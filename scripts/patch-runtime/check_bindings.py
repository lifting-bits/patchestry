#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Check a lifted unit's binding manifest against the ELF it was lifted from.

`patchir-decomp -emit-bindings` lists every global and function the lifted
code refers to, with the address, size and kind Ghidra recovered.  Reinsertion
resolves those names against the original ELF.  This script checks that each
name resolves to the object the binary actually has, not to something else and
not to nothing.

Verdict per binding:

  match              an ELF symbol of that name has the recovered address
                     and kind, and a compatible size (an object's sizes are
                     equal; a function's recovered body fits in the symbol,
                     which also covers the literal pool after the body)
  alias              no symbol of that name, but another symbol starts at the
                     recovered address
  interior           no symbol of that name; the address lies inside another
                     symbol
  unnamed            no symbol covers the address, but a loaded segment does;
                     the name needs an absolute definition at that address
  outside-image      an object no symbol or loaded segment covers, such as
                     a memory-mapped register; the ELF cannot confirm it and
                     the name needs an absolute definition
  new                the lifter introduced the entity; nothing to resolve
  size-mismatch      the symbol's size differs from the recovered size
  address-mismatch   a symbol of that name exists at a different address
  kind-mismatch      a function resolves to an object symbol, or the reverse
  ambiguous-symbol   the linker name stands for more than one address: the
                     ELF defines it at several (file-local symbols that share
                     a name), or two bindings lifted under it were recovered
                     at different addresses.  Reinsertion resolves a name to
                     one address, so at least one reference would go to the
                     wrong place
  not-executable     a function's address is not in an executable segment
  unmapped           a function no symbol or loaded segment covers
  imported           Ghidra placed the entity outside the binary; a static
                     image has nothing to resolve it against
  plt-stub           an imported function whose PLT stub in the image was
                     confirmed (dynamically linked executables only)
  unexplained        the lifted unit declares it but the program model does
                     not account for it

The manifest also lists the calls whose target nothing binds: indirect calls
and tail calls to an address that is not a known function.  They are copied
into the report and counted, not judged: where such a call goes is decided at
run time.

Only fixed-address ELF images are supported.  A shared object or
position-independent executable (ET_DYN) is always rejected: its symbols
resolve through a load bias this check does not model.

An image with a dynamic section or a program interpreter is rejected unless
`--allow-dynamic` is given.  Its own code and data are still at fixed
addresses; what differs is an imported function, which the image reaches
through a PLT stub.  With `--function-json`, the stub address the lift
recorded for each import is checked against the image: the stub must load its
target from the GOT slot whose relocation names that import.  Such a binding
gets the verdict `plt-stub`; an import with no stub, or with a stub of another
import, keeps the failing verdict `imported`.

Exit status is 1 when any binding gets a verdict from address-mismatch
onward in the list above; `--strict` adds size-mismatch.  With `--rewritten`,
every original binding the unit does not define must also still be at the
same address in the rewritten ELF.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from elftools.elf.dynamic import DynamicSegment
from elftools.elf.elffile import ELFFile

FAILING = {
    "address-mismatch",
    "kind-mismatch",
    "ambiguous-symbol",
    "not-executable",
    "unmapped",
    "imported",
    "unexplained",
    "moved",
}
STRICT_FAILING = {"size-mismatch"}

PF_X = 0x1
PF_W = 0x2
PF_R = 0x4


class UnsupportedImage(RuntimeError):
    """The ELF is not a static, fixed-address image."""


@dataclass(frozen=True)
class ElfSymbol:
    name: str
    addr: int
    size: int
    kind: str  # STT_FUNC, STT_OBJECT, STT_NOTYPE, ...
    thumb: bool = False  # ARM only: the function symbol had bit 0 set


@dataclass(frozen=True)
class Segment:
    start: int
    end: int  # exclusive
    flags: int

    def permissions(self) -> str:
        return (
            ("r" if self.flags & PF_R else "-")
            + ("w" if self.flags & PF_W else "-")
            + ("x" if self.flags & PF_X else "-")
        )


@dataclass
class Image:
    symbols: dict[str, list[ElfSymbol]]
    by_address: list[ElfSymbol]
    segments: list[Segment]

    def segment_at(self, address: int) -> Segment | None:
        for segment in self.segments:
            if segment.start <= address < segment.end:
                return segment
        return None

    def symbol_at(self, address: int) -> ElfSymbol | None:
        """A sized symbol that starts at `address`, preferring a named kind."""
        best = None
        for symbol in self.by_address:
            if symbol.addr == address and symbol.kind in ("STT_FUNC", "STT_OBJECT"):
                return symbol
            if symbol.addr == address and best is None:
                best = symbol
        return best

    def symbol_covering(self, address: int) -> ElfSymbol | None:
        for symbol in self.by_address:
            if symbol.size and symbol.addr < address < symbol.addr + symbol.size:
                return symbol
        return None


def load_image(path: Path, allow_dynamic: bool = False) -> Image:
    """Read symbols and loaded segments; reject anything not at fixed addresses.

    A dynamically linked executable is accepted only with `allow_dynamic`.
    """
    with path.open("rb") as handle:
        elf = ELFFile(handle)
        if elf["e_type"] != "ET_EXEC":
            raise UnsupportedImage(
                f"{path}: ELF type is {elf['e_type']}; only a fixed-address "
                "executable (ET_EXEC) is supported"
            )
        segments = []
        for segment in elf.iter_segments():
            if segment["p_type"] in ("PT_DYNAMIC", "PT_INTERP") and not allow_dynamic:
                raise UnsupportedImage(
                    f"{path}: has a {segment['p_type']} segment; dynamically linked "
                    "images are not supported unless allowed"
                )
            if segment["p_type"] == "PT_LOAD":
                start = int(segment["p_vaddr"])
                segments.append(
                    Segment(start, start + int(segment["p_memsz"]), int(segment["p_flags"]))
                )

        is_arm = elf["e_machine"] == "EM_ARM"
        symbols: dict[str, list[ElfSymbol]] = {}
        symtab = elf.get_section_by_name(".symtab")
        if symtab is not None:
            for symbol in symtab.iter_symbols():
                if not symbol.name or symbol["st_shndx"] == "SHN_UNDEF":
                    continue
                kind = symbol["st_info"]["type"]
                if kind in ("STT_SECTION", "STT_FILE"):
                    continue
                addr = int(symbol["st_value"])
                # ARM marks a Thumb function by setting bit 0 of its address.
                thumb = is_arm and kind == "STT_FUNC" and bool(addr & 1)
                if is_arm and kind == "STT_FUNC":
                    addr &= ~1
                symbols.setdefault(symbol.name, []).append(
                    ElfSymbol(symbol.name, addr, int(symbol["st_size"]), kind, thumb)
                )
        # ARM mapping symbols ($t, $d, $a) mark code and data runs, not objects.
        by_address = sorted(
            (s for group in symbols.values() for s in group if not s.name.startswith("$")),
            key=lambda s: (s.addr, s.name),
        )
        return Image(symbols, by_address, segments)


def _jump_slots(elf: ELFFile) -> dict[int, str]:
    """GOT slot address -> name of the import the loader binds there."""
    slots: dict[int, str] = {}
    for segment in elf.iter_segments():
        if not isinstance(segment, DynamicSegment):
            continue
        table = segment.get_relocation_tables().get("JMPREL")
        if table is None:
            continue
        for relocation in table.iter_relocations():
            symbol = segment.get_symbol(relocation["r_info_sym"])
            if symbol is not None and symbol.name:
                slots[int(relocation["r_offset"])] = symbol.name
    return slots


def _read(elf: ELFFile, address: int, size: int) -> bytes | None:
    for segment in elf.iter_segments():
        start = int(segment["p_vaddr"])
        if (
            segment["p_type"] == "PT_LOAD"
            and segment["p_flags"] & PF_X
            and start <= address
            and address + size <= start + int(segment["p_filesz"])
        ):
            return segment.data()[address - start:address - start + size]
    return None


def plt_stub_import(path: Path, address: int) -> str | None:
    """The import an ARM PLT stub at `address` jumps to, or None if it is not one.

    The stub is three ARM instructions that build the address of a GOT slot in
    ip and load pc from it:

        add ip, pc, #a
        add ip, ip, #b
        ldr pc, [ip, #c]!

    The slot is `address + 8 + a + b + c`, and the import is the symbol of the
    jump-slot relocation at that slot.
    """
    with path.open("rb") as handle:
        elf = ELFFile(handle)
        if elf["e_machine"] != "EM_ARM":
            return None
        code = _read(elf, address, 12)
        if code is None:
            return None
        words = [int.from_bytes(code[i:i + 4], "little") for i in (0, 4, 8)]
        # add ip, pc, #a / add ip, ip, #b / ldr pc, [ip, #c]!
        if [word & 0xFFFFF000 for word in words] != [0xE28FC000, 0xE28CC000, 0xE5BCF000]:
            return None

        def rotated(word: int) -> int:
            value, turn = word & 0xFF, 2 * ((word >> 8) & 0xF)
            return ((value >> turn) | (value << (32 - turn))) & 0xFFFFFFFF

        slot = address + 8 + rotated(words[0]) + rotated(words[1]) + (words[2] & 0xFFF)
        return _jump_slots(elf).get(slot & 0xFFFFFFFF)


def stubs_from_function_json(function_json: dict) -> dict[str, int]:
    """Name -> stub address for every import the lift recorded an entry point for."""
    stubs = {}
    for key, function in function_json.get("functions", {}).items():
        entry = function.get("entry_point")
        if key.startswith("EXTERNAL:") and entry:
            stubs[function["name"]] = int(entry.rsplit(":", 1)[-1], 16)
    return stubs


def binding_address(binding: dict) -> int | None:
    text = binding.get("address")
    return int(text, 16) if text else None


def kind_matches(binding_kind: str, elf_kind: str) -> bool:
    if elf_kind == "STT_NOTYPE":
        return True  # assembly labels carry no type
    if binding_kind == "function":
        return elf_kind == "STT_FUNC"
    return elf_kind in ("STT_OBJECT", "STT_COMMON", "STT_TLS")


def size_differs(binding: dict, symbol: ElfSymbol) -> bool:
    size = binding.get("size")
    if not size or not symbol.size:
        return False  # one side has no size to compare
    if binding["kind"] == "function":
        # The symbol covers the body and its literal pool; Ghidra's function
        # body stops at the last instruction.
        return size > symbol.size
    return size != symbol.size


def check_binding(
    binding: dict, image: Image, original: Path | None = None, stubs: dict | None = None
) -> dict:
    """Return the report row for one binding.

    `stubs` gives the PLT stub address claimed for each import; it is checked
    against `original`.
    """
    row = {
        "name": binding["name"],
        "symbol": binding.get("symbol", binding["name"]),
        "kind": binding["kind"],
        "origin": binding["origin"],
    }
    address = binding_address(binding)
    if address is not None:
        row["address"] = f"0x{address:x}"

    if binding["origin"] == "patch-owned":
        row["verdict"] = "new"
        return row
    if binding["origin"] == "imported":
        row["verdict"] = "imported"
        stub = (stubs or {}).get(row["symbol"])
        if stub is not None and original is not None:
            resolved = plt_stub_import(original, stub)
            if resolved == row["symbol"]:
                row["verdict"] = "plt-stub"
                row["stub"] = f"0x{stub:x}"
            else:
                row["stub_claimed"] = f"0x{stub:x}"
                row["stub_resolves_to"] = resolved
        return row
    if binding["origin"] != "original" or address is None:
        row["verdict"] = "unexplained"
        return row

    segment = image.segment_at(address)
    if segment is not None:
        row["segment"] = segment.permissions()

    candidates = image.symbols.get(row["symbol"], [])
    if candidates:
        # A name can be defined more than once (file-local statics); report
        # the definition at the recovered address when there is one.
        symbol = next((s for s in candidates if s.addr == address), candidates[0])
        row["elf"] = {"addr": f"0x{symbol.addr:x}", "size": symbol.size, "type": symbol.kind}
        defined_at = sorted({s.addr for s in candidates})
        if symbol.addr != address:
            row["verdict"] = "address-mismatch"
        elif len(defined_at) > 1:
            # One of the definitions is the right one, but a name resolves to
            # a single address and nothing says it is this one.
            row["verdict"] = "ambiguous-symbol"
            row["elf_definitions"] = [f"0x{addr:x}" for addr in defined_at]
        elif not kind_matches(binding["kind"], symbol.kind):
            row["verdict"] = "kind-mismatch"
        elif size_differs(binding, symbol):
            row["verdict"] = "size-mismatch"
        else:
            row["verdict"] = "match"
    else:
        at = image.symbol_at(address)
        covering = None if at else image.symbol_covering(address)
        if at is not None:
            row["verdict"] = "alias"
            row["elf"] = {"name": at.name, "addr": f"0x{at.addr:x}", "size": at.size,
                          "type": at.kind}
            if not kind_matches(binding["kind"], at.kind):
                row["verdict"] = "kind-mismatch"
        elif covering is not None:
            row["verdict"] = "interior"
            row["elf"] = {"name": covering.name, "addr": f"0x{covering.addr:x}",
                          "size": covering.size, "type": covering.kind,
                          "offset": address - covering.addr}
        elif segment is not None:
            row["verdict"] = "unnamed"
        else:
            row["verdict"] = "unmapped" if binding["kind"] == "function" else "outside-image"

    if (
        binding["kind"] == "function"
        and row["verdict"] not in FAILING
        and segment is not None
        and not segment.flags & PF_X
    ):
        row["verdict"] = "not-executable"
    return row


def check_bindings(
    manifest: dict,
    original: Path,
    rewritten: Path | None = None,
    allow_dynamic: bool = False,
    stubs: dict[str, int] | None = None,
) -> dict:
    """Check every binding in `manifest` against `original`.

    With `rewritten`, also require that each original entity the unit does not
    define is at the same address there.  Raises UnsupportedImage for an ELF
    that is not a fixed-address image, or is dynamically linked and
    `allow_dynamic` is not set.  `stubs` (name -> address) says where the PLT
    stub of each import is; only a stub the image confirms binds the import.
    """
    image = load_image(original, allow_dynamic)
    rows = [
        check_binding(binding, image, original, stubs if allow_dynamic else None)
        for binding in manifest.get("bindings", [])
    ]

    rewritten_checked = None
    if rewritten is not None:
        after = load_image(rewritten, allow_dynamic)
        # A rewriter that strips the symbol table leaves nothing to compare.
        rewritten_checked = bool(after.symbols)
    if rewritten_checked:
        defined = {b["name"] for b in manifest.get("bindings", []) if b.get("defined")}
        for row in rows:
            if row["verdict"] != "match" or row["name"] in defined:
                continue
            address = int(row["address"], 16)
            candidates = after.symbols.get(row["symbol"], [])
            if not any(symbol.addr == address for symbol in candidates):
                row["verdict"] = "moved"
                row["rewritten"] = [f"0x{symbol.addr:x}" for symbol in candidates]

    # Two entities lifted under one symbol resolve to the same place at link
    # time, whatever their recovered addresses say.  When the addresses
    # differ, at most one of them gets the address it was recovered at.
    sharing: dict[str, int] = {}
    addresses: dict[str, set[str]] = {}
    for row in rows:
        if row["verdict"] != "new":
            sharing[row["symbol"]] = sharing.get(row["symbol"], 0) + 1
            if "address" in row:
                addresses.setdefault(row["symbol"], set()).add(row["address"])
    for row in rows:
        if sharing.get(row["symbol"], 0) > 1:
            row["shared_symbol"] = sharing[row["symbol"]]
        if (
            row["verdict"] != "new"
            and row["verdict"] not in FAILING
            and len(addresses.get(row["symbol"], ())) > 1
        ):
            row["verdict"] = "ambiguous-symbol"
            row["binding_addresses"] = sorted(
                addresses[row["symbol"]], key=lambda text: int(text, 16)
            )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    # Names with no ELF symbol of their own: the patch link has to be given
    # these as absolute symbols, or they resolve to nothing.
    absolute = {
        row["symbol"]: row["address"]
        for row in rows
        if row["verdict"] in ("alias", "interior", "unnamed", "outside-image")
    }
    absolute.update({row["symbol"]: row["stub"] for row in rows if row["verdict"] == "plt-stub"})
    return {
        "original": str(original),
        "rewritten": str(rewritten) if rewritten else None,
        "rewritten_checked": rewritten_checked,
        "summary": dict(sorted(counts.items())),
        "absolute_symbols": dict(sorted(absolute.items())),
        "unresolved_calls": manifest.get("unresolved_calls", []),
        "bindings": rows,
    }


def failures(report: dict, strict: bool) -> list[dict]:
    failing = FAILING | (STRICT_FAILING if strict else set())
    return [row for row in report["bindings"] if row["verdict"] in failing]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bindings", type=Path, required=True,
                        help="manifest from patchir-decomp -emit-bindings")
    parser.add_argument("--elf", type=Path, required=True, help="ELF the unit was lifted from")
    parser.add_argument("--rewritten", type=Path, help="ELF after reinsertion")
    parser.add_argument("--output", type=Path, help="write the report here (default: stdout)")
    parser.add_argument("--strict", action="store_true", help="also fail on size-mismatch")
    parser.add_argument("--allow-dynamic", action="store_true",
                        help="accept a dynamically linked fixed-address executable")
    parser.add_argument("--function-json", type=Path,
                        help="Ghidra JSON of the lift; gives the PLT stub of each import")
    args = parser.parse_args()

    manifest = json.loads(args.bindings.read_text(encoding="utf-8"))
    stubs = None
    if args.function_json:
        stubs = stubs_from_function_json(
            json.loads(args.function_json.read_text(encoding="utf-8"))
        )
    try:
        report = check_bindings(manifest, args.elf, args.rewritten, args.allow_dynamic, stubs)
    except UnsupportedImage as error:
        print(f"check-bindings: {error}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    if report["rewritten_checked"] is False:
        print(f"check-bindings: {args.rewritten} has no symbol table; "
              "rewritten addresses were not compared", file=sys.stderr)
    failed = failures(report, args.strict)
    summary = ", ".join(f"{count} {verdict}" for verdict, count in report["summary"].items())
    print(f"check-bindings: {len(report['bindings'])} bindings: {summary}", file=sys.stderr)
    if report["unresolved_calls"]:
        print(f"check-bindings: {len(report['unresolved_calls'])} call(s) with an "
              "unresolved target", file=sys.stderr)
    for row in failed:
        print(
            f"check-bindings: {row['verdict']}: {row['kind']} '{row['name']}'"
            f" at {row.get('address', '?')}",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
