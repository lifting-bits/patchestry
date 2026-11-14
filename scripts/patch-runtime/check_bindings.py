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
  not-executable     a function's address is not in an executable segment
  unmapped           a function no symbol or loaded segment covers
  imported           Ghidra placed the entity outside the binary; a static
                     image has nothing to resolve it against
  unexplained        the lifted unit declares it but the program model does
                     not account for it

Only static, fixed-address ELF images are supported.  A shared object or
position-independent executable (ET_DYN), or an image with a dynamic section
or a program interpreter, is rejected: its symbols resolve through a load
bias and PLT/GOT entries that this check does not model.

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

from elftools.elf.elffile import ELFFile

FAILING = {
    "address-mismatch",
    "kind-mismatch",
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


def load_image(path: Path) -> Image:
    """Read symbols and loaded segments; reject anything not statically placed."""
    with path.open("rb") as handle:
        elf = ELFFile(handle)
        if elf["e_type"] != "ET_EXEC":
            raise UnsupportedImage(
                f"{path}: ELF type is {elf['e_type']}; only a static, fixed-address "
                "executable (ET_EXEC) is supported"
            )
        segments = []
        for segment in elf.iter_segments():
            if segment["p_type"] in ("PT_DYNAMIC", "PT_INTERP"):
                raise UnsupportedImage(
                    f"{path}: has a {segment['p_type']} segment; dynamically linked "
                    "images are not supported"
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
                if is_arm and kind == "STT_FUNC":
                    addr &= ~1
                symbols.setdefault(symbol.name, []).append(
                    ElfSymbol(symbol.name, addr, int(symbol["st_size"]), kind)
                )
        # ARM mapping symbols ($t, $d, $a) mark code and data runs, not objects.
        by_address = sorted(
            (s for group in symbols.values() for s in group if not s.name.startswith("$")),
            key=lambda s: (s.addr, s.name),
        )
        return Image(symbols, by_address, segments)


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


def check_binding(binding: dict, image: Image) -> dict:
    """Return the report row for one binding."""
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
        return row
    if binding["origin"] != "original" or address is None:
        row["verdict"] = "unexplained"
        return row

    segment = image.segment_at(address)
    if segment is not None:
        row["segment"] = segment.permissions()

    candidates = image.symbols.get(row["symbol"], [])
    if candidates:
        # A name can be defined more than once (file-local statics); take the
        # definition at the recovered address when there is one.
        symbol = next((s for s in candidates if s.addr == address), candidates[0])
        row["elf"] = {"addr": f"0x{symbol.addr:x}", "size": symbol.size, "type": symbol.kind}
        if symbol.addr != address:
            row["verdict"] = "address-mismatch"
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


def check_bindings(manifest: dict, original: Path, rewritten: Path | None = None) -> dict:
    """Check every binding in `manifest` against `original`.

    With `rewritten`, also require that each original entity the unit does not
    define is at the same address there.  Raises UnsupportedImage for an ELF
    that is not a static, fixed-address image.
    """
    image = load_image(original)
    rows = [check_binding(binding, image) for binding in manifest.get("bindings", [])]

    rewritten_checked = None
    if rewritten is not None:
        after = load_image(rewritten)
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
    # time, whatever their recovered addresses say.
    sharing: dict[str, int] = {}
    for row in rows:
        if row["verdict"] != "new":
            sharing[row["symbol"]] = sharing.get(row["symbol"], 0) + 1
    for row in rows:
        if sharing.get(row["symbol"], 0) > 1:
            row["shared_symbol"] = sharing[row["symbol"]]

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
    return {
        "original": str(original),
        "rewritten": str(rewritten) if rewritten else None,
        "rewritten_checked": rewritten_checked,
        "summary": dict(sorted(counts.items())),
        "absolute_symbols": dict(sorted(absolute.items())),
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
    args = parser.parse_args()

    manifest = json.loads(args.bindings.read_text(encoding="utf-8"))
    try:
        report = check_bindings(manifest, args.elf, args.rewritten)
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
    for row in failed:
        print(
            f"check-bindings: {row['verdict']}: {row['kind']} '{row['name']}'"
            f" at {row.get('address', '?')}",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
