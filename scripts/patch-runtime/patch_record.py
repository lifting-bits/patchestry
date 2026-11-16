#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Record of what the rewriter did to an image, and a check of it.

Patcherex2 writes no record of a patch: it prints where it put each inserted
function and does not update the symbol table.  It does keep every write it
makes (`binfmt_tool.file_updates`) and the symbols each link can see
(`p.symbols` and the image's functions from the binary analyzer).  `Recorder` watches each patch being applied and attributes the
writes and the symbols to it.  `build_record` turns that into a JSON document:

- the original and rewritten image, by hash;
- the target function and the compiler invocation;
- for every function, in insertion order: where it was placed, the hash of the
  IR snippet it was compiled from, the bytes written (offset, address, size,
  hash), and the address of each symbol it references as its link saw it;
- writes that no watched patch made.

`check_record` compares the record with the rewritten image and with the
report of check_trampoline.py, which finds the new body independently by
decoding the branch at the old entry:

| Problem | Meaning |
| --- | --- |
| `image-mismatch` | the rewritten image is not the file the record was made for |
| `missing-target` | the record has no entry for the target function |
| `unmapped-write` | a write is outside the loaded segments of the rewritten image |
| `write-mismatch` | the image does not hold the recorded bytes at a recorded write |
| `destination-mismatch` | the target's recorded placement or address is not what the image's entry shows |
| `symbol-mismatch` | a referenced symbol had no address at link time, or an inserted function was linked at an address other than where it was placed |
| `unattributed-write` | the rewriter wrote bytes that belong to no recorded function and are not the ELF header or its header tables |
| `entry-changed` | the entry point differs from the original image's |
| `segment-changed` | a loaded segment of the original image is no longer loaded at the same addresses with the same permissions |

The record says what was written, not that it is the right code: comparing the
bytes with a fresh compile of the snippets is a separate check.

CLI: patch_record.py --record patch_record.json --rewritten rewritten.elf
     --trampoline trampoline.json
Exit 0 when the record is consistent, 1 when not, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

RECORD_VERSION = 1


@dataclass
class Applied:
    """One function the rewriter placed."""

    name: str
    role: str  # "inserted" or "target"
    snippet: str
    symbols: dict[str, int] = field(default_factory=dict)
    writes: list[tuple[int, bytes]] = field(default_factory=list)


class Recorder:
    """Attributes the rewriter's writes and link symbols to the patch that made them."""

    def __init__(self) -> None:
        self.applied: list[Applied] = []
        self._claimed: set[int] = set()

    def watch(self, patch, name: str, role: str, snippet: str) -> None:
        """Wrap `patch.apply` so that applying it adds an entry to `applied`."""
        original_apply = patch.apply

        def apply(p):
            before = len(p.binfmt_tool.file_updates)
            # What the link of this patch resolves against, in the rewriter's
            # own order: its symbol table, then the image's functions on top.
            symbols = dict(p.symbols)
            symbols.update(p.binary_analyzer.get_all_symbols())
            entry = Applied(name=name, role=role, snippet=snippet, symbols=symbols)
            original_apply(p)
            for index in range(before, len(p.binfmt_tool.file_updates)):
                update = p.binfmt_tool.file_updates[index]
                entry.writes.append((int(update["offset"]), bytes(update["content"])))
                self._claimed.add(index)
            self.applied.append(entry)

        patch.apply = apply

    def unattributed(self, p) -> list[tuple[int, bytes]]:
        """Writes the rewriter made outside every watched patch."""
        return [
            (int(update["offset"]), bytes(update["content"]))
            for index, update in enumerate(p.binfmt_tool.file_updates)
            if index not in self._claimed
        ]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex(value: int | None) -> str | None:
    return None if value is None else f"0x{value:x}"


def _load_segments(elf: Path) -> list[tuple[int, int, int]]:
    """(file offset, file size, address) of each loaded segment."""
    with elf.open("rb") as handle:
        return [
            (int(segment["p_offset"]), int(segment["p_filesz"]), int(segment["p_vaddr"]))
            for segment in ELFFile(handle).iter_segments()
            if segment["p_type"] == "PT_LOAD" and segment["p_filesz"]
        ]


def _layout(elf: Path) -> dict:
    """Entry point, loaded segments and the extent of the ELF's own tables."""
    with elf.open("rb") as handle:
        parsed = ELFFile(handle)
        header = parsed.header
        return {
            "entry": _hex(int(header["e_entry"])),
            "segments": [
                {
                    "address": _hex(int(segment["p_vaddr"])),
                    "load_address": _hex(int(segment["p_paddr"])),
                    "file_size": int(segment["p_filesz"]),
                    "memory_size": int(segment["p_memsz"]),
                    "flags": int(segment["p_flags"]),
                }
                for segment in parsed.iter_segments()
                if segment["p_type"] == "PT_LOAD"
            ],
            "structure": {
                "elf-header": (0, int(header["e_ehsize"])),
                "program-headers": (
                    int(header["e_phoff"]),
                    int(header["e_phnum"]) * int(header["e_phentsize"]),
                ),
                "section-headers": (
                    int(header["e_shoff"]),
                    int(header["e_shnum"]) * int(header["e_shentsize"]),
                ),
            },
        }


def _address_of(segments: list[tuple[int, int, int]], offset: int, size: int) -> int | None:
    for start, length, address in segments:
        if start <= offset and offset + size <= start + length:
            return address + offset - start
    return None


def _function_symbol(elf: Path, name: str) -> tuple[int, int]:
    with elf.open("rb") as handle:
        for section in ELFFile(handle).iter_sections():
            if not isinstance(section, SymbolTableSection):
                continue
            for symbol in section.iter_symbols():
                if symbol.name == name and symbol["st_info"]["type"] == "STT_FUNC":
                    return int(symbol["st_value"]) & ~1, int(symbol["st_size"])
    raise LookupError(f"{elf} has no function symbol {name}")


def _write_rows(
    segments: list[tuple[int, int, int]], writes: list[tuple[int, bytes]]
) -> list[dict]:
    return [
        {
            "kind": "body",
            "file_offset": _hex(offset),
            "address": _hex(_address_of(segments, offset, len(content))),
            "size": len(content),
            "sha256": _sha256(content),
        }
        for offset, content in writes
    ]


def build_record(
    original: Path,
    rewritten: Path,
    target: str,
    applied: list[Applied],
    compiler: dict,
    references: Mapping[str, list[str]],
    unattributed: list[tuple[int, bytes]],
    target_extent: tuple[int, int] | None = None,
    mode: str = "thumb",
) -> dict:
    """The record for one rewritten image.

    `references` gives, per function, the symbols its snippet leaves undefined
    (from the dependency report); the record keeps the address each had when
    that function was linked.  `target_extent` is the (address, size) of the
    target for an image with no symbol for it, and `mode` the instruction set
    it was compiled for ("thumb" or "arm").
    """
    segments = _load_segments(rewritten)
    target_address, target_size = target_extent or _function_symbol(original, target)
    original_layout = _layout(original)
    rewritten_layout = _layout(rewritten)

    # A write outside every patch is expected only where saving the image
    # rewrites the ELF header and its header tables.
    other_writes = _write_rows(segments, unattributed)
    for row, (offset, content) in zip(other_writes, unattributed):
        # The header and its tables can themselves be in a loaded segment.
        row["kind"] = "unknown" if row["address"] is None else "loaded"
        for kind, (start, size) in rewritten_layout["structure"].items():
            if start <= offset and offset + len(content) <= start + size:
                row["kind"] = kind

    functions = []
    for order, entry in enumerate(applied):
        writes = _write_rows(segments, entry.writes)
        placement = "inserted"
        body = writes
        if entry.role == "target":
            at_entry = [row for row in writes if row["address"] == _hex(target_address)]
            elsewhere = [row for row in writes if row["address"] != _hex(target_address)]
            if at_entry and elsewhere:
                placement = "trampoline"
                for row in at_entry:
                    row["kind"] = "entry-branch"
                body = elsewhere
            else:
                placement = "in-place"
        functions.append(
            {
                "name": entry.name,
                "role": entry.role,
                "order": order,
                "placement": placement,
                "address": body[0]["address"] if body else None,
                "size": sum(row["size"] for row in body),
                "snippet_sha256": _sha256(entry.snippet.encode("utf-8")),
                "symbols": {
                    name: _hex(entry.symbols.get(name)) for name in references.get(entry.name, [])
                },
                "writes": writes,
            }
        )

    return {
        "version": RECORD_VERSION,
        "original": {
            "path": str(original),
            "sha256": _sha256(original.read_bytes()),
            "entry": original_layout["entry"],
            "segments": original_layout["segments"],
        },
        "rewritten": {
            "path": str(rewritten),
            "sha256": _sha256(rewritten.read_bytes()),
            "entry": rewritten_layout["entry"],
        },
        "target": {
            "name": target, "address": _hex(target_address), "size": target_size, "mode": mode
        },
        "compiler": compiler,
        "functions": functions,
        "unattributed_writes": other_writes,
    }


def link_address_problems(record: dict, addresses: Mapping[str, int]) -> list[str]:
    """References that were linked against an address other than the checked one.

    `addresses` gives, by name, where the binding check found each entity of
    the original image.  The rewriter resolves names through its own analysis
    of the image; this compares the two.
    """
    problems = []
    for row in record["functions"]:
        for name, value in row["symbols"].items():
            wanted = addresses.get(name)
            if wanted is None or value is None:
                continue
            if int(value, 16) & ~1 != wanted & ~1:
                problems.append(
                    f"{row['name']} was linked with {name} at {value}; "
                    f"the binding check has it at 0x{wanted:x}"
                )
    return problems


def check_record(record: dict, rewritten: Path, trampoline: dict) -> dict[str, str]:
    """Problems found, keyed by name; empty when the record is consistent."""
    found: dict[str, list[str]] = {}

    def problem(kind: str, detail: str) -> None:
        found.setdefault(kind, []).append(detail)

    image = rewritten.read_bytes()
    if _sha256(image) != record["rewritten"]["sha256"]:
        problem("image-mismatch", f"{rewritten} is not the image the record was made for")

    functions = record["functions"]
    placed = {row["name"]: row for row in functions}
    target_name = record["target"]["name"]

    for row in functions:
        for write in row["writes"]:
            where = f"{row['name']} at file offset {write['file_offset']}"
            if write["address"] is None:
                problem("unmapped-write", where)
                continue
            offset = int(write["file_offset"], 16)
            if _sha256(image[offset:offset + write["size"]]) != write["sha256"]:
                problem("write-mismatch", f"{where} (address {write['address']})")

        for name, value in row["symbols"].items():
            if value is None:
                problem("symbol-mismatch", f"{row['name']} was linked with no address for {name}")
                continue
            other = placed.get(name)
            if other is None or other["role"] != "inserted" or other["address"] is None:
                continue
            if int(value, 16) & ~1 != int(other["address"], 16):
                problem(
                    "symbol-mismatch",
                    f"{row['name']} was linked with {name} at {value}; "
                    f"{name} was placed at {other['address']}",
                )

    target = placed.get(target_name)
    if target is None or target["role"] != "target":
        problem("missing-target", f"no record of {target_name}")
    else:
        seen = trampoline.get("verdict")
        expected = (
            trampoline.get("destination") if seen == "trampoline" else record["target"]["address"]
        )
        if target["placement"] != seen or target["address"] != expected:
            problem(
                "destination-mismatch",
                f"recorded {target['placement']} at {target['address']}; "
                f"the image shows {seen} at {expected}",
            )

    # The header tables are rewritten on save, so what they say about the
    # original image has to be checked: same entry point, and every original
    # segment still loaded at the same addresses with the same permissions.
    layout = _layout(rewritten)
    if layout["entry"] != record["original"]["entry"]:
        problem(
            "entry-changed",
            f"entry point was {record['original']['entry']}, is {layout['entry']}",
        )
    for old in record["original"]["segments"]:
        start = int(old["address"], 16)
        delta = int(old["load_address"], 16) - start
        if not any(
            int(new["address"], 16) <= start
            and start + old["file_size"] <= int(new["address"], 16) + new["file_size"]
            and start + old["memory_size"] <= int(new["address"], 16) + new["memory_size"]
            and int(new["load_address"], 16) - int(new["address"], 16) == delta
            and new["flags"] == old["flags"]
            for new in layout["segments"]
        ):
            problem(
                "segment-changed",
                f"original segment at {old['address']} ({old['memory_size']} bytes, "
                f"flags {old['flags']}) is not loaded the same way",
            )

    for write in record["unattributed_writes"]:
        if write["kind"] in ("elf-header", "program-headers", "section-headers"):
            continue
        problem(
            "unattributed-write",
            f"{write['size']} bytes at file offset {write['file_offset']} "
            f"(address {write['address']})",
        )

    return {kind: "; ".join(details) for kind, details in found.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--rewritten", type=Path, required=True)
    parser.add_argument("--trampoline", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
        trampoline = json.loads(args.trampoline.read_text(encoding="utf-8"))
        problems = check_record(record, args.rewritten, trampoline)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for kind, detail in problems.items():
        print(f"{kind}: {detail}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
