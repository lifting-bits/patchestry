#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Check that the bytes in a rewritten image are the intended replacement.

The patch record (patch_record.py) says which function went where and which
address each symbol had at link time.  This check takes the IR snippets again,
compiles each one with the recorded flags, links it at its recorded address
against its recorded symbols, and compares the result with what the image
holds, byte for byte.  The link is a plain `ld.lld` link; the rewriter gets
its bytes another way (a relocatable link that a loader then relocates), so
agreement is between two independent paths from the same object code.

What must hold:

| Problem | Meaning |
| --- | --- |
| `snippet-mismatch` | a snippet is missing or is not the one the record was made from |
| `link-failed` | the snippet does not compile or link with the recorded symbols |
| `byte-mismatch` | the image differs from the expected bytes of a function, or the recorded size is not the expected size |
| `entry-branch-mismatch` | the bytes at the old entry are not a Thumb `b.w` to the new body |
| `unrecorded-change` | a byte of the original image changed outside every recorded write |
| `unrecorded-code` | a non-zero byte was added outside every recorded write |

Together with the record check this accounts for every loaded byte of the
rewritten image: it is unchanged from the original, zero fill, or equal to
code compiled here from a known snippet.

Not covered: that the compiler translated the IR correctly (both paths use the
same compiler), and anything outside the loaded segments.

CLI: check_applied_bytes.py --record patch_record.json --elf firmware.elf
     --rewritten rewritten.elf --module patched.ll [--output report.json]
Exit 0 when everything matches, 1 when not, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping

from elftools.elf.elffile import ELFFile

import check_trampoline
import patch_dependencies


def find_lld(tools: patch_dependencies.Tools) -> Path:
    configured = os.environ.get("PATCHESTRY_LD_LLD")
    return Path(configured) if configured else tools.clang.parent / "ld.lld"


def thumb_branch_wide(source: int, target: int) -> bytes:
    """Encoding of Thumb-2 `b.w target` placed at `source` (encoding T4)."""
    offset = (target - (source + 4)) & 0x1FFFFFF
    sign = (offset >> 24) & 1
    i1, i2 = (offset >> 23) & 1, (offset >> 22) & 1
    j1, j2 = (~(i1 ^ sign)) & 1, (~(i2 ^ sign)) & 1
    high = 0xF000 | (sign << 10) | ((offset >> 12) & 0x3FF)
    low = 0x9000 | (j1 << 13) | (j2 << 11) | ((offset >> 1) & 0x7FF)
    return high.to_bytes(2, "little") + low.to_bytes(2, "little")


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed ({result.returncode}):\n{result.stderr}")


def expected_bytes(
    snippet: str,
    address: int,
    symbols: Mapping[str, int],
    compiler: dict,
    tools: patch_dependencies.Tools,
) -> bytes:
    """The bytes of one snippet linked at `address` against `symbols`.

    The layout is the reinsertion's: the code, then each read-only data
    section at its own alignment.  Writable data has no place in it.
    """
    with tempfile.TemporaryDirectory(prefix="applied-bytes-") as directory:
        work = Path(directory)
        source, obj, script, linked = (
            work / "snippet.ll", work / "snippet.o", work / "link.ld", work / "linked.elf"
        )
        source.write_text(snippet, encoding="utf-8")
        _run(
            [str(tools.clang), "-target", compiler["target"], *compiler["flags"],
             "-c", str(source), "-o", str(obj)]
        )
        with obj.open("rb") as handle:
            read_only = " ".join(
                f". = ALIGN({section['sh_addralign']}); *({section.name})"
                for section in ELFFile(handle).iter_sections()
                if section.name.startswith(".rodata")
            )
        definitions = " ".join(f"{name} = {value:#x};" for name, value in symbols.items())
        script.write_text(
            f"SECTIONS {{ .patcherex2 {address:#x} : SUBALIGN(0) {{ *(.text) {read_only} }} "
            "/DISCARD/ : { *(.ARM.exidx*) *(.ARM.extab*) *(.ARM.attributes) *(.comment) *(.note*) } }\n"
            f"{definitions}\n",
            encoding="utf-8",
        )
        _run([str(find_lld(tools)), str(obj), "-T", str(script), "--entry=0", "-o", str(linked)])
        with linked.open("rb") as handle:
            section = ELFFile(handle).get_section_by_name(".patcherex2")
            if section is None:
                raise RuntimeError("the linked snippet has no code")
            return section.data()


def expected_functions(
    record: dict, snippets: Mapping[str, str], tools: patch_dependencies.Tools
) -> dict[str, tuple[str, bytes | str]]:
    """Per recorded function: ("ok", bytes) or (problem, detail)."""
    expected: dict[str, tuple[str, bytes | str]] = {}
    for row in record["functions"]:
        name = row["name"]
        snippet = snippets.get(name)
        if snippet is None:
            expected[name] = ("snippet-mismatch", "no snippet for this function")
            continue
        if hashlib.sha256(snippet.encode("utf-8")).hexdigest() != row["snippet_sha256"]:
            expected[name] = ("snippet-mismatch", "the snippet is not the recorded one")
            continue
        if row["address"] is None or any(value is None for value in row["symbols"].values()):
            expected[name] = ("link-failed", "the record has no address for it or for a symbol")
            continue
        try:
            code = expected_bytes(
                snippet,
                int(row["address"], 16),
                {symbol: int(value, 16) for symbol, value in row["symbols"].items()},
                record["compiler"],
                tools,
            )
        except RuntimeError as error:
            expected[name] = ("link-failed", str(error))
            continue
        expected[name] = ("ok", code)
    return expected


def _unrecorded(record: dict, original: Path, rewritten: Path) -> list[dict]:
    """Runs of bytes that differ from the original, or were added, outside the record."""
    recorded = [
        (int(write["address"], 16), int(write["address"], 16) + write["size"])
        for row in record["functions"]
        for write in row["writes"]
        if write["address"] is not None
    ]
    before = check_trampoline.Memory(original)
    after = check_trampoline.Memory(rewritten)
    runs: list[dict] = []
    for start, data in after.chunks:
        for index, value in enumerate(data):
            address = start + index
            old = before.byte(address)
            if old is None:
                kind = "unrecorded-code" if value else None
            else:
                kind = "unrecorded-change" if value != old else None
            if kind is None or any(low <= address < high for low, high in recorded):
                continue
            last = runs[-1] if runs else None
            if last and last["kind"] == kind and last["end"] == address:
                last["end"] = address + 1
            else:
                runs.append({"kind": kind, "start": address, "end": address + 1})
    return [
        {"kind": run["kind"], "address": f"0x{run['start']:x}", "size": run["end"] - run["start"]}
        for run in runs
    ]


def check_applied_bytes(
    record: dict,
    original: Path,
    rewritten: Path,
    snippets: Mapping[str, str],
    tools: patch_dependencies.Tools,
    expected: dict | None = None,
) -> dict:
    """Compare the rewritten image with a fresh compile of every recorded function."""
    if expected is None:
        expected = expected_functions(record, snippets, tools)
    image = check_trampoline.Memory(rewritten)

    functions = []
    for row in record["functions"]:
        status, value = expected[row["name"]]
        entry = {"name": row["name"], "address": row["address"], "size": row["size"]}
        if status != "ok":
            entry.update({"verdict": status, "detail": value})
        else:
            address = int(row["address"], 16)
            found = image.read(address, len(value))
            entry["expected_size"] = len(value)
            if found is None:
                entry.update({"verdict": "byte-mismatch", "detail": "not in the loaded image"})
            elif found != value:
                first = next(i for i in range(len(value)) if found[i] != value[i])
                entry.update(
                    {"verdict": "byte-mismatch", "first_difference": f"0x{address + first:x}"}
                )
            elif row["size"] != len(value):
                entry.update(
                    {"verdict": "byte-mismatch",
                     "detail": f"recorded size {row['size']}, expected {len(value)}"}
                )
            else:
                entry["verdict"] = "match"
        functions.append(entry)

    entry_branch = {"verdict": "none"}
    target = next(
        (row for row in record["functions"]
         if row["role"] == "target" and row["placement"] == "trampoline"),
        None,
    )
    if target is not None and target["address"] is not None:
        source = int(record["target"]["address"], 16)
        wanted = thumb_branch_wide(source, int(target["address"], 16))
        found = image.read(source, len(wanted))
        entry_branch = {
            "address": record["target"]["address"],
            "expected": wanted.hex(),
            "found": found.hex() if found is not None else None,
            "verdict": "match" if found == wanted else "entry-branch-mismatch",
        }

    return {
        "original": str(original),
        "rewritten": str(rewritten),
        "functions": functions,
        "entry_branch": entry_branch,
        "unrecorded": _unrecorded(record, original, rewritten),
    }


def failures(report: dict) -> list[str]:
    """The problems in the report, each once."""
    found = {row["verdict"] for row in report["functions"] if row["verdict"] != "match"}
    if report["entry_branch"]["verdict"] not in ("match", "none"):
        found.add(report["entry_branch"]["verdict"])
    found.update(run["kind"] for run in report["unrecorded"])
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--elf", type=Path, required=True, help="original firmware ELF")
    parser.add_argument("--rewritten", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True, help="patched LLVM IR (.ll)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        tools = patch_dependencies.Tools.from_environment()
        record = json.loads(args.record.read_text(encoding="utf-8"))
        snippets = patch_dependencies.split_module(args.module, tools)
        report = check_applied_bytes(record, args.elf, args.rewritten, snippets, tools)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    problems = failures(report)
    if problems:
        print(f"applied bytes: {', '.join(problems)}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
