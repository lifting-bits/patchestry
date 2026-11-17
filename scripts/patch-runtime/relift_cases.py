#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Run the re-lift check on cases a suite has already rewritten.

The serial-qemu and identity suites leave a directory per case with
`rewritten.elf`, the intended IR and `trampoline.json`.  This reads a list of
such cases, runs `check_relift` in each directory (`relift.json`), and builds
the case's evidence again so `validation.json` has the `relift` check.

A case may list seeded harms: single bytes of the inserted code changed in a
copy of the rewritten image.  Each is re-lifted in `<case>--relift-<name>`
and must fail.  That is the gate of the check: a semantic change in the
inserted bytes is found without looking at the patched IR's own toolchain.

Cases file, a JSON list:

    {"suite": "qemu-firmware-runtime",   directory under --builds
     "case": "length",
     "function": "qemu_target_length",
     "intended": "qemu_target_length.patched.ll",
     "whole_image": true,                 lift the inserted helpers too
     "moved": false,                      find the body at the trampoline destination
     "declare": {"mmio_regions": [[low, high]], "balance": [[a, r]],
                 "arg_buffers": {"0": 64}},
     "expect": "tested",
     "note": "why a verdict other than tested is expected",
     "seeded": [{"name": "limit", "address": "0x100a", "byte": "0x41"}]}

Needs Docker (Ghidra runs in the project's image), the Debug or Release
tools, and what `check_equivalence` needs from the environment.

CLI: relift_cases.py --cases FILE --builds DIR [--build-type Debug]
     [--only CASE]
Exit 0 when every verdict is the expected one, 1 otherwise, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elftools.elf.elffile import ELFFile

import check_equivalence
import check_relift
import evidence


def with_byte(image: Path, address: int, value: int, output: Path) -> int:
    """Copy `image` with the byte at virtual `address` set to `value`; returns the old byte."""
    data = bytearray(image.read_bytes())
    with image.open("rb") as handle:
        for segment in ELFFile(handle).iter_segments():
            start = segment["p_vaddr"]
            if segment["p_type"] == "PT_LOAD" and start <= address < start + segment["p_filesz"]:
                offset = segment["p_offset"] + address - start
                old = data[offset]
                data[offset] = value
                output.write_bytes(data)
                return old
    raise ValueError(f"{image}: no loaded file byte at {address:#x}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--builds", type=Path, required=True,
                        help="directory holding the suites' output directories")
    parser.add_argument("--build-type", default="Debug")
    parser.add_argument("--only", action="append", help="run only the named case")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    tool_root = repo_root / "builds" / "default" / "tools"
    lift = check_relift.LiftTools(
        repo_root / "scripts" / "ghidra" / "decompile-headless.sh",
        tool_root / "patchir-decomp" / args.build_type / "patchir-decomp",
        tool_root / "patchir-cir2llvm" / args.build_type / "patchir-cir2llvm",
    )
    try:
        tools = check_equivalence.Tools.from_environment()
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
    except (check_equivalence.MissingTool, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.only:
        cases = [case for case in cases if case["case"] in args.only]

    rows, unexpected = [], 0

    def record(name: str, report: dict, expected: str) -> None:
        nonlocal unexpected
        detail = report["reason"] or ""
        rows.append((name, report["verdict"], expected, detail))
        unexpected += report["verdict"] != expected
        print(f"{name}\t{report['verdict']}\t{expected}\t{detail}", flush=True)

    for case in cases:
        directory = args.builds / case["suite"] / case["case"]
        rewritten = directory / "rewritten.elf"
        if not rewritten.is_file():
            print(f"error: {rewritten} is missing; run the {case['suite']} suite first",
                  file=sys.stderr)
            return 2
        declared = case.get("declare", {})
        options = {
            "mmio_regions": [tuple(row) for row in declared.get("mmio_regions", [])],
            "balance": [tuple(row) for row in declared.get("balance", [])],
            "arg_buffers": {int(index): size for index, size in
                            declared.get("arg_buffers", {}).items()} or None,
            "whole_image": case.get("whole_image", True),
            "address": None,
        }
        if case.get("moved"):
            trampoline = json.loads((directory / "trampoline.json").read_text(encoding="utf-8"))
            options["address"] = int(trampoline["destination"], 0)
        intended = directory / case["intended"]
        report = check_relift.check_relift(
            rewritten, case["function"], intended, directory, tools, lift, **options
        )
        (directory / "relift.json").write_text(json.dumps(report, indent=2) + "\n",
                                               encoding="utf-8")
        if (directory / "case.json").is_file():
            evidence.write_evidence(directory)
        record(f"{case['suite']}/{case['case']}", report, case.get("expect", "tested"))

        for harm in case.get("seeded", []):
            out = args.builds / case["suite"] / f"{case['case']}--relift-{harm['name']}"
            out.mkdir(parents=True, exist_ok=True)
            old = with_byte(rewritten, int(harm["address"], 0), int(harm["byte"], 0),
                            out / "rewritten.elf")
            report = check_relift.check_relift(
                out / "rewritten.elf", case["function"], intended, out, tools, lift, **options
            )
            report["seeded"] = {**harm, "was": f"{old:#04x}"}
            (out / "relift.json").write_text(json.dumps(report, indent=2) + "\n",
                                             encoding="utf-8")
            record(f"{case['suite']}/{out.name}", report, "failed")

    summary = args.builds / "relift-summary.tsv"
    summary.write_text(
        "case\tverdict\texpected\tdetail\n"
        + "".join("\t".join(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"{len(rows)} re-lifts, {unexpected} unexpected; {summary}", file=sys.stderr)
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
