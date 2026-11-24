#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Compare the stack frames Ghidra recovered with the frames of the recompiled
lifted functions.

Inputs are the Ghidra JSON the lifter consumed (it must carry the `frame`
record on functions and `storage` on local declarations) and the LLVM IR the
lifter produced from it.  The IR is compiled with `-fstack-usage`, which
makes the backend write one line per function with its frame size and whether
the frame is static or dynamic.

One row per lifted function:

  recovered    frame size Ghidra recovered, in bytes
  recompiled   frame size of the recompiled function, in bytes
  frame        `static`, `dynamic` or `dynamic,bounded` as the backend reports
  caller_reads locals read before the function defines them that live in the
               caller's frame (stack-passed values the prototype does not
               cover)
  verdict      relative-layout-suffices: the function only reaches its own
                 frame through its own locals, which the lifter keeps at their
                 relative offsets where that matters
               needs-exact-placement: the function reads fixed offsets in its
                 caller's frame; no relative layout reproduces that
               dynamic-frame: the recompiled frame is not a fixed size
               not-compiled: the function has no body in the IR

With `--strict` the exit status is 1 when any function needs exact placement
or has a dynamic frame.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def strip_comments(text):
    """Drop the `//` header lines lit fixtures carry before the JSON."""
    return "\n".join(
        line for line in text.split("\n") if not line.lstrip().startswith("//")
    )


def caller_frame_reads(function):
    """Names of live-in locals stored in the caller's frame."""
    grows_negative = function.get("frame", {}).get("grows_negative", True)
    names = []
    for block in function.get("basic_blocks", {}).values():
        for operation in block.get("operations", {}).values():
            if operation.get("mnemonic") != "DECLARE_LOCAL" or not operation.get("live_in"):
                continue
            for piece in operation.get("storage", []):
                if piece.get("kind") != "stack":
                    continue
                offset = piece.get("offset", 0)
                if (offset >= 0) if grows_negative else (offset < 0):
                    names.append(operation.get("name", "?"))
                    break
    return names


def stack_usage(clang, ll_path, opt_level, extra_flags):
    """Compile the IR and return {function: (size, qualifier)}."""
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / "frame-audit.o"
        command = [
            str(clang), "-c", f"-{opt_level}", "-fstack-usage", "-Wno-override-module",
            *extra_flags, str(ll_path), "-o", str(obj),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            raise SystemExit(f"frame-audit: compile failed: {' '.join(command)}")
        su_path = obj.with_suffix(".su")
        if not su_path.exists():
            raise SystemExit(f"frame-audit: {clang} wrote no stack usage file")
        usage = {}
        for line in su_path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) != 3:
                continue
            usage[fields[0].rsplit(":", 1)[-1]] = (int(fields[1]), fields[2])
        return usage


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", type=Path, required=True, help="Ghidra JSON the lifter read")
    parser.add_argument("--ll", type=Path, required=True, help="LLVM IR the lifter wrote")
    parser.add_argument("--clang", type=Path, required=True, help="clang used to recompile")
    parser.add_argument("--opt-level", default="O1", choices=["O0", "O1", "O2", "Os", "Oz"])
    parser.add_argument(
        "--flag", action="append", default=[],
        help="extra flag for the compile, e.g. --flag=--target=armv7m-none-eabi (repeatable)",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    program = json.loads(strip_comments(args.json.read_text(encoding="utf-8")))
    usage = stack_usage(args.clang, args.ll, args.opt_level, args.flag)

    rows = []
    failures = 0
    for key in sorted(program.get("functions", {})):
        function = program["functions"][key]
        if not function.get("basic_blocks"):
            continue
        name = function.get("display_name") or function.get("name", key)
        frame = function.get("frame")
        recovered = str(frame["size"]) if frame else "-"
        reads = caller_frame_reads(function)
        if name not in usage:
            verdict, recompiled, qualifier = "not-compiled", "-", "-"
        else:
            size, qualifier = usage[name]
            recompiled = str(size)
            if "dynamic" in qualifier:
                verdict = "dynamic-frame"
            elif reads:
                verdict = "needs-exact-placement"
            else:
                verdict = "relative-layout-suffices"
        if verdict in ("dynamic-frame", "needs-exact-placement"):
            failures += 1
        rows.append((name, recovered, recompiled, qualifier, ",".join(reads) or "-", verdict))

    print("function\trecovered\trecompiled\tframe\tcaller_reads\tverdict")
    for row in rows:
        print("\t".join(row))

    counts = {}
    for row in rows:
        counts[row[-1]] = counts.get(row[-1], 0) + 1
    summary = ", ".join(f"{count} {verdict}" for verdict, count in sorted(counts.items()))
    print(f"frame-audit: {len(rows)} functions: {summary}", file=sys.stderr)
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    sys.exit(main())
