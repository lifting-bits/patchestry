#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Judge one KLEE run against what its test expects.

KLEE leaves an output directory with one `*.ktest` file per path it kept and
one `*.<kind>.err` file per path that ended in an error (`abort`, `assert`,
`ptr`, ...).  This reads that directory and gives the run a verdict:

| Verdict | Meaning |
| --- | --- |
| `failed` | at least one path ended in an error |
| `proven` | no error, the run was not cut short, and at least one path was kept |
| `incomplete` | no error, but the time limit was reached |
| `not-run` | the directory is missing or holds no result |

`proven` is bounded: it covers every path KLEE explored within the harness's
bounds (buffer sizes, stub values, loop limits), and nothing beyond them.

A test states what it expects in a comment:

    ; KLEE-EXPECT: no-errors
    ; KLEE-EXPECT: errors: abort

`no-errors` holds when KLEE kept at least one path and none ended in an error.
`errors:` lists the kinds that must be found, and no other kind may be.  A test
with no such comment is not judged.

CLI: klee_gate.py judge --klee-out DIR [--test FILE | --expect TEXT]
     [--elapsed SECONDS --max-time SECONDS] [--json FILE]
Exit 0 when the run is what the test expects, 1 when it is not, 3 when the
test expects nothing, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DIRECTIVE = re.compile(r"^\s*;\s*KLEE-EXPECT:\s*(.+?)\s*$", re.MULTILINE)


def expectation(text: str) -> dict | None:
    """What a test file expects of its KLEE run, or None."""
    found = DIRECTIVE.search(text)
    if not found:
        return None
    return parse_expectation(found.group(1))


def parse_expectation(directive: str) -> dict:
    if directive == "no-errors":
        return {"errors": []}
    if directive.startswith("errors:"):
        kinds = sorted(kind.strip() for kind in directive[len("errors:"):].split(",")
                       if kind.strip())
        if kinds:
            return {"errors": kinds}
    raise ValueError(f"KLEE-EXPECT: cannot read {directive!r}")


def judge(klee_out: Path, elapsed: float | None = None, max_time: float | None = None) -> dict:
    """Read a KLEE output directory."""
    result = {"directory": str(klee_out), "tests": 0, "errors": [],
              "timed_out": False, "verdict": "not-run"}
    if not klee_out.is_dir():
        return result
    result["tests"] = sum(1 for _ in klee_out.glob("*.ktest"))
    for path in sorted(klee_out.glob("*.err")):
        parts = path.name.split(".")
        result["errors"].append(
            {"file": path.name, "kind": parts[-2] if len(parts) >= 3 else "unknown"}
        )
    result["timed_out"] = bool(
        elapsed is not None and max_time is not None and elapsed >= max_time
    )
    if result["errors"]:
        result["verdict"] = "failed"
    elif result["tests"] == 0:
        result["verdict"] = "not-run"
    elif result["timed_out"]:
        result["verdict"] = "incomplete"
    else:
        result["verdict"] = "proven"
    return result


def meets(result: dict, expected: dict) -> tuple[bool, str]:
    """Whether a judged run is what the test expects, and why not."""
    kinds = sorted({row["kind"] for row in result["errors"]})
    if result["verdict"] == "not-run":
        return False, "KLEE left no result"
    if kinds != expected["errors"]:
        return False, (f"expected {expected['errors'] or 'no errors'}, "
                       f"found {kinds or 'no errors'}")
    if not expected["errors"] and result["tests"] == 0:
        return False, "KLEE kept no path"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("judge", help="judge one KLEE output directory")
    one.add_argument("--klee-out", type=Path, required=True)
    one.add_argument("--test", type=Path, help="test file with a KLEE-EXPECT comment")
    one.add_argument("--expect", help="the expectation itself, as after KLEE-EXPECT:")
    one.add_argument("--elapsed", type=float)
    one.add_argument("--max-time", type=float)
    one.add_argument("--json", type=Path, help="write the judged run here")
    args = parser.parse_args()
    try:
        if args.expect is not None:
            expected = parse_expectation(args.expect)
        elif args.test is not None:
            expected = expectation(args.test.read_text(encoding="utf-8"))
        else:
            expected = None
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    result = judge(args.klee_out, args.elapsed, args.max_time)
    result["expected"] = expected
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    kinds = sorted({row["kind"] for row in result["errors"]})
    print(f"verdict: {result['verdict']}; paths kept: {result['tests']}; "
          f"errors: {', '.join(kinds) or 'none'}")
    if expected is None:
        print("not judged: the test has no KLEE-EXPECT comment")
        return 3
    ok, why = meets(result, expected)
    if not ok:
        print(f"FAIL: {why}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
