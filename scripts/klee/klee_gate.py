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
| `incomplete` | no error, but the time limit was reached or a path was inconclusive |
| `not-run` | the directory is missing or holds no result |

`proven` is bounded: it covers every path KLEE explored within the harness's
bounds (buffer sizes, stub values, loop limits), and nothing beyond them.

A test states what it expects in a comment:

    ; KLEE-EXPECT: no-errors
    ; KLEE-EXPECT: errors: abort

`no-errors` holds when KLEE kept at least one path and none ended in an error.
`errors:` lists the kinds that must be found, and no other kind may be.  A test
with no such comment is not judged.

A product harness linked with lib/patchestry/klee/models/compare_models.c
aborts from a function named after the check that failed.  KLEE's error file
prints the stack, so the check is read from the name: `checks` in the result
maps each of P1 to P4 and the firmware checks to `failed` or to the verdict
of the run.

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

# Function in an error file's stack -> the check it stands for.
FAILS = {
    "__patchestry_fail_p1": "P1",
    "__patchestry_fail_p2": "P2",
    "__patchestry_fail_p3": "P3",
    "__patchestry_fail_p4": "P4",
    "__patchestry_fail_mmio_order": "mmio-order",
    "__patchestry_fail_lock_balance": "lock-balance",
    "__patchestry_fail_reject_hygiene": "reject-hygiene",
    "__patchestry_fail_unmodelled": "unmodelled",
}
CHECKS = ("P1", "P2", "P3", "P4", "mmio-order", "lock-balance", "reject-hygiene")
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
    result = {"directory": str(klee_out), "tests": 0, "errors": [], "inconclusive": 0,
              "timed_out": False, "verdict": "not-run", "checks": {}}
    if not klee_out.is_dir():
        return result
    result["tests"] = sum(1 for _ in klee_out.glob("*.ktest"))
    for path in sorted(klee_out.glob("*.err")):
        parts = path.name.split(".")
        text = path.read_text(encoding="utf-8", errors="replace")
        # The longest name first: one is a prefix of none, but be exact.
        check = next((FAILS[name] for name in sorted(FAILS, key=len, reverse=True)
                      if re.search(rf"\b{name}\b", text)), None)
        row = {"file": path.name, "kind": parts[-2] if len(parts) >= 3 else "unknown",
               "check": check}
        if check == "unmodelled":
            result["inconclusive"] += 1
        else:
            result["errors"].append(row)
    result["timed_out"] = bool(
        elapsed is not None and max_time is not None and elapsed >= max_time
    )
    if result["errors"]:
        result["verdict"] = "failed"
    elif result["tests"] == 0 and result["inconclusive"] == 0:
        result["verdict"] = "not-run"
    elif result["timed_out"] or result["inconclusive"]:
        result["verdict"] = "incomplete"
    else:
        result["verdict"] = "proven"
    failed = {row["check"] for row in result["errors"] if row["check"]}
    # A harness error with no check named (a memory error inside a version,
    # say) is a failure of the run, and of no single check.
    rest = "proven" if result["verdict"] == "proven" else "not-run"
    result["checks"] = {name: "failed" if name in failed else rest for name in CHECKS}
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


def equivalence_report(result: dict, harness: dict | None = None) -> dict:
    """The run as `equivalence.json`, for evidence.py.

    A check is `proven` only when the whole run is; after a time limit or an
    inconclusive path the checks that did not fail are `not-run`, and the
    concrete runner's `tested` is the fallback.
    """
    reason = None
    if result["verdict"] == "incomplete":
        reason = ("KLEE reached its time limit" if result["timed_out"]
                  else f"{result['inconclusive']} paths were inconclusive")
    elif result["verdict"] == "not-run":
        reason = "KLEE left no result"
    elif result["verdict"] == "failed":
        reason = "the run found a failure; the paths were not all explored without one"
    return {
        "engine": "klee",
        "inputs": 0,
        "outcomes": {"paths-kept": result["tests"], "errors": len(result["errors"]),
                     "inconclusive": result["inconclusive"]},
        "violations": [
            {"seed": None, "property": row["check"], "kind": f"klee-{row['kind']}",
             "file": row["file"]}
            for row in result["errors"] if row["check"]
        ],
        "properties": {
            name: {"verdict": verdict, "reason": None if verdict != "not-run" else reason}
            for name, verdict in result["checks"].items()
        },
        "bounds": {"complete": result["verdict"] == "proven"},
        "harness": harness,
        "cut_short": reason if result["timed_out"] else None,
    }


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
