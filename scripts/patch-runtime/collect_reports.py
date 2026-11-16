#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Collect the per-case validation reports of a run into tables for analysis.

The serial-qemu runtime (scripts/test-qemu-firmware-runtime.sh) and the
identity run (scripts/test-identity-reinsertion.sh) each write one directory
per case with a JSON report per check.  This reads every case directory under
the given output directories and writes:

| File | One row per | Holds |
| --- | --- | --- |
| `cases.csv` | case | status, target, placement, sizes, frames, counts from each report |
| `functions.csv` | inserted or replaced function | address, size, byte verdict, static verdict, frame, exits |
| `checks.csv` | case and check | group, verdict (`proven`, `tested`, `failed`, `not-run`), the problems or the reason, the report it was read from |
| `assumptions.csv` | case and assumption | name, source, statement |
| `problems.csv` | reported problem or not-run reason | which check, which function, the detail |
| `collected.json` | case | the rows above plus every report as written |

A report a case did not get to is absent, and its columns are empty: the case
was rejected before that check ran.  Check verdicts and assumptions are those
of the case's evidence (`validation.json`, see evidence.py); a case directory
without one gets its evidence built from the reports that are there.  Nothing
is filled in as passed.

Only the standard library and evidence.py are needed.

CLI: collect_reports.py OUTPUT_DIR [OUTPUT_DIR ...] --output DIR
Exit 0 when at least one case was read, 1 when none was, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import evidence

REPORTS = (
    "validation", "bindings", "dependencies", "abi", "trampoline", "patch_record",
    "applied_bytes", "static", "bindings.rewritten",
)

CASE_COLUMNS = (
    "suite", "case", "status", "expected", "detail", "verdict", "integrity", "static",
    "runtime", "equivalence", "firmware", "cross_check", "binary", "function", "mode",
    "placement", "destination", "original_code_bytes", "new_code_bytes", "inserted_functions",
    "inserted_code_bytes", "original_frame_bytes", "new_frame_bytes", "frame_growth",
    "inserted_frame_bytes", "static_verdict", "static_not_run", "inlined_copies",
    "code_budget", "stack_budget", "bindings", "binding_verdicts", "unresolved_calls",
    "references", "incoming_branches", "changed_ranges", "header_changes", "tail_split",
    "entry_branch", "unrecorded_ranges", "unattributed_writes", "abi_flags", "compiler_flags",
    "assumptions",
)
FUNCTION_COLUMNS = (
    "suite", "case", "function", "role", "placement", "order", "address", "size", "references",
    "bytes_verdict", "static_verdict", "frame_bytes", "instructions", "exits", "static_problems",
    "static_not_run",
)
CHECK_COLUMNS = ("suite", "case", "check", "group", "verdict", "problem", "detail", "report")
ASSUMPTION_COLUMNS = ("suite", "case", "name", "source", "detail")
PROBLEM_COLUMNS = ("suite", "case", "check", "function", "kind", "problem", "detail")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_summary(directory: Path) -> dict[str, dict]:
    """Rows of `summary.tsv`, by case name."""
    path = directory / "summary.tsv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["case"]: row for row in csv.DictReader(handle, delimiter="\t")}


def text(value) -> str:
    """A cell: lists joined, mappings as `key=value`, None as empty."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return "; ".join(f"{key}={text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return "; ".join(text(item) for item in value)
    return str(value)


def collect_case(suite: str, directory: Path, summary: dict) -> dict:
    """Everything known about one case directory."""
    reports = {name: read_json(directory / f"{name}.json") for name in REPORTS}
    reports = {name: value for name, value in reports.items() if value is not None}
    validation = reports.get("validation")
    if not validation or validation.get("schema") != evidence.SCHEMA:
        validation = evidence.build_evidence(directory)
    groups = validation["groups"]
    trampoline = reports.get("trampoline", {})
    record = reports.get("patch_record", {})
    static = reports.get("static", {})
    applied = reports.get("applied_bytes", {})
    bindings = reports.get("bindings", {})
    recorded = record.get("functions", [])
    target_row = next((row for row in recorded if row["role"] == "target"), {})
    analysed = {row["name"]: row for row in static.get("functions", [])}
    byte_verdicts = {row["name"]: row["verdict"] for row in applied.get("functions", [])}
    target_static = analysed.get(target_row.get("name"), {})

    case = {
        "suite": suite,
        "case": directory.name,
        "status": summary.get("status") or validation.get("suite_status"),
        "expected": summary.get("expected") or validation.get("suite_expected"),
        "detail": summary.get("detail"),
        "verdict": validation["verdict"],
        "integrity": groups["integrity"],
        "static": groups["static"],
        "runtime": groups["runtime"],
        "equivalence": groups["equivalence"],
        "firmware": groups["firmware"],
        "cross_check": groups["cross-check"],
        "binary": validation["target"]["binary"] or trampoline.get("original"),
        "function": validation["target"]["function"],
        "mode": trampoline.get("mode") or record.get("target", {}).get("mode"),
        "placement": trampoline.get("verdict"),
        "destination": trampoline.get("destination"),
        "original_code_bytes": trampoline.get("size", record.get("target", {}).get("size")),
        "new_code_bytes": target_row.get("size"),
        "inserted_functions": len(recorded) or None,
        "inserted_code_bytes": static.get("code_bytes"),
        "original_frame_bytes": static.get("original", {}).get("stack_depth"),
        "new_frame_bytes": target_static.get("stack_depth"),
        "frame_growth": static.get("stack_growth"),
        "inserted_frame_bytes": static.get("stack_bytes"),
        "static_verdict": None,
        "static_not_run": sorted({r for row in analysed.values() for r in row["not_run"]}),
        "inlined_copies": len(static["inlined_copies"]) if static else None,
        "code_budget": static.get("budgets", {}).get("code_bytes", {}).get("verdict"),
        "stack_budget": static.get("budgets", {}).get("stack_bytes", {}).get("verdict"),
        "bindings": len(bindings["bindings"]) if bindings else None,
        "binding_verdicts": bindings.get("summary"),
        "unresolved_calls": len(bindings["unresolved_calls"]) if bindings else None,
        "references": sum(len(row["symbols"]) for row in recorded) if recorded else None,
        "incoming_branches": len(trampoline["incoming"]) if trampoline else None,
        "changed_ranges": len(trampoline["changed"]) if trampoline else None,
        "header_changes": len(trampoline.get("header_changes", [])) if trampoline else None,
        "tail_split": trampoline.get("tail_split"),
        "entry_branch": applied.get("entry_branch", {}).get("verdict"),
        "unrecorded_ranges": len(applied["unrecorded"]) if applied else None,
        "unattributed_writes": [
            row["kind"] for row in record.get("unattributed_writes", [])
        ] if record else None,
        "abi_flags": reports.get("abi", {}).get("flags"),
        "compiler_flags": record.get("compiler", {}).get("flags"),
        "assumptions": [row["name"] for row in validation["assumptions"]],
    }

    checks = validation["checks"]
    case["static_verdict"] = checks["calling-convention"]["verdict"]
    assumptions = [
        {"suite": suite, "case": directory.name, **row} for row in validation["assumptions"]
    ]

    functions = []
    for row in recorded:
        analysis = analysed.get(row["name"], {})
        functions.append({
            "suite": suite, "case": directory.name, "function": row["name"],
            "role": row["role"], "placement": row["placement"], "order": row["order"],
            "address": row["address"], "size": row["size"], "references": len(row["symbols"]),
            "bytes_verdict": byte_verdicts.get(row["name"]),
            "static_verdict": analysis.get("verdict"),
            "frame_bytes": analysis.get("stack_depth"),
            "instructions": analysis.get("instructions"),
            "exits": [exit_["kind"] for exit_ in analysis.get("exits", [])] or None,
            "static_problems": sorted(analysis.get("problems", {})),
            "static_not_run": sorted(analysis.get("not_run", {})),
        })
    if static:
        original = static["original"]
        functions.append({
            "suite": suite, "case": directory.name, "function": original["name"],
            "role": "original", "placement": "original", "order": None,
            "address": original["address"], "size": original["size"], "references": None,
            "bytes_verdict": None, "static_verdict": original["verdict"],
            "frame_bytes": original["stack_depth"], "instructions": original["instructions"],
            "exits": [exit_["kind"] for exit_ in original["exits"]],
            "static_problems": sorted(original["problems"]),
            "static_not_run": sorted(original["not_run"]),
        })

    check_rows = [
        {"suite": suite, "case": directory.name, "check": name, "group": row["group"],
         "verdict": row["verdict"], "problem": row["problems"], "detail": row["reason"],
         "report": row["report"]}
        for name, row in checks.items()
    ]

    problems = []

    def problem(check: str, function, kind: str, name: str, detail) -> None:
        problems.append({"suite": suite, "case": directory.name, "check": check,
                         "function": function, "kind": kind, "problem": name,
                         "detail": text(detail)})

    for name, detail in trampoline.get("problems", {}).items():
        problem("reinsertion", trampoline.get("function"), "problem", name, detail)
    for row in static.get("functions", []):
        for name, detail in row["problems"].items():
            problem("static", row["name"], "problem", name, detail)
        for name, detail in row["not_run"].items():
            problem("static", row["name"], "not-run", name, detail)
    for row in applied.get("functions", []):
        if row["verdict"] != "match":
            problem("applied-bytes", row["name"], "problem", row["verdict"],
                    row.get("detail") or row.get("first_difference"))
    for run in applied.get("unrecorded", []):
        problem("applied-bytes", None, "problem", run["kind"],
                f"{run['size']} bytes at {run['address']}")
    for row in reports.get("dependencies", {}).get("functions", []):
        if row.get("unresolved"):
            problem("dependencies", row["function"], "problem", "unresolved-dependency",
                    row["unresolved"])
    for row in reports.get("dependencies", {}).get("writable_state", []):
        problem("dependencies", None, "problem", "writable-state", row)
    for name, detail in reports.get("abi", {}).get("problems", {}).items():
        problem("abi", None, "problem", name, detail)
    # Failures only the evidence holds: a check the suite itself stopped at.
    known = {"reinsertion", "calling-convention", "applied-bytes", "dependencies", "abi"}
    for name, row in checks.items():
        if row["verdict"] == "failed" and name not in known:
            for found in row["problems"]:
                problem(name, validation["target"]["function"], "problem", found, row["reason"])
        elif row["verdict"] == "failed" and not any(
            p["check"] in (name, "static") for p in problems
        ):
            for found in row["problems"]:
                problem(name, validation["target"]["function"], "problem", found, row["reason"])

    reports["validation"] = validation
    return {"case": case, "functions": functions, "checks": check_rows, "problems": problems,
            "assumptions": assumptions, "reports": reports}


def collect(directories: list[Path]) -> list[dict]:
    """Every case directory under each output directory."""
    collected = []
    for directory in directories:
        summary = read_summary(directory)
        names = sorted(
            {path.name for path in directory.iterdir() if path.is_dir()
             and any((path / f"{name}.json").exists() for name in (*REPORTS, "case"))}
            | {name for name in summary if (directory / name).is_dir()}
        )
        # Keep the order the suite ran the cases in, then anything else.
        order = [name for name in summary if name in names]
        order += [name for name in names if name not in order]
        for name in order:
            collected.append(collect_case(directory.name, directory / name, summary.get(name, {})))
    return collected


def write_table(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([text(row.get(column)) for column in columns])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directories", type=Path, nargs="+",
                        help="output directory of a run (holds one directory per case)")
    parser.add_argument("--output", type=Path, required=True, help="directory for the tables")
    args = parser.parse_args()
    missing = [str(path) for path in args.directories if not path.is_dir()]
    if missing:
        print(f"error: not a directory: {', '.join(missing)}", file=sys.stderr)
        return 2

    collected = collect(args.directories)
    args.output.mkdir(parents=True, exist_ok=True)
    write_table(args.output / "cases.csv", CASE_COLUMNS, [row["case"] for row in collected])
    write_table(args.output / "functions.csv", FUNCTION_COLUMNS,
                [row for case in collected for row in case["functions"]])
    write_table(args.output / "checks.csv", CHECK_COLUMNS,
                [row for case in collected for row in case["checks"]])
    write_table(args.output / "problems.csv", PROBLEM_COLUMNS,
                [row for case in collected for row in case["problems"]])
    write_table(args.output / "assumptions.csv", ASSUMPTION_COLUMNS,
                [row for case in collected for row in case["assumptions"]])
    (args.output / "collected.json").write_text(
        json.dumps(collected, indent=2) + "\n", encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for row in collected:
        status = row["case"]["status"] or "unknown"
        counts[status] = counts.get(status, 0) + 1
    print(f"{len(collected)} cases from {len(args.directories)} run(s): "
          + ", ".join(f"{count} {status}" for status, count in sorted(counts.items())),
          file=sys.stderr)
    print(f"tables: {args.output}", file=sys.stderr)
    return 0 if collected else 1


if __name__ == "__main__":
    sys.exit(main())
