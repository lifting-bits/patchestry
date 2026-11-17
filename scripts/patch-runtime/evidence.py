#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Build the evidence artifact of one validated case: `<case>/validation.json`.

The checks each write a report into the case directory.  This reads those
reports, and nothing else, and states what they establish: a verdict for every
check of the validation plan, the bounds the verdicts hold under, the reports
they were read from, and every assumption by name.

Verdicts:

| Verdict | Meaning |
| --- | --- |
| `proven` | the check decided its claim for everything the claim covers: a comparison of every byte, an analysis of every path.  It holds under the listed assumptions |
| `tested` | the claim held on the inputs that were run, and only on those |
| `failed` | the check found a problem |
| `not-run` | the check did not run, could not decide, or does not exist yet |

The verdict of a group, and of the case, is the weakest verdict in it, in the
order `failed`, `not-run`, `tested`, `proven`.  A check whose report is absent
is `not-run`: nothing is taken as passed because nobody looked.  `groups`
says what each group of checks established.  P1 to
P4 are `tested` when the two versions were run on concrete inputs; `proven`
is accepted only from a KLEE run.

Checks and where each verdict is read from:

| Group | Check | Report |
| --- | --- | --- |
| integrity | `bindings` | `bindings.json` |
| integrity | `dependencies` | `dependencies.json` |
| integrity | `abi` | `abi.json` |
| integrity | `reinsertion` | `trampoline.json` |
| integrity | `patch-record`, `link-addresses` | `patch_record_check.json` |
| integrity | `applied-bytes` | `applied_bytes.json` |
| integrity | `rewritten-bindings` | `bindings.rewritten.json`, or `bindings.json` when it covers the rewritten image |
| static | `calling-convention`, `code-size-budget`, `stack-budget` | `static.json` |
| runtime | `runtime-transcript` | `case.json` (`runtime`), written by the suite |
| equivalence | `P1`, `P2`, `P3`, `P4` | `equivalence.json` (check_equivalence.py) |
| firmware | `mmio-order`, `lock-balance`, `reject-hygiene` | `equivalence.json` |
| cross-check | `relift` | `relift.json` |

`case.json` is what the suite knows and no report holds: the case, the image
and its digest, the patch, where the fixture came from, the outcome of running
the image, and a check the suite itself stopped at (`stopped`).  A suite can
say a run was `tested`, `failed` or `not-run`; it cannot say `proven`.

Assumptions come from the reports: `lift-faithful` whenever lifted code was
used, `compiler-faithful` when bytes were compared against a recompilation,
`extent-from-lifter` and `mode-declared` when the image has no symbol to
confirm them, `plt-stubs-from-fixture`, `addresses-outside-image`,
`fixtures-hand-written`, `fixtures-not-regenerated`, and
`validation-block-absent` while the patch specification declares no intent.
From a run of the equivalence harness: `inputs-sampled`, one
`stubbed-callee:<name>` or `modelled-callee:<name>` per external function,
`variadic-arguments:<name>`, `memory-scope`, `locals-by-kind`,
`globals-as-lifted`, `layout-adopted`, and `retargeted-layout` when the
harness was not run in the image's own layout.  Each declaration that relaxes
the call comparison is one too: `pure-callee:<name>` and
`substitution:<original>-><patched>`, the latter with its condition.  So is
each declaration of how the patch rejects and what the firmware checks look
at: `reject-path:handler:<name>`, `reject-path:return:<value>`,
`reject-path:skipped:<name>`, `noreturn-handler:<name>`,
`reject-effect:<name>`, `device-registers` and `balanced-pair:<a>/<b>`.

Only the standard library is needed.

CLI: evidence.py CASE_DIR [--output validation.json]
Exit 0 when the case is `proven` or `tested`, 1 when `failed`, 3 when
`not-run`, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = 1
VERDICTS = ("failed", "not-run", "tested", "proven")  # weakest first

CHECKS = (
    ("bindings", "integrity"),
    ("dependencies", "integrity"),
    ("abi", "integrity"),
    ("reinsertion", "integrity"),
    ("patch-record", "integrity"),
    ("link-addresses", "integrity"),
    ("applied-bytes", "integrity"),
    ("rewritten-bindings", "integrity"),
    ("calling-convention", "static"),
    ("code-size-budget", "static"),
    ("stack-budget", "static"),
    ("runtime-transcript", "runtime"),
    ("P1", "equivalence"),
    ("P2", "equivalence"),
    ("P3", "equivalence"),
    ("P4", "equivalence"),
    ("mmio-order", "firmware"),
    ("lock-balance", "firmware"),
    ("reject-hygiene", "firmware"),
    ("relift", "cross-check"),
)

CLAIMS = {
    "bindings": "every name the lifted code uses is the entity at that address in the image",
    "dependencies": "the inserted code needs nothing the image does not provide, and no "
                    "writable state of its own",
    "abi": "the inserted code is built for the CPU and calling convention the image declares",
    "reinsertion": "the rewrite changed only the target function, and its entry reaches "
                   "the new body",
    "patch-record": "the record of what the rewriter wrote agrees with the rewritten image",
    "link-addresses": "every reference was linked against the address the binding check found",
    "applied-bytes": "the rewritten image holds the intended code and no other change",
    "rewritten-bindings": "nothing the lifted code is bound to moved in the rewritten image",
    "calling-convention": "every inserted function returns to its caller with the stack "
                          "pointer and the callee-saved registers as it found them",
    "code-size-budget": "the inserted code fits the declared code budget",
    "stack-budget": "the inserted functions' frames fit the declared stack budget",
    "runtime-transcript": "the rewritten image, run, produces the expected output",
    "P1": "any input the patched function accepts, the original accepted",
    "P2": "on accepted inputs, the final non-local memory is the same",
    "P3": "on accepted inputs, the return value is the same",
    "P4": "on accepted inputs, the external calls and their arguments are the same",
    "mmio-order": "writes to declared device regions keep their order",
    "lock-balance": "locks and interrupt masks are balanced on every exit",
    "reject-hygiene": "a reject path writes nothing it did not declare",
    "relift": "the function lifted back from the rewritten image matches the intended code",
}

BINDING_FAILURES = {
    "address-mismatch", "kind-mismatch", "not-executable", "unmapped", "imported", "unexplained",
}
SUITE_RUNTIME_VERDICTS = ("tested", "failed", "not-run")
# Every report the evidence reads; a suite removes stale ones before a case runs.
REPORT_FILES = (
    "bindings.json", "bindings.rewritten.json", "dependencies.json", "abi.json",
    "trampoline.json", "patch_record.json", "patch_record_check.json", "applied_bytes.json",
    "static.json", "equivalence.json", "relift.json",
)


def weakest(verdicts) -> str:
    """The weakest of the verdicts; `not-run` when there are none."""
    verdicts = list(verdicts)
    if not verdicts:
        return "not-run"
    return min(verdicts, key=VERDICTS.index)


class Reports:
    """The JSON reports of a case directory, read once, with what was read."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.used: dict[str, str] = {}
        self.unreadable: set[str] = set()
        self._cache: dict[str, dict | None] = {}

    def get(self, name: str) -> dict | None:
        if name not in self._cache:
            path = self.directory / name
            value = None
            if path.is_file():
                data = path.read_bytes()
                try:
                    value = json.loads(data)
                except ValueError:
                    self.unreadable.add(name)
                if not isinstance(value, dict):
                    value = None
                    self.unreadable.add(name)
                else:
                    self.used[name] = hashlib.sha256(data).hexdigest()
            self._cache[name] = value
        return self._cache[name]

    def absent(self, name: str) -> str:
        if name in self.unreadable:
            return f"{name} is unreadable"
        return f"{name} was not written: the check did not run"


def _row(verdict: str, report: str | None = None, problems=(), reason: str | None = None,
         facts: dict | None = None) -> dict:
    return {"verdict": verdict, "report": report, "problems": sorted(set(problems)),
            "reason": reason, "facts": facts or {}}


def _decided(report: str, problems, facts: dict | None = None) -> dict:
    problems = list(problems)
    return _row("failed" if problems else "proven", report, problems, facts=facts)


def _bindings(reports: Reports) -> dict:
    report = reports.get("bindings.json")
    if report is None:
        return _row("not-run", reason=reports.absent("bindings.json"))
    found = {row["verdict"] for row in report["bindings"]} & BINDING_FAILURES
    return _decided("bindings.json", found, {
        "verdicts": report.get("summary", {}),
        "unresolved_calls": len(report.get("unresolved_calls", [])),
    })


def _dependencies(reports: Reports) -> dict:
    report = reports.get("dependencies.json")
    if report is None:
        return _row("not-run", reason=reports.absent("dependencies.json"))
    problems = []
    unresolved = sorted({name for row in report["functions"] for name in row.get("unresolved", [])})
    if unresolved or report.get("mutually_dependent"):
        problems.append("unresolved-dependency")
    if report.get("writable_state"):
        problems.append("writable-state")
    return _decided("dependencies.json", problems, {
        "functions": [row["function"] for row in report["functions"]],
        "unresolved": unresolved,
        "writable_state": [row["name"] for row in report.get("writable_state", [])],
    })


def _abi(reports: Reports) -> dict:
    report = reports.get("abi.json")
    if report is None:
        return _row("not-run", reason=reports.absent("abi.json"))
    return _decided("abi.json", report["problems"], {"flags": report.get("flags", [])})


def _reinsertion(reports: Reports) -> dict:
    report = reports.get("trampoline.json")
    if report is None:
        return _row("not-run", reason=reports.absent("trampoline.json"))
    return _decided("trampoline.json", report["problems"], {
        "placement": report.get("verdict"), "destination": report.get("destination"),
    })


def _record(reports: Reports, key: str, problem: str | None) -> dict:
    report = reports.get("patch_record_check.json")
    if report is None:
        return _row("not-run", reason=reports.absent("patch_record_check.json"))
    if key not in report:
        return _row("not-run", "patch_record_check.json",
                    reason=f"patch_record_check.json has no '{key}'")
    found = report[key]
    if problem is None:
        return _decided("patch_record_check.json", found)
    row = _decided("patch_record_check.json", [problem] if found else [])
    row["facts"] = {"mismatches": list(found)}
    return row


def _applied_bytes(reports: Reports) -> dict:
    report = reports.get("applied_bytes.json")
    if report is None:
        return _row("not-run", reason=reports.absent("applied_bytes.json"))
    problems = {row["verdict"] for row in report["functions"] if row["verdict"] != "match"}
    if report["entry_branch"]["verdict"] not in ("match", "none"):
        problems.add(report["entry_branch"]["verdict"])
    problems.update(run["kind"] for run in report["unrecorded"])
    return _decided("applied_bytes.json", problems, {
        "functions": len(report["functions"]),
        "entry_branch": report["entry_branch"]["verdict"],
    })


def _rewritten_bindings(reports: Reports) -> dict:
    before = reports.get("bindings.json")
    after = reports.get("bindings.rewritten.json")
    if before is not None and after is not None:
        def place(row: dict):
            return row["verdict"], row.get("address"), row.get("stub")

        moved = len(before["bindings"]) != len(after["bindings"]) or any(
            place(old) != place(new)
            for old, new in zip(before["bindings"], after["bindings"])
        )
        return _decided("bindings.rewritten.json", ["moved"] if moved else [])
    if before is not None and before.get("rewritten_checked"):
        moved = any(row["verdict"] == "moved" for row in before["bindings"])
        return _decided("bindings.json", ["moved"] if moved else [])
    if before is None:
        return _row("not-run", reason=reports.absent("bindings.json"))
    return _row("not-run", "bindings.json",
                reason="the bindings were not checked against the rewritten image")


def _calling_convention(reports: Reports) -> dict:
    report = reports.get("static.json")
    if report is None:
        return _row("not-run", reason=reports.absent("static.json"))
    problems = {name for row in report["functions"] for name in row["problems"]}
    facts = {
        "functions": {row["name"]: row["verdict"] for row in report["functions"]},
        "original": report["original"]["verdict"],
        "stack_growth": report.get("stack_growth"),
    }
    if problems:
        return _row("failed", "static.json", problems, facts=facts)
    skipped = sorted(
        f"{row['name']}: {reason}" for row in report["functions"] for reason in row["not_run"]
    )
    if skipped:
        return _row("not-run", "static.json", facts=facts,
                    reason="not analyzed: " + "; ".join(skipped))
    return _row("proven", "static.json", facts=facts)


def _equivalence(reports: Reports, name: str) -> dict:
    """P1 to P4 or a firmware check, from a run of the product harness."""
    report = reports.get("equivalence.json")
    if report is None:
        return _row("not-run", reason="equivalence.json was not written: the original and "
                                      "the patched version were not compared")
    if name not in report.get("properties", {}):
        raise ValueError(f"equivalence.json: no verdict for {name}")
    judged = report["properties"][name]
    # Concrete inputs test; only a complete symbolic exploration proves.
    if judged["verdict"] not in ("tested", "failed", "not-run") and not (
        judged["verdict"] == "proven" and report.get("engine") == "klee"
    ):
        raise ValueError(
            f"equivalence.json: verdict {judged['verdict']!r} from engine "
            f"{report.get('engine')!r} is not accepted"
        )
    kinds = sorted({row["kind"] for row in report["violations"] if row["property"] == name})
    facts = {
        "engine": report.get("engine"),
        "inputs": report.get("inputs"),
        "outcomes": {key: count for key, count in report.get("outcomes", {}).items() if count},
    }
    if judged["verdict"] == "failed":
        seeds = [row["seed"] for row in report["violations"] if row["property"] == name]
        facts["first_seeds"] = seeds[:5]
        where = sorted({
            str(row[key]) for row in report["violations"]
            for key in ("region", "call")
            if row["property"] == name and row.get(key) is not None
        })
        if where:
            facts["memory_regions" if name == "P2" else "calls"] = where
        return _row("failed", "equivalence.json", kinds, facts=facts)
    return _row(judged["verdict"], "equivalence.json", reason=judged.get("reason"), facts=facts)


def _relift(reports: Reports) -> dict:
    """The function lifted back from the rewritten image, against the intended code."""
    report = reports.get("relift.json")
    if report is None:
        return _row("not-run", reason="relift.json was not written: the rewritten image "
                                      "was not lifted again")
    if report["verdict"] not in ("tested", "failed", "not-run"):
        raise ValueError(f"relift.json: verdict {report['verdict']!r} is not accepted")
    facts = {"lifted": report.get("lifted", []), "outcomes": report.get("outcomes", {})}
    if report["verdict"] == "failed":
        kinds = sorted({f"{row['property']}:{row['kind']}" for row in report["violations"]})
        return _row("failed", "relift.json", kinds, facts=facts)
    return _row(report["verdict"], "relift.json", reason=report.get("reason"), facts=facts)


def _budget(reports: Reports, key: str) -> dict:
    report = reports.get("static.json")
    if report is None:
        return _row("not-run", reason=reports.absent("static.json"))
    budget = report["budgets"][key]
    facts = {"limit": budget["limit"], "measured": budget["measured"]}
    if budget["limit"] is None:
        return _row("not-run", "static.json", facts=facts,
                    reason="no budget is declared; the size is measured, not judged")
    if budget["measured"] is None:
        return _row("not-run", "static.json", facts=facts,
                    reason="the size could not be measured")
    return _decided("static.json", [] if budget["verdict"] == "pass" else [budget["verdict"]],
                    facts)


def _runtime(context: dict) -> dict:
    runtime = context.get("runtime")
    if not runtime:
        return _row("not-run", reason="the suite recorded no run of the rewritten image")
    verdict = runtime.get("verdict")
    if verdict not in SUITE_RUNTIME_VERDICTS:
        raise ValueError(
            f"case.json: runtime verdict {verdict!r} is not one of {SUITE_RUNTIME_VERDICTS}"
        )
    detail = runtime.get("detail") or ""
    if verdict == "failed":
        return _row("failed", runtime.get("report"), [detail or "run failed"])
    return _row(verdict, runtime.get("report"),
                reason=detail if verdict == "not-run" else None,
                facts={"detail": detail} if verdict == "tested" else None)


def _assumptions(reports: Reports, context: dict) -> list[dict]:
    rows: list[dict] = []

    def assume(name: str, source: str, detail: str) -> None:
        if all(row["name"] != name for row in rows):
            rows.append({"name": name, "source": source, "detail": detail})

    bindings = reports.get("bindings.json")
    trampoline = reports.get("trampoline.json")
    fixture = context.get("fixture") or {}
    if (bindings is not None or reports.get("dependencies.json") is not None
            or reports.get("equivalence.json") is not None):
        assume("lift-faithful", "pipeline",
               "the lifted code is taken as the meaning of the function's machine code; "
               "the lifter and the decompiler are not checked")
    if reports.get("applied_bytes.json") is not None:
        assume("compiler-faithful", "applied_bytes.json",
               "the bytes are compared against a second compilation of the same code with "
               "the same compiler; the step from IR to machine code is not checked")
    if trampoline is not None and trampoline.get("extent_source") not in (None, "symbol"):
        assume("extent-from-lifter", "trampoline.json",
               f"the extent of {trampoline.get('function')} is the lifter's address range "
               f"({trampoline['extent_source']}); the image has no symbol size to confirm it")
    if trampoline is not None and trampoline.get("mode_source") not in (None, "symbol"):
        assume("mode-declared", "trampoline.json",
               f"{trampoline.get('function')} is taken to be {trampoline.get('mode')}-mode "
               f"code ({trampoline['mode_source']}); the image has no symbol to confirm it")
    if bindings is not None:
        stubs = sorted(row["name"] for row in bindings["bindings"]
                       if row["verdict"] == "plt-stub")
        if stubs:
            assume("plt-stubs-from-fixture", "bindings.json",
                   "imports are reached through the stub addresses the fixture names, each "
                   "confirmed to resolve to that import: " + ", ".join(stubs))
        outside = sorted(row["name"] for row in bindings["bindings"]
                         if row["verdict"] == "outside-image")
        if outside:
            assume("addresses-outside-image", "bindings.json",
                   "these names are addresses no segment of the image maps; nothing about "
                   "them is checked: " + ", ".join(outside))
    if fixture.get("hand_written"):
        assume("fixtures-hand-written", "case.json",
               f"the function JSON ({fixture.get('path')}) was written by hand, not "
               "produced by the lifter from this image")
    if fixture and not fixture.get("regenerated"):
        assume("fixtures-not-regenerated", "case.json",
               f"the function JSON ({fixture.get('path')}) is a checked-in file; it was "
               "not lifted from the image in this run")
    assume("validation-block-absent", "patch specification",
           "the patch specification declares no reject paths, strict callees, device "
           "regions or budgets; every check that needs declared intent is not-run")
    equivalence = reports.get("equivalence.json")
    harness = (equivalence or {}).get("harness") or {}
    if equivalence is not None and equivalence.get("inputs"):
        assume("inputs-sampled", "equivalence.json",
               f"the versions were compared on {equivalence['inputs']} concrete inputs "
               "derived from seeds, not on every input; a difference on another input "
               "is not excluded")
    for stub in harness.get("stubs", []):
        if stub["kind"] in ("values", "void"):
            assume(f"stubbed-callee:{stub['name']}", "equivalence.json",
                   f"{stub['name']} is not run; it returns the same value at the same call "
                   "index in both versions, whatever its arguments, and writes nothing")
        if stub["kind"] == "model":
            assume(f"modelled-callee:{stub['name']}", "equivalence.json",
                   f"{stub['name']} is run from the harness's own implementation, not the "
                   "image's")
        if stub.get("variadic") and not stub.get("pure"):
            assume(f"variadic-arguments:{stub['name']}", "equivalence.json",
                   f"only the fixed arguments of {stub['name']} are compared")
    for name in (equivalence or {}).get("pure_callees", []):
        assume(f"pure-callee:{name}", "equivalence.json",
               f"declared: calls of {name} are left out of the call comparison")
    for swap in (equivalence or {}).get("substitutions", []):
        assume(f"substitution:{swap['original']}->{swap['patched']}", "equivalence.json",
               f"declared: a call of {swap['patched']} by the patched version counts as a "
               f"call of {swap['original']}, compared on the first {swap['compare_args']} "
               f"arguments, when {swap['condition']}; the condition is not checked")
    declared = equivalence or {}
    for name in declared.get("reject_handlers", []):
        assume(f"reject-path:handler:{name}", "equivalence.json",
               f"declared: a call of {name} means the input was rejected")
    for value in declared.get("reject_returns", []):
        assume(f"reject-path:return:{value}", "equivalence.json",
               f"declared: a return of {value} means the input was rejected")
    for name in declared.get("reject_skips", []):
        assume(f"reject-path:skipped:{name}", "equivalence.json",
               f"declared: the patched version rejects an input by not calling {name}; "
               "on such an input only the other calls are compared")
    for name in declared.get("noreturn_handlers", []):
        assume(f"noreturn-handler:{name}", "equivalence.json",
               f"declared: {name} does not return")
    for name in declared.get("reject_effects", []):
        assume(f"reject-effect:{name}", "equivalence.json",
               f"declared: {name} may be called before a rejection")
    if declared.get("mmio_regions"):
        assume("device-registers", "equivalence.json",
               "declared: the device registers are "
               + ", ".join(f"{low:#x} to {high:#x}" for low, high in declared["mmio_regions"])
               + "; a write there is recorded and not performed, a read returns a value "
               "from a series both versions share, and no other address is a device")
    for acquire, release in declared.get("balance", []):
        assume(f"balanced-pair:{acquire}/{release}", "equivalence.json",
               f"declared: calls of {acquire} and {release} must balance; neither is run")
    if "memory_regions" in harness:
        assume("memory-scope", "equivalence.json",
               "P2 compares " + (", ".join(harness["memory_regions"]) or "nothing")
               + "; writes through an address computed from a constant, and globals that "
               "hold pointers, are not compared")
    if any(stub["kind"] not in ("reject", "reject-noreturn")
           for stub in harness.get("stubs", [])):
        assume("locals-by-kind", "equivalence.json",
               "the address of a local passed to an external function is compared only "
               "as the address of a local; what the local holds is not compared")
    if harness.get("concrete_globals"):
        assume("globals-as-lifted", "equivalence.json",
               "these globals keep their lifted initial value instead of an arbitrary one: "
               + ", ".join(harness["concrete_globals"]))
    if harness.get("retargeted"):
        assume("retargeted-layout", "equivalence.json",
               "the versions were compared in a 64-bit layout, not the image's own")
    if harness.get("layout_adopted"):
        assume("layout-adopted", "equivalence.json",
               "the patched module was given the original's data layout; the two agree on "
               "endianness, pointer size and the alignment of 64-bit values")
    for row in context.get("assumptions", []):
        assume(row["name"], "case.json", row["detail"])
    return rows


def build_evidence(directory: Path) -> dict:
    """The evidence for the case whose reports are in `directory`."""
    reports = Reports(directory)
    context = reports.get("case.json") or {}
    names = [name for name, _ in CHECKS]

    checks = {
        "bindings": _bindings(reports),
        "dependencies": _dependencies(reports),
        "abi": _abi(reports),
        "reinsertion": _reinsertion(reports),
        "patch-record": _record(reports, "problems", None),
        "link-addresses": _record(reports, "link_addresses", "link-address-mismatch"),
        "applied-bytes": _applied_bytes(reports),
        "rewritten-bindings": _rewritten_bindings(reports),
        "calling-convention": _calling_convention(reports),
        "code-size-budget": _budget(reports, "code_bytes"),
        "stack-budget": _budget(reports, "stack_bytes"),
        "runtime-transcript": _runtime(context),
    }
    for name in ("P1", "P2", "P3", "P4", "mmio-order", "lock-balance", "reject-hygiene"):
        checks[name] = _equivalence(reports, name)
    checks["relift"] = _relift(reports)

    stopped = context.get("stopped")
    if stopped:
        if stopped.get("check") not in checks:
            raise ValueError(f"case.json: stopped at unknown check {stopped.get('check')!r}")
        # The suite's own finding: a failure no report holds.  It never
        # improves a verdict.
        row = checks[stopped["check"]]
        if row["verdict"] != "failed":
            checks[stopped["check"]] = _row(
                "failed", row["report"], [stopped["problem"]], stopped.get("detail") or None,
                row["facts"],
            )

    groups: dict[str, list[str]] = {}
    ordered = {}
    for name, group in CHECKS:
        ordered[name] = {"group": group, "claim": CLAIMS[name], **checks[name]}
        groups.setdefault(group, []).append(checks[name]["verdict"])
    assert list(ordered) == names

    static = reports.get("static.json")
    trampoline = reports.get("trampoline.json") or {}
    record = reports.get("patch_record.json") or {}
    budgets = (static or {}).get("budgets", {})
    warnings = [
        f"{copy['instructions']} instructions at {copy['address']} of the original image "
        "repeat the replaced function's body; a copy is not reached by the reinsertion"
        for copy in (static or {}).get("inlined_copies", [])
    ]
    assumptions = _assumptions(reports, context)
    return {
        "schema": SCHEMA,
        "suite": context.get("suite"),
        "case": context.get("case", directory.name),
        "target": {
            "binary": context.get("binary") or record.get("original", {}).get("path"),
            "binary_sha256": context.get("binary_sha256")
            or record.get("original", {}).get("sha256"),
            "function": context.get("function") or trampoline.get("function")
            or record.get("target", {}).get("name"),
            "address": trampoline.get("address") or record.get("target", {}).get("address"),
            "mode": trampoline.get("mode") or record.get("target", {}).get("mode"),
        },
        "patch": context.get("patch"),
        "suite_status": context.get("status"),
        "suite_expected": context.get("expected"),
        "verdict": weakest(row["verdict"] for row in ordered.values()),
        "groups": {group: weakest(verdicts) for group, verdicts in groups.items()},
        "checks": ordered,
        "bounds": {
            "code_bytes": {key: budgets.get("code_bytes", {}).get(key)
                           for key in ("limit", "measured")},
            "stack_bytes": {key: budgets.get("stack_bytes", {}).get(key)
                            for key in ("limit", "measured")},
            "stack_scope": "the inserted functions' own frames; frames of the image's "
                           "functions they call are not included",
            "compiler": record.get("compiler"),
            "equivalence": None if reports.get("equivalence.json") is None else {
                "engine": reports.get("equivalence.json").get("engine"),
                **reports.get("equivalence.json").get("bounds", {}),
                "arguments": (reports.get("equivalence.json").get("harness") or {})
                .get("arguments"),
                "cut_short": reports.get("equivalence.json").get("cut_short"),
            },
        },
        "assumptions": assumptions,
        "warnings": warnings,
        "reports": dict(sorted(reports.used.items())),
    }


def write_evidence(directory: Path, output: Path | None = None) -> dict:
    """Build the evidence for `directory` and write it; returns it."""
    built = build_evidence(directory)
    output = output or directory / "validation.json"
    output.write_text(json.dumps(built, indent=2) + "\n", encoding="utf-8")
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, help="case directory holding the reports")
    parser.add_argument("--output", type=Path, help="default: CASE_DIR/validation.json")
    args = parser.parse_args()
    if not args.directory.is_dir():
        print(f"error: not a directory: {args.directory}", file=sys.stderr)
        return 2
    try:
        built = write_evidence(args.directory, args.output)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    summary = ", ".join(f"{group} {verdict}" for group, verdict in built["groups"].items())
    print(f"{built['case']}: {built['verdict']} ({summary})", file=sys.stderr)
    return {"failed": 1, "not-run": 3}.get(built["verdict"], 0)


if __name__ == "__main__":
    sys.exit(main())
