#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests for evidence.py.

Each test writes the reports of one case into a directory, in the shape the
checks write them, and reads the evidence built from that directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import evidence

SCRIPT = Path(__file__).with_name("evidence.py")

INTEGRITY = (
    "bindings", "dependencies", "abi", "reinsertion", "patch-record", "link-addresses",
    "applied-bytes", "rewritten-bindings",
)
EQUIVALENCE = ("P1", "P2", "P3", "P4")


def good_reports() -> dict[str, dict]:
    """The reports of a case every implemented check passes, by file name."""
    binding = {"name": "target", "symbol": "target", "kind": "function", "origin": "original",
               "address": "0x1a4", "verdict": "match"}
    return {
        "case.json": {
            "suite": "unit",
            "case": "good",
            "function": "target",
            "binary": "image.elf",
            "binary_sha256": "ab" * 32,
            "patch": "spec.yaml",
            "fixture": {"path": "target.json", "hand_written": False, "regenerated": True},
            "runtime": {"verdict": "tested", "report": "qemu.log",
                        "detail": "transcript matched under QEMU"},
        },
        "bindings.json": {
            "rewritten_checked": True, "summary": {"match": 1}, "absolute_symbols": {},
            "unresolved_calls": [], "bindings": [binding],
        },
        "dependencies.json": {
            "functions": [{"function": "target", "undefined": ["helper"], "unresolved": []}],
            "writable_state": [], "insertion_order": [], "mutually_dependent": [],
        },
        "abi.json": {
            "image": {"cpu_arch": 10, "fp_arch": 0, "vfp_args": 0},
            "object": {"cpu_arch": 10, "fp_arch": 0, "vfp_args": 0},
            "flags": ["-march=armv7-a", "-mfloat-abi=soft"], "problems": {},
        },
        "trampoline.json": {
            "function": "target", "address": "0x1a4", "size": 8, "extent_source": "symbol",
            "mode": "thumb", "mode_source": "symbol", "destination": "0x1000",
            "verdict": "trampoline", "problems": {},
        },
        "patch_record.json": {
            "version": 1,
            "original": {"path": "image.elf", "sha256": "ab" * 32},
            "target": {"name": "target", "address": "0x1a4", "size": 8, "mode": "thumb"},
            "compiler": {"path": "clang", "version": "clang version 22.1.4",
                         "target": "arm-linux-gnueabihf", "flags": ["-O0", "-mthumb"]},
            "functions": [{"name": "target", "role": "target", "order": 0,
                           "placement": "trampoline", "address": "0x1000", "size": 10,
                           "symbols": {"helper": "0x200"}}],
            "unattributed_writes": [],
        },
        "patch_record_check.json": {"problems": {}, "link_addresses": []},
        "applied_bytes.json": {
            "functions": [{"name": "target", "address": "0x1000", "size": 10,
                           "expected_size": 10, "verdict": "match"}],
            "entry_branch": {"address": "0x1a4", "verdict": "match"},
            "unrecorded": [],
        },
        "static.json": {
            "functions": [{"name": "target", "address": "0x1000", "size": 10, "mode": "thumb",
                           "instructions": 4, "stack_depth": 8,
                           "exits": [{"address": "0x1008", "kind": "return"}],
                           "problems": {}, "not_run": {}, "verdict": "pass"}],
            "original": {"name": "target", "address": "0x1a4", "size": 8, "stack_depth": 8,
                         "instructions": 3, "exits": [], "problems": {}, "not_run": {},
                         "verdict": "pass"},
            "code_bytes": 10, "stack_bytes": 8, "stack_growth": 0,
            "budgets": {
                "code_bytes": {"limit": None, "measured": 10, "verdict": "not-judged"},
                "stack_bytes": {"limit": None, "measured": 8, "verdict": "not-judged"},
            },
            "inlined_copies": [],
        },
    }


class Evidence(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "good"
        self.directory.mkdir()

    def build(self, change=None, drop=()) -> dict:
        reports = good_reports()
        if change:
            change(reports)
        for name in drop:
            del reports[name]
        for name, value in reports.items():
            (self.directory / name).write_text(json.dumps(value), encoding="utf-8")
        return evidence.build_evidence(self.directory)

    def verdicts(self, built: dict, names) -> list[str]:
        return [built["checks"][name]["verdict"] for name in names]

    # What a case with every implemented check passing can claim.

    def test_implemented_checks_of_a_good_case(self):
        built = self.build()
        self.assertEqual(self.verdicts(built, INTEGRITY), ["proven"] * len(INTEGRITY))
        self.assertEqual(built["checks"]["calling-convention"]["verdict"], "proven")
        self.assertEqual(built["checks"]["runtime-transcript"]["verdict"], "tested")
        self.assertEqual(built["groups"]["integrity"], "proven")
        self.assertEqual(built["groups"]["runtime"], "tested")

    def test_equivalence_is_not_run_and_the_case_cannot_pass(self):
        built = self.build()
        self.assertEqual(self.verdicts(built, EQUIVALENCE), ["not-run"] * 4)
        self.assertEqual(built["groups"]["equivalence"], "not-run")
        self.assertEqual(built["verdict"], "not-run")
        for name in EQUIVALENCE:
            self.assertTrue(built["checks"][name]["reason"])

    def test_every_check_has_a_verdict_from_the_fixed_set(self):
        built = self.build()
        self.assertEqual(list(built["checks"]), [name for name, _ in evidence.CHECKS])
        for row in built["checks"].values():
            self.assertIn(row["verdict"], evidence.VERDICTS)
            self.assertIn(row["group"], built["groups"])

    def test_overall_is_the_weakest_verdict(self):
        self.assertEqual(evidence.weakest(["proven", "tested"]), "tested")
        self.assertEqual(evidence.weakest(["proven", "tested", "not-run"]), "not-run")
        self.assertEqual(evidence.weakest(["not-run", "failed", "proven"]), "failed")
        self.assertEqual(evidence.weakest(["proven"]), "proven")
        self.assertEqual(evidence.weakest([]), "not-run")

    # A failing sub-report.

    def failed(self, change, check: str, problem: str) -> dict:
        built = self.build(change)
        self.assertEqual(built["checks"][check]["verdict"], "failed", built["checks"][check])
        self.assertIn(problem, built["checks"][check]["problems"])
        self.assertEqual(built["verdict"], "failed")
        return built

    def test_failing_bindings(self):
        def change(reports):
            reports["bindings.json"]["bindings"][0]["verdict"] = "address-mismatch"
        self.failed(change, "bindings", "address-mismatch")

    def test_failing_dependencies(self):
        def change(reports):
            reports["dependencies.json"]["functions"][0]["unresolved"] = ["helper"]
            reports["dependencies.json"]["writable_state"] = [{"name": "counter"}]
        built = self.failed(change, "dependencies", "unresolved-dependency")
        self.assertIn("writable-state", built["checks"]["dependencies"]["problems"])

    def test_failing_abi(self):
        def change(reports):
            reports["abi.json"]["problems"] = {"float-abi-mismatch": "detail"}
        self.failed(change, "abi", "float-abi-mismatch")

    def test_failing_reinsertion(self):
        def change(reports):
            reports["trampoline.json"]["problems"] = {"span-outside-function": []}
        built = self.failed(change, "reinsertion", "span-outside-function")
        self.assertEqual(built["groups"]["integrity"], "failed")

    def test_failing_patch_record_and_link_addresses(self):
        def change(reports):
            reports["patch_record_check.json"] = {
                "problems": {"size-mismatch": "detail"},
                "link_addresses": ["target was linked with helper at 0x204"],
            }
        built = self.failed(change, "patch-record", "size-mismatch")
        self.assertEqual(built["checks"]["link-addresses"]["verdict"], "failed")

    def test_failing_applied_bytes(self):
        def change(reports):
            reports["applied_bytes.json"]["functions"][0]["verdict"] = "bytes-differ"
        self.failed(change, "applied-bytes", "bytes-differ")

    def test_unrecorded_change_fails_applied_bytes(self):
        def change(reports):
            reports["applied_bytes.json"]["unrecorded"] = [
                {"kind": "unrecorded-change", "address": "0x300", "size": 1}
            ]
        self.failed(change, "applied-bytes", "unrecorded-change")

    def test_binding_moved_in_the_rewritten_image(self):
        def change(reports):
            reports["bindings.json"]["bindings"][0]["verdict"] = "moved"
        built = self.failed(change, "rewritten-bindings", "moved")
        self.assertEqual(built["checks"]["bindings"]["verdict"], "proven")

    def test_separate_rewritten_bindings_report(self):
        def change(reports):
            reports["bindings.json"]["rewritten_checked"] = None
            after = copy.deepcopy(reports["bindings.json"])
            reports["bindings.rewritten.json"] = after
        built = self.build(change)
        self.assertEqual(built["checks"]["rewritten-bindings"]["verdict"], "proven")
        self.assertEqual(built["checks"]["rewritten-bindings"]["report"],
                         "bindings.rewritten.json")

    def test_separate_rewritten_bindings_report_that_differs(self):
        def change(reports):
            reports["bindings.json"]["rewritten_checked"] = None
            after = copy.deepcopy(reports["bindings.json"])
            after["bindings"][0]["address"] = "0x1a8"
            reports["bindings.rewritten.json"] = after
        self.failed(change, "rewritten-bindings", "moved")

    def test_rewritten_image_not_checked_is_not_run(self):
        def change(reports):
            reports["bindings.json"]["rewritten_checked"] = None
        built = self.build(change)
        self.assertEqual(built["checks"]["rewritten-bindings"]["verdict"], "not-run")

    def test_clobbered_register_fails_the_calling_convention(self):
        def change(reports):
            row = reports["static.json"]["functions"][0]
            row["problems"] = {"callee-saved-clobbered": [{"registers": ["r4"]}]}
            row["verdict"] = "fail"
        self.failed(change, "calling-convention", "callee-saved-clobbered")

    def test_function_not_analyzed_is_not_run(self):
        def change(reports):
            row = reports["static.json"]["functions"][0]
            row["not_run"] = {"indirect-jump": ["0x1002: mov pc, r0"]}
            row["verdict"] = "not-run"
        built = self.build(change)
        check = built["checks"]["calling-convention"]
        self.assertEqual(check["verdict"], "not-run")
        self.assertIn("indirect-jump", check["reason"])
        self.assertEqual(built["groups"]["static"], "not-run")

    def test_transcript_mismatch(self):
        def change(reports):
            reports["case.json"]["runtime"] = {"verdict": "failed", "report": "qemu.log",
                                               "detail": "transcript mismatch"}
        self.failed(change, "runtime-transcript", "transcript mismatch")

    def test_binary_not_executed(self):
        def change(reports):
            reports["case.json"]["runtime"] = {
                "verdict": "not-run", "detail": "the binary was not executed"}
        built = self.build(change)
        self.assertEqual(built["checks"]["runtime-transcript"]["verdict"], "not-run")
        self.assertEqual(built["groups"]["runtime"], "not-run")

    def test_runtime_verdict_cannot_be_proven(self):
        def change(reports):
            reports["case.json"]["runtime"]["verdict"] = "proven"
        with self.assertRaises(ValueError):
            self.build(change)

    # Budgets.

    def test_budgets_not_declared_are_not_run(self):
        built = self.build()
        self.assertEqual(built["checks"]["code-size-budget"]["verdict"], "not-run")
        self.assertEqual(built["checks"]["stack-budget"]["verdict"], "not-run")
        self.assertEqual(built["bounds"]["code_bytes"], {"limit": None, "measured": 10})
        self.assertEqual(built["bounds"]["stack_bytes"], {"limit": None, "measured": 8})

    def test_budget_met_and_exceeded(self):
        def change(reports):
            budgets = reports["static.json"]["budgets"]
            budgets["code_bytes"] = {"limit": 64, "measured": 10, "verdict": "pass"}
            budgets["stack_bytes"] = {"limit": 4, "measured": 8, "verdict": "stack-over-budget"}
        built = self.build(change)
        self.assertEqual(built["checks"]["code-size-budget"]["verdict"], "proven")
        self.assertEqual(built["checks"]["stack-budget"]["verdict"], "failed")
        self.assertEqual(built["bounds"]["stack_bytes"], {"limit": 4, "measured": 8})

    # Reports that are not there.

    def test_missing_report_is_not_run(self):
        built = self.build(drop=("applied_bytes.json", "static.json"))
        self.assertEqual(built["checks"]["applied-bytes"]["verdict"], "not-run")
        self.assertEqual(built["checks"]["calling-convention"]["verdict"], "not-run")
        self.assertEqual(built["groups"]["integrity"], "not-run")
        self.assertNotIn("applied_bytes.json", built["reports"])

    def test_unreadable_report_is_not_run(self):
        self.build()
        (self.directory / "trampoline.json").write_text("{ not json", encoding="utf-8")
        built = evidence.build_evidence(self.directory)
        self.assertEqual(built["checks"]["reinsertion"]["verdict"], "not-run")
        self.assertIn("unreadable", built["checks"]["reinsertion"]["reason"])

    def test_empty_directory(self):
        built = evidence.build_evidence(self.directory)
        self.assertEqual({row["verdict"] for row in built["checks"].values()}, {"not-run"})
        self.assertEqual(built["verdict"], "not-run")

    def test_check_stopped_by_the_driver(self):
        def change(reports):
            reports["case.json"]["stopped"] = {
                "check": "bindings", "problem": "unsupported-image",
                "detail": "ELF type is ET_DYN"}
        built = self.build(change, drop=[n for n in good_reports() if n != "case.json"])
        self.assertEqual(built["checks"]["bindings"]["verdict"], "failed")
        self.assertEqual(built["checks"]["bindings"]["problems"], ["unsupported-image"])
        self.assertEqual(built["checks"]["dependencies"]["verdict"], "not-run")
        self.assertEqual(built["verdict"], "failed")

    def test_driver_cannot_stop_an_unknown_check(self):
        def change(reports):
            reports["case.json"]["stopped"] = {"check": "nonsense", "problem": "p", "detail": ""}
        with self.assertRaises(ValueError):
            self.build(change)

    # Reports the evidence was built from.

    def test_reports_are_listed_with_their_digests(self):
        built = self.build()
        self.assertEqual(built["checks"]["reinsertion"]["report"], "trampoline.json")
        digest = hashlib.sha256((self.directory / "trampoline.json").read_bytes()).hexdigest()
        self.assertEqual(built["reports"]["trampoline.json"], digest)
        self.assertEqual(set(built["reports"]), set(good_reports()))

    def test_target_and_measurements(self):
        built = self.build()
        self.assertEqual(built["target"]["function"], "target")
        self.assertEqual(built["target"]["binary_sha256"], "ab" * 32)
        self.assertEqual(built["target"]["mode"], "thumb")
        self.assertEqual(built["target"]["address"], "0x1a4")
        self.assertEqual(built["bounds"]["compiler"]["flags"], ["-O0", "-mthumb"])

    # Assumptions.

    def names(self, built: dict) -> list[str]:
        return [row["name"] for row in built["assumptions"]]

    def test_standing_assumptions(self):
        names = self.names(self.build())
        self.assertIn("lift-faithful", names)
        self.assertIn("compiler-faithful", names)
        self.assertIn("validation-block-absent", names)
        self.assertNotIn("fixtures-hand-written", names)
        self.assertNotIn("fixtures-not-regenerated", names)
        self.assertEqual(len(names), len(set(names)))

    def test_every_assumption_has_a_source_and_a_statement(self):
        for row in self.build()["assumptions"]:
            self.assertTrue(row["name"] and row["source"] and row["detail"], row)

    def test_compiler_faithful_only_when_bytes_were_compared(self):
        self.assertNotIn("compiler-faithful",
                         self.names(self.build(drop=("applied_bytes.json",))))

    def test_hand_written_fixture(self):
        def change(reports):
            reports["case.json"]["fixture"] = {
                "path": "target.json", "hand_written": True, "regenerated": False}
        names = self.names(self.build(change))
        self.assertIn("fixtures-hand-written", names)
        self.assertIn("fixtures-not-regenerated", names)

    def test_extent_and_mode_not_from_symbols(self):
        def change(reports):
            reports["trampoline.json"]["extent_source"] = "function-json"
            reports["trampoline.json"]["mode_source"] = "declared"
        names = self.names(self.build(change))
        self.assertIn("extent-from-lifter", names)
        self.assertIn("mode-declared", names)

    def test_extent_and_mode_from_symbols_assume_nothing(self):
        names = self.names(self.build())
        self.assertNotIn("extent-from-lifter", names)
        self.assertNotIn("mode-declared", names)

    def test_plt_stubs_and_unchecked_addresses(self):
        def change(reports):
            rows = reports["bindings.json"]["bindings"]
            rows.append({"name": "strlen", "kind": "function", "verdict": "plt-stub",
                         "stub": "0xeb20"})
            rows.append({"name": "DAT_00000004", "kind": "object", "verdict": "outside-image",
                         "address": "0x4"})
        built = self.build(change)
        by_name = {row["name"]: row for row in built["assumptions"]}
        self.assertIn("strlen", by_name["plt-stubs-from-fixture"]["detail"])
        self.assertIn("DAT_00000004", by_name["addresses-outside-image"]["detail"])

    def test_assumptions_of_the_driver_are_kept(self):
        def change(reports):
            reports["case.json"]["assumptions"] = [
                {"name": "stubbed-callee:helper", "detail": "helper is modelled"}]
        built = self.build(change)
        row = next(r for r in built["assumptions"] if r["name"] == "stubbed-callee:helper")
        self.assertEqual(row["source"], "case.json")

    def test_warnings_carry_inlined_copies(self):
        def change(reports):
            reports["static.json"]["inlined_copies"] = [
                {"address": "0x400", "instructions": 20, "body_instruction": 0}]
        built = self.build(change)
        self.assertEqual(len(built["warnings"]), 1)
        self.assertIn("0x400", built["warnings"][0])

    # Equivalence of the patched version with the original.

    def equivalence(self, **changes) -> dict:
        report = {
            "engine": "concrete", "inputs": 200,
            "outcomes": {"accepted": 150, "patched-rejected": 50, "timeout": 0},
            "violations": [],
            "properties": {
                **{name: {"verdict": "tested", "reason": None}
                   for name in ("P1", "P2", "P3", "P4")},
                **{name: {"verdict": "not-run", "reason": "nothing is declared"}
                   for name in ("mmio-order", "lock-balance", "reject-hygiene")},
            },
            "pure_callees": [], "substitutions": [],
            "bounds": {"inputs": 200, "pointer_size": 64, "seeds": [0, 200], "stub_values": 8},
            "cut_short": None,
            "harness": {"stubs": [{"name": "read_reg", "kind": "values"},
                                  {"name": "reject", "kind": "reject-noreturn"}],
                        "memory_regions": ["state", "arg0"],
                        "concrete_globals": [], "retargeted": False, "layout_adopted": False,
                        "arguments": [{"index": 0, "kind": "integer", "size": 4}]},
        }
        report.update(changes)
        return report

    def test_compared_on_concrete_inputs_is_tested(self):
        def change(reports):
            reports["equivalence.json"] = self.equivalence()
        built = self.build(change)
        self.assertEqual(self.verdicts(built, ("P1", "P2", "P3", "P4")),
                         ["tested"] * 4)
        self.assertEqual(built["groups"]["equivalence"], "tested")
        self.assertEqual(built["checks"]["P1"]["report"], "equivalence.json")
        self.assertEqual(built["verdict"], "not-run")
        self.assertEqual(built["bounds"]["equivalence"]["inputs"], 200)
        names = self.names(built)
        self.assertIn("inputs-sampled", names)
        self.assertIn("lift-faithful", names)
        self.assertIn("stubbed-callee:read_reg", names)
        self.assertNotIn("stubbed-callee:reject", names)
        self.assertNotIn("retargeted-layout", names)
        self.assertIn("memory-scope", names)
        self.assertIn("locals-by-kind", names)

    def properties(self, firmware: dict | None = None, **changed) -> dict:
        rows = {name: {"verdict": "tested", "reason": None}
                for name in ("P1", "P2", "P3", "P4")}
        for name in ("mmio-order", "lock-balance", "reject-hygiene"):
            rows[name] = {"verdict": "not-run", "reason": "nothing is declared"}
        for name, verdict in changed.items():
            rows[name] = {"verdict": verdict,
                          "reason": "harness not built" if verdict == "not-run" else None}
        rows.update(firmware or {})
        return rows

    def test_memory_difference_fails_p2(self):
        def change(reports):
            reports["equivalence.json"] = self.equivalence(
                violations=[{"seed": 3, "property": "P2", "kind": "memory-differs",
                             "region": "state"}],
                properties=self.properties(P2="failed"))
        built = self.failed(change, "P2", "memory-differs")
        self.assertEqual(built["checks"]["P2"]["facts"]["memory_regions"], ["state"])
        self.assertEqual(built["checks"]["P4"]["verdict"], "tested")

    def test_call_difference_fails_p4(self):
        def change(reports):
            reports["equivalence.json"] = self.equivalence(
                violations=[{"seed": 3, "property": "P4", "kind": "calls-differ", "call": 1}],
                properties=self.properties(P4="failed"))
        built = self.failed(change, "P4", "calls-differ")
        self.assertEqual(built["checks"]["P4"]["facts"]["calls"], ["1"])

    def test_declarations_are_assumptions(self):
        def change(reports):
            report = self.equivalence(
                pure_callees=["strlen"],
                substitutions=[{"original": "strcpy", "patched": "strncpy",
                                "compare_args": 2, "condition": "strlen(src) < n"}])
            report["harness"]["stubs"] = [
                {"name": "strlen", "kind": "model", "pure": True},
                {"name": "sprintf", "kind": "values", "variadic": True},
            ]
            reports["equivalence.json"] = report
        built = self.build(change)
        names = self.names(built)
        for name in ("pure-callee:strlen", "substitution:strcpy->strncpy",
                     "modelled-callee:strlen", "variadic-arguments:sprintf",
                     "stubbed-callee:sprintf"):
            self.assertIn(name, names)
        detail = next(row["detail"] for row in built["assumptions"]
                      if row["name"] == "substitution:strcpy->strncpy")
        self.assertIn("strlen(src) < n", detail)

    def test_firmware_checks_come_from_the_equivalence_report(self):
        def change(reports):
            reports["equivalence.json"] = self.equivalence(
                reject_handlers=["qemu_reject"], noreturn_handlers=["qemu_reject"],
                reject_returns=[-1], reject_effects=["log_reject"], reject_skips=["forward"],
                mmio_regions=[[0x4000C000, 0x4000CFFF]], balance=[["irq_off", "irq_on"]],
                violations=[{"seed": 9, "property": "mmio-order", "kind": "writes-differ"}],
                properties=self.properties(firmware={
                    "mmio-order": {"verdict": "failed", "reason": None},
                    "lock-balance": {"verdict": "tested", "reason": None},
                    "reject-hygiene": {"verdict": "tested", "reason": None},
                }))
        built = self.failed(change, "mmio-order", "writes-differ")
        self.assertEqual(self.verdicts(built, ("lock-balance", "reject-hygiene")),
                         ["tested", "tested"])
        self.assertEqual(built["groups"]["firmware"], "failed")
        names = self.names(built)
        for name in ("reject-path:handler:qemu_reject", "reject-path:return:-1",
                     "reject-path:skipped:forward", "noreturn-handler:qemu_reject",
                     "reject-effect:log_reject", "device-registers",
                     "balanced-pair:irq_off/irq_on"):
            self.assertIn(name, names)
        detail = next(row["detail"] for row in built["assumptions"]
                      if row["name"] == "device-registers")
        self.assertIn("0x4000c000 to 0x4000cfff", detail)

    def test_undeclared_firmware_checks_are_not_run(self):
        def change(reports):
            reports["equivalence.json"] = self.equivalence()
        built = self.build(change)
        self.assertEqual(self.verdicts(built, ("mmio-order", "lock-balance", "reject-hygiene")),
                         ["not-run"] * 3)
        self.assertIn("declared", built["checks"]["mmio-order"]["reason"])

    def test_relift_comes_from_its_report(self):
        def tested(reports):
            reports["relift.json"] = {"verdict": "tested", "reason": None, "violations": [],
                                      "lifted": ["f", "FUN_00001000"],
                                      "outcomes": {"accepted": 200}}
        built = self.build(tested)
        self.assertEqual(built["checks"]["relift"]["verdict"], "tested")
        self.assertEqual(built["groups"]["cross-check"], "tested")
        self.assertEqual(built["checks"]["relift"]["facts"]["lifted"], ["f", "FUN_00001000"])

        def differs(reports):
            reports["relift.json"] = {
                "verdict": "failed", "reason": "differs", "lifted": ["f"], "outcomes": {},
                "violations": [{"seed": 1, "property": "P3", "kind": "return-differs"}]}
        self.failed(differs, "relift", "P3:return-differs")

        def stopped(reports):
            reports["relift.json"] = {"verdict": "not-run", "reason": "ghidra failed",
                                      "violations": [], "lifted": []}
        built = self.build(stopped)
        self.assertEqual(built["checks"]["relift"]["verdict"], "not-run")
        self.assertIn("ghidra", built["checks"]["relift"]["reason"])

    def test_report_without_a_property_is_refused(self):
        def change(reports):
            report = self.equivalence()
            del report["properties"]["P4"]
            reports["equivalence.json"] = report
        with self.assertRaises(ValueError):
            self.build(change)

    def test_violation_fails_the_property(self):
        def change(reports):
            reports["equivalence.json"] = self.equivalence(
                violations=[{"seed": 7, "property": "P1", "kind": "original-rejects"}],
                properties=self.properties(P1="failed"))
        built = self.failed(change, "P1", "original-rejects")
        self.assertEqual(built["checks"]["P1"]["facts"]["first_seeds"], [7])
        self.assertEqual(built["checks"]["P3"]["verdict"], "tested")

    def test_comparison_that_reached_nothing_is_not_run(self):
        def change(reports):
            reports["equivalence.json"] = self.equivalence(
                inputs=0,
                properties=self.properties(P1="not-run", P2="not-run", P3="not-run",
                                           P4="not-run"))
        built = self.build(change)
        self.assertEqual(built["checks"]["P3"]["verdict"], "not-run")
        self.assertIn("harness not built", built["checks"]["P3"]["reason"])
        self.assertNotIn("inputs-sampled", self.names(built))

    def test_concrete_run_cannot_claim_proven(self):
        def change(reports):
            reports["equivalence.json"] = self.equivalence(
                properties=self.properties(P1="proven"))
        with self.assertRaises(ValueError):
            self.build(change)

    def test_retargeted_harness_is_an_assumption(self):
        def change(reports):
            report = self.equivalence()
            report["harness"]["retargeted"] = True
            report["harness"]["concrete_globals"] = ["names"]
            reports["equivalence.json"] = report
        names = self.names(self.build(change))
        self.assertIn("retargeted-layout", names)
        self.assertIn("globals-as-lifted", names)

    # The command.

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(SCRIPT), *arguments],
                              capture_output=True, text=True, check=False)

    def test_command_writes_validation_json(self):
        self.build()
        result = self.run_cli(str(self.directory))
        self.assertEqual(result.returncode, 3, result.stderr)  # not-run
        written = json.loads((self.directory / "validation.json").read_text(encoding="utf-8"))
        self.assertEqual(written["verdict"], "not-run")
        self.assertEqual(written["schema"], evidence.SCHEMA)
        self.assertNotIn("validation.json", written["reports"])

    def test_command_exit_code_on_failure(self):
        def change(reports):
            reports["abi.json"]["problems"] = {"float-abi-mismatch": "detail"}
        self.build(change)
        self.assertEqual(self.run_cli(str(self.directory)).returncode, 1)

    def test_command_on_a_missing_directory(self):
        self.assertEqual(self.run_cli(str(self.directory / "absent")).returncode, 2)

    def test_building_twice_gives_the_same_evidence(self):
        self.build()
        first = evidence.build_evidence(self.directory)
        evidence.write_evidence(self.directory)
        self.assertEqual(evidence.build_evidence(self.directory), first)


if __name__ == "__main__":
    unittest.main()
