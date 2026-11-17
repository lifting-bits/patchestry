#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests of klee_gate.py on made-up KLEE output directories.

The directories are written by hand: KLEE does not run where these tests do.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import klee_gate  # noqa: E402

GATE = Path(__file__).resolve().parent / "klee_gate.py"


class Gate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "klee-last"
        self.out.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def paths(self, count):
        for index in range(1, count + 1):
            (self.out / f"test{index:06}.ktest").write_bytes(b"KTEST")

    def error(self, index, kind, stack=""):
        (self.out / f"test{index:06}.{kind}.err").write_text(
            f"Error: {kind}\nStack:\n{stack}\n", encoding="utf-8"
        )

    def test_missing_directory_is_not_run(self):
        self.assertEqual(klee_gate.judge(self.out / "absent")["verdict"], "not-run")

    def test_empty_directory_is_not_run(self):
        result = klee_gate.judge(self.out)
        self.assertEqual(result["verdict"], "not-run")
        self.assertFalse(klee_gate.meets(result, {"errors": []})[0])

    def test_paths_without_errors_are_proven(self):
        self.paths(3)
        result = klee_gate.judge(self.out, elapsed=2, max_time=30)
        self.assertEqual(result["verdict"], "proven")
        self.assertEqual(set(result["checks"].values()), {"proven"})
        self.assertTrue(klee_gate.meets(result, {"errors": []})[0])

    def test_time_limit_is_incomplete_not_proven(self):
        self.paths(3)
        result = klee_gate.judge(self.out, elapsed=30, max_time=30)
        self.assertEqual(result["verdict"], "incomplete")
        self.assertEqual(set(result["checks"].values()), {"not-run"})
        report = klee_gate.equivalence_report(result)
        self.assertIn("time limit", report["properties"]["P1"]["reason"])

    def test_error_file_fails_a_test_that_expects_none(self):
        self.paths(2)
        self.error(2, "abort")
        result = klee_gate.judge(self.out)
        self.assertEqual(result["verdict"], "failed")
        ok, why = klee_gate.meets(result, {"errors": []})
        self.assertFalse(ok)
        self.assertIn("abort", why)

    def test_expected_error_must_be_found(self):
        self.paths(2)
        ok, why = klee_gate.meets(klee_gate.judge(self.out), {"errors": ["abort"]})
        self.assertFalse(ok)
        self.assertIn("found no errors", why)
        self.error(1, "abort")
        self.assertTrue(klee_gate.meets(klee_gate.judge(self.out), {"errors": ["abort"]})[0])

    def test_another_kind_of_error_is_unexpected(self):
        self.paths(2)
        self.error(1, "abort")
        self.error(2, "ptr")
        self.assertFalse(klee_gate.meets(klee_gate.judge(self.out), {"errors": ["abort"]})[0])

    def test_check_is_read_from_the_stack(self):
        self.paths(4)
        self.error(1, "abort", "#0 in __patchestry_fail_p2 ()\n#1 in __patchestry_path_end (code=66)")
        self.error(2, "abort", "#0 in __patchestry_fail_lock_balance ()")
        result = klee_gate.judge(self.out)
        self.assertEqual(result["checks"]["P2"], "failed")
        self.assertEqual(result["checks"]["lock-balance"], "failed")
        self.assertEqual(result["checks"]["P1"], "not-run")
        report = klee_gate.equivalence_report(result)
        self.assertEqual([row["property"] for row in report["violations"]],
                         ["P2", "lock-balance"])
        self.assertEqual(report["properties"]["P2"]["verdict"], "failed")

    def test_unmodelled_path_is_inconclusive_not_a_failure(self):
        self.paths(2)
        self.error(1, "abort", "#0 in __patchestry_fail_unmodelled ()")
        result = klee_gate.judge(self.out)
        self.assertEqual(result["verdict"], "incomplete")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["inconclusive"], 1)

    def test_expectation_is_read_from_a_test_file(self):
        self.assertEqual(klee_gate.expectation("; KLEE-EXPECT: no-errors\n"), {"errors": []})
        self.assertEqual(klee_gate.expectation("x\n ; KLEE-EXPECT: errors: ptr, abort \n"),
                         {"errors": ["abort", "ptr"]})
        self.assertIsNone(klee_gate.expectation("; RUN: true\n"))
        with self.assertRaises(ValueError):
            klee_gate.expectation("; KLEE-EXPECT: maybe\n")

    def test_cli_exit_codes(self):
        def run(*extra):
            return subprocess.run([sys.executable, str(GATE), "judge", "--klee-out",
                                   str(self.out), *extra], capture_output=True, text=True)
        self.paths(1)
        self.assertEqual(run("--expect", "no-errors").returncode, 0)
        self.assertEqual(run().returncode, 3)
        self.assertEqual(run("--expect", "errors: abort").returncode, 1)
        self.assertEqual(run("--expect", "nonsense").returncode, 2)
        self.error(1, "abort")
        seeded = run("--expect", "no-errors")
        self.assertEqual(seeded.returncode, 1)
        self.assertIn("FAIL", seeded.stdout)


if __name__ == "__main__":
    unittest.main()
