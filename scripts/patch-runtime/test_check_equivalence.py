#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests for check_equivalence.py.

Each test compiles an original and a patched version of one small function to
ARM IR, builds the product harness, and runs it on concrete inputs as ARM code
under QEMU.  Needs PATCHESTRY_LLVM_PREFIX, a built patchir-klee-verifier and
qemu-system-arm; the tests are skipped without them.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import check_equivalence

COMMON = """
int state;
extern int read_reg(int);
void reject(void) __attribute__((noreturn));
void soft_reject(void);
"""

BODY = """
    int sum = read_reg(len) & 0xff;
    for (int i = 0; i < len && i < 16; i++) sum += buf[i];
    state = sum;
    return sum;
}
"""

# Accepts any length from 0 up; returns -1 for a negative one.
ORIGINAL = COMMON + """
int target(unsigned char *buf, int len) {
    if (len < 0) return -1;
""" + BODY

VERSIONS = {
    "identical": ORIGINAL,
    # Input-restricting patches, one per way of rejecting.
    "early_return": COMMON + """
int target(unsigned char *buf, int len) {
    if (len < 0) return -1;
    if (len > 8) return -1;
""" + BODY,
    "noreturn_handler": COMMON + """
int target(unsigned char *buf, int len) {
    if (len < 0) return -1;
    if (len > 8) reject();
""" + BODY,
    "returning_handler": COMMON + """
int target(unsigned char *buf, int len) {
    if (len < 0) return -1;
    if (len > 8) { soft_reject(); return 0; }
""" + BODY,
    # Harms.
    "changed_return": COMMON + """
int target(unsigned char *buf, int len) {
    if (len < 0) return -1;
    int sum = read_reg(len) & 0xff;
    for (int i = 0; i < len && i < 16; i++) sum += buf[i];
    state = sum;
    return sum + 1;
}
""",
    "crashes": COMMON + """
int target(unsigned char *buf, int len) {
    if (len < 0) return -1;
    if (len == 3) *(volatile int *)0xE0000000 = 1;
""" + BODY,
    # The restricted function, and a version that accepts more than it.
    "limit_8": COMMON + """
int target(unsigned char *buf, int len) {
    if (len < 0) return -1;
    if (len > 8) return -1;
""" + BODY,
    "limit_12": COMMON + """
int target(unsigned char *buf, int len) {
    if (len < 0) return -1;
    if (len > 12) return -1;
""" + BODY,
    "always_faults": COMMON + """
int target(unsigned char *buf, int len) {
    return *(volatile int *)0xE0000000 + len + buf[0];
}
""",
    # An address passed as an integer, as lifted code has them.
    "integer_pointer": """
int target(int address, int limit) {
    const char *text = (const char *)address;
    int n = 0;
    while (text[n] != 0 && n < 31) n++;
    return n + (limit & 1);
}
""",
    "needs_pointer": """
extern int *find_entry(int);
int target(int key) {
    if (key < 0) return -1;
    return *find_entry(key);
}
""",
}


class Equivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tools = check_equivalence.Tools.from_environment()
        except check_equivalence.MissingTool as error:
            raise unittest.SkipTest(str(error)) from None
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.modules = {}
        for name, source in VERSIONS.items():
            c_file = cls.directory / f"{name}.c"
            c_file.write_text(source, encoding="utf-8")
            module = cls.directory / f"{name}.ll"
            subprocess.run(
                [str(cls.tools.clang), "--target=armv7a-none-eabi", "-mfloat-abi=soft", "-O0",
                 "-S", "-emit-llvm", str(c_file), "-o", str(module)],
                check=True, capture_output=True,
            )
            cls.modules[name] = module

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def compare(self, original: str, patched: str, **options) -> dict:
        out = self.directory / f"{self._testMethodName}"
        out.mkdir(exist_ok=True)
        options.setdefault("inputs", 150)
        options.setdefault("pointer_size", 32)
        return check_equivalence.check_equivalence(
            self.modules[original], self.modules[patched], "target", out, self.tools, **options
        )

    def verdicts(self, report: dict) -> tuple[str, str]:
        return report["properties"]["P1"]["verdict"], report["properties"]["P3"]["verdict"]

    def kinds(self, report: dict, name: str) -> set[str]:
        return {row["kind"] for row in report["violations"] if row["property"] == name}

    # Versions that do no harm.

    def test_identical_versions(self):
        report = self.compare("identical", "identical")
        self.assertEqual(self.verdicts(report), ("tested", "tested"), report["outcomes"])
        self.assertEqual(report["violations"], [])
        self.assertGreater(report["outcomes"]["accepted"], 0)
        self.assertEqual(report["engine"], "concrete")
        self.assertEqual(report["inputs"], 150)

    def test_restriction_by_early_return(self):
        report = self.compare("identical", "early_return", reject_returns=[-1])
        self.assertEqual(self.verdicts(report), ("tested", "tested"), report["outcomes"])
        self.assertGreater(report["outcomes"]["patched-rejected"], 0)
        self.assertGreater(report["outcomes"]["accepted"], 0)

    def test_restriction_by_handler_that_does_not_return(self):
        report = self.compare("identical", "noreturn_handler", reject_handlers=["reject"],
                              reject_returns=[-1])
        self.assertEqual(self.verdicts(report), ("tested", "tested"), report["outcomes"])
        self.assertGreater(report["outcomes"]["patched-rejected"], 0)

    def test_restriction_by_handler_that_returns(self):
        report = self.compare("identical", "returning_handler",
                              reject_handlers=["soft_reject"], reject_returns=[-1])
        self.assertEqual(self.verdicts(report), ("tested", "tested"), report["outcomes"])

    # Harms.

    def test_undeclared_restriction_fails_p3(self):
        # The same patch with no reject value declared: the return value changed.
        report = self.compare("identical", "early_return")
        self.assertEqual(report["properties"]["P3"]["verdict"], "failed")
        self.assertEqual(self.kinds(report, "P3"), {"return-differs"})

    def test_widened_condition_fails_p1(self):
        report = self.compare("limit_8", "limit_12", reject_returns=[-1])
        self.assertEqual(report["properties"]["P1"]["verdict"], "failed", report["outcomes"])
        self.assertEqual(self.kinds(report, "P1"), {"original-rejects"})
        row = report["violations"][0]
        self.assertIn("seed", row)

    def test_narrowed_condition_passes(self):
        report = self.compare("limit_12", "limit_8", reject_returns=[-1])
        self.assertEqual(self.verdicts(report), ("tested", "tested"), report["outcomes"])

    def test_changed_return_fails_p3(self):
        report = self.compare("identical", "changed_return", reject_returns=[-1])
        self.assertEqual(report["properties"]["P3"]["verdict"], "failed")
        self.assertEqual(report["properties"]["P1"]["verdict"], "tested")

    def test_patched_version_that_crashes_fails_p3(self):
        report = self.compare("identical", "crashes", reject_returns=[-1])
        self.assertEqual(report["properties"]["P3"]["verdict"], "failed", report["outcomes"])
        self.assertEqual(self.kinds(report, "P3"), {"patched-faults"})

    def test_original_that_crashes_fails_p1(self):
        report = self.compare("crashes", "identical", reject_returns=[-1])
        self.assertEqual(report["properties"]["P1"]["verdict"], "failed", report["outcomes"])
        self.assertEqual(self.kinds(report, "P1"), {"original-faults"})

    # What cannot be decided.

    def test_no_input_reaches_the_comparison(self):
        report = self.compare("always_faults", "always_faults")
        self.assertEqual(self.verdicts(report), ("not-run", "not-run"), report["outcomes"])
        self.assertEqual(report["violations"], [])
        self.assertTrue(report["properties"]["P1"]["reason"])

    def test_external_function_that_returns_a_pointer(self):
        report = self.compare("needs_pointer", "needs_pointer", reject_returns=[-1])
        self.assertEqual(self.verdicts(report), ("not-run", "not-run"), report["outcomes"])
        self.assertGreater(report["outcomes"]["unmodelled"], 0)
        self.assertEqual(report["harness"]["stubs"], [{"kind": "unmodelled", "name": "find_entry"}])

    def test_harness_that_cannot_be_built(self):
        report = self.compare("identical", "identical", reject_handlers=["absent"])
        self.assertEqual(self.verdicts(report), ("not-run", "not-run"))
        self.assertIn("absent", report["properties"]["P1"]["reason"])
        self.assertEqual(report["inputs"], 0)

    # Inputs and bounds.

    def test_integer_argument_declared_as_buffer(self):
        report = self.compare("integer_pointer", "integer_pointer", arg_buffers={0: 32})
        self.assertEqual(self.verdicts(report), ("tested", "tested"), report["outcomes"])
        self.assertEqual(report["harness"]["arguments"][0],
                         {"index": 0, "kind": "buffer", "size": 32})

    def test_report_states_bounds_and_stubs(self):
        report = self.compare("identical", "identical")
        self.assertEqual(report["bounds"]["inputs"], 150)
        self.assertEqual(report["bounds"]["pointer_size"], 32)
        self.assertEqual(report["harness"]["stubs"], [{"kind": "values", "name": "read_reg"}])
        self.assertEqual(report["harness"]["symbolic_globals"], ["state"])
        self.assertFalse(report["harness"]["retargeted"])
        self.assertEqual(sum(report["outcomes"].values()), 150)

    def test_same_seeds_give_the_same_outcomes(self):
        first = self.compare("limit_8", "limit_12", reject_returns=[-1])
        second = self.compare("limit_8", "limit_12", reject_returns=[-1])
        self.assertEqual(first["outcomes"], second["outcomes"])
        self.assertEqual(first["violations"], second["violations"])


if __name__ == "__main__":
    unittest.main()
