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
int other;
extern int read_reg(int);
extern void note(int);
extern void other_note(int);
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

COPY = """
int state;
char kept[128];
extern char *strcpy(char *, const char *);
extern char *strncpy(char *, const char *, unsigned);
extern unsigned strlen(const char *);
"""
SWAP = {"original": "strcpy", "patched": "strncpy", "compare_args": 2,
        "condition": "strlen(src) < n"}

# A device register written twice with interrupts off.
DEVICE = """
#define REG (*(volatile unsigned *) 0x4000C000u)
#define STATUS (*(volatile unsigned *) 0x4000C004u)
extern void irq_off(void);
extern void irq_on(void);
extern void log_reject(int);
void reject(void);
int state;
"""
SEND = """
    irq_off();
    REG = byte;
    REG = 10;
    irq_on();
    return 0;
}
"""
UART = [(0x4000C000, 0x4000CFFF)]
IRQ = [("irq_off", "irq_on")]

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
    # More harms: what the version writes and whom it calls.
    "extra_global_write": COMMON + """
int target(unsigned char *buf, int len) {
    if (len < 0) return -1;
    other = len;
""" + BODY,
    "buffer_write": COMMON + """
int target(unsigned char *buf, int len) {
    if (len < 0) return -1;
    int sum = read_reg(len) & 0xff;
    for (int i = 0; i < len && i < 16; i++) sum += buf[i];
    state = sum;
    buf[0] = 7;
    return sum;
}
""",
    "extra_call": COMMON + """
int target(unsigned char *buf, int len) {
    if (len < 0) return -1;
    note(len);
""" + BODY,
    "altered_call": COMMON + """
int target(unsigned char *buf, int len) {
    if (len < 0) return -1;
    int sum = read_reg(len + 1) & 0xff;
    for (int i = 0; i < len && i < 16; i++) sum += buf[i];
    state = sum;
    return sum;
}
""",
    # A copy into a local, bounded by the patch.
    "copy_original": COPY + """
int target(const char *text, int unused) {
    char local[128];
    strcpy(local, text);
    state = local[0];
    return local[0];
}
""",
    "copy_patched": COPY + """
int target(const char *text, int unused) {
    char local[128];
    if (strlen(text) >= 16) return -1;
    strncpy(local, text, 16);
    state = local[0];
    return local[0];
}
""",
    # The same swap with a global as the destination: strncpy pads it.
    "copy_global_original": COPY + """
int target(const char *text, int unused) {
    strcpy(kept, text);
    return kept[0];
}
""",
    "copy_global_patched": COPY + """
int target(const char *text, int unused) {
    if (strlen(text) >= 16) return -1;
    strncpy(kept, text, 16);
    return kept[0];
}
""",
    # A reject by leaving out the second call.
    "forward_original": COMMON + """
int target(unsigned char *buf, int len) {
    note(len + 1);
    other_note(len * 2);
    return 0;
}
""",
    "forward_patched": COMMON + """
int target(unsigned char *buf, int len) {
    note(len + 1);
    if ((unsigned) len * 2 <= 32) other_note(len * 2);
    return 0;
}
""",
    "send_original": DEVICE + "int target(int byte) {" + SEND,
    "send_patched": DEVICE + "int target(int byte) { if (byte >= 97) reject();" + SEND,
    "send_reordered": DEVICE + """
int target(int byte) {
    irq_off();
    REG = 10;
    REG = byte;
    irq_on();
    return 0;
}
""",
    "send_missing_enable": DEVICE + """
int target(int byte) {
    irq_off();
    REG = byte;
    REG = 10;
    return 0;
}
""",
    "send_reject_inside": DEVICE + """
int target(int byte) {
    irq_off();
    if (byte >= 97) { irq_on(); return -1; }
    REG = byte;
    REG = 10;
    irq_on();
    return 0;
}
""",
    "send_skips_enable": DEVICE + """
int target(int byte) {
    irq_off();
    if (byte >= 97) return -1;
    REG = byte;
    REG = 10;
    irq_on();
    return 0;
}
""",
    "send_reject_writes_device": DEVICE
    + "int target(int byte) { if (byte >= 97) { REG = 33; reject(); }" + SEND,
    "send_reject_writes_memory": DEVICE
    + "int target(int byte) { if (byte >= 97) { state = 1; return -1; }" + SEND,
    "send_reject_logs": DEVICE
    + "int target(int byte) { if (byte >= 97) { log_reject(byte); return -1; }" + SEND,
    "status_read": DEVICE + """
int target(int byte) {
    if (STATUS & 1) return 1;
    REG = byte;
    return 0;
}
""",
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

    def all_verdicts(self, report: dict) -> list[str]:
        return [report["properties"][name]["verdict"] for name in ("P1", "P2", "P3", "P4")]

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
        self.assertEqual(self.all_verdicts(report), ["tested"] * 4)
        self.assertEqual(report["harness"]["memory_regions"], ["state", "other", "arg0"])

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

    def test_extra_global_write_fails_p2(self):
        report = self.compare("identical", "extra_global_write", reject_returns=[-1])
        self.assertEqual(self.all_verdicts(report), ["tested", "failed", "tested", "tested"],
                         report["outcomes"])
        self.assertEqual(self.kinds(report, "P2"), {"memory-differs"})
        self.assertEqual({row["region"] for row in report["violations"]}, {"other"})

    def test_write_into_argument_buffer_fails_p2(self):
        report = self.compare("identical", "buffer_write", reject_returns=[-1])
        self.assertEqual(self.all_verdicts(report), ["tested", "failed", "tested", "tested"],
                         report["outcomes"])
        self.assertEqual({row["region"] for row in report["violations"]}, {"arg0"})

    def test_extra_call_fails_p4(self):
        report = self.compare("identical", "extra_call", reject_returns=[-1])
        self.assertEqual(self.all_verdicts(report), ["tested", "tested", "tested", "failed"],
                         report["outcomes"])
        self.assertEqual(self.kinds(report, "P4"), {"calls-differ"})
        self.assertEqual({row["call"] for row in report["violations"]}, {0})

    def test_altered_call_argument_fails_p4(self):
        report = self.compare("identical", "altered_call", reject_returns=[-1])
        self.assertEqual(self.all_verdicts(report), ["tested", "tested", "tested", "failed"],
                         report["outcomes"])

    # Declared relaxations of the call comparison.

    def test_declared_substitution_passes(self):
        report = self.compare("copy_original", "copy_patched", reject_returns=[-1],
                              pure_callees=["strlen"], substitutions=[SWAP])
        self.assertEqual(self.all_verdicts(report), ["tested"] * 4, report["outcomes"])
        self.assertGreater(report["outcomes"]["patched-rejected"], 0)
        self.assertGreater(report["outcomes"]["accepted"], 0)
        kinds = {row["name"]: row["kind"] for row in report["harness"]["stubs"]}
        self.assertEqual(kinds, {"strcpy": "model", "strncpy": "model", "strlen": "model"})

    def test_undeclared_substitution_fails_p4(self):
        report = self.compare("copy_original", "copy_patched", reject_returns=[-1],
                              pure_callees=["strlen"])
        self.assertEqual(self.all_verdicts(report), ["tested", "tested", "tested", "failed"],
                         report["outcomes"])

    def test_call_not_declared_pure_fails_p4(self):
        report = self.compare("copy_original", "copy_patched", reject_returns=[-1],
                              substitutions=[SWAP])
        self.assertEqual(report["properties"]["P4"]["verdict"], "failed", report["outcomes"])

    def test_substitution_does_not_hide_a_memory_difference(self):
        # strncpy pads the destination with zeros; strcpy does not.
        report = self.compare("copy_global_original", "copy_global_patched",
                              reject_returns=[-1], pure_callees=["strlen"],
                              substitutions=[SWAP])
        self.assertEqual(report["properties"]["P4"]["verdict"], "tested", report["outcomes"])
        self.assertEqual(report["properties"]["P2"]["verdict"], "failed", report["outcomes"])
        self.assertEqual({row["region"] for row in report["violations"]}, {"kept"})

    def test_substitution_without_condition_is_refused(self):
        report = self.compare("copy_original", "copy_patched", reject_returns=[-1],
                              pure_callees=["strlen"],
                              substitutions=[{**SWAP, "condition": " "}])
        self.assertEqual(self.all_verdicts(report), ["not-run"] * 4)
        self.assertIn("condition", report["properties"]["P4"]["reason"])

    def test_strict_callee_cannot_be_relaxed(self):
        report = self.compare("copy_original", "copy_patched", reject_returns=[-1],
                              pure_callees=["strlen"], substitutions=[SWAP],
                              strict_callees=["strcpy"])
        self.assertEqual(self.all_verdicts(report), ["not-run"] * 4)
        self.assertIn("strict", report["properties"]["P4"]["reason"])

    def test_pure_callee_nobody_calls_is_refused(self):
        report = self.compare("identical", "identical", pure_callees=["absent"])
        self.assertEqual(self.all_verdicts(report), ["not-run"] * 4)
        self.assertIn("absent", report["properties"]["P1"]["reason"])

    def test_reject_by_skipping_a_declared_call(self):
        report = self.compare("forward_original", "forward_patched",
                              reject_skips=["other_note"])
        self.assertEqual(self.all_verdicts(report), ["tested"] * 4, report["outcomes"])
        self.assertGreater(report["outcomes"]["patched-skipped"], 0)
        self.assertGreater(report["outcomes"]["accepted"], 0)
        self.assertTrue(report["reject_exercised"])

    def test_skipped_call_that_is_not_declared_fails_p4(self):
        report = self.compare("forward_original", "forward_patched")
        self.assertEqual(self.all_verdicts(report), ["tested", "tested", "tested", "failed"],
                         report["outcomes"])
        other = self.compare("forward_original", "forward_patched", reject_skips=["note"])
        self.assertEqual(other["properties"]["P4"]["verdict"], "failed", other["outcomes"])

    # Firmware checks.

    def firmware(self, report: dict) -> list[str]:
        return [report["properties"][name]["verdict"]
                for name in ("mmio-order", "lock-balance", "reject-hygiene")]

    def send(self, patched: str, **options) -> dict:
        options.setdefault("mmio_regions", UART)
        options.setdefault("balance", IRQ)
        return self.compare("send_original", patched, **options)

    def test_same_device_writes(self):
        report = self.send("send_original")
        self.assertEqual(self.all_verdicts(report), ["tested"] * 4, report["outcomes"])
        self.assertEqual(self.firmware(report), ["tested", "tested", "not-run"])
        self.assertEqual(report["harness"]["device_accesses"], 4)

    def test_restricting_patch_passes_the_firmware_checks(self):
        report = self.send("send_patched", reject_handlers=["reject"],
                           noreturn_handlers=["reject"])
        self.assertEqual(self.all_verdicts(report), ["tested"] * 4, report["outcomes"])
        self.assertEqual(self.firmware(report), ["tested"] * 3, report["outcomes"])
        self.assertGreater(report["outcomes"]["patched-rejected"], 0)

    def test_reordered_register_writes_fail_mmio_order(self):
        report = self.send("send_reordered")
        self.assertEqual(self.all_verdicts(report), ["tested"] * 4, report["outcomes"])
        self.assertEqual(report["properties"]["mmio-order"]["verdict"], "failed")
        self.assertEqual(self.kinds(report, "mmio-order"), {"writes-differ"})

    def test_undeclared_device_is_not_seen(self):
        report = self.send("send_reordered", mmio_regions=[])
        self.assertEqual(self.all_verdicts(report), ["tested"] * 4, report["outcomes"])
        self.assertEqual(report["properties"]["mmio-order"]["verdict"], "not-run")
        self.assertIn("declared", report["properties"]["mmio-order"]["reason"])

    def test_missing_enable_fails_lock_balance(self):
        report = self.send("send_missing_enable")
        self.assertEqual(report["properties"]["lock-balance"]["verdict"], "failed")
        self.assertEqual(self.kinds(report, "lock-balance"), {"balance-differs"})
        self.assertEqual(report["properties"]["P4"]["verdict"], "failed")

    def test_reject_that_restores_the_balance_passes(self):
        report = self.send("send_reject_inside", reject_returns=[-1])
        self.assertEqual(self.firmware(report), ["tested"] * 3, report["outcomes"])

    def test_early_return_that_skips_the_enable_fails_lock_balance(self):
        report = self.send("send_skips_enable", reject_returns=[-1])
        self.assertEqual(report["properties"]["lock-balance"]["verdict"], "failed",
                         report["outcomes"])
        self.assertEqual(self.kinds(report, "lock-balance"), {"unbalanced-on-reject"})
        # The reject paths were seen, and wrote and called nothing else.
        self.assertEqual(report["properties"]["reject-hygiene"]["verdict"], "tested")
        self.assertGreater(report["outcomes"]["reject-violation"], 0)

    def test_device_write_on_a_reject_path_fails_reject_hygiene(self):
        report = self.send("send_reject_writes_device", reject_handlers=["reject"],
                           noreturn_handlers=["reject"])
        self.assertEqual(report["properties"]["reject-hygiene"]["verdict"], "failed",
                         report["outcomes"])
        self.assertEqual(self.kinds(report, "reject-hygiene"), {"device-written"})
        self.assertEqual(self.all_verdicts(report), ["tested"] * 4)

    def test_memory_write_on_a_reject_path_fails_reject_hygiene(self):
        report = self.send("send_reject_writes_memory", reject_returns=[-1])
        self.assertEqual(self.kinds(report, "reject-hygiene"), {"memory-written"})
        self.assertEqual({row.get("region") for row in report["violations"]}, {"state"})

    def test_call_on_a_reject_path_fails_unless_declared(self):
        report = self.send("send_reject_logs", reject_returns=[-1])
        self.assertEqual(self.kinds(report, "reject-hygiene"), {"call-made"})
        declared = self.send("send_reject_logs", reject_returns=[-1],
                             reject_effects=["log_reject"])
        self.assertEqual(self.firmware(declared), ["tested"] * 3, declared["outcomes"])

    def test_handler_that_returns_runs_on(self):
        # Declared as a handler but not as one that does not return: the
        # patched version goes on to write the device after rejecting.
        report = self.send("send_patched", reject_handlers=["reject"])
        self.assertEqual(report["properties"]["reject-hygiene"]["verdict"], "failed")

    def test_device_read_returns_the_same_series_to_both(self):
        report = self.compare("status_read", "status_read", mmio_regions=UART)
        self.assertEqual(self.all_verdicts(report), ["tested"] * 4, report["outcomes"])
        self.assertEqual(report["properties"]["mmio-order"]["verdict"], "tested")

    def test_balanced_callee_nobody_calls_is_refused(self):
        report = self.compare("identical", "identical", balance=[("lock", "unlock")])
        self.assertEqual(self.firmware(report), ["not-run"] * 3)
        self.assertIn("lock", report["properties"]["P1"]["reason"])

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
        self.assertEqual(report["harness"]["symbolic_globals"], ["state", "other"])
        self.assertFalse(report["harness"]["retargeted"])
        self.assertEqual(sum(report["outcomes"].values()), 150)

    def test_same_seeds_give_the_same_outcomes(self):
        first = self.compare("limit_8", "limit_12", reject_returns=[-1])
        second = self.compare("limit_8", "limit_12", reject_returns=[-1])
        self.assertEqual(first["outcomes"], second["outcomes"])
        self.assertEqual(first["violations"], second["violations"])


if __name__ == "__main__":
    unittest.main()
