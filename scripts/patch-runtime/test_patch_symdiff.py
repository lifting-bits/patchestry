#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests for patch_symdiff.py.

Run in the environment the script needs (see its docstring).  The tests that
compare binaries are skipped without cozy, and those that compile need
PATCHESTRY_LLVM_PREFIX (clang) and PATCHESTRY_LD_LLD (ld.lld, defaults to the
one beside clang).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import patch_symdiff as psd

ROOT = Path(__file__).resolve().parents[2]
FIRMWARE = ROOT / "firmwares/serial-qemu/build/serial.elf"
LENGTH = ROOT / "builds/qemu-firmware-runtime/length/rewritten.elf"
LLVM = Path(os.environ.get("PATCHESTRY_LLVM_PREFIX", "/nonexistent"))
LLD = Path(os.environ.get("PATCHESTRY_LD_LLD", LLVM / "ld.lld"))
HAVE_COZY = importlib.util.find_spec("cozy") is not None
HAVE_TOOLS = (LLVM / "clang").exists() and LLD.exists()


def build(directory: Path, name: str, sources: list[str], flags: list[str] = ()) -> Path:
    """A Thumb ELF from C sources, each its own object; the entry is `put`."""
    objects = []
    for index, source in enumerate(sources):
        path = directory / f"{name}{index}.c"
        path.write_text(source, encoding="utf-8")
        objects.append(directory / f"{name}{index}.o")
        subprocess.run([str(LLVM / "clang"), "--target=thumbv7m-none-eabi", "-O1",
                        "-ffreestanding", "-fno-builtin", *flags, "-c", str(path),
                        "-o", str(objects[-1])], check=True)
    elf = directory / f"{name}.elf"
    subprocess.run([str(LLD), "-Ttext=0x8000", "-e", "put", *map(str, objects),
                    "-o", str(elf)], check=True)
    return elf


ADD = "int put(int x) { return x + ADDEND; }\n"
SAVED = "__attribute__((naked)) int put(int x) { __asm__(BODY); }\n"
STORE = "int cell; int put(int x) { cell = x + ADDEND; return 0; }\n"
LENGTH_OF = "int put(const unsigned char *s) { int n = 0; while (s[n]) n++; return n; }\n"
# LIMIT is the largest accepted value; above it the function returns -1.
CHECKED = """
int put(unsigned n) {
#ifdef LIMIT
    if (n > LIMIT) return -1;
#endif
    return (int)(n & 0xff) + ADDEND;
}
"""
CALLER = "int helper(int x); int put(int x) { return helper(x) + ADDEND; }\n"
# FIRST is written, then 10, REPEAT times.  The last value is always 10.
DEVICE = """
#define DATA (*(volatile unsigned *) 0x40001000u)
int put(unsigned x) {
    DATA = x + FIRST;
    for (int i = 0; i < REPEAT; i++) DATA = 10;
    return 0;
}
"""
GUARDED = """
void stop(void);
int put(unsigned n) {
#ifdef LIMIT
    if (n > LIMIT) stop();
#endif
    return n & 0xff;
}
"""
STOP = "void stop(void) { for (;;) { } }\n"
FAULTS = """
int put(int x) {
#ifdef LIMIT
    if (x == LIMIT) *(volatile int *)0 = 0xdead;
#endif
    return x + 1;
}
"""
HELPER = "int helper(int x) { return x * FACTOR; }\n"
TWO_CALLEES = "int first(int x); int second(int x); int put(int x) { return ORDER; }\n"
TWO_HELPERS = "int first(int x) { return x + 1; } int second(int x) { return x * 3; }\n"
PASSES = "int helper(int x); int put(int x) { return helper(x + ADDEND); }\n"
MAYBE_CALLS = "int helper(int x); int put(int x) { USE; return 0; }\n"
PASSES_LOCAL = ("int helper(int *p); int put(int x) "
                "{ volatile int pad[PAD]; int cell = x; pad[0] = 0; return helper(&cell); }\n")
PASSES_TEXT = ("int helper(const char *s); const char pad[PAD] = {1}; "
               "int put(int x) { return helper(TEXT) + pad[x & 1]; }\n")
SKIPS = "void helper(int x); void put(unsigned x) { USE; }\n"
VOID_HELPER = "void helper(int x) { }\n"
THIRD_BYTE = "int put(const char *s) { return s[2] == 'A' ? 1 : ADDEND; }\n"
WIDE_HELPER = "unsigned long long helper(TYPE x) { return 0; }\n"
USES_UPPER_HALF = ("unsigned long long helper(unsigned x); "
                   "unsigned put(void) { return (unsigned)(helper(0) >> 32); }\n")
DROPS_RESULT = ("unsigned long long helper(unsigned x); "
                "unsigned put(void) { helper(0); return 0; }\n")
PASSES_WIDE = ("unsigned long long helper(unsigned long long x); "
               "unsigned put(void) { return (unsigned)helper(VALUE) + 1; }\n")
# Stands for the library function: the comparison never runs a stubbed callee.
STRNCMP = "int strncmp(const char *a, const char *b, unsigned n) { return 0; }\n"
MATCHES = """
int strncmp(const char *a, const char *b, unsigned n);
int put(const char *s) { return strncmp(s, "ab", 2) == 0 EXTRA; }
"""
THROUGH_POINTER = "int put(int x, int (*f)(void)) { if (x) return f() + ADDEND; return 0; }\n"
WIDE_REJECT = """
unsigned long long put(unsigned n) {
    if (n > 8) return REJECTED;
    return n;
}
"""
EITHER_DEVICE = """
int put(unsigned x) {
    volatile unsigned *p = (volatile unsigned *)(0x40001000u + 4 * (x & 1));
    *p = FIRST;
    *p = 0;
    return 0;
}
"""
# A store the processor makes on some inputs only, and the same store made
# on all of them.  r1 is the device register.
STORES = """
__attribute__((naked)) int put(unsigned x) {
    __asm__("movw r1, #0x1000\\n movt r1, #0x4000\\n movs r2, #0\\n cmp r0, #0\\n"
            FIRST_STORE "\\n movs r2, #1\\n str r2, [r1]\\n movs r0, #0\\n bx lr");
}
"""
INDEXED = ("int put(const unsigned char *p, unsigned i) "
           "{ return p[i & MASK] + ((i & MASK) == 4 ? ADDEND : 0); }\n")
SIGNED_REJECT = """
int put(int x) {
#ifdef LIMIT
    if (x > LIMIT) return -1;
#endif
    return x & 0xff;
}
"""
POINTER_CALLER = "int *helper(int x); int put(int x) { return helper(x)[INDEX]; }\n"
POINTER_HELPER = "int table[4]; int *helper(int x) { return table; }\n"


class Pieces(unittest.TestCase):
    def test_verdict(self):
        self.assertEqual(psd.verdict_of([], []), "proven")
        self.assertEqual(psd.verdict_of([], ["a note"]), "incomplete")
        self.assertEqual(psd.verdict_of([{"kind": "memory"}], ["a note"]), "failed")

    def test_arguments(self):
        self.assertEqual(psd.indexed_size("1:0x10"), (1, 16))
        self.assertEqual(psd.indexed_range("0:1:0x20"), (0, (1, 32)))
        self.assertEqual(psd.stub_name("0x8001"), 0x8001)
        self.assertEqual(psd.stub_name("memcpy"), "memcpy")
        self.assertEqual(psd.address_range("0x10:0x1f"), (16, 31))
        with self.assertRaises(ValueError):
            psd.indexed_size("1")

    def test_settings(self):
        psd.Settings(args=2, arg_buffers={0: 8}, arg_ranges={1: (0, 4)}).check()
        for wrong in (psd.Settings(args=5),
                      psd.Settings(args=1, arg_buffers={1: 8}),
                      psd.Settings(args=1, arg_buffers={0: 0}),
                      psd.Settings(args=1, arg_buffers={0: 8}, arg_ranges={0: (0, 1)}),
                      psd.Settings(args=1, arg_ranges={0: (2, 1)}),
                      psd.Settings(mmio_regions=[(2, 1)]),
                      psd.Settings(loop_bound=-1)):
            with self.assertRaises(ValueError):
                wrong.check()

    def test_describe(self):
        line = psd.describe({"kind": "register", "register": "r0", "original": "0x1",
                             "rewritten": "0x2", "input": {"arg0": "0x0"}, "paths": {}})
        self.assertIn("register (r0): 0x1 originally, 0x2 rewritten", line)
        line = psd.describe({"kind": "unmatched-path", "side": "rewritten", "input": None,
                             "paths": {}})
        self.assertIn("rewritten only", line)
        line = psd.describe({"kind": "device-writes", "writes": {"original": 3, "rewritten": 2},
                             "input": {}, "paths": {}})
        self.assertIn("device-writes (3 originally, 2 rewritten)", line)

    def test_skipped(self):
        names = {"check"}
        self.assertTrue(psd.skipped(["a"], ["a", "check"], names))
        self.assertTrue(psd.skipped([], ["check", "check"], names))
        self.assertTrue(psd.skipped(["a", "b"], ["a", "check", "b"], names))
        self.assertFalse(psd.skipped(["a", "check"], ["a", "check"], names))  # nothing left out
        self.assertFalse(psd.skipped(["b"], ["a", "check"], names))  # another call differs
        self.assertFalse(psd.skipped([], ["a"], names))  # what is left out is not named
        self.assertFalse(psd.skipped([], ["check"], set()))

    def test_argument_layout(self):
        layout = psd.argument_layout
        self.assertEqual(layout([]), ((), ""))
        self.assertEqual(layout(["i32 noundef", "ptr"]), (("r0", "r1"), ""))
        self.assertEqual(layout(["i64 noundef"]), (("r0", "r1"), ""))
        # A 64-bit integer starts at an even register; r1 is skipped.
        self.assertEqual(layout(["i32", "i64"]), (("r0", "r2", "r3"), ""))
        self.assertEqual(layout(["ptr", "ptr", "..."]),
                         (("r0", "r1"), "... (variable arguments)"))
        self.assertEqual(layout(["i32", "i32", "i32", "i64"]),
                         (("r0", "r1", "r2"), "i64 (on the stack)"))
        self.assertEqual(layout(["i32", "double noundef"]), (("r0",), "double noundef"))
        self.assertEqual(layout(["ptr byval(%struct.S) align 4"]),
                         ((), "ptr byval(%struct.S) align 4"))

    def test_return_layout(self):
        self.assertEqual(psd.return_layout("void"), ((), ""))
        self.assertEqual(psd.return_layout("ptr"), (("r0",), ""))
        self.assertEqual(psd.return_layout("i1"), (("r0",), ""))
        self.assertEqual(psd.return_layout("i64"), (("r0", "r1"), ""))
        self.assertEqual(psd.return_layout("double"), (("r0",), "double"))
        self.assertEqual(psd.return_layout("{ i32, i32 }"), (("r0",), "{ i32, i32 }"))

    def test_declared_functions(self):
        text = ("declare i32 @f(ptr byval(%struct.S) align 4, { i32, i32 }, i64 noundef)\n"
                "declare void @g()\ndeclare noundef ptr @h(i32, ...)\n"
                "define dso_local { i32, i32 } @k(i32 %0) {\n")
        self.assertEqual(psd.declared_functions(text), {
            "f": ("i32", ["ptr byval(%struct.S) align 4", "{ i32, i32 }", "i64 noundef"]),
            "g": ("void", []), "h": ("ptr", ["i32", "..."]),
        })
        self.assertEqual(psd.declared_functions(text, "define"),
                         {"k": ("{ i32, i32 }", ["i32 %0"])})

    IR = ("declare i64 @wide(i32, i64)\ndeclare i32 @printf(ptr, ...)\n"
          "declare void @check(i32)\ndeclare double @real(i32)\n"
          "define dso_local void @target(i32 %0) {\n")

    def test_declaration_from_ir(self):
        declared = psd.declaration_from_ir(
            {"reject_skips": ["check"], "arg_buffers": {0: 4}}, self.IR, "target",
            {"wide": 0x8000, "printf": "printf", "real": "real", "absent": "absent"})
        # A function that returns nothing: r0 is not compared.
        self.assertEqual(declared["registers"], [*psd.SAVED_REGISTERS, "sp"])
        self.assertEqual(declared["stubs"], [0x8000, "printf", "real", "absent", "check"])
        self.assertEqual(declared["stub_registers"], {
            "0x8000": ["r0", "r2", "r3"], "printf": ["r0"], "real": ["r0"], "check": ["r0"]})
        self.assertEqual(declared["stub_returns"], {
            "0x8000": ["r0", "r1"], "printf": ["r0"], "real": ["r0"], "check": []})
        self.assertIn("variable arguments", declared["stub_notes"]["printf"])
        self.assertIn("double", declared["stub_notes"]["real"])
        self.assertEqual(declared["notes"], [])
        settings = psd.settings_from_declaration(declared)
        settings.check()
        self.assertEqual(settings.stub_registers[0x8000], ("r0", "r2", "r3"))
        self.assertEqual(settings.arg_buffers, {0: 4})
        self.assertEqual(settings.reject_skips, ["check"])

    def test_declaration_of_a_function_with_a_result(self):
        wide = self.IR.replace("void @target", "i64 @target")
        self.assertEqual(psd.declaration_from_ir({}, wide, "target")["registers"][:3],
                         ["r0", "r1", "r4"])
        real = self.IR.replace("void @target", "double @target")
        declared = psd.declaration_from_ir({}, real, "target")
        self.assertEqual(declared["registers"][:2], ["r0", "r4"])
        self.assertIn("double", declared["notes"][0])
        # A function the IR does not define, and registers the case gives, are left alone.
        self.assertNotIn("registers", psd.declaration_from_ir({}, self.IR, "other"))
        self.assertEqual(
            psd.declaration_from_ir({"registers": ["r0"]}, self.IR, "target")["registers"],
            ["r0"])

    def test_settings_from_declaration(self):
        settings = psd.settings_from_declaration({
            "arg_buffers": {"1": 4}, "arg_suffixes": {"1": "39393b"},
            "arg_ranges": {"2": [0, 4]}, "reject_returns": [0xffffffff],
            "mmio_regions": [[0x4000C000, 0x4000CFFF]], "stubs": ["helper", 0x8000],
            "stub_args": {"helper": 2, "0x8000": 1}, "max_seconds": 120,
        })
        self.assertEqual(settings.arg_buffers, {1: 4})
        self.assertEqual(settings.arg_suffixes, {1: b"99;"})
        self.assertEqual(settings.arg_ranges, {2: (0, 4)})
        self.assertEqual(settings.mmio_regions, [(0x4000C000, 0x4000CFFF)])
        self.assertEqual(settings.stub_args, {"helper": 2, 0x8000: 1})
        self.assertEqual(settings.max_seconds, 120)
        settings.check()
        with self.assertRaisesRegex(psd.UsageError, "max_paths"):
            psd.settings_from_declaration({"max_paths": 5})
        with self.assertRaisesRegex(psd.UsageError, "not a buffer"):
            psd.settings_from_declaration({"arg_suffixes": {"1": "00"}}).check()

    def test_stubs_from_bindings(self):
        bindings = {"bindings": [
            {"name": "target", "kind": "function", "address": "0x100"},
            {"name": "strchr", "symbol": "strchr", "kind": "function", "origin": "imported",
             "address": "0x22", "stub": "0x94b0"},
            {"name": "FUN_00000200", "kind": "function", "address": "0x200"},
            {"name": "FUN_00000300", "kind": "function", "address": "0x300"},
            {"name": "DAT_00000400", "kind": "object", "address": "0x400"},
        ]}
        self.assertEqual(psd.stubs_from_bindings(bindings, "target", ["FUN_00000300"]),
                         {"strchr": "strchr", "FUN_00000200": 0x200})

    @unittest.skipUnless(HAVE_COZY, "needs claripy")
    def test_guarded_difference(self):
        import claripy
        x = claripy.BVS("x", 32)
        zero, one = claripy.BVV(0, 32), claripy.BVV(1, 32)

        def differs(first, second, *assumed):
            solver = claripy.Solver()
            solver.add([psd.guarded_difference(claripy, first, second), *assumed])
            return solver.satisfiable()

        always = [("d", zero, None), ("d", one, None)]
        sometimes = [("d", zero, x != 0), ("d", one, None)]
        self.assertFalse(differs(always, always))
        self.assertTrue(differs(sometimes, always))
        self.assertFalse(differs(sometimes, always, x != 0))
        self.assertTrue(differs(sometimes, always, x == 0))
        # An event that never happens is no event.
        self.assertFalse(differs(sometimes, [("d", one, None)], x == 0))
        self.assertFalse(differs([("d", zero, x != x)], []))
        self.assertTrue(differs([("d", zero, None)], []))
        self.assertTrue(differs([("d", zero, None)], [("e", zero, None)]))
        self.assertTrue(differs([("d", zero, None)], [("d", one, None)]))

    def test_kinds_of(self):
        report = {"differences": [{"kind": "calls"}, {"kind": "memory"}, {"kind": "calls"}]}
        self.assertEqual(psd.kinds_of(report), ["memory", "calls"])

    def test_usage_errors(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(psd.main(["--original", "/nonexistent/a", "--rewritten",
                                       "/nonexistent/b", "--function", "f"]), 2)
        self.assertIn("is not a file", errors.getvalue())


@unittest.skipUnless(HAVE_COZY and HAVE_TOOLS, "needs cozy, clang and ld.lld")
class Built(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.scratch.name)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def image(self, name: str, sources: list[str], *flags: str) -> Path:
        return build(self.directory, name, sources, list(flags))

    def run_pair(self, original: Path, rewritten: Path, **settings) -> dict:
        entry, thumb = psd.find_function(original, "put")
        self.assertTrue(thumb)
        return psd.compare(original, rewritten, entry, thumb, psd.Settings(**settings))

    def kinds(self, report: dict) -> set[str]:
        return {difference["kind"] for difference in report["differences"]}

    def test_same_image(self):
        image = self.image("same", [ADD], "-DADDEND=1")
        report = self.run_pair(image, image, args=1)
        self.assertEqual(report["verdict"], "proven", report)
        self.assertEqual(report["states"]["pairs"], 1)
        self.assertEqual(report["mode"], "thumb")
        self.assertIn("cozy", report["engine"])

    def test_changed_result(self):
        report = self.run_pair(self.image("add1", [ADD], "-DADDEND=1"),
                               self.image("add2", [ADD], "-DADDEND=2"), args=1)
        self.assertEqual(report["verdict"], "failed", report)
        difference, = report["differences"]
        self.assertEqual((difference["kind"], difference["register"]), ("register", "r0"))
        # The witness is an input, and the two values are what each side returns for it.
        argument = int(difference["input"]["arg0"], 16)
        self.assertEqual(int(difference["original"], 16), (argument + 1) & 0xFFFFFFFF)
        self.assertEqual(int(difference["rewritten"], 16), (argument + 2) & 0xFFFFFFFF)

    def test_result_not_compared(self):
        report = self.run_pair(self.image("add1", [ADD], "-DADDEND=1"),
                               self.image("add2", [ADD], "-DADDEND=2"),
                               args=1, registers=("r4", "sp"))
        self.assertEqual(report["verdict"], "proven", report)

    def test_saved_register(self):
        # A saved register has an unknown value at entry, so leaving any
        # fixed value in it, zero included, is seen.
        plain = self.image("keep", [SAVED], "-DBODY=\"bx lr\"")
        for value in (0, 7):
            with self.subTest(value=value):
                report = self.run_pair(plain, self.image(
                    f"lose{value}", [SAVED], f"-DBODY=\"movs r4, #{value}\\n bx lr\""), args=1)
                self.assertEqual(report["verdict"], "failed", report)
                difference, = report["differences"]
                self.assertEqual((difference["kind"], difference["register"]),
                                 ("register", "r4"))
                self.assertEqual(int(difference["rewritten"], 16), value)
                self.assertNotEqual(int(difference["original"], 16), value)
        # Using the register and restoring it is not a difference.
        report = self.run_pair(plain, self.image(
            "restore", [SAVED], "-DBODY=\"push {r4, lr}\\n movs r4, #0\\n pop {r4, pc}\""),
            args=1)
        self.assertEqual(report["verdict"], "proven", report)

    def test_changed_store(self):
        report = self.run_pair(self.image("store1", [STORE], "-DADDEND=1"),
                               self.image("store2", [STORE], "-DADDEND=2"), args=1)
        self.assertEqual(report["verdict"], "failed", report)
        self.assertEqual(self.kinds(report), {"memory"})
        self.assertEqual(report["differences"][0]["size"], 4)

    def test_buffer_and_loop_bound(self):
        image = self.image("length", [LENGTH_OF])
        report = self.run_pair(image, image, args=1, arg_buffers={0: 3})
        self.assertEqual(report["verdict"], "proven", report)
        # A terminator in each of the three bytes, or in the zero after them.
        self.assertEqual(report["states"]["original"]["returned"], 4)

        report = self.run_pair(image, image, args=1, arg_buffers={0: 3}, loop_bound=1)
        self.assertEqual(report["verdict"], "incomplete", report)
        self.assertGreater(report["states"]["original"]["loop-bound"], 0)
        self.assertTrue(any("loop bound" in note for note in report["incomplete"]))

    def test_argument_range(self):
        # The two differ only above 0xff, where the low byte is taken.
        masked = "int put(unsigned n) { return n & 0xff; }\n"
        whole = "int put(unsigned n) { return n; }\n"
        images = self.image("masked", [masked]), self.image("whole", [whole])
        self.assertEqual(self.run_pair(*images, args=1)["verdict"], "failed")
        report = self.run_pair(*images, args=1, arg_ranges={0: (0, 0xFF)})
        self.assertEqual(report["verdict"], "proven", report)

    def test_added_rejection(self):
        original = self.image("open", [CHECKED], "-DADDEND=0")
        checked = self.image("checked", [CHECKED], "-DADDEND=0", "-DLIMIT=16")
        report = self.run_pair(original, checked, args=1)
        self.assertEqual(report["verdict"], "failed", report)

        report = self.run_pair(original, checked, args=1, reject_returns=[0xFFFFFFFF])
        self.assertEqual(report["verdict"], "proven", report)
        self.assertEqual(report["rejections"], 1)

        # The other way round, the rewritten function accepts what was rejected.
        report = self.run_pair(checked, original, args=1, reject_returns=[0xFFFFFFFF])
        self.assertEqual(report["verdict"], "failed", report)
        self.assertEqual(self.kinds(report), {"accepts-rejected-input"})
        self.assertGreater(int(report["differences"][0]["input"]["arg0"], 16), 16)

    def test_rejection_does_not_hide_a_change(self):
        report = self.run_pair(self.image("open0", [CHECKED], "-DADDEND=0"),
                               self.image("checked1", [CHECKED], "-DADDEND=1", "-DLIMIT=16"),
                               args=1, reject_returns=[0xFFFFFFFF])
        self.assertEqual(report["verdict"], "failed", report)
        self.assertEqual(self.kinds(report), {"register"})
        self.assertLessEqual(int(report["differences"][0]["input"]["arg0"], 16), 16)

    def test_stub(self):
        doubles = self.image("doubles", [CALLER, HELPER], "-DADDEND=1", "-DFACTOR=2")
        triples = self.image("triples", [CALLER, HELPER], "-DADDEND=1", "-DFACTOR=3")
        self.assertEqual(self.run_pair(doubles, triples, args=1)["verdict"], "failed")

        report = self.run_pair(doubles, triples, args=1, stubs=["helper"])
        self.assertEqual(report["verdict"], "proven", report)
        self.assertEqual(report["assumptions"]["stubbed_callees"], ["helper"])

        # The stub's result is the same symbol on both sides, so what the
        # caller does with it is still compared.
        other = self.image("other", [CALLER, HELPER], "-DADDEND=2", "-DFACTOR=2")
        report = self.run_pair(doubles, other, args=1, stubs=["helper"])
        self.assertEqual(report["verdict"], "failed", report)

        with self.assertRaises(psd.UsageError):
            self.run_pair(doubles, triples, args=1, stubs=["absent"])

    def test_stubs_are_told_apart(self):
        # Each stubbed callee has its own results: taking the difference of
        # two callees the other way round is a change.
        one = self.image("order1", [TWO_CALLEES, TWO_HELPERS], "-DORDER=first(x) - second(x)")
        other = self.image("order2", [TWO_CALLEES, TWO_HELPERS], "-DORDER=second(x) - first(x)")
        report = self.run_pair(one, other, args=1, stubs=["first", "second"])
        self.assertEqual(report["verdict"], "failed", report)
        # The result differs, and so does the order of the two calls.
        self.assertEqual(self.kinds(report), {"register", "calls"})
        report = self.run_pair(one, one, args=1, stubs=["first", "second"])
        self.assertEqual(report["verdict"], "proven", report)

    def test_stub_returning_a_pointer(self):
        # The callee's result is read through: it has to point at memory, and
        # that memory is unknown and the same on both sides.
        second = self.image("second", [POINTER_CALLER, POINTER_HELPER], "-DINDEX=1")
        third = self.image("third", [POINTER_CALLER, POINTER_HELPER], "-DINDEX=2")
        report = self.run_pair(second, second, args=1, stub_pointers={"helper": 16})
        self.assertEqual(report["verdict"], "proven", report)
        self.assertEqual(report["assumptions"]["stubbed_callees"], ["helper"])
        report = self.run_pair(second, third, args=1, stub_pointers={"helper": 16})
        self.assertEqual(report["verdict"], "failed", report)
        difference, = report["differences"]
        self.assertEqual(difference["register"], "r0")
        self.assertNotEqual(difference["original"], difference["rewritten"])
        with self.assertRaises(psd.UsageError):
            self.run_pair(second, third, args=1, stubs=["helper"], stub_pointers={"helper": 16})

    def test_calls_to_stubs(self):
        # The arguments of a stubbed callee are compared when their number is given.
        one = self.image("pass1", [PASSES, HELPER], "-DADDEND=1", "-DFACTOR=2")
        two = self.image("pass2", [PASSES, HELPER], "-DADDEND=2", "-DFACTOR=2")
        report = self.run_pair(one, two, args=1, stubs=["helper"])
        self.assertEqual(report["verdict"], "proven", report)
        report = self.run_pair(one, two, args=1, stubs=["helper"], stub_args={"helper": 1})
        self.assertEqual(report["verdict"], "failed", report)
        difference, = report["differences"]
        self.assertEqual(difference["kind"], "calls")
        self.assertEqual(difference["calls"], {"original": 1, "rewritten": 1})
        report = self.run_pair(one, one, args=1, stubs=["helper"], stub_args={"helper": 1})
        self.assertEqual(report["verdict"], "proven", report)

        # A call that is gone is a difference whether or not arguments are compared.
        calls = self.image("calls", [MAYBE_CALLS, HELPER], "-DUSE=helper(x)", "-DFACTOR=2")
        silent = self.image("silent", [MAYBE_CALLS, HELPER], "-DUSE=(void)x", "-DFACTOR=2")
        report = self.run_pair(calls, silent, args=1, stubs=["helper"])
        self.assertEqual(self.kinds(report), {"calls"}, report)
        self.assertEqual(report["differences"][0]["calls"], {"original": 1, "rewritten": 0})

        # The address of a local differs with the frame and is not compared.
        small = self.image("small", [PASSES_LOCAL, HELPER], "-DPAD=1", "-DFACTOR=2")
        large = self.image("large", [PASSES_LOCAL, HELPER], "-DPAD=9", "-DFACTOR=2")
        report = self.run_pair(small, large, args=1, stubs=["helper"], stub_args={"helper": 1})
        self.assertEqual(report["verdict"], "proven", report)

        # Constant text is compared by what it says, not by where it is.
        text = HELPER.replace("int x", "const char *x").replace("x *", "x[0] *")
        here = self.image("here", [PASSES_TEXT, text], "-DPAD=2", '-DTEXT="one"', "-DFACTOR=2")
        there = self.image("there", [PASSES_TEXT, text], "-DPAD=2", '-DTEXT=&"xxone"[2]',
                           "-DFACTOR=2")
        said = self.image("said", [PASSES_TEXT, text], "-DPAD=2", '-DTEXT="two"', "-DFACTOR=2")
        report = self.run_pair(here, there, args=1, stubs=["helper"], stub_args={"helper": 1})
        self.assertEqual(report["verdict"], "proven", report)
        report = self.run_pair(here, said, args=1, stubs=["helper"], stub_args={"helper": 1})
        self.assertEqual(self.kinds(report), {"calls"}, report)

        with self.assertRaises(psd.UsageError):
            self.run_pair(one, two, args=1, stub_args={"helper": 1})

    def test_leaving_out_a_call_is_a_rejection(self):
        always = self.image("always", [SKIPS, VOID_HELPER], "-DUSE=helper(x)")
        checked = self.image("checked", [SKIPS, VOID_HELPER], "-DUSE=if (x <= 32) helper(x)")
        registers = (*psd.SAVED_REGISTERS, "sp")
        report = self.run_pair(always, checked, args=1, stubs=["helper"], registers=registers)
        self.assertEqual(self.kinds(report), {"calls"}, report)
        report = self.run_pair(always, checked, args=1, reject_skips=["helper"],
                               registers=registers)
        self.assertEqual(report["verdict"], "proven", report)
        self.assertEqual(report["rejections"], 1)
        self.assertEqual(report["assumptions"]["reject_skips"], ["helper"])
        # The other way round the rewritten function makes a call the original
        # left out: that is not a rejection.
        report = self.run_pair(checked, always, args=1, reject_skips=["helper"],
                               registers=registers)
        self.assertEqual(self.kinds(report), {"calls"}, report)

    def test_bytes_after_a_buffer(self):
        one = self.image("third1", [THIRD_BYTE], "-DADDEND=2")
        two = self.image("third2", [THIRD_BYTE], "-DADDEND=3")
        # With nothing after the two symbolic bytes the third is zero.
        report = self.run_pair(one, two, args=1, arg_buffers={0: 2})
        self.assertEqual(self.kinds(report), {"register"}, report)
        report = self.run_pair(one, two, args=1, arg_buffers={0: 2}, arg_suffixes={0: b"A"})
        self.assertEqual(report["verdict"], "proven", report)
        self.assertEqual(report["assumptions"]["arg_suffixes"], {"0": "41"})

    def test_upper_half_of_a_stub_result(self):
        helper = WIDE_HELPER.replace("TYPE", "unsigned")
        uses = self.image("upper", [helper, USES_UPPER_HALF])
        drops = self.image("drops", [helper, DROPS_RESULT])
        wide = dict(args=0, stubs=["helper"], stub_returns={"helper": ("r0", "r1")})
        report = self.run_pair(uses, drops, **wide)
        self.assertEqual(self.kinds(report), {"register"}, report)
        self.assertEqual(self.run_pair(uses, uses, **wide)["verdict"], "proven")

    def test_wide_argument_of_a_stub(self):
        helper = WIDE_HELPER.replace("TYPE", "unsigned long long")
        one = self.image("warg1", [helper, PASSES_WIDE], "-DVALUE=0x100000000ULL")
        two = self.image("warg2", [helper, PASSES_WIDE], "-DVALUE=0x200000000ULL")
        low = dict(args=0, stubs=["helper"], stub_registers={"helper": ("r0",)})
        self.assertEqual(self.run_pair(one, two, **low)["verdict"], "proven")
        both = dict(args=0, stubs=["helper"], stub_registers={"helper": ("r0", "r1")})
        self.assertEqual(self.kinds(self.run_pair(one, two, **both)), {"calls"})
        self.assertEqual(self.run_pair(one, one, **both)["verdict"], "proven")

    def test_called_stub_with_an_unplaced_parameter_is_incomplete(self):
        calls = self.image("note1", [MAYBE_CALLS, HELPER], "-DUSE=helper(x)", "-DFACTOR=2")
        silent = self.image("note2", [MAYBE_CALLS, HELPER], "-DUSE=(void)x", "-DFACTOR=2")
        noted = dict(args=1, stubs=["helper"], stub_notes={"helper": "its parameter ... "
                                                           "(variable arguments) is not compared"})
        report = self.run_pair(calls, calls, **noted)
        self.assertEqual(report["verdict"], "incomplete", report)
        self.assertIn("helper", report["incomplete"][0])
        # A stub nothing calls leaves the run as it is.
        self.assertEqual(self.run_pair(silent, silent, **noted)["verdict"], "proven")

    def test_notes_given_with_the_settings(self):
        image = self.image("given", [ADD], "-DADDEND=1")
        report = self.run_pair(image, image, args=1, notes=["the return type double ..."])
        self.assertEqual(report["verdict"], "incomplete", report)

    def test_modelled_string_function(self):
        plain = self.image("match1", [MATCHES, STRNCMP], "-DEXTRA=")
        # The added test is true exactly when the call returns zero.
        again = self.image("match2", [MATCHES, STRNCMP],
                           "-DEXTRA=|| (s[0] == 'a' && s[1] == 'b')")
        settings = dict(args=1, arg_buffers={0: 2}, stubs=["strncmp"])
        report = self.run_pair(plain, again, **settings)
        self.assertEqual(report["verdict"], "proven", report)
        self.assertEqual(report["assumptions"]["modelled_callees"], ["strncmp"])
        # Left unknown, the result can be nonzero for "ab".
        report = self.run_pair(plain, again, models=False, **settings)
        self.assertEqual(self.kinds(report), {"register"}, report)
        self.assertEqual(report["assumptions"]["modelled_callees"], [])

    def test_path_lost_at_a_call_through_a_pointer(self):
        one = self.image("lost1", [THROUGH_POINTER], "-DADDEND=0")
        two = self.image("lost2", [THROUGH_POINTER], "-DADDEND=1")
        report = self.run_pair(one, two, args=2)
        self.assertEqual(report["verdict"], "incomplete", report)
        self.assertEqual(report["states"]["original"]["lost"], 1)
        self.assertTrue(any("can be anywhere" in note for note in report["incomplete"]), report)
        # A function that loses no path says so.
        plain = self.image("lost3", [ADD], "-DADDEND=1")
        self.assertEqual(self.run_pair(plain, plain, args=1)["states"]["original"]["lost"], 0)

    def test_reject_value_wider_than_a_register(self):
        rejects = self.image("wide1", [WIDE_REJECT], "-DREJECTED=0x100000000ULL")
        accepts = self.image("wide2", [WIDE_REJECT], "-DREJECTED=0")
        wide = dict(args=1, reject_returns=[0x100000000],
                    registers=("r0", "r1", *psd.SAVED_REGISTERS, "sp"))
        # The rewritten function returns 0 where the original refused.
        report = self.run_pair(rejects, accepts, **wide)
        self.assertEqual(self.kinds(report), {"accepts-rejected-input"}, report)
        self.assertEqual(self.run_pair(rejects, rejects, **wide)["verdict"], "proven")
        # The other way round the rewritten function refuses more: no difference.
        self.assertEqual(self.run_pair(accepts, rejects, **wide)["verdict"], "proven")

    def test_device_write_through_an_address_that_depends_on_the_input(self):
        one = self.image("either1", [EITHER_DEVICE], "-DFIRST=1")
        two = self.image("either2", [EITHER_DEVICE], "-DFIRST=2")
        report = self.run_pair(one, two, args=1, mmio_regions=[(0x40001000, 0x40001fff)])
        self.assertNotEqual(report["verdict"], "proven", report)
        if report["verdict"] == "incomplete":
            self.assertTrue(any("device range 0x40001000:0x40001fff" in note
                                for note in report["incomplete"]), report)
        # A range the write cannot reach is not mentioned.
        report = self.run_pair(one, one, args=1, mmio_regions=[(0x50000000, 0x50000fff)])
        self.assertEqual(report["verdict"], "proven", report)

    def test_conditional_device_write(self):
        guarded = self.image("guard1", [STORES], '-DFIRST_STORE="it ne\\n strne r2, [r1]"')
        always = self.image("guard2", [STORES], '-DFIRST_STORE="str r2, [r1]"')
        region = dict(args=1, mmio_regions=[(0x40001000, 0x40001fff)])
        report = self.run_pair(guarded, always, **region)
        self.assertEqual(self.kinds(report), {"device-writes"}, report)
        self.assertEqual(report["differences"][0]["input"]["arg0"], "0x0")
        self.assertEqual(self.run_pair(guarded, guarded, **region)["verdict"], "proven")
        self.assertEqual(self.run_pair(always, guarded, **region)["verdict"], "failed")

    def test_read_through_an_address_with_many_values(self):
        # 256 addresses are all followed: the one input that differs is found.
        one = self.image("index1", [INDEXED], "-DMASK=255", "-DADDEND=0")
        two = self.image("index2", [INDEXED], "-DMASK=255", "-DADDEND=1")
        report = self.run_pair(one, two, args=2, arg_buffers={0: 256})
        self.assertEqual(self.kinds(report), {"register"}, report)
        self.assertEqual(self.run_pair(one, one, args=2, arg_buffers={0: 256})["verdict"],
                         "proven")
        # With more, the run does not claim to have looked at all of them.
        wide = self.image("index3", [INDEXED], "-DMASK=511", "-DADDEND=0")
        report = self.run_pair(wide, wide, args=2, arg_buffers={0: 512})
        self.assertEqual(report["verdict"], "incomplete", report)
        self.assertTrue(any("more than 256 possible values" in note
                            for note in report["incomplete"]), report)

    def test_negative_reject_value(self):
        plain = self.image("signed1", [SIGNED_REJECT])
        checked = self.image("signed2", [SIGNED_REJECT], "-DLIMIT=16")
        self.assertEqual(self.kinds(self.run_pair(plain, checked, args=1)), {"register"})
        for value in (-1, 0xFFFFFFFF):
            report = self.run_pair(plain, checked, args=1, reject_returns=[value])
            self.assertEqual(report["verdict"], "proven", (value, report))
            self.assertEqual(report["rejections"], 1)

    def test_device_writes(self):
        region = [(0x40001000, 0x40001FFF)]
        base = self.image("device", [DEVICE], "-DFIRST=0", "-DREPEAT=2")
        for name, flags in (("first", ("-DFIRST=1", "-DREPEAT=2")),     # another value
                            ("fewer", ("-DFIRST=0", "-DREPEAT=1")),     # a write removed
                            ("more", ("-DFIRST=0", "-DREPEAT=3"))):     # a write added
            with self.subTest(name):
                other = self.image(name, [DEVICE], *flags)
                # The device register ends with the same value, so the final
                # memory says nothing.
                self.assertEqual(self.run_pair(base, other, args=1)["verdict"], "proven")
                report = self.run_pair(base, other, args=1, mmio_regions=region)
                self.assertEqual(report["verdict"], "failed", report)
                self.assertEqual(self.kinds(report), {"device-writes"})
                self.assertEqual(report["differences"][0]["writes"]["original"], 3)
        report = self.run_pair(base, base, args=1, mmio_regions=region)
        self.assertEqual(report["verdict"], "proven", report)

    def test_reject_handler(self):
        original = self.image("unguarded", [GUARDED, STOP])
        guarded = self.image("guarded", [GUARDED, STOP], "-DLIMIT=16")
        report = self.run_pair(original, guarded, args=1, reject_handlers=["stop"])
        self.assertEqual(report["verdict"], "proven", report)
        self.assertEqual(report["rejections"], 1)
        self.assertEqual(report["states"]["rewritten"]["returned"], 2)

        report = self.run_pair(guarded, original, args=1, reject_handlers=["stop"])
        self.assertEqual(report["verdict"], "failed", report)
        self.assertEqual(self.kinds(report), {"accepts-rejected-input"})

        # Without the declaration the handler runs, and never returns.
        report = self.run_pair(original, guarded, args=1, loop_bound=2)
        self.assertEqual(report["verdict"], "incomplete", report)

    def test_reject_write(self):
        original = self.image("unchecked", [FAULTS])
        checked = self.image("faulting", [FAULTS], "-DLIMIT=16")
        fault = [(0, 3)]
        report = self.run_pair(original, checked, args=1, reject_writes=fault)
        self.assertEqual(report["verdict"], "proven", report)
        self.assertEqual(report["rejections"], 1)

        # Removing the check accepts the input it refused.
        report = self.run_pair(checked, original, args=1, reject_writes=fault)
        self.assertEqual(report["verdict"], "failed", report)
        self.assertEqual(self.kinds(report), {"accepts-rejected-input"})

        # Without the declaration the store is a change to memory.
        report = self.run_pair(original, checked, args=1)
        self.assertEqual(report["verdict"], "failed", report)
        self.assertEqual(self.kinds(report), {"memory"})

    def test_command_line(self):
        first = self.image("cli1", [ADD], "-DADDEND=1")
        second = self.image("cli2", [ADD], "-DADDEND=2")
        output, text, graph = (self.directory / name for name in
                               ("cli.json", "cli.txt", "cli.graph.json"))
        # A process of its own: cozy reads the descriptor of the real stdout.
        script = [sys.executable, str(Path(psd.__file__))]
        done = subprocess.run([*script, "--original", str(first), "--rewritten", str(second),
                               "--function", "put", "--args", "1", "--output", str(output),
                               "--text", str(text), "--graph", str(graph)],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertIn("failed: 1 difference(s)", done.stdout)
        self.assertEqual(json.loads(output.read_text())["verdict"], "failed")
        self.assertIn("STATE PAIR", text.read_text())
        json.loads(graph.read_text())

        done = subprocess.run([*script, "--original", str(first), "--rewritten", str(first),
                               "--address", hex(psd.find_function(first, "put")[0]),
                               "--mode", "thumb", "--args", "1"],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("proven", done.stdout)

        done = subprocess.run([*script, "--original", str(first), "--rewritten", str(first),
                               "--address", "0x8000"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertIn("--address needs --mode", done.stderr)


@unittest.skipUnless(HAVE_COZY and FIRMWARE.exists() and LENGTH.exists(),
                     "needs cozy and the firmware suite's output")
class Firmware(unittest.TestCase):
    def test_length(self):
        entry, thumb = psd.find_function(FIRMWARE, "qemu_target_length")
        settings = dict(arg_buffers={0: 4})
        report = psd.compare(FIRMWARE, LENGTH, entry, thumb,
                             psd.Settings(reject_returns=[0xFFFFFFFF], **settings))
        self.assertEqual(report["verdict"], "proven", report)
        self.assertEqual(report["rejections"], 1)
        report = psd.compare(FIRMWARE, LENGTH, entry, thumb, psd.Settings(**settings))
        self.assertEqual(report["verdict"], "failed", report)


if __name__ == "__main__":
    unittest.main()
