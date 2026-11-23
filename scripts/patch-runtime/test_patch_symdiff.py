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
    """A Thumb ELF from C sources, each its own object; the entry is `put`.
    The same helper as in test_check_binary_equivalence.py, which cannot be
    imported here because it loads the other angr."""
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
