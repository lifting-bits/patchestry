#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests of check_binary_equivalence.py on the serial-qemu firmware.

Needs angr (the Patcherex2 environment has it) and the firmware ELF,
`firmwares/serial-qemu/build/serial.elf`.  The rewritten side is the original
image with bytes of the function changed, so no patcher is involved.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import claripy
from elftools.elf.elffile import ELFFile

import check_binary_equivalence as cbe

FIRMWARE = Path(__file__).resolve().parents[2] / "firmwares/serial-qemu/build/serial.elf"
LLVM = Path(os.environ.get("PATCHESTRY_LLVM_PREFIX", "/nonexistent"))
LLD = Path(os.environ.get("PATCHESTRY_LD_LLD", LLVM / "ld.lld"))

# Stores one byte at an index taken from the input.
INDEXED_STORE = """
void put(unsigned char *out, unsigned index, unsigned char value) {
    out[index & 7] = value + ADDEND;
}
"""


def with_byte(image: Path, address: int, value: int, output: Path) -> int:
    """Copy `image` with the byte at virtual `address` set to `value`; returns the old byte."""
    data = bytearray(image.read_bytes())
    with image.open("rb") as handle:
        for segment in ELFFile(handle).iter_segments():
            start = segment["p_vaddr"]
            if segment["p_type"] == "PT_LOAD" and start <= address < start + segment["p_filesz"]:
                offset = segment["p_offset"] + address - start
                old = data[offset]
                data[offset] = value
                output.write_bytes(data)
                return old
    raise ValueError(f"{image}: no loaded file byte at {address:#x}")


class Pieces(unittest.TestCase):
    def test_skipped_calls(self):
        call = lambda name: (name, ())
        first, second = call("first"), call("second")
        self.assertTrue(cbe.skipped((first,), (first, second), ("second",)))
        self.assertFalse(cbe.skipped((first,), (first, second), ()))
        self.assertFalse(cbe.skipped((second,), (first, second), ("second",)))
        self.assertFalse(cbe.skipped((first, second), (first, second), ("second",)))

    def test_sequences(self):
        x, y = claripy.BVS("x", 32), claripy.BVS("y", 32)
        differ = lambda a, b: cbe.sequences_differ(a, b, lambda call: (call[0], call[1]))
        self.assertTrue(differ((("f", (x,)),), ()).is_true())
        self.assertTrue(differ((("f", (x,)),), (("g", (x,)),)).is_true())
        self.assertTrue(differ((("f", (x,)),), (("f", (x,)),)).is_false())
        self.assertFalse(differ((("f", (x,)),), (("f", (y,)),)).is_false())
        # The address of a local equals the address of another local.
        self.assertTrue(differ((("f", (("local",),)),), (("f", (("local",),)),)).is_false())
        self.assertTrue(differ((("f", (("constant", b"a"),)),),
                               (("f", (("constant", b"b"),)),)).is_true())

    def test_prototypes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ir = Path(tmp) / "f.ll"
            ir.write_text("declare i32 @sprintf(ptr, ptr, ...)\ndeclare void @g()\n"
                          "define dso_local void @target(i32 %0) {\n", encoding="utf-8")
            counts, returns_value = cbe.read_prototypes(ir, "target")
        self.assertEqual(counts, {"sprintf": 2, "g": 0})
        self.assertFalse(returns_value)


@unittest.skipUnless(FIRMWARE.is_file(), f"{FIRMWARE} is not built")
class Firmware(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.copy = Path(self.tmp.name) / "serial.elf"

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self, function: str, rewritten: Path, **settings) -> dict:
        import angr
        symbol = angr.Project(str(FIRMWARE), auto_load_libs=False).loader.find_symbol(function)
        return cbe.check_binary_equivalence(
            FIRMWARE, rewritten, symbol.rebased_addr & ~1, bool(symbol.rebased_addr & 1),
            cbe.Settings(**settings))

    def test_an_image_equals_itself(self):
        shutil.copyfile(FIRMWARE, self.copy)
        report = self.run_check("qemu_target_send", self.copy, returns_value=False,
                                prototypes={"qemu_irq_disable": 0, "qemu_irq_enable": 0},
                                mmio_regions=((0x4000C000, 0x4000CFFF),))
        self.assertEqual(report["verdict"], "proven", report)
        self.assertEqual(report["checks"]["mmio-order"], "proven")
        self.assertEqual(report["checks"]["P3"], "not-run")

    def test_a_changed_device_write_is_found(self):
        # The second store of qemu_target_send writes a newline (movs rN, #10).
        import angr
        project = angr.Project(str(FIRMWARE), auto_load_libs=False)
        symbol = project.loader.find_symbol("qemu_target_send")
        start = symbol.rebased_addr & ~1
        code = project.loader.memory.load(start, symbol.size)
        at = next(i for i in range(0, len(code) - 1, 2) if code[i] == 10 and code[i + 1] >> 3 == 4)
        with_byte(FIRMWARE, start + at, 11, self.copy)
        report = self.run_check("qemu_target_send", self.copy, returns_value=False,
                                prototypes={"qemu_irq_disable": 0, "qemu_irq_enable": 0},
                                mmio_regions=((0x4000C000, 0x4000CFFF),))
        self.assertEqual(report["verdict"], "failed", report)
        self.assertEqual(report["checks"]["mmio-order"], "failed")
        self.assertEqual(report["checks"]["P4"], "proven")

    def test_undeclared_device_writes_are_memory(self):
        shutil.copyfile(FIRMWARE, self.copy)
        report = self.run_check("qemu_target_send", self.copy, returns_value=False,
                                prototypes={"qemu_irq_disable": 0, "qemu_irq_enable": 0})
        self.assertEqual(report["checks"]["mmio-order"], "not-run")
        self.assertEqual(report["checks"]["P2"], "proven")


@unittest.skipUnless((LLVM / "clang").is_file() and LLD.is_file(),
                     "PATCHESTRY_LLVM_PREFIX has no clang and ld.lld")
class SymbolicAddress(unittest.TestCase):
    """A write through an address that depends on the input."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def image(self, name: str, addend: int, source: str = INDEXED_STORE) -> Path:
        (self.directory / "put.c").write_text(source, encoding="utf-8")
        obj, elf = self.directory / f"{name}.o", self.directory / f"{name}.elf"
        subprocess.run([str(LLVM / "clang"), "--target=thumbv7m-none-eabi", "-O1",
                        "-ffreestanding", f"-DADDEND={addend}", "-c",
                        str(self.directory / "put.c"), "-o", str(obj)], check=True)
        subprocess.run([str(LLD), "-Ttext=0x8000", "-e", "put", str(obj), "-o", str(elf)],
                       check=True)
        return elf

    def run_check(self, original: Path, rewritten: Path, **settings) -> dict:
        import angr
        symbol = angr.Project(str(original), auto_load_libs=False).loader.find_symbol("put")
        return cbe.check_binary_equivalence(
            original, rewritten, symbol.rebased_addr & ~1, bool(symbol.rebased_addr & 1),
            cbe.Settings(returns_value=False, **settings))

    def test_same_store_is_proven(self):
        report = self.run_check(self.image("a", 0), self.image("b", 0), arg_buffers={0: 8})
        self.assertEqual(report["verdict"], "proven", report)
        self.assertEqual(report["checks"]["P2"], "proven")

    def test_changed_value_fails_p2(self):
        report = self.run_check(self.image("a", 0), self.image("b", 1), arg_buffers={0: 8})
        self.assertEqual(report["checks"]["P2"], "failed", report)
        self.assertEqual(report["violations"][0]["size"], 8)

    def test_changed_index_fails_p2(self):
        other = INDEXED_STORE.replace("index & 7", "(index + 1) & 7")
        report = self.run_check(self.image("a", 0), self.image("b", 0, other),
                                arg_buffers={0: 8})
        self.assertEqual(report["checks"]["P2"], "failed", report)

    def test_declared_range_bounds_the_address(self):
        free = INDEXED_STORE.replace("index & 7", "index")
        images = self.image("a", 0, free), self.image("b", 0, free)
        report = self.run_check(*images, arg_buffers={0: 8})
        self.assertEqual(report["verdict"], "incomplete", report)
        report = self.run_check(*images, arg_buffers={0: 8}, arg_ranges={1: (0, 7)})
        self.assertEqual(report["verdict"], "proven", report)

    def test_too_many_addresses_is_incomplete(self):
        wide = INDEXED_STORE.replace("index & 7", "index & 0xfff")
        report = self.run_check(self.image("a", 0, wide), self.image("b", 0, wide),
                                arg_buffers={0: 8})
        self.assertEqual(report["verdict"], "incomplete", report)
        self.assertTrue(any("possible values" in note for note in report["incomplete"]))


if __name__ == "__main__":
    unittest.main()
