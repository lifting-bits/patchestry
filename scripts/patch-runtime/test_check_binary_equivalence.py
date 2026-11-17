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

import shutil
import tempfile
import unittest
from pathlib import Path

import claripy
from elftools.elf.elffile import ELFFile

import check_binary_equivalence as cbe

FIRMWARE = Path(__file__).resolve().parents[2] / "firmwares/serial-qemu/build/serial.elf"


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


if __name__ == "__main__":
    unittest.main()
