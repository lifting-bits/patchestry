#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests for check_bindings.py on a small hand-made Thumb image.

Needs clang and ld.lld with the ARM target.  They are taken from
PATCHESTRY_LLVM_PREFIX (a bin directory) and PATCHESTRY_LD_LLD, the same
variables scripts/test-qemu-firmware-runtime.sh uses.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_bindings  # noqa: E402

# `helper` is file-local in both objects, so the image has two symbols of that
# name at different addresses.  `unique` and `counter` are defined once.
FIRST = r"""
    .syntax unified
    .thumb
    .text

    .global unique
    .type unique, %function
    .thumb_func
unique:
    push {r7, lr}
    bl helper
    pop {r7, pc}
    .size unique, . - unique

    .type helper, %function
    .thumb_func
helper:
    bx lr
    nop
    .size helper, . - helper

    .data
    .global counter
    .type counter, %object
counter:
    .word 0
    .size counter, . - counter
"""

SECOND = r"""
    .syntax unified
    .thumb
    .text

    .type helper, %function
    .thumb_func
helper:
    movs r0, #1
    bx lr
    .size helper, . - helper
"""

LINKER_SCRIPT = r"""
SECTIONS {
  .text 0x10000 : { *(.text*) }
  .data 0x100000 : { *(.data*) }
  /DISCARD/ : { *(.ARM.exidx*) *(.ARM.attributes) }
}
"""


def find_tools() -> tuple[str, str]:
    prefix = os.environ.get("PATCHESTRY_LLVM_PREFIX")
    clang = str(Path(prefix) / "clang") if prefix else shutil.which("clang")
    lld = os.environ.get("PATCHESTRY_LD_LLD")
    if not lld:
        lld = str(Path(prefix) / "ld.lld") if prefix else shutil.which("ld.lld")
    if not clang or not lld or not Path(clang).exists() or not Path(lld).exists():
        raise unittest.SkipTest(
            "clang and ld.lld not found; set PATCHESTRY_LLVM_PREFIX and PATCHESTRY_LD_LLD"
        )
    return clang, lld


def binding(name: str, address: int, kind: str = "function", size: int = 0) -> dict:
    return {"name": name, "symbol": name, "kind": kind, "origin": "original",
            "address": f"0x{address:x}", "size": size}


class CheckBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clang, lld = find_tools()
        cls.directory = Path(tempfile.mkdtemp(prefix="check-bindings-"))
        (cls.directory / "first.S").write_text(FIRST, encoding="utf-8")
        (cls.directory / "second.S").write_text(SECOND, encoding="utf-8")
        (cls.directory / "link.ld").write_text(LINKER_SCRIPT, encoding="utf-8")
        for name in ("first", "second"):
            subprocess.run(
                [clang, "--target=thumbv7m-none-eabi", "-mcpu=cortex-m3", "-c",
                 f"{name}.S", "-o", f"{name}.o"],
                cwd=cls.directory, check=True,
            )
        subprocess.run([lld, "-T", "link.ld", "first.o", "second.o", "-o", "image.elf"],
                       cwd=cls.directory, check=True)
        cls.elf = cls.directory / "image.elf"
        cls.image = check_bindings.load_image(cls.elf)
        cls.helpers = sorted(symbol.addr for symbol in cls.image.symbols["helper"])
        cls.unique = cls.image.symbols["unique"][0].addr
        cls.counter = cls.image.symbols["counter"][0].addr

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def report(self, *bindings: dict) -> dict:
        return check_bindings.check_bindings({"bindings": list(bindings)}, self.elf)

    def verdicts(self, *bindings: dict) -> list[str]:
        return [row["verdict"] for row in self.report(*bindings)["bindings"]]

    def test_image_has_two_helpers(self):
        self.assertEqual(len(self.helpers), 2)
        self.assertNotEqual(self.helpers[0], self.helpers[1])

    def test_single_definition_matches(self):
        self.assertEqual(
            self.verdicts(binding("unique", self.unique),
                          binding("counter", self.counter, "object", 4)),
            ["match", "match"],
        )

    def test_name_defined_at_two_addresses_is_ambiguous(self):
        for address in self.helpers:
            report = self.report(binding("helper", address))
            row = report["bindings"][0]
            self.assertEqual(row["verdict"], "ambiguous-symbol")
            self.assertEqual(row["elf_definitions"], [f"0x{a:x}" for a in self.helpers])
            self.assertEqual(len(check_bindings.failures(report, strict=False)), 1)

    def test_both_helpers_bound_is_ambiguous(self):
        report = self.report(binding("helper", self.helpers[0]),
                             binding("helper", self.helpers[1]))
        self.assertEqual([row["verdict"] for row in report["bindings"]],
                         ["ambiguous-symbol", "ambiguous-symbol"])
        self.assertEqual(len(check_bindings.failures(report, strict=False)), 2)

    def test_wrong_address_stays_address_mismatch(self):
        self.assertEqual(self.verdicts(binding("helper", self.unique)), ["address-mismatch"])
        self.assertEqual(self.verdicts(binding("unique", self.helpers[0])), ["address-mismatch"])

    def test_two_bindings_one_absolute_name_two_addresses(self):
        # Neither name is an ELF symbol, so both would be given to the patch
        # link as absolute symbols under one name.
        report = self.report(binding("DAT_shared", self.counter, "object"),
                             binding("DAT_shared", self.counter + 2, "object"))
        rows = report["bindings"]
        self.assertEqual([row["verdict"] for row in rows],
                         ["ambiguous-symbol", "ambiguous-symbol"])
        self.assertEqual(rows[0]["binding_addresses"],
                         [f"0x{self.counter:x}", f"0x{self.counter + 2:x}"])
        self.assertNotIn("DAT_shared", report["absolute_symbols"])

    def test_two_bindings_same_address_share_without_failing(self):
        report = self.report(binding("unique", self.unique), binding("unique", self.unique))
        self.assertEqual([row["verdict"] for row in report["bindings"]], ["match", "match"])
        self.assertEqual(report["bindings"][0]["shared_symbol"], 2)


if __name__ == "__main__":
    unittest.main()
