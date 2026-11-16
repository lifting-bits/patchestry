#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests for check_abi.py: objects built for different ARM ABIs are told apart.

Needs clang from PATCHESTRY_LLVM_PREFIX.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_abi  # noqa: E402
import patch_dependencies  # noqa: E402

SNIPPET = r"""
target triple = "armv7-unknown-linux-gnueabihf"

define dso_local double @half(double %0) {
  %2 = fmul double %0, 5.000000e-01
  ret double %2
}
"""

SOFT_V7 = ["-march=armv7-a", "-mfloat-abi=soft"]


class CheckAbi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix = os.environ.get("PATCHESTRY_LLVM_PREFIX")
        if not prefix:
            raise unittest.SkipTest("set PATCHESTRY_LLVM_PREFIX to the LLVM bin directory")
        cls.tools = patch_dependencies.Tools.from_prefix(Path(prefix))
        cls.directory = Path(tempfile.mkdtemp(prefix="check-abi-"))
        cls.soft = cls.build("soft", SOFT_V7)
        cls.hard = cls.build("hard", [])  # the patch target's default
        cls.old = cls.build("old", ["-march=armv5te", "-mfloat-abi=soft"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    @classmethod
    def build(cls, name: str, flags: list[str]) -> Path:
        obj = cls.directory / f"{name}.o"
        patch_dependencies.compile_snippet(SNIPPET, obj, cls.tools, flags)
        return obj

    def test_attributes(self):
        soft = check_abi.abi_attributes(self.soft)
        self.assertEqual((soft["cpu_arch"], soft["fp_arch"], soft["vfp_args"]), (10, 0, 0))
        hard = check_abi.abi_attributes(self.hard)
        self.assertEqual(hard["vfp_args"], 1)
        self.assertGreater(hard["fp_arch"], 0)

    def test_file_without_attributes(self):
        plain = self.directory / "plain.bin"
        plain.write_bytes(b"not an ELF")
        self.assertIsNone(check_abi.abi_attributes(plain))

    def test_same_abi_has_no_problems(self):
        self.assertEqual(check_abi.abi_problems(self.soft, self.soft), {})

    def test_hard_float_code_in_a_soft_float_image(self):
        problems = check_abi.abi_problems(self.soft, self.hard)
        self.assertIn("float-abi-mismatch", problems)
        self.assertIn("fp-unit-assumed", problems)

    def test_code_for_a_newer_cpu(self):
        self.assertEqual(list(check_abi.abi_problems(self.old, self.soft)), ["cpu-arch-newer"])

    def test_code_for_an_older_cpu_is_accepted(self):
        self.assertEqual(check_abi.abi_problems(self.soft, self.old), {})

    def test_flags_reproduce_the_image_abi(self):
        for image in (self.soft, self.hard, self.old):
            flags = check_abi.compiler_flags(check_abi.abi_attributes(image))
            rebuilt = self.build("rebuilt", flags)
            self.assertEqual(check_abi.abi_problems(image, rebuilt), {}, flags)
        self.assertEqual(check_abi.compiler_flags(check_abi.abi_attributes(self.soft)), SOFT_V7)


if __name__ == "__main__":
    unittest.main()
