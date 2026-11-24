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

    def test_a_profile_code_in_an_m_profile_image(self):
        # Both are Tag_CPU_arch v7; only the profile tells them apart.
        m3 = self.build("m3", ["--target=thumbv7m-none-eabi", "-mcpu=cortex-m3"])
        self.assertEqual(check_abi.abi_attributes(m3)["cpu_arch"],
                         check_abi.abi_attributes(self.soft)["cpu_arch"])
        self.assertEqual(list(check_abi.abi_problems(m3, self.soft)), ["cpu-profile-mismatch"])
        self.assertEqual(list(check_abi.abi_problems(self.soft, m3)), ["cpu-profile-mismatch"])
        self.assertEqual(check_abi.abi_problems(m3, m3), {})

    def test_m_profile_architectures(self):
        m0 = self.build("m0", ["--target=thumbv6m-none-eabi"])
        m3 = self.build("m3b", ["--target=thumbv7m-none-eabi", "-mcpu=cortex-m3"])
        m4 = self.build("m4", ["--target=thumbv7em-none-eabi", "-mcpu=cortex-m4",
                               "-mfloat-abi=soft"])
        base = self.build("v8mbase", ["--target=thumbv8m.base-none-eabi"])
        self.assertEqual(check_abi.abi_problems(m3, m0), {})
        self.assertEqual(check_abi.abi_problems(m4, m3), {})
        self.assertEqual(list(check_abi.abi_problems(m0, m3)), ["cpu-arch-newer"])
        self.assertEqual(list(check_abi.abi_problems(m3, m4)), ["cpu-arch-newer"])
        # Neither is a subset of the other.
        self.assertEqual(list(check_abi.abi_problems(base, m3)), ["cpu-arch-unknown"])
        self.assertEqual(list(check_abi.abi_problems(m3, base)), ["cpu-arch-unknown"])

    def test_r_profile_code_in_an_a_profile_image(self):
        r = self.build("r", ["--target=armv7r-none-eabi", "-mfloat-abi=soft"])
        self.assertEqual(list(check_abi.abi_problems(self.soft, r)), ["cpu-profile-mismatch"])

    def test_code_for_a_newer_fp_unit(self):
        hard = ["--target=armv7-linux-gnueabihf"]
        vfp2 = self.build("vfp2", [*hard, "-mfpu=vfpv2"])
        vfp3d16 = self.build("vfp3d16", [*hard, "-mfpu=vfpv3-d16"])
        vfp3 = self.build("vfp3", [*hard, "-mfpu=vfpv3"])
        vfp4 = self.build("vfp4", [*hard, "-mfpu=vfpv4"])
        neon = self.build("neon", [*hard, "-mfpu=neon"])
        self.assertEqual(list(check_abi.abi_problems(vfp2, vfp4)), ["fp-unit-newer"])
        self.assertEqual(check_abi.abi_problems(vfp4, vfp2), {})
        # Same version, but the code may use d16 to d31.
        self.assertEqual(list(check_abi.abi_problems(vfp3d16, vfp3)), ["fp-unit-newer"])
        self.assertEqual(check_abi.abi_problems(vfp3, vfp3d16), {})
        self.assertEqual(list(check_abi.abi_problems(vfp3, neon)), ["fp-unit-newer"])
        self.assertEqual(check_abi.abi_problems(neon, vfp3), {})

    def test_double_precision_code_with_a_single_precision_unit(self):
        m4 = ["--target=thumbv7em-none-eabihf", "-mcpu=cortex-m4", "-mfloat-abi=hard"]
        m7 = ["--target=thumbv7em-none-eabihf", "-mcpu=cortex-m7", "-mfloat-abi=hard",
              "-mfpu=fpv5-d16"]
        single, double = self.build("m4f", m4), self.build("m7f", m7)
        self.assertTrue(check_abi.abi_attributes(single)["fp_single_only"])
        self.assertFalse(check_abi.abi_attributes(double)["fp_single_only"])
        self.assertEqual(list(check_abi.abi_problems(single, double)), ["fp-unit-newer"])

    def test_unknown_tag_values_are_rejected(self):
        known = check_abi.abi_attributes(self.hard)
        self.assertEqual(check_abi.cpu_problems(known, dict(known, cpu_arch=99)).keys(),
                         {"cpu-arch-unknown"})
        self.assertEqual(check_abi.fp_problems(known, dict(known, fp_arch=99)).keys(),
                         {"fp-unit-newer"})
        self.assertEqual(check_abi.fp_problems(dict(known, fp_arch=99), known).keys(),
                         {"fp-unit-newer"})

    def test_flags_for_an_m_profile_image(self):
        m3 = self.build("m3c", ["--target=thumbv7m-none-eabi", "-mcpu=cortex-m3"])
        self.assertEqual(check_abi.compiler_flags(check_abi.abi_attributes(m3)),
                         ["-march=armv7-m", "-mfloat-abi=soft"])

    def test_flags_reproduce_the_image_abi(self):
        hard = ["--target=armv7-linux-gnueabihf"]
        others = [
            self.build("img-vfp2", ["--target=armv6-linux-gnueabihf", "-mfpu=vfpv2"]),
            self.build("img-vfp3d16", [*hard, "-mfpu=vfpv3-d16"]),
            self.build("img-vfp4", [*hard, "-mfpu=vfpv4"]),
            self.build("img-m4f", ["--target=thumbv7em-none-eabihf", "-mcpu=cortex-m4",
                                   "-mfloat-abi=hard"]),
        ]
        for image in (self.soft, self.hard, self.old, *others):
            flags = check_abi.compiler_flags(check_abi.abi_attributes(image))
            rebuilt = self.build("rebuilt", flags)
            self.assertEqual(check_abi.abi_problems(image, rebuilt), {}, flags)
        self.assertEqual(check_abi.compiler_flags(check_abi.abi_attributes(self.soft)), SOFT_V7)


if __name__ == "__main__":
    unittest.main()
