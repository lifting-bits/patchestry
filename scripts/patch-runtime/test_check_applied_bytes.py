#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests for check_applied_bytes.py.

Builds on the hand-made Thumb image of test_check_trampoline.py, with a block
of spare executable bytes for inserted code.  Needs clang, ld.lld and the LLVM
tools from PATCHESTRY_LLVM_PREFIX.
"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import capstone
from elftools.elf.elffile import ELFFile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_applied_bytes  # noqa: E402
import patch_dependencies  # noqa: E402
import patch_record  # noqa: E402
import test_check_trampoline as images  # noqa: E402

# Room for the rewriter's code: zero bytes, as in the block Patcherex2 adds.
SPARE = r"""
    .section .patch, "ax", %progbits
    .space 1024
"""

HELPER_SNIPPET = r"""
target triple = "armv7-unknown-linux-gnueabihf"

@.str = private unnamed_addr constant [6 x i8] c"PATCH\00", align 1

declare void @helper(ptr)

define dso_local void @patch_fn() {
  call void @helper(ptr @.str)
  ret void
}
"""

TARGET_SNIPPET = r"""
target triple = "armv7-unknown-linux-gnueabihf"

declare void @patch_fn()

define dso_local void @plain() {
  call void @patch_fn()
  ret void
}
"""

COMPILER = {
    "path": None,
    "version": None,
    "target": "arm-linux-gnueabihf",
    "flags": ["-fno-pic", "-O0", "-mthumb"],
}
HELPER_AT = 0x1000
BODY_AT = 0x1100


def find_tools() -> patch_dependencies.Tools:
    prefix = os.environ.get("PATCHESTRY_LLVM_PREFIX")
    if not prefix:
        raise unittest.SkipTest("set PATCHESTRY_LLVM_PREFIX to the LLVM bin directory")
    tools = patch_dependencies.Tools.from_prefix(Path(prefix))
    if tools.missing():
        raise unittest.SkipTest(f"missing LLVM tools: {', '.join(tools.missing())}")
    return tools


class ExpectedBytes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = find_tools()

    def compile(self, address, symbols):
        return check_applied_bytes.expected_bytes(
            HELPER_SNIPPET, address, symbols, COMPILER, self.tools
        )

    def calls(self, code, address):
        decoder = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_THUMB)
        return [
            int(instruction.op_str.lstrip("#"), 16)
            for instruction in decoder.disasm(code, address)
            if instruction.mnemonic in ("bl", "blx")
        ]

    def test_code_is_linked_at_the_address_against_the_symbols(self):
        code = self.compile(0x1000, {"helper": 0x20D})
        self.assertEqual(self.calls(code, 0x1000), [0x20C])
        moved = self.compile(0x2000, {"helper": 0x20D})
        self.assertEqual(self.calls(moved, 0x2000), [0x20C])
        self.assertEqual(len(moved), len(code))
        self.assertNotEqual(moved, code)
        self.assertEqual(self.calls(self.compile(0x1000, {"helper": 0x301}), 0x1000), [0x300])

    def test_read_only_data_follows_the_code(self):
        code = self.compile(0x1000, {"helper": 0x20D})
        self.assertIn(b"PATCH\x00", code)
        # The code loads the string's own address, inside the same block.
        at = 0x1000 + code.index(b"PATCH\x00")
        self.assertIn(at.to_bytes(4, "little"), code)

    def test_same_input_same_bytes(self):
        self.assertEqual(
            self.compile(0x1000, {"helper": 0x20D}), self.compile(0x1000, {"helper": 0x20D})
        )

    def test_missing_symbol_is_an_error(self):
        with self.assertRaises(RuntimeError):
            self.compile(0x1000, {})

    def test_wide_branch_encoding(self):
        for source, target in ((0x108, 0x1040), (0x1040, 0x108), (0x190, 0x1174)):
            self.assertEqual(
                check_applied_bytes.thumb_branch_wide(source, target),
                images.branch_wide(source, target),
            )


class AppliedBytes(unittest.TestCase):
    """`plain` replaced through a trampoline by code that calls one inserted helper."""

    @classmethod
    def setUpClass(cls):
        cls.tools = find_tools()
        clang, lld = images.find_tools()
        cls.directory = Path(tempfile.mkdtemp(prefix="check-applied-bytes-"))
        (cls.directory / "firmware.S").write_text(images.FIRMWARE, encoding="utf-8")
        (cls.directory / "spare.S").write_text(SPARE, encoding="utf-8")
        (cls.directory / "link.ld").write_text(images.LINKER_SCRIPT, encoding="utf-8")
        for name in ("firmware", "spare"):
            subprocess.run(
                [clang, "--target=thumbv7m-none-eabi", "-mcpu=cortex-m3", "-c",
                 f"{name}.S", "-o", f"{name}.o"],
                cwd=cls.directory, check=True,
            )
        subprocess.run([lld, "-T", "link.ld", "firmware.o", "-o", "original.elf"],
                       cwd=cls.directory, check=True)
        subprocess.run([lld, "-T", "link.ld", "firmware.o", "spare.o", "-o", "spare.elf"],
                       cwd=cls.directory, check=True)
        cls.original = cls.directory / "original.elf"
        with cls.original.open("rb") as handle:
            cls.symbols = {
                symbol.name: symbol["st_value"]
                for symbol in ELFFile(handle).get_section_by_name(".symtab").iter_symbols()
                if symbol.name and not symbol.name.startswith("$")
            }
        cls.plain = cls.symbols["plain"] & ~1

        # The linker fills the gap between the firmware's code and the spare
        # block with trap bytes.  The rewriter's own segment has no such fill;
        # zero it so the only non-zero added bytes are the ones a test writes.
        end = max(
            start + len(data)
            for start, data in check_applied_bytes.check_trampoline.Memory(cls.original).chunks
            if start < HELPER_AT
        )
        spare = cls.directory / "spare.elf"
        data = bytearray(spare.read_bytes())
        data[cls.offset(end):cls.offset(HELPER_AT)] = bytes(HELPER_AT - end)
        spare.write_bytes(bytes(data))

        cls.snippets = {"patch_fn": HELPER_SNIPPET, "plain": TARGET_SNIPPET}
        cls.link_symbols = {
            "patch_fn": {"helper": cls.symbols["helper"]},
            "plain": {"patch_fn": HELPER_AT | 1},
        }
        cls.helper_code = check_applied_bytes.expected_bytes(
            HELPER_SNIPPET, HELPER_AT, cls.link_symbols["patch_fn"], COMPILER, cls.tools
        )
        cls.body_code = check_applied_bytes.expected_bytes(
            TARGET_SNIPPET, BODY_AT, cls.link_symbols["plain"], COMPILER, cls.tools
        )
        cls.branch = images.branch_wide(cls.plain, BODY_AT)
        cls.writes = {HELPER_AT: cls.helper_code, BODY_AT: cls.body_code, cls.plain: cls.branch}
        cls.rewritten = cls.image("rewritten", cls.writes)
        cls.record = cls.record_for(cls.rewritten, cls.writes)
        cls.expected = check_applied_bytes.expected_functions(cls.record, cls.snippets, cls.tools)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    @classmethod
    def offset(cls, address: int) -> int:
        with (cls.directory / "spare.elf").open("rb") as handle:
            for segment in ELFFile(handle).iter_segments():
                start = segment["p_vaddr"]
                if segment["p_type"] == "PT_LOAD" and start <= address < start + segment["p_filesz"]:
                    return segment["p_offset"] + address - start
        raise AssertionError(hex(address))

    @classmethod
    def image(cls, name: str, writes: dict[int, bytes]) -> Path:
        path = cls.directory / f"{name}.elf"
        data = bytearray((cls.directory / "spare.elf").read_bytes())
        for address, content in writes.items():
            start = cls.offset(address)
            data[start:start + len(content)] = content
        path.write_bytes(bytes(data))
        return path

    @classmethod
    def record_for(cls, rewritten: Path, writes: dict[int, bytes]) -> dict:
        applied = [
            patch_record.Applied(
                "patch_fn", "inserted", HELPER_SNIPPET, cls.link_symbols["patch_fn"],
                [(cls.offset(HELPER_AT), writes[HELPER_AT])],
            ),
            patch_record.Applied(
                "plain", "target", TARGET_SNIPPET, cls.link_symbols["plain"],
                [(cls.offset(cls.plain), writes[cls.plain]), (cls.offset(BODY_AT), writes[BODY_AT])],
            ),
        ]
        return patch_record.build_record(
            cls.original, rewritten, "plain", applied, COMPILER,
            {"patch_fn": ["helper"], "plain": ["patch_fn"]}, [],
        )

    def check(self, rewritten=None, record=None, snippets=None, expected=None):
        report = check_applied_bytes.check_applied_bytes(
            record or self.record,
            self.original,
            rewritten or self.rewritten,
            snippets or self.snippets,
            self.tools,
            expected=expected,
        )
        return report, check_applied_bytes.failures(report)

    def test_image_with_the_intended_bytes_passes(self):
        report, problems = self.check()
        self.assertEqual(problems, [])
        self.assertEqual(
            [(row["name"], row["verdict"], row["size"]) for row in report["functions"]],
            [("patch_fn", "match", len(self.helper_code)), ("plain", "match", len(self.body_code))],
        )
        self.assertEqual(report["entry_branch"]["verdict"], "match")
        self.assertEqual(report["unrecorded"], [])

    def test_every_flipped_byte_is_caught(self):
        # Step B's gate: one corrupted byte anywhere in what was applied.
        missed = []
        for address, content in self.writes.items():
            for index in range(len(content)):
                damaged = bytearray(content)
                damaged[index] ^= 0x01
                rewritten = self.image("flipped", {**self.writes, address: bytes(damaged)})
                # The record stays the one made for the intended bytes.
                _, problems = self.check(rewritten=rewritten, expected=self.expected)
                if not problems:
                    missed.append(hex(address + index))
        self.assertEqual(missed, [])

    def test_flipped_byte_is_named(self):
        damaged = bytearray(self.helper_code)
        damaged[3] ^= 0x80
        rewritten = self.image("named", {**self.writes, HELPER_AT: bytes(damaged)})
        report, problems = self.check(rewritten=rewritten, expected=self.expected)
        self.assertEqual(problems, ["byte-mismatch"])
        helper = report["functions"][0]
        self.assertEqual(helper["verdict"], "byte-mismatch")
        self.assertEqual(helper["first_difference"], f"0x{HELPER_AT + 3:x}")

    def test_entry_branch_must_lead_to_the_body(self):
        elsewhere = images.branch_wide(self.plain, BODY_AT + 0x40)
        rewritten = self.image("elsewhere", {**self.writes, self.plain: elsewhere})
        report, problems = self.check(rewritten=rewritten, expected=self.expected)
        self.assertEqual(problems, ["entry-branch-mismatch"])
        self.assertEqual(report["entry_branch"]["expected"], self.branch.hex())

    def test_snippet_must_be_the_recorded_one(self):
        other = TARGET_SNIPPET.replace("ret void", "call void @patch_fn()\n  ret void")
        _, problems = self.check(snippets={**self.snippets, "plain": other})
        self.assertIn("snippet-mismatch", problems)
        _, problems = self.check(snippets={"patch_fn": HELPER_SNIPPET})
        self.assertIn("snippet-mismatch", problems)

    def test_code_linked_against_another_address_does_not_match(self):
        # The image holds code linked with helper at its real address; a record
        # that claims another address describes different bytes.
        record = copy.deepcopy(self.record)
        record["functions"][0]["symbols"]["helper"] = "0x301"
        _, problems = self.check(record=record)
        self.assertEqual(problems, ["byte-mismatch"])

    def test_changed_byte_of_the_original_image_must_be_recorded(self):
        tiny = self.symbols["tiny"] & ~1
        rewritten = self.image("stray", {**self.writes, tiny: b"\x00\xbf"})
        report, problems = self.check(rewritten=rewritten, expected=self.expected)
        self.assertEqual(problems, ["unrecorded-change"])
        self.assertEqual(report["unrecorded"][0]["address"], f"0x{tiny:x}")

    def test_added_bytes_must_be_recorded(self):
        rewritten = self.image("extra", {**self.writes, 0x1200: b"\x70\x47"})
        report, problems = self.check(rewritten=rewritten, expected=self.expected)
        self.assertEqual(problems, ["unrecorded-code"])
        self.assertEqual(
            report["unrecorded"], [{"kind": "unrecorded-code", "address": "0x1200", "size": 2}]
        )

    def test_in_place_replacement(self):
        # A body that fits is written over the function; there is no branch.
        body = check_applied_bytes.expected_bytes(
            TARGET_SNIPPET, self.plain, self.link_symbols["plain"], COMPILER, self.tools
        )
        self.assertLessEqual(len(body), 8)
        writes = {HELPER_AT: self.helper_code, self.plain: body}
        rewritten = self.image("inplace", writes)
        applied = [
            patch_record.Applied("patch_fn", "inserted", HELPER_SNIPPET,
                                 self.link_symbols["patch_fn"],
                                 [(self.offset(HELPER_AT), self.helper_code)]),
            patch_record.Applied("plain", "target", TARGET_SNIPPET, self.link_symbols["plain"],
                                 [(self.offset(self.plain), body)]),
        ]
        record = patch_record.build_record(
            self.original, rewritten, "plain", applied, COMPILER,
            {"patch_fn": ["helper"], "plain": ["patch_fn"]}, [],
        )
        report, problems = self.check(rewritten=rewritten, record=record)
        self.assertEqual(problems, [])
        self.assertEqual(report["entry_branch"]["verdict"], "none")


if __name__ == "__main__":
    unittest.main()
