#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests for patch_record.py.

Uses the hand-made Thumb images of test_check_trampoline.py, so it needs the
same tools: clang and ld.lld from PATCHESTRY_LLVM_PREFIX / PATCHESTRY_LD_LLD.
"""

from __future__ import annotations

import copy
import hashlib
import shutil
import sys
import unittest
from pathlib import Path

from elftools.elf.elffile import ELFFile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_trampoline  # noqa: E402
import patch_record  # noqa: E402
import test_check_trampoline as images  # noqa: E402


class FakeBinfmt:
    def __init__(self):
        self.file_updates = []


class FakeAnalyzer:
    def __init__(self, symbols):
        self.symbols = symbols

    def get_all_symbols(self):
        return dict(self.symbols)


class FakeRewriter:
    """The three things the recorder reads from a Patcherex instance."""

    def __init__(self, symbols, image_symbols=None):
        self.symbols = dict(symbols)
        self.binfmt_tool = FakeBinfmt()
        self.binary_analyzer = FakeAnalyzer(image_symbols or {})


class FakePatch:
    def __init__(self, writes, defines=None):
        self.writes = writes
        self.defines = defines or {}
        self.applied = 0

    def apply(self, p):
        self.applied += 1
        for offset, content in self.writes:
            p.binfmt_tool.file_updates.append({"offset": offset, "content": content})
        p.symbols.update(self.defines)


class Recorder(unittest.TestCase):
    def test_writes_and_symbols_are_attributed_to_the_patch_that_made_them(self):
        # The rewriter links against its own table and the image's functions;
        # where both name a symbol, the image's address wins.
        p = FakeRewriter({"qemu_callback": 0x2A8, "shadowed": 0x10}, {"uart_write_line": 0x20D, "shadowed": 0x20})
        helper = FakePatch([(0x11000, b"\x01\x02")], defines={"helper": 0x1001})
        target = FakePatch([(0x10108, b"\xaa\xbb\xcc\xdd"), (0x11010, b"\x03\x04\x05\x06")])
        recorder = patch_record.Recorder()
        recorder.watch(helper, "helper", "inserted", "helper snippet")
        recorder.watch(target, "target", "target", "target snippet")

        # Something else writes before and between the patches.
        p.binfmt_tool.file_updates.append({"offset": 0, "content": b"\xff"})
        helper.apply(p)
        p.binfmt_tool.file_updates.append({"offset": 4, "content": b"\xff"})
        target.apply(p)

        self.assertEqual((helper.applied, target.applied), (1, 1))
        first, second = recorder.applied
        self.assertEqual((first.name, first.role, first.snippet), ("helper", "inserted", "helper snippet"))
        self.assertEqual(first.writes, [(0x11000, b"\x01\x02")])
        # Symbols are the ones the link of this patch could see: before it ran.
        image = {"qemu_callback": 0x2A8, "uart_write_line": 0x20D, "shadowed": 0x20}
        self.assertEqual(first.symbols, image)
        self.assertEqual(second.name, "target")
        self.assertEqual(second.writes, [(0x10108, b"\xaa\xbb\xcc\xdd"), (0x11010, b"\x03\x04\x05\x06")])
        self.assertEqual(second.symbols, {**image, "helper": 0x1001})

    def test_unattributed_writes_are_kept(self):
        p = FakeRewriter({})
        patch = FakePatch([(8, b"\x01")])
        recorder = patch_record.Recorder()
        recorder.watch(patch, "f", "target", "")
        p.binfmt_tool.file_updates.append({"offset": 0, "content": b"\xff"})
        patch.apply(p)
        p.binfmt_tool.file_updates.append({"offset": 4, "content": b"\xee\xee"})
        self.assertEqual(recorder.unattributed(p), [(0, b"\xff"), (4, b"\xee\xee")])


class Record(unittest.TestCase):
    """A record of `plain` replaced through a trampoline, with one helper."""

    @classmethod
    def setUpClass(cls):
        images.CheckTrampoline.setUpClass.__func__(cls)
        s = cls.symbols
        cls.branch = images.branch_wide(s["plain"], s["patch_body"])
        cls.rewritten = images.CheckTrampoline.rewrite(cls, "recorded", {s["plain"]: cls.branch})
        memory = check_trampoline.Memory(cls.rewritten)
        cls.body = memory.read(s["patch_body"], 6)
        cls.helper = memory.read(s["not_code"], 8)
        cls.trampoline = check_trampoline.check_trampoline(cls.original, cls.rewritten, "plain", None)
        cls.applied = [
            patch_record.Applied(
                name="helper",
                role="inserted",
                snippet="helper snippet",
                symbols={"uart_write_line": 0x20D, "unrelated": 0x400},
                writes=[(cls.offset(s["not_code"]), cls.helper)],
            ),
            patch_record.Applied(
                name="plain",
                role="target",
                snippet="target snippet",
                symbols={"uart_write_line": 0x20D, "helper": s["not_code"] | 1, "unrelated": 0x400},
                writes=[(cls.offset(s["plain"]), cls.branch), (cls.offset(s["patch_body"]), cls.body)],
            ),
        ]
        cls.compiler = {"path": "/x/clang", "version": "clang 22", "target": "arm-linux-gnueabihf",
                        "flags": ["-fno-pic", "-O0", "-mthumb"]}
        cls.references = {"helper": ["uart_write_line"], "plain": ["helper"]}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    @classmethod
    def offset(cls, address: int) -> int:
        with cls.rewritten.open("rb") as handle:
            for segment in ELFFile(handle).iter_segments():
                start = segment["p_vaddr"]
                if segment["p_type"] == "PT_LOAD" and start <= address < start + segment["p_filesz"]:
                    return segment["p_offset"] + address - start
        raise AssertionError(hex(address))

    def build(self, applied=None, unattributed=()):
        return patch_record.build_record(
            self.original, self.rewritten, "plain", applied or self.applied,
            self.compiler, self.references, list(unattributed),
        )

    def check(self, record, rewritten=None, trampoline=None):
        return patch_record.check_record(
            record, rewritten or self.rewritten, trampoline or self.trampoline
        )

    def test_record_contents(self):
        record = self.build()
        s = self.symbols
        self.assertEqual(record["target"], {"name": "plain", "address": f"0x{s['plain']:x}", "size": 8, "mode": "thumb"})
        self.assertEqual(record["compiler"], self.compiler)
        self.assertEqual(record["original"]["sha256"],
                         hashlib.sha256(self.original.read_bytes()).hexdigest())
        self.assertEqual(record["rewritten"]["sha256"],
                         hashlib.sha256(self.rewritten.read_bytes()).hexdigest())
        self.assertEqual([f["name"] for f in record["functions"]], ["helper", "plain"])

        helper, plain = record["functions"]
        self.assertEqual(helper["role"], "inserted")
        self.assertEqual(helper["address"], f"0x{s['not_code']:x}")
        self.assertEqual(helper["size"], 8)
        self.assertEqual(helper["snippet_sha256"], hashlib.sha256(b"helper snippet").hexdigest())
        # Only the symbols the snippet references, at the address its link saw.
        self.assertEqual(helper["symbols"], {"uart_write_line": "0x20d"})
        self.assertEqual(plain["symbols"], {"helper": f"0x{s['not_code'] | 1:x}"})

        self.assertEqual(plain["role"], "target")
        self.assertEqual(plain["placement"], "trampoline")
        self.assertEqual(plain["address"], f"0x{s['patch_body']:x}")
        self.assertEqual(plain["size"], 6)
        self.assertEqual(
            plain["writes"],
            [
                {"kind": "entry-branch", "file_offset": f"0x{self.offset(s['plain']):x}",
                 "address": f"0x{s['plain']:x}", "size": 4,
                 "sha256": hashlib.sha256(self.branch).hexdigest()},
                {"kind": "body", "file_offset": f"0x{self.offset(s['patch_body']):x}",
                 "address": f"0x{s['patch_body']:x}", "size": 6,
                 "sha256": hashlib.sha256(self.body).hexdigest()},
            ],
        )

    def test_consistent_record_has_no_problems(self):
        self.assertEqual(self.check(self.build()), {})

    def test_in_place_target(self):
        s = self.symbols
        new_body = b"\x00\xbf" * 4
        rewritten = images.CheckTrampoline.rewrite(self, "inplace", {s["plain"]: new_body})
        applied = [patch_record.Applied("plain", "target", "t", {}, [(self.offset(s["plain"]), new_body)])]
        record = patch_record.build_record(
            self.original, rewritten, "plain", applied, self.compiler, {"plain": []}, []
        )
        plain = record["functions"][0]
        self.assertEqual(plain["placement"], "in-place")
        self.assertEqual(plain["address"], f"0x{s['plain']:x}")
        self.assertEqual([w["kind"] for w in plain["writes"]], ["body"])
        trampoline = check_trampoline.check_trampoline(self.original, rewritten, "plain", None)
        self.assertEqual(patch_record.check_record(record, rewritten, trampoline), {})

    def test_destination_must_be_where_the_entry_branch_leads(self):
        record = self.build()
        moved = copy.deepcopy(record)
        moved["functions"][1]["address"] = "0x1100"
        self.assertIn("destination-mismatch", self.check(moved))
        # Also when the placement itself disagrees with what the image shows.
        flipped = copy.deepcopy(record)
        flipped["functions"][1]["placement"] = "in-place"
        self.assertIn("destination-mismatch", self.check(flipped))

    def test_recorded_bytes_must_be_in_the_image(self):
        record = self.build()
        s = self.symbols
        damaged = images.CheckTrampoline.rewrite(
            self, "damaged", {s["plain"]: self.branch, s["patch_body"] + 2: b"\xff"}
        )
        trampoline = check_trampoline.check_trampoline(self.original, damaged, "plain", None)
        problems = patch_record.check_record(record, damaged, trampoline)
        self.assertIn("write-mismatch", problems)
        self.assertIn("plain", problems["write-mismatch"])
        self.assertIn("image-mismatch", problems)  # the file is not the one recorded

    def test_write_outside_the_loaded_image(self):
        applied = copy.deepcopy(self.applied)
        applied[0].writes = [(0x7FFFFF0, b"\x00\x00")]
        problems = self.check(self.build(applied))
        self.assertIn("unmapped-write", problems)

    def test_inserted_function_must_be_linked_at_its_recorded_address(self):
        applied = copy.deepcopy(self.applied)
        applied[1].symbols["helper"] = 0x2001  # the invented-address failure of G13/G14
        problems = self.check(self.build(applied))
        self.assertIn("symbol-mismatch", problems)
        self.assertIn("helper", problems["symbol-mismatch"])

    def test_reference_without_an_address_is_reported(self):
        applied = copy.deepcopy(self.applied)
        del applied[0].symbols["uart_write_line"]
        problems = self.check(self.build(applied))
        self.assertIn("symbol-mismatch", problems)
        self.assertIn("uart_write_line", problems["symbol-mismatch"])

    def test_target_must_be_recorded(self):
        problems = self.check(self.build(self.applied[:1]))
        self.assertIn("missing-target", problems)

    def test_writes_no_patch_made_are_reported(self):
        record = self.build(unattributed=[(self.offset(self.symbols["tiny"]), b"\x00")])
        self.assertEqual(record["unattributed_writes"][0]["address"], f"0x{self.symbols['tiny']:x}")
        self.assertEqual(record["unattributed_writes"][0]["kind"], "loaded")
        self.assertIn("unattributed-write", self.check(record))
        # So is a write that is neither loaded nor part of the ELF structure.
        record = self.build(unattributed=[(0x7FFFFF0, b"\x00")])
        self.assertEqual(record["unattributed_writes"][0]["kind"], "unknown")
        self.assertIn("unattributed-write", self.check(record))

    def test_rewriting_the_elf_structure_is_expected(self):
        # Saving the image rewrites its header and header tables.
        with self.rewritten.open("rb") as handle:
            header = ELFFile(handle).header
        writes = [
            (0, b"\x00" * header["e_ehsize"]),
            (header["e_phoff"], b"\x00" * (header["e_phnum"] * header["e_phentsize"])),
            (header["e_shoff"], b"\x00" * (header["e_shnum"] * header["e_shentsize"])),
        ]
        record = self.build(unattributed=writes)
        self.assertEqual(
            [w["kind"] for w in record["unattributed_writes"]],
            ["elf-header", "program-headers", "section-headers"],
        )
        self.assertEqual(self.check(record), {})

    def test_entry_point_must_not_move(self):
        record = self.build()
        self.assertEqual(record["original"]["entry"], record["rewritten"]["entry"])
        moved = self.directory / "moved-entry.elf"
        data = bytearray(self.rewritten.read_bytes())
        data[0x18:0x1C] = (0x1001).to_bytes(4, "little")  # e_entry of a 32-bit ELF
        moved.write_bytes(bytes(data))
        problems = patch_record.check_record(record, moved, self.trampoline)
        self.assertIn("entry-changed", problems)

    def test_original_segments_must_still_load_the_same_way(self):
        record = self.build()
        self.assertTrue(record["original"]["segments"])
        altered = copy.deepcopy(record)
        altered["original"]["segments"][0]["flags"] ^= 2  # was writable, or was not
        self.assertIn("segment-changed", self.check(altered))
        shifted = copy.deepcopy(record)
        shifted["original"]["segments"][0]["address"] = "0x40000000"
        self.assertIn("segment-changed", self.check(shifted))


if __name__ == "__main__":
    unittest.main()
