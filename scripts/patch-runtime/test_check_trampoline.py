#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests for check_trampoline.py on small hand-made Thumb images.

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

from elftools.elf.elffile import ELFFile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_trampoline  # noqa: E402
from check_bindings import UnsupportedImage  # noqa: E402

# Every function is 8 bytes unless its name says otherwise.
FIRMWARE = r"""
    .syntax unified
    .thumb
    .text

    .macro function name
    .global \name
    .type \name, %function
    .thumb_func
\name:
    .endm

    function tiny
    bx lr
    .size tiny, . - tiny

    function neighbour
    push {r7, lr}
    bl helper
    pop {r7, pc}
    .size neighbour, . - neighbour

    function plain
    push {r7, lr}
plain_middle:
    bl helper
    pop {r7, pc}
    .size plain, . - plain

    function helper
    bx lr
    .size helper, . - helper

    function looped
    push {r7, lr}
    nop
    nop
looped_middle:
    nop
    bne.n looped_middle
    pop {r7, pc}
    .size looped, . - looped

    function jumper
    b.n looped_middle
    .size jumper, . - jumper

    function outer
    nop
    nop
    function inner
    nop
    bx lr
    .size inner, . - inner
    .size outer, . - outer

    function padded
    nop
    bx lr
    .size padded, . - padded
    @ Stands in for linker fill: bytes after the end of a sized symbol that
    @ happen to read as a branch into plain.
    bmi.n plain_middle
    nop

    function wide
    b.w helper
    .size wide, . - wide

    function unsized
    nop
    nop
    nop
    bx lr

    .data
    .global word
word:
    .long 0x12345678
"""

# Linked only into the "rewritten" image: code the rewriter would have added.
ADDED = r"""
    .syntax unified
    .thumb
    .section .patch, "ax", %progbits
    .global patch_body
    .type patch_body, %function
    .thumb_func
patch_body:
    push {r7, lr}
    nop
    pop {r7, pc}

    .section .ronly, "a", %progbits
    .global not_code
not_code:
    .long 0, 0
"""

LINKER_SCRIPT = """
ENTRY(plain)
SECTIONS
{
  .text 0x0 : { *(.text*) }
  .patch 0x1000 : { KEEP(*(.patch)) }
  .ronly 0x3000 : { KEEP(*(.ronly)) }
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


def branch_wide(source: int, target: int, exchange: bool = False) -> bytes:
    """Encode Thumb-2 `b.w target` (or `blx target`) placed at `source`."""
    offset = target - (source + 4)
    sign = 1 if offset < 0 else 0
    offset &= 0x1FFFFFF
    i1, i2 = (offset >> 23) & 1, (offset >> 22) & 1
    j1, j2 = (~(i1 ^ sign)) & 1, (~(i2 ^ sign)) & 1
    high = 0xF000 | (sign << 10) | ((offset >> 12) & 0x3FF)
    low = (0xC000 if exchange else 0x9000) | (j1 << 13) | (j2 << 11) | ((offset >> 1) & 0x7FF)
    if exchange:
        low &= ~1
    return high.to_bytes(2, "little") + low.to_bytes(2, "little")


class CheckTrampoline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clang, lld = find_tools()
        cls.directory = Path(tempfile.mkdtemp(prefix="check-trampoline-"))
        (cls.directory / "firmware.S").write_text(FIRMWARE, encoding="utf-8")
        (cls.directory / "added.S").write_text(ADDED, encoding="utf-8")
        (cls.directory / "link.ld").write_text(LINKER_SCRIPT, encoding="utf-8")
        for name in ("firmware", "added"):
            subprocess.run(
                [clang, "--target=thumbv7m-none-eabi", "-mcpu=cortex-m3", "-c",
                 f"{name}.S", "-o", f"{name}.o"],
                cwd=cls.directory, check=True,
            )
        # Same script and same first object, so .text is identical in both.
        subprocess.run([lld, "-T", "link.ld", "firmware.o", "-o", "original.elf"],
                       cwd=cls.directory, check=True)
        subprocess.run([lld, "-T", "link.ld", "firmware.o", "added.o", "-o", "added.elf"],
                       cwd=cls.directory, check=True)
        cls.original = cls.directory / "original.elf"
        cls.symbols = {}
        with cls.original.open("rb") as handle:
            for symbol in ELFFile(handle).get_section_by_name(".symtab").iter_symbols():
                if symbol.name and not symbol.name.startswith("$"):
                    cls.symbols[symbol.name] = symbol["st_value"] & ~1
        with (cls.directory / "added.elf").open("rb") as handle:
            for symbol in ELFFile(handle).get_section_by_name(".symtab").iter_symbols():
                if symbol.name in ("patch_body", "not_code"):
                    cls.symbols[symbol.name] = symbol["st_value"] & ~1

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def rewrite(self, name: str, writes: dict[int, bytes]) -> Path:
        """Copy the image with the added code and write bytes at addresses."""
        path = self.directory / f"{name}.elf"
        shutil.copyfile(self.directory / "added.elf", path)
        with path.open("r+b") as handle:
            segments = [
                (s["p_vaddr"], s["p_filesz"], s["p_offset"])
                for s in ELFFile(handle).iter_segments() if s["p_type"] == "PT_LOAD"
            ]
            for address, data in writes.items():
                start, _, offset = next(
                    s for s in segments if s[0] <= address and address + len(data) <= s[0] + s[1]
                )
                handle.seek(offset + address - start)
                handle.write(data)
        return path

    def check(self, name: str, function: str, writes: dict[int, bytes], function_json=None):
        rewritten = self.rewrite(name, writes)
        return check_trampoline.check_trampoline(self.original, rewritten, function, function_json)

    def trampoline(self, function: str, target: str | int) -> dict[int, bytes]:
        source = self.symbols[function]
        if isinstance(target, str):
            target = self.symbols[target]
        return {source: branch_wide(source, target)}

    def test_layout(self):
        # The cases below depend on these sizes and neighbours.
        s = self.symbols
        self.assertEqual(s["neighbour"] - s["tiny"], 2)
        self.assertEqual(s["helper"] - s["plain"], 8)
        self.assertEqual(s["patch_body"], 0x1000)

    def test_trampoline(self):
        report = self.check("trampoline", "plain", self.trampoline("plain", "patch_body"))
        self.assertEqual(report["verdict"], "trampoline")
        self.assertEqual(report["problems"], {})
        self.assertEqual(report["destination"], "0x1000")
        start = self.symbols["plain"]
        self.assertEqual(report["span"], [f"0x{start:x}", f"0x{start + 4:x}"])
        # The cut is inside the 4-byte bl that starts at +2.
        self.assertTrue(report["tail_split"])
        self.assertEqual(report["entry"]["instruction"], "b.w #0x1000")

    def test_padding_is_not_a_branch(self):
        report = self.check("padding", "plain", self.trampoline("plain", "patch_body"))
        self.assertEqual(report["incoming"], [])
        # The same bytes do count when the gap is not excluded.
        image = check_trampoline.load_image(self.original)
        gap = self.symbols["padded"] + 4
        self.assertIn((gap, gap + 4), check_trampoline.padding_gaps(image))
        memory = check_trampoline.Memory(self.original)
        decoded = next(check_trampoline.decode_thumb(memory, gap, gap + 2))
        self.assertEqual(check_trampoline.direct_target(decoded[2], decoded[3]),
                         self.symbols["plain"] + 2)

    def test_branch_sharing_a_byte_with_the_old_code(self):
        # `wide` is itself one b.w, so the new branch repeats its last byte.
        start = self.symbols["wide"]
        branch = branch_wide(start, self.symbols["patch_body"])
        original = check_trampoline.Memory(self.original).read(start, 4)
        self.assertEqual(original[3], branch[3])
        self.assertNotEqual(original[:3], branch[:3])
        report = self.check("same-byte", "wide", {start: branch})
        self.assertEqual(report["problems"], {})
        self.assertEqual(report["changed"][-1][1], f"0x{start + 3:x}")
        self.assertEqual(report["span"], [f"0x{start:x}", f"0x{start + 4:x}"])
        self.assertFalse(report["tail_split"])

    def test_in_place(self):
        start = self.symbols["plain"]
        body = bytes.fromhex("00bf00bf00bf7047")  # nop; nop; nop; bx lr
        report = self.check("in-place", "plain", {start: body})
        self.assertEqual(report["verdict"], "in-place")
        self.assertEqual(report["problems"], {})
        self.assertNotIn("destination", report)

    def test_unchanged(self):
        report = self.check("unchanged", "plain", {})
        self.assertEqual(list(report["problems"]), ["unchanged"])

    def test_function_shorter_than_branch(self):
        report = self.check("short", "tiny", self.trampoline("tiny", "patch_body"))
        self.assertEqual(report["verdict"], "trampoline")
        self.assertEqual(list(report["problems"]), ["span-outside-function"])
        row = report["problems"]["span-outside-function"][0]
        self.assertEqual(row["belongs_to"], "neighbour")

    def test_change_in_another_function(self):
        writes = self.trampoline("plain", "patch_body")
        writes[self.symbols["helper"]] = bytes.fromhex("00bf")
        report = self.check("stray", "plain", writes)
        self.assertEqual(list(report["problems"]), ["span-outside-function"])
        self.assertEqual(report["problems"]["span-outside-function"][0]["belongs_to"], "helper")

    def test_destination_in_original_image(self):
        report = self.check("dest-original", "plain", self.trampoline("plain", "helper"))
        self.assertEqual(list(report["problems"]), ["bad-destination"])
        self.assertIn("inside the original image", report["problems"]["bad-destination"][0])

    def test_destination_unmapped(self):
        report = self.check("dest-unmapped", "plain", self.trampoline("plain", 0x2000))
        self.assertEqual(list(report["problems"]), ["bad-destination"])
        self.assertIn("not in a loaded segment", report["problems"]["bad-destination"][0])

    def test_destination_not_executable(self):
        report = self.check("dest-data", "plain", self.trampoline("plain", "not_code"))
        self.assertEqual(list(report["problems"]), ["bad-destination"])
        self.assertIn("permissions r--", report["problems"]["bad-destination"][0])

    def test_mode_switch(self):
        start = self.symbols["plain"]
        writes = {start: branch_wide(start, self.symbols["patch_body"], exchange=True)}
        report = self.check("blx", "plain", writes)
        self.assertEqual(list(report["problems"]), ["not-thumb"])

    def test_incoming_branch(self):
        report = self.check("incoming", "looped", self.trampoline("looped", "patch_body"))
        self.assertEqual(list(report["problems"]), ["incoming-branch"])
        rows = report["problems"]["incoming-branch"]
        self.assertEqual([row["in"] for row in rows], ["jumper"])
        # The function's own backward branch is listed but is not a problem.
        self.assertEqual([row["in"] for row in report["incoming"] if row["internal"]], ["looped"])

    def test_alternate_entry_symbol(self):
        report = self.check("alternate", "outer", self.trampoline("outer", "patch_body"))
        self.assertEqual(list(report["problems"]), ["alternate-entry"])
        self.assertIn("inner", report["problems"]["alternate-entry"][0])

    def test_alternate_entry_from_ghidra_ranges(self):
        start = self.symbols["plain"]
        record = {"functions": {f"ram:{start:08x}": {
            "name": "plain",
            "address_ranges": [
                {"start": f"ram:{start:08x}", "end": f"ram:{start + 3:08x}"},
                {"start": f"ram:{start + 6:08x}", "end": f"ram:{start + 7:08x}"},
            ],
        }}}
        writes = self.trampoline("plain", "patch_body")
        report = self.check("ranges", "plain", writes, record)
        self.assertEqual(list(report["problems"]), ["alternate-entry"])

        record["functions"][f"ram:{start:08x}"]["address_ranges"] = [
            {"start": f"ram:{start:08x}", "end": f"ram:{start + 7:08x}"}
        ]
        report = self.check("one-range", "plain", writes, record)
        self.assertEqual(report["problems"], {})

    def test_unsized_function(self):
        report = self.check("unsized", "unsized", self.trampoline("unsized", "patch_body"))
        self.assertIn("unknown-extent", report["problems"])

    def test_missing_function(self):
        with self.assertRaises(LookupError):
            self.check("missing", "no_such_function", {})

    def test_relocatable_input_is_rejected(self):
        with self.assertRaises(UnsupportedImage):
            check_trampoline.check_trampoline(
                self.directory / "firmware.o", self.directory / "added.elf", "plain"
            )


if __name__ == "__main__":
    unittest.main()
