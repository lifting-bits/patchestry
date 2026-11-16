#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests for the checks on a stripped, dynamically linked ARM-mode executable.

The other test files use bare-metal Thumb images with a symbol table.  A Linux
user-space binary differs in four ways the checks have to be told about: it
has a program interpreter and a dynamic section, it may have no symbol table,
its functions may be ARM-mode code, and it reaches imported functions through
PLT stubs.

Needs clang, ld.lld and llvm-objcopy from PATCHESTRY_LLVM_PREFIX.
"""

from __future__ import annotations

import hashlib
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
import check_bindings  # noqa: E402
import check_trampoline  # noqa: E402
import patch_record  # noqa: E402
from check_bindings import UnsupportedImage  # noqa: E402

LIBRARY = r"""
    .arm
    .text
    .global first_import
    .type first_import, %function
first_import:
    bx lr
    .global second_import
    .type second_import, %function
second_import:
    bx lr
"""

PROGRAM = r"""
    .arm
    .text

    .macro function name
    .global \name
    .type \name, %function
\name:
    .endm

    function plain
    push {r11, lr}
    bl first_import
    bl second_import
    pop {r11, pc}
    .size plain, . - plain

    function looped
    push {r11, lr}
looped_middle:
    subs r0, r0, #1
    bne looped_middle
    pop {r11, pc}
    .size looped, . - looped

    function jumper
    b looped_middle
    .size jumper, . - jumper

    function _start
    bl plain
    bl looped
    bx lr
    .size _start, . - _start
"""

# Linked only into the "rewritten" image: room for the rewriter's code.
SPARE = r"""
    .section .patch, "ax", %progbits
    .global patch_area
patch_area:
    .space 256
"""

# Every section is placed by name so that adding the spare block moves nothing.
LINKER_SCRIPT = """
SECTIONS
{
  . = 0x10000 + SIZEOF_HEADERS;
  .interp : { *(.interp) }
  .dynsym : { *(.dynsym) }
  .gnu.hash : { *(.gnu.hash) }
  .hash : { *(.hash) }
  .dynstr : { *(.dynstr) }
  .rel.dyn : { *(.rel.dyn) }
  .rel.plt : { *(.rel.plt) }
  .text 0x20000 : { *(.text*) }
  .plt : { *(.plt) *(.iplt) }
  .dynamic 0x30000 : { *(.dynamic) }
  .got : { *(.got) }
  .got.plt : { *(.got.plt) }
  .data : { *(.data*) }
  .bss : { *(.bss*) }
  .patch 0x80000 : { KEEP(*(.patch)) }
  /DISCARD/ : { *(.ARM.exidx*) *(.ARM.attributes) }
}
"""

TEXT_AT = 0x20000
PATCH_AT = 0x80000
RETURN = bytes.fromhex("1eff2fe1")  # bx lr


def find_tools() -> dict[str, str]:
    prefix = os.environ.get("PATCHESTRY_LLVM_PREFIX")
    if not prefix:
        raise unittest.SkipTest("set PATCHESTRY_LLVM_PREFIX to the LLVM bin directory")
    tools = {
        "clang": str(Path(prefix) / "clang"),
        "lld": os.environ.get("PATCHESTRY_LD_LLD") or str(Path(prefix) / "ld.lld"),
        "objcopy": str(Path(prefix) / "llvm-objcopy"),
    }
    missing = [name for name, path in tools.items() if not Path(path).exists()]
    if missing:
        raise unittest.SkipTest(f"missing LLVM tools: {', '.join(missing)}")
    return tools


def blx_arm(source: int, target: int) -> bytes:
    """ARM-mode `blx target`, which enters Thumb code."""
    offset = target - (source + 8)
    word = 0xFA000000 | (((offset >> 1) & 1) << 24) | ((offset >> 2) & 0xFFFFFF)
    return word.to_bytes(4, "little")


class LinuxArmImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tools = find_tools()
        cls.directory = Path(tempfile.mkdtemp(prefix="linux-arm-images-"))
        for name, text in (("library", LIBRARY), ("program", PROGRAM), ("spare", SPARE)):
            (cls.directory / f"{name}.S").write_text(text, encoding="utf-8")
            subprocess.run(
                [tools["clang"], "--target=armv7-linux-gnueabihf", "-c", f"{name}.S",
                 "-o", f"{name}.o"],
                cwd=cls.directory, check=True,
            )
        subprocess.run([tools["lld"], "-shared", "library.o", "-o", "library.so"],
                       cwd=cls.directory, check=True)
        (cls.directory / "link.ld").write_text(LINKER_SCRIPT, encoding="utf-8")
        link = [tools["lld"], "--dynamic-linker=/lib/ld.so.1", "-T", "link.ld",
                "program.o", "library.so"]
        subprocess.run([*link, "spare.o", "-o", "added.sym.elf"], cwd=cls.directory, check=True)
        cls.symbols = {}
        with (cls.directory / "added.sym.elf").open("rb") as handle:
            for symbol in ELFFile(handle).get_section_by_name(".symtab").iter_symbols():
                if symbol.name and not symbol.name.startswith("$"):
                    cls.symbols[symbol.name] = int(symbol["st_value"])
        subprocess.run(
            [tools["objcopy"], "--strip-all", "added.sym.elf", "added.elf"],
            cwd=cls.directory, check=True,
        )
        # The spare block's segment is made the last program header.  The
        # original is then the same file with one program header fewer, so the
        # two differ in e_phnum only, as after a rewriter appends a segment.
        added = cls.directory / "added.elf"
        with added.open("r+b") as handle:
            elf = ELFFile(handle)
            start, size, count = elf["e_phoff"], elf["e_phentsize"], elf["e_phnum"]
            handle.seek(start)
            entries = [handle.read(size) for _ in range(count)]
            spare = next(
                index for index, segment in enumerate(elf.iter_segments())
                if segment["p_type"] == "PT_LOAD" and segment["p_vaddr"] == PATCH_AT
            )
            entries.append(entries.pop(spare))
            handle.seek(start)
            handle.write(b"".join(entries))
        cls.original = cls.directory / "original.elf"
        shutil.copyfile(added, cls.original)
        with cls.original.open("r+b") as handle:
            handle.seek(0x2C)
            handle.write((count - 1).to_bytes(2, "little"))  # e_phnum
        cls.header_at = 0x10000  # the first segment loads the file from offset 0
        cls.library = cls.directory / "library.so"

        # Where the two calls in `plain` go: the PLT stubs of the two imports.
        memory = check_trampoline.Memory(cls.original)
        decoder = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM)
        plain = cls.symbols["plain"]
        calls = [
            int(instruction.op_str.lstrip("#"), 16)
            for instruction in decoder.disasm(memory.read(plain, 16), plain)
            if instruction.mnemonic == "bl"
        ]
        cls.first_stub, cls.second_stub = calls

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def function_json(self, name: str, size: int) -> dict:
        start = self.symbols[name]
        return {
            "functions": {
                f"ram:{start:08x}": {
                    "name": name,
                    "entry_point": f"ram:{start:08x}",
                    "address_ranges": [
                        {"start": f"ram:{start:08x}", "end": f"ram:{start + size - 1:08x}"}
                    ],
                }
            }
        }

    def rewrite(self, name: str, writes: dict[int, bytes]) -> Path:
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

    def moved(self, function: str) -> dict[int, bytes]:
        """The function's entry branches to a body in the spare block."""
        source = self.symbols[function]
        return {
            source: check_applied_bytes.arm_branch(source, PATCH_AT),
            PATCH_AT: RETURN,
        }

    def check(self, name: str, function: str, size: int, writes: dict[int, bytes], **options):
        options.setdefault("mode", "arm")
        options.setdefault("allow_dynamic", True)
        return check_trampoline.check_trampoline(
            self.original, self.rewrite(name, writes), function,
            self.function_json(function, size), **options,
        )

    # The image.

    def test_layout(self):
        s = self.symbols
        self.assertEqual(s["plain"], TEXT_AT)
        self.assertEqual(s["looped"] - s["plain"], 16)
        self.assertEqual(s["patch_area"], PATCH_AT)
        original = check_trampoline.Memory(self.original)
        added = check_trampoline.Memory(self.directory / "added.elf")
        self.assertIsNone(original.read(PATCH_AT, 4))
        self.assertEqual(added.read(PATCH_AT, 256), bytes(256))
        self.assertEqual(original.read(self.header_at, 4), b"\x7fELF")
        with self.original.open("rb") as handle:
            self.assertIsNone(ELFFile(handle).get_section_by_name(".symtab"))

    def test_dynamic_executable_needs_to_be_allowed(self):
        with self.assertRaises(UnsupportedImage):
            check_bindings.load_image(self.original)
        image = check_bindings.load_image(self.original, allow_dynamic=True)
        self.assertTrue(image.segment_at(TEXT_AT).flags & check_bindings.PF_X)

    def test_shared_object_is_never_supported(self):
        with self.assertRaises(UnsupportedImage):
            check_bindings.load_image(self.library, allow_dynamic=True)

    # Imports.

    def test_stub_names_the_import_it_jumps_through(self):
        self.assertEqual(
            check_bindings.plt_stub_import(self.original, self.first_stub), "first_import"
        )
        self.assertEqual(
            check_bindings.plt_stub_import(self.original, self.second_stub), "second_import"
        )

    def test_other_code_is_not_a_stub(self):
        self.assertIsNone(check_bindings.plt_stub_import(self.original, self.symbols["plain"]))
        self.assertIsNone(check_bindings.plt_stub_import(self.original, self.first_stub + 4))
        self.assertIsNone(check_bindings.plt_stub_import(self.original, 0x7000000))

    def bindings(self, stubs):
        manifest = {
            "bindings": [
                {"name": "first_import", "symbol": "first_import", "kind": "function",
                 "origin": "imported", "address": "0x1", "defined": False},
                {"name": "plain", "symbol": "plain", "kind": "function", "origin": "original",
                 "address": f"0x{self.symbols['plain']:x}", "size": 16, "defined": True},
            ]
        }
        report = check_bindings.check_bindings(
            manifest, self.original, allow_dynamic=True, stubs=stubs
        )
        return report, {row["name"]: row for row in report["bindings"]}

    def test_import_is_bound_through_its_stub(self):
        report, rows = self.bindings({"first_import": self.first_stub})
        self.assertEqual(rows["first_import"]["verdict"], "plt-stub")
        self.assertEqual(rows["first_import"]["stub"], f"0x{self.first_stub:x}")
        self.assertEqual(check_bindings.failures(report, strict=False), [])
        # The patch link has to be given both: neither has an ELF symbol.
        self.assertEqual(
            report["absolute_symbols"],
            {"first_import": f"0x{self.first_stub:x}", "plain": f"0x{self.symbols['plain']:x}"},
        )

    def test_stripped_function_is_unnamed(self):
        _, rows = self.bindings({"first_import": self.first_stub})
        self.assertEqual(rows["plain"]["verdict"], "unnamed")

    def test_import_without_a_stub_fails(self):
        report, rows = self.bindings({})
        self.assertEqual(rows["first_import"]["verdict"], "imported")
        self.assertEqual(len(check_bindings.failures(report, strict=False)), 1)

    def test_stub_of_another_import_fails(self):
        report, rows = self.bindings({"first_import": self.second_stub})
        self.assertEqual(rows["first_import"]["verdict"], "imported")
        self.assertEqual(rows["first_import"]["stub_resolves_to"], "second_import")

    # Reinsertion.

    def test_arm_trampoline(self):
        report = self.check("moved", "plain", 16, self.moved("plain"))
        self.assertEqual(report["verdict"], "trampoline")
        self.assertEqual(report["problems"], {})
        self.assertEqual(report["destination"], f"0x{PATCH_AT:x}")
        self.assertEqual(report["mode"], "arm")
        self.assertEqual(report["extent_source"], "function-json")
        self.assertEqual(report["size"], 16)
        self.assertEqual(report["span"], [f"0x{TEXT_AT:x}", f"0x{TEXT_AT + 4:x}"])

    def test_stripped_function_needs_its_extent(self):
        rewritten = self.rewrite("no-extent", self.moved("plain"))
        with self.assertRaises(LookupError):
            check_trampoline.check_trampoline(
                self.original, rewritten, "plain", None, mode="arm", allow_dynamic=True
            )

    def test_arm_function_is_refused_without_the_mode(self):
        report = self.check("thumb-policy", "plain", 16, self.moved("plain"), mode=None)
        self.assertIn("not-thumb", report["problems"])

    def test_change_past_the_extent(self):
        writes = self.moved("plain")
        writes[self.symbols["looped"]] = RETURN
        report = self.check("spill", "plain", 16, writes)
        self.assertIn("span-outside-function", report["problems"])

    def test_header_table_fields_may_change(self):
        # The added segment shows as one more program header in the loaded
        # copy of the ELF header.
        report = self.check("tables", "plain", 16, self.moved("plain"))
        self.assertEqual(report["problems"], {})
        count = f"0x{self.header_at + 0x2c:x}"
        self.assertEqual(report["header_changes"], [[count, count]])
        self.assertEqual(report["changed"], [[f"0x{TEXT_AT:x}", f"0x{TEXT_AT + 4:x}"]])

    def test_other_header_fields_may_not_change(self):
        writes = self.moved("plain")
        writes[self.header_at + 0x18] = bytes.fromhex("00008000")  # e_entry
        report = self.check("entry", "plain", 16, writes)
        self.assertIn("span-outside-function", report["problems"])

    def test_incoming_branch(self):
        report = self.check("incoming", "looped", 16, self.moved("looped"))
        sources = [row["from"] for row in report["problems"]["incoming-branch"]]
        self.assertEqual(sources, [f"0x{self.symbols['jumper']:x}"])

    def test_entry_that_switches_mode(self):
        source = self.symbols["plain"]
        report = self.check(
            "switch", "plain", 16, {source: blx_arm(source, PATCH_AT), PATCH_AT: RETURN}
        )
        self.assertIn("mode-switch", report["problems"])

    def test_branch_mnemonics(self):
        target = check_trampoline.direct_target
        self.assertEqual(target("b", "#0x100"), 0x100)
        self.assertEqual(target("bne", "#0x100"), 0x100)
        self.assertEqual(target("bls", "#0x100"), 0x100)
        self.assertEqual(target("blt", "#0x100"), 0x100)
        self.assertEqual(target("bleq", "#0x100"), 0x100)
        self.assertEqual(target("blxne", "#0x100"), 0x100)
        self.assertEqual(target("bne.w", "#0x100"), 0x100)
        self.assertEqual(target("cbz", "r0, #0x100"), 0x100)
        # Not branches, whatever their last operand looks like.
        self.assertIsNone(target("bic", "r0, r0, #0x100"))
        self.assertIsNone(target("bfc", "r0, #0, #0x100"))
        self.assertIsNone(target("bkpt", "#0x100"))
        self.assertIsNone(target("bx", "lr"))

    # Bytes.

    def test_arm_branch_encoding(self):
        s = self.symbols
        memory = check_trampoline.Memory(self.original)
        # `jumper` is `b looped_middle`, and the assembler encoded it.
        self.assertEqual(
            check_applied_bytes.arm_branch(s["jumper"], s["looped"] + 4),
            memory.read(s["jumper"], 4),
        )
        decoder = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM)
        for source, target in ((0x7AE8C, 0x118F4C), (0x118F4C, 0x7AE8C), (0x20000, 0x20008)):
            instruction = next(
                decoder.disasm(check_applied_bytes.arm_branch(source, target), source)
            )
            self.assertEqual((instruction.mnemonic, instruction.op_str), ("b", f"#{target:#x}"))

    def test_rewritten_header_is_structure_even_when_loaded(self):
        rewritten = self.rewrite("header", self.moved("plain"))
        record = self.record(rewritten, "arm")
        self.assertEqual(
            [(row["kind"], row["address"]) for row in record["unattributed_writes"]],
            [("elf-header", f"0x{self.header_at:x}")],
        )
        trampoline = check_trampoline.check_trampoline(
            self.original, rewritten, "plain", self.function_json("plain", 16),
            mode="arm", allow_dynamic=True,
        )
        self.assertEqual(patch_record.check_record(record, rewritten, trampoline), {})
        report = check_applied_bytes.check_applied_bytes(
            record, self.original, rewritten, {"plain": "snippet"}, None,
            expected={"plain": ("ok", RETURN)},
        )
        self.assertEqual(report["unrecorded"], [])

    def test_header_change_no_write_accounts_for(self):
        rewritten = self.rewrite("header-unrecorded", self.moved("plain"))
        record = self.record(rewritten, "arm", header_write=False)
        report = check_applied_bytes.check_applied_bytes(
            record, self.original, rewritten, {"plain": "snippet"}, None,
            expected={"plain": ("ok", RETURN)},
        )
        self.assertEqual(
            report["unrecorded"],
            [{"kind": "unrecorded-change", "address": f"0x{self.header_at + 0x2c:x}", "size": 1}],
        )

    def record(self, rewritten: Path, mode: str, header_write: bool = True) -> dict:
        source = self.symbols["plain"]
        offsets = {}
        with rewritten.open("rb") as handle:
            for segment in ELFFile(handle).iter_segments():
                if segment["p_type"] != "PT_LOAD":
                    continue
                for address in (source, PATCH_AT):
                    if segment["p_vaddr"] <= address < segment["p_vaddr"] + segment["p_filesz"]:
                        offsets[address] = segment["p_offset"] + address - segment["p_vaddr"]
        record = patch_record.build_record(
            self.original, rewritten, "plain",
            [patch_record.Applied(
                "plain", "target", "snippet", {},
                [(offsets[source], check_applied_bytes.arm_branch(source, PATCH_AT)),
                 (offsets[PATCH_AT], RETURN)],
            )],
            {"path": None, "version": None, "target": "arm-linux-gnueabihf", "flags": []},
            {},
            # Saving the image rewrites the ELF header, as the rewriter does.
            [(0, rewritten.read_bytes()[:52])] if header_write else [],
            target_extent=(source, 16), mode=mode,
        )
        return record

    def test_record_of_a_stripped_target(self):
        rewritten = self.rewrite("recorded", self.moved("plain"))
        record = self.record(rewritten, "arm")
        self.assertEqual(
            record["target"],
            {"name": "plain", "address": f"0x{TEXT_AT:x}", "size": 16, "mode": "arm"},
        )
        self.assertEqual(record["functions"][0]["placement"], "trampoline")
        self.assertEqual(record["functions"][0]["address"], f"0x{PATCH_AT:x}")
        with self.assertRaises(LookupError):
            patch_record.build_record(
                self.original, rewritten, "plain", [], record["compiler"], {}, []
            )

    def test_link_addresses_are_compared_with_the_bindings(self):
        record = {"functions": [
            {"name": "plain", "symbols": {"first_import": "0x20050", "helper": "0x21", "gone": None}}
        ]}
        check = patch_record.link_address_problems
        self.assertEqual(check(record, {"first_import": 0x20050, "helper": 0x20}), [])
        self.assertEqual(check(record, {"unrelated": 0x1}), [])
        self.assertEqual(
            check(record, {"first_import": 0x20060}),
            ["plain was linked with first_import at 0x20050; the binding check has it at 0x20060"],
        )

    def applied(self, rewritten: Path, mode: str) -> dict:
        record = self.record(rewritten, mode)
        self.assertEqual(
            record["functions"][0]["snippet_sha256"], hashlib.sha256(b"snippet").hexdigest()
        )
        return check_applied_bytes.check_applied_bytes(
            record, self.original, rewritten, {"plain": "snippet"}, None,
            expected={"plain": ("ok", RETURN)},
        )

    def test_arm_entry_branch_matches(self):
        report = self.applied(self.rewrite("bytes", self.moved("plain")), "arm")
        self.assertEqual(report["entry_branch"]["verdict"], "match")
        self.assertEqual(check_applied_bytes.failures(report), [])

    def test_entry_branch_is_judged_in_the_recorded_mode(self):
        report = self.applied(self.rewrite("bytes-thumb", self.moved("plain")), "thumb")
        self.assertEqual(report["entry_branch"]["verdict"], "entry-branch-mismatch")

    def test_arm_entry_branch_to_another_place(self):
        source = self.symbols["plain"]
        writes = self.moved("plain")
        writes[source] = check_applied_bytes.arm_branch(source, PATCH_AT + 8)
        rewritten = self.rewrite("bytes-elsewhere", writes)
        # The record still says what the rewriter meant to write.
        record = self.record(rewritten, "arm")
        report = check_applied_bytes.check_applied_bytes(
            record, self.original, rewritten, {"plain": "snippet"}, None,
            expected={"plain": ("ok", RETURN)},
        )
        self.assertEqual(report["entry_branch"]["verdict"], "entry-branch-mismatch")


if __name__ == "__main__":
    unittest.main()
