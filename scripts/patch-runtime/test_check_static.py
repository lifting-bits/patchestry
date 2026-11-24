#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests for check_static.py on small hand-written ARM and Thumb functions.

Needs clang and ld.lld from PATCHESTRY_LLVM_PREFIX.
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

import check_static  # noqa: E402
import check_trampoline  # noqa: E402

SOURCE = r"""
    .syntax unified
    .text

    .macro arm_function name
    .arm
    .balign 4
    .global \name
    .type \name, %function
\name:
    .endm

    .macro thumb_function name
    .thumb
    .balign 4
    .global \name
    .type \name, %function
    .thumb_func
\name:
    .endm

    .macro end name
    .size \name, . - \name
    .endm

    arm_function callee
    bx lr
    end callee

    @ ---- ARM mode ----

    arm_function good
    push {r4, r11, lr}
    add r11, sp, #4
    sub sp, sp, #16
    mov r4, r0
    bl callee
    mov r0, r4
    sub sp, r11, #4
    pop {r4, r11, pc}
    end good

    arm_function leaf
    add r0, r0, #1
    bx lr
    end leaf

    arm_function drops_register
    push {r11, lr}
    mov r4, r0
    bl callee
    mov r0, r4
    pop {r11, pc}
    end drops_register

    arm_function unbalanced
    push {r4, r11, lr}
    sub sp, sp, #8
    mov r4, r0
    pop {r4, r11, pc}
    end unbalanced

    arm_function wrong_order
    push {r4, r5, lr}
    mov r4, r0
    mov r5, r1
    pop {r5}
    pop {r4}
    pop {pc}
    end wrong_order

    arm_function conditional_return
    cmp r0, #0
    bxeq lr
    push {r4, lr}
    mov r4, r0
    bl callee
    pop {r4, pc}
    end conditional_return

    arm_function two_exits
    push {r4, r5, r11, lr}
    cmp r0, #0
    beq 1f
    mov r4, r0
    pop {r4, r5, r11, pc}
1:  mov r5, r0
    pop {r4, r5, r11, lr}
    bx lr
    end two_exits

    arm_function tail_call
    push {r4, lr}
    mov r4, #1
    pop {r4, lr}
    b callee
    end tail_call

    arm_function tail_call_clobbers
    mov r6, #1
    b callee
    end tail_call_clobbers

    arm_function single_slot
    str lr, [sp, #-4]!
    bl callee
    ldr pc, [sp], #4
    end single_slot

    arm_function multiple_load_and_store
    .inst 0xe92d4800         @ stmdb sp!, {r11, lr}
    mov r11, #0
    bl callee
    .inst 0xe8bd0800         @ ldm sp!, {r11}
    .inst 0xe8bd8000         @ ldm sp!, {pc}
    end multiple_load_and_store

    arm_function looped
    push {r4, lr}
    mov r4, #8
1:  bl callee
    subs r4, r4, #1
    bne 1b
    pop {r4, pc}
    end looped

    arm_function literal
    ldr r0, =0xe92d4010      @ the encoding of push {r4, lr}
    bx lr
    .ltorg
    end literal

    arm_function jump_table
    add pc, pc, r0, lsl #2
    nop
    bx lr
    bx lr
    end jump_table

    arm_function dynamic_frame
    push {r11, lr}
    mov r11, sp
    sub sp, sp, r0
    mov sp, r11
    pop {r11, pc}
    end dynamic_frame

    arm_function loses_return
    push {r4, lr}
    mov lr, #0
    pop {r4}
    add sp, sp, #4
    bx lr
    end loses_return

    @ A body with enough distinct instructions to be recognized elsewhere.
    arm_function distinctive
    push {r4, r5, lr}
    mov r4, r0
    eor r5, r4, r4, lsl #3
    add r5, r5, #0x35
    orr r4, r5, r4, lsr #7
    rsb r4, r4, #0x1100
    bic r5, r4, #0xf0
    mul r4, r5, r4
    eor r4, r4, #0x5a
    add r4, r4, r5, lsl #2
    sub r5, r5, r4, asr #1
    orr r0, r4, r5
    mvn r0, r0
    add r0, r0, #9
    pop {r4, r5, pc}
    end distinctive

    arm_function has_inlined_copy
    push {r4, r5, r6, lr}
    mov r6, r1
    mov r4, r0
    eor r5, r4, r4, lsl #3
    add r5, r5, #0x35
    orr r4, r5, r4, lsr #7
    rsb r4, r4, #0x1100
    bic r5, r4, #0xf0
    mul r4, r5, r4
    eor r4, r4, #0x5a
    add r4, r4, r5, lsl #2
    sub r5, r5, r4, asr #1
    orr r0, r4, r5
    mvn r0, r0
    add r0, r0, #9
    add r0, r0, r6
    pop {r4, r5, r6, pc}
    end has_inlined_copy

    @ ---- Thumb ----

    thumb_function t_good
    push {r4, r7, lr}
    add r7, sp, #4
    sub sp, #8
    mov r4, r0
    bl callee
    cbz r0, 1f
    mov r0, r4
1:  add sp, #8
    pop {r4, r7, pc}
    end t_good

    thumb_function t_drops_register
    push {r7, lr}
    mov r5, r0
    bl callee
    pop {r7, pc}
    end t_drops_register

    thumb_function t_frame_pointer
    push {r7, lr}
    mov r7, sp
    sub sp, #24
    mov sp, r7
    pop {r7, pc}
    end t_frame_pointer

    thumb_function t_conditional
    push {r4, lr}
    cmp r0, #0
    it ne
    movne r4, r0
    pop {r4, pc}
    end t_conditional

    thumb_function t_it_return
    push {r4, lr}
    cmp r0, #0
    it eq
    popeq {r4, pc}
    mov r4, r0
    pop {r4, pc}
    end t_it_return

    thumb_function t_it_return_then_clobber
    push {r4, lr}
    cmp r0, #0
    it eq
    popeq {r4, pc}
    mov r4, r0
    pop {r3, pc}
    end t_it_return_then_clobber

    thumb_function t_it_else
    push {r4, lr}
    cmp r0, #0
    ite eq
    moveq r0, #1
    movne r5, r0
    pop {r4, pc}
    end t_it_else

    thumb_function t_table
    tbb [pc, r0]
    .byte 1, 2
    bx lr
    bx lr
    end t_table

    @ ---- Floating-point registers that overlap d8 to d15 ----
    .fpu neon

    arm_function fp_half
    vadd.f32 s16, s0, s0
    bx lr
    end fp_half

    arm_function fp_upper_half
    vadd.f32 s17, s0, s0
    bx lr
    end fp_upper_half

    arm_function fp_quad
    vadd.i32 q4, q0, q0
    bx lr
    end fp_quad

    arm_function fp_caller_saved
    vadd.f32 s15, s0, s0
    vadd.i32 q3, q0, q0
    bx lr
    end fp_caller_saved

    arm_function fp_saved
    vpush {d8}
    vadd.f32 s16, s0, s0
    vpop {d8}
    bx lr
    end fp_saved

    arm_function fp_saved_as_singles
    vpush {s16, s17}
    vadd.f32 s17, s0, s0
    vpop {s16, s17}
    bx lr
    end fp_saved_as_singles

    arm_function fp_restores_other_half
    vpush {s0}
    vpop {s16}
    bx lr
    end fp_restores_other_half

    arm_function fp_saved_half
    vpush {s16}
    vadd.f32 s16, s0, s0
    vpop {s16}
    bx lr
    end fp_saved_half

    @ ---- Recursion ----

    arm_function recursive
    push {r4, lr}
    cmp r0, #0
    beq 1f
    sub r0, r0, #1
    bl recursive
1:  pop {r4, pc}
    end recursive

    arm_function mutual_a
    push {r4, lr}
    cmp r0, #0
    beq 1f
    bl mutual_b
1:  pop {r4, pc}
    end mutual_a

    arm_function mutual_b
    push {r4, lr}
    sub r0, r0, #1
    bl mutual_a
    pop {r4, pc}
    end mutual_b

    arm_function calls_leaf
    push {r4, lr}
    bl callee
    pop {r4, pc}
    end calls_leaf
"""

LINKER_SCRIPT = """
ENTRY(good)
SECTIONS
{
  .text 0x10000 : { *(.text*) }
  /DISCARD/ : { *(.ARM.exidx*) *(.ARM.attributes) }
}
"""


class StaticChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix = os.environ.get("PATCHESTRY_LLVM_PREFIX")
        if not prefix:
            raise unittest.SkipTest("set PATCHESTRY_LLVM_PREFIX to the LLVM bin directory")
        clang = str(Path(prefix) / "clang")
        lld = os.environ.get("PATCHESTRY_LD_LLD") or str(Path(prefix) / "ld.lld")
        cls.directory = Path(tempfile.mkdtemp(prefix="check-static-"))
        (cls.directory / "code.S").write_text(SOURCE, encoding="utf-8")
        (cls.directory / "link.ld").write_text(LINKER_SCRIPT, encoding="utf-8")
        subprocess.run([clang, "--target=armv7-none-eabi", "-c", "code.S", "-o", "code.o"],
                       cwd=cls.directory, check=True)
        subprocess.run([lld, "-T", "link.ld", "code.o", "-o", "code.elf"],
                       cwd=cls.directory, check=True)
        cls.image = cls.directory / "code.elf"
        cls.memory = check_trampoline.Memory(cls.image)
        cls.functions = {}
        with cls.image.open("rb") as handle:
            for symbol in ELFFile(handle).get_section_by_name(".symtab").iter_symbols():
                if symbol["st_info"]["type"] == "STT_FUNC":
                    value = int(symbol["st_value"])
                    cls.functions[symbol.name] = {
                        "name": symbol.name,
                        "address": value & ~1,
                        "size": int(symbol["st_size"]),
                        "mode": "thumb" if value & 1 else "arm",
                    }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def analyze(self, name: str) -> dict:
        function = self.functions[name]
        return check_static.analyze_function(
            self.memory, function["address"], function["size"], function["mode"]
        )

    def assert_passes(self, name: str, depth: int):
        result = self.analyze(name)
        self.assertEqual((result["problems"], result["not_run"]), ({}, {}), name)
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["stack_depth"], depth)
        return result

    # Functions that keep the calling convention.

    def test_frame_with_saved_registers(self):
        result = self.assert_passes("good", 28)
        self.assertEqual(result["instructions"], 8)
        self.assertEqual([row["kind"] for row in result["exits"]], ["return"])

    def test_leaf(self):
        self.assert_passes("leaf", 0)

    def test_conditional_return(self):
        result = self.assert_passes("conditional_return", 8)
        self.assertEqual(len(result["exits"]), 2)

    def test_two_exits_restoring_differently(self):
        self.assert_passes("two_exits", 16)

    def test_tail_call(self):
        result = self.assert_passes("tail_call", 8)
        self.assertEqual([row["kind"] for row in result["exits"]], ["tail-call"])

    def test_single_register_push_and_pop(self):
        self.assert_passes("single_slot", 4)

    def test_push_and_pop_written_as_multiple_load_and_store(self):
        self.assert_passes("multiple_load_and_store", 8)

    def test_loop(self):
        self.assert_passes("looped", 8)

    def test_literal_pool_is_not_code(self):
        result = self.assert_passes("literal", 0)
        self.assertEqual(result["instructions"], 2)

    def test_thumb_frame(self):
        self.assert_passes("t_good", 20)

    def test_thumb_frame_pointer_restores_the_stack(self):
        self.assert_passes("t_frame_pointer", 32)

    def test_thumb_conditional_write_to_a_saved_register(self):
        self.assert_passes("t_conditional", 8)

    # Functions that break it.

    def test_register_used_without_being_saved(self):
        result = self.analyze("drops_register")
        self.assertEqual(result["verdict"], "fail")
        rows = result["problems"]["callee-saved-clobbered"]
        self.assertEqual([row["registers"] for row in rows], [["r4"]])

    def test_thumb_register_used_without_being_saved(self):
        rows = self.analyze("t_drops_register")["problems"]["callee-saved-clobbered"]
        self.assertEqual([row["registers"] for row in rows], [["r5"]])

    def test_return_with_the_frame_still_allocated(self):
        result = self.analyze("unbalanced")
        self.assertEqual(result["problems"]["unbalanced-stack"][0]["depth"], 8)
        self.assertEqual(result["verdict"], "fail")

    def test_registers_restored_from_each_others_slots(self):
        rows = self.analyze("wrong_order")["problems"]["callee-saved-clobbered"]
        self.assertEqual([row["registers"] for row in rows], [["r4", "r5"]])

    def test_tail_call_with_a_clobbered_register(self):
        rows = self.analyze("tail_call_clobbers")["problems"]["callee-saved-clobbered"]
        self.assertEqual([row["registers"] for row in rows], [["r6"]])

    def test_return_to_an_overwritten_address(self):
        self.assertIn("return-address-lost", self.analyze("loses_return")["problems"])

    # Functions the analysis cannot follow.

    def test_computed_jump_is_not_run(self):
        result = self.analyze("jump_table")
        self.assertEqual(result["verdict"], "not-run")
        self.assertIn("indirect-jump", result["not_run"])
        self.assertIsNone(result["stack_depth"])

    def test_thumb_table_branch_is_not_run(self):
        self.assertIn("indirect-jump", self.analyze("t_table")["not_run"])

    def test_variable_sized_frame_is_not_run(self):
        result = self.analyze("dynamic_frame")
        self.assertEqual(result["verdict"], "not-run")
        self.assertIn("untracked-stack-pointer", result["not_run"])

    # Copies of a body elsewhere in the image.

    def copies(self, name: str) -> list[dict]:
        function = self.functions[name]
        return check_static.inlined_copies(
            self.image, function["address"], function["size"], function["mode"]
        )

    def test_inlined_copy_is_found(self):
        copies = self.copies("distinctive")
        self.assertEqual(len(copies), 1)
        host = self.functions["has_inlined_copy"]
        self.assertEqual(copies[0]["address"], f"0x{host['address'] + 8:x}")
        self.assertEqual(copies[0]["instructions"], 13)

    def test_short_run_of_a_long_body_is_not_a_copy(self):
        # 13 instructions repeat, which is less than the share asked for.
        function = self.functions["distinctive"]
        self.assertEqual(
            check_static.inlined_copies(
                self.image, function["address"], function["size"], "arm", share=0.95
            ),
            [],
        )

    def test_no_copy_of_a_body_that_occurs_once(self):
        self.assertEqual(self.copies("good"), [])
        self.assertEqual(self.copies("looped"), [])

    def test_copy_is_found_from_either_side(self):
        copies = self.copies("has_inlined_copy")
        self.assertEqual(len(copies), 1)
        self.assertEqual(
            copies[0]["address"], f"0x{self.functions['distinctive']['address'] + 4:x}"
        )

    # The report for a rewritten image.

    def report(self, names, target="good", budgets=None) -> dict:
        return check_static.check_static(
            self.image, self.image, [self.functions[name] for name in names],
            self.functions[target], budgets,
        )

    def test_report_without_budgets(self):
        report = self.report(["good", "leaf"])
        self.assertEqual(check_static.failures(report), [])
        self.assertEqual(report["code_bytes"], 32 + 8)
        self.assertEqual(report["stack_bytes"], 28)
        self.assertEqual(report["budgets"]["code_bytes"]["verdict"], "not-judged")
        self.assertEqual(report["budgets"]["stack_bytes"]["verdict"], "not-judged")
        self.assertEqual(report["original"]["stack_depth"], 28)

    def test_within_budget(self):
        report = self.report(["good"], budgets={"code_bytes": 32, "stack_bytes": 28})
        self.assertEqual(check_static.failures(report), [])
        self.assertEqual(report["budgets"]["code_bytes"]["verdict"], "pass")

    def test_code_size_over_budget(self):
        report = self.report(["good", "leaf"], budgets={"code_bytes": 36})
        self.assertEqual(check_static.failures(report), ["code-size-over-budget"])

    def test_stack_over_budget(self):
        report = self.report(["good"], budgets={"stack_bytes": 24})
        self.assertEqual(check_static.failures(report), ["stack-over-budget"])

    def test_stack_budget_needs_a_known_depth(self):
        report = self.report(["jump_table"], budgets={"stack_bytes": 64})
        self.assertEqual(report["budgets"]["stack_bytes"]["verdict"], "not-run")
        self.assertEqual(check_static.failures(report), [])
        self.assertEqual(check_static.not_run(report), ["indirect-jump"])

    # Registers that overlap a callee-saved floating-point register.

    def clobbered(self, name: str) -> list[str]:
        result = self.analyze(name)
        self.assertEqual(list(result["problems"]), ["callee-saved-clobbered"], name)
        return result["problems"]["callee-saved-clobbered"][0]["registers"]

    def test_single_precision_half_of_a_saved_register(self):
        self.assertEqual(self.clobbered("fp_half"), ["d8"])
        self.assertEqual(self.clobbered("fp_upper_half"), ["d8"])

    def test_quad_register_over_two_saved_registers(self):
        self.assertEqual(self.clobbered("fp_quad"), ["d8", "d9"])

    def test_caller_saved_floating_point_registers(self):
        self.assert_passes("fp_caller_saved", 0)

    def test_saved_register_written_through_its_half(self):
        self.assert_passes("fp_saved", 8)

    def test_register_saved_as_its_two_halves(self):
        self.assert_passes("fp_saved_as_singles", 8)

    def test_saving_one_half_is_not_saving_the_register(self):
        result = self.analyze("fp_saved_half")
        self.assertEqual(result["stack_depth"], 4)
        self.assertNotIn("unbalanced-stack", result["problems"])
        self.assertEqual(result["problems"]["callee-saved-clobbered"][0]["registers"], ["d8"])

    def test_pop_into_half_of_a_saved_register(self):
        self.assertEqual(self.clobbered("fp_restores_other_half"), ["d8"])

    # Thumb IT blocks.

    def test_conditional_return_in_an_it_block(self):
        result = self.assert_passes("t_it_return", 8)
        self.assertEqual(len(result["exits"]), 2)

    def test_path_after_a_conditional_return_in_an_it_block(self):
        result = self.analyze("t_it_return_then_clobber")
        self.assertEqual(len(result["exits"]), 2)
        self.assertEqual(
            [row["registers"] for row in result["problems"]["callee-saved-clobbered"]], [["r4"]])

    def test_every_instruction_of_an_it_block_is_followed(self):
        self.assertEqual(self.clobbered("t_it_else"), ["r5"])

    def test_whole_registers(self):
        whole = check_static._whole_registers
        self.assertEqual(whole("s16"), {"d8"})
        self.assertEqual(whole("s31"), {"d15"})
        self.assertEqual(whole("q4"), {"d8", "d9"})
        self.assertEqual(whole("q7"), {"d14", "d15"})
        self.assertEqual(whole("d8"), {"d8"})
        self.assertEqual(whole("sp"), {"sp"})
        self.assertEqual(whole("r4"), {"r4"})

    # Recursion and the stack budget.

    def test_recursive_function_has_no_stack_bound(self):
        self.assert_passes("recursive", 8)  # one activation keeps the convention
        report = self.report(["recursive"], budgets={"stack_bytes": 8})
        self.assertEqual(report["recursion"], ["recursive"])
        self.assertIsNone(report["stack_bytes"])
        self.assertEqual(report["budgets"]["stack_bytes"]["verdict"], "not-run")
        self.assertEqual(check_static.failures(report), [])
        self.assertEqual(check_static.not_run(report), ["recursive-call"])

    def test_mutual_recursion(self):
        report = self.report(["mutual_a", "mutual_b", "leaf"], budgets={"stack_bytes": 64})
        self.assertEqual(report["recursion"], ["mutual_a", "mutual_b"])
        self.assertEqual(report["budgets"]["stack_bytes"]["verdict"], "not-run")

    def test_call_out_of_the_inserted_code_is_not_recursion(self):
        report = self.report(["calls_leaf", "leaf"], budgets={"stack_bytes": 8})
        self.assertEqual(report["recursion"], [])
        self.assertEqual(report["stack_bytes"], 8)
        self.assertEqual(report["budgets"]["stack_bytes"]["verdict"], "pass")

    def test_call_of_the_replaced_address_is_a_call_of_the_replacement(self):
        # The replaced function is `mutual_a`.  Its replacement is the code
        # of `mutual_b`, which calls the address of `mutual_a`: the entry
        # there now branches to the replacement, so the call comes back.
        original = dict(self.functions["mutual_a"], name="target")
        replacement = dict(self.functions["mutual_b"], name="target")
        report = check_static.check_static(
            self.image, self.image, [replacement], original, {"stack_bytes": 64})
        self.assertEqual(report["recursion"], ["target"])
        self.assertEqual(report["budgets"]["stack_bytes"]["verdict"], "not-run")
        # The same code is not recursive when it replaces something else.
        other = dict(self.functions["leaf"], name="target")
        report = check_static.check_static(
            self.image, self.image, [replacement], other, {"stack_bytes": 64})
        self.assertEqual(report["recursion"], [])

    def test_failing_function_fails_the_report(self):
        report = self.report(["good", "drops_register"])
        self.assertEqual(check_static.failures(report), ["callee-saved-clobbered"])

    def test_copies_of_the_target_are_a_warning(self):
        report = self.report(["good"], target="distinctive")
        self.assertEqual(len(report["inlined_copies"]), 1)
        self.assertEqual(check_static.failures(report), [])


if __name__ == "__main__":
    unittest.main()
