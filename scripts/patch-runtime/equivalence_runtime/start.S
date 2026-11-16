@ Copyright (c) 2026, Trail of Bits, Inc.
@
@ This source code is licensed in accordance with the terms specified in
@ the LICENSE file found in the root directory of this source tree.
@
@ Entry, exception vectors and the restart point of the equivalence runtime
@ (QEMU `virt` machine, Cortex-A15, ARM state).

    .syntax unified
    .arm
    .section .vectors, "ax"
    .global _start
_start:
    b reset
    b fault             @ undefined instruction
    b fault             @ supervisor call
    b fault             @ prefetch abort
    b fault             @ data abort
    b fault
    b fault             @ interrupt
    b fault             @ fast interrupt

reset:
    ldr r0, =_start
    mrc p15, 0, r1, c1, c0, 0
    bic r1, r1, #(1 << 13)          @ vectors at VBAR, not at 0xffff0000
    mcr p15, 0, r1, c1, c0, 0
    mcr p15, 0, r0, c12, c0, 0      @ VBAR
    ldr sp, =__stack_top
    bl runtime_main
    b .

@ Any exception while a version runs: back to supervisor mode on a fresh
@ stack, and report the input as faulted.
fault:
    cps #0x13
    ldr sp, =__stack_top
    bl runtime_fault
    b .

@ Abandon the current input and go on with the next one.
    .global runtime_resume
runtime_resume:
    ldr sp, =__stack_top
    b runtime_loop
