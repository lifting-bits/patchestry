/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 *
 * Memory layout of the equivalence runtime on QEMU's `virt` machine (RAM at
 * 0x40000000).  The harness's writable data is kept apart from the runtime's,
 * so the runtime can put it back before every input.
 */
ENTRY(_start)
SECTIONS
{
    . = 0x40010000;
    .text : { KEEP(*(.vectors)) *(.text .text.*) }
    .rodata : { *(.rodata .rodata.*) }
    . = ALIGN(16);
    __harness_data_start = .;
    .harness_data : { *harness.o(.data .data.*) }
    .harness_bss : { *harness.o(.bss .bss.* COMMON) }
    . = ALIGN(16);
    __harness_data_end = .;
    __harness_snapshot = .;
    . += __harness_data_end - __harness_data_start;
    .data : { *(.data .data.*) }
    .bss : { *(.bss .bss.* COMMON) }
    . = ALIGN(16);
    __heap_start = .;
    . += 0x100000;
    __heap_end = .;
    . += 0x100000;
    __stack_top = .;
}
