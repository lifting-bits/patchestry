/* MPS2-AN386 (Cortex-M4) memory map for QEMU `mps2-an386`.
 * SSRAM1  0x00000000  4M  code (treated as flash)
 * SSRAM23 0x20000000  4M  data (RAM)
 */
ENTRY(Reset_Handler)

MEMORY {
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS {
    .isr_vector : {
        KEEP(*(.isr_vector))
    } > FLASH

    .text : {
        *(.text*)
        *(.rodata*)
        *(.glue_7) *(.glue_7t)
        *(.eh_frame)
        KEEP(*(.init)) KEEP(*(.fini))
        . = ALIGN(4);
        _etext = .;
    } > FLASH

    .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > FLASH
    .ARM : {
        __exidx_start = .;
        *(.ARM.exidx*)
        __exidx_end = .;
    } > FLASH

    _sidata = LOADADDR(.data);
    .data : {
        . = ALIGN(4);
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    .bss : {
        . = ALIGN(4);
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    /* Heap follows BSS, grows toward stack. */
    .heap (NOLOAD) : {
        . = ALIGN(8);
        _end = .;
        end = .;
        _heap_start = .;
    } > RAM
}
