#include <stdint.h>

#include "targets.h"

#include "uart.h"

/* Anchor a non-zero byte in .data so the linker emits a writable PT_LOAD
   segment for SRAM. Patcherex2 needs one to infer the RAM region. */
__attribute__((used, section(".data")))
static volatile uint8_t patcherex_ram_anchor = 1;

int main(void) {
    uart_write_line("BOOT");
    qemu_target_before();
    qemu_target_after();
    qemu_target_replace();
    qemu_target_contract();
    uart_write_line("DONE");

    return 0;
}
