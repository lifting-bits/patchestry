#include <stdint.h>

#include "targets.h"

#include "uart.h"

/* Anchor a non-zero byte in .data so the linker emits a writable PT_LOAD
   segment for SRAM. Patcherex2 needs one to infer the RAM region. */
__attribute__((used, section(".data")))
static volatile uint8_t patcherex_ram_anchor = 1;

/* qemu_target_pointer is called through this stored address, never by a
   direct branch. */
static void (*const volatile pointer_targets[])(void) = {
    qemu_target_pointer,
};

int main(void) {
    uart_write_line("BOOT");
    qemu_target_before();
    qemu_target_after();
    qemu_target_replace();
    qemu_target_contract();
    qemu_target_short();
    qemu_target_tiny();
    pointer_targets[0]();
    qemu_target_dispatch();
    uart_write_line("DONE");

    return 0;
}
