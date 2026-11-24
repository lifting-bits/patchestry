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

static const unsigned char length_input[] = { 3 };

/* The callers of the validation targets are never patched; they report what
   they observe at the function boundary. */
static void report_length(unsigned len) {
    unsigned result = qemu_target_length(length_input, len);
    if (result == QEMU_LENGTH_ERROR) {
        uart_write_line("LEN:error");
    } else if (result == length_input[0] + len && qemu_length_last == len) {
        uart_write_line("LEN:ok");
    } else {
        uart_write_line("LEN:bad");
    }
}

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
    report_length(4);
    report_length(200);
    qemu_target_forward(5);
    qemu_target_forward(100);
    qemu_target_send('S');
    qemu_target_send('z');
    uart_write_line("DONE");

    return 0;
}
