#include "targets.h"

#include "uart.h"

static void emit_base_prefix(const char *name) {
    uart_write_string("BASE:");
    uart_write_line(name);
}

void qemu_target_before(void) {
    emit_base_prefix("before");
}

void qemu_target_after(void) {
    emit_base_prefix("after");
}

void qemu_target_replace(void) {
    emit_base_prefix("replace");
}

void qemu_target_contract(void) {
    emit_base_prefix("contract");
}
