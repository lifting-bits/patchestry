#include "targets.h"

#include "uart.h"

static void emit_before_base(void) {
    uart_write_line("BASE:before");
}

static void emit_after_base(void) {
    uart_write_line("BASE:after");
}

static void emit_replace_base(void) {
    uart_write_line("BASE:replace");
}

static void emit_contract_base(void) {
    uart_write_line("BASE:contract");
}

void qemu_target_before(void) {
    emit_before_base();
}

void qemu_target_after(void) {
    emit_after_base();
}

void qemu_target_replace(void) {
    emit_replace_base();
}

void qemu_target_contract(void) {
    emit_contract_base();
}
