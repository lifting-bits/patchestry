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

/* The targets below exist for their size and for how they are reached.
   Reinsertion puts a 4-byte branch at the entry of a function it has to
   move. */

__attribute__((used))
static void emit_short_base(void) {
    uart_write_line("BASE:short");
}

/* 4 bytes: one wide branch to a callee in another section. The entry
   branch fits exactly. */
__attribute__((naked))
void qemu_target_short(void) {
    __asm__("b emit_short_base");
}

/* 2 bytes: one narrow branch to the function placed directly after it in
   the same section. An entry branch does not fit and would overwrite the
   start of emit_tiny_base. */
#define TINY_SECTION __attribute__((section(".text.qemu_target_tiny")))

static void emit_tiny_base(void);

TINY_SECTION __attribute__((naked))
void qemu_target_tiny(void) {
    __asm__("b.n emit_tiny_base");
}

/* aligned(2) drops the padding clang would put between the two. */
TINY_SECTION __attribute__((used, aligned(2)))
static void emit_tiny_base(void) {
    uart_write_line("BASE:tiny");
}

static void emit_pointer_base(void) {
    uart_write_line("BASE:pointer");
}

/* Only reached through the address stored in main.c's table. */
void qemu_target_pointer(void) {
    emit_pointer_base();
}
