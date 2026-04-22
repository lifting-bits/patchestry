// RUN: true

#define USE_C99_TYPES
#include "patchestry/intrinsics/patchestry_intrinsics.h"

extern void uart_write_line(const char *text);
extern void qemu_target_replace(void);
extern void qemu_target_before(void);
extern void qemu_target_after(void);

void patch__before__qemu_target_before(void) {
    uart_write_line("PATCH:before");
}

void patch__after__qemu_target_after(void) {
    uart_write_line("PATCH:after");
}

void patch__replace__qemu_target_replace(void) {
    uart_write_line("PATCH:replace");
}

void contract__entrypoint__qemu_target_contract(void) {
    uart_write_line("CONTRACT:entry");
}
