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

void patch__replace__qemu_target_short(void) {
    uart_write_line("PATCH:short");
}

void patch__replace__qemu_target_tiny(void) {
    uart_write_line("PATCH:tiny");
}

void patch__replace__qemu_target_pointer(void) {
    uart_write_line("PATCH:pointer");
}

void patch__replace__qemu_target_handler(void) {
    uart_write_line("PATCH:handler");
}

void patch__before__qemu_target_dispatch(void) {
    uart_write_line("PATCH:dispatch");
}

/* Dependencies of inserted code.  The three patches below replace the body of
   qemu_target_replace with code that needs more than a call to the firmware:
   constants and a switch table, a compiler runtime helper, and a variable of
   its own. */

static const char *switch_arm_name(unsigned value) {
    switch (value) {
        case 0:
            return "PATCH:switch:zero";
        case 1:
            return "PATCH:switch:one";
        case 2:
            return "PATCH:switch:two";
        case 3:
            return "PATCH:switch:three";
        case 4:
            return "PATCH:switch:four";
        default:
            return "PATCH:switch:other";
    }
}

void patch__replace__switch(void) {
    for (unsigned value = 0; value < 6; ++value) {
        uart_write_line(switch_arm_name(value));
    }
}

/* 64-bit division is a call to __aeabi_uldivmod on 32-bit ARM.  The firmware
   is linked with -nostdlib and has no such function. */
void patch__replace__divide(void) {
    volatile unsigned long long numerator = 100;
    volatile unsigned long long denominator = 7;
    if (numerator / denominator == 14) {
        uart_write_line("PATCH:divide");
    }
}

/* A counter that has to live in RAM the firmware never set aside. */
static unsigned patch_call_count = 0;

void patch__replace__state(void) {
    patch_call_count = patch_call_count + 1;
    if (patch_call_count == 1) {
        uart_write_line("PATCH:state");
    }
}
