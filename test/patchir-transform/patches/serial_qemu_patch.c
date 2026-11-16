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

/* The same, with a value that has to be there before the first call: the
   variable needs a startup copy from flash, not only cleared RAM. */
static unsigned patch_retry_limit = 3;

void patch__replace__initialized(void) {
    if (patch_retry_limit == 3) {
        uart_write_line("PATCH:initialized");
    }
    patch_retry_limit = patch_retry_limit - 1;
}

/* Validation corpus.  Each patch below only restricts the inputs its target
   accepts, and each rejects in a different way. */

extern volatile unsigned qemu_length_last;
extern void forward_second(unsigned value);
extern void qemu_reject(void) __attribute__((noreturn));

/* Reject by early return: the replacement body returns the error value for a
   length over the limit and is the original body otherwise. */
unsigned patch__definition__qemu_target_length(const unsigned char *buf, unsigned len) {
    if (len > 16) {
        return 0xffffffffu;
    }
    qemu_length_last = len;
    return buf[0] + len;
}

/* Reject by skipping the operation: the call is made only for a value within
   the limit. */
void patch__replace__forward_second(unsigned value) {
    if (value <= 32) {
        forward_second(value);
    }
}

/* Reject by calling a handler that does not return. */
void patch__entry__qemu_target_send(unsigned byte) {
    if (byte >= 'a') {
        qemu_reject();
    }
}
