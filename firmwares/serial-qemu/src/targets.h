#pragma once

void qemu_target_before(void);
void qemu_target_after(void);
void qemu_target_replace(void);
void qemu_target_contract(void);
void qemu_target_short(void);
void qemu_target_tiny(void);
void qemu_target_pointer(void);
void qemu_target_handler(void);
void qemu_target_dispatch(void);

extern void (*const volatile qemu_callback)(void);

#define QEMU_LENGTH_ERROR 0xffffffffu

extern volatile unsigned qemu_length_last;

unsigned qemu_target_length(const unsigned char *buf, unsigned len);
void forward_first(unsigned value);
void forward_second(unsigned value);
void qemu_target_forward(unsigned value);
void qemu_irq_disable(void);
void qemu_irq_enable(void);
void qemu_target_send(unsigned byte);
void qemu_reject(void) __attribute__((noreturn));
