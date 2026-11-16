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
