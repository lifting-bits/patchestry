// RUN: true
typedef unsigned char  uint8_t;
typedef unsigned int   uint32_t;

// Underlying UART read primitive — extern from src/uart.c, used by the
// existing HardwareSerial methods. Returns -1 when no byte is buffered,
// otherwise 0..255. Unlike the secpump patches this TU is intentionally
// not extern-free: a working `available()` cannot exist without the UART
// read, and Patcherex2 already resolves uart_getc_nonblock in the firmware.
int uart_getc_nonblock(void);

// HardwareSerial::available() in firmwares/makair-qemu/src/arduino_shim.cpp
// had a stash-clobber bug: it called uart_getc_nonblock() unconditionally
// and overwrote `m_uart_id`'s low 9 bits, so a second call before read()
// silently dropped the previously buffered byte and the function never
// returned more than 1. Both the in-tree fast path
// (src/main.cpp::serial_control_loop_v0) and the upstream parser
// (firmwares/repos/makair-firmware/srcs/serial_control.cpp:66) gate on
// `Serial6.available() >= 11`, so neither path could ever assemble a
// Control frame — Heartbeat / VentilationMode / PEEP / etc. were silently
// consumed and discarded.
//
// peek() in the same file already implements the correct pattern: check
// the stash bit (m_uart_id & 0x100) before pulling a new byte. This
// replacement mirrors that guard.
//
// HardwareSerial declares a single uint32_t member (m_uart_id) and no
// virtual methods, so the C++ this-pointer is ABI-compatible with a
// `uint32_t *` and the offset to m_uart_id is 0. ARM AAPCS thiscall passes
// `this` in r0 and returns in r0 — identical to a free C function with
// the same signature, so the lifted/lowered patch slots in cleanly.
int patch__replace__HardwareSerial__available(uint32_t *self) {
    if (*self & 0x100u) {
        return 1;                    // byte already buffered — don't drain UART
    }
    int b = uart_getc_nonblock();
    if (b < 0) {
        return 0;
    }
    *self = (*self & ~0x1FFu) | 0x100u | (uint32_t)((unsigned)b & 0xFFu);
    return 1;
}
