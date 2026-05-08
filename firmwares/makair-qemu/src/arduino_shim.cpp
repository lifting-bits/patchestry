// arduino_shim.cpp — implements just enough of the Arduino API for
// upstream MakAir telemetry.cpp and friends to compile and run on QEMU
// mps2-an386. Real GPIO/ADC operations are no-ops; time is SysTick-derived;
// Serial / Serial6 both route through src/uart.c so the host harness sees
// every byte the firmware would have emitted on its UART pin.

#include "Arduino.h"

extern "C" {
    void uart_init(void);
    void uart_putc(char c);
    int  uart_getc_nonblock(void);
    int  uart_getc(void);
}

// SysTick is configured by src/startup.c at boot to tick every millisecond.
extern "C" volatile uint32_t systick_ms;

// ---- C API ---------------------------------------------------------------
extern "C" {

void pinMode(uint32_t /*pin*/, uint32_t /*mode*/) { /* no-op */ }
void digitalWrite(uint32_t /*pin*/, uint32_t /*val*/) { /* no-op */ }
int  digitalRead(uint32_t /*pin*/) { return 0; }
int  analogRead(uint32_t /*pin*/) { return 0; }
void analogWrite(uint32_t /*pin*/, int /*val*/) { /* no-op */ }

uint32_t millis(void)  { return systick_ms; }
uint32_t micros(void)  { return systick_ms * 1000u; }

void delay(uint32_t ms) {
    uint32_t start = systick_ms;
    while ((systick_ms - start) < ms) { /* spin */ }
}
void delayMicroseconds(uint32_t us) {
    // mps2-an386 cycle timing isn't precise enough to honour microseconds;
    // round up to the nearest millisecond. Telemetry/control loops only need
    // coarse delays.
    if (us == 0) return;
    delay((us + 999u) / 1000u);
}

// STM32 LL UID stubs — return a deterministic 96-bit "device id". Real STM32
// pulls this from 0x1FFF7A10; we just synthesise a value that round-trips
// through telemetry.cpp::computeDeviceId without referencing any STM32 reg.
uint32_t LL_GetUID_Word0(void) { return 0x4D414B41u; /* "MAKA" */ }
uint32_t LL_GetUID_Word1(void) { return 0x49524F45u; /* "IROE" */ }
uint32_t LL_GetUID_Word2(void) { return 0x6D752D31u; /* "mu-1" */ }

}  // extern "C"

// ---- HardwareSerial (C++ class) -----------------------------------------
void HardwareSerial::begin(unsigned long /*baud*/) { uart_init(); }
void HardwareSerial::end(void)                     { /* no-op */ }
int  HardwareSerial::available(void)               {
    int b = uart_getc_nonblock();
    if (b < 0) return 0;
    // Stash the read byte for the next read()/peek().
    m_uart_id = (m_uart_id & ~0x1FFu) | (uint32_t)(0x100u | (b & 0xFF));
    return 1;
}
int HardwareSerial::peek(void) {
    if (!(m_uart_id & 0x100u)) {
        int b = uart_getc_nonblock();
        if (b < 0) return -1;
        m_uart_id = (m_uart_id & ~0x1FFu) | 0x100u | (uint32_t)(b & 0xFF);
    }
    return (int)(m_uart_id & 0xFFu);
}
int HardwareSerial::read(void) {
    if (m_uart_id & 0x100u) {
        int b = (int)(m_uart_id & 0xFFu);
        m_uart_id &= ~0x1FFu;
        return b;
    }
    return uart_getc_nonblock();
}
size_t HardwareSerial::readBytes(uint8_t *buf, size_t len) {
    size_t n = 0;
    while (n < len) {
        int b = read();
        if (b < 0) break;
        buf[n++] = (uint8_t)b;
    }
    return n;
}
size_t HardwareSerial::readBytes(char *buf, size_t len) {
    return readBytes((uint8_t *)buf, len);
}
size_t HardwareSerial::write(uint8_t b) {
    uart_putc((char)b);
    return 1;
}
size_t HardwareSerial::write(const uint8_t *buf, size_t len) {
    for (size_t i = 0; i < len; ++i) uart_putc((char)buf[i]);
    return len;
}

static size_t print_signed(HardwareSerial *s, long v) {
    char buf[16];
    int n = snprintf(buf, sizeof(buf), "%ld", v);
    return s->write((const uint8_t *)buf, (size_t)n);
}
static size_t print_unsigned(HardwareSerial *s, unsigned long v) {
    char buf[16];
    int n = snprintf(buf, sizeof(buf), "%lu", v);
    return s->write((const uint8_t *)buf, (size_t)n);
}
size_t HardwareSerial::print(int v)            { return print_signed(this, v); }
size_t HardwareSerial::print(unsigned int v)   { return print_unsigned(this, v); }
size_t HardwareSerial::print(long v)           { return print_signed(this, v); }
size_t HardwareSerial::print(unsigned long v)  { return print_unsigned(this, v); }

// Default debug Serial and the telemetry Serial6 — both share UART0 in QEMU.
HardwareSerial Serial;
HardwareSerial Serial6;
