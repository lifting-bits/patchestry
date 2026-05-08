// Arduino.h — minimal QEMU shim covering only the API surface the MakAir
// protocol layer (telemetry.cpp + serial_control.cpp + rpi_watchdog.cpp +
// alarm.cpp + cycle.cpp) actually pulls in. Anything wider is left out on
// purpose so the link picks up the matching shim from src/arduino_shim.cpp
// rather than dragging in the real Arduino-STM32 framework.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>      // upstream main_controller.cpp uses sqrt()
#ifdef __cplusplus
#include <algorithm>   // upstream alarm_controller.cpp uses std::sort
using std::min;        // main_controller.cpp uses unqualified min/max
using std::max;
extern "C" {
#endif

typedef uint8_t  byte;
typedef bool     boolean;
typedef uint8_t  word;

// -- pin / digital / analog ------------------------------------------------
#define INPUT          0
#define OUTPUT         1
#define INPUT_PULLUP   2
#define LOW            0
#define HIGH           1
#define LSBFIRST       0
#define MSBFIRST       1

void pinMode(uint32_t pin, uint32_t mode);
void digitalWrite(uint32_t pin, uint32_t val);
int  digitalRead(uint32_t pin);
int  analogRead(uint32_t pin);
void analogWrite(uint32_t pin, int val);

// -- time ------------------------------------------------------------------
uint32_t millis(void);
uint32_t micros(void);
void     delay(uint32_t ms);
void     delayMicroseconds(uint32_t us);

// STM32 Nucleo F411RE pin-name macros referenced by includes/parameters.h.
// Arduino-STM32 normally defines them via PinNamesTypes.h; we substitute
// dense small integers so anything that uses them as `digitalWrite(PIN_X,...)`
// arguments compiles cleanly. The shimmed digitalWrite/digitalRead/analogRead
// in arduino_shim.cpp ignore the value.
#define PA0  0x00
#define PA1  0x01
#define PA2  0x02
#define PA3  0x03
#define PA4  0x04
#define PA5  0x05
#define PA6  0x06
#define PA7  0x07
#define PA8  0x08
#define PA9  0x09
#define PA10 0x0A
#define PA11 0x0B
#define PA12 0x0C
#define PA13 0x0D
#define PA14 0x0E
#define PA15 0x0F
#define PB0  0x10
#define PB1  0x11
#define PB2  0x12
#define PB3  0x13
#define PB4  0x14
#define PB5  0x15
#define PB6  0x16
#define PB7  0x17
#define PB8  0x18
#define PB9  0x19
#define PB10 0x1A
#define PB11 0x1B
#define PB12 0x1C
#define PB13 0x1D
#define PB14 0x1E
#define PB15 0x1F
#define PC0  0x20
#define PC1  0x21
#define PC2  0x22
#define PC3  0x23
#define PC4  0x24
#define PC5  0x25
#define PC6  0x26
#define PC7  0x27
#define PC8  0x28
#define PC9  0x29
#define PC10 0x2A
#define PC11 0x2B
#define PC12 0x2C
#define PC13 0x2D
#define PC14 0x2E
#define PC15 0x2F
#define PD0  0x30
#define PD1  0x31
#define PD2  0x32
#define PD3  0x33

// STM32 LL UID stubs used by telemetry.cpp::computeDeviceId
uint32_t LL_GetUID_Word0(void);
uint32_t LL_GetUID_Word1(void);
uint32_t LL_GetUID_Word2(void);

#ifdef __cplusplus
}  // extern "C"

// -- HardwareSerial / Serial / Serial6 (C++ class) ------------------------
class HardwareSerial {
public:
    HardwareSerial(uint32_t rx_pin, uint32_t tx_pin) : m_uart_id(rx_pin ^ tx_pin) {}
    HardwareSerial() : m_uart_id(0) {}

    void   begin(unsigned long baud);
    void   end(void);
    int    available(void);
    int    peek(void);
    int    read(void);
    size_t readBytes(uint8_t *buf, size_t len);
    size_t readBytes(char    *buf, size_t len);

    size_t write(uint8_t b);
    size_t write(const uint8_t *buf, size_t len);
    size_t write(const char *s) { return write((const uint8_t *)s, strlen(s)); }
    size_t write(const char *buf, size_t len) {
        return write((const uint8_t *)buf, len);
    }

    // Arduino-flavoured print() overloads. Just enough for DBG_DO traces and
    // sendBootMessage's "B:" / "\t" / "\n" markers.
    size_t print(const char *s)   { return write(s); }
    size_t print(uint8_t b)       { return write(b); }
    size_t print(int v);
    size_t print(unsigned int v);
    size_t print(long v);
    size_t print(unsigned long v);

    size_t println(void)              { return write("\r\n"); }
    size_t println(const char *s)     { size_t n = write(s); return n + write("\r\n"); }
    size_t println(int v)             { size_t n = print(v); return n + write("\r\n"); }
    size_t println(unsigned int v)    { size_t n = print(v); return n + write("\r\n"); }
    size_t println(long v)            { size_t n = print(v); return n + write("\r\n"); }
    size_t println(unsigned long v)   { size_t n = print(v); return n + write("\r\n"); }

    void flush(void) {}

private:
    uint32_t m_uart_id;
};

extern HardwareSerial Serial;
extern HardwareSerial Serial6;

// -- HardwareTimer stub ----------------------------------------------------
// MakAir's blower.h / pressure_valve.h declare HardwareTimer pointers as
// member fields and constructor arguments; we never actually drive any
// timer in QEMU, so a hollow class that satisfies the type system is enough.
enum TimerFormat { MICROSEC_FORMAT = 0, HERTZ_FORMAT = 1, MICROSEC_COMPARE_FORMAT = 2 };

class HardwareTimer {
public:
    HardwareTimer(void * /*tim*/) {}
    HardwareTimer() {}
    void setOverflow(uint32_t /*v*/, TimerFormat /*f*/ = MICROSEC_FORMAT) {}
    void setMode(uint32_t /*ch*/, uint32_t /*mode*/, uint32_t /*pin*/ = 0) {}
    void setCaptureCompare(uint32_t /*ch*/, uint32_t /*v*/, TimerFormat /*f*/ = MICROSEC_COMPARE_FORMAT) {}
    void resume() {}
    void pause() {}
    void refresh() {}
    void attachInterrupt(void (* /*cb*/)()) {}
    void attachInterrupt(uint32_t /*ch*/, void (* /*cb*/)()) {}
    void detachInterrupt() {}
    uint32_t getOverflow(TimerFormat /*f*/ = MICROSEC_FORMAT) { return 0; }
};

// MakAir's pressure_valve.h types
typedef enum { TIMER_OUTPUT_COMPARE_PWM1 = 0, TIMER_OUTPUT_COMPARE_PWM2 = 1, TIMER_OUTPUT_COMPARE_FORCED_ACTIVE = 2 } TimerModes_t;
typedef enum { actuator = 0, sensor = 1 } HwLayer_t;

// Some MakAir headers reference `TIM3` etc. as `void *` magic numbers.
#define TIM1 ((void *)1)
#define TIM2 ((void *)2)
#define TIM3 ((void *)3)
#define TIM4 ((void *)4)

// Forward types referenced in pressure_valve.h / blower.h that we never call.
typedef int32_t (*MICROSEC_COMPARE_FORMAT_t)(int);

#endif  // __cplusplus
