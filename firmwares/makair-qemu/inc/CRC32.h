// CRC32.h — minimal shim for the Arduino CRC32 library used by upstream
// telemetry.cpp + serial_control.cpp. Implements CRC-32/IEEE (poly 0xEDB88320)
// via a straightforward bit-by-bit update — fine for the throughput we run at
// in QEMU (a handful of frames per second).
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

class CRC32 {
public:
    CRC32() : m_state(0xFFFFFFFFu) {}

    void reset(void) { m_state = 0xFFFFFFFFu; }

    void update(uint8_t b) {
        m_state ^= b;
        for (int i = 0; i < 8; ++i) {
            uint32_t mask = -(m_state & 1u);
            m_state = (m_state >> 1) ^ (0xEDB88320u & mask);
        }
    }

    void update(const uint8_t *buf, size_t len) {
        for (size_t i = 0; i < len; ++i) update(buf[i]);
    }
    void update(const char *buf, size_t len) {
        update((const uint8_t *)buf, len);
    }
    void update(const char *s) {
        update((const uint8_t *)s, strlen(s));
    }

    uint32_t finalize(void) const { return m_state ^ 0xFFFFFFFFu; }

private:
    uint32_t m_state;
};
