// RUN: true
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;

extern void halt(void);

// CWE-190: Replace unchecked 16-bit multiply with overflow-checked version.
// The vulnerable pattern in PeekHandler::Process() multiplies two uint16_t
// values (count * block_size) without checking for overflow. If the product
// exceeds 65535, it wraps around and the subsequent bounds check passes
// with the wrong value, enabling out-of-bounds memory reads.
//
// This patch widens the operands to 32-bit, performs the multiply, and
// returns UINT16_MAX if the result would overflow uint16_t.
unsigned int patch__replace__int_mul16(unsigned int a, unsigned int b) {
    uint32_t wide_result = a * b;
    if (wide_result > (uint32_t)0xFFFF) {
        return (unsigned int)0xFFFF;
    }
    return (unsigned int)wide_result;
}
