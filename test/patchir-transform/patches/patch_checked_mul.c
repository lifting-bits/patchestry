// RUN: true
typedef unsigned int uint32_t;

// CWE-190: Replace unchecked multiply with overflow-checked version.
// If the multiplication would overflow, clamp to max value.
unsigned int patch__replace__int_mul(unsigned int a, unsigned int b) {
    if (b != 0 && a > (unsigned int)0xFFFFFFFF / b) {
        return (unsigned int)0xFFFFFFFF;
    }
    return a * b;
}
