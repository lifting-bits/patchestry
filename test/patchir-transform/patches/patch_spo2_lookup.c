// RUN: true
typedef float SFtype;

extern void halt();

extern void write_str(const char *str);

extern unsigned char spo2_lookup(SFtype param_0);

#define patch_assert(cond) \
    do { \
        if (!(cond)) { \
            const char *msg = "Assertion failed: " #cond "\n"; \
            write_str(msg); \
            halt(); \
        } \
    } while (0)

int isfinite_float(float f) {
    union {
        float f;
        unsigned int u;
    } u = { f };

    unsigned int exponent = (u.u >> 23) & 0xFF;
    return exponent != 0xFF;
}

void patch__before__spo2_lookup(SFtype param_0) {
    // Your patch code here
    if (isfinite_float(param_0)) {
        // Do something with param_0
        patch_assert(0);
    }
    return;
}

void patch__after__spo2_lookup(unsigned char return_value) {
    // Your patch code here
    return;
}

unsigned char patch__replace__spo2_lookup(SFtype param_0) {
    // Test replacement: clamp to valid range without calling original
    if (param_0 < 0.0f || param_0 > 100.0f) {
        return (unsigned char)0;
    }
    return (unsigned char)param_0;
}
