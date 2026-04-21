// RUN: true
typedef unsigned int uint32_t;

void patch__before__test_function(void) {
    // Intentionally empty — the patch exists to verify instrumentation
    // inserts a call at the correct site.
}
