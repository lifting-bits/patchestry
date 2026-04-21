// RUN: true
// Test-only patch function: accepts a pointer and a size, used by capture
// tests to verify bound values flow correctly.

typedef unsigned long size_t;

void patch__before__ptr_log(void *ptr, size_t n) {
    (void) ptr;
    (void) n;
}
