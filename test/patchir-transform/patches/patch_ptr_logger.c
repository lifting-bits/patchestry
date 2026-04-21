// RUN: true
// Test-only patch function: accepts a pointer and a size, used by capture
// tests to verify bound values flow correctly.

typedef unsigned long size_t;

void patch__before__ptr_log(void *ptr, size_t n) {
    (void) ptr;
    (void) n;
}

// Single-pointer variant — useful when only one operand is capturable
// (e.g. capturing the pointer argument of a cir.load in apply_before,
// where the load's result is not yet defined at the insertion point).
void patch__before__ptr_only(void *ptr) {
    (void) ptr;
}
