// RUN: true
extern void halt();

// CWE-476: Injected before get_name — checks pointer is non-null.
// The vulnerable pattern is calling a method on a pointer that may be
// NULL after a failed lookup, without checking first.
void patch__before__get_name(const void *var_ptr) {
    if (var_ptr == (void *)0) {
        halt();
    }
}
