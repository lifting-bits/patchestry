// RUN: true
typedef unsigned long size_t;

// Per-TU loud-fail: lowers to a single trapping instruction (UDF on ARMv7-M)
// and never returns. `__builtin_trap` is a compiler intrinsic — no extern
// symbol — so patchir-transform / patchir-cir2llvm emits its body inside
// the patch module and Patcherex2's relocatable link does not need to
// resolve any extern. The trap routes the CPU into HardFault instead of
// silently spinning, so overflow is externally observable.
__attribute__((noreturn))
static void secpump_assert_fail(void) {
    __builtin_trap();
}

// SecPump seeded vuln: MaliciousMemCpy copies `n` bytes into `dest` with no
// bound on `dest`'s allocation. Replace with a bounds-checked variant that
// fails loud (named `secpump_assert_fail`) when `n` exceeds the caller-
// declared destination capacity. Silent truncation is forbidden — overflow
// must become a visible firmware halt rather than partial/corrupt copy.
void patch__replace__MaliciousMemCpy(void *dest, void *src, size_t n,
                                     size_t dest_cap) {
    if (n > dest_cap) {
        secpump_assert_fail();
    }
    char *cdest = (char *)dest;
    char *csrc  = (char *)src;
    for (size_t i = 0; i < n; ++i) {
        cdest[i] = csrc[i];
    }
}
