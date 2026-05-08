// RUN: true
typedef unsigned long size_t;

// Per-TU assert: never returns. `static` keeps it at internal linkage so
// patchir-transform / patchir-cir2llvm emits its body inside the patch
// module and Patcherex2's relocatable link does not need to resolve any
// extern when embedding the patched function into the firmware.
//
// The patchestry-embedded clang frontend (lib/patchestry/Passes/Compiler.cpp)
// rejects `__builtin_trap` with "use of unknown builtin", so we stick with
// the infinite-loop form that every other patch in this tree uses.
__attribute__((noreturn))
static void secpump_assert_fail(void) {
    for (;;) { }                         // halt visibly; no return path.
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
