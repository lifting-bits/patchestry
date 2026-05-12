// RUN: true
// Simplified implementation for cross-compilation without relying on libc headers
// We'll provide minimal forward declarations needed for vsnprintf forwarding.
// For ARM32, size_t is typically unsigned int (32-bit).

#define USE_C99_TYPES
#include "patchestry/intrinsics/patchestry_intrinsics.h"

// Use the compiler's built-in va_list support so the ABI-correct lowering
// is selected per target. The previous hand-rolled char*-based va_list
// performed manual word-aligned pointer arithmetic that clangir 22 rejected
// with 'cir.binop op requires all operands to have the same type' (size_t
// vs int promotion in __va_argsiz).
typedef __builtin_va_list va_list;
#define va_start(ap, last) __builtin_va_start(ap, last)
#define va_end(ap)         __builtin_va_end(ap)

// External declarations - will be provided by the target system's C library
int vsnprintf(char *str, size_t size, const char *format, va_list ap);


/*
 * Replacement for legacy sprintf usage that enforces buffer length handling.
 * This version directly uses snprintf which is safer than sprintf.
 * Returns the number of characters that would have been written, excluding the
 * terminating null byte, mirroring snprintf semantics.
 *
 * Note: This simplified version assumes the format string and arguments are
 * passed through directly. The original variadic behavior is maintained by
 * the patch framework during replacement.
 */
int patch__replace__sprintf(char *dest, size_t dest_size, const char *format, ...)
{
    // In practice, this function signature will be matched by the patch system
    // and the call will be rewritten to vsnprintf with the proper arguments.
    // For compilation purposes, we just need the type signature to match.

    if (dest_size == 0) {
        return 0;
    }

    va_list args;
    va_start(args, format);
    int result = vsnprintf(dest, dest_size, format, args);
    va_end(args);

    // Only forcibly null-terminate if truncation occurred
    if (dest_size > 0 && result >= (int)dest_size) {
        dest[dest_size - 1] = '\0';
    }

    return result;
}


// Bounds check for the replaced sprintf's return value, invoked
// apply_after. Migrated from the runtime-contract version
// (was contract__sprintf) when runtime contracts were merged into
// patches.
void patch__after__sprintf_bounds(int return_value, size_t dest_size)
{
    if (return_value < 0 || (size_t) return_value > dest_size) {
        PATCHESTRY_ASSERT(0, "sprintf returned invalid value");
    }
}

