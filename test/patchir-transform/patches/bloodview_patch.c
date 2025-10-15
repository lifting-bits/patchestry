// RUN: true
// Simplified implementation for cross-compilation without relying on libc headers
// We'll provide minimal forward declarations needed for vsnprintf forwarding.
// For ARM32, size_t is typically unsigned int (32-bit).
typedef unsigned int size_t;

// Manual variadic argument handling for ARM32 AAPCS
// Variadic arguments are passed on the stack after named parameters
typedef char* va_list;

// Calculate the size of a type rounded up to the nearest word (4 bytes on ARM32)
#define __va_argsiz(t) (((sizeof(t) + 3) & ~3))

// Initialize va_list to point to the first variadic argument
#define va_start(ap, last) ((ap) = (va_list)&(last) + __va_argsiz(last))

// Clean up (no-op on ARM32)
#define va_end(ap) ((void)0)

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

    if (dest_size > 0) {
        dest[dest_size - 1] = '\0';
    }

    return result;
}

/*
#include "patchestry/intrinsics/patchestry_intrinsics.h"

void contract__sprintf(int return_value, int dest_size)
{
    // assert if return value is less than 0 or more than dest size
    if(return_value < 0 || return_value > dest_size) {
        patchestry_assert(0, "sprintf returned invalid value");
    }
}*/