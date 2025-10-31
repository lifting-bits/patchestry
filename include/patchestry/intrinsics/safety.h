/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN

#define PATCHESTRY_IS_NULL(ptr)                ((ptr) == ((void *) 0))
#define PATCHESTRY_PTR_ALIGN_CHECK(ptr, align) (((uintptr_t) (ptr) & ((align) - 1)) == 0)

bool __patchestry_is_null_pointer(const volatile void *ptr);

static inline bool __patchestry_is_valid_pointer(const volatile void *ptr) {
    return !__patchestry_is_null_pointer(ptr);
}

// Pointer Validation - optimized versions
bool __patchestry_is_readable(const void *ptr, size_t size);
bool __patchestry_is_writable(const void *ptr, size_t size);

bool __patchestry_check_bounds(const void *ptr, size_t offset, size_t size);
bool __patchestry_check_buffer_write(void *buffer, size_t buffer_size, size_t write_size);

bool __patchestry_check_string_bounds(const char *str, size_t max_len);

bool __patchestry_is_initialized(void *ptr, size_t size);

typedef enum {
    PATCHESTRY_ACCESS_READ    = 1,
    PATCHESTRY_ACCESS_WRITE   = 2,
    PATCHESTRY_ACCESS_EXECUTE = 4
} patchestry_access_t;

bool __patchestry_check_access(const void *ptr, patchestry_access_t access);

extern long write(int, const void*, unsigned long);

static inline unsigned long strlen_custom(const char *s) {
    unsigned long n = 0;
    while (s && s[n]) n++;
    return n;
}

static inline void
__patchestry_assert_fail(const char *a, const char *f, int l, const char *m)
{
    if (a) write(2, "Assert: ", 8), write(2, a, strlen_custom(a)), write(2, "\n", 1);
    if (f) write(2, " at ", 4),   write(2, f, strlen_custom(f)), write(2, ":", 1);
    if (m) write(2, "  ", 2), write(2, m, strlen_custom(m)), write(2, "\n", 1);
    *(volatile int *)0 = 0;
}

// Assertion Macros
#define PATCHESTRY_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
	    if(msg) write(2, "Assert: ", 8), write(2, msg, strlen_custom(msg)), write(2, "\n", 1); \
            *(volatile int *)0 = 0; \
        } \
    } while (0)

#define PATCHESTRY_ASSERT_NOT_NULL(ptr, msg) \
    PATCHESTRY_ASSERT(!__patchestry_is_null_pointer(ptr), msg)

#define PATCHESTRY_ASSERT_BOUNDS(ptr, size, msg) \
    PATCHESTRY_ASSERT(__patchestry_check_bounds(ptr, 0, size), msg)

// Conditional Guards
#define PATCHESTRY_IF_VALID(ptr, block) \
    do { \
        if (__patchestry_is_valid_pointer(ptr)) { \
            block \
        } \
    } while (0)

#define PATCHESTRY_RETURN_IF_NULL(ptr, retval) \
    do { \
        if (__patchestry_is_null_pointer(ptr)) { \
            __patchestry_set_error("Null pointer encountered"); \
            return retval; \
        } \
    } while (0)

#define PATCHESTRY_RETURN_IF_INVALID(cond, retval) \
    do { \
        if (!(cond)) { \
            __patchestry_set_error("Validation condition failed: " #cond); \
            return retval; \
        } \
    } while (0)

// Array Operations
// =============================================================================
bool __patchestry_check_array_bounds(
    const void *array, size_t array_size, size_t index, size_t elem_size
);

#define PATCHESTRY_ARRAY_GET(array, index, size, type) \
    (__patchestry_check_array_bounds(array, size, index, sizeof(type)) \
         ? &((type *) (array))[index] \
         : NULL)

uint32_t __patchestry_random_u32(void);
void __patchestry_random_bytes(void *buffer, size_t size);

// NOLINTEND
#ifdef __cplusplus
}
#endif
