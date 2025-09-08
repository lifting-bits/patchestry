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

// =============================================================================
// Memory Safety APIs - Optimized for Medical Device Firmware
// =============================================================================

// Compact pointer validation for binary injection
#define PATCHESTRY_IS_NULL(ptr)                ((ptr) == ((void *) 0))
#define PATCHESTRY_PTR_ALIGN_CHECK(ptr, align) (((uintptr_t) (ptr) & ((align) - 1)) == 0)

// Fast inline pointer validation
bool __patchestry_is_null_pointer(const volatile void *ptr);

// Medical device specific pointer validation (minimal overhead)
static inline bool __patchestry_is_valid_pointer(const volatile void *ptr) {
    return __patchestry_is_null_pointer(ptr);
}

// Pointer Validation - optimized versions
bool __patchestry_is_readable(const void *ptr, size_t size); // NOLINT
bool __patchestry_is_writable(const void *ptr, size_t size); // NOLINT

bool __patchestry_check_bounds(const void *ptr, size_t offset, size_t size); // NOLINT
bool __patchestry_check_buffer_write(
    void *buffer, size_t buffer_size, size_t write_size
); // NOLINT

bool __patchestry_check_string_bounds(const char *str, size_t max_len); // NOLINT

// Additional safety helpers
bool __patchestry_is_initialized(void *ptr, size_t size);

// =============================================================================
// Access Control
// =============================================================================

typedef enum {
    PATCHESTRY_ACCESS_READ    = 1,
    PATCHESTRY_ACCESS_WRITE   = 2,
    PATCHESTRY_ACCESS_EXECUTE = 4
} patchestry_access_t;

bool __patchestry_check_access(const void *ptr, patchestry_access_t access);

// =============================================================================
// Assertion and Validation APIs
// =============================================================================

// Core assertion function
void __patchestry_assert_fail(
    const char *assertion, const char *file, int line, const char *msg
);

// Assertion Macros
#define PATCHESTRY_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            __patchestry_assert_fail(#cond, __FILE__, __LINE__, msg); \
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

// =============================================================================
// Security / Hash / RNG
// =============================================================================

uint32_t __patchestry_crc32(void *data, size_t size);
uint64_t __patchestry_hash64(void *data, size_t size);
bool __patchestry_verify_checksum(void *data, size_t size, uint32_t expected_crc);

uint32_t __patchestry_random_u32(void);
void __patchestry_random_bytes(void *buffer, size_t size);

// =============================================================================
// Array Operations
// =============================================================================
bool __patchestry_check_array_bounds(
    const void *array, size_t array_size, size_t index, size_t elem_size
);

// Safe array access macro
#define PATCHESTRY_ARRAY_GET(array, index, size, type) \
    (__patchestry_check_array_bounds(array, size, index, sizeof(type)) \
         ? &((type *) (array))[index] \
         : NULL)

#ifdef __cplusplus
}
#endif
