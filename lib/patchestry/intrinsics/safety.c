/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#define _GNU_SOURCE        1
#define _DEFAULT_SOURCE    1
#define _POSIX_C_SOURCE    200809L

#include <patchestry/intrinsics/patchestry_intrinsics.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// POSIX/Linux headers
#include <unistd.h>        // for sysconf, getpagesize
#include <sys/mman.h>      // for mincore, mmap, etc.
#include <sys/types.h>     // for size_t, off_t
#include <errno.h>         // for errno

// =============================================================================
// Memory Safety Implementation
// =============================================================================
bool __patchestry_is_null_pointer(const volatile void *ptr) {
    return ptr == NULL;
}

bool __patchestry_is_readable(const void* ptr, size_t size) {
    if (ptr == NULL || size == 0) { // NOLINT
        return false;
    }
    
    size_t page_size = getpagesize();
    uintptr_t start = ((uintptr_t)ptr) & ~(page_size - 1);
    size_t pages = (((uintptr_t)ptr + size - start) + page_size - 1) / page_size;
    
    unsigned char* vec = malloc(pages);
    if (!vec) {
        return false;
    }
    
    int result = mincore((void*)start, pages * page_size, vec);
    free(vec);
    
    return result == 0;
}

bool __patchestry_is_writable(const void* ptr, size_t size) { // NOLINT
    if (!__patchestry_is_readable(ptr, size)) {
        return false;
    }
    
    // For now, assume readable memory is writable
    // This should be enhanced with proper memory protection checking
    return true;
}

bool __patchestry_check_bounds(const void* ptr, size_t offset, size_t size) {
    if (ptr == NULL) { // NOLINT
        return false;
    }
    
    // Check for integer overflow
    uintptr_t base = (uintptr_t)ptr;
    uintptr_t end;
    
    if (__builtin_add_overflow(base, offset, &end) ||
        __builtin_add_overflow(end, size, &end)) {
        return false;
    }
    
    return __patchestry_is_readable((const char*)ptr + offset, size);
}

bool __patchestry_check_buffer_write(void* buffer, size_t buffer_size, size_t write_size) {
    if (buffer == NULL || buffer_size == 0) { // NOLINT
        return false;
    }
    
    return write_size <= buffer_size && 
           __patchestry_is_writable(buffer, write_size);
}

bool __patchestry_check_string_bounds(const char* str, size_t max_len) {
    if (str == NULL) { // NOLINT
        return false;
    }
    
    // Check if we can read at least the first byte
    if (!__patchestry_is_readable(str, 1)) {
        return false;
    }
    
    // Find string length within bounds
    for (size_t i = 0; i < max_len; i++) {
        if (!__patchestry_is_readable(str + i, 1)) {
            return false;
        }
        if (str[i] == '\0') {
            return true;
        }
    }
    
    return false; // String too long
}

bool __patchestry_is_initialized(void* ptr, size_t size) {
    // This is a stub implementation
    // Real implementation would need memory tracking or valgrind-like functionality
    return __patchestry_is_readable(ptr, size);
}

// =============================================================================
// Assertion Implementation
// =============================================================================

void __patchestry_assert_fail(const char* assertion, const char* file, int line, const char* msg) {
    fprintf(stderr, "PATCHESTRY ASSERTION FAILED: %s\n", assertion);
    fprintf(stderr, "  File: %s:%d\n", file, line);
    if (msg) {
        fprintf(stderr, "  Message: %s\n", msg);
    }
    
    // Set error for later retrieval
    char error_buffer[512];
    snprintf(error_buffer, sizeof(error_buffer), 
             "Assertion failed: %s at %s:%d", assertion, file, line);
    __patchestry_set_error(error_buffer);
    
    // In debug builds, abort; in release, just continue
    #ifdef DEBUG
    abort();
    #endif
}

// =============================================================================
// Access Control
// =============================================================================

bool __patchestry_check_access(const void* ptr, patchestry_access_t access) {
    if (ptr == NULL) return false;
    
    // Simplified implementation
    if (access & PATCHESTRY_ACCESS_READ) {
        if (!__patchestry_is_readable(ptr, 1)) {
            return false;
        }
    }
    
    if (access & PATCHESTRY_ACCESS_WRITE) {
        if (!__patchestry_is_writable(ptr, 1)) {
            return false;
        }
    }
    
    // Execute permission check would need more sophisticated implementation
    if (access & PATCHESTRY_ACCESS_EXECUTE) {
        // Stub - would need to check memory protection flags
        return true;
    }
    
    return true;
}

uint32_t __patchestry_random_u32(void) {
    // Use standard library rand() function
    // rand() returns int in range [0, RAND_MAX]
    // Combine two rand() calls to get full 32-bit range

    uint32_t high = (uint32_t)rand() & 0xFFFF;
    uint32_t low = (uint32_t)rand() & 0xFFFF;
    return (high << 16) | low;
}

void __patchestry_random_bytes(void* buffer, size_t size) {
    if (buffer == NULL || size == 0) return;
    
    unsigned char* bytes = (unsigned char*)buffer;
    for (size_t i = 0; i < size; i++) {
        bytes[i] = (unsigned char)__patchestry_random_u32();
    }
}

// =============================================================================
// Array Operations
// =============================================================================

bool __patchestry_check_array_bounds(const void* array, size_t array_size, size_t index, size_t elem_size) {
    if (array == NULL || elem_size == 0) {
        return false;
    }
    
    // Check for overflow
    size_t offset;
    if (__builtin_mul_overflow(index, elem_size, &offset)) {
        return false;
    }
    
    if (offset + elem_size > array_size) {
        return false;
    }
    
    return __patchestry_is_readable((const char*)array + offset, elem_size);
}
