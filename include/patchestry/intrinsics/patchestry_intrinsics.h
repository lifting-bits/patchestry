/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

// Compiler attributes for intrinsic functions must be defined
// before including any headers that rely on them.
#ifdef __clang__
    #define PATCHESTRY_INTRINSIC __attribute__((always_inline)) __attribute__((flatten))
    #define PATCHESTRY_NOINLINE  __attribute__((noinline))
    #define PATCHESTRY_CONST     __attribute__((const))
    #define PATCHESTRY_CRITICAL  __attribute__((section(".critical")))
#else
    #define PATCHESTRY_INTRINSIC inline
    #define PATCHESTRY_NOINLINE
    #define PATCHESTRY_CONST
    #define PATCHESTRY_CRITICAL
#endif

// Medical device firmware specific macros for ultra-compact code
#ifdef __clang__
    #define PATCHESTRY_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define PATCHESTRY_UNLIKELY(x) __builtin_expect(!!(x), 0)
    #ifdef __has_builtin
        #if __has_builtin(__builtin_assume)
            #define PATCHESTRY_ASSUME(x) __builtin_assume(!!(x))
        #else
            #define PATCHESTRY_ASSUME(x) \
                do { \
                    if (!(x)) \
                        __builtin_unreachable(); \
                } while (0)
        #endif
    #else
        #define PATCHESTRY_ASSUME(x) \
            do { \
                if (!(x)) \
                    __builtin_unreachable(); \
            } while (0)
    #endif
#else
    // Fallback for non-clang compilers
    #define PATCHESTRY_LIKELY(x)   (x)
    #define PATCHESTRY_UNLIKELY(x) (x)
    #define PATCHESTRY_ASSUME(x)   ((void) 0)
#endif

// Include common types before anything else
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN
// Version information
#define PATCHESTRY_INTRINSICS_VERSION_MAJOR 1
#define PATCHESTRY_INTRINSICS_VERSION_MINOR 0
#define PATCHESTRY_INTRINSICS_VERSION_PATCH 0

// Include all intrinsic subsystems
#include "device.h"
#include "patch_utils.h"
#include "runtime.h"
#include "safety.h"

// Global error handling
extern const char *__patchestry_get_last_error(void);
extern void __patchestry_set_error(const char *error_msg);
extern void __patchestry_clear_error(void);

// NOLINTEND
#ifdef __cplusplus
}
#endif
