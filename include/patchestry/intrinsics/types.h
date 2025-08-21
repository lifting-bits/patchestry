/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

// NOLINTBEGIN
// =============================================================================
// Common C11 Type Definitions for Patchestry Intrinsics
// =============================================================================
// This header consolidates all C11 type definitions to reduce redundancy
// across the intrinsic library headers.

// Prefer system headers when available to avoid conflicts with C99 types
// and not defined use_patchestry_types macro
#ifndef USE_C99_TYPES
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
#else
    // C99 compliant types - avoid external dependencies
    #ifndef _PATCHESTRY_STDINT_H
        #define _PATCHESTRY_STDINT_H

// C99 integer types
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// Pointer arithmetic type
typedef unsigned long uintptr_t;

        // Standard size_t definition
        #ifndef _SIZE_T_DEFINED
            #define _SIZE_T_DEFINED
typedef unsigned long size_t;
        #endif

        // Boolean type for C99
        #ifndef __cplusplus
            #if !defined(bool) && !defined(__bool_true_false_are_defined)
typedef _Bool bool;
                #define true  1
                #define false 0
            #endif
        #endif

        // offsetof macro for C99
        #ifndef offsetof
            #define offsetof(type, member) ((size_t) &((type *) 0)->member)
        #endif

        // NULL definition
        #ifndef NULL
            #define NULL ((void *) 0)
        #endif
    #endif // _PATCHESTRY_STDINT_H
#endif

typedef uint64_t patchestry_time_t;

// =============================================================================
// Common Attribute Macros
// =============================================================================

#ifndef PATCHESTRY_CONST
    #define PATCHESTRY_CONST
#endif

// NOLINTEND
