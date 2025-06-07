/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

// =============================================================================
// Compiler Detection
// =============================================================================

#if defined(__GNUC__)
    #define COMPILER_GCC
#elif defined(__clang__)
    #define COMPILER_CLANG
#elif defined(_MSC_VER)
    #define COMPILER_MSVC
#else
    #error "Unknown compiler"
#endif

// =============================================================================
// Diagnostic Pragma Helpers
// =============================================================================

#define PRAGMA(X) _Pragma(#X)

#if defined(COMPILER_GCC)
    #define PRAGMA_DIAGNOSTIC_PUSH PRAGMA(GCC diagnostic push)
    #define PRAGMA_DIAGNOSTIC_POP  PRAGMA(GCC diagnostic pop)
#elif defined(COMPILER_CLANG)
    #define PRAGMA_DIAGNOSTIC_PUSH PRAGMA(clang diagnostic push)
    #define PRAGMA_DIAGNOSTIC_POP  PRAGMA(clang diagnostic pop)
#elif defined(COMPILER_MSVC)
    #define PRAGMA_DIAGNOSTIC_PUSH PRAGMA(warning(push))
    #define PRAGMA_DIAGNOSTIC_POP  PRAGMA(warning(pop))
#else
    #define PRAGMA_DIAGNOSTIC_PUSH
    #define PRAGMA_DIAGNOSTIC_POP
#endif

// =============================================================================
// Null Dereference Warnings
// =============================================================================

#if defined(COMPILER_GCC)
    #define DISABLE_NULL_DEREF_WARNING_IMPL PRAGMA(GCC diagnostic ignored "-Wnull-dereference")
#elif defined(COMPILER_CLANG)
    #define DISABLE_NULL_DEREF_WARNING_IMPL \
        PRAGMA(clang diagnostic ignored "-Wnull-dereference") \
        PRAGMA(clang diagnostic ignored "-Wnullable-to-nonnull-conversion")
#elif defined(COMPILER_MSVC)
    #define DISABLE_NULL_DEREF_WARNING_IMPL \
        PRAGMA(warning(disable : 6011)) \
        PRAGMA(warning(disable : 28182))
#else
    #define DISABLE_NULL_DEREF_WARNING_IMPL
#endif

#define DISABLE_NULL_DEREF_WARNING_BEGIN \
    PRAGMA_DIAGNOSTIC_PUSH \
    DISABLE_NULL_DEREF_WARNING_IMPL

#define DISABLE_NULL_DEREF_WARNING_END PRAGMA_DIAGNOSTIC_POP

// =============================================================================
// Unused Variable/Parameter Warnings
// =============================================================================

#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
    #define DISABLE_UNUSED_WARNING_IMPL \
        PRAGMA(GCC diagnostic ignored "-Wunused-variable") \
        PRAGMA(GCC diagnostic ignored "-Wunused-parameter")
#elif defined(COMPILER_MSVC)
    #define DISABLE_UNUSED_WARNING_IMPL \
        PRAGMA(warning(disable : 4101)) \
        PRAGMA(warning(disable : 4100))
#else
    #define DISABLE_UNUSED_WARNING_IMPL
#endif

#define DISABLE_UNUSED_WARNING_BEGIN \
    PRAGMA_DIAGNOSTIC_PUSH \
    DISABLE_UNUSED_WARNING_IMPL

#define DISABLE_UNUSED_WARNING_END PRAGMA_DIAGNOSTIC_POP

// =============================================================================
// Sign Conversion Warnings
// =============================================================================

#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
    #define DISABLE_SIGN_CONVERSION_WARNING_IMPL \
        PRAGMA(GCC diagnostic ignored "-Wsign-conversion") \
        PRAGMA(GCC diagnostic ignored "-Wsign-compare")
#elif defined(COMPILER_MSVC)
    #define DISABLE_SIGN_CONVERSION_WARNING_IMPL \
        PRAGMA(warning(disable : 4018)) \
        PRAGMA(warning(disable : 4389))
#else
    #define DISABLE_SIGN_CONVERSION_WARNING_IMPL
#endif

#define DISABLE_SIGN_CONVERSION_WARNING_BEGIN \
    PRAGMA_DIAGNOSTIC_PUSH \
    DISABLE_SIGN_CONVERSION_WARNING_IMPL

#define DISABLE_SIGN_CONVERSION_WARNING_END PRAGMA_DIAGNOSTIC_POP

// =============================================================================
// Deprecation Warnings
// =============================================================================

#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
    #define DISABLE_DEPRECATION_WARNING_IMPL \
        PRAGMA(GCC diagnostic ignored "-Wdeprecated-declarations")
#elif defined(COMPILER_MSVC)
    #define DISABLE_DEPRECATION_WARNING_IMPL PRAGMA(warning(disable : 4996))
#else
    #define DISABLE_DEPRECATION_WARNING_IMPL
#endif

#define DISABLE_DEPRECATION_WARNING_BEGIN \
    PRAGMA_DIAGNOSTIC_PUSH \
    DISABLE_DEPRECATION_WARNING_IMPL

#define DISABLE_DEPRECATION_WARNING_END PRAGMA_DIAGNOSTIC_POP

// =============================================================================
// Convenience Function Attributes
// =============================================================================

#if defined(COMPILER_CLANG) || defined(COMPILER_GCC)
    #define NO_SANITIZE_NULL __attribute__((no_sanitize("null")))
    #define UNUSED_FUNCTION  __attribute__((unused))
    #define MAYBE_UNUSED     [[maybe_unused]]
#elif defined(COMPILER_MSVC)
    #define NO_SANITIZE_NULL
    #define UNUSED_FUNCTION
    #define MAYBE_UNUSED
#else
    #define NO_SANITIZE_NULL
    #define UNUSED_FUNCTION
    #define MAYBE_UNUSED
#endif

// =============================================================================
// Patchestry utility Macros
// =============================================================================
#define PE_RELAX_WARNINGS_BEGIN \
    PRAGMA_DIAGNOSTIC_PUSH \
    DISABLE_NULL_DEREF_WARNING_IMPL \
    DISABLE_UNUSED_WARNING_IMPL \
    DISABLE_SIGN_CONVERSION_WARNING_IMPL \
    DISABLE_DEPRECATION_WARNING_IMPL

#define PE_RELAX_WARNINGS_END PRAGMA_DIAGNOSTIC_POP
