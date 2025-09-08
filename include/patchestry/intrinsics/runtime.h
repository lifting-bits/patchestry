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
// Runtime Introspection APIs
// =============================================================================

// Caller Information
const char *__patchestry_get_caller_name(void);
const char *__patchestry_get_caller_at_depth(int depth);

// Thread and CPU Information
uint32_t __patchestry_get_thread_id(void);
uint32_t __patchestry_get_cpu_id(void);
uint32_t __patchestry_get_process_id(void);

// Function Metadataint __patchestry_get_param_count(const char *func_name);
const char *__patchestry_get_return_type(const char *func_name);

// =============================================================================
// Patch Context and Metadata APIs
// =============================================================================

// Context Management
typedef struct
{
    const char *patch_name;
    const char *target_function;
    void *user_data;
} patchestry_context_t;

patchestry_context_t *__patchestry_get_context(void);
const char *__patchestry_get_patch_name(void);
const char *__patchestry_get_patch_version(void);

// ClangIR Metadata Generation
void __patchestry_emit_metadata(const char *key, const char *value);
void __patchestry_emit_annotation(const char *annotation);

// Execution Tracking
void __patchestry_mark_patch_start(const char *patch_name);
void __patchestry_mark_patch_end(const char *patch_name);
bool __patchestry_is_patch_active(const char *patch_name);

// =============================================================================
// Logging and Debugging APIs
// =============================================================================

// Logging Levels
typedef enum {
    PATCHESTRY_LOG_DEBUG = 0,
    PATCHESTRY_LOG_INFO  = 1,
    PATCHESTRY_LOG_WARN  = 2,
    PATCHESTRY_LOG_ERROR = 3,
    PATCHESTRY_LOG_FATAL = 4
} patchestry_log_level_t;

// Core logging function
void __patchestry_log(patchestry_log_level_t level, const char *fmt, ...);

// Logging Macros
#define PATCHESTRY_DEBUG(fmt, ...) __patchestry_log(PATCHESTRY_LOG_DEBUG, fmt, ##__VA_ARGS__)
#define PATCHESTRY_INFO(fmt, ...)  __patchestry_log(PATCHESTRY_LOG_INFO, fmt, ##__VA_ARGS__)
#define PATCHESTRY_WARN(fmt, ...)  __patchestry_log(PATCHESTRY_LOG_WARN, fmt, ##__VA_ARGS__)
#define PATCHESTRY_ERROR(fmt, ...) __patchestry_log(PATCHESTRY_LOG_ERROR, fmt, ##__VA_ARGS__)
#define PATCHESTRY_FATAL(fmt, ...) __patchestry_log(PATCHESTRY_LOG_FATAL, fmt, ##__VA_ARGS__)

// Debug Utilities
void __patchestry_dump_memory(void *ptr, size_t size, const char *label);
void __patchestry_dump_struct(void *struct_ptr, const char *struct_type);

// =============================================================================
// Performance and Profiling APIs
// =============================================================================

// High-precision timing functions
patchestry_time_t __patchestry_get_time(void);
double __patchestry_time_diff_ms(patchestry_time_t start, patchestry_time_t end);

// Performance profiling
void __patchestry_profile_start(const char *name);
void __patchestry_profile_end(const char *name);
void __patchestry_profile_report(void);

#ifdef __cplusplus
}
#endif
