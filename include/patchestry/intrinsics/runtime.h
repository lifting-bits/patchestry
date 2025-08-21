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

// Caller Information
const char *__patchestry_get_caller_name(void);
const char *__patchestry_get_caller_at_depth(int depth);

// Thread and CPU Information
uint32_t __patchestry_get_thread_id(void);
uint32_t __patchestry_get_cpu_id(void);
uint32_t __patchestry_get_process_id(void);

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

typedef struct
{
    const char *patch_name;
    const char *target_function;
    void *user_data;
} patchestry_context_t;

// High-precision timing functions
patchestry_time_t __patchestry_get_time(void);
double __patchestry_time_diff_ms(patchestry_time_t start, patchestry_time_t end);

// Performance profiling
void __patchestry_profile_start(const char *name);
void __patchestry_profile_end(const char *name);
void __patchestry_profile_report(void);

// NOLINTEND
#ifdef __cplusplus
}
#endif
