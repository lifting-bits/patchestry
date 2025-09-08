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
// Patch Application Utilities
// =============================================================================

// Patch lifecycle management
typedef enum {
    PATCHESTRY_PATCH_INACTIVE  = 0,
    PATCHESTRY_PATCH_ACTIVE    = 1,
    PATCHESTRY_PATCH_FAILED    = 2,
    PATCHESTRY_PATCH_SUSPENDED = 3
} patchestry_patch_state_t;

// Hook type for callbacks during patch transitions
typedef void (*patchestry_patch_hook_t)(const char *patch_name);

// Monitoring callback for external observers
typedef void (*patchestry_patch_monitor_t)(
    const char *patch_name, patchestry_patch_state_t state
);

// Patch statistics structure to aggregate metrics
typedef struct
{
    const char *patch_name;
    unsigned long apply_count;
    unsigned long success_count;
    unsigned long failure_count;
    double avg_execution_time_ms;
    unsigned long total_execution_time_ms;
} patchestry_patch_stats_t;

// Patch management
bool __patchestry_patch_init(const char *patch_name, const char *version);
void __patchestry_patch_cleanup(const char *patch_name);
patchestry_patch_state_t __patchestry_patch_get_state(const char *patch_name);
bool __patchestry_patch_set_state(const char *patch_name, patchestry_patch_state_t state);

// Patch data access
bool __patchestry_patch_store_data(const char *patch_name, const void *data, size_t size);
void *__patchestry_patch_get_data(const char *patch_name);
size_t __patchestry_patch_get_data_size(const char *patch_name);

// Conditional patching
bool __patchestry_check_env_variable(const char *var_name, const char *expected_value);
bool __patchestry_check_build_config(const char *config_name);
bool __patchestry_check_target_arch(const char *arch_name);
bool __patchestry_check_function_exists(const char *func_name);
bool __patchestry_check_symbol_exists(const char *symbol_name);

// Validation and testing
bool __patchestry_validate_patch_target(const char *patch_name, void *target_ptr);
bool __patchestry_validate_patch_signature(const char *patch_name, const char *expected_sig);
bool __patchestry_validate_patch_checksum(const char *patch_name, uint32_t expected_crc);

#ifdef __cplusplus
}
#endif
