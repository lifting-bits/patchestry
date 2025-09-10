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

// Patch management
bool __patchestry_patch_init(const char *patch_name, const char *version);
void __patchestry_patch_cleanup(const char *patch_name);

// Patch data access
bool __patchestry_patch_store_data(const char *patch_name, const void *data, size_t size);
void *__patchestry_patch_get_data(const char *patch_name);
size_t __patchestry_patch_get_data_size(const char *patch_name);

bool __patchestry_check_env_variable(
    const char *var_name, const char *expected_value, size_t size
);
bool __patchestry_check_build_config(const char *config_name, size_t size);
bool __patchestry_check_target_arch(const char *arch_name, size_t size);
bool __patchestry_check_function_exists(const char *func_name);
bool __patchestry_check_symbol_exists(const char *symbol_name);

#ifdef __cplusplus
}
#endif
// NOLINTEND
