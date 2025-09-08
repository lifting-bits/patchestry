/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/intrinsics/patchestry_intrinsics.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <dlfcn.h>
#include <unistd.h>

// =============================================================================
// Patch Management Implementation
// =============================================================================

typedef struct patch_entry {
    char name[64];
    char version[32];
    char target_function[64];
    patchestry_patch_state_t state;
    void* patch_data;
    size_t data_size;
    
    // Dependencies and conflicts
    char dependencies[8][64];
    int dependency_count;
    char conflicts[8][64];
    int conflict_count;
    
    struct patch_entry* next;
} patch_entry_t;

static patch_entry_t* patch_list = NULL;
static pthread_mutex_t patch_mutex = PTHREAD_MUTEX_INITIALIZER;

static patch_entry_t* find_patch_entry(const char* patch_name) {
    patch_entry_t* entry = patch_list;
    while (entry) {
        if (strcmp(entry->name, patch_name) == 0) {
            return entry;
        }
        entry = entry->next;
    }
    return NULL;
}

bool __patchestry_patch_init(const char* patch_name, const char* version) {
    if (patch_name == NULL || version == NULL) {
        return false;
    }
    
    pthread_mutex_lock(&patch_mutex);
    
    // Check if patch already exists
    if (find_patch_entry(patch_name)) {
        pthread_mutex_unlock(&patch_mutex);
        return false;
    }
    
    patch_entry_t* entry = malloc(sizeof(patch_entry_t));
    if (!entry) {
        pthread_mutex_unlock(&patch_mutex);
        return false;
    }
    
    strncpy(entry->name, patch_name, sizeof(entry->name) - 1);
    entry->name[sizeof(entry->name) - 1] = '\0';
    
    strncpy(entry->version, version, sizeof(entry->version) - 1);
    entry->version[sizeof(entry->version) - 1] = '\0';
    
    entry->target_function[0] = '\0';
    entry->state = PATCHESTRY_PATCH_INACTIVE;
    entry->patch_data = NULL;
    entry->data_size = 0;
    entry->dependency_count = 0;
    entry->conflict_count = 0;
    entry->next = patch_list;
    patch_list = entry;
    
    pthread_mutex_unlock(&patch_mutex);
    return true;
}

void __patchestry_patch_cleanup(const char* patch_name) {
    if (patch_name == NULL) return;
    
    pthread_mutex_lock(&patch_mutex);
    
    patch_entry_t** current = &patch_list;
    while (*current) {
        if (strcmp((*current)->name, patch_name) == 0) {
            patch_entry_t* to_remove = *current;
            *current = to_remove->next;
            
            if (to_remove->patch_data) {
                free(to_remove->patch_data);
            }
            free(to_remove);
            break;
        }
        current = &(*current)->next;
    }
    
    pthread_mutex_unlock(&patch_mutex);
}

patchestry_patch_state_t __patchestry_patch_get_state(const char* patch_name) {
    if (patch_name == NULL) {
        return PATCHESTRY_PATCH_FAILED;
    }
    
    pthread_mutex_lock(&patch_mutex);
    patch_entry_t* entry = find_patch_entry(patch_name);
    patchestry_patch_state_t state = entry ? entry->state : PATCHESTRY_PATCH_FAILED;
    pthread_mutex_unlock(&patch_mutex);
    
    return state;
}

bool __patchestry_patch_set_state(const char* patch_name, patchestry_patch_state_t state) {
    if (patch_name == NULL) {
        return false;
    }
    
    pthread_mutex_lock(&patch_mutex);
    patch_entry_t* entry = find_patch_entry(patch_name);
    if (entry) {
        entry->state = state;
    }
    pthread_mutex_unlock(&patch_mutex);
    
    return entry != NULL;
}

bool __patchestry_patch_store_data(const char* patch_name, const void* data, size_t size) {
    if (patch_name == NULL || data == NULL || size == 0) {
        return false;
    }
    
    pthread_mutex_lock(&patch_mutex);
    patch_entry_t* entry = find_patch_entry(patch_name);
    if (entry) {
        if (entry->patch_data) {
            free(entry->patch_data);
        }
        
        entry->patch_data = malloc(size);
        if (entry->patch_data) {
            memcpy(entry->patch_data, data, size);
            entry->data_size = size;
        }
    }
    pthread_mutex_unlock(&patch_mutex);
    
    return entry != NULL && entry->patch_data != NULL;
}

void* __patchestry_patch_get_data(const char* patch_name) {
    if (patch_name == NULL) {
        return NULL;
    }
    
    pthread_mutex_lock(&patch_mutex);
    patch_entry_t* entry = find_patch_entry(patch_name);
    void* data = entry ? entry->patch_data : NULL;
    pthread_mutex_unlock(&patch_mutex);
    
    return data;
}

size_t __patchestry_patch_get_data_size(const char* patch_name) {
    if (patch_name == NULL) {
        return 0;
    }
    
    pthread_mutex_lock(&patch_mutex);
    patch_entry_t* entry = find_patch_entry(patch_name);
    size_t size = entry ? entry->data_size : 0;
    pthread_mutex_unlock(&patch_mutex);
    
    return size;
}

// =============================================================================
// Conditional Patching Implementation
// =============================================================================

bool __patchestry_check_env_variable(const char* var_name, const char* expected_value) {
    if (var_name == NULL) {
        return false;
    }
    
    const char* value = getenv(var_name);
    if (value == NULL) {
        return expected_value == NULL;
    }
    
    if (expected_value == NULL) {
        return false;
    }
    
    return strcmp(value, expected_value) == 0;
}

bool __patchestry_check_build_config(const char* config_name) {
    if (config_name == NULL) {
        return false;
    }
    
    // Check common build configurations
    if (strcmp(config_name, "DEBUG") == 0) {
        #ifdef DEBUG
        return true;
        #else
        return false;
        #endif
    }
    
    if (strcmp(config_name, "RELEASE") == 0) {
        #ifdef NDEBUG
        return true;
        #else
        return false;
        #endif
    }
    
    return false;
}

bool __patchestry_check_target_arch(const char* arch_name) {
    if (arch_name == NULL) {
        return false;
    }
    
    // Check architecture
    #ifdef __x86_64__
    if (strcmp(arch_name, "x86_64") == 0) {
        return true;
    }
    #endif
    
    #ifdef __aarch64__
    if (strcmp(arch_name, "aarch64") == 0) {
        return true;
    }
    #endif
    
    #ifdef __arm__
    if (strcmp(arch_name, "arm") == 0) {
        return true;
    }
    #endif
    
    return false;
}

bool __patchestry_check_function_exists(const char* func_name) {
    if (func_name == NULL) {
        return false;
    }
    
    // Use dlsym to check if function exists
    void* handle = dlopen(NULL, RTLD_LAZY);
    if (!handle) {
        return false;
    }
    
    void* func_ptr = dlsym(handle, func_name);
    dlclose(handle);
    
    return func_ptr != NULL;
}

bool __patchestry_check_symbol_exists(const char* symbol_name) {
    // Similar to function check but for any symbol
    return __patchestry_check_function_exists(symbol_name);
}

// =============================================================================
// Patch Validation and Testing
// =============================================================================

bool __patchestry_validate_patch_target(const char* patch_name, void* target_ptr) {
    if (patch_name == NULL || target_ptr == NULL) {
        return false;
    }
    
    // Basic validation - check if target is readable
    return __patchestry_is_readable(target_ptr, sizeof(void*));
}

bool __patchestry_validate_patch_signature(const char* patch_name, const char* expected_sig) {
    if (patch_name == NULL || expected_sig == NULL) {
        return false;
    }
    
    // Stub implementation - would check function signatures
    return true;
}

bool __patchestry_validate_patch_checksum(const char* patch_name, uint32_t expected_crc) {
    if (patch_name == NULL) {
        return false;
    }
    
    void* patch_data = __patchestry_patch_get_data(patch_name);
    size_t data_size = __patchestry_patch_get_data_size(patch_name);
    
    if (patch_data == NULL || data_size == 0) {
        return false;
    }
    
    return __patchestry_verify_checksum(patch_data, data_size, expected_crc);
}
