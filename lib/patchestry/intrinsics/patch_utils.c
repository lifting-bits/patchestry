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

// NOLINTBEGIN
// =============================================================================
// Patch Management Implementation
// =============================================================================

typedef struct patch_entry {
    char name[64];
    char version[32];
    void* patch_data;
    size_t data_size;
    struct patch_entry* next;
} patch_entry_t;

static patch_entry_t* patch_list = NULL;
static pthread_mutex_t patch_mutex = PTHREAD_MUTEX_INITIALIZER;

static patch_entry_t* find_patch_entry(const char* patch_name) {
    patch_entry_t* entry = patch_list;
    while (entry) {
        if (strncmp(entry->name, patch_name, sizeof(entry->name)) == 0) {
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
    
    entry->patch_data = NULL;
    entry->data_size = 0;
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

bool __patchestry_check_env_variable(const char* var_name, const char* expected_value, size_t size) {
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

    return strncmp(value, expected_value, size) == 0;
}

bool __patchestry_check_build_config(const char* config_name, size_t size) {
    if (config_name == NULL) {
        return false;
    }
    
    // Check common build configurations
    if (strncmp(config_name, "DEBUG", size) == 0) {
        #ifdef DEBUG
        return true;
        #else
        return false;
        #endif
    }
    
    if (strncmp(config_name, "RELEASE", size) == 0) {
        #ifdef NDEBUG
        return true;
        #else
        return false;
        #endif
    }
    
    return false;
}

bool __patchestry_check_target_arch(const char* arch_name, size_t size) {
    if (arch_name == NULL) {
        return false;
    }
    
    // Check architecture
    #ifdef __x86_64__
    if (strncmp(arch_name, "x86_64", size) == 0) {
        return true;
    }
    #endif
    
    #ifdef __aarch64__
    if (strncmp(arch_name, "aarch64", size) == 0) {
        return true;
    }
    #endif
    
    #ifdef __arm__
    if (strncmp(arch_name, "arm", size) == 0) {
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

// NOLINTEND