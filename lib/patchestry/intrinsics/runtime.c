/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#ifdef __linux__
    #define _GNU_SOURCE        1
    #define _DEFAULT_SOURCE    1
    #define _POSIX_C_SOURCE    200809L
#endif

#include <patchestry/intrinsics/patchestry_intrinsics.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#ifdef __linux__
#include <sched.h>
#endif

// NOLINTBEGIN
// Global error state
static __thread char g_last_error[256] = {0};

// Global context
static patchestry_context_t g_context = {
    .patch_name = "unknown",
    .target_function = "unknown", 
    .user_data = NULL
};

const char* __patchestry_get_last_error(void) {
    return g_last_error[0] ? g_last_error : NULL;
}

void __patchestry_set_error(const char* error_msg) {
    if (error_msg) {
        strncpy(g_last_error, error_msg, sizeof(g_last_error) - 1);
        g_last_error[sizeof(g_last_error) - 1] = '\0';
    }
}

void __patchestry_clear_error(void) {
    g_last_error[0] = '\0';
}

// =============================================================================
// Runtime Introspection Implementation
// =============================================================================

const char* __patchestry_get_caller_name(void) {
    // This is a stub implementation
    // In a real implementation, this would use stack unwinding
    return "unknown_caller";
}

const char* __patchestry_get_caller_at_depth(int depth) {
    // Stub implementation
    static char caller_buffer[64];
    snprintf(caller_buffer, sizeof(caller_buffer), "caller_at_depth_%d", depth);
    return caller_buffer;
}

uint32_t __patchestry_get_thread_id(void) {
#ifdef __linux__
    return (uint32_t)syscall(SYS_gettid);
#elif defined(__APPLE__)
    // On macOS, use pthread_threadid_np for thread ID
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
    return (uint32_t)tid;
#else
    // Fallback for other platforms
    return (uint32_t)pthread_self();
#endif
}

uint32_t __patchestry_get_cpu_id(void) {
#ifdef __linux__
    return (uint32_t)sched_getcpu();
#else
    // CPU affinity APIs are platform-specific, return 0 as fallback
    return 0;
#endif
}

uint32_t __patchestry_get_process_id(void) {
    return (uint32_t)getpid();
}

const char* __patchestry_get_return_type(const char* func_name) {
    // Stub implementation
    return "unknown";
}

// =============================================================================
// Context Management
// =============================================================================

patchestry_context_t* __patchestry_get_context(void) {
    return &g_context;
}

const char* __patchestry_get_patch_name(void) {
    return g_context.patch_name;
}

const char* __patchestry_get_patch_version(void) {
    return "1.0.0"; // Stub version
}

void __patchestry_emit_metadata(const char* key, const char* value) {
    // dummy implementation
    printf("// @patchestry-metadata: %s = %s\n", key, value);
}

void __patchestry_emit_annotation(const char* annotation) {
    // dummy implementation
    printf("// @patchestry-annotation: %s\n", annotation);
}

// =============================================================================
// Execution Tracking
// =============================================================================

// Simple patch tracking (in real implementation would be more sophisticated)
static char active_patches[16][64];
static int active_patch_count = 0;
static pthread_mutex_t patch_mutex = PTHREAD_MUTEX_INITIALIZER;

void __patchestry_mark_patch_start(const char* patch_name) {
    pthread_mutex_lock(&patch_mutex);
    if (active_patch_count < 16) {
        strncpy(active_patches[active_patch_count], patch_name, 63);
        active_patches[active_patch_count][63] = '\0';
        active_patch_count++;
    }
    pthread_mutex_unlock(&patch_mutex);
}

void __patchestry_mark_patch_end(const char* patch_name) {
    pthread_mutex_lock(&patch_mutex);
    for (int i = 0; i < active_patch_count; i++) {
        if (strcmp(active_patches[i], patch_name) == 0) {
            // Shift remaining patches down
            for (int j = i; j < active_patch_count - 1; j++) {
                strcpy(active_patches[j], active_patches[j + 1]);
            }
            active_patch_count--;
            break;
        }
    }
    pthread_mutex_unlock(&patch_mutex);
}

bool __patchestry_is_patch_active(const char* patch_name) {
    pthread_mutex_lock(&patch_mutex);
    bool found = false;
    for (int i = 0; i < active_patch_count; i++) {
        if (strcmp(active_patches[i], patch_name) == 0) {
            found = true;
            break;
        }
    }
    pthread_mutex_unlock(&patch_mutex);
    return found;
}

// =============================================================================
// Logging Implementation
// =============================================================================

static const char* log_level_names[] = {
    "DEBUG", "INFO", "WARN", "ERROR", "FATAL"
};

void __patchestry_log(patchestry_log_level_t level, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    
    printf("[PATCHESTRY:%s] ", log_level_names[level]);
    vprintf(fmt, args);
    printf("\n");
    
    va_end(args);
}

void __patchestry_dump_memory(void* ptr, size_t size, const char* label) {
    if (!ptr || size == 0) return;
    
    printf("=== Memory Dump: %s ===\n", label ? label : "Unknown");
    unsigned char* bytes = (unsigned char*)ptr;
    for (size_t i = 0; i < size; i += 16) {
        printf("%08lx: ", (unsigned long)i);
        
        // Hex dump
        size_t n = (size - i >= 16) ? 16 : (size - i);
        for (size_t j = 0; j < n; j++) {
            printf("%02x ", bytes[i + j]);
        }
        
        // Padding
        for (size_t j = n; j < 16; j++) {
            printf("   ");
        }
        
        printf(" |");
        
        // ASCII dump
        for (size_t j = 0; j < n; j++) {
            unsigned char c = bytes[i + j];
            printf("%c", (c >= 32 && c <= 126) ? c : '.');
        }
        
        printf("|\n");
    }
    printf("========================\n");
}

void __patchestry_dump_struct(void* struct_ptr, const char* struct_type) {
    printf("=== Struct Dump: %s @ %p ===\n", struct_type, struct_ptr);
    // Stub implementation - would need type information
    printf("(Structure introspection not implemented)\n");
    printf("================================\n");
}

// =============================================================================
// Performance and Profiling
// =============================================================================

patchestry_time_t __patchestry_get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (patchestry_time_t)(ts.tv_sec * 1000000000ULL + ts.tv_nsec);
}

double __patchestry_time_diff_ms(patchestry_time_t start, patchestry_time_t end) {
    return (double)(end - start) / 1000000.0; // Convert nanoseconds to milliseconds
}

// Simple profiling implementation
typedef struct {
    char name[64];
    patchestry_time_t start_time;
    patchestry_time_t total_time;
    int call_count;
} profile_entry_t;

static profile_entry_t profiles[32];
static int profile_count = 0;
static pthread_mutex_t profile_mutex = PTHREAD_MUTEX_INITIALIZER;

void __patchestry_profile_start(const char* name) {
    pthread_mutex_lock(&profile_mutex);
    
    // Find existing entry or create new one
    profile_entry_t* entry = NULL;
    for (int i = 0; i < profile_count; i++) {
        if (strcmp(profiles[i].name, name) == 0) {
            entry = &profiles[i];
            break;
        }
    }
    
    if (!entry && profile_count < 32) {
        entry = &profiles[profile_count++];
        strncpy(entry->name, name, sizeof(entry->name) - 1);
        entry->name[sizeof(entry->name) - 1] = '\0';
        entry->total_time = 0;
        entry->call_count = 0;
    }
    
    if (entry) {
        entry->start_time = __patchestry_get_time();
    }
    
    pthread_mutex_unlock(&profile_mutex);
}

void __patchestry_profile_end(const char* name) {
    patchestry_time_t end_time = __patchestry_get_time();
    
    pthread_mutex_lock(&profile_mutex);
    
    for (int i = 0; i < profile_count; i++) {
        if (strcmp(profiles[i].name, name) == 0) {
            profiles[i].total_time += (end_time - profiles[i].start_time);
            profiles[i].call_count++;
            break;
        }
    }
    
    pthread_mutex_unlock(&profile_mutex);
}

void __patchestry_profile_report(void) {
    pthread_mutex_lock(&profile_mutex);
    
    printf("=== Patchestry Profile Report ===\n");
    printf("%-20s %10s %15s %15s\n", "Name", "Calls", "Total (ms)", "Avg (ms)");
    printf("------------------------------------------------------------\n");
    
    for (int i = 0; i < profile_count; i++) {
        double total_ms = (double)profiles[i].total_time / 1000000.0;
        double avg_ms = profiles[i].call_count > 0 ? total_ms / profiles[i].call_count : 0.0;
        
        printf("%-20s %10d %15.3f %15.3f\n", 
               profiles[i].name, profiles[i].call_count, total_ms, avg_ms);
    }
    
    printf("==================================\n");
    
    pthread_mutex_unlock(&profile_mutex);
}
// NOLINTEND