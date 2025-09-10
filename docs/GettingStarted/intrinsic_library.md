# Patchestry Intrinsics Library

## Overview

The **Patchestry Intrinsics Library** provides a stable, comprehensive runtime API for developers to query the **state of the patched program** and access the **YAML/MLIR metadata** that drives patch insertion. It is specifically designed for **ClangIR-based patching** workflows, enabling runtime analysis and safe modification of target binaries.

The library bridges the gap between patch specifications and runtime execution by providing functionality including runtime introspection for caller analysis and execution context, memory safety through validation/corruption detection, program state access for structures, fields, and registers, and patch orchestration with mutability controls and error handling.

These intrinsics enable developers to write robust, debuggable patches with comprehensive visibility into program behavior and patch execution context.

## Architecture

### Core Components

1. **Runtime Introspection** - Functions for runtime analysis and metadata access
2. **Safety Checks** - Nullability, bounds checking, and assertion utilities  
3. **Device Management** - Structured field access and device state
4. **Patch Utilities** - Common patch application helpers

### Header Structure

```
include/patchestry/intrinsics/
├── patchestry_intrinsics.h     # Main header for patch inclusion
├── runtime.h                   # Runtime introspection functions
├── safety.h                    # Safety and validation checks
├── device.h                    # Device structure access utilities
└── patch_utils.h              # Patch application utilities
```

## Intrinsic API Reference

The intrinsic libraries will support the list of functions that can be embedded directly to the target binary during patching:

### 1. Runtime Introspection APIs

```c
// Caller Information
const char* __patchestry_get_caller_name(void);
const char* __patchestry_get_caller_at_depth(int depth);

// Thread and CPU Information
uint32_t __patchestry_get_thread_id(void);
uint32_t __patchestry_get_cpu_id(void);
uint32_t __patchestry_get_process_id(void);

const char* __patchestry_get_return_type(const char* func_name);
```

### 2. Memory Safety APIs

```c
// Pointer Validation
bool __patchestry_is_null_pointer(void* ptr);
bool __patchestry_is_valid_pointer(void* ptr);
bool __patchestry_is_readable(void* ptr, size_t size);
bool __patchestry_is_writable(void* ptr, size_t size);

// Bounds Checking
bool __patchestry_check_bounds(void* ptr, size_t offset, size_t size);
bool __patchestry_check_buffer_write(void* buffer, size_t buffer_size, size_t write_size);
bool __patchestry_check_string_bounds(const char* str, size_t max_len);

// Memory State
bool __patchestry_is_initialized(void* ptr, size_t size);
```

### 3. Assertion and Validation APIs

```c
// Assertion Macros
#define PATCHESTRY_ASSERT(cond, msg)
#define PATCHESTRY_ASSERT_NOT_NULL(ptr, msg)
#define PATCHESTRY_ASSERT_BOUNDS(ptr, size, msg)

// Conditional Guards
#define PATCHESTRY_IF_VALID(ptr, block)
#define PATCHESTRY_RETURN_IF_NULL(ptr, retval)
#define PATCHESTRY_RETURN_IF_INVALID(cond, retval)
```

### 4. Structure and Field Access APIs

```c
// Generic Field Access
#define PATCHESTRY_GET_FIELD(struct_ptr, field_name, field_type)
#define PATCHESTRY_SET_FIELD(struct_ptr, field_name, value, field_type)

// Array Operations
#define PATCHESTRY_ARRAY_GET(array, index, size, type)
bool __patchestry_check_array_bounds(void* array, size_t array_size, size_t index, size_t elem_size);
```

### 5. Device and Hardware APIs

```c
// Device State Management
typedef enum {
    PATCHESTRY_DEVICE_VALID,
    PATCHESTRY_DEVICE_NULL,
    PATCHESTRY_DEVICE_BUSY,
    PATCHESTRY_DEVICE_ERROR
} patchestry_device_state_t;

patchestry_device_state_t __patchestry_device_state(void* device);

// Register Access
uint32_t __patchestry_read_reg32(volatile uint32_t* reg);
void __patchestry_write_reg32(volatile uint32_t* reg, uint32_t value);
bool __patchestry_is_valid_register(void* reg_addr);
```

### 6. Mutability and Access Control APIs

```c
// Mutability Control
bool __patchestry_is_mutable(void* ptr, size_t size);

// Access Permissions
typedef enum {
    PATCHESTRY_ACCESS_READ    = 1,
    PATCHESTRY_ACCESS_WRITE   = 2,
    PATCHESTRY_ACCESS_EXECUTE = 4
} patchestry_access_t;

bool __patchestry_check_access(void* ptr, patchestry_access_t access);
```

### 7. Logging and Debugging APIs

```c
// Logging Levels
typedef enum {
    PATCHESTRY_LOG_DEBUG,
    PATCHESTRY_LOG_INFO,
    PATCHESTRY_LOG_WARN,
    PATCHESTRY_LOG_ERROR,
    PATCHESTRY_LOG_FATAL
} patchestry_log_level_t;

// Logging Macros
#define PATCHESTRY_DEBUG(fmt, ...)
#define PATCHESTRY_INFO(fmt, ...)
#define PATCHESTRY_ERROR(fmt, ...)

// Debug Utilities
void __patchestry_dump_memory(void* ptr, size_t size, const char* label);
void __patchestry_dump_struct(void* struct_ptr, const char* struct_type);
```

### 8. Patch Context and Metadata APIs

```c
// Context Management
typedef struct {
    const char* patch_name;
    const char* target_function;
    void* user_data;
} patchestry_context_t;

patchestry_context_t* __patchestry_get_context(void);
const char* __patchestry_get_patch_name(void);
const char* __patchestry_get_patch_version(void);

// ClangIR Metadata Generation (Not implemented yet)
void __patchestry_emit_metadata(const char* key, const char* value);
void __patchestry_emit_annotation(const char* annotation);

// Execution Tracking
void __patchestry_mark_patch_start(const char* patch_name);
void __patchestry_mark_patch_end(const char* patch_name);
bool __patchestry_is_patch_active(const char* patch_name);
```

### 9. Security APIs

```c
// Random Number Generation
uint32_t __patchestry_random_u32(void);
void __patchestry_random_bytes(void* buffer, size_t size);
```

### 10. Performance and Profiling APIs

```c
// Timing
typedef uint64_t patchestry_time_t;
patchestry_time_t __patchestry_get_time(void);
double __patchestry_time_diff_ms(patchestry_time_t start, patchestry_time_t end);

// Profiling
void __patchestry_profile_start(const char* name);
void __patchestry_profile_end(const char* name);
void __patchestry_profile_report(void);
```

### Basic Patch Template
```c
#include <patchestry/intrinsics/patchestry_intrinsics.h>

void device_security_patch(void* device) {
    __patchestry_profile_start("device_security_patch");
    
    // Log patch entry with metadata from YAML specification
    PATCHESTRY_INFO("Applying patch: %s v%s", 
                    __patchestry_get_patch_name(),
                    __patchestry_get_patch_version());
    
    // Validate device with comprehensive safety checks
    PATCHESTRY_RETURN_IF_NULL(device, -1);
    
    if (__patchestry_device_state(device) != PATCHESTRY_DEVICE_VALID) {
        PATCHESTRY_ERROR("Invalid device state");
        __patchestry_set_error("Device validation failed");
        return;
    }
    
    // Safe field modification
    uint32_t* config = PATCHESTRY_GET_FIELD(device, config, uint32_t);
    if (__patchestry_is_mutable(config, sizeof(uint32_t))) {
        *config |= SECURITY_ENABLED;
        PATCHESTRY_DEBUG("Security configuration updated");
        
        // Emit metadata for audit trail
        __patchestry_emit_metadata("security_flag", "enabled");
    } else {
        PATCHESTRY_WARN("Config field is immutable");
    }
    
    // End profiling
    __patchestry_profile_end("device_security_patch");
}
```

### Clang IR Integration
- Intrinsic functions will be declared as external symbols in generated IR
- Runtime library provides implementations linked at final binary creation
- Compile-time constant folding for static analysis where possible