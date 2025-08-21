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

// NOLINTBEGIN

// =============================================================================
// Device State Management
// =============================================================================

patchestry_device_state_t __patchestry_device_state(void* device) {
    if (device == NULL) {
        return PATCHESTRY_DEVICE_NULL;
    }
    
    // Basic validation - check if pointer is readable
    if (!__patchestry_is_readable(device, sizeof(void*))) {
        return PATCHESTRY_DEVICE_ERROR;
    }
    
    // In a real implementation, this would check device-specific state
    return PATCHESTRY_DEVICE_VALID;
}

// =============================================================================
// Register Access Implementation
// =============================================================================

uint32_t __patchestry_read_reg32(volatile uint32_t* reg) {
    if (!__patchestry_is_valid_pointer(reg)) {
        __patchestry_set_error("Invalid register pointer for 32-bit read");
        return 0;
    }
    return *reg;
}

void __patchestry_write_reg32(volatile uint32_t* reg, uint32_t value) {
    if (!__patchestry_is_valid_pointer(reg)) {
        __patchestry_set_error("Invalid register pointer for 32-bit write");
        return;
    }
    *reg = value;
}

uint8_t __patchestry_read_reg8(volatile uint8_t* reg) {
    if (!__patchestry_is_valid_pointer(reg)) {
        __patchestry_set_error("Invalid register pointer for 8-bit read");
        return 0;
    }
    return *reg;
}

uint16_t __patchestry_read_reg16(volatile uint16_t* reg) {
    if (!__patchestry_is_valid_pointer(reg)) {
        __patchestry_set_error("Invalid register pointer for 16-bit read");
        return 0;
    }
    return *reg;
}

uint64_t __patchestry_read_reg64(volatile uint64_t* reg) {
    if (!__patchestry_is_valid_pointer(reg)) {
        __patchestry_set_error("Invalid register pointer for 64-bit read");
        return 0;
    }
    return *reg;
}

void __patchestry_write_reg8(volatile uint8_t* reg, uint8_t value) {
    if (!__patchestry_is_valid_pointer(reg)) {
        __patchestry_set_error("Invalid register pointer for 8-bit write");
        return;
    }
    *reg = value;
}

void __patchestry_write_reg16(volatile uint16_t* reg, uint16_t value) {
    if (!__patchestry_is_valid_pointer(reg)) {
        __patchestry_set_error("Invalid register pointer for 16-bit write");
        return;
    }
    *reg = value;
}

void __patchestry_write_reg64(volatile uint64_t* reg, uint64_t value) {
    if (!__patchestry_is_valid_pointer(reg)) {
        __patchestry_set_error("Invalid register pointer for 64-bit write");
        return;
    }
    *reg = value;
}

bool __patchestry_is_valid_register(void* reg_addr) {
    // Basic pointer validation
    return __patchestry_is_valid_pointer(reg_addr);
}

// =============================================================================
// Memory-Mapped I/O Implementation
// =============================================================================

uint32_t __patchestry_mmio_read32(void* mmio_base, size_t offset) {
    if (mmio_base == NULL) {
        __patchestry_set_error("NULL MMIO base address");
        return 0;
    }
    
    volatile uint32_t* reg = (volatile uint32_t*)((char*)mmio_base + offset);
    return __patchestry_read_reg32(reg);
}

void __patchestry_mmio_write32(void* mmio_base, size_t offset, uint32_t value) {
    if (mmio_base == NULL) {
        __patchestry_set_error("NULL MMIO base address");
        return;
    }
    
    volatile uint32_t* reg = (volatile uint32_t*)((char*)mmio_base + offset);
    __patchestry_write_reg32(reg, value);
}

bool __patchestry_is_valid_mmio_range(void* mmio_base, size_t offset, size_t size) {
    if (mmio_base == NULL || size == 0) {
        return false;
    }
    
    // Check for overflow
    uintptr_t addr = (uintptr_t)mmio_base;
    uintptr_t end;
    
    if (__builtin_add_overflow(addr, offset, &end) ||
        __builtin_add_overflow(end, size, &end)) {
        return false;
    }
    
    return __patchestry_is_readable((const char*)mmio_base + offset, size);
}

// NOLINTEND