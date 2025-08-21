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
// Device and Hardware APIs
// =============================================================================

typedef enum patchestry_device_state_e {
    PATCHESTRY_DEVICE_VALID,
    PATCHESTRY_DEVICE_NULL,
    PATCHESTRY_DEVICE_BUSY,
    PATCHESTRY_DEVICE_ERROR
} patchestry_device_state_t;

patchestry_device_state_t __patchestry_device_state(void *device);

// Register access for different sizes
bool __patchestry_is_valid_register(void *reg_addr);

uint8_t __patchestry_read_reg8(volatile uint8_t *reg);
uint16_t __patchestry_read_reg16(volatile uint16_t *reg);
uint32_t __patchestry_read_reg32(volatile uint32_t *reg);
uint64_t __patchestry_read_reg64(volatile uint64_t *reg);

void __patchestry_write_reg8(volatile uint8_t *reg, uint8_t value);
void __patchestry_write_reg16(volatile uint16_t *reg, uint16_t value);
void __patchestry_write_reg32(volatile uint32_t *reg, uint32_t value);
void __patchestry_write_reg64(volatile uint64_t *reg, uint64_t value);

// =============================================================================
// Structure and Field Access APIs
// =============================================================================

#ifndef PATCHESTRY_OFFSET_OF
    #define PATCHESTRY_OFFSET_OF(type, field) ((size_t) offsetof(type, field))
#endif

// Get pointer to a field using type and field name
#define PATCHESTRY_GET_FIELD_T(struct_ptr, type, field) \
    ((void *) ((char *) (struct_ptr) + PATCHESTRY_OFFSET_OF(type, field)))

// Read a field value with explicit field type
#define PATCHESTRY_READ_FIELD_T(struct_ptr, type, field, field_type) \
    (*(field_type *) ((char *) (struct_ptr) + PATCHESTRY_OFFSET_OF(type, field)))

// Write a field value with explicit field type
#define PATCHESTRY_WRITE_FIELD_T(struct_ptr, type, field, value, field_type) \
    do { \
        *(field_type *) ((char *) (struct_ptr) + PATCHESTRY_OFFSET_OF(type, field)) = (value); \
    } while (0)

// =============================================================================
// Memory-Mapped I/O Helpers
// =============================================================================

static inline uint32_t __patchestry_mmio_read32_fast(void *base, size_t offset) {
    return *(volatile uint32_t *) ((uint8_t *) base + offset);
}

static inline void __patchestry_mmio_write32_fast(void *base, size_t offset, uint32_t value) {
    *(volatile uint32_t *) ((uint8_t *) base + offset) = value;
}

// Traditional MMIO operations with full validation
uint32_t __patchestry_mmio_read32(void *mmio_base, size_t offset);
void __patchestry_mmio_write32(void *mmio_base, size_t offset, uint32_t value);
bool __patchestry_is_valid_mmio_range(void *mmio_base, size_t offset, size_t size);

// MMIO access with bounds checking
#define PATCHESTRY_MMIO_READ32(base, offset) \
    (__patchestry_is_valid_mmio_range(base, offset, 4) \
         ? __patchestry_mmio_read32(base, offset) \
         : 0)

#define PATCHESTRY_MMIO_WRITE32(base, offset, value) \
    do { \
        if (__patchestry_is_valid_mmio_range(base, offset, 4)) { \
            __patchestry_mmio_write32(base, offset, value); \
        } \
    } while (0)

#ifdef __cplusplus
}
#endif
