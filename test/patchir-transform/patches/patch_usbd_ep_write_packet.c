// RUN: true

#define USE_C99_TYPES
#include <patchestry/intrinsics/patchestry_intrinsics.h>

// Simple assert implementation with error messages
void patch_assert_fail(const char* message, const char* file, int line) {
    *(volatile int*)0 = 0; // Intentional segfault to stop execution
}

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            patch_assert_fail(message, __FILE__, __LINE__); \
        } \
    } while(0)


void patch__before__usbd_ep_write_packet(void* operand_0, void* variable_2) {
    // Basic sanity checks for USB endpoint write packet operation with error messages
    
    // Check that buffer pointer is not NULL
    ASSERT(variable_2 != 0, "USB write packet: buffer pointer is NULL");
    
    // Check that endpoint pointer is not NULL
    ASSERT(operand_0 != 0, "USB write packet: endpoint pointer is NULL");
}

void patch__before__usbd_cp_write_packet__update_state(void** operand_0, void** variable_2/*, uint32_t *variable_4*/) {
    if (__patchestry_is_null_pointer(variable_2)) {
        __patchestry_set_error("USB write packet: buffer pointer is NULL");
        return;
    }
    if (__patchestry_is_null_pointer(operand_0)) {
        __patchestry_set_error("USB write packet: endpoint pointer is NULL");
        return;
    }
    // update buffer pointer with the new value
    *variable_2 = (void*)0x1000;
    // update endpoint pointer with the new value
    *operand_0 = (void*)0x2000;
    // update packet length with the new value
    //*variable_4 = 1024;
}

void patch__after__usbd_ep_write_packet(uint32_t return_value) {
    ASSERT(return_value == 0, "USB write packet: operation failed with error code");
}
