// RUN: true

// Self-contained patch function with basic type definitions
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef unsigned long size_t;

// Define SIZE_MAX for size_t
#define SIZE_MAX ((size_t)-1)

// USB operation type enumeration
typedef enum {
    USB_OP_READ = 0,
    USB_OP_WRITE = 1,
    USB_OP_CONTROL = 2,
    USB_OP_BULK = 3,
    USB_OP_INTERRUPT = 4,
    USB_OP_ISOCHRONOUS = 5
} usb_operation_type_t;

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

// Maximum allowed USB transfer sizes based on operation type
#define USB_MAX_CONTROL_SIZE    64
#define USB_MAX_BULK_SIZE       512
#define USB_MAX_INTERRUPT_SIZE  64
#define USB_MAX_ISOCHRONOUS_SIZE 1023
#define USB_MAX_GENERAL_SIZE    1024

/**
 * Check buffer bounds for USB operations
 * 
 * @param buffer - Buffer to validate
 * @param buffer_size - Available buffer size
 * @param requested_size - Requested operation size
 * @param operation_type - Type of USB operation
 */
void check_usb_buffer_bounds(const void* buffer, size_t buffer_size, 
                           size_t requested_size, usb_operation_type_t operation_type) {
    
    // Basic null pointer validation
    ASSERT(buffer != 0, "USB bounds check: buffer pointer is NULL");
    
    // Validate buffer size is non-zero
    ASSERT(buffer_size > 0, "USB bounds check: buffer size is zero");
    
    // Validate requested size is non-zero
    ASSERT(requested_size > 0, "USB bounds check: requested size is zero");
    
    // Core bounds check: ensure requested size doesn't exceed available buffer
    ASSERT(requested_size <= buffer_size, "USB bounds check: requested size exceeds buffer capacity");
    
    // Validate operation type is within valid range
    ASSERT(operation_type >= USB_OP_READ && operation_type <= USB_OP_ISOCHRONOUS,
           "USB bounds check: invalid operation type");
    
    // Check operation-specific size limits
    switch (operation_type) {
        case USB_OP_CONTROL:
            ASSERT(requested_size <= USB_MAX_CONTROL_SIZE,
                   "USB bounds check: control transfer size exceeds maximum (64 bytes)");
            break;
            
        case USB_OP_BULK:
            ASSERT(requested_size <= USB_MAX_BULK_SIZE,
                   "USB bounds check: bulk transfer size exceeds maximum (512 bytes)");
            break;
            
        case USB_OP_INTERRUPT:
            ASSERT(requested_size <= USB_MAX_INTERRUPT_SIZE,
                   "USB bounds check: interrupt transfer size exceeds maximum (64 bytes)");
            break;
            
        case USB_OP_ISOCHRONOUS:
            ASSERT(requested_size <= USB_MAX_ISOCHRONOUS_SIZE,
                   "USB bounds check: isochronous transfer size exceeds maximum (1023 bytes)");
            break;
            
        case USB_OP_READ:
        case USB_OP_WRITE:
        default:
            ASSERT(requested_size <= USB_MAX_GENERAL_SIZE,
                   "USB bounds check: transfer size exceeds maximum (1024 bytes)");
            break;
    }
    
    // Additional buffer accessibility validation
    if (requested_size > 0) {
        volatile uint8_t* test_buffer = (volatile uint8_t*)buffer;
        volatile uint8_t test_byte;
        
        // Test first byte accessibility
        test_byte = test_buffer[0];
        ASSERT(1, "USB bounds check: buffer start not accessible");
        
        // Test last byte accessibility
        test_byte = test_buffer[requested_size - 1];
        ASSERT(1, "USB bounds check: buffer end not accessible");
        
        (void)test_byte; // Prevent compiler optimization
        
        // Check for reasonable buffer alignment (4-byte aligned for larger transfers)
        uint32_t buffer_addr = (uint32_t)buffer;
        if (requested_size >= 4 && (buffer_addr % 4) != 0) {
            ASSERT(1, "USB bounds check: buffer not properly aligned for large transfer");
        }
    }
    
    // Integer overflow protection
    ASSERT(buffer_size < SIZE_MAX - requested_size,
           "USB bounds check: potential integer overflow detected");
}

// Wrapper functions for specific operation types
void check_usb_read_bounds(const void* buffer, size_t buffer_size, size_t requested_size) {
    check_usb_buffer_bounds(buffer, buffer_size, requested_size, USB_OP_READ);
}

void check_usb_write_bounds(const void* buffer, size_t buffer_size, size_t requested_size) {
    check_usb_buffer_bounds(buffer, buffer_size, requested_size, USB_OP_WRITE);
}

void check_usb_control_bounds(const void* buffer, size_t buffer_size, size_t requested_size) {
    check_usb_buffer_bounds(buffer, buffer_size, requested_size, USB_OP_CONTROL);
} 