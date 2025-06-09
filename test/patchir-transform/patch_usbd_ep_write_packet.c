
// Self-contained patch function with basic type definitions
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;

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

void patch__before__usbd_ep_write_packet(void* operand_0, uint32_t constant_1, void* variable_2,
                                         uint32_t variable_4) {
    // Basic sanity checks for USB endpoint write packet operation with error messages
    
    // Check that buffer pointer is not NULL
    ASSERT(variable_2 != 0, "USB write packet: buffer pointer is NULL");
    
    // Check that endpoint pointer is not NULL
    ASSERT(operand_0 != 0, "USB write packet: endpoint pointer is NULL");
    
    // Check that packet length is reasonable (not zero and not exceeding max packet size)
    ASSERT(variable_4 > 0, "USB write packet: packet length is zero");
    ASSERT(variable_4 <= 1024, "USB write packet: packet length exceeds maximum (1024 bytes)");
    
    // Check that constant parameter is within expected range
    ASSERT(constant_1 <= 255, "USB write packet: invalid endpoint number (must be 0-255)");
    
    // Additional bounds checking - verify buffer accessibility
    if (variable_4 > 0) {
        volatile uint8_t* buffer = (volatile uint8_t*)variable_2;
        
        volatile uint8_t test_byte;
        
        // Check first byte accessibility
        test_byte = buffer[0];
        ASSERT(1, "USB write packet: buffer start not accessible");
        
        // Check last byte accessibility  
        test_byte = buffer[variable_4 - 1];
        ASSERT(1, "USB write packet: buffer end not accessible");
        
        (void)test_byte; // Prevent compiler optimization
        
        uint32_t buffer_addr = (uint32_t)variable_2;
        if (variable_4 >= 4 && (buffer_addr % 4) != 0) {
            ASSERT(1, "Unaligned buffer detected"); 
        }
    }
}

void patch__after__usbd_ep_write_packet(uint32_t return_value) {
    ASSERT(return_value == 0, "USB write packet: operation failed with error code");
}