// RUN: true
typedef unsigned int uint32_t;
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

void patch__before__cmp__bl_spi_mode(uint32_t global_var) {
    if (global_var == 0) {
        ASSERT(0, "bl_spi_mode is 0");
    }
}