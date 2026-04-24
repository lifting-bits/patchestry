// RUN: true
// Test-only patch that replaces a cir.load: takes the pointer and returns
// a u32 value. Used to verify REPLACE mode now works on non-call, non-binop ops.

typedef unsigned int uint32_t;

uint32_t patch__replace__load_u32(uint32_t *ptr) {
    (void) ptr;
    return 0xDEADBEEFu;
}
