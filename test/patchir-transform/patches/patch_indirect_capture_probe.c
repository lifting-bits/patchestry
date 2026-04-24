// RUN: true

// Probe patch used to verify that a capture against operand 0 of an
// indirect cir.call binds the user argument, not the callee function
// pointer. Takes a uint32_t* so it matches the type of the matched
// call's actual arg 0 and no intervening cast is synthesized.

typedef unsigned int uint32_t;

void patch__before__indirect_capture_probe(uint32_t *user_arg0) {
    (void) user_arg0;
}
