// RUN: true

// Runtime contract inserted at function entrypoint.
// Applied to bl_usb__send_message to validate the incoming message pointer
// is non-null before any processing occurs.

typedef struct { unsigned char type; } bl_msg_data_t;

static void contract_assert_fail(void) {
    *(volatile int *)0 = 0; // deliberate trap on contract violation
}

void contract__entrypoint__message_entry_check(void *msg) {
    if (msg == (void *)0) {
        contract_assert_fail();
    }
}
