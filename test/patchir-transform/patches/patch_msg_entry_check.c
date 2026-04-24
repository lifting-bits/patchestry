// RUN: true

// Runtime patch inserted at function entrypoint.
// Applied to bl_usb__send_message to validate the incoming message pointer
// is non-null before any processing occurs.

static void patch_assert_fail(void) {
    *(volatile int *)0 = 0; // deliberate trap on patch violation
}

void patch__entrypoint__message_entry_check(void *msg) {
    if (msg == (void *)0) {
        patch_assert_fail();
    }
}
