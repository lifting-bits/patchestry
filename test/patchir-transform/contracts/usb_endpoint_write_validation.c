// RUN: true

// Simple assert implementation with error messages
void patch_assert_fail(const char *message, const char *file, int line) {
    *(volatile int *) 0 = 0; // Intentional segfault to stop execution
}

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            patch_assert_fail(message, __FILE__, __LINE__); \
        } \
    } while (0)

typedef enum usb_device_state {
    /* NOTATTACHED isn't in the USB spec, and this state acts
     * the same as ATTACHED ... but it's clearer this way.
     */
    USB_STATE_NOTATTACHED = 0,

    /* chapter 9 and authentication (wireless) device states */
    USB_STATE_ATTACHED,
    USB_STATE_POWERED,         /* wired */
    USB_STATE_RECONNECTING,    /* auth */
    USB_STATE_UNAUTHENTICATED, /* auth */
    USB_STATE_DEFAULT,         /* limited function */
    USB_STATE_ADDRESS,
    USB_STATE_CONFIGURED, /* most functions */

    USB_STATE_SUSPENDED

    /* NOTE:  there are actually four different SUSPENDED
     * states, returning to POWERED, DEFAULT, ADDRESS, or
     * CONFIGURED respectively when SOF tokens flow again.
     * At this level there's no difference between L1 and L2
     * suspend states.  (L2 being original USB 1.1 suspend.)
     */
} usb_device_state_t;

typedef struct usb_device {
    usb_device_state_t state;
} usb_device_t;

void contract__before__test_contract(usb_device_t *usb_device, const void *buffer) {
    ASSERT(usb_device->state == USB_STATE_CONFIGURED, "USB not ready for use");
}
