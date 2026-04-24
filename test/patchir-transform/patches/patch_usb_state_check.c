// RUN: true

// Runtime USB device-state validation, invoked before usbd_ep_write_packet.
// Migrated from the previous runtime-contract version (contract::before::
// test_contract) when runtime contracts were merged into patches.

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
    USB_STATE_NOTATTACHED = 0,
    USB_STATE_ATTACHED,
    USB_STATE_POWERED,
    USB_STATE_RECONNECTING,
    USB_STATE_UNAUTHENTICATED,
    USB_STATE_DEFAULT,
    USB_STATE_ADDRESS,
    USB_STATE_CONFIGURED,
    USB_STATE_SUSPENDED
} usb_device_state_t;

typedef struct usb_device {
    usb_device_state_t state;
} usb_device_t;

void patch__before__usb_state_check(usb_device_t *usb_device, const void *buffer) {
    ASSERT(usb_device->state == USB_STATE_CONFIGURED, "USB not ready for use");
}
