// RUN: true

#define assert(cond) do { if (!(cond)) abort(); } while (0)

typedef __SIZE_TYPE__ size_t;
typedef struct usb_device {
    enum usb_device_state   state;
};

void contract(usb_device *usb_device, const void *buffer) {
    assert(0);
}