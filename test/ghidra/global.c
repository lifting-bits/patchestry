// UNSUPPORTED: system-windows
// RUN: %cc-x86_64 %s -g -c -o %t.o
// RUN: %decompile-headless --input %t.o --function main --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?main}}"

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "architecture":"{{.*}}","format":"{{.*}}","functions":
// DECOMPILEA-SAME: "name":"{{_?usbd_init}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --input %t.o --list-functions --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":
// LISTFNS-SAME: "name":"{{_?usbd_init}}"
// LISTFNS-SAME: "name":"{{_?main}}"

#include <stdio.h>
#include <stdint.h>

struct usbd_device;
struct usbd_config;

struct {
	struct usbd_device *handle;
} usb_g;

struct usbd_device {
    void* base_addr;
    struct usbd_config *registered_config;
};

struct usbd_config {
    int dummy_config_value;
};


void __attribute__((noinline))
usbd_register_config(struct usbd_device *handle, struct usbd_config* config) {
    if (handle == NULL) {
        printf("handle is NULL\n");
        return;
    }
    handle->registered_config = config;
}

void __attribute__((noinline)) bl_usb__setup() {
    printf("initializing usb\n");
}

struct usbd_device* __attribute__((noinline))
usbd_init() {
    static struct usbd_device dev = {
        .base_addr = (void*)(0xdeadbeef)
    };
    return &dev;
}

int main() {
    bl_usb__setup();

    usb_g.handle = usbd_init();
    usbd_register_config(usb_g.handle, NULL);
    return 0;
}
