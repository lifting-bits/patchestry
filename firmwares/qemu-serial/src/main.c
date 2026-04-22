#include "targets.h"

#include "uart.h"

int main(void) {
    uart_write_line("BOOT");
    qemu_target_before();
    qemu_target_after();
    qemu_target_replace();
    qemu_target_contract();
    uart_write_line("DONE");

    return 0;
}
