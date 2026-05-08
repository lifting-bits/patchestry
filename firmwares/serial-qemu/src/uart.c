#include "uart.h"

#include <stdint.h>

enum {
    UART0_BASE = 0x4000C000u,
    UARTDR     = 0x000u,
};

static volatile uint32_t *uart_reg(uint32_t offset) {
    return (volatile uint32_t *)(uintptr_t)(UART0_BASE + offset);
}

void uart_write_char(char ch) {
    *uart_reg(UARTDR) = (uint32_t)(unsigned char)ch;
}

void uart_write_string(const char *text) {
    while (*text != '\0') {
        uart_write_char(*text++);
    }
}

void uart_write_line(const char *text) {
    uart_write_string(text);
    uart_write_string("\r\n");
}
