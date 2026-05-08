// Copyright (C) 2024 -- QEMU re-host shim for SecPump (GPLv3, derived from
// SecPump @ https://github.com/r3glisss/SecPump).
//
// This main loop replaces the upstream STM32 startup + MX_BlueNRG_MS_Process
// loop. The BlueNRG SPI/HCI radio is replaced by a UART line parser that
// synthesizes EVT_BLUE_GATT_ATTRIBUTE_MODIFIED events. From the application
// layer downward (user_notify -> Attribute_Modified_CB -> ProcessXxxReq) the
// dispatch is byte-identical to upstream.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "InsulinController.h"
#include "PumpService.h"

/* Provided by uart.c */
extern void uart_init(void);
extern int  uart_getc(void);
extern void uart_putc(char c);

/* OPERATING_MODE is referenced as extern from PumpService.c; T_iterator and
 * BolusConfig are defined in InsulinController.c. */
uint8_t OPERATING_MODE = 0;        /* 0 = Auto PID, 1 = Manual bolus */

/* Clean exit via ARM semihosting SYS_EXIT. Triggered by the 'Q' command.
 * Requires QEMU to be invoked with -semihosting; without it BKPT 0xAB will
 * fault. The default exit reason ADP_Stopped_ApplicationExit (0x20026) makes
 * QEMU exit with status 0. */
__attribute__((noreturn))
static void semihosting_exit(void) {
    register uint32_t r0 __asm__("r0") = 0x18;     /* SYS_EXIT */
    register uint32_t r1 __asm__("r1") = 0x20026;  /* ADP_Stopped_ApplicationExit */
    __asm__ volatile("bkpt 0xAB" : : "r"(r0), "r"(r1) : "memory");
    for (;;) { /* unreachable when -semihosting is enabled */ }
}

/* Read one CR/LF-terminated line from UART0 into buf (NUL-terminated). */
static size_t uart_readline(char *buf, size_t cap) {
    size_t n = 0;
    while (n + 1 < cap) {
        int c = uart_getc();
        if (c == '\r' || c == '\n') break;
        if (c == 0x08 || c == 0x7f) {                  /* backspace */
            if (n > 0) { --n; printf("\b \b"); fflush(stdout); }
            continue;
        }
        buf[n++] = (char)c;
        uart_putc((char)c);                            /* echo */
    }
    buf[n] = '\0';
    printf("\r\n");
    return n;
}

/* Decode an even-length ASCII-hex string into bytes (best-effort, max `cap`). */
static size_t hex_decode(const char *s, uint8_t *out, size_t cap) {
    size_t n = 0;
    while (s[0] && s[1] && n < cap) {
        char a = s[0], b = s[1];
        int hi = (a >= '0' && a <= '9') ? a - '0'
               : (a >= 'a' && a <= 'f') ? a - 'a' + 10
               : (a >= 'A' && a <= 'F') ? a - 'A' + 10 : -1;
        int lo = (b >= '0' && b <= '9') ? b - '0'
               : (b >= 'a' && b <= 'f') ? b - 'a' + 10
               : (b >= 'A' && b <= 'F') ? b - 'A' + 10 : -1;
        if (hi < 0 || lo < 0) break;
        out[n++] = (uint8_t)((hi << 4) | lo);
        s += 2;
    }
    return n;
}

static void banner(void) {
    printf("\r\n");
    printf("============================================================\r\n");
    printf(" SecPump QEMU re-host  (mps2-an386, Cortex-M4)\r\n");
    printf(" Derived from https://github.com/r3glisss/SecPump (GPLv3)\r\n");
    printf("------------------------------------------------------------\r\n");
    printf(" Commands:\r\n");
    printf("   M:0 | M:1          mode (0=Auto PID, 1=Manual bolus)\r\n");
    printf("   B:<float>          set manual bolus, e.g. B:1.5\r\n");
    printf("   G:<float>          inject glucose sample, e.g. G:200\r\n");
    printf("   V:<hex16>          write 16 bytes to vuln characteristic\r\n");
    printf("                      (7 calls trigger seeded overflow)\r\n");
    printf("   C[:<12 hex>]       fake LE connect (-> GAP_ConnectionComplete_CB)\r\n");
    printf("   D                  fake LE disconnect (-> GAP_DisconnectionComplete_CB)\r\n");
    printf("   R:M | R:B | R:V    fake GATT read permit (-> Read_Request_CB)\r\n");
    printf("   Q                  exit QEMU cleanly (semihosting)\r\n");
    printf("   ?                  show this banner\r\n");
    printf("============================================================\r\n");
    printf(" (Ctrl-A then x also exits QEMU; Ctrl-A then c opens the monitor.)\r\n");
}

int main(void) {
    uart_init();
    /* Prime PID tables exactly like the upstream firmware does on boot. */
    ResetController();

    /* Bring up the upstream GATT service (allocates pumpServ/mode/bolus/vuln
     * char handles via the shim's no-op aci_gatt_add_* implementations). */
    if (Add_Pump_Service() != BLE_STATUS_SUCCESS) {
        printf("[!] Add_Pump_Service failed; aborting.\r\n");
        for (;;) { /* hang */ }
    }
    /* Mirror the upstream "general discoverable" log without any radio. */
    setConnectable();

    banner();

    char  line[128];
    uint8_t arg[64];

    for (;;) {
        printf("secpump> "); fflush(stdout);
        size_t n = uart_readline(line, sizeof line);
        if (n == 0) continue;

        if (line[0] == '?') { banner(); continue; }
        /* Letter-only commands: Q (quit), C (connect, optional bdaddr),
         * D (disconnect). */
        if (line[0] == 'Q' && (n == 1 || line[1] == '\0')) {
            printf("[*] Goodbye.\r\n");
            semihosting_exit();
        }
        if (line[0] == 'D' && (n == 1 || line[1] == '\0')) {
            shim_post_disconnect();
            continue;
        }
        if (line[0] == 'C' && (n == 1 || line[1] == '\0' || line[1] == ':')) {
            uint8_t bdaddr[6] = {0xAA,0xBB,0xCC,0xDD,0xEE,0xFF};
            if (line[1] == ':') {
                uint8_t parsed[6];
                if (hex_decode(line + 2, parsed, 6) == 6) {
                    memcpy(bdaddr, parsed, 6);
                }
            }
            shim_post_le_connect(bdaddr, 0x0001);
            continue;
        }
        if (n < 2 || line[1] != ':') {
            printf("[!] bad command (need 'X:...'): %s\r\n", line);
            continue;
        }

        char  tag = line[0];
        char *val = line + 2;
        size_t val_len = strlen(val);

        switch (tag) {
        case 'M':
            /* Upstream path: GATT write to mode characteristic value handle
             * (modeCharHandle + 1). Drive Attribute_Modified_CB via the same
             * user_notify dispatch the BlueNRG stack would. */
            shim_post_attr_modified((uint16_t)(modeCharHandle + 1),
                                    (uint8_t)(val_len + 1),
                                    (const uint8_t *)val);
            break;
        case 'B':
            shim_post_attr_modified((uint16_t)(bolusCharHandle + 1),
                                    (uint8_t)(val_len + 1),
                                    (const uint8_t *)val);
            printf("[+] BolusConfig set\r\n");
            break;
        case 'G':
            /* Glucose samples come in via UART2 IRQ in upstream — never
             * through GATT. Route directly to the controller. */
            if (OPERATING_MODE == 0) {
                InsulinController((uint8_t *)val);
            } else {
                InsulinManualController((uint8_t *)val);
            }
            break;
        case 'V': {
            size_t got = hex_decode(val, arg, sizeof arg);
            if (got < 16) {
                printf("[!] V: needs 32 hex chars (16 bytes), got %u\r\n",
                       (unsigned)got);
                break;
            }
            shim_post_attr_modified((uint16_t)(vulnCharHandle + 1),
                                    (uint8_t)16, arg);
            break;
        }
        case 'R': {
            /* Drive Read_Request_CB. Map the friendly tag to the char value
             * handle the upstream service registered. */
            uint16_t handle;
            switch (val[0]) {
            case 'M': handle = (uint16_t)(modeCharHandle  + 1); break;
            case 'B': handle = (uint16_t)(bolusCharHandle + 1); break;
            case 'V': handle = (uint16_t)(vulnCharHandle  + 1); break;
            default:
                printf("[!] R: expected R:M, R:B, or R:V\r\n");
                continue;
            }
            shim_post_gatt_read_permit(handle);
            break;
        }
        default:
            printf("[!] unknown tag '%c'\r\n", tag);
        }
    }
}
