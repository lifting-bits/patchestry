/* uart.c - CMSDK APB UART0 driver + newlib syscall glue for QEMU mps2-an386. */
#include <stdint.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>

typedef struct {
    volatile uint32_t DATA;
    volatile uint32_t STATE;
    volatile uint32_t CTRL;
    volatile uint32_t INTSTATUS;
    volatile uint32_t BAUDDIV;
} cmsdk_uart_t;

#define UART0 ((cmsdk_uart_t *)0x40004000UL)
#define UART_STATE_TX_FULL  (1u << 0)
#define UART_STATE_RX_FULL  (1u << 1)
#define UART_CTRL_TX_EN     (1u << 0)
#define UART_CTRL_RX_EN     (1u << 1)

void uart_init(void) {
    UART0->BAUDDIV = 16;          /* QEMU ignores baud, just needs >=16. */
    UART0->CTRL    = UART_CTRL_TX_EN | UART_CTRL_RX_EN;
}

void uart_putc(char c) {
    while (UART0->STATE & UART_STATE_TX_FULL) { }
    UART0->DATA = (uint32_t)c;
}

int uart_getc_nonblock(void) {
    if (!(UART0->STATE & UART_STATE_RX_FULL)) return -1;
    return (int)(UART0->DATA & 0xFF);
}

int uart_getc(void) {
    while (!(UART0->STATE & UART_STATE_RX_FULL)) { }
    return (int)(UART0->DATA & 0xFF);
}

/* --- newlib syscall stubs --- */
extern char _heap_start;
static char *heap_end = 0;

void *_sbrk(int incr) {
    if (heap_end == 0) heap_end = &_heap_start;
    char *prev = heap_end;
    heap_end += incr;
    return prev;
}

int _write(int fd, const char *buf, int len) {
    (void)fd;
    for (int i = 0; i < len; ++i) {
        if (buf[i] == '\n') uart_putc('\r');
        uart_putc(buf[i]);
    }
    return len;
}

int _read(int fd, char *buf, int len) {
    (void)fd;
    if (len <= 0) return 0;
    buf[0] = (char)uart_getc();
    return 1;
}

int _close(int fd)                 { (void)fd; return -1; }
int _fstat(int fd, struct stat *s) { (void)fd; s->st_mode = S_IFCHR; return 0; }
int _isatty(int fd)                { (void)fd; return 1; }
int _lseek(int fd, int o, int w)   { (void)fd; (void)o; (void)w; return 0; }
int _getpid(void)                  { return 1; }
int _kill(int pid, int sig)        { (void)pid; (void)sig; errno = EINVAL; return -1; }
void _exit(int status)             { (void)status; while (1) { } }
