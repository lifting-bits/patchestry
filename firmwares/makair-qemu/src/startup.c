/* startup.c — ARMv7-M reset + vector table for MPS2-AN386, with SysTick
 * driving a 1 ms tick that the Arduino shim's millis()/micros() reads. */
#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);

volatile uint32_t systick_ms = 0;

void Default_Handler(void) { while (1) { } }

void SysTick_Handler(void) { systick_ms += 1u; }

/* mps2-an386 system clock is 25 MHz. SysTick reload of 25000 -> 1 kHz. */
#define SYST_CSR   (*(volatile uint32_t *)0xE000E010UL)
#define SYST_RVR   (*(volatile uint32_t *)0xE000E014UL)
#define SYST_CVR   (*(volatile uint32_t *)0xE000E018UL)

void Reset_Handler(void) {
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;
    for (dst = &_sbss; dst < &_ebss; ++dst) *dst = 0;

    /* SysTick: reload = 25_000 - 1, processor clock, exception enabled, run. */
    SYST_RVR = 25000u - 1u;
    SYST_CVR = 0u;
    SYST_CSR = 0x7u;  /* CLKSOURCE | TICKINT | ENABLE */

    (void)main();
    while (1) { }
}

#define ALIAS(x) __attribute__((weak, alias(#x)))
void NMI_Handler(void)         ALIAS(Default_Handler);
void HardFault_Handler(void)   ALIAS(Default_Handler);
void MemManage_Handler(void)   ALIAS(Default_Handler);
void BusFault_Handler(void)    ALIAS(Default_Handler);
void UsageFault_Handler(void)  ALIAS(Default_Handler);
void SVC_Handler(void)         ALIAS(Default_Handler);
void DebugMon_Handler(void)    ALIAS(Default_Handler);
void PendSV_Handler(void)      ALIAS(Default_Handler);

__attribute__((section(".isr_vector"), used))
void (* const g_pfnVectors[])(void) = {
    (void (*)(void))(&_estack),
    Reset_Handler,
    NMI_Handler,
    HardFault_Handler,
    MemManage_Handler,
    BusFault_Handler,
    UsageFault_Handler,
    0, 0, 0, 0,
    SVC_Handler,
    DebugMon_Handler,
    0,
    PendSV_Handler,
    SysTick_Handler,
};
