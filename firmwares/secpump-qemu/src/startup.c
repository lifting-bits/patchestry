/* startup.c - ARMv7-M reset & vector table for MPS2-AN386. */
#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);

void Default_Handler(void) { while (1) { } }

void Reset_Handler(void) {
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;
    for (dst = &_sbss; dst < &_ebss; ++dst) *dst = 0;
    /* No __libc_init_array(): we have no C++ ctors and no preinit_array,
     * and pulling it in would require crti.o for _init/_fini. */
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
void SysTick_Handler(void)     ALIAS(Default_Handler);

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
