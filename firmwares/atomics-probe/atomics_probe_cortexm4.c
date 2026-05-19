/*
 * Minimal Cortex-M4 probe binary for grounded atomic-intrinsic tests.
 *
 * Each public function executes one ARMv7-M synchronization instruction
 * (DMB/DSB/ISB/CLREX) or LL/SC primitive (LDREX/STREX) so that Ghidra
 * exposes the corresponding CALLOTHER userop (DataMemoryBarrier, etc.) in
 * the serialized P-Code.
 */

#include <stdint.h>

volatile uint32_t shared;

void __attribute__((used)) probe_dmb(void) {
    __asm__ volatile("dmb" ::: "memory");
}

void __attribute__((used)) probe_dsb(void) {
    __asm__ volatile("dsb" ::: "memory");
}

void __attribute__((used)) probe_isb(void) {
    __asm__ volatile("isb" ::: "memory");
}

void __attribute__((used)) probe_clrex(void) {
    __asm__ volatile("clrex" ::: "memory");
}

uint32_t __attribute__((used)) probe_ldrex(volatile uint32_t *p) {
    uint32_t v;
    __asm__ volatile("ldrex %0, [%1]" : "=r"(v) : "r"(p));
    return v;
}

uint32_t __attribute__((used)) probe_strex(volatile uint32_t *p, uint32_t v) {
    uint32_t status;
    __asm__ volatile("strex %0, %2, [%1]" : "=&r"(status) : "r"(p), "r"(v));
    return status;
}

uint32_t __attribute__((used)) probe_ll_sc_increment(volatile uint32_t *p) {
    uint32_t old, status;
    do {
        __asm__ volatile("ldrex %0, [%1]" : "=r"(old) : "r"(p));
        __asm__ volatile("strex %0, %2, [%1]"
                         : "=&r"(status)
                         : "r"(p), "r"(old + 1));
    } while (status != 0);
    __asm__ volatile("dmb" ::: "memory");
    return old;
}
