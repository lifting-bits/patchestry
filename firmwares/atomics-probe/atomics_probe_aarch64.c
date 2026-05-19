/*
 * AArch64 freestanding probe binary for grounded atomic-intrinsic tests.
 *
 * Every public function executes one AArch64 synchronization instruction
 * (DMB / DSB / ISB / CLREX) or a single LDXR / LDAR that exercises the
 * exclusive-monitor and load-acquire SLEIGH userops. STXR / STLXR are
 * intentionally omitted: stock Ghidra returns their status flag through a
 * CAST chain that the current ClangIR pipeline cannot lower
 * ("Not Yet Implemented: bitcast on function return value"). Reintroduce
 * them once that limitation is lifted.
 */

#include <stdint.h>

volatile uint32_t shared32;

void __attribute__((used)) probe_dmb(void) {
    __asm__ volatile("dmb sy" ::: "memory");
}

void __attribute__((used)) probe_dsb(void) {
    __asm__ volatile("dsb sy" ::: "memory");
}

void __attribute__((used)) probe_isb(void) {
    __asm__ volatile("isb" ::: "memory");
}

void __attribute__((used)) probe_clrex(void) {
    __asm__ volatile("clrex" ::: "memory");
}

uint32_t __attribute__((used)) probe_ldxr(volatile uint32_t *p) {
    uint32_t v;
    __asm__ volatile("ldxr %w0, [%1]" : "=r"(v) : "r"(p));
    return v;
}

uint32_t __attribute__((used)) probe_ldaxr(volatile uint32_t *p) {
    uint32_t v;
    __asm__ volatile("ldaxr %w0, [%1]" : "=r"(v) : "r"(p));
    return v;
}

uint32_t __attribute__((used)) probe_ldar(volatile uint32_t *p) {
    uint32_t v;
    __asm__ volatile("ldar %w0, [%1]" : "=r"(v) : "r"(p));
    return v;
}
