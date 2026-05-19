/*
 * x86-64 freestanding probe binary for grounded atomic-intrinsic tests.
 *
 * Every public function executes one x86-64 synchronization instruction
 * (MFENCE / LFENCE / SFENCE / PAUSE) or LOCK-prefixed RMW (XADD / CMPXCHG /
 * XCHG / INC / DEC / OR) so that stock Ghidra emits the corresponding
 * CALLOTHER userops in the serialized P-Code. The Intel/AMD x86 SLEIGH
 * exposes only LOCK / UNLOCK as bare CALLOTHER userops surrounding the
 * RMW body; the fence and pause instructions lower to inline P-Code.
 */

#include <stdint.h>

volatile uint32_t shared32;

void __attribute__((used)) probe_mfence(void) {
    __asm__ volatile("mfence" ::: "memory");
}

void __attribute__((used)) probe_lfence(void) {
    __asm__ volatile("lfence" ::: "memory");
}

void __attribute__((used)) probe_sfence(void) {
    __asm__ volatile("sfence" ::: "memory");
}

void __attribute__((used)) probe_pause(void) {
    __asm__ volatile("pause" ::: "memory");
}

uint32_t __attribute__((used)) probe_lock_xadd(volatile uint32_t *p, uint32_t v) {
    uint32_t old = v;
    __asm__ volatile("lock xaddl %0, %1"
                     : "+r"(old), "+m"(*p)
                     :
                     : "memory");
    return old;
}

uint32_t __attribute__((used))
probe_lock_cmpxchg(volatile uint32_t *p, uint32_t expected, uint32_t desired) {
    uint32_t prev;
    __asm__ volatile("lock cmpxchgl %2, %1"
                     : "=a"(prev), "+m"(*p)
                     : "r"(desired), "0"(expected)
                     : "memory");
    return prev;
}

uint32_t __attribute__((used)) probe_lock_xchg(volatile uint32_t *p, uint32_t v) {
    /* XCHG with a memory operand is implicitly locked on x86-64. */
    __asm__ volatile("xchgl %0, %1" : "+r"(v), "+m"(*p) : : "memory");
    return v;
}

void __attribute__((used)) probe_lock_inc(volatile uint32_t *p) {
    __asm__ volatile("lock incl %0" : "+m"(*p) : : "memory");
}

void __attribute__((used)) probe_lock_dec(volatile uint32_t *p) {
    __asm__ volatile("lock decl %0" : "+m"(*p) : : "memory");
}

void __attribute__((used)) probe_lock_or(volatile uint32_t *p, uint32_t v) {
    __asm__ volatile("lock orl %1, %0" : "+m"(*p) : "ir"(v) : "memory");
}
