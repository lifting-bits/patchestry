/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

/*
 * Runs a product harness (patchir-klee-verifier --compare-original) on
 * concrete inputs, as ARM code under QEMU.  It stands in for KLEE: the
 * harness's `klee_make_symbolic` gets bytes derived from a seed, one seed per
 * input, and every path end is printed.
 *
 * One line per input on the semihosting console:
 *
 *     I <seed> end <code>      the harness ended the path with <code>
 *     I <seed> fault <phase>   an exception while <phase> ran (1 original, 2 patched)
 *     I <seed> abort <phase>   an abort-like call while <phase> ran
 *     I <seed> discard 0       klee_assume was false for this input
 *     I <seed> memory <n>      before `end`: memory region <n> differs, or was written
 *     I <seed> call <n>        before `end`: the <n>-th calls differ, or is not allowed
 *
 * After a fault or abort in the patched version the same input is run again
 * with the original alone; that line carries the same seed and code 30 or 31,
 * or a fault or abort in phase 1.
 *
 * FIRST_SEED and SEED_COUNT are given at compile time.
 */

#include <stddef.h>
#include <stdint.h>

extern int main(void);
extern int __patchestry_phase;
extern int __patchestry_only_original;
extern char __harness_data_start[], __harness_data_end[], __harness_snapshot[];
extern char __heap_start[], __heap_end[], __stack_top[];
extern int __patchestry_first_memory, __patchestry_first_call; /* compare.c */
void __patchestry_compare_reset(void);
void runtime_resume(void) __attribute__((noreturn));

static unsigned next_seed = FIRST_SEED;
static unsigned current_seed;
static int rerun_original;
static int in_rerun;
static uint32_t random_state;
static char *heap_next;
static char *side_heap;

static int semihost(int operation, void *argument) {
    register int r0 __asm__("r0")   = operation;
    register void *r1 __asm__("r1") = argument;
    __asm__ volatile("svc 0x123456" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

static void put(const char *text) { semihost(0x04, (void *) text); }

static void put_number(unsigned value) {
    char digits[12];
    int at     = 11;
    digits[at] = 0;
    do {
        digits[--at] = (char) ('0' + value % 10);
        value /= 10;
    } while (value != 0);
    put(&digits[at]);
}

static void report(const char *kind, unsigned detail) {
    put("I ");
    put_number(current_seed);
    put(" ");
    put(kind);
    put(" ");
    put_number(detail);
    put("\n");
}

void *memcpy(void *to, const void *from, size_t count) {
    char *out      = to;
    const char *in = from;
    while (count-- != 0) {
        *out++ = *in++;
    }
    return to;
}

void *memmove(void *to, const void *from, size_t count) {
    char *out      = to;
    const char *in = from;
    if (out < in) {
        while (count-- != 0) {
            *out++ = *in++;
        }
    } else {
        while (count-- != 0) {
            out[count] = in[count];
        }
    }
    return to;
}

void *memset(void *to, int value, size_t count) {
    char *out = to;
    while (count-- != 0) {
        *out++ = (char) value;
    }
    return to;
}

void __aeabi_memcpy(void *to, const void *from, size_t count) { memcpy(to, from, count); }
void __aeabi_memcpy4(void *to, const void *from, size_t count) { memcpy(to, from, count); }
void __aeabi_memcpy8(void *to, const void *from, size_t count) { memcpy(to, from, count); }
void __aeabi_memmove(void *to, const void *from, size_t count) { memmove(to, from, count); }
void __aeabi_memmove4(void *to, const void *from, size_t count) { memmove(to, from, count); }
void __aeabi_memmove8(void *to, const void *from, size_t count) { memmove(to, from, count); }
void __aeabi_memset(void *to, size_t count, int value) { memset(to, value, count); }
void __aeabi_memset4(void *to, size_t count, int value) { memset(to, value, count); }
void __aeabi_memset8(void *to, size_t count, int value) { memset(to, value, count); }
void __aeabi_memclr(void *to, size_t count) { memset(to, 0, count); }
void __aeabi_memclr4(void *to, size_t count) { memset(to, 0, count); }
void __aeabi_memclr8(void *to, size_t count) { memset(to, 0, count); }

void *malloc(size_t size) {
    size = (size + 7U) & ~(size_t) 7U;
    if (size > (size_t) (__heap_end - heap_next)) {
        return NULL;
    }
    void *block = heap_next;
    heap_next += size;
    return block;
}

void free(void *block) { (void) block; }

/*
 * Called by the harness before each version runs.  Both get the same heap,
 * so a pointer one of them allocates and returns equals the other's.
 */
void __patchestry_side_begin(void) {
    if (__patchestry_phase == 2 || __patchestry_only_original) {
        side_heap = heap_next;
    } else {
        heap_next = side_heap;
    }
}

int __patchestry_stack_address(uintptr_t address) {
    return address >= (uintptr_t) __heap_end && address < (uintptr_t) __stack_top;
}

static uint32_t random_word(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}

/*
 * The bytes of one "symbolic" object.  Shapes that reach the branches of
 * checking code: zero, small numbers, values next to a power of two, all
 * ones, text, and noise.
 */
void klee_make_symbolic(void *address, size_t size, const char *name) {
    unsigned char *bytes = address;
    uint32_t shape       = random_word() % 8;
    (void) name;
    memset(bytes, 0, size);
    if (size <= 8) {
        uint32_t value = 0;
        switch (shape) {
            case 0: value = 0; break;
            case 1: value = random_word() % 16; break;
            case 2: value = random_word() % 40; break;
            case 3: value = random_word() % 300; break;
            case 4: value = (1U << (random_word() % 32)) + (random_word() % 3) - 1; break;
            case 5: value = 0U - (random_word() % 16); break;
            case 6: value = 0xffffffffU; break;
            default: value = random_word(); break;
        }
        for (size_t at = 0; at < size; at++) {
            bytes[at] = (unsigned char) (at < 4 ? value >> (8 * at) : (shape >= 5 ? 0xff : 0));
        }
        return;
    }
    switch (shape) {
        case 0: break;
        case 1: memset(bytes, 0xff, size); break;
        case 2:
        case 3:
        case 4: { /* text, ended inside the buffer */
            static const char alphabet[] = "abcXYZ019 /:.-_=&?%\r\n\t";
            size_t length                = random_word() % size;
            for (size_t at = 0; at < length; at++) {
                bytes[at] = (unsigned char) alphabet[random_word() % (sizeof alphabet - 1)];
            }
            break;
        }
        case 5: /* small numbers */
            for (size_t at = 0; at < size; at++) {
                bytes[at] = (unsigned char) (random_word() % 16);
            }
            break;
        default: /* noise, ended by a zero byte */
            for (size_t at = 0; at + 1 < size; at++) {
                bytes[at] = (unsigned char) random_word();
            }
            break;
    }
}

void klee_assume(uintptr_t condition) {
    if (condition == 0) {
        report("discard", 0);
        runtime_resume();
    }
}

static void stopped(const char *kind) __attribute__((noreturn));
static void stopped(const char *kind) {
    unsigned phase = (unsigned) __patchestry_phase;
    report(kind, phase);
    /* Whether the original handles an input the patched version died on. */
    rerun_original = phase == 2 && !in_rerun;
    runtime_resume();
}

void klee_abort(void) { stopped("abort"); }

void runtime_fault(void) { stopped("fault"); }

void __patchestry_path_end(int code) {
    if (code >= 64 && __patchestry_first_memory >= 0) {
        report("memory", (unsigned) __patchestry_first_memory);
    }
    if (code >= 64 && __patchestry_first_call >= 0) {
        report("call", (unsigned) __patchestry_first_call);
    }
    report("end", (unsigned) code);
    runtime_resume();
}

void runtime_loop(void) {
    for (;;) {
        if (rerun_original) {
            rerun_original = 0;
            in_rerun       = 1;
        } else {
            in_rerun = 0;
            if (next_seed >= FIRST_SEED + SEED_COUNT) {
                break;
            }
            current_seed = next_seed++;
        }
        memcpy(__harness_data_start, __harness_snapshot,
               (size_t) (__harness_data_end - __harness_data_start));
        __patchestry_only_original = in_rerun;
        heap_next                  = __heap_start;
        __patchestry_compare_reset();
        random_state               = current_seed * 2654435761U + 0x9e3779b9U;
        if (random_state == 0) {
            random_state = 1;
        }
        main();
        report("end", 99); /* the harness never returns */
    }
    put("DONE\n");
    semihost(0x18, (void *) 0x20026);
    for (;;) { }
}

void runtime_main(void) {
    memcpy(__harness_snapshot, __harness_data_start,
           (size_t) (__harness_data_end - __harness_data_start));
    runtime_loop();
}
