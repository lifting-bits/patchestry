/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

/*
 * What a product harness (patchir-klee-verifier --compare-original) calls to
 * compare the two versions beyond their return values: the calls each made
 * to external functions, and the memory each left behind.
 *
 * Each version has its own copy of every global, so the address of a global
 * differs between the two.  An argument that falls inside a copy is compared
 * as (global, offset); `__patchestry_pairs` lists the copies.  An argument
 * that points into constant data is compared by the bytes from there to the
 * end of the object, so equal string literals of the two modules are equal.
 * The address of a local is compared only as "a local": the two versions lay out their
 * frames differently, and what the local holds is not looked at.
 *
 * Nothing here depends on the machine: `__patchestry_stack_address` is the
 * runner's.
 */

#include <stddef.h>
#include <stdint.h>

#define MAX_CALLS 64
#define MAX_ARGS 8
#define KIND_PLAIN 0U
#define KIND_LOCAL 0xffffffffU
#define KIND_CONSTANT 0xfffffffeU

struct pair {
    const char *original;
    const char *patched;
    uintptr_t size;
};

extern const struct pair __patchestry_pairs[];
extern const int __patchestry_pair_count;

struct constant {
    const unsigned char *base;
    uintptr_t size;
};

extern const struct constant __patchestry_constants[];
extern const int __patchestry_constant_count;
extern int __patchestry_phase; /* 1 original, 2 patched */
extern int __patchestry_stack_address(uintptr_t address);

struct call {
    int callee;
    int count;
    int filled;
    uintptr_t kind[MAX_ARGS];
    uintptr_t value[MAX_ARGS];
};

struct trace {
    int calls;
    int overflow;
    struct call call[MAX_CALLS];
};

static struct trace traces[2];

/* Where the first difference was found, for the runner's report. */
int __patchestry_first_memory = -1;
int __patchestry_first_call   = -1;

static struct trace *current(void) {
    if (__patchestry_phase == 1) {
        return &traces[0];
    }
    if (__patchestry_phase == 2) {
        return &traces[1];
    }
    return NULL;
}

void __patchestry_compare_reset(void) {
    traces[0].calls = traces[0].overflow = 0;
    traces[1].calls = traces[1].overflow = 0;
    __patchestry_first_memory           = -1;
    __patchestry_first_call             = -1;
}

void __patchestry_trace_call(int callee, int count) {
    struct trace *trace = current();
    if (trace == NULL) {
        return;
    }
    if (trace->calls >= MAX_CALLS || count > MAX_ARGS) {
        trace->overflow = 1;
        return;
    }
    struct call *call = &trace->call[trace->calls++];
    call->callee      = callee;
    call->count       = count;
    call->filled      = 0;
}

void __patchestry_trace_arg(uintptr_t value) {
    struct trace *trace = current();
    if (trace == NULL || trace->overflow || trace->calls == 0) {
        return;
    }
    struct call *call = &trace->call[trace->calls - 1];
    if (call->filled >= call->count) {
        return;
    }
    uintptr_t kind = KIND_PLAIN;
    for (int at = 0; at < __patchestry_constant_count; at++) {
        const struct constant *constant = &__patchestry_constants[at];
        uintptr_t offset                = value - (uintptr_t) constant->base;
        if (offset < constant->size) {
            uintptr_t hash = 2166136261U;
            for (; offset < constant->size; offset++) {
                hash = (hash ^ constant->base[offset]) * 16777619U;
            }
            kind  = KIND_CONSTANT;
            value = hash;
            break;
        }
    }
    for (int at = 0; kind == KIND_PLAIN && at < __patchestry_pair_count; at++) {
        const struct pair *pair = &__patchestry_pairs[at];
        uintptr_t base =
            (uintptr_t) (__patchestry_phase == 1 ? pair->original : pair->patched);
        if (value - base < pair->size) {
            kind  = (uintptr_t) at + 1;
            value = value - base;
            break;
        }
    }
    if (kind == KIND_PLAIN && __patchestry_stack_address(value)) {
        kind  = KIND_LOCAL;
        value = 0;
    }
    call->kind[call->filled]  = kind;
    call->value[call->filled] = value;
    call->filled++;
}

/* 0: the same calls; 1: they differ; 2: a trace did not hold every call. */
int __patchestry_trace_differs(void) {
    const struct trace *original = &traces[0];
    const struct trace *patched  = &traces[1];
    if (original->overflow || patched->overflow) {
        return 2;
    }
    int shared = original->calls < patched->calls ? original->calls : patched->calls;
    for (int at = 0; at < shared; at++) {
        const struct call *ours   = &original->call[at];
        const struct call *theirs = &patched->call[at];
        int differs               = ours->callee != theirs->callee || ours->count != theirs->count;
        for (int arg = 0; !differs && arg < ours->count; arg++) {
            differs = ours->kind[arg] != theirs->kind[arg] || ours->value[arg] != theirs->value[arg];
        }
        if (differs) {
            __patchestry_first_call = at;
            return 1;
        }
    }
    if (original->calls != patched->calls) {
        __patchestry_first_call = shared;
        return 1;
    }
    return 0;
}

int __patchestry_memory_differs(const char *original, const char *patched, uintptr_t size,
                                int region) {
    for (uintptr_t at = 0; at < size; at++) {
        if (original[at] != patched[at]) {
            if (__patchestry_first_memory < 0) {
                __patchestry_first_memory = region;
            }
            return 1;
        }
    }
    return 0;
}
