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
 * Device registers are the address ranges of `__patchestry_mmio`.  The
 * harness asks before every load and store that may reach one: a write is
 * recorded, in order, and not performed; a read returns the next entry of
 * `__patchestry_mmio_values`, the same series for both versions.  Calls of a
 * declared acquire and release pair are counted per version.
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
#define MAX_WRITES 64
#define MAX_PAIRS 8
#define EFFECT 0x40000000 /* id bit of a callee allowed before a rejection */
#define SKIP 0x20000000   /* id bit of a callee the patched version may not call */

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

struct range {
    uintptr_t low;
    uintptr_t high;
};

extern const struct range __patchestry_mmio[];
extern const int __patchestry_mmio_count;
extern unsigned long long __patchestry_mmio_values[];
extern const int __patchestry_mmio_value_count;

struct write {
    uintptr_t address;
    unsigned long long value;
    int size;
};

struct device {
    int writes;
    int reads;
    int overflow;
    struct write write[MAX_WRITES];
};

static struct device devices[2];
static int depth[2][MAX_PAIRS];

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

static int side(void) { return __patchestry_phase == 1 ? 0 : __patchestry_phase == 2 ? 1 : -1; }

static int in_device(uintptr_t address) {
    for (int at = 0; at < __patchestry_mmio_count; at++) {
        if (address >= __patchestry_mmio[at].low && address <= __patchestry_mmio[at].high) {
            return 1;
        }
    }
    return 0;
}

/* Nonzero when the address is a device register: the store is not performed. */
int __patchestry_mmio_write(uintptr_t address, unsigned long long value, int size) {
    if (!in_device(address)) {
        return 0;
    }
    if (side() < 0) {
        return 1;
    }
    struct device *device = &devices[side()];
    if (device->writes >= MAX_WRITES) {
        device->overflow = 1;
        return 1;
    }
    if (size < 8) {
        value &= (1ULL << (8 * size)) - 1;
    }
    struct write *write = &device->write[device->writes++];
    write->address      = address;
    write->value        = value;
    write->size         = size;
    return 1;
}

/* Nonzero when the address is a device register: `*value` is what was read. */
int __patchestry_mmio_read(uintptr_t address, int size, unsigned long long *value) {
    (void) size;
    if (!in_device(address)) {
        return 0;
    }
    int index = 0;
    if (side() >= 0) {
        index = devices[side()].reads++;
    }
    if (index >= __patchestry_mmio_value_count) {
        index = __patchestry_mmio_value_count - 1;
    }
    *value = __patchestry_mmio_values[index];
    return 1;
}

int __patchestry_mmio_differs(void) {
    const struct device *original = &devices[0];
    const struct device *patched  = &devices[1];
    if (original->writes != patched->writes) {
        return 1;
    }
    for (int at = 0; at < original->writes; at++) {
        const struct write *ours   = &original->write[at];
        const struct write *theirs = &patched->write[at];
        if (ours->address != theirs->address || ours->value != theirs->value
            || ours->size != theirs->size) {
            return 1;
        }
    }
    return 0;
}

void __patchestry_balance(int pair, int delta) {
    if (side() >= 0 && pair >= 0 && pair < MAX_PAIRS) {
        depth[side()][pair] += delta;
    }
}

int __patchestry_balance_differs(void) {
    for (int at = 0; at < MAX_PAIRS; at++) {
        if (depth[0][at] != depth[1][at]) {
            return 1;
        }
    }
    return 0;
}

/* What the patched version did before it rejected the input. */
int __patchestry_reject_calls(void) {
    for (int at = 0; at < traces[1].calls; at++) {
        if ((traces[1].call[at].callee & EFFECT) == 0) {
            __patchestry_first_call = at;
            return 1;
        }
    }
    return traces[1].overflow;
}

int __patchestry_reject_device(void) { return devices[1].writes != 0 || devices[1].overflow; }

int __patchestry_reject_unbalanced(void) {
    for (int at = 0; at < MAX_PAIRS; at++) {
        if (depth[1][at] != 0) {
            return 1;
        }
    }
    return 0;
}

void __patchestry_compare_reset(void) {
    for (int which = 0; which < 2; which++) {
        devices[which].writes = devices[which].reads = devices[which].overflow = 0;
        for (int at = 0; at < MAX_PAIRS; at++) {
            depth[which][at] = 0;
        }
    }
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

static int same_call(const struct call *ours, const struct call *theirs) {
    if (ours->callee != theirs->callee || ours->count != theirs->count) {
        return 0;
    }
    for (int arg = 0; arg < ours->count; arg++) {
        if (ours->kind[arg] != theirs->kind[arg] || ours->value[arg] != theirs->value[arg]) {
            return 0;
        }
    }
    return 1;
}

/*
 * 0: the same calls; 1: they differ; 2: a trace did not hold every call;
 * 3: the same, except that the patched version left out calls of a callee
 * declared for that.
 */
int __patchestry_trace_differs(void) {
    const struct trace *original = &traces[0];
    const struct trace *patched  = &traces[1];
    if (original->overflow || patched->overflow || devices[0].overflow || devices[1].overflow) {
        return 2;
    }
    int ours = 0, theirs = 0, skipped = 0;
    while (ours < original->calls || theirs < patched->calls) {
        if (ours < original->calls && theirs < patched->calls
            && same_call(&original->call[ours], &patched->call[theirs])) {
            ours++;
            theirs++;
        } else if (ours < original->calls && (original->call[ours].callee & SKIP) != 0) {
            ours++;
            skipped = 1;
        } else {
            __patchestry_first_call = ours;
            return 1;
        }
    }
    return skipped ? 3 : 0;
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
