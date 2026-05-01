/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// Symbolic models for libc functions used by the bloodview KLEE demo.
// The sprintf/vsnprintf models return a symbolic length capped at 1024 and
// stamp a NUL terminator, mirroring real-write semantics so out-of-bounds
// returns surface as concrete violations of any contract bounding the
// destination size. bl_device__match returns a symbolic status and, on the
// success path, allocates a heap buffer for the output handle.

#include "klee_stub.h"
#include <stdint.h>

extern void *malloc(size_t size);

int vsnprintf(char *dest, size_t size, const char *fmt, void *ap)
{
    (void) fmt;
    (void) ap;

    int n;
    klee_make_symbolic(&n, sizeof(n), "vsnprintf_n");
    klee_assume(n >= 0);
    klee_assume(n <= 1024);

    if (size == 0) {
        return n;
    }

    size_t writelen = ((size_t) n < size - 1) ? (size_t) n : size - 1;
    dest[writelen] = '\0';
    return (int) writelen;
}

int sprintf(char *dest, const char *fmt, ...)
{
    (void) fmt;

    int n;
    klee_make_symbolic(&n, sizeof(n), "sprintf_n");
    klee_assume(n >= 0);
    klee_assume(n <= 1024);

    dest[n] = '\0';
    return n;
}

char bl_device__match(unsigned int id, void **out_handle)
{
    (void) id;

    char ret;
    klee_make_symbolic(&ret, sizeof(ret), "bl_device__match_ret");
    if (ret == 1) {
        *out_handle = malloc(64);
    } else {
        *out_handle = (void *) 0;
    }
    return ret;
}
