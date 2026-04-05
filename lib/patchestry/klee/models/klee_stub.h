/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// Minimal KLEE header for compiling model libraries without a full KLEE install.
// Declares the subset of KLEE intrinsics used by symbolic models.

#ifndef KLEE_KLEE_H
#define KLEE_KLEE_H

#include <stddef.h>

void klee_make_symbolic(void *addr, size_t nbytes, const char *name);
void klee_assume(unsigned long long condition);
void klee_abort(void) __attribute__((noreturn));

// klee_assert is a macro in KLEE's real header (not a function), so model
// it the same way here to keep source-compatible with models compiled
// against the real KLEE runtime.
#define klee_assert(cond)                                                      \
    do {                                                                       \
        if (!(cond))                                                           \
            klee_abort();                                                      \
    } while (0)

#endif // KLEE_KLEE_H
