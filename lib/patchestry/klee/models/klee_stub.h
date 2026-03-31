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
void klee_assert(int condition);

#endif // KLEE_KLEE_H
