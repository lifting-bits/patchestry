/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

/*
 * What a product harness (patchir-klee-verifier --compare-original) needs
 * from its runner when the runner is KLEE.  Link this, and compare.c and
 * models.c of scripts/patch-runtime/equivalence_runtime, with the harness.
 *
 * A path that found a violation ends in klee_abort, called from a function
 * named after the check that failed: KLEE writes an error file for the path,
 * and its stack names that function.  scripts/klee/klee_gate.py reads the
 * check from there.  Any other path ends with exit(0).
 *
 * Not run under KLEE yet; see docs/klee-integration.md.
 */

#include <stdint.h>

void exit(int status) __attribute__((noreturn));

#define FAIL(name) __attribute__((noinline)) void __patchestry_fail_##name(void) { klee_abort(); }

FAIL(p1)
FAIL(p2)
FAIL(p3)
FAIL(p4)
FAIL(mmio_order)
FAIL(lock_balance)
FAIL(reject_hygiene)
/* Not a violation: the harness could not model something on this path. */
FAIL(unmodelled)

/* The codes are those of tools/patchir-klee-verifier/Compare.cpp. */
void __patchestry_path_end(int code) {
    if (code == 10) {
        __patchestry_fail_p1();
    }
    if (code == 20) {
        __patchestry_fail_unmodelled();
    }
    if (code > 128) {
        if (code & (1 | 2 | 4)) {
            __patchestry_fail_reject_hygiene();
        }
        __patchestry_fail_lock_balance();
    }
    if (code > 64 && code < 128) {
        if (code & 2) {
            __patchestry_fail_p2();
        }
        if (code & 1) {
            __patchestry_fail_p3();
        }
        if (code & 4) {
            __patchestry_fail_p4();
        }
        if (code & 8) {
            __patchestry_fail_mmio_order();
        }
        __patchestry_fail_lock_balance();
    }
    exit(0);
}

/* Each allocation of a KLEE state is its own object; there is no heap to rewind. */
void __patchestry_side_begin(void) { }

/*
 * KLEE's locals have no address range to test against.  The address of a
 * local passed to an external function is therefore compared as a number,
 * and differs between the two versions: such a path fails P4 under KLEE
 * where the concrete runner passes it.
 */
int __patchestry_stack_address(uintptr_t address) {
    (void) address;
    return 0;
}
