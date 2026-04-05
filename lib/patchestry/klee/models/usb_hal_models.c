/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// Symbolic models for USB HAL functions.
// Provides constrained symbolic return values and memory effects
// for hardware abstraction layer functions used in USB firmware.

#include "klee_stub.h"
#include <stdint.h>

// Model: usbd_ep_write_packet
//   Memory effect: makes buffer contents symbolic (models HAL writing response)
//   Return: symbolic value constrained to [0, 255] (valid USB packet length)
int32_t usbd_ep_write_packet(void *dev, void *buf, int32_t len) {
    (void) dev;
    (void) len;
    if (buf)
        klee_make_symbolic(buf, 256, "ep_write_buf");
    int32_t ret;
    klee_make_symbolic(&ret, sizeof(ret), "ep_write_ret");
    klee_assume(ret >= 0);
    klee_assume(ret <= 255);
    return ret;
}
