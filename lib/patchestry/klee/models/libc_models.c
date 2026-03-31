/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// Symbolic models for common libc functions.
// Each function provides symbolic output constraints and memory effects
// suitable for KLEE symbolic execution.

#include "klee_stub.h"
#include <stdint.h>

// --- Memory ---

void *memcpy(void *dst, const void *src, size_t n) {
    if (dst && n > 0)
        klee_make_symbolic(dst, n, "memcpy_dst");
    return dst;
}

void *memset(void *s, int c, size_t n) {
    unsigned char *p = (unsigned char *) s;
    for (size_t i = 0; i < n; i++)
        p[i] = (unsigned char) c;
    return s;
}

void *malloc(size_t size) {
    static unsigned char pool[4096];
    static size_t offset = 0;
    if (offset + size > sizeof(pool))
        return (void *) 0;
    void *ptr = &pool[offset];
    offset += size;
    klee_make_symbolic(ptr, size, "malloc_buf");
    return ptr;
}

void free(void *ptr) {
    (void) ptr;
}

// --- String ---

size_t strlen(const char *s) {
    (void) s;
    size_t len;
    klee_make_symbolic(&len, sizeof(len), "strlen_ret");
    klee_assume(len <= 1024);
    return len;
}

int strcmp(const char *a, const char *b) {
    (void) a;
    (void) b;
    int ret;
    klee_make_symbolic(&ret, sizeof(ret), "strcmp_ret");
    return ret;
}

char *strdup(const char *s) {
    (void) s;
    char *buf = malloc(256);
    if (buf)
        klee_make_symbolic(buf, 256, "strdup_buf");
    return buf;
}

int snprintf(char *dst, size_t size, const char *fmt, ...) {
    (void) fmt;
    if (dst && size > 0)
        klee_make_symbolic(dst, size, "snprintf_dst");
    int ret;
    klee_make_symbolic(&ret, sizeof(ret), "snprintf_ret");
    klee_assume(ret >= 0);
    klee_assume(ret <= (int) size);
    return ret;
}

// --- I/O (no-ops for symbolic execution) ---

int printf(const char *fmt, ...) {
    (void) fmt;
    return 0;
}

int fprintf(void *stream, const char *fmt, ...) {
    (void) stream;
    (void) fmt;
    return 0;
}

size_t fwrite(const void *ptr, size_t size, size_t nmemb, void *stream) {
    (void) ptr;
    (void) stream;
    return size * nmemb;
}

int puts(const char *s) {
    (void) s;
    return 0;
}
