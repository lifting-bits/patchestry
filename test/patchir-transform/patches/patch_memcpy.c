// RUN: true
typedef unsigned long size_t;

extern void *memcpy(void *dest, const void *src, size_t n);

// CWE-122: Replace unbounded memcpy with size-capped version.
// The vulnerable pattern is memcpy into a heap buffer where the copy
// size is attacker-controlled and can exceed the allocation size.
void *patch__replace__memcpy(void *dest, const void *src,
                             size_t n, size_t max_size) {
    if (n > max_size) {
        n = max_size;
    }
    return memcpy(dest, src, n);
}
