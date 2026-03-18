// RUN: true
typedef unsigned int uint32_t;

extern void *strncpy(void *dest, const void *src, uint32_t n);

// CWE-676: Replace unbounded strcpy with strncpy + null terminator.
// The vulnerable pattern is strcpy into a fixed-size member buffer
// without bounds checking.
void *patch__replace__strcpy(void *dest, const void *src, uint32_t max_size) {
    char *d = (char *)dest;
    void *result = strncpy(dest, src, max_size - 1);
    d[max_size - 1] = '\0';
    return result;
}
