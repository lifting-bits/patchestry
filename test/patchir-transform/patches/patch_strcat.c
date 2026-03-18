// RUN: true
typedef unsigned long size_t;

extern size_t strlen(const char *s);
extern char *strncat(char *dest, const char *src, size_t n);

// CWE-121: Replace unbounded strcat with strncat that respects buffer capacity.
// The vulnerable pattern is a loop calling strcat to accumulate a hex string
// into a fixed-size stack buffer without tracking remaining capacity.
char *patch__replace__strcat(char *dest, const char *src, size_t buf_size) {
    size_t current_len = strlen(dest);
    if (current_len >= buf_size) return dest;
    size_t remaining = buf_size - current_len - 1;
    return strncat(dest, src, remaining);
}
