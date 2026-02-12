/* SeaHorn libc models: constrain what callers rely on; keep side-effects minimal. */

#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>

/* Non-determinism */
extern int nd_int(void);
extern unsigned int nd_uint(void);
extern _Bool nd_bool(void);
extern size_t nd_size_t(void);

/* Verification */
extern void __VERIFIER_assume(_Bool);
extern void __VERIFIER_error(void) __attribute__((noreturn));

/* Helpers */
static inline size_t __sea_min(size_t a, size_t b) { return a < b ? a : b; }
static inline size_t __sea_align_up(size_t x, size_t a) { return (x + (a - 1)) & ~(a - 1); }

/* Simple bump heap: fresh, non-NULL, writable regions. */
#ifndef SEA_HEAP_SIZE
#define SEA_HEAP_SIZE (1u << 20) /* 1 MiB */
#endif

#ifndef SEA_MAX_STRLEN
#define SEA_MAX_STRLEN 4096u
#endif

static unsigned char __sea_heap[SEA_HEAP_SIZE];
static size_t __sea_brk;

static void *__sea_alloc(size_t n) {
    _Bool ok = nd_bool();
    if (!ok) return (void *)0;
  
    size_t a = sizeof(void *);
    size_t start = __sea_align_up(__sea_brk, a);
    size_t end = start + n;
  
    if (end < start) return (void *)0;              /* overflow */
    if (end > (size_t)SEA_HEAP_SIZE) return (void *)0;
  
    __sea_brk = end;
    return (void *)&__sea_heap[start];
  }

/* printf-family */

/* snprintf: Safe bounded string formatting.
 *
 * Models realistic snprintf behavior:
 * - Returns number of chars that would be written (excluding NUL)
 * - Only writes up to (size-1) chars + NUL terminator
 * - Always NUL-terminates if size > 0
 * - Return value may be >= size-1 (indicates truncation)
 *
 */
int snprintf(char *str, size_t size, const char *fmt, ...) {
    int r = nd_int();
    __VERIFIER_assume(r >= 0);
    __VERIFIER_assume(r <= (int)SEA_MAX_STRLEN);  /* Bound output size */
    __VERIFIER_assume(fmt != 0);

    if (size == 0) return r;
    __VERIFIER_assume(str != 0);

    /* Always NUL-terminate within buffer bounds. */
    size_t pos = __sea_min((size_t)r, size - 1);
    str[pos] = '\0';

    /* Return value indicates if truncation occurred: r >= size means truncated */
    return r;
}

/* vsnprintf: Safe bounded string formatting with va_list.
 *
 * See snprintf comment above for behavior and security notes.
 */
int vsnprintf(char *str, size_t size, const char *fmt, va_list ap) {
    int r = nd_int();
    __VERIFIER_assume(r >= 0);
    __VERIFIER_assume(r <= (int)SEA_MAX_STRLEN);  /* Bound output size */
    __VERIFIER_assume(fmt != 0);

    if (size == 0) return r;
    __VERIFIER_assume(str != 0);

    size_t pos = __sea_min((size_t)r, size - 1);
    str[pos] = '\0';
    return r;
}

int sprintf(char *str, const char *fmt, ...) {
    int r = nd_int();
    __VERIFIER_assume(r >= 0);
    __VERIFIER_assume(r <= (int)SEA_MAX_STRLEN);  /* Bound output size */
    __VERIFIER_assume(str != 0);
    __VERIFIER_assume(fmt != 0);

    /* Model conservative behavior: assumes buffer has at least r+1 capacity.
     * Verification will fail if contracts don't ensure adequate buffer size.
     * This catches potential buffer overflows during verification. */
    str[(size_t)r] = '\0';
    return r;
}

/* 
 * See sprintf warning above for security considerations.
 */
int vsprintf(char *str, const char *fmt, va_list ap) {
    int r = nd_int();
    __VERIFIER_assume(r >= 0);
    __VERIFIER_assume(r <= (int)SEA_MAX_STRLEN);  /* Bound output size */
    __VERIFIER_assume(str != 0);
    __VERIFIER_assume(fmt != 0);

    /* Conservative model - see sprintf comment above. */
    str[(size_t)r] = '\0';
    return r;
}

int printf(const char *fmt, ...) {
    int r = nd_int();
    __VERIFIER_assume(r >= 0);
    __VERIFIER_assume(fmt != 0);
    return r;
}
  
int fprintf(void *stream, const char *fmt, ...) {
    int r = nd_int();
    __VERIFIER_assume(r >= 0);
    __VERIFIER_assume(stream != 0);
    __VERIFIER_assume(fmt != 0);
    return r;
}

/* strings */
size_t strlen(const char *s) {
    size_t len = nd_size_t();
    __VERIFIER_assume(s != 0);
    // keep strlen bounded
    __VERIFIER_assume(len <= (size_t)SEA_MAX_STRLEN);
    __VERIFIER_assume(s[len] == '\0');

    /* Enforce canonical strlen semantics: no earlier terminator. */
    for (size_t i = 0; i < len; ++i) {
        __VERIFIER_assume(s[i] != '\0');
    }

    return len;
}

size_t strnlen(const char *s, size_t maxlen) {
    __VERIFIER_assume(s != 0);
    size_t len = nd_size_t();
    __VERIFIER_assume(len <= maxlen);
    __VERIFIER_assume(len <= (size_t)SEA_MAX_STRLEN);
    for (size_t i = 0; i < len; ++i) {
        __VERIFIER_assume(s[i] != '\0');
    }
    if (len < maxlen) __VERIFIER_assume(s[len] == '\0');
    return len;
}

char *strcpy(char *dst, const char *src) {
    __VERIFIER_assume(dst != 0);
    __VERIFIER_assume(src != 0);
    size_t len = strlen(src);
    /* Conservative model assumes dst has capacity for len+1 bytes.
     * Real strcpy will overflow if dst is too small. */
    for (size_t i = 0; i < len; ++i) {
        dst[i] = src[i];
    }
    dst[len] = '\0';
    return dst;
}
  
char *strncpy(char *dst, const char *src, size_t n) {
    __VERIFIER_assume(dst != 0);
    __VERIFIER_assume(src != 0);
    if (n == 0) return dst;
    size_t copy = strnlen(src, n);
    for (size_t i = 0; i < copy; ++i) {
        dst[i] = src[i];
    }
    if (copy < n) dst[copy] = '\0';
    return dst;
}

char *strcat(char *dst, const char *src) {
    __VERIFIER_assume(dst != 0);
    __VERIFIER_assume(src != 0);
    size_t dlen = strlen(dst);
    size_t slen = strlen(src);
    /* Conservative model assumes dst has capacity for dlen+slen+1 bytes.
     * Real strcat will overflow if dst buffer is too small. */
    for (size_t i = 0; i < slen; ++i) {
        dst[dlen + i] = src[i];
    }
    dst[dlen + slen] = '\0';
    return dst;
}
  
char *strncat(char *dst, const char *src, size_t n) {
    __VERIFIER_assume(dst != 0);
    __VERIFIER_assume(src != 0);
    size_t dlen = strlen(dst);
    size_t copy = strnlen(src, n);
    for (size_t i = 0; i < copy; ++i) {
        dst[dlen + i] = src[i];
    }
    dst[dlen + copy] = '\0';
    return dst;
}
  
int strcmp(const char *a, const char *b) {
    if (a == b) return 0;
    __VERIFIER_assume(a != 0);
    __VERIFIER_assume(b != 0);
    int r = nd_int();
    __VERIFIER_assume(r >= -1 && r <= 1);
    if (r == 0) __VERIFIER_assume(a[0] == b[0]);
    if (a[0] != b[0]) __VERIFIER_assume(r != 0);
    return r;
}
  
int strncmp(const char *a, const char *b, size_t n) {
    if (n == 0) return 0;
    if (a == b) return 0;
    __VERIFIER_assume(a != 0);
    __VERIFIER_assume(b != 0);
    int r = nd_int();
    __VERIFIER_assume(r >= -1 && r <= 1);
    if (r == 0) __VERIFIER_assume(a[0] == b[0]);
    if (a[0] != b[0]) __VERIFIER_assume(r != 0);
    return r;
}
  
char *strstr(const char *hay, const char *needle) {
    __VERIFIER_assume(hay != 0);
    __VERIFIER_assume(needle != 0);
    if (needle[0] == '\0') return (char *)hay;
    if (!nd_bool()) return (char *)0;
    size_t hlen = strlen(hay);
    size_t nlen = strlen(needle);
    __VERIFIER_assume(nlen > 0);
    __VERIFIER_assume(nlen <= hlen);
    size_t off = nd_size_t();
    __VERIFIER_assume(off <= hlen);
    __VERIFIER_assume(nlen <= hlen - off);
    __VERIFIER_assume(hay[off] == needle[0]);
    return (char *)hay + off;
}
  
char *strchr(const char *s, int c) {
    __VERIFIER_assume(s != 0);
    size_t len = strlen(s);
    if ((char)c == '\0') return (char *)s + len;
    if (!nd_bool()) return (char *)0;
    size_t off = nd_size_t();
    __VERIFIER_assume(off < len);
    __VERIFIER_assume(s[off] == (char)c);
    return (char *)s + off;
}
  
char *strrchr(const char *s, int c) {
    __VERIFIER_assume(s != 0);
    size_t len = strlen(s);
    if ((char)c == '\0') return (char *)s + len;
    if (!nd_bool()) return (char *)0;
    size_t off = nd_size_t();
    __VERIFIER_assume(off < len);
    __VERIFIER_assume(s[off] == (char)c);
    return (char *)s + off;
}

/* memory */
void *memcpy(void *dst, const void *src, size_t n) {
    if (n > 0) {
        __VERIFIER_assume(dst != 0);
        __VERIFIER_assume(src != 0);
        ((unsigned char *)dst)[0] = ((const unsigned char *)src)[0];
        ((unsigned char *)dst)[n - 1] = ((const unsigned char *)src)[n - 1];
    }
    return dst;
}
  
void *memmove(void *dst, const void *src, size_t n) {
    if (n > 0) {
        __VERIFIER_assume(dst != 0);
        __VERIFIER_assume(src != 0);
        ((unsigned char *)dst)[0] = ((const unsigned char *)src)[0];
        ((unsigned char *)dst)[n - 1] = ((const unsigned char *)src)[n - 1];
    }
    return dst;
}
  
void *memset(void *s, int c, size_t n) {
    if (n > 0) {
        __VERIFIER_assume(s != 0);
        ((unsigned char *)s)[0] = (unsigned char)c;
        ((unsigned char *)s)[n - 1] = (unsigned char)c;
    }
    return s;
}
  
int memcmp(const void *a, const void *b, size_t n) {
    if (n == 0) return 0;
    if (a == b) return 0;
    __VERIFIER_assume(a != 0);
    __VERIFIER_assume(b != 0);
    int r = nd_int();
    __VERIFIER_assume(r >= -1 && r <= 1);
    if (((const unsigned char *)a)[0] != ((const unsigned char *)b)[0]) {
        __VERIFIER_assume(r != 0);
    }
    if (r == 0) {
        __VERIFIER_assume(((const unsigned char *)a)[0] == ((const unsigned char *)b)[0]);
        __VERIFIER_assume(((const unsigned char *)a)[n - 1] == ((const unsigned char *)b)[n - 1]);
    }
    return r;
}
  
void *memchr(const void *s, int c, size_t n) {
    if (n == 0) return (void *)0;
    __VERIFIER_assume(s != 0);
    if (!nd_bool()) return (void *)0;
    size_t off = nd_size_t();
    __VERIFIER_assume(off < n);
    __VERIFIER_assume(((const unsigned char *)s)[off] == (unsigned char)c);
    return (void *)((const unsigned char *)s + off);
}
  
/* alloc */
  
void *malloc(size_t size) {
    if (size == 0 && nd_bool()) return (void *)0;
    void *p = __sea_alloc(size == 0 ? 1 : size);
    return p;
}
  
void *calloc(size_t nmemb, size_t size) {
    if (nmemb == 0 || size == 0) {
        if (nd_bool()) return (void *)0;
        return __sea_alloc(1);
    }
    size_t total = nmemb * size;
    if (nmemb != 0 && total / nmemb != size) return (void *)0; /* overflow */
    void *p = __sea_alloc(total);
    if (p && total > 0) {
        ((unsigned char *)p)[0] = 0;
        ((unsigned char *)p)[total - 1] = 0;
    }
    return p;
}
  
void *realloc(void *ptr, size_t size) {
    if (size == 0) {
        return (void *)0;
    }
    if (ptr == 0) {
        return __sea_alloc(size);
    }
    if (nd_bool()) return ptr; /* may reuse same block */
    return __sea_alloc(size);
}
  
void free(void *ptr) { (void)ptr; }

/* ctype */

int isdigit(int c) { return (c >= 48 && c <= 57) ? 1 : 0; }
int isalpha(int c) { return ((c >= 65 && c <= 90) || (c >= 97 && c <= 122)) ? 1 : 0; }
int isalnum(int c) { return (isdigit(c) || isalpha(c)) ? 1 : 0; }
int isspace(int c) { return (c == 32 || (c >= 9 && c <= 13)) ? 1 : 0; }
int toupper(int c) { return (c >= 97 && c <= 122) ? (c - 32) : c; }
int tolower(int c) { return (c >= 65 && c <= 90) ? (c + 32) : c; }

/* conversion */

int atoi(const char *s) {
    __VERIFIER_assume(s != 0);
    int r = nd_int();
    if (!(isdigit((unsigned char)s[0]) || s[0] == '+' || s[0] == '-')) {
        __VERIFIER_assume(r == 0);
    }
    return r;
}
  
long atol(const char *s) {
    __VERIFIER_assume(s != 0);
    long r = (long)nd_int();
    if (!(isdigit((unsigned char)s[0]) || s[0] == '+' || s[0] == '-')) {
        __VERIFIER_assume(r == 0);
    }
    return r;
}
  
long strtol(const char *s, char **endp, int base) {
    __VERIFIER_assume(s != 0);
    __VERIFIER_assume(base == 0 || (base >= 2 && base <= 36));
    long r = (long)nd_int();
    size_t n = strnlen(s, SEA_MAX_STRLEN);
    size_t off = nd_size_t();
    __VERIFIER_assume(off <= n);
    if (off == 0) __VERIFIER_assume(r == 0);
    if (endp) {
        *endp = (char *)s + off;
    }
    return r;
}
  
unsigned long strtoul(const char *s, char **endp, int base) {
    __VERIFIER_assume(s != 0);
    __VERIFIER_assume(base == 0 || (base >= 2 && base <= 36));
    unsigned long r = (unsigned long)nd_uint();
    size_t n = strnlen(s, SEA_MAX_STRLEN);
    size_t off = nd_size_t();
    __VERIFIER_assume(off <= n);
    if (off == 0) __VERIFIER_assume(r == 0);
    if (endp) {
        *endp = (char *)s + off;
    }
    return r;
}

/* stdio-ish */

void *fopen(const char *path, const char *mode) {
    __VERIFIER_assume(path != 0);
    __VERIFIER_assume(mode != 0);
    if (!nd_bool()) return (void *)0;
    return __sea_alloc(8);
}
  
int fclose(void *stream) {
    __VERIFIER_assume(stream != 0);
    return nd_bool() ? 0 : -1;
}
  
size_t fread(void *ptr, size_t size, size_t nmemb, void *stream) {
    if (size == 0 || nmemb == 0) return 0;
    if (nmemb > 0 && size > 0) __VERIFIER_assume(ptr != 0);
    __VERIFIER_assume(stream != 0);
    size_t k = nd_size_t();
    __VERIFIER_assume(k <= nmemb);
    size_t bytes = k * size;
    if (k != 0) __VERIFIER_assume(bytes / k == size); /* overflow guard */
    if (bytes > 0) {
        ((unsigned char *)ptr)[0] = (unsigned char)nd_uint();
        ((unsigned char *)ptr)[bytes - 1] = (unsigned char)nd_uint();
    }
    return k;
}
  
size_t fwrite(const void *ptr, size_t size, size_t nmemb, void *stream) {
    if (size == 0 || nmemb == 0) return 0;
    if (nmemb > 0 && size > 0) __VERIFIER_assume(ptr != 0);
    __VERIFIER_assume(stream != 0);
    size_t k = nd_size_t();
    __VERIFIER_assume(k <= nmemb);
    return k;
}
  
char *fgets(char *s, int size, void *stream) {
    if (size <= 0) return (char *)0;
    __VERIFIER_assume(s != 0);
    __VERIFIER_assume(stream != 0);
    if (!nd_bool()) return (char *)0;
    if (size == 1) {
        s[0] = '\0';
        return s;
    }
    size_t len = nd_size_t();
    __VERIFIER_assume(len >= 1);
    __VERIFIER_assume(len < (size_t)size);
    s[0] = (char)nd_uint();
    __VERIFIER_assume(s[0] != '\0');
    s[len] = '\0';
    return s;
}
  
int fputs(const char *s, void *stream) {
    __VERIFIER_assume(s != 0);
    __VERIFIER_assume(stream != 0);
    if (!nd_bool()) return -1;
    int r = nd_int();
    __VERIFIER_assume(r >= 0);
    return r;
}
  
/* misc */

char *getenv(const char *name) {
    __VERIFIER_assume(name != 0);
    if (!nd_bool()) return (char *)0;
    char *p = (char *)__sea_alloc(2);
    if (p) {
        p[0] = nd_bool() ? 'X' : '\0';
        p[1] = '\0';
    }
    return p;
}
  
void exit(int status) __attribute__((noreturn));
void exit(int status) { (void)status; __VERIFIER_error(); }
  
void abort(void) __attribute__((noreturn));
void abort(void) { __VERIFIER_error(); }
