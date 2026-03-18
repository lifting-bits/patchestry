// RUN: true
typedef unsigned int uint32_t;

extern unsigned int snprintf(void *buf, uint32_t size, const void *fmt, ...);

// CWE-134: Replace snprintf(buf, size, user_str) with snprintf(buf, size, "%s", user_str).
// The vulnerable pattern is snprintf called with a user-controlled format
// string that can contain %x, %n, etc.
// Note: unsigned int return matches CIR type from decompiled binary.
unsigned int patch__replace__snprintf(void *buf, uint32_t size, const void *user_str) {
    if (!buf || !user_str) return (unsigned int)-1;
    return snprintf(buf, size, "%s", user_str);
}
