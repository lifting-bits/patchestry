/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

/*
 * C library functions a product harness runs instead of stubbing.  The stub
 * of `name` records the call and then calls `__patchestry_model_name`.
 * check_equivalence.py lists these names (MODELS); keep the two in step.
 */

#include <stddef.h>

static int lower(int c) { return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c; }

size_t __patchestry_model_strlen(const char *text) {
    size_t length = 0;
    while (text[length] != 0) {
        length++;
    }
    return length;
}

char *__patchestry_model_strcpy(char *to, const char *from) {
    char *out = to;
    while ((*out++ = *from++) != 0) { }
    return to;
}

char *__patchestry_model_strncpy(char *to, const char *from, size_t count) {
    size_t at = 0;
    for (; at < count && from[at] != 0; at++) {
        to[at] = from[at];
    }
    for (; at < count; at++) {
        to[at] = 0;
    }
    return to;
}

char *__patchestry_model_strcat(char *to, const char *from) {
    __patchestry_model_strcpy(to + __patchestry_model_strlen(to), from);
    return to;
}

int __patchestry_model_strcmp(const char *left, const char *right) {
    while (*left != 0 && *left == *right) {
        left++;
        right++;
    }
    return (unsigned char) *left - (unsigned char) *right;
}

int __patchestry_model_strncmp(const char *left, const char *right, size_t count) {
    for (; count != 0; count--, left++, right++) {
        if (*left != *right || *left == 0) {
            return (unsigned char) *left - (unsigned char) *right;
        }
    }
    return 0;
}

int __patchestry_model_strcasecmp(const char *left, const char *right) {
    while (*left != 0 && lower((unsigned char) *left) == lower((unsigned char) *right)) {
        left++;
        right++;
    }
    return lower((unsigned char) *left) - lower((unsigned char) *right);
}

int __patchestry_model_strncasecmp(const char *left, const char *right, size_t count) {
    for (; count != 0; count--, left++, right++) {
        int a = lower((unsigned char) *left);
        int b = lower((unsigned char) *right);
        if (a != b || a == 0) {
            return a - b;
        }
    }
    return 0;
}

char *__patchestry_model_strchr(const char *text, int wanted) {
    for (;; text++) {
        if (*text == (char) wanted) {
            return (char *) text;
        }
        if (*text == 0) {
            return NULL;
        }
    }
}

char *__patchestry_model_strrchr(const char *text, int wanted) {
    const char *found = NULL;
    for (;; text++) {
        if (*text == (char) wanted) {
            found = text;
        }
        if (*text == 0) {
            return (char *) found;
        }
    }
}

char *__patchestry_model_strstr(const char *text, const char *wanted) {
    size_t length = __patchestry_model_strlen(wanted);
    for (; *text != 0 || length == 0; text++) {
        if (__patchestry_model_strncmp(text, wanted, length) == 0) {
            return (char *) text;
        }
        if (*text == 0) {
            break;
        }
    }
    return NULL;
}

int __patchestry_model_memcmp(const void *left, const void *right, size_t count) {
    const unsigned char *a = left;
    const unsigned char *b = right;
    for (; count != 0; count--, a++, b++) {
        if (*a != *b) {
            return *a - *b;
        }
    }
    return 0;
}
