// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t && %decompile-headless %t string_concat %t1 && %file-check %s --input-file %t1
// CHECK: {{...}}

#include <stdio.h>
#include <string.h>

char* string_concat(char* dest, const char* src) {
    strcat(dest, src);
    return dest;
}

int main() {
    char dest[50] = "Hello, ";
    char src[] = "World!";
    printf("%s\n", string_concat(dest, src));
    return 0;
}

