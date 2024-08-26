// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t
// RUN %t; if [ "$(uname)" = "Linux" ]; then %decompile-headless %t string_concat %t1 fi
// RUN %t; if [ "$(uname)" = "Darwin" ]; then %decompile-headless %t _string_concat %t1 fi
// RUN %t1; %file-check %s --input-file %t1
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

