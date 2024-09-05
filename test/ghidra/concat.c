// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function string_concat --output %t %ci_output_folder
// RUN: %file-check -vv %s --input-file %t
// CHECK: "name":"{{_?string_concat}}"

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
