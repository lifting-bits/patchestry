// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t.o
// RUN: %decompile-headless --input %t.o --function fread_test --output %t
// RUN: %file-check -vv %s --input-file %t
// CHECK: "name":"{{_?fread_test}}"

#include <stdio.h>

int fread_test(void) {
    FILE *file = fopen("example.txt", "r");
    if (!file) {
        perror("Error opening file");
        return 1;
    }

    char ch;
    while ((ch = fgetc(file)) != EOF) {
        putchar(ch);
    }

    fclose(file);
    return 0;
}

int main(void) {
    return fread_test();
}
