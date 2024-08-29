// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t && %decompile-headless %t fread_test %t1 && %file-check %s --input-file %t1
// CHECK: {{...}}

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