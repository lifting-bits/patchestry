// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function reverse_string --output %t %ci_output_folder
// RUN: %file-check -vv %s --input-file %t
// CHECK: "name":"{{_?reverse_string}}"

#include <stdio.h>
#include <string.h>

void reverse_string(char *str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; ++i) {
        char temp = str[i];
        str[i] = str[len - i - 1];
        str[len - i - 1] = temp;
    }
}

int main(void) {
    char str[] = "hello";
    reverse_string(str);
    printf("Reversed: %s\n", str);
    return 0;
}
