// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function insert_substring --output %t %ci_output_folder
// RUN: %file-check -vv %s --input-file %t
// CHECK: "name":"{{_?insert_substring}}"

#include <stdio.h>
#include <string.h>

void insert_substring(char *str, char *sub, int pos) {
    char temp[100];
    strncpy(temp, str, pos);
    temp[pos] = '\0';
    strcat(temp, sub);
    strcat(temp, str + pos);
    strcpy(str, temp);
}

int main() {
    char str[100] = "Hello World!";
    char sub[] = ", ";
    insert_substring(str, sub, 3);
    printf("After insertion: %s\n", str);
    return 0;
}
