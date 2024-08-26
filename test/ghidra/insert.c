// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t
// RUN %t; if [ "$(uname)" = "Linux" ]; then %decompile-headless %t insert_substring %t1 fi
// RUN %t; if [ "$(uname)" = "Darwin" ]; then %decompile-headless %t _insert_substring %t1 fi
// RUN %t1; %file-check %s --input-file %t1
// CHECK: {{...}}

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

