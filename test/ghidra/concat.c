// UNSUPPORTED: system-windows
// RUN: %cc-x86_64 %s -g -c -o %t.o
// RUN: %decompile-headless --input %t.o --function string_concat --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?string_concat}}"

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "arch":"{{.*}}","format":"{{.*}}","functions":
// DECOMPILEA-SAME: "name":"{{_?string_concat}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --input %t.o --list-functions --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":
// LISTFNS-SAME: "name":"{{_?string_concat}}"
// LISTFNS-SAME: "name":"{{_?main}}"

#include <stdio.h>
#include <string.h>

char *string_concat(char *dest, const char *src) {
    strcat(dest, src);
    return dest;
}

int main() {
    char dest[50] = "Hello, ";
    char src[]    = "World!";
    printf("%s\n", string_concat(dest, src));
    return 0;
}
