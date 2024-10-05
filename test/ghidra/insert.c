// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function insert_substring --output %t %ci_output_folder

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "arch":"{{.*}}","os":"{{.*}}","functions":{{...}}
// DECOMPILEA-SAME: "name":"{{_?insert_substring}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --high-pcode --input %t.o --function insert_substring --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEHS %s --input-file %t
// DECOMPILEHS: "name":"{{_?insert_substring}}"

// RUN: %decompile-headless --high-pcode --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEHA %s --input-file %t
// DECOMPILEHA: "arch":"{{.*}}","os":"{{.*}}","functions":{{...}}
// DECOMPILEHA-SAME: "name":"{{_?insert_substring}}"
// DECOMPILEHA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --list-functions --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":{{...}}
// LISTFNS-SAME: "name":"{{_?insert_substring}}"
// LISTFNS-SAME: "name":"{{_?main}}"

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
    char sub[]    = ", ";
    insert_substring(str, sub, 3);
    printf("After insertion: %s\n", str);
    return 0;
}
