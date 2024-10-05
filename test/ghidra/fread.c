// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function fread_test --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?fread_test}}"

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "arch":"{{.*}}","os":"{{.*}}","functions":{{...}}
// DECOMPILEA-SAME: "name":"{{_?fread_test}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --high-pcode --input %t.o --function fread_test --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEHS %s --input-file %t
// DECOMPILEHS: "name":"{{_?fread_test}}"

// RUN: %decompile-headless --high-pcode --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEHA %s --input-file %t
// DECOMPILEHA: "arch":"{{.*}}","os":"{{.*}}","functions":{{...}}
// DECOMPILEHA-SAME: "name":"{{_?fread_test}}"
// DECOMPILEHA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --input %t.o --list-functions --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":{{...}}
// LISTFNS-SAME: "name":"{{_?fread_test}}"
// LISTFNS-SAME: "name":"{{_?main}}"

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

int main(void) { return fread_test(); }
