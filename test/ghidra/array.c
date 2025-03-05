// UNSUPPORTED: system-windows
// RUN: %cc-x86_64 %s -g -c -o %t.o
// RUN: %decompile-headless --input %t.o --function array --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?array}}"

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "arch":"{{.*}}","format":"{{.*}}","functions":
// DECOMPILEA-SAME: "name":"{{_?array}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --input %t.o --list-functions --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":
// LISTFNS-SAME: "name":"{{_?array}}"
// LISTFNS-SAME: "name":"{{_?main}}"

#include <stdio.h>
#include <stdlib.h>

int array(int argc, char **argv) {
    int size   = 5;
    int *array = malloc(size * sizeof(int));

    for (int i = 0; i < size; ++i) {
        array[i] = i * 2;
    }

    for (int i = 0; i < size; ++i) {
        printf("%d ", array[i]);
    }
    printf("\n");

    free(array);
    return 0;
}

int main(int a, char **argv) { return array(a, argv); }
