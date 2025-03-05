// UNSUPPORTED: system-windows
// RUN: %cc-x86_64 %s -g -c -o %t.o
// RUN: %decompile-headless --input %t.o --function sort_test --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?sort_test}}"

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "arch":"{{.*}}","format":"{{.*}}","functions":
// DECOMPILEA-SAME: "name":"{{_?sort_test}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --input %t.o --list-functions --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":
// LISTFNS-SAME: "name":"{{_?sort_test}}"
// LISTFNS-SAME: "name":"{{_?main}}"

#include <stdio.h>

int sort_test() {
    int array[100], n, c, d, swap;
    scanf("%d", &n);
    for (c = 0; c < n; c++) {
        scanf("%d", &array[c]);
    }

    for (c = 0; c < n - 1; c++) {
        for (d = 0; d < n - c - 1; d++) {
            if (array[d] > array[d + 1]) {
                swap         = array[d];
                array[d]     = array[d + 1];
                array[d + 1] = swap;
            }
        }
    }

    for (c = 0; c < n; c++) {
        printf("%d\n", array[c]);
    }

    return 0;
}

int main(void) { return sort_test(); }
