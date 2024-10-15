// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
/* clang-format off */ 
// RUN: %decompile-headless --input %t.o --function multiply_matrices --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?multiply_matrices}}"

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "arch":"{{.*}}","format":"{{.*}}","functions":{{...}}
// DECOMPILEA-SAME: "name":"{{_?multiply_matrices}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --list-functions --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":{{...}}
// LISTFNS-SAME: "name":"{{_?multiply_matrices}}"
// LISTFNS-SAME: "name":"{{_?main}}"

#include <stdio.h>

#define SIZE 3

void multiply_matrices(int a[SIZE][SIZE], int b[SIZE][SIZE], int result[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < SIZE; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main(void) {
    int a[SIZE][SIZE] = {
        { 1, 2, 3 },
        { 4, 5, 6 },
        { 7, 8, 9 }
    };
    int b[SIZE][SIZE] = {
        { 9, 8, 7 },
        { 6, 5, 4 },
        { 3, 2, 1 }
    };
    int result[SIZE][SIZE];

    multiply_matrices(a, b, result);

    printf("Resultant matrix:\n");
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%d ", result[i][j]);
        }
        printf("\n");
    }

    return 0;
}
