// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t && %decompile-headless %t main %t1 && %file-check %s --input-file %t1
// CHECK: {{...}}

#include <stdio.h>
#include <stdlib.h>

// CHECK: {{...}}
int main(void) {
    int size = 5;
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

