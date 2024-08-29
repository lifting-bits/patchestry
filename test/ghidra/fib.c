// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t && %decompile-headless %t fibonacci %t1 && %file-check %s --input-file %t1
// CHECK: {{...}}

#include <stdio.h>

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    int n = 10;
    printf("%d: %d\n", n, fibonacci(n));
    return 0;
}

