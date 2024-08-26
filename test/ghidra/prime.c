// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t
// RUN %t; if [ "$(uname)" = "Linux" ]; then %decompile-headless %t is_prime %t1 fi
// RUN %t; if [ "$(uname)" = "Darwin" ]; then %decompile-headless %t _is_prime %t1 fi
// RUN %t1; %file-check %s --input-file %t1
// CHECK: {{...}}

#include <stdio.h>
#include <stdbool.h>

bool is_prime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i < n; i++) {
        if (n % i == 0) return false;
    }
    return true;
}

int main() {
    int n = 29;
    printf("is prime: %d\n", is_prime(n));
    return 0;
}

