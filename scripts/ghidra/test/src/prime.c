#include <stdio.h>
#include <stdbool.h>

// CHECK: {{...}}
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

