#include <stdio.h>
#include <string.h>

// CHECK: {{...}}
char* string_concat(char* dest, const char* src) {
    strcat(dest, src);
    return dest;
}

int main() {
    char dest[50] = "Hello, ";
    char src[] = "World!";
    printf("%s\n", string_concat(dest, src));
    return 0;
}

