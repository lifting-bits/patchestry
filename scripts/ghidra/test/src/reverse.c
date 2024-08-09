#include <stdio.h>
#include <string.h>

// CHECK: {{...}}
void reverse_string(char *str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; ++i) {
        char temp = str[i];
        str[i] = str[len - i - 1];
        str[len - i - 1] = temp;
    }
}

int main(void) {
    char str[] = "hello";
    reverse_string(str);
    printf("Reversed: %s\n", str);
    return 0;
}
