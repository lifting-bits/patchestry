#include <stdio.h>

//CHECK: {{...}}
int main(void) {
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

