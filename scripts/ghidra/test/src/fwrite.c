#include <stdio.h>

// CHECK: {{...}}
void write_file(const char* filename, const char* content) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        return;
    }
    fprintf(file, "%s", content);
    fclose(file);
}

int main() {
    const char* filename = "test.txt";
    const char* content = "Hello, World!";
    write_file(filename, content);
    return 0;
}

