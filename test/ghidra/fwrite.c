// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t.o
// RUN: %decompile-headless --input %t.o --function write_file --output %t
// RUN: %file-check -vv %s --input-file %t
// CHECK: "name":"{{_?write_file}}"

#include <stdio.h>

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
