// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t
// RUN %t; if [ "$(uname)" = "Linux" ]; then %decompile-headless %t write_file %t1 fi
// RUN %t; if [ "$(uname)" = "Darwin" ]; then %decompile-headless %t _write_file %t1 fi
// RUN %t1; %file-check %s --input-file %t1
// CHECK: {{...}}

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

