// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function write_file --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?write_file}}"

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "arch":"{{.*}}","format":"{{.*}}","functions":{{...}}
// DECOMPILEA-SAME: "name":"{{_?write_file}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --input %t.o --list-functions --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":{{...}}
// LISTFNS-SAME: "name":"{{_?write_file}}"
// LISTFNS-SAME: "name":"{{_?main}}"

#include <stdio.h>

void write_file(const char *filename, const char *content) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        return;
    }
    fprintf(file, "%s", content);
    fclose(file);
}

int main() {
    const char *filename = "test.txt";
    const char *content  = "Hello, World!";
    write_file(filename, content);
    return 0;
}
