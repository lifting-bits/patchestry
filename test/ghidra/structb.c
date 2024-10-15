// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function structb --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?structb}}"

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "arch":"{{.*}}","format":"{{.*}}","functions":{{...}}
// DECOMPILEA-SAME: "name":"{{_?structb}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --input %t.o --list-functions --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":{{...}}
// LISTFNS-SAME: "name":"{{_?structb}}"
// LISTFNS-SAME: "name":"{{_?main}}"

#include <stdio.h>

typedef struct
{
    char name[50];
    int age;
} Person;

int structb(void) {
    Person p;
    snprintf(p.name, sizeof(p.name), "John");
    p.age = 30;

    printf("Name: %s\n", p.name);
    printf("Age: %d\n", p.age);

    return 0;
}

int main(void) { return structb(); }
