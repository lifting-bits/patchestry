// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function structb --output %t %ci_output_folder
// RUN: %file-check -vv %s --input-file %t
// CHECK: "name":"{{_?structb}}"


#include <stdio.h>

typedef struct {
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

int main(void) {
    return structb();
}
