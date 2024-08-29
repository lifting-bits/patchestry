// UNSUPPORTED: system-windows
// RUN: %cc -ggdb %s -o %t && %decompile-headless %t main %t1 && %file-check %s --input-file %t1
// CHECK: {{...}}

#include <stdio.h>

typedef struct {
    char name[50];
    int age;
} Person;

int main(void) {
    Person p;
    snprintf(p.name, sizeof(p.name), "John");
    p.age = 30;

    printf("Name: %s\n", p.name);
    printf("Age: %d\n", p.age);

    return 0;
}

