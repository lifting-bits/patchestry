#include <stdio.h>

typedef struct {
    char name[50];
    int age;
} Person;

// CHECK: {{...}}
int main(void) {
    Person p;
    snprintf(p.name, sizeof(p.name), "John");
    p.age = 30;

    printf("Name: %s\n", p.name);
    printf("Age: %d\n", p.age);

    return 0;
}

