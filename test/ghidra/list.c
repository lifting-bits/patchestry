// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function print_list --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?print_list}}"

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "arch":"{{.*}}","os":"{{.*}}","functions":{{...}}
// DECOMPILEA-SAME: "name":"{{_?print_list}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --high-pcode --input %t.o --function print_list --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEHS %s --input-file %t
// DECOMPILEHS: "name":"{{_?print_list}}"

// RUN: %decompile-headless --high-pcode --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEHA %s --input-file %t
// DECOMPILEHA: "arch":"{{.*}}","os":"{{.*}}","functions":{{...}}
// DECOMPILEHA-SAME: "name":"{{_?print_list}}"
// DECOMPILEHA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --list-functions --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":{{...}}
// LISTFNS-SAME: "name":"{{_?print_list}}"
// LISTFNS-SAME: "name":"{{_?main}}"

#include <stdio.h>
#include <stdlib.h>

typedef struct Node
{
    int data;
    struct Node *next;
} Node;

void print_list(Node *head) {
    Node *current = head;
    while (current) {
        printf("%d -> ", current->data);
        current = current->next;
    }
    printf("NULL\n");
}

int main(void) {
    Node *head             = malloc(sizeof(Node));
    head->data             = 1;
    head->next             = malloc(sizeof(Node));
    head->next->data       = 2;
    head->next->next       = malloc(sizeof(Node));
    head->next->next->data = 3;
    head->next->next->next = NULL;

    print_list(head);

    // Free the memory
    Node *current = head;
    Node *next;
    while (current) {
        next = current->next;
        free(current);
        current = next;
    }

    return 0;
}
