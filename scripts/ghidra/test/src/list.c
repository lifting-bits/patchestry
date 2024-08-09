#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

// CHECK: {{...}}
void print_list(Node *head) {
    Node *current = head;
    while (current) {
        printf("%d -> ", current->data);
        current = current->next;
    }
    printf("NULL\n");
}

int main(void) {
    Node *head = malloc(sizeof(Node));
    head->data = 1;
    head->next = malloc(sizeof(Node));
    head->next->data = 2;
    head->next->next = malloc(sizeof(Node));
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

