// UNSUPPORTED: system-windows
// RUN: %cc-x86_64 %s -g -c -o %t.o
// RUN: %decompile-headless --input %t.o --function dequeue --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?dequeue}}"

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "architecture":"{{.*}}","format":"{{.*}}","functions":
// DECOMPILEA-SAME: "name":"{{_?init_queue}}"
// DECOMPILEA-SAME: "name":"{{_?enqueue}}"
// DECOMPILEA-SAME: "name":"{{_?dequeue}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --list-functions --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":
// LISTFNS-SAME: "name":"{{_?init_queue}}"
// LISTFNS-SAME: "name":"{{_?enqueue}}"
// LISTFNS-SAME: "name":"{{_?dequeue}}"
// LISTFNS-SAME: "name":"{{_?main}}"

#include <stdio.h>
#include <stdlib.h>

#define SIZE 5

typedef struct
{
    int data[SIZE];
    int front, rear, count;
} Queue;

void init_queue(Queue *q) {
    q->front = 0;
    q->rear  = 0;
    q->count = 0;
}

int is_empty(Queue *q) { return q->count == 0; }

int is_full(Queue *q) { return q->count == SIZE; }

void enqueue(Queue *q, int value) {
    if (is_full(q)) {
        printf("Queue is full\n");
        return;
    }
    q->data[q->rear] = value;
    q->rear          = (q->rear + 1) % SIZE;
    q->count++;
}

int dequeue(Queue *q) {
    if (is_empty(q)) {
        printf("Queue is empty\n");
        return -1; // Error value
    }
    int value = q->data[q->front];
    q->front  = (q->front + 1) % SIZE;
    q->count--;
    return value;
}

int main(void) {
    Queue q;
    init_queue(&q);

    enqueue(&q, 1);
    enqueue(&q, 2);
    enqueue(&q, 3);

    printf("Dequeued: %d\n", dequeue(&q));
    printf("Dequeued: %d\n", dequeue(&q));
    printf("Dequeued: %d\n", dequeue(&q));

    return 0;
}
