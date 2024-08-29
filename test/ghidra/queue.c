// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t && %decompile-headless %t dequeue %t1 && %file-check %s --input-file %t1
// CHECK: {{...}}

#include <stdio.h>
#include <stdlib.h>

#define SIZE 5

typedef struct {
    int data[SIZE];
    int front, rear, count;
} Queue;

void init_queue(Queue *q) {
    q->front = 0;
    q->rear = 0;
    q->count = 0;
}

int is_empty(Queue *q) {
    return q->count == 0;
}

int is_full(Queue *q) {
    return q->count == SIZE;
}

void enqueue(Queue *q, int value) {
    if (is_full(q)) {
        printf("Queue is full\n");
        return;
    }
    q->data[q->rear] = value;
    q->rear = (q->rear + 1) % SIZE;
    q->count++;
}

int dequeue(Queue *q) {
    if (is_empty(q)) {
        printf("Queue is empty\n");
        return -1;  // Error value
    }
    int value = q->data[q->front];
    q->front = (q->front + 1) % SIZE;
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
