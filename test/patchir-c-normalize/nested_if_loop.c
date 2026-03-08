// RUN: %patchir-c-normalize -input %s -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=NESTED %s --input-file %t.c
// NESTED: while (
// NESTED: if (
// NESTED-NOT: goto

// Pattern: if inside while — both WhileLoopStructurizePass and
// ConditionalStructurizePass must fire (in different pipeline iterations).
//
// Outer while: loop_head (condition) / loop_exit / loop_body (body + back-edge)
// Inner if-else: branch inside loop_body, else arm first in RPO

void nested_if_loop(int x, int y) {
loop_head:
    if (x > 0) goto loop_body; else goto loop_exit;
loop_exit:
    return;
loop_body:
    if (y > 0) goto then_block; else goto else_block;
else_block:
    y = 0;
    goto join;
then_block:
    y = 1;
    goto join;
join:
    x--;
    goto loop_head;
}
