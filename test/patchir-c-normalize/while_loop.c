// RUN: %patchir-c-normalize -input %s -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=WHILE %s --input-file %t.c
// WHILE: while (
// WHILE-NOT: goto

// Pattern: goto back-edge → WhileLoopStructurizePass
// Flat-block structure mirrors P-Code lifter output:
//   block_head: if (cond) goto body; else goto exit;  ← RPO: exit precedes body
//   block_exit: <post-loop>
//   block_body: <loop body>; goto block_head;         ← back-edge

void while_loop(int x) {
block_head:
    if (x > 0) goto block_body; else goto block_exit;
block_exit:
    return;
block_body:
    x--;
    goto block_head;
}
