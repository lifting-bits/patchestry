// RUN: %patchir-c-normalize -input %s -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=LOOP %s --input-file %t.c
// LOOP:     while (
// LOOP:         switch (
// LOOP:         case
// LOOP:         break
// LOOP-NOT: goto

// Pattern: while loop whose body dispatches via a switch-goto table.
// The loop head label wraps a non-IfStmt sub-stmt (cmd assignment), and the
// loop-condition IfStmt is a separate flat stmt — WhileLoopStructurizePass
// must include the head's sub-stmt in the while body, skip the dead IfStmt in
// the outer rewritten list, and SwitchGotoInliningPass must inline case bodies
// replacing trailing goto-to-join with break.

int loop_with_switch_goto_cases(int n) {
    int result = 0;
    int cmd;
loop_head:
    cmd = n % 4;
    if (n > 0) goto loop_body; else goto loop_exit;
loop_exit:
    return result;
loop_body:
    switch (cmd) {
        case 0: goto op_inc;
        case 1: goto op_double;
        case 2: goto op_reset;
        default: goto op_nop;
    }
op_nop:
    goto loop_continue;
op_reset:
    result = 0;
    goto loop_continue;
op_double:
    result = result * 2;
    goto loop_continue;
op_inc:
    result = result + 1;
loop_continue:
    n = n - 1;
    goto loop_head;
}
