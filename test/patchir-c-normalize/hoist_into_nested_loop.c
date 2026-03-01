// RUN: %patchir-c-normalize -input %s -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=NESTEDLOOP %s --input-file %t.c
// NESTEDLOOP: while (
// NESTEDLOOP: x = 1
// NESTEDLOOP: return x

// Pattern 1 — Loop-exit inlining into nested loop:
//
//    +------------------+
//    | outer_loop       |
//    |   +------------+ |
//    |   | inner_loop | |
//    |   |   ...      | |
//    |   |   if (p)   |--goto L_exit--+
//    |   |   ...      | |              |
//    |   +------------+ |              v
//    +------------------+      +-------------+
//                              | L_exit:     |
//                              | x = 1;      |
//                              | goto outer  |
//                              +-------------+
//
// When L_exit is control-equivalent to the goto site inside the inner loop,
// the body of L_exit (x = 1; goto outer) is inlined at the goto, replacing
// it with a break from the inner loop followed by the inlined body.
//
// Safety: L_exit has exactly one incoming goto from the inner loop. Every path
// through the inner loop that reaches the goto also reaches L_exit, and
// vice versa (dom + postdom).
//
// If incorrect: x = 1 would be lost on the exit path, or duplicated on
// paths that do not exit.
//
// Note: When nested-loop hoisting is fully implemented (recursion into loop
// bodies), the inner loop would be structurized and NESTEDLOOP-NOT: goto L_exit
// would apply.
//
// Structure mirrors test/AST/loop_nested_while.c but with L_exit as a
// forward label for the inner-loop exit (single ref from inner_exit).

int hoist_into_nested_loop(int rows, int cols) {
    int x = 0;
outer_head:
    if (rows > 0) goto outer_body; else goto outer_exit;
outer_exit:
    return x;
outer_body:
    goto inner_head;
inner_head:
    if (cols > 0) goto inner_body; else goto inner_exit;
inner_exit:
    goto L_exit;
inner_body:
    x = x + 1;
    cols = cols - 1;
    goto inner_head;
L_exit:
    x = 1;
    rows = rows - 1;
    goto outer_head;
}
