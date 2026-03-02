// RUN: %patchir-c-normalize -input %s -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=NESTEDIF %s --input-file %t.c
// NESTEDIF: if (
// NESTEDIF: y = 1
// NESTEDIF: return

// Pattern 3 — Label body into nested if-branch:
//
//    +----------------+
//    | if (outer)     |
//    +----T-----F-----+
//         |     |
//         v     +---> ...
//    +----------------+
//    | if (inner)     |
//    +----T-----F-----+
//         |     |
//         v     v
//    goto L  fall
//    L_inner: ...
//      y = 1
//      ...
//
// When L_inner is control-equivalent to the then-branch of the inner if
// (the branch containing the goto), the body of L_inner can be moved into
// that branch, replacing the goto.
//
// Safety: The goto to L_inner is the only path into L_inner from this
// region. L_inner post-dominates the inner if-head and the inner if-head
// dominates L_inner on that path.
//
// If incorrect: y = 1 would be skipped when inner is true, or duplicated
// when inner is false.
//
// RPO diamond: else before then, both goto join. Join has y=1 and return.

int hoist_nested_if(int outer, int inner) {
    int y = 0;
entry:
    if (outer > 0) goto then_arm; else goto else_arm;
else_arm:
    return y;
then_arm:
    if (inner > 0) goto inner_then; else goto inner_else;
inner_else:
    goto L_join;
inner_then:
    goto L_inner;
L_inner:
    y = 1;
L_join:
    return y + 1;
}
