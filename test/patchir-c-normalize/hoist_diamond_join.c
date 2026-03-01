// RUN: %patchir-c-normalize -input %s -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=DIAMOND %s --input-file %t.c
// DIAMOND: if (
// DIAMOND: else
// DIAMOND: x = 2
// DIAMOND-NOT: goto L_join
// DIAMOND-NOT: L_join:

// Pattern 2 — Diamond-goto join absorption:
//
//    +----------+
//    | if (c)   |
//    +--T---F---+
//       |   |
//       v   v
//     L_t: fall
//     ...  ...
//     goto goto
//     L_j  L_j
//       |   |
//       +---+
//       v
//    +----------+
//    | L_join:  |  <-- post-dominates if-head; head dominates L_join
//    | x = 2    |
//    +----------+
//
// The join code (x = 2) is control-equivalent to the if-head. Both branches
// reconverge at L_join before any other exit. The join body is moved after
// the structured if-else, absorbing the label.
//
// Safety: L_join post-dominates the if condition and the if-head dominates
// L_join. Every path through the if reaches L_join exactly once.
//
// If incorrect: x = 2 would be lost (wrong result) or executed multiple
// times (duplication on one branch).

int hoist_diamond_join(int c) {
    int x = 0;
entry:
    if (c > 0) goto L_then; else goto L_else;
L_else:
    x = 1;
    goto L_join;
L_then:
    x = 0;
    goto L_join;
L_join:
    x = 2;
    return x;
}
