// RUN: %patchir-c-normalize -input %s -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=IFTHEN %s --input-file %t.c
// IFTHEN: if (
// IFTHEN-NOT: goto
// IFTHEN-NOT: else

// Pattern: single-sided BrTerm → ConditionalStructurizePass (no-else arm)
// Structure: if (cond) goto skip; <fallthrough body>; skip: <tail>
// The pass recognises this as: if (!cond) { <fallthrough body> }
// Condition: no intermediate labels between the if and the skip label.

void if_then_only(int x) {
    if (x <= 0) goto skip;
    x = x * 2;
skip:
    return;
}
