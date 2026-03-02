// RUN: %patchir-c-normalize -input %s -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=MULTI %s --input-file %t.c
// MULTI: goto
// MULTI-NOT: else

// Pattern: goto preserved when structurization is blocked by an intermediate label.
// The ConditionalStructurizePass single-sided check requires NO intermediate labels
// between the if-stmt and the skip label.  Here 'middle:' is in that range, so the
// pass must not fire and the explicit goto must remain in the output.

int multi_pred_goto(int a, int b) {
first:
    if (a > 0) goto join;
middle:
    b = 0;
join:
    return b;
}
