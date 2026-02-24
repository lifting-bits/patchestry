// RUN: %patchir-c-normalize -input %s -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=MERGE %s --input-file %t.c
// MERGE: if (
// MERGE-NOT: goto out
// MERGE: x = 42

// RUN: %patchir-c-normalize -input %s -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=MERGE2 %s --input-file %t.c
// MERGE2: if (
// MERGE2-NOT: goto merge2_yes
// MERGE2: z = 99

// Pattern 1: two consecutive "if (c) goto out;" with the same target label must
// be merged into "if (c0 || c1) goto out;" and then structurized by the
// existing single-sided skip region path.

int if_merge_same_target(int a, int b) {
    int x = 0;
    if (a > 0) goto out;
    if (b > 0) goto out;
    x = 42;
    return x;
out:
    return x + 100;
}

// Pattern 2: plain if + two-sided if with same then-target must be merged into
// "if (c0 || c1) goto L; else goto M;" preserving the else arm.

int if_merge_two_sided_terminator(int a, int b) {
    int z = 0;
    if (a > 0) goto merge2_yes;
    if (b > 0) goto merge2_yes;
    else goto merge2_no;
merge2_no:
    z = 99;
    return z;
merge2_yes:
    return z + 1;
}
