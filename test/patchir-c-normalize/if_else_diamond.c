// RUN: %patchir-c-normalize -input %s -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=IFELSE %s --input-file %t.c
// IFELSE: if (
// IFELSE: else
// IFELSE-NOT: goto

// Pattern: BrTerm diamond → ConditionalStructurizePass (two-sided)
// RPO-ordered flat blocks:
//   entry: if (cond) goto then_block; else goto else_block;
//   else_block: <else body>; goto join;   ← else arm (false branch) comes first in RPO
//   then_block: <then body>; goto join;   ← then arm (true branch) comes second
//   join: <tail>

int if_else_diamond(int x) {
entry:
    if (x > 0) goto then_block; else goto else_block;
else_block:
    x = 0;
    goto join;
then_block:
    x = 1;
    goto join;
join:
    return x;
}
