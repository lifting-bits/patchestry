// RUN: %patchir-c-normalize -input %s -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=HOIST %s --input-file %t.c
// HOIST: while (
// HOIST: x =
// HOIST: y =

// Pattern: HoistControlEquivalentStmtsIntoLoopPass must NOT absorb op_target when
// a surviving label (survive) implicitly falls through to it.
//
// How the bug trigger is constructed:
//   (1) loop_exit block has `goto survive; n=-1;` where n=-1 is dead code.
//   (2) After WhileLoopStructurizePass, loop_exit's label is stripped by
//       AstCleanupPass (no explicit goto to loop_exit remains after structuring);
//       `goto survive` becomes a top-level statement.
//   (3) DeadCfgPruningPass removes n=-1 (dead after the unconditional `goto survive`).
//   (4) GotoCanonicalizePass sees `goto survive;` now immediately before `survive:`
//       and removes the trivial jump.  survive: is left as a LabelStmt with
//       ref_count == 0 (no remaining explicit gotos).
//   (5) When HoistPass runs, survive is still a LabelStmt that falls through to
//       op_target.  Phase 2 of the fix detects this and prevents absorption.
//
// Without the Phase 2 guard, op_target would be absorbed (total_inlined == 1 ==
// ref_count == 1, no explicit goto targets in replacement body) and y = 2 would
// vanish from the normal loop-exit path.

int hoist_loop_fallthrough_guard(int n) {
    int x = 0, y = 0;
loop_head:
    if (n > 0) goto loop_body; else goto loop_exit;
loop_exit:
    goto survive;
    n = -1;   /* dead code: unreachable after goto survive; pruned so that
                  goto survive becomes adjacent to survive: for canonicalization */
survive:
    x = 1;
loop_body:
    n--;
    if (n > 1) goto op_target;
    goto loop_head;
op_target:
    y = 2;
    return x + y;
}
