# Phase 3 Audit — ClangEmitterCleanup.cpp pass-to-family mapping

Audit of the post-emission Clang-AST cleanup layer (`ClangEmitterCleanup.cpp`,
7417 lines) ahead of the Phase 3 family consolidation. Scope per the Phase 1
hybrid decision: the Clang-AST layer only; the SNode-layer duplicates are
resolved later in Phase 5 (cross-layer resolution).

## Method

Every top-level function in the file was enumerated and split into two classes:

- **Driver passes** — invoked from `CleanupPrettyPrint` (the driver, lines
  7058–7415). 27 of them, plus the one-shot `CleanupStmtTree` prologue.
- **Helpers** — recursion bodies / sub-rewrites private to one driver pass.

`MeasureClangCleanup` (7029) and `CleanupReportName` are instrumentation, not
transforms, and stay as-is.

## Family mapping

Eight target families per the Phase 3 plan table. Line ranges are
`[def-start, next-def-start)`.

### F1 — `InlineGotoTarget` (clone label payload to the goto site)

| Driver pass | Lines | ~LoC |
|---|---|---|
| `InlineSingleRefTerminalLabelBlocks` | 3765–3877 | 113 |
| `CloneFallthroughTerminalLabelGotos` | 4121–4199 | 79 |
| `CloneNoFallthroughTerminalLabelGotos` | 4200–4357 | 158 |
| `CloneCleanupLabelBeforeJoinGotos` | 4435–4532 | 98 |
| `CloneSmallStraightLineLabelBeforeJoinGotos` | 4533–4642 | 110 |

Helpers: `CloneStraightLineSeqAsStmt` (3878), `CloneGotoFreeSeqAsStmt` (3936),
`ReplaceGotoWithClonedJoinTail` (3948), `ReplaceGotoWithClonedGotoFreeSeq`
(4036), `ReplaceGotoWithClonedCleanupThenJoin` (4371),
`SeqHasUnsafeGuardedJoinMoveControl` (4358).
**5 driver passes — the largest family, and the highest-risk (this is the
clone-payload family the reverted Phase 2b regression came from).**

### F2 — `CollapseGotoForwarder` (fold a forwarder/single-ref region away)

| Driver pass | Lines | ~LoC |
|---|---|---|
| `FoldConditionalFallthroughChains` | 3202–3285 | 84 |
| `FoldForwardSingleRefLabelRegions` | 3286–3469 | 184 |
| `SinkCommonTerminalEpilogues` | 3696–3764 | 69 |
| `FoldGuardedJoinLabelChains` | 4853–4920 | 68 |
| `FoldCrossCompoundIfLabelDiamonds` | 4921–5027 | 107 |
| `FoldCrossCompoundDispatchChains` | 5333–5436 | 104 |
| `FoldLocalGotoDiamonds` | 5702–5927 | 226 |

Helpers: the `TryFold*` group (4643, 4749, 4803, 5047, 5190),
`BuildDispatchGuardChain` (5028), `Sink*`/epilogue helpers (3470–3695).
**7 driver passes. `Local` vs `CrossCompound` is the artificial split the plan
calls out — same transform, different `CompoundStmt` nesting.**

### F3 — `EliminateGotoToNextLabel` (drop goto to immediately-following label)

| Driver pass | Lines | ~LoC |
|---|---|---|
| `EliminateGotoToNextLabel` (wrapper + `ProcessCompound` + overload) | 1572–1856 | 285 |

Helper: `StripGotoToFollowingLabelFromTailPosition` (865–1021).
**1 driver pass. Plan note: a second copy exists in the SNode layer — dedup is
Phase 5, not here.**

### F4 — `RemoveDeadControlFlow` (delete unreachable labels/gotos/blocks)

| Driver pass | Lines | ~LoC |
|---|---|---|
| `RemoveDeadLabels` | 358–476 | 119 |
| `RemoveEmptyBlocks` | 477–836 | 360 |
| `RemoveOrphanedGotos` | 5950–6108 | 159 |

Helpers: `CollectGotoTargets` (43), `CollectDefinedLabels` (5928).
**3 driver passes. Pure monotone deletion — idempotent and mutually confluent.**

### F5 — `ScopeifyConditionalGoto`

| Driver pass | Lines | ~LoC |
|---|---|---|
| `ScopeifyIfGotos` | 1917–2102 | 186 |

Helper: `ClangScopeifyLabelsAreRegionLocal` (1872).
**1 driver pass.**

### F6 — `HoistCrossScopeLabels`

| Driver pass | Lines | ~LoC |
|---|---|---|
| `HoistCrossScopeLabelEntries` | 2453–2548 | 96 |
| `RepairCrossScopeLabelEntries` | 2549–2649 | 101 |

Helpers: `FindDirectNestedEntryLabel` (2144), `CollectScopedControlTransfers`
(2239), `ExtractFirstNestedTargetLabel` (2356).
**2 driver passes.**

### F7 — `RecoverLoop`

| Driver pass | Lines | ~LoC |
|---|---|---|
| `ConvertImmediateLoopExitGotosToBreak` | 5491–5570 | 80 |
| `PromoteLocalBackwardGotoLoops` | 5598–5693 | 96 |

Helpers: `ReplaceGotoWithBreakInCurrentLoop` (5437),
`LocalLoopRegionHasUnsafeControl` (5571), `SeqHasLocalLabelOrControl` (5694).
**2 driver passes.**

### Unplaced / cosmetic

| Pass | Lines | Disposition |
|---|---|---|
| `FoldClangSwitchLocalCaseTargets` | 2950–3076 | **Open** — `Fold*` by name but redirects gotos into switch cases. Candidate F2, or its own F8 switch family. Decide before merging F2. Helpers 2650–2949. |
| `PromoteSimpleCounterWhileToFor` | 6355–6424 | Cosmetic, keep separate (plan). |
| `RemoveRedundantTerminalForContinues` | 6425–6506 | Cosmetic. |
| `PushLabelsIntoCompounds` | 6507–6638 | Cosmetic / layout. |
| `AttachEmptyLabelsToFollowingStmt` | 6639–6785 | Cosmetic / layout. |
| `NormalizeConditions` | 6995–7028 | Cosmetic, runs last (plan). |
| `CleanupStmtTree` | 188–357 | One-shot prologue, not in the loop. Untouched. |

## Tally

27 in-loop driver passes → F1:5, F2:7, F3:1, F4:3, F5:1, F6:2, F7:2,
switch:1, cosmetic:5 — collapsing to ~8 canonical transforms + ~5 cosmetic
passes left standalone.

## Recommended first family: F4 `RemoveDeadControlFlow`

Reasons:

1. **Lowest risk.** All three are pure deletion — they only remove dead
   labels, empty `CompoundStmt`/`NullStmt`, and gotos with no defined target.
   They cannot drop a live block, so they cannot reproduce the Phase 2b
   `decode_basic_field` regression.
2. **Provably confluent.** Monotone deletion over a finite node set: order
   between the three does not change the fixpoint. This is the one family
   where confluence needs no argument — a clean proof-of-process.
3. **Validates the merge harness first.** It exercises the full Phase 3 loop —
   merge into one parameterized transform, rebuild, run lit (81/81), check the
   goto budget — on a family that cannot fail for semantic reasons. The
   harness is then trusted before F1/F2 (the dangerous clone/fold families).
4. **Self-contained.** Helpers `CollectGotoTargets` / `CollectDefinedLabels`
   are leaf utilities; no entanglement with other families.

Proposed F4 shape: one `RemoveDeadControlFlow(ctx, body)` doing a single
post-order walk that drops dead labels, orphaned gotos, and empty blocks in
one pass, returning a real changed-flag (Phase 2 contract). The driver's
`run_remove_dead_labels` / `run_remove_empty_blocks` / `RemoveOrphanedGotos`
call sites collapse to one.

After F4 proves the process: F6 (2 passes), F7 (2 passes), then the hard
ones — F2 (7 passes, the `Local`/`CrossCompound` split) and F1 (5 clone
passes), each with an explicit confluence check before merge.

## F4 merge — DONE

`RemoveDeadControlFlow(ctx, body, bool &mutated)` added before
`CleanupPrettyPrint`, composing the three sub-rewrites in fixed order
(dead labels → orphaned gotos → empty blocks) with an honest changed-flag
detected via `Stmt::Profile`. The three sub-functions are now internal
(no driver-facing callers). 81/81 lit; goto budget holds; `/patchir-inspect
--debug --batch` VERDICT PASS (0 lost calls/conditions/blocks).

**Confluence caveat discovered.** The three F4 passes are confluent *with
each other*, but the family is **not** position-independent w.r.t. non-F4
passes. The first attempt collapsed every F4 call site into the full trio;
this moved `RemoveEmptyBlocks` (which merges `if`/`else`) to three legacy
tail sites that historically ran a dead-label sweep only, reshaping the
tree ahead of downstream goto-elimination and regressing
`cve_2016_6563_fun_0000b920` (0 → 2 gotos). Fix: those three sites keep a
`run_remove_dead_labels()` dead-label-only sweep, to be folded into
`run_dead_control_flow()` once Phase 4 removes the downstream order-
dependence. Lesson for F1/F2/F5–F7: a family being internally confluent
does **not** license moving its passes past non-family passes — preserve
call-site position.

## F6 merge — DONE

`HoistCrossScopeLabels(ctx, fn, body, bool &mutated)` added after
`RemoveDeadControlFlow`, composing `RepairCrossScopeLabelEntries` then
`HoistCrossScopeLabelEntries` (refs computed internally) with an honest
changed-flag via `Stmt::Profile`. Both sub-functions are now internal.

Two driver sites: the schedule's two adjacent steps (Repair, Hoist)
collapsed into one; the tail Hoist-only call also moved to the full
transform. Unlike F4, adding the sibling pass (Repair) at the tail
Hoist-only site did **not** regress — verified, so F6 is a clean full
merge with no position-preservation carve-out. 81/81 lit, goto budget
holds, `cve_2016_6563_fun_0000b920` rc=0 in goto/struct/CIR modes,
`/patchir-inspect --debug --batch` VERDICT PASS.

## F7 merge — DONE

`RecoverLoop(ctx, body, bool &mutated)` added after `HoistCrossScopeLabels`,
composing `ConvertImmediateLoopExitGotosToBreak` then
`PromoteLocalBackwardGotoLoops` (refs computed internally) with an honest
changed-flag via `Stmt::Profile`. Both sub-functions are now internal.

Two driver sites — the schedule's two adjacent steps (Convert, Promote)
collapsed into one, and the tail Convert+Promote block collapsed into one.
Both were already adjacent and in the same order, so the merge is exactly
position-preserving — a clean full merge, no carve-out. 81/81 lit, goto
budget holds, `/patchir-inspect --debug --batch` VERDICT PASS (loops
correctly recovered into while/for; 0 lost calls/conditions/blocks).
