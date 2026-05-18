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

### F1 — `InlineGotoTarget` (clone label payload to the goto site) — IN PROGRESS

> Like F2, F1's 5 passes are used à la carte — no single fixed-order
> wrapper. F1 lands as sub-merges of always-adjacent pairs.
>
> **Sub-merge 1 — DONE.** `CloneTerminalLabelGotos` composition wrapper
> for `CloneFallthroughTerminalLabelGotos` + `CloneNoFallthroughTerminalLabelGotos`
> — two complementary passes (fallthrough-predecessor vs not), always
> invoked as an adjacent `Fallthrough → NoFallthrough` pair at four
> driver sites (2 schedule, 2 tail). `CloneNoFallthrough` is now
> internal; `CloneFallthrough` keeps one lone driver call (the
> switch-guarded tail site) to preserve call-site position. 81/81 lit,
> goto budget holds, `/patchir-inspect --debug --batch` VERDICT PASS
> (cloning verified correct on highest-clone-count fixtures).
> **Remaining:** `InlineSingleRefTerminalLabelBlocks`,
> `CloneCleanupLabelBeforeJoinGotos`,
> `CloneSmallStraightLineLabelBeforeJoinGotos` — all standalone,
> scattered; deferred behind Phase 4.

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

### F2 — `CollapseGotoForwarder` (fold a forwarder/single-ref region away) — IN PROGRESS

> **Scope correction (found while merging).** F2 cannot be a single
> fixed-order wrapper like F4/F6/F7. Its 7 passes are used **à la carte**
> — different subsets in different orders at each of the schedule and
> tail call sites. A single `CollapseGotoForwarder(all 7)` would move
> passes past non-family passes (the F4 hazard). F2 therefore lands as a
> sequence of sub-merges of always-adjacent-same-order pairs; the rest is
> blocked on either genuine fold-logic unification or the Phase 4
> worklist (which dissolves the hand-tuned à-la-carte ordering). The
> audit's "Local vs CrossCompound = same transform" line was an
> optimistic name-level guess: reading the code, `FoldLocalGotoDiamonds`
> (224 lines, iterates to fixpoint, two sub-cases) and
> `FoldCrossCompoundIfLabelDiamonds` (107 lines, single-pass) are two
> *distinct* algorithms — composable, not trivially unifiable.
>
> **Sub-merge 1 — DONE.** `FoldGotoDiamonds` composition wrapper for
> `FoldLocalGotoDiamonds` + `FoldCrossCompoundIfLabelDiamonds` (always
> adjacent, same order, all 3 driver sites — position-preserving).
> Both sub-functions internal. 81/81 lit, goto budget holds,
> `/patchir-inspect --debug --batch` VERDICT PASS.
> **Remaining:** `FoldConditionalFallthroughChains`,
> `FoldForwardSingleRefLabelRegions`, `SinkCommonTerminalEpilogues`,
> `FoldCrossCompoundDispatchChains`, `FoldGuardedJoinLabelChains` — no
> clean always-adjacent grouping; deferred behind Phase 4.

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
**1 driver pass — already a single canonical transform. Nothing to merge
on the Clang-AST layer.** The driver invokes it through two lambda modes
(`run_goto_to_next_label_fixed_point` / `_once`); that is one transform with
two call modes, not two passes. A second copy exists in the SNode layer —
dedup is Phase 5, not Phase 3. **F3 status: DONE (no-op — confirmed canonical).**

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
**1 driver pass — already a single canonical transform. Nothing to merge.**
The canonical name in the plan table is `ScopeifyConditionalGoto`; the code
keeps the existing name `ScopeifyIfGotos` (a rename would be cosmetic churn
with no structural value). **F5 status: DONE (no-op — confirmed canonical).**

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

### F8 — `FoldClangSwitchLocalCaseTargets` (switch-case-target goto fold)

| Driver pass | Lines | ~LoC |
|---|---|---|
| `FoldClangSwitchLocalCaseTargets` | 2950–3076 | 127 |

Helpers: the switch-local-label group, 2650–2949.
**1 driver pass — already a single canonical transform. Nothing to merge.**

**Placement decision (settled).** The audit left this open between "fold into
F2" and "its own family." It is **its own family, F8** — not part of F2:

- Mechanism differs. F2 `CollapseGotoForwarder` collapses goto-forwarders /
  single-ref label regions into their predecessor (a layout/forwarding
  transform). F8 redirects a goto whose target is a *switch-case-local* label
  into the switch itself. Different domain (switch-specific), different shape.
- Layer placement differs. Phase 5 classifies switch-case folding as a
  *structural region transform* — a candidate to live on the SNode tree —
  whereas F2's forwarder-collapses are layout/adjacency transforms that stay
  post-emission. Folding F8 into F2 would mix transforms Phase 5 must split to
  different layers. Keeping F8 separate keeps that decision clean.

**F8 status: DONE (no-op — confirmed canonical, its own family).**

### Unplaced / cosmetic

| Pass | Lines | Disposition |
|---|---|---|
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

## Phase 3 — final state

The 27 in-loop Clang-AST cleanup passes resolve to **8 canonical transform
families** plus the 5 standalone cosmetic passes and the `CleanupStmtTree`
prologue.

| Family | Canonical transform | Passes | Phase 3 outcome |
|---|---|---|---|
| F1 | `CloneTerminalLabelGotos` (+`InlineGotoTarget` deferred) | 5 | sub-merge 1 done (2→1); 3 deferred |
| F2 | `FoldGotoDiamonds` (+`CollapseGotoForwarder` deferred) | 7 | sub-merge 1 done (2→1); 5 deferred |
| F3 | `EliminateGotoToNextLabel` | 1 | already canonical |
| F4 | `RemoveDeadControlFlow` | 3 | merged (3→1) |
| F5 | `ScopeifyIfGotos` | 1 | already canonical |
| F6 | `HoistCrossScopeLabels` | 2 | merged (2→1) |
| F7 | `RecoverLoop` | 2 | merged (2→1) |
| F8 | `FoldClangSwitchLocalCaseTargets` | 1 | already canonical |

Commits: F4 `c99b39e`, F6 `2605c82`, F7 `78ca42f`, F2-1 `7194a85`,
F1-1 `b7d986d`. Every commit: 81/81 lit, goto budget holds,
`/patchir-inspect` VERDICT PASS.

**What Phase 3 achieved.** Five families (F3, F4, F5, F6, F7, F8) are now
each exactly one canonical transform. Driver-facing cleanup entry points
dropped by 9 (F4 −2, F6 −2, F7 −2, F2 −1, F1 −2). Every merged transform
carries an honest `Stmt::Profile` changed-contract — a Phase 4 prerequisite.

**What Phase 3 cannot finish without Phase 4.** F2 and F1 each retain a
remainder (5 and 3 passes) that resists consolidation: their passes are
invoked **à la carte** by the hand-unrolled driver tail — different subsets
in different orders per site — so no position-preserving wrapper can group
them. This is the hard dependency the reorder did not foresee: the *final*
F2/F1 consolidation needs the Phase 4 worklist to dissolve the à-la-carte
ordering first. The F2/F1 remainders are the explicit hand-off into Phase 4.

**Confluence lesson (carried from F4).** A family being internally confluent
does not license moving its passes past non-family passes. F4 needed a
dead-label-only carve-out (3 sites); F1 keeps one lone `CloneFallthrough`
site. F6/F7 and the two sub-merges were exactly position-preserving and
needed none. Phase 4's worklist must establish confluence for the surviving
~8 transforms before iterating them.
