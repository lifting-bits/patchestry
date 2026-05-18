# Structuring Cleanup Redesign Plan

## Context

The `kumarak/redesign_structuring` branch reduced emitted `goto` statements
from 222 (`main`) to 25 across the 79 patchir-decomp fixtures — a real, measured
89% win. But it bought that win with a maintainability problem this plan
addresses.

### The problem, with evidence

The decompilation pipeline now runs **two parallel control-flow cleanup
engines on two different IRs**:

```
JSON → CGraph → CFGStructure collapse → SNode tree
    → [SNode cleanup]      ~22 passes, scheduled by RunSNodeCleanupSchedule
                           (ASTConsumer.cpp), transaction-guarded via
                           SNodeRegion legality/ownership/rollback
    → ClangEmitter → Clang AST
    → [Clang-AST cleanup]  27 passes, CleanupPrettyPrint (ClangEmitterCleanup.cpp),
                           unguarded peephole, fake 8× "fixed point" + unrolled tail
    → C
```

Measured contribution (branch, identical 79 fixtures, counting emitted `goto`):

| Configuration | Emitted gotos |
|---|---|
| SNode cleanup ON, Clang-AST cleanup OFF | **196** |
| Both ON | **25** |
| (reference: `main` whole pipeline) | 222 |

**The Clang-AST layer performs ~87% of all goto elimination (196 → 25).** The
principled, transaction-guarded SNode layer does the minority. The redesign's
headline result is delivered by the *unprincipled* layer.

Further evidence (instrumented fire-counts over 116 functions):

- All 27 Clang-AST passes fire; **none is dead** — redundancy is structural overlap.
- ~10 passes return a rebuilt body on *every* call → the `if (body)` guard
  wrapping each pass is dead logic, and the "fixed-point" loop's convergence
  check never triggers (`RepairCrossScopeLabelEntries` fires exactly 8×/function
  = the hard cap). It is a fixed 8-iteration loop plus a 180-line hand-unrolled
  tail.
- The ~49 passes across both layers are really ~8 transforms duplicated per
  syntactic shape. Pass *names* are duplicated across the two layers:
  `EliminateGotoToNextLabel`, `ScopeifyIfGotos`, `RepairCrossScopeLabelEntries`,
  `FoldSwitchLocalCaseTargets`/`FoldClangSwitchLocalCaseTargets`; the SNode
  `Duplicate*Targets` family ≈ the Clang-AST `Clone*` family.

Cost: `ClangEmitterCleanup.cpp` grew 52 KB → 332 KB (6.4×); ~49 cleanup passes;
two IRs; no single source of truth for "eliminate a goto."

### Goal

Converge on **one cleanup engine, one IR, a real fixed point, ~8 parameterized
transforms** — without regressing the goto KPI (≤ 25) or the 80/80 lit suite.

## Non-goals

- Rewriting the `CFGStructure` graph-collapse algorithm (interval/loop-nest
  structuring). A stronger collapse would leave fewer residual gotos for *any*
  cleanup layer — tracked as future work (Phase 7), not part of this redesign.
- Changing the vector-slot `SNode` data model — it is a sound foundation and
  stays.

## Phases

Every phase ends green: Debug build clean, `llvm-lit` 81/81, goto budget holds,
one atomic commit. Phases 0–2 deliver standalone value even if 3–6 stall.

> **Ordering note.** Consolidation (Phase 3) now runs *before* the worklist
> driver (Phase 4) — the reverse of the original plan. Rationale in Phase 3.

### Phase 0 — Guardrails (prerequisite)

The `C-NOT: goto` FileCheck assertions were removed earlier, so the 25-goto
result is currently unpinned. Before touching the engine:

1. Add a regression guard — a lit/script test that runs all fixtures, sums
   emitted `goto`, and asserts a total budget (≤ 25, or per-fixture budgets).
2. Promote the throwaway instrumentation into a permanent gated diagnostic:
   a `-cleanup-stats` flag dumping per-stage goto counts (post-collapse,
   post-SNode-cleanup, post-Clang-cleanup) and per-pass fire counts.
3. Commit a baseline snapshot: per-fixture goto at each stage.

**Exit:** KPI is pinned in CI and every later phase is measurable.

### Phase 1 — Diagnose the SNode layer (the decision phase) — DONE

**Status: complete.** Full results in
`docs/structuring_cleanup_phase1_findings.md`.

The pivotal question was: *why does the SNode cleanup eliminate only ~16% of
gotos?* The `ShouldApplySNodeRewrite` preflight gate was instrumented and
measured over all fixtures.

Verdicts:
- **H1 (timid legality/ownership gating) — REJECTED.** The gate accepts 85% of
  rewrite requests; all 15% rejections are genuine legality constraints, zero
  are profitability, zero transactions roll back. There is no over-caution to
  relax.
- **H2 (the SNode tree cannot see post-linearization adjacency) — CONFIRMED**
  as the dominant cause. `EliminateGotoToNextLabel` is the hottest Clang-AST
  pass (3852 fires); "goto to the next label" is a linear-layout property a
  tree of nested body slots structurally cannot expose.
- **H3 (SNode passes incomplete) — minor**, not primary.

**Decision: hybrid. Direction A is rejected** — a post-emission cleanup layer
is architecturally necessary, not debt. The redesign therefore:
- keeps a post-emission layer, rebuilt as one consolidated engine (Phases 2–3);
- trims the SNode cleanup layer to genuinely structural rewrites;
- resolves cross-layer duplicate passes *asymmetrically* — layout/adjacency
  transforms live post-emission, structural transforms live on the tree.

### Phase 2 — Honest changed-contract — DONE

Enforce the changed-contract: every pass returns "changed" only when it
actually mutated, so the `if (body)` guard becomes meaningful (a prerequisite
for any real fixed point).

**Status: complete.** Converted 9/10 always-rebuild passes to an honest
`bool &mutated` contract (commits `f56784d`, `3e64290`, `405f0a4`).
`CleanupStmtTree` was left alone — it is a one-shot prologue, not part of the
loop. 81/81 lit holds.

> Phase 0's "10 always-rebuild passes" undercount was found here: passes
> invoked N×/function (`RepairCrossScopeLabelEntries`,
> `FoldClangSwitchLocalCaseTargets`, the `Fold*Diamonds` group) are *also*
> always-rebuild and were missed. The full always-rebuild set is only knowable
> after consolidation.

### Phase 3 — Consolidate transform families — DONE (pre-Phase-4 ceiling)

**Reorder rationale.** The original plan put the worklist driver (old Phase 2b)
*before* consolidation. A Phase 2b attempt was built — a `Stmt::Profile`
fingerprint worklist running to a structural fixed point — and **reverted**: it
regressed `decode_basic_field` (dropped `if (param_2 == 5){... pb_decode_fixed32
...}` blocks). Root cause: **the cleanup transforms are not confluent.** The
hand-tuned "schedule ×8 then tail ×1" order is load-bearing; iterating the tail
to a fixpoint converges to a *different, wrong* result. Building a worklist over
a non-confluent transform set is backwards. Consolidate first — the ~8 survivors
are few enough to reason about confluence directly, then a worklist over them is
cheap and safe. Consolidation therefore moves ahead of the driver work.

Collapse ~49 passes into ~8 canonical, parameterized transforms, on the
post-emission Clang-AST layer (per the Phase 1 hybrid decision), family by
family (each family = one build + budget check):

| Canonical transform | Subsumes |
|---|---|
| `InlineGotoTarget` (profitability-parameterized) | 4 Clang `Clone*` + SNode `Duplicate*Targets` (~9 passes) |
| `CollapseGotoForwarder` | 6 Clang `Fold*` + SNode `Fold*` / `CollapsePassThroughLabels` (~9 passes) |
| `EliminateGotoToNextLabel` | the two duplicated copies → one |
| `RemoveDeadControlFlow` | `RemoveDeadLabels` + `RemoveEmptyBlocks` + `RemoveOrphanedGotos` |
| `ScopeifyConditionalGoto` | `ScopeifyIfGotos` (deduped across layers) |
| `HoistCrossScopeLabels` | `Hoist*`/`RepairCrossScopeLabelEntries` (deduped) |
| `RecoverLoop` | `PromoteLocalBackwardGotoLoops`, `ConvertImmediateLoopExitGotosToBreak` |
| (cosmetic, kept separate) | `NormalizeConditions`, `PromoteSimpleCounterWhileToFor` |

The `Local` vs `CrossCompound` pass split disappears: it exists only because the
Clang AST makes one control-flow shape look different across `CompoundStmt`
boundaries — a non-issue on the SNode tree.

**Exit:** ~8 transforms; goto budget holds at each family merge.

**Status: DONE to the pre-Phase-4 ceiling.** Full record in
`docs/structuring_cleanup_phase3_audit.md`. The 27 in-loop Clang-AST passes
resolve to 8 families (F1–F8). F3/F4/F5/F6/F7/F8 are each now one canonical
transform; F4/F6/F7 were merges (commits `c99b39e`, `2605c82`, `78ca42f`),
F3/F5/F8 were already canonical. F1 and F2 each got a first sub-merge
(`b7d986d`, `7194a85`) but retain a remainder (3 and 5 passes) that **cannot**
be consolidated by Phase 3 alone: the hand-unrolled driver tail invokes those
passes *à la carte* — different subsets and orders per call site — so no
position-preserving wrapper can group them. Two correctness facts surfaced
that the original plan missed: (1) family-internal confluence does **not**
license moving a pass past non-family passes (F4 needed a carve-out); (2) the
F2/F1 remainders are gated on Phase 4 dissolving the à-la-carte ordering — so
consolidation and the worklist are *mutually* entangled, not strictly
sequential. Every Phase 3 commit held 81/81 lit, the goto budget, and
`/patchir-inspect` VERDICT PASS.

### Phase 4 — Real fixed-point worklist driver — PARTIAL (worklist refuted)

Original goal: replace the fake 8× loop + 180-line unrolled tail in
`CleanupPrettyPrint` with a genuine worklist fixed point.

**Outcome (see Status below): half achieved, half refuted.** The "fake 8×
loop" half *was* a fake loop and is now a genuine fixed point (step 1). The
"unrolled tail" half is **not** a fake loop — step 3 measured that iterating
its transforms to a fixed point diverges catastrophically — so it stays. The
"replace the unrolled tail with a worklist" goal is withdrawn as empirically
refuted.

This is the old "Phase 2b", moved after consolidation. Confluence must be
established for the ~8 survivors *first* — verify the order-independence the
Phase 2b attempt assumed and the old 49-pass set lacked. If a pair is provably
non-confluent, fix it (pick a canonical orientation) rather than freezing an
order; a worklist over a still-non-confluent set will reproduce the
`decode_basic_field` regression.

**Inherited from Phase 3.** Phase 4 also absorbs the F2/F1 consolidation
remainders: once the worklist replaces the à-la-carte unrolled tail, the
5 residual F2 folds (`FoldConditionalFallthroughChains`,
`FoldForwardSingleRefLabelRegions`, `SinkCommonTerminalEpilogues`,
`FoldCrossCompoundDispatchChains`, `FoldGuardedJoinLabelChains`) and the
3 residual F1 inline/clone passes (`InlineSingleRefTerminalLabelBlocks`,
`CloneCleanupLabelBeforeJoinGotos`, `CloneSmallStraightLineLabelBeforeJoinGotos`)
become worklist entries and the per-site ordering constraint disappears.
Finishing F2/F1 is therefore a Phase 4 deliverable, not a Phase 3 gap.

**Exit (revised):** the schedule loop is a real fixed point (done); the
unrolled tail stays (worklist refuted); 81/81 lit; goto budget holds.

**Status: steps 1–3 DONE; worklist refuted.** Done incrementally rather than
as a big-bang (the Phase 2b big-bang regressed `decode_basic_field`).

- *Step 1 — real schedule fixed point* (commit `42149ac`). The schedule
  loop's convergence check moved from body-pointer identity (never
  converged — always-rebuild passes bumped the pointer, so it always ran the
  cap) to a `Stmt::Profile` structural fingerprint. Output-identical: a
  schedule pass reporting no change is a true fixed point and the schedule is
  deterministic. The schedule now converges in 1–4 iterations (cap 8) across
  all fixtures, and **none oscillate** — positive confluence evidence for the
  schedule subset. Count exported as `schedule_iterations=` in
  `CLANG_CLEANUP_SUMMARY`.
- *Step 2 — cross-pass confluence audit* DONE (commit `70467b7`). Recorded in
  `docs/structuring_cleanup_phase4_confluence_audit.md`. It classified most of
  the tail as confluent and gave step 3 a (later-refuted) GREEN.
- *Step 3 — tail worklist: IMPLEMENTED then REFUTED* (commit `ec4a9b8`). The
  three-phase worklist was built behind a default-off `-cleanup-worklist`
  flag and run through the mandated empirical gate (per-fixture diff vs the
  unrolled tail, all 80 fixtures). The gate **FAILED**: 9 fixtures diverge
  catastrophically — the worklist deletes ~80% of `decode_basic_field`,
  over-clones `cwe22_init_logger` 3–5×, regresses `cve_2016_6563`'s goto KPI.
  The step-2 verdict was wrong: the schedule loop converging proves only that
  the *fixed-order composition* converges, not that the transforms are
  confluent under free re-iteration. The unrolled tail is a deliberately-
  ordered run-once pipeline; a worklist is the wrong model. The flag and
  worklist branch were then **removed** (commit `48371a0`) — the gate result
  is recorded in the confluence-audit doc; the dead branch was not kept.
  Detail in the confluence-audit doc, "Step 3 result" section.

**Net Phase 4 result.** Step 1 is a real, kept win — the schedule loop is now
a genuine fixed point. The tail-worklist goal is withdrawn. Any future tail
rework must make individual transforms idempotent / iteration-safe, one at a
time — not a wholesale worklist.

### Phase 5 — Resolve the cross-layer duplication — REFUTED

Original plan: place each cross-layer-duplicated transform in exactly one
layer and delete the other copy — adjacency transforms keep the Clang-AST
copy and drop the inert SNode copy; structural transforms keep the SNode copy
and drop the Clang-AST duplicate.

**Outcome: the premise is refuted by measurement.** Full record in
`docs/structuring_cleanup_phase5_findings.md`. The flagship "delete the SNode
copy" candidate — the SNode-layer `EliminateGotoToNextLabel`, which Phase 1
argued the tree "structurally cannot expose" so should be inert — was removed
from the SNode cleanup schedule and tested: **2 lit failures, including the
`zz-goto-budget` guard.** The SNode copy is load-bearing, not inert.

The reason generalizes: the SNode tree *can* see sibling-order adjacency
within an `SSeq` (the SNode copy eliminates those gotos), and the Clang-AST
copy handles the cases that only become adjacent after linearization. Same
name, two IRs, **disjoint and complementary** adjacency — not duplication.
Corroborated by Phase 1's own finding that no pass on either layer is dead.
The "two parallel engines doing duplicate work" framing was a high-level
Phase-1 impression that does not survive per-pass measurement.

**Verdict:** no safe cross-layer deletion exists; the delete-one-copy goal is
withdrawn. The shared *names* signal a shared *idea*, not redundant work. The
intentional two-IR design should instead be **documented** (Phase 6).

**Exit (revised):** Phase 5 produces no code change; its deliverable is the
findings doc and the architecture note carried into Phase 6.

### Phase 6 — Split monoliths, document — DOCUMENTATION DONE; SPLIT DEFERRED

- **Documentation — DONE.** `AGENTS.md` now carries a "`patchir-decomp`
  structuring and cleanup pipeline" subsection: the CGraph → CFGStructure →
  SNode → ClangEmitter flow, the *two intentional cleanup layers* (and why
  they are complementary, not duplicate — the Phase 5 finding), the 8
  Clang-AST transform families, the schedule fixed point + run-once tail, and
  the goto-budget guard. The `docs/structuring_cleanup_*.md` set is linked
  from "Related Docs". Note: the plan originally said "single-engine
  pipeline" — that wording is dropped; Phase 5 established the two layers are
  both load-bearing, so the docs describe the real two-layer design.
- **File split — DEFERRED.** Splitting `ClangEmitterCleanup.cpp` (7,574 lines)
  and `CFGStructure.cpp` (8,725 lines) by family is a large, dedicated
  refactor: nearly all of `ClangEmitterCleanup.cpp`'s passes live in one
  anonymous namespace (lines 782–7057), so a split also requires restructuring
  that namespace and introducing internal headers for the shared helpers. It
  is pure code motion with **zero behavioral value** and produces a very large
  diff. Recommended as its own standalone PR, not bundled with the redesign.

### Phase 7 — Strengthen CGraph collapse (future, separate effort)

A stronger `CFGStructure` collapse (proper interval/loop-nest structuring,
irreducible-region handling) emits cleaner control flow directly, shrinking the
residual gotos any cleanup layer must handle. Tracked separately.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Goto KPI regresses mid-migration | Phase 0 budget guard in CI; per-phase verification; atomic, revertable commits |
| Trimming a tree-level SNode pass drops a goto it was quietly handling | Phase 4 deletes tree-level copies family-by-family, each gated by the goto budget |
| Clang-AST layer is unguarded per-rewrite (no rollback) | Keep `-verify-no-node-loss` in CI; consider making it default-on |
| Effort overrun | Phases 0–2 are independently valuable; 3–5 can land incrementally, one family per PR |

## Success metrics

| Metric | Now | Target |
|---|---|---|
| Cleanup passes | ~49 across 2 layers | ~8, one engine |
| Cleanup IRs | 2 (SNode tree + Clang AST) | 1 (+ tiny cosmetic pass) |
| Duplicated pass names across layers | ~5 | 0 |
| Driver | fake 8× loop + unrolled tail | real worklist fixed point |
| `ClangEmitterCleanup.cpp` | 332 KB | < 50 KB (or deleted) |
| Goto KPI | 25 | ≤ 25, guarded by CI |

## Verification protocol (every phase)

```sh
cmake --build builds/default --config Debug --target patchir-decomp
builds/default/bin/llvm-lit builds/default/test/patchir-decomp/ --no-progress-bar   # 80/80
# goto budget guard (Phase 0 artifact) must pass
```

Rough sizing: Phase 0 ~0.5d, Phase 1 ~2–3d, Phase 2 ~2d, Phase 3 ~1wk,
Phase 4 ~3d, Phase 5 ~3d, Phase 6 ~2d — on the order of 4–5 weeks total,
front-loaded value.
