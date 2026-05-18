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
  cleanup layer — tracked as future work (Phase 6), not part of this redesign.
- Changing the vector-slot `SNode` data model — it is a sound foundation and
  stays.

## Phases

Every phase ends green: Debug build clean, `llvm-lit` 80/80, goto budget holds,
one atomic commit. Phases 0–2 deliver standalone value even if 3–5 stall.

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

### Phase 2 — Real fixed-point driver

Independent of the Phase 4 layer-resolution work; fixes the post-emission
(Clang-AST) driver.

1. **Phase 2a** — Enforce the changed-contract: every pass returns "changed"
   only when it actually mutated. Fix the ~10 always-rebuild passes so the
   `if (body)` guard becomes meaningful (prerequisite for a real fixed point).
2. **Phase 2b** — Replace the fake 8× loop + 180-line unrolled tail with a
   genuine worklist fixed point: run transforms until none reports a change.

**Exit:** convergence iteration count drops; 80/80 lit; goto budget holds.

### Phase 3 — Consolidate transform families

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

### Phase 4 — Resolve the cross-layer duplication (hybrid)

Phase 1 rejected the "delete one whole layer" framing — both layers are
needed. Instead, place each transform in exactly one layer, decided by class:

- **Layout / adjacency transforms** (`EliminateGotoToNextLabel`, cross-compound
  forwarder folds): keep the post-emission Clang-AST copy; delete the
  tree-level SNode copy — the tree cannot see the adjacency, so the SNode copy
  spins without effect.
- **Purely structural region transforms** (clone/move payload, switch-case
  folding): keep the SNode copy (its gate accepts 85%); delete the Clang-AST
  duplicate.
- Result: every duplicated pass name collapses to one definition, in the layer
  that can actually do the work.

**Exit:** no cross-layer duplicate passes; goto budget holds; 80/80 lit.

### Phase 5 — Split monoliths, document

- Split the surviving cleanup file by transform family (the current 7,337-line
  `ClangEmitterCleanup.cpp` / ~9,000-line `CFGStructure.cpp` are unmaintainable
  monoliths).
- Update `AGENTS.md` / architecture docs to describe the single-engine pipeline.

### Phase 6 — Strengthen CGraph collapse (future, separate effort)

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
Phase 4 ~3d, Phase 5 ~2d — on the order of 4–5 weeks total, front-loaded value.
