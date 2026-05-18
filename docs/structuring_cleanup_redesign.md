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

### Phase 1 — Diagnose the SNode layer (the decision phase)

The pivotal question: **why does the SNode cleanup achieve only ~12%?**

Measurements:
- Third data point: both cleanups off → raw post-collapse goto count. Fixes the
  SNode layer's true contribution (currently only bounded: it reaches 196).
- Instrument `ValidateSNodeRewriteLegality` / `ShouldApplySNodeRewrite`: how
  often does a pass *attempt* a rewrite vs how often does `SNodeRegion`
  legality/profitability *reject* it?

Hypotheses:
- **H1** — legality/ownership gating rejects most rewrites (timid by construction).
- **H2** — the SNode IR doesn't expose adjacency ("goto to the next label")
  that only emerges after statements are laid out linearly at emission.
- **H3** — the SNode passes are simply less complete than their Clang-AST twins.

**Decision gate** — pick the consolidation direction:
- **Direction A — SNode becomes the single engine.** Choose if H1 dominates and
  the gating is over-cautious / safely relaxable. Port the effective Clang-AST
  transforms down to SNode passes; `ClangEmitterCleanup` keeps only cosmetics.
- **Direction B — Clang-AST becomes the single engine.** Choose if H2 dominates
  (some cleanup genuinely needs post-emission linear layout). Delete the weak
  SNode cleanup layer and its transaction machinery; make `ClangEmitterCleanup`
  the single, well-structured engine.
- **Likely hybrid:** structural transforms (forwarder collapse, target inlining)
  on the SNode tree; a *small* layout-dependent post-emission pass for
  goto-to-next-label / label adjacency only.

**Exit:** a written decision (A / B / hybrid) backed by the rejection data.

### Phase 2 — Real fixed-point driver

Independent of the A/B choice; fixes whichever driver survives.

1. Enforce the changed-contract: every pass returns "changed" only when it
   actually mutated. Fix the ~10 always-rebuild passes so the `if (body)` guard
   becomes meaningful.
2. Replace the fake 8× loop + 180-line unrolled tail with a genuine worklist
   fixed point: run transforms until none reports a change.

**Exit:** convergence iteration count drops (measured via `-cleanup-stats`);
80/80 lit; goto budget holds.

### Phase 3 — Consolidate transform families

Collapse ~49 passes into ~8 canonical, parameterized transforms, on the IR
chosen in Phase 1, family by family (each family = one build + budget check):

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

### Phase 4 — Delete the redundant layer

- **Direction A:** delete `ClangEmitterCleanup`'s goto/label passes; it retains
  only genuinely-Clang-AST cosmetics (`NormalizeConditions`,
  `PushLabelsIntoCompounds`, while→for, empty-compound flatten). Target < 50 KB.
- **Direction B:** delete the SNode cleanup scheduler and the cleanup-only parts
  of the `SNodeRegion` transaction/legality machinery.
- Either way: duplicated pass names collapse to one definition each.

**Exit:** one cleanup engine; goto budget holds; 80/80 lit.

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
| Direction A picked but SNode gating can't be safely relaxed → lose the 87% | Phase 1 decision gate — do not commit to A without H1 evidence |
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
