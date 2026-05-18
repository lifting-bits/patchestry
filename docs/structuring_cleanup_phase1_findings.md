# Phase 1 Findings — Why the SNode Cleanup Layer Is Weak

Phase 1 of `docs/structuring_cleanup_redesign.md`. The pivotal question:
**why does the SNode-level cleanup layer eliminate only ~16% of gotos
(233 → 196) while the Clang-AST layer does ~87% (196 → 25)?**

Measured 2026-05-18 on branch `kumarak/structuring-phase1-diagnosis`, Debug
build, across the 80 `test/patchir-decomp` fixtures (116 structured functions).

## How it was measured

- `ShouldApplySNodeRewrite` (the per-pass preflight gate, called from 9 sites
  in the SNode cleanup passes) was instrumented to log every call as
  `accept` / `reject_legality` / `reject_profitability`.
- Transaction rollbacks were counted from the existing
  `RunSNodeRewriteTransaction` "Rolling back …" log.
- The cleanup funnel was read from the existing `STRUCTURING_IMPROVEMENT_REPORT`
  `cleanup_*` fields.

The instrumentation lives in `lib/patchestry/AST/SNodeRegion.cpp` marked
`[PHASE-1 INSTRUMENTATION]` — temporary; behavior-neutral (81/81 lit passes).

## Data

### Preflight gate — `ShouldApplySNodeRewrite` (6468 calls)

| Outcome | Count | Share |
|---|---|---|
| accept (clone 5440 + move 72) | 5512 | **85.2%** |
| reject_legality (clone 895 + move 61) | 956 | 14.8% |
| reject_profitability | **0** | 0.0% |

### Transaction rollbacks

`RunSNodeRewriteTransaction` rolled back **0** times.

### Cleanup funnel (summed over 116 functions)

| candidates_queued | candidates_run | changed_passes | profitable_rewrites | rewrite_candidates_consumed |
|---|---|---|---|---|
| 6438 | 6438 | 241 | 25 | 619 |

## Hypothesis verdicts

**H1 — legality/ownership gating rejects most rewrites (timid by construction):
REJECTED.** The gate accepts 85% of all rewrite requests. Every one of the 15%
rejections is a *legality* rejection — a payload operation that genuinely
cannot be safely cloned/moved — and **zero** are profitability rejections. The
post-hoc transaction validator rolled back **zero** times. The
`SNodeRegion` legality / ownership / transaction machinery is correct and
cheap; it is **not** what makes the SNode layer timid. There is no
over-caution to relax.

**H2 — the SNode tree IR does not expose post-linearization adjacency:
SUPPORTED (dominant cause).** The Clang-AST layer's single hottest pass is
`EliminateGotoToNextLabel` — 3852 fires, more than any other of the 27. "goto
to the immediately following label" is a property of the *linear statement
layout*, which exists only after the SNode tree is flattened into a
`CompoundStmt`. The SNode tree is a tree of nested vector-slot bodies; a goto
and its target label that sit in different body slots become physically
adjacent only post-emission. A large class of goto eliminations is therefore
structurally invisible to *any* tree-level pass — including the SNode-level
`EliminateGotoToNextLabel`, which can only see intra-body-vector adjacency.

**H3 — SNode passes simply incomplete: minor, not primary.** The SNode passes
run heavily (6438 pass-runs, 241 effective changes) and the gate feeds them at
an 85% accept rate — they are not starved for candidates and not blocked. The
shortfall is not "passes never tried"; it is that the tree form does not
surface the dominant transform.

## Decision — NOT Direction A; adopt the hybrid

The Phase-0 plan framed Direction A ("make the SNode layer the single engine
by relaxing its timid gating") as viable if H1 held. **H1 is dead.** There is
no timid gating; relaxing it would change nothing. Direction A is rejected.

A post-emission cleanup layer is **architecturally necessary** — not debt — for
layout-dependent goto elimination (`EliminateGotoToNextLabel` and the
cross-compound adjacency folds). It cannot be moved onto the tree.

**Adopt the hybrid.** The redesign goal is restated:

1. **Keep a post-emission cleanup layer**, but rebuild it as *one consolidated
   engine* — ~8 parameterized transforms with a real worklist fixed point —
   instead of 27 ad-hoc passes plus a fake 8× loop. (Plan Phases 2–3 still
   apply, targeting the Clang-AST layer.)
2. **Trim the SNode cleanup layer to its competent scope** — the genuinely
   structural region rewrites (clone/move payload, switch-case folding) that
   the gate already accepts at 85%. Stop running tree-level copies of
   layout-dependent passes (`EliminateGotoToNextLabel` and similar) at the
   SNode layer: the tree cannot see the adjacency, so they mostly spin without
   effect and only add cost.
3. The duplicated pass *names* across the two layers are real redundancy, but
   the resolution is asymmetric, decided per transform class:
   - layout / adjacency transforms → keep the post-emission copy, delete the
     tree-level copy;
   - purely structural region transforms → keep the SNode copy, delete the
     Clang-AST copy.

## Impact on the redesign plan

- Plan **Phase 4** ("delete the redundant layer") is re-scoped: it is *not*
  "delete one whole layer." It is "delete the weak tree-level adjacency passes,
  delete the structural Clang-AST duplicates, and keep each transform in
  exactly one layer."
- Plan **Phases 2–3** (real fixed point, family consolidation) are unchanged
  and now clearly target the **post-emission** engine.
- The vector-slot `SNode` model remains sound; this finding is about *which
  cleanup belongs at which stage*, not about the data model.

## Reproduce

```sh
git checkout kumarak/structuring-phase1-diagnosis
cmake --build builds/default --config Debug --target patchir-decomp
for j in test/patchir-decomp/*.json; do
  bash test/scripts/strip-json-comments.sh "$j" > /tmp/in.json
  patchir-decomp -input /tmp/in.json -use-structuring-pass \
      -structuring-improvement-report -print-tu -output /tmp/o 2>&1
done | grep -oE 'SREWRITE_GATE action=[a-z]+ outcome=[a-z_]+' | sort | uniq -c
```
