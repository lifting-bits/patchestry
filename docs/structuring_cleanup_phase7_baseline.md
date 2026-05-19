# Phase 7 Step 0 — Baseline + premise pin

Phase 7 of `docs/structuring_cleanup_redesign.md` (the deferred "strengthen
CGraph collapse" phase, re-scoped — see the Phase 7 plan). Step 0 establishes
the per-stage goto funnel and *pins the premise* the rest of Phase 7 depends
on: that the SNode-layer / Clang-AST-layer cleanup gap is not a representational
limit of the SNode tree.

Measured 2026-05-18, Debug build, all 80 `test/patchir-decomp` fixtures.

## Instrumentation added

- `STRUCTURING_IMPROVEMENT_REPORT` gains `post_cleanup_gotos` — `SGoto` count of
  the SNode tree *after* the SNode cleanup schedule, before `ClangEmitter`
  (`CountSNodeGotos`, `ASTConsumer.cpp`). It already carried `pre_cleanup_gotos`
  (post-collapse).
- `CLANG_CLEANUP_SUMMARY` gains `prologue_compound_splices` /
  `prologue_label_pushes` — counts of the two adjacency-changing operations the
  emission-adjacent `CleanupStmtTree` prologue performs
  (`CleanupStmtTreeStats` / `TakeCleanupStmtTreeStats`).

All behavior-neutral: `llvm-lit` 81/81, `zz-goto-budget` holds.

## The four-stage goto funnel (summed over 80 fixtures)

| Stage | What it counts | Total |
|---|---|---|
| s1 post-collapse | `SGoto` nodes, pre SNode cleanup | 233 |
| s2 post-SNode-cleanup | `SGoto` nodes, post SNode cleanup | **62** |
| s3 post-emission / pre-Clang-cleanup | all `clang::GotoStmt` | 196 |
| s4 post-Clang-cleanup | all `clang::GotoStmt` | 25 |

`CleanupStmtTree` prologue, all 80 fixtures: **0** compound splices, **81**
label pushes.

## The decisive finding

`CountSNodeGotos` counts only `SNodeKind::kGoto` (`SGoto`) nodes. `SStmt` is an
opaque leaf wrapping a raw `clang::Stmt*` with no SNode children, so a goto that
lives as a raw `clang::GotoStmt` inside an `SStmt` is invisible to
`CountSNodeGotos` **and to every SNode-level cleanup pass**.

`ClangEmitter` cannot synthesize a goto: the only `clang::GotoStmt` sources are
`EmitGoto(SGoto)` and raw `clang::GotoStmt` passed through `SStmt::Stmt()`.
Therefore, exactly:

```
s3 (all clang gotos) = #SGoto + #raw-goto-in-SStmt
196                  = 62     + 134
```

**134 of the 196 gotos entering the Clang-AST cleanup — 68% — were never
visible to the SNode cleanup layer at all.** They are raw `clang::GotoStmt`
embedded in `SStmt` leaves (P-Code block-internal if-gotos and CNode terminal
gotos that `BuildLeafSNode` wraps as `SStmt(node.terminal)`), not first-class
`SGoto` nodes.

The SNode layer is **not weak — it is half-blind.** On the goto population it
*can* see, it is strong: 233 → 62 `SGoto` is a 73% reduction. It simply never
sees the other 134.

## Premise verdict — GREEN

The redesign's Phase 1 read the SNode/Clang gap as representational ("the tree
cannot see post-linearization adjacency"). Step 0 refutes that mechanism:

- **Emission is goto-count-neutral** — `s3 = #SGoto + #raw`, no goto created or
  destroyed by `ClangEmitter`.
- **`CleanupStmtTree` does no compound flattening** (0 splices across all 80
  fixtures) — post-`SSeq`, the emitter never nests a `CompoundStmt` directly in
  a `CompoundStmt`. Its only adjacency work is 81 label-pushes
  (`SLabel`-with-multi-child-body → splice tail into parent), trivially
  replicable on the flat-vector SNode tree before emission.
- **The real gap is representational *within* the SNode tree, not between
  IRs**: gotos exist in two forms — `SGoto` nodes and raw `clang::GotoStmt` in
  opaque `SStmt` leaves — and the cleanup only handles the first.

Therefore Phase 7 Step 1's "normalize every goto to `SGoto`, every label to
`SLabel`" is **the critical-path enabler**, not a cosmetic cleanup. Once the
SNode tree has a uniform first-class control-flow representation, the SNode
cleanup layer sees the full 196-goto population, and there is no representational
reason the Clang-AST post-emission layer must exist. The plan proceeds.

## Per-fixture residuals (the 12 fixtures with cleanup activity)

| Fixture | s1 | s2 | s3 | s4 | label pushes |
|---|---|---|---|---|---|
| cve_2016_6563_fun_0000b920 | 55 | 11 | 12 | 0 | 3 |
| pb_decode_inner | 47 | 15 | 19 | **10** | 3 |
| encode_basic_field | 32 | 11 | 50 | **4** | 20 |
| decode_basic_field | 29 | 7 | 23 | **5** | 11 |
| cwe22_init_logger | 14 | 10 | 56 | **6** | 23 |
| bl_device__match | 8 | 1 | 13 | 0 | 10 |
| decode_frame | 8 | 3 | 4 | 0 | 1 |
| load_descriptor_values | 6 | 4 | 13 | 0 | 8 |
| bloodview_device__query | 3 | 0 | 1 | 0 | 0 |
| cwe121_eeprom_handler_write | 3 | 0 | 1 | 0 | 1 |
| cve_2016_6563_fun_00011708 | 1 | 0 | 3 | 0 | 0 |
| advance_iterator | 1 | 0 | 1 | 0 | 1 |

The remaining 68 fixtures: s4 = 0. The s4 total of 25 = the four budgeted
fixtures (pb_decode_inner 10, cwe22_init_logger 6, decode_basic_field 5,
encode_basic_field 4), consistent with `goto-budget.txt`.

## Reproduce

```sh
cmake --build builds/default --config Debug --target patchir-decomp
for j in test/patchir-decomp/*.json; do
  bash test/scripts/strip-json-comments.sh "$j" > /tmp/in.json
  builds/default/tools/patchir-decomp/Debug/patchir-decomp -input /tmp/in.json \
    -use-structuring-pass -structuring-improvement-report -print-tu -output /tmp/o 2>&1
done | grep -E 'STRUCTURING_IMPROVEMENT_REPORT|CLANG_CLEANUP_SUMMARY'
```
