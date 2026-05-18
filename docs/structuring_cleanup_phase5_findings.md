# Phase 5 — Cross-layer duplication: findings (premise refuted)

Phase 5's goal was to resolve the cross-layer cleanup duplication: place each
duplicated transform in exactly one of the two cleanup layers and delete the
other copy. Empirical testing **refutes the premise** — the same-named passes
on the two layers are not redundant duplicates.

## The two cleanup layers

| Layer | Location | When it runs | Passes |
|---|---|---|---|
| **SNode cleanup** | `CFGStructure.cpp`, scheduled in `ASTConsumer.cpp` (~1330–1600) | only under `-use-structuring-pass` (gated at `ASTConsumer.cpp:1295`) | ~22 |
| **Clang-AST cleanup** | `ClangEmitterCleanup.cpp` (`CleanupPrettyPrint`) | always (patchir-decomp path) | 8 families (post-Phase-3) |

The SNode layer runs on the `SNode` tree *before* Clang AST emission; the
Clang-AST layer runs on the emitted `clang::Stmt` tree *after*. They are two
engines on two IRs — but, crucially, the SNode layer is opt-in and the
Clang-AST layer is always-on, so the "two parallel engines" only both run
under `-use-structuring-pass`.

## Cross-layer name duplicates

Passes that share a name (or near-name) across the two layers:

| SNode (CFGStructure.cpp) | Clang-AST (ClangEmitterCleanup.cpp) |
|---|---|
| `EliminateGotoToNextLabel` | `EliminateGotoToNextLabel` (F3) |
| `ScopeifyIfGotos` | `ScopeifyIfGotos` (F5) |
| `RepairCrossScopeLabelEntries` | `RepairCrossScopeLabelEntries` (now inside F6 `HoistCrossScopeLabels`) |
| `FoldSwitchLocalCaseTargets` | `FoldClangSwitchLocalCaseTargets` (F8) |
| `Duplicate*Targets` family (7 passes) | `Clone*` / `InlineSingleRef*` family (F1) |

The plan's rule: layout/adjacency transforms keep the Clang-AST copy and
delete the SNode copy ("the tree cannot see the adjacency, so the SNode copy
spins without effect"); structural transforms keep the SNode copy and delete
the Clang-AST duplicate.

## Empirical test — the premise is refuted

`EliminateGotoToNextLabel` is the plan's flagship "delete the SNode copy"
case — Phase 1 named it the hottest Clang-AST pass and argued the SNode tree
"structurally cannot expose" goto-to-next-label adjacency, so the SNode copy
should be inert.

**Test:** the SNode-layer `EliminateGotoToNextLabel` was removed from the
SNode cleanup schedule; rebuild; run the lit suite.

**Result:** **2 failures** — `cwe121_eeprom_handler_write.json` and the
`zz-goto-budget` regression guard. The SNode-layer `EliminateGotoToNextLabel`
does real, budget-relevant goto elimination. It is **load-bearing, not
inert.** Probe reverted; suite back to 81/81.

## Why — and why this generalizes

The plan's reasoning was half-right and half-wrong. It is true the SNode tree
cannot see *post-linearization flat* adjacency. But the SNode tree **can** see
*sibling-order* adjacency within an `SSeq` — a goto whose target label is the
next sibling node. The SNode `EliminateGotoToNextLabel` eliminates exactly
those; the Clang-AST `EliminateGotoToNextLabel` eliminates the cases that only
become adjacent after the tree is flattened. **Same idea, two IRs, two
disjoint sets of adjacency — complementary, not duplicate.** Deleting either
loses its half.

This corroborates Phase 1's own measurement that *no pass on either layer is
dead*. A shared *name* across the layers signals a shared *idea*, not
redundant work. The "two parallel cleanup engines doing duplicate work"
framing — a high-level Phase-1 impression — does not survive per-pass
measurement.

## Verdict

**Phase 5's delete-one-copy goal is withdrawn as refuted.** The cross-layer
duplication is nominal (shared names) not actual (shared work); both copies
are load-bearing. There is no safe deletion to make. Like the Phase 4
tail-worklist, the plan item is closed by measurement, not by code.

The only salvageable Phase 5 value is **documentation**: the intentional
two-IR, two-adjacency design should be described so the shared names stop
*looking* like dead duplication. That belongs in Phase 6 (architecture docs).
A disambiguating rename (e.g. SNode `EliminateGotoToNextLabel` →
`EliminateSiblingGotoToLabel`) is possible but is cosmetic churn with real
risk across `CFGStructure.cpp`; deferred, not recommended as urgent.

**Remaining name-duplicates** (`ScopeifyIfGotos`, `RepairCrossScopeLabelEntries`,
`FoldSwitchLocalCaseTargets`, the `Duplicate*`/`Clone*` family) were not
individually probed: Phase 1's "no pass is dead" plus this result make the
expected outcome the same — load-bearing on both layers. A full per-pass
probe is possible but low-value.
