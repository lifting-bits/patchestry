# Phase 4 Step 2 — Cross-pass confluence audit of the cleanup tail

Analysis deliverable. Gates Phase 4 step 3 (replacing the unrolled tail of
`CleanupPrettyPrint` with a worklist). The question: which tail transforms
are **confluent** (order-independent — a worklist reaches the same normal
form regardless of firing order) and which are not. The old Phase 2b
big-bang worklist regressed `decode_basic_field` precisely because it
iterated a non-confluent set; this audit isolates the non-confluent part so
step 3 can quarantine it.

## Method

The cleanup runs in two parts inside `CleanupPrettyPrint`:

1. **The schedule** — a 12-step `fixed_point_cleanup_schedule` run in a loop.
   Since Phase 4 step 1 this loop is a real `Stmt::Profile` fixed point.
2. **The unrolled tail** — a ~150-line straight-line sequence (~33 transform
   invocations), run once.

Key evidence used here: **step 1 measured the schedule loop converging in
1–4 iterations on every fixture, with none hitting the cap of 8.** A loop
that converges (`before == after`) reached a genuine fixed point; one that
oscillated would never satisfy `before == after` and would hit the cap. So
**every transform in the schedule is empirically confluent-as-a-group** —
the schedule loop *is* a proven-terminating worklist over them.

## Schedule set — PROVEN CONFLUENT (step 1)

The schedule iterates, to a no-oscillation fixed point:

| Transform | Family |
|---|---|
| `HoistCrossScopeLabels` | F6 |
| `FoldClangSwitchLocalCaseTargets` (×2) | F8 |
| `ScopeifyIfGotos` | F5 |
| `FoldConditionalFallthroughChains` | F2 |
| `RecoverLoop` | F7 |
| `FoldGotoDiamonds` | F2 |
| `CloneTerminalLabelGotos` (×2) | F1 |
| `FoldForwardSingleRefLabelRegions` | F2 |
| `SinkCommonTerminalEpilogues` | F2 |
| `run_goto_to_next_label_once` | F3 |

These 8 transforms are worklist-safe — the schedule already *is* a worklist
over them and step 1 proved it converges.

## Tail transform inventory & classification

The ~33 tail invocations (after the schedule loop) map to three classes.

### Class A — confluent, worklist-safe

- **All schedule-set transforms re-invoked in the tail** (`FoldForwardSingleRefLabelRegions`,
  `SinkCommonTerminalEpilogues`, `FoldConditionalFallthroughChains`,
  `RecoverLoop`, `FoldGotoDiamonds`, `CloneTerminalLabelGotos`,
  `HoistCrossScopeLabels`, `ScopeifyIfGotos`, `run_goto_to_next_label_*`) —
  already proven (above).
- **`RemoveDeadControlFlow` (F4)** — pure monotone deletion (dead labels,
  orphaned gotos, empty blocks). Deletion only ever shrinks the node set;
  order between deletions cannot change the fixed point. Trivially confluent
  (established in the F4 merge). The F4 merge already needed *one* carve-out:
  `RemoveEmptyBlocks`'s `if`/`else` *merge* is not pure deletion — see the
  caveat under Class B.
- **`FoldCrossCompoundDispatchChains` (F2, tail-only)** — a `Fold*` forwarder
  collapse of the same shape family as the schedule-proven `Fold*` set. Not
  in the schedule, so not *proven*, but structurally homogeneous with proven
  peers. **Step-3 action: add it to the schedule worklist and confirm
  convergence is unaffected.**
- **`InlineSingleRefTerminalLabelBlocks` (F1, tail-only)** — inlines a
  single-ref terminal label block. Same payload-clone family as the proven
  `CloneTerminalLabelGotos`; the `single-ref` precondition makes it strongly
  monotone (a ref count of 1 → 0 cannot reappear). **Step-3 action: add to
  the worklist; gate with the empirical check below.**

### Class B — NOT confluent: the join transforms (the real finding)

Three tail transforms are each wrapped in the driver as
`X(...); if (changed) run_late_join_fixups();` where
`run_late_join_fixups = run_goto_to_next_label_fixed_point() + run_dead_control_flow()`:

| Transform | Family | Driver coupling |
|---|---|---|
| `FoldGuardedJoinLabelChains` | F2 | `if (folded_guarded_join) run_late_join_fixups()` |
| `CloneSmallStraightLineLabelBeforeJoinGotos` | F1 | `if (cloned_small_join) run_late_join_fixups()` |
| `CloneCleanupLabelBeforeJoinGotos` | F1 | `if (cloned_cleanup_join) run_late_join_fixups()` |

The conditional-repair wrapper is the **non-confluence signature.** Each of
these transforms clones a cleanup/join tail into goto sites; the clone leaves
the AST in a transient state with fresh goto-to-next-label adjacencies and
newly-dead labels. `run_late_join_fixups` (F3 + F4) immediately normalizes
that state. The repair is *behavior-preserving* (F3 and F4 are both
behavior-preserving), so the coupling is **not a correctness hazard** — but
it **is a confluence hazard**: if any *other* fold/clone transform fires
against the un-repaired transient state, it can match a shape that should not
exist and rewrite wrongly. This is the most plausible mechanism for the
Phase 2b `decode_basic_field` block loss — a clone transform firing on a
transient join shape and dropping a block.

The three join transforms are also **mutually order-sensitive**: each runs
its own repair before the next, i.e. the hand-tuned order
`FoldGuardedJoin → fixup → CloneSmall → fixup → CloneCleanup → fixup` is
load-bearing. A flat worklist would interleave them and their fixups freely.

`RemoveEmptyBlocks`'s `if`/`else` merge (inside F4) is a lesser Class-B
member — the F4 merge already proved that running it at extra positions
regressed `cve_2016_6563` (the F4 carve-out). It is confluent *with the other
F4 sub-rewrites* but not position-free against fold/clone passes.

### Class C — terminal cosmetics: must run last, once, NOT in the worklist

| Transform | Why terminal |
|---|---|
| `PromoteSimpleCounterWhileToFor` | rewrites `while`→`for`; readability, after structure settles |
| `RemoveRedundantTerminalForContinues` | depends on the `for` shape above |
| `PushLabelsIntoCompounds` | label *layout* normalization |
| `AttachEmptyLabelsToFollowingStmt` | label *layout* normalization |
| `NormalizeConditions` | explicitly "runs last — purely a readability pass" |

These change *surface form*, not goto/label structure. Iterating them in a
worklist is pointless and risks oscillation against the structural passes
(e.g. a `for` that a fold pass would re-shape). They belong in a fixed
once-only epilogue after the worklist converges.

## Recommended step-3 worklist design

Replace the unrolled tail with **three fixed phases**, not one flat worklist:

1. **Core worklist** — the schedule set ∪ {`FoldCrossCompoundDispatchChains`,
   `InlineSingleRefTerminalLabelBlocks`, `RemoveDeadControlFlow`}. Iterate to
   a `Stmt::Profile` fixed point (same driver as step 1), cap-bounded.
2. **Join sub-sequence** — `FoldGuardedJoinLabelChains`,
   `CloneSmallStraightLineLabelBeforeJoinGotos`,
   `CloneCleanupLabelBeforeJoinGotos`, each *immediately* followed by its
   `run_late_join_fixups`, in this fixed order. Then re-run phase 1's core
   worklist once (the join clones may expose new structural folds).
3. **Cosmetic epilogue** — the Class C passes, once, in order.

This keeps the non-confluent Class B passes quarantined behind their repairs
(preserving the load-bearing order) while still delivering a genuine
fixed-point worklist for the confluent majority. It also absorbs the Phase 3
F2/F1 remainders: `FoldCrossCompoundDispatchChains` and
`InlineSingleRefTerminalLabelBlocks` become core-worklist entries; the join
clones become the phase-2 sub-sequence.

## Step-3 entry gate (empirical confluence proof)

Reasoning narrows the risk; it cannot *prove* confluence of AST rewrites.
Step 3 must build the worklist **behind a default-off flag** and prove
equivalence empirically before flipping the default:

1. For every fixture, diff worklist-output vs current-tail-output. Expect
   byte-identical, or a strict goto-count improvement with `/patchir-inspect`
   PASS.
2. Idempotence: re-running the core worklist after it converges must be a
   no-op (it is a fixed point by construction — verify).
3. `zz-goto-budget` holds; 81/81 lit; `/patchir-inspect --debug --batch`
   VERDICT PASS.

Only when all three hold does step 3 flip the flag default and delete the
unrolled tail.

## Verdict

The cleanup tail is **mostly confluent**: 8 transforms proven by step 1, plus
F4 (monotone) and two structurally-homogeneous tail-only folds. The
**non-confluent core is small and identified** — the three join transforms
(`FoldGuardedJoinLabelChains`, `CloneSmallStraightLineLabelBeforeJoinGotos`,
`CloneCleanupLabelBeforeJoinGotos`) whose `run_late_join_fixups` coupling and
mutual ordering are load-bearing. Step 3 is **GREEN to proceed** with the
three-phase design above: a real worklist for the confluent majority, the
join transforms quarantined as a fixed repaired sub-sequence, cosmetics as a
terminal epilogue — gated by the empirical equivalence check.
