# Structuring Cleanup — Phase 0 Baseline Snapshot

Captured 2026-05-18 on `kumarak/redesign_structuring`, Debug build, across the
80 `test/patchir-decomp` fixtures (116 structured functions). See
`docs/structuring_cleanup_redesign.md` for the redesign this baseline guards.

## Per-stage emitted-goto totals

Goto count of the program at each cleanup-pipeline stage, summed over all
structured functions (from `STRUCTURING_IMPROVEMENT_REPORT` +
`CLANG_CLEANUP_SUMMARY`, with the `pre_cleanup_gotos` field added in Phase 0):

| Stage | Total gotos | Eliminated by this stage |
|---|---|---|
| Post-collapse (CFGStructure output) | 233 | — |
| Post-SNode-cleanup (~22 passes)     | 196 | 37  (16% of its input) |
| Post-Clang-cleanup (27 passes)      | 25  | 171 (87% of its input) |

This quantifies the redesign's central problem: the principled,
transaction-guarded **SNode cleanup layer eliminates only 16%**, while the
unguarded **Clang-AST peephole layer does 87%** of the remaining work.

## Per-fixture final-goto budget

Final emitted `goto` count is 0 for all fixtures except:

| Fixture | Final gotos |
|---|---|
| `pb_decode_inner`    | 10 |
| `cwe22_init_logger`  | 6  |
| `decode_basic_field` | 5  |
| `encode_basic_field` | 4  |
| **Total**            | **25** |

These are the upper bounds enforced by `test/patchir-decomp/goto-budget.txt`
via the `zz-goto-budget.test` lit regression guard.

## Reproduce

```sh
# Regression guard (also runs as part of the lit suite):
builds/default/bin/llvm-lit builds/default/test/patchir-decomp/zz-goto-budget.test -v

# Per-stage breakdown for one function:
patchir-decomp -input f.json -use-structuring-pass \
    -structuring-improvement-report -print-tu -output /tmp/o 2>&1 \
  | grep -E 'STRUCTURING_IMPROVEMENT_REPORT|CLANG_CLEANUP_SUMMARY'
#   pre_cleanup_gotos = post-collapse
#   CLANG_CLEANUP_SUMMARY initial_gotos = post-SNode-cleanup
#   CLANG_CLEANUP_SUMMARY final_gotos   = post-Clang-cleanup
```
