---
description: Decompilation pipeline development (AST, Ghidra, patchir-decomp)
paths:
  - "lib/patchestry/AST/**"
  - "include/patchestry/AST/**"
  - "lib/patchestry/Ghidra/**"
  - "include/patchestry/Ghidra/**"
  - "tools/patchir-decomp/**"
  - "test/patchir-decomp/**"
---

# Decompilation Pipeline Development

## Key Files

| Task | Files |
|------|-------|
| Add P-Code operation handler | `lib/patchestry/AST/OperationStmt.cpp`, `lib/patchestry/AST/OperationBuilder.cpp` |
| Modify function building | `lib/patchestry/AST/FunctionBuilder.cpp`, `include/patchestry/AST/FunctionBuilder.hpp` |
| Modify AST consumer | `lib/patchestry/AST/ASTConsumer.cpp`, `include/patchestry/AST/ASTConsumer.hpp` |
| Add intrinsic handler | `lib/patchestry/AST/IntrinsicHandlers.cpp`, `include/patchestry/AST/IntrinsicHandlers.hpp` |
| Modify Ghidra data model | `include/patchestry/Ghidra/JsonDeserialize.hpp`, `lib/patchestry/Ghidra/` |
| Add LIT decomp test | `test/patchir-decomp/` — copy an existing JSON file and embed `// RUN:` directives |

## Build & Test

```sh
# macOS
cmake --build builds/default --config Debug --target patchestry_ast
lit ./builds/default/test/patchir-decomp -D BUILD_TYPE=Debug -v

# Linux (dev container)
cmake --build builds/ci --config Release --target patchestry_ast -j$(nproc)
lit ./builds/ci/test/patchir-decomp -D BUILD_TYPE=Release -v
```

## Structuring Validation

After modifying CFGStructure, ClangEmitter, CGraphBuilder, or ASTConsumer,
run `/patchir-debugger --batch` to verify functional equivalence between
the goto baseline and structured output across all test fixtures.

The batch run checks:
- Function signature preservation
- Call graph preservation (no lost function calls)
- Condition preservation (no lost if-guards — structuring should add ifs, never remove them)
- Duplicate assignment detection (same lvalue written twice without guard = lost condition)
- Return preservation
- Goto elimination metrics

**When to run:**
- Before committing any change to structuring rules or post-passes
- After adding a new CGraph rule or modifying BuildLoopBodySNode
- When goto counts change (verify no correctness regression alongside improvement)

**Interpreting results:**
- `Conds: OK` — all conditions preserved
- `Conds: LOST:N` — investigate: check for condition inversion (false positive) vs real guard loss
- `Conds: FP:-N` — triaged as false positive from condition inversion/merging

If `/patchir-debugger` is not installed, skip structuring validation.

## Inspection

```sh
# Decompile P-Code JSON to C (full AST pipeline output)
patchir-decomp -input func.json -print-tu -output /tmp/out -verbose

# Decompile to CIR
patchir-decomp -input func.json -emit-cir -output /tmp/out

# Decompile to LLVM IR
patchir-decomp -input func.json -emit-llvm -output /tmp/out
```
