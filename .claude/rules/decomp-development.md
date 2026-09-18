---
description: Decompilation pipeline development (AST, Ghidra, patchir-decomp)
paths:
  - "lib/patchestry/AST/**"
  - "include/patchestry/AST/**"
  - "lib/patchestry/Frontend/**"
  - "include/patchestry/Frontend/**"
  - "lib/patchestry/Codegen/**"
  - "include/patchestry/Codegen/**"
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
| Change how the lifter's CompilerInstance is set up | `lib/patchestry/Frontend/ClangFrontend.cpp`, `include/patchestry/Frontend/ClangFrontend.hpp` |
| Add an AST source or change the AST-to-lowering handoff | `include/patchestry/AST/TranslationUnit.hpp`, `lib/patchestry/AST/PcodeLifter.cpp`, `include/patchestry/AST/LiftOptions.hpp` |
| Change CIR lowering or output writing | `lib/patchestry/Codegen/Codegen.cpp`, `include/patchestry/Codegen/Codegen.hpp` |
| Add intrinsic handler | `lib/patchestry/AST/IntrinsicHandlers.cpp`, `include/patchestry/AST/IntrinsicHandlers.hpp` |
| Modify Ghidra data model | `include/patchestry/Ghidra/JsonDeserialize.hpp`, `lib/patchestry/Ghidra/` |
| Add LIT decomp test | `test/patchir-decomp/` — copy an existing JSON file and embed `// RUN:` directives |

## Pipeline entry points

`tools/patchir-decomp/main.cpp` is a thin driver: JSON -> `ast::LiftProgram`
(a `PcodeASTConsumer` inside `frontend::createSyntheticCompilerInstance`) ->
`ast::TranslationUnit` -> `-print-tu` written by the driver ->
`codegen::CodeGenerator::lower_ast_to_mlir` -> `emit_outputs`.  The lowering
takes only an `ASTContext` and `CodeGenOptions`, so any other producer of a
`TranslationUnit` can feed it without touching `patchestry_codegen`.

## Build & Test

```sh
# macOS
cmake --build builds/default --config Debug --target patchir-decomp
lit ./builds/default/test/patchir-decomp -D BUILD_TYPE=Debug -v

# Linux (dev container)
cmake --build builds/ci --config Release --target patchir-decomp -j$(nproc)
lit ./builds/ci/test/patchir-decomp -D BUILD_TYPE=Release -v
```

## Structuring Validation

After modifying CFGStructure, ClangEmitter, CGraphBuilder, or ASTConsumer,
run `/patchir-inspect --debug --batch` to verify functional equivalence between
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

If `/patchir-inspect` is not installed, skip structuring validation.

## Printed C round trip

`-print-tu` output must re-parse with clang: `test/patchir-decomp/zz-roundtrip.test`
runs every fixture through `check-roundtrip.sh`.  A new failure means the
printer (`lib/patchestry/AST/TUPrinter.cpp`) or the lifted AST regressed; a
fixture that starts passing must be removed from `roundtrip-known-failures.txt`.

## Inspection

```sh
# Decompile P-Code JSON to C (full AST pipeline output)
patchir-decomp -input func.json -print-tu -output /tmp/out -verbose

# Lean lift only: goto CFG, no structuring, no cleanup (LLM-stage input)
patchir-decomp -input func.json -emit-flat-baseline -print-tu -output /tmp/out

# Decompile to CIR
patchir-decomp -input func.json -emit-cir -output /tmp/out

# Decompile to LLVM IR
patchir-decomp -input func.json -emit-llvm -output /tmp/out
```
