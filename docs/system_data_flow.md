# Patchestry System Data Flow

This document describes the repository-owned data flow through Patchestry.
It is intentionally scoped to patchestry code, interfaces, tools, scripts, and
documented downstream handoff points. It does not attempt to document the
internals of LLVM, MLIR, or vendored dependencies.

## End-to-End Data Flow

```text
[Input firmware binary]
    |
    | scripts/ghidra/decompile-headless.sh
    | scripts/ghidra/PatchestryDecompileFunctions.java
    | scripts/ghidra/util/{FunctionSerializer,PcodeSerializer}.java
    v
[Ghidra-exported JSON]
    |
    | patchir-decomp
    |   uses:
    |   - patchestry_ghidra: JSON -> in-memory program/P-Code model
    |   - patchestry_frontend: clang CompilerInstance for the lifter's synthetic unit
    |   - patchestry_ast: Ghidra model -> CGraph -> SNode -> Clang translation unit
    |   - patchestry_codegen: Clang AST -> CIR, then the selected output form
    v
[Decompilation outputs]
    |- High-level MLIR        (--emit-mlir)      [tested]
    |- CIR                    (--emit-cir)       [tested]
    |- LLVM IR                (--emit-llvm)      [tested]
    |- Pretty-printed C TU    (--print-tu)       [tested]
    |- Assembly               (--emit-asm)       [flag exists, path unimplemented]
    \- Object file            (--emit-obj)       [flag exists, path unimplemented]

C re-entry path (LLM refinement stage):
[Pretty-printed C TU from -emit-flat-baseline -print-tu]
    |
    | out-of-process refinement: renames, types, structuring
    | (scripts/llm, not yet in tree)
    v
[Refined C TU, patchestry:tu header and function markers intact]
    +
[Original Ghidra JSON (-input)]
    |
    | patchir-decomp -from-c refined.c -input func.json -validate-pcode
    |   uses:
    |   - patchestry_frontend + patchestry_ast CSourceUnit: parse the C under the target triple
    |   - patchestry_ast PcodeValidator: check the parsed C against the P-Code model
    |   - patchestry_codegen: lower to CIR / LLVM IR
    v
[CIR / LLVM IR]  +  [<output>.validation.json]                 [tested]

Main patching path:
[CIR from patchir-decomp or hand-provided CIR]
    +
[YAML patch/contract spec]
    +
[Patch C code / contract C code]
    |
    | patchir-transform
    |   uses:
    |   - patchestry_yaml: parse config/specs
    |   - patchestry_passes: match + apply patch/contract actions
    |   - patchestry_frontend + patchestry_codegen: compile `code_file:` C into CIR
    |   - MLIRContracts: contract attrs/metadata
    v
[Patched CIR]
    |
    | patchir-cir2llvm
    |   uses:
    |   - CIR lowering
    |   - MLIRContracts metadata preservation
    v
[LLVM output]
    |- LLVM IR text (.ll)     (-S)              [supported]
    \- LLVM bitcode (.bc)     (default mode)    [supported]

Downstream of this repo:
[LLVM IR / bitcode with patch + contract metadata]
    |
    \- external binary rewriting / verification tools
       e.g. final patched binary, KLEE/SeaHorn-style analysis

Current firmware runtime-validation path:
[Checked-in Ghidra JSON fixture]
    |
    | patchir-decomp --emit-cir
    v
[CIR]
    |
    | patchir-transform + YAML spec + patch/contract C code
    v
[Patched CIR]
    |
    | patchir-cir2llvm -S
    v
[Patched LLVM IR]
    |
    \- downstream whole-function replacement flow
       target object for affected function
       -> linked patch blob at reserved firmware patch arena address
       -> patcherex2 raw-byte rewrite of original ELF
       -> qemu-system-arm runtime validation
```

## Notes on Outputs

- `patchir-decomp` can stop at multiple output layers depending on flags:
  high-level MLIR, CIR, LLVM IR, and pretty-printed C are exercised by tests.
- `--emit-asm` and `--emit-obj` are currently exposed by CLI parsing but are not
  implemented end-to-end in the reviewed PR branch.
- `-from-c` re-enters a printed or refined C translation unit and produces the
  same CIR/LLVM outputs as an `-input` run; with `-validate-pcode` it also
  writes `<output>.validation.json`.
- `patchir-transform` produces patched CIR.
- `patchir-cir2llvm` produces LLVM IR text or LLVM bitcode.
- `patchir-yaml-parser` produces validation/inspection output, not a transformed
  firmware artifact.
- Final patched binaries are downstream of this repository's core toolchain and
  are not the primary in-repo artifact produced by the tested flows here.
- Current runtime validation is intentionally scoped to whole-function replacement of affected functions in the original ELF, even when the original Patchestry patch semantics are sub-function (`apply_before`, `apply_after`, `replace`, runtime contract insertion).

## Decompilation Semantics

### What the JSON is

- The Ghidra-side serializer does not emit raw assembly text.
- It emits a repository-specific JSON schema that contains:
  - recovered type information
  - functions and globals
  - basic blocks and operation order
  - high P-Code operations and varnodes
  - branch metadata such as `taken_block`, `not_taken_block`, `target_block`
  - switch metadata such as `switch_input`, `switch_cases`, and fallback edges
- The schema boundary is defined by `include/patchestry/Ghidra/PcodeOperations.hpp`
  and loaded by `patchestry_ghidra`.

### How `patchir-decomp` bridges to C

`patchir-decomp` is not translating assembly directly to C syntax in one step.
The flow is layered:

1. `patchestry_ghidra` deserializes Ghidra JSON into an in-memory typed P-Code
   and CFG model.
2. `patchestry_ast` lifts individual P-Code operations into Clang expressions
   and statements mechanically where possible.
3. Branch terminals are separated from block contents and turned into CFG edges.
4. `CGraph` represents that CFG explicitly for control-flow structuring.
5. `SNode` represents structured control flow (`if`, `switch`, loops, labels,
   gotos, break, continue, return).
6. Clang AST emission lowers the `SNode` tree into a function body.  The
   finished unit is handed over as an `ast::TranslationUnit`; `--print-tu` is
   written from it by the driver before lowering.
7. `patchestry_codegen` lowers the unit's `ASTContext` to CIR and emits
   CIR / MLIR / LLVM.  It does not depend on how the unit was produced.

### Frontend and lowering contracts

`FrontendConfig` requires an explicit `CompilationPolicy`: `LiftedCode` disables
strict-return assumptions and uses the lifter's language settings; `PatchCode`
retains the existing patch compiler settings. Both frontend factories apply the
selected policy, so parsing C does not implicitly select patch-code semantics.

Ghidra language-to-target interpretation belongs to `patchestry_ghidra`
(`Ghidra/Target.hpp`). The lifter explicitly ignores subarchitecture variants and
uses the Program architecture; patch compilation preserves the existing variant
mapping from the language id. The generic frontend consumes only the resulting
LLVM triple and compilation policy.

`CodeGenerator::lower_ast_to_mlir()` returns no module on new codegen errors
or failed verification.
`emit_outputs()` returns failure for invalid modules, failed conversion or LLVM
translation, and file open/write/close failures. The decompiler exits nonzero for
these failures and for failed C output. LLVM emission continues to lower the
supplied CIR module in place; callers must finish CIR inspection first.

`-emit-flat-baseline` is the lean lift: steps 1-3 only, with one label and
one `goto` per CFG edge and no structuring or cleanup.  It is the parity
baseline for `/patchir-inspect --debug` and the input of the out-of-process
LLM structuring stage.

### What the printed C is

`-print-tu` writes `<output>.c` through `lib/patchestry/AST/TUPrinter.cpp`,
not `TranslationUnitDecl::print`.  The file is meant to be re-parsed, by an
LLM stage and by clang, so:

- top-level declarations come out in dependency order: tag forward
  declarations, type definitions, globals, prototypes, definitions;
- expressions are reparenthesized before printing (the lifted AST has no
  `ParenExpr`, and clang's printer never inserts parentheses);
- integer literals narrower than `int` print as plain decimals and NaN/Inf
  as `__builtin_nan*`; unions print as unions; C++ type, field and
  enumerator spellings are sanitized to C identifiers;
- the file starts with `// patchestry:tu format=1 target=<ghidra-id>
  arch=<arch>` and every definition is wrapped in
  `// patchestry:function-begin <key> name=<c-name> symbol=<linker-symbol>`
  and `// patchestry:function-end <key>`; a `comment` field on the JSON
  function prints as a block comment above the definition.

`test/patchir-decomp/zz-roundtrip.test` re-parses every fixture's `.c` with
clang under its target triple; `roundtrip-known-failures.txt` lists the
fixtures whose lifted AST is itself not valid C.

### C re-entry and P-Code validation

`-from-c <file.c>` compiles a translation unit written by `-print-tu`, or
refined from it, instead of lifting JSON.  The target comes from
`-target-lang`, else the `patchestry:tu` header, else `-input`.  The file is
parsed as gnu23 under the same triple and codegen options as an `-input` run,
with warnings off, through `lib/patchestry/AST/CSourceUnit.cpp` and the
`ReenteredCode` policy of `lib/patchestry/Frontend/ClangFrontend.cpp`, the
frontend setup that `patchir-transform` also uses for patch C code.  Target
intrinsics the lifter declares as plain functions (`__wfi`, `__dmb`) lose the
builtin identity Sema gives them on re-parse; library builtins (`memcpy`,
`ceilf`) keep it, as in an `-input` run.

Every `patchestry:function-begin` marker must name a function with a body in
the C and a `cir.func` with a body in the lowered module.
`-strict-symbols=false` turns a missing one into a warning.  `-print-tu` on a
`-from-c` run re-prints the parsed unit with a fresh header and markers.

`-validate-pcode` (requires `-input`) checks each marker-listed function
against the P-Code model it was lifted from, through
`lib/patchestry/AST/PcodeValidator.cpp`.  It is a fact-preservation check,
not an equivalence proof: signature arity and types, callee names and call
counts, string literals, global references, return arity, store and condition
counts, switch cases, reachability and duplicate stores.  Finding codes:
`SIGNATURE_MISMATCH`, `SIGNATURE_TYPE_MISMATCH`, `CALL_LOST`,
`CALL_HALLUCINATED`, `CALL_REWRITTEN`, `CALL_COUNT_DELTA`,
`CALL_ARGC_MISMATCH`, `CALLIND_DEFICIT`, `STRING_LOST`, `GLOBAL_LOST`,
`GLOBAL_HALLUCINATED`, `RET_MISMATCH`, `RET_COUNT_DELTA`, `STORE_DEFICIT`,
`COND_DEFICIT`, `SWITCH_CASE_LOST`, `UNREACHABLE_CODE`, `DUP_ASSIGN`,
`FUNCTION_MISSING`, `UNKNOWN_FUNCTION`.  Each finding is a warning or
critical; the report goes to `<output>.validation.json` (stdout without
`-output`).  The bare flag exits non-zero on any critical finding;
`-validate-pcode=report` writes the report and exits zero.

`test/patchir-decomp/zz-flat-validate.test` runs every fixture through
`-emit-flat-baseline -print-tu`, feeds the C back through
`-from-c -validate-pcode -emit-cir`, and requires zero critical findings.
This pins the re-entry path and the validator's false-positive rate on the one
body known to preserve every P-Code fact.  `flat-validate-known-failures.txt`
lists the fixtures whose lifted AST is not valid C.

The lifter also emits `refine:`-prefixed warnings when the JSON disagrees with
itself: `DECLARE_PARAMETER` count or type versus the prototype, and record
field offsets or size versus the C layout.  An out-of-process refinement
driver can grep for them.

### Mechanical vs semantic recovery

- Mechanical recovery:
  opcode-level lifting such as integer arithmetic, comparisons, loads, stores,
  calls, casts, and pointer operations.
- Semantic recovery:
  recovering higher-level control flow from CFG and Ghidra metadata, especially
  conditional branches and jump-table-based `switch` statements.
- The direct JSON -> CGraph path exists so control-flow structuring happens
  before CIR/LLVM lowering, while branch and switch intent is still explicit.


## Maintenance Contract

- This diagram should match the current code, tests, and documented workflows.
- If a change adds, removes, or reroutes an input, output, tool boundary,
  script entrypoint, or interface handoff, update this document in the same PR.
- If a PR changes an affected interface and does not update this diagram, treat
  that as documentation debt to fix before merge.
