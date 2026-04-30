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
    |   - patchestry_ast: Ghidra model -> CGraph -> SNode -> Clang AST/CIR-ready structures
    |   - patchestry_codegen: emit selected output form
    v
[Decompilation outputs]
    |- High-level MLIR        (--emit-mlir)      [tested]
    |- CIR                    (--emit-cir)       [tested]
    |- LLVM IR                (--emit-llvm)      [tested]
    |- Pretty-printed C TU    (--print-tu)       [tested]
    |- Assembly               (--emit-asm)       [flag exists, path unimplemented]
    \- Object file            (--emit-obj)       [flag exists, path unimplemented]

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
6. Clang AST emission lowers the `SNode` tree into a function body, and later
   codegen emits CIR / MLIR / LLVM / pretty-printed C.

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
