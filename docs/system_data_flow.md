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
    |   - patchestry_ast: Ghidra model -> Clang AST/CIR-ready structures
    |   - patchestry_codegen: emit selected output form
    v
[Decompilation outputs]
    |- High-level MLIR        (--emit-mlir)      [tested]
    |- CIR                    (--emit-cir)       [tested]
    |- LLVM IR                (--emit-llvm)      [tested]
    |- Pretty-printed C TU    (--print-tu)       [tested]
    |- Assembly               (--emit-asm)       [supported in code]
    \- Object file            (--emit-obj)       [supported in code]

Optional side path:
[Ghidra-exported JSON]
    |
    | pcode-translate
    |   uses patchestry_ghidra + MLIRPcode registration
    v
[P-Code translation output]
    \- textual translation output via mlir-translate driver   [tested]

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
```

## Notes on Outputs

- `patchir-decomp` can stop at multiple output layers depending on flags:
  high-level MLIR, CIR, LLVM IR, pretty-printed C, assembly, or object output.
- `patchir-transform` produces patched CIR.
- `patchir-cir2llvm` produces LLVM IR text or LLVM bitcode.
- `patchir-yaml-parser` produces validation/inspection output, not a transformed
  firmware artifact.
- Final patched binaries are downstream of this repository's core toolchain and
  are not the primary in-repo artifact produced by the tested flows here.

## Maintenance Contract

- This diagram should match the current code, tests, and documented workflows.
- If a change adds, removes, or reroutes an input, output, tool boundary,
  script entrypoint, or interface handoff, update this document in the same PR.
- If a PR changes an affected interface and does not update this diagram, treat
  that as documentation debt to fix before merge.
