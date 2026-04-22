# System Data Flow

Patchestry’s primary flow is:

```text
firmware ELF
  -> Ghidra headless export
  -> JSON
  -> patchir-decomp
  -> CIR
  -> patchir-transform
  -> patched CIR
  -> patchir-cir2llvm
  -> patched LLVM IR / bitcode
```

The repository now also has an opt-in runtime validation path for the QEMU ARM32 fixture:

```text
patched LLVM IR
  -> target object code for the affected function
  -> linked patch blob at a reserved firmware patch arena address
  -> patcherex2 raw-byte rewrite of the original ELF
  -> qemu-system-arm runtime validation
```

Current runtime validation is intentionally scoped to whole-function replacement of affected functions in the original ELF, even when the original Patchestry patch semantics are sub-function (`apply_before`, `apply_after`, `replace`, runtime contract insertion).
