---
description: Patch & contract development (Passes, YAML spec, Contracts dialect, patchir-transform, patchir-cir2llvm)
paths:
  - "lib/patchestry/Passes/**"
  - "include/patchestry/Passes/**"
  - "lib/patchestry/YAML/**"
  - "include/patchestry/YAML/**"
  - "include/patchestry/Dialect/Contracts/**"
  - "tools/patchir-transform/**"
  - "tools/patchir-cir2llvm/**"
  - "test/patchir-transform/**"
---

# Patch & Contract Development

## Key Files

| Task | Files |
|------|-------|
| Extend YAML patch spec | `include/patchestry/YAML/PatchSpec.hpp`, `include/patchestry/YAML/BaseSpec.hpp` |
| Extend YAML contract spec | `include/patchestry/YAML/ContractSpec.hpp` |
| Add patch/contract mode | `lib/patchestry/Passes/InstrumentationPass.cpp`, `lib/patchestry/Passes/ContractOperationImpl.cpp` |
| Change argument handling | `lib/patchestry/Passes/InstrumentationPass.cpp` — `prepare_contract_call_arguments` / `prepare_patch_call_arguments` |
| Add LIT transform test | `test/patchir-transform/` — copy an existing YAML test as template |
| Modify CIR→LLVM lowering | `tools/patchir-cir2llvm/main.cpp` — CIR to LLVM IR/bitcode with contract metadata |
| Update spec docs | `docs/GettingStarted/patch_specifications.md` |

## Adding a New Argument Source

`ArgumentSourceType` controls where a value comes from when building the argument list for a patch/contract call. Each `arguments:` entry in a YAML `action:` block has a `source:` field mapping to one of: `operand` (Nth operand by index — at `apply_at_entrypoint`, index N maps to the enclosing function's Nth block arg), `variable` (named local alloca), `symbol` (module-level global), `constant` (inline literal), `return_value` (call result — invalid with `apply_at_entrypoint`), `capture` (named match binding — invalid with `apply_at_entrypoint`).

To add a new source:

1. Add enumerator to `ArgumentSourceType` in `include/patchestry/YAML/BaseSpec.hpp`
2. Add `source_str == "my_source"` branch in `MappingTraits<patch::ArgumentSource>::mapping` in `include/patchestry/YAML/PatchSpec.hpp`; mirror in `ContractSpec.hpp` if contracts also use it
3. Add a `case ArgumentSourceType::MY_SOURCE:` in both `prepare_patch_call_arguments` (~line 720) and `prepare_contract_call_arguments` (~line 1102) in `lib/patchestry/Passes/InstrumentationPass.cpp`; if the source requires call-site context, reject it when `entrypoint_func` is set (like `return_value` and `capture` do). Both `prepare_patch_call_arguments` and `prepare_contract_call_arguments` take an `std::optional<cir::FuncOp> entrypoint_func` — thread it through to your handler.
4. Add a row to the Argument Source Types table in `docs/GettingStarted/patch_specifications.md`

## Invariants

- **SSA Dominance**: A value must be defined in a block that dominates every use. Never reference a value defined at a call site from an earlier insertion point (e.g., the function entry block).
- **APPLY_AT_ENTRYPOINT sources**: `operand` → remapped to enclosing function args; `variable`/`symbol`/`constant` → valid; `return_value` and `capture` → **rejected** (only exist at call site). Supported by both `patches:` and `contracts:`. Implementation: `prepare_patch_call_arguments` / `prepare_contract_call_arguments` in `InstrumentationPass.cpp` (both take `std::optional<cir::FuncOp> entrypoint_func`); dispatch lives in `PatchOperationImpl::applyPatchAtEntrypoint` and `ContractOperationImpl::applyContractAtEntrypoint`.

## Build & Test

```sh
# macOS
cmake --build builds/default --config Debug --target patchestry_passes patchestry_yaml patchir-cir2llvm
lit ./builds/default/test/patchir-transform -D BUILD_TYPE=Debug -v

# Linux (dev container)
cmake --build builds/ci --config Release --target patchestry_passes patchestry_yaml patchir-cir2llvm -j$(nproc)
lit ./builds/ci/test/patchir-transform -D BUILD_TYPE=Release -v
```

## Inspection

```sh
# Apply patches and inspect resulting CIR
patchir-transform input.cir -spec patch.yaml -o patched.cir

# Lower patched CIR to LLVM IR
patchir-cir2llvm -S patched.cir -o patched.ll

# Validate a YAML spec
patchir-yaml-parser myspec.yaml --validate
```
