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
| Add patch/contract mode | `lib/patchestry/Passes/InstrumentationPass.cpp`, `lib/patchestry/Passes/ContractOperationImpl.cpp` (static contracts only — runtime validators go under `patches:`) |
| Change argument handling | `lib/patchestry/Passes/InstrumentationPass.cpp` — `prepare_patch_call_arguments` |
| Add LIT transform test | `test/patchir-transform/` — copy an existing YAML test as template |
| Modify CIR→LLVM lowering | `tools/patchir-cir2llvm/main.cpp` — CIR to LLVM IR/bitcode with contract metadata |
| Update spec docs | `docs/GettingStarted/patch_specifications.md` |

## Adding a New Argument Source

`ArgumentSourceType` controls where a value comes from when building the argument list for a patch call. Each `arguments:` entry in a YAML `action:` block has a `source:` field mapping to one of: `operand` (Nth operand by index — at `apply_at_entrypoint`, index N maps to the enclosing function's Nth block arg; also accepted as the legacy alias `argument`), `variable` (named local alloca), `symbol` (module-level global), `constant` (inline literal), `return_value` (call result — invalid with `apply_at_entrypoint`), `capture` (named match binding — invalid with `apply_at_entrypoint`). Static contracts carry no call arguments; they attach declarative predicates as MLIR attributes.

To add a new source:

1. Add enumerator to `ArgumentSourceType` in `include/patchestry/YAML/BaseSpec.hpp`
2. Add `source_str == "my_source"` branch in `MappingTraits<patch::ArgumentSource>::mapping` in `include/patchestry/YAML/PatchSpec.hpp`
3. Add a `handle_<my>_argument` helper on `InstrumentationPass` — declare it in `include/patchestry/Passes/InstrumentationPass.hpp` alongside the existing `handle_operand_argument` / `handle_variable_argument` / `handle_symbol_argument` / `handle_return_value_argument` / `handle_constant_argument` / `handle_capture_argument` helpers, and implement it in `lib/patchestry/Passes/InstrumentationPass.cpp`. The helper does the per-source work (resolve the value, create a cast if needed, write into `arg_map`). Inlining the logic into the switch instead of adding a helper diverges from the house style — always follow the `handle_*` pattern.
4. Add a `case ArgumentSourceType::MY_SOURCE:` in the `switch (arg_spec.source)` inside `InstrumentationPass::prepare_patch_call_arguments` (`lib/patchestry/Passes/InstrumentationPass.cpp`) that calls your new helper.
5. If the source needs data harvested at match time (`capture` is the precedent — it reads values bound by `match.captures`), populate that binding in `OperationMatcher::patch_action_matches` (`lib/patchestry/Passes/OperationMatcher.cpp`, the overload that takes `llvm::StringMap<mlir::Value> &captures_out`) and expose it on `PatchInformation` (`include/patchestry/Passes/InstrumentationPass.hpp`) so the helper can read it. `apply_patch_action_to_targets` stashes the per-match binding onto a `PatchInformation` copy before dispatch — mirror that flow. Sources whose value comes from the call site or module scope (operand/variable/symbol/constant/return_value) don't need any OperationMatcher changes.
6. If the source is only meaningful at the matched call site (like `return_value` and `capture`), have the helper reject it when `entrypoint_func` is set — add `std::optional<cir::FuncOp> entrypoint_func = std::nullopt` to both the declaration and definition, thread it from the switch (mirror how OPERAND / RETURN_VALUE / CAPTURE do it), and early-return with a descriptive `LOG(ERROR)` message pointing to the allowed sources. Sources whose value lives in function or module scope (like `variable`, `symbol`, `constant`) don't need to take `entrypoint_func`.
7. Add a row to the Argument Source Types table in `docs/GettingStarted/patch_specifications.md`.

## Invariants

- **SSA Dominance**: A value must be defined in a block that dominates every use. Never reference a value defined at a call site from an earlier insertion point (e.g., the function entry block).
- **Contracts are static-only**: the YAML parser rejects, with a migration message pointing at `patches:`, any `contracts:` entry that carries `code_file` / `function_name` or sets `mode: "apply_at_entrypoint"`. Valid contract modes are `apply_before` and `apply_after`; both attach the same `contract.static` attribute and only differ in which op it lands on. (The legacy `type: "RUNTIME"` spelling is rejected the same way; `type:` is no longer a schema field.)
- **APPLY_AT_ENTRYPOINT is patch-only**: dispatch in `PatchOperationImpl::applyPatchAtEntrypoint`; argument resolution in `prepare_patch_call_arguments` (which takes `std::optional<cir::FuncOp> entrypoint_func`). See the source list above for per-source behavior at entrypoint.

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
