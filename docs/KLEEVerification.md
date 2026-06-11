# KLEE-Based Verification Pipeline for Patchestry Patches and Contracts

---

## 1. Executive Summary

Patchestry's `patchir-transform` + `patchir-cir2llvm` pipeline embeds formal contracts (preconditions, postconditions, and invariants) into patched LLVM IR as `!static_contract` metadata strings. A SeaHorn integration exists in `analysis/seahorn/` using `__VERIFIER_assert` / `__VERIFIER_assume`. What is currently **missing** is a standalone tool that bridges these contracts to KLEE-based symbolic execution.

We plan to implement **`patchir-klee-harness`**: a tool that reads the `!static_contract` metadata embedded by `patchir-cir2llvm` and generates a `@klee_main_<func>()` driver appended to the patched `.ll` file. KLEE then executes the driver symbolically to verify that all contracts hold on every reachable path.

---

## 2. Background and Motivation

### 2.1 Why Symbolic Execution?

Firmware patches introduce security-critical logic into low-level code paths that must be correct across every possible input the deployed device will ever encounter. Two common verification approaches fall short here:

- **Testing** can only cover inputs the developer thought of. Edge cases — malformed inputs, boundary values, unexpected device states — are routinely missed and can cause patient harm.
- **Static analysis** reasons over abstractions and is deliberately over-approximate; it cannot definitively prove that a contract holds on every path through real binary code, and it cannot produce the concrete input that would trigger a violation.

Symbolic execution closes this gap. KLEE treats inputs as symbolic variables and explores all feasible execution paths through the patched function simultaneously. Applied to Patchestry patches, this means:

- **Preconditions** are checked at every call site, on every reachable path, not just the ones a test suite exercises.
- **Postconditions** are verified on every return path, including error exits that tests rarely reach.
- **Runtime contract functions** are checked exhaustively, not just for sampled inputs.
- **Patch interactions** with surrounding firmware logic are validated end-to-end: the harness drives the patched function with the full range of inputs the binary could receive, catching regressions that appear only at the interface between new patch code and existing firmware.

When a contract is violated, KLEE produces a **concrete counterexample input** — the exact byte sequence or argument values that trigger the failure — which dramatically reduces debugging time compared to reproducing violations by hand on real hardware.

### 2.2 Contract Encoding in Patchestry

Patchestry encodes contracts at two levels:

**Static contracts** — encoded as two-operand LLVM MDNode metadata on instructions.
`patchir-cir2llvm` emits a `!static_contract` MDNode with two operands: a tag
string (`"static_contract"`) and the serialized contract string:

```llvm
!static_contract !56
!56 = !{!"static_contract", !"preconditions=[{...}], postconditions=[{...}]"}
```

The harness parser must extract the **second operand** of the MDNode to obtain
the contract string (see `tools/patchir-cir2llvm/main.cpp` lines 432-440).

**Runtime contracts** — compiled C functions merged into the IR module:

```llvm
define void @contract__sprintf(i32 %0, i32 %1) { ... }
```

Both contract types can be present in the patched `.ll` output and must be handled by the harness generator.

---

## 3. Existing Pipeline and Gap Analysis

```text
Firmware Binary
  → [Ghidra]                     → P-Code JSON
  → [patchir-decomp]             → .cir
  → [patchir-transform -spec]    → patched.cir
  → [patchir-cir2llvm -S]        → patched.ll  (with !static_contract metadata)
```

The planned **`patchir-klee-harness`** tool will read the `!static_contract` metadata from patched IR, generate a symbolic driver (`@klee_main_<func>()`) that makes all function arguments symbolic, injects `klee_assume` for preconditions, and asserts postconditions and invariants on every return path, producing a self-contained harness ready for KLEE execution.

```text
patched.ll
  → [patchir-klee-harness -input patched.ll -spec patch.yaml]  → harness.ll
  → [klee harness.bc --entry-point=klee_main_<func>]           → verification results
```

---

## 4. Tool Design: `patchir-klee-harness`

### 4.1 Command-Line Interface

```sh
patchir-klee-harness \
  -input  patched.ll          # patched LLVM IR from patchir-cir2llvm
  -output harness.ll          # original module + klee_main_* functions appended
  [-function <name>]          # only generate harness for this function (default: all)
  [-buffer-size <N>]          # bytes allocated for each pointer argument (default: 64)
  [-spec patch.yaml]          # optional: YAML spec for additional contract overrides;
                              #   contracts are read from !static_contract metadata
                              #   when omitted
```

### 4.2 Shared Predicate Infrastructure

The SeaHorn integration and KLEE harness tool both need to parse 
`!static_contract` metadata and translate predicates into
verification-tool-specific IR. To avoid duplicating this logic, the following
should be extracted into a shared library (e.g. `patchestry_contract_ir`):

- **Metadata parsing**: reading `!static_contract` MDNodes and deserializing
  them into in-memory `Predicate` structures.
- **Predicate-to-IR translation**: converting `PredicateKind` values (range,
  non-null, alignment, expr, etc.) into LLVM IR `icmp`/`and`/`or` sequences if needed.
- **Function-level contract collection**: walking a module to associate
  contracts with their enclosing functions.

Each verification backend then only needs a thin driver layer: SeaHorn emits
`__VERIFIER_assert`/`__VERIFIER_assume`, while KLEE emits
`klee_assume`/`abort()`.

### 4.3 Contract Source

Contracts are always read from the `!static_contract` metadata strings embedded on call instructions by `patchir-cir2llvm`, using the shared metadata parsing helpers described above.

If `-spec <yaml>` is provided, the `contracts:` section of the spec is also parsed and merged with the metadata-derived contracts. Spec entries take precedence over metadata for the same function, allowing per-deployment overrides without re-running the full transform pipeline.

### 4.3 Function Discovery

The tool walks all LLVM `Function`s in the module. For each function, it walks all `Instruction`s checking for a `!static_contract` metadata string, since `patchir-cir2llvm` can attach static contract metadata to any CIR operation (loads, stores, arithmetic, calls, etc.) that carried a `contract.static` attribute in the CIR module. Enclosing functions with at least one such instruction are collected as harness targets. If `-function <name>` is specified, the tool restricts to that function only.

### 4.4 Argument Type Handling

| LLVM Type | Harness Strategy |
|---|---|
| `i8` – `i32` / `i64` | `alloca` + `klee_make_symbolic` + `load` |
| Pointer (`ptr`) | `alloca [buffer_size x i8]` + `klee_make_symbolic` on array + `getelementptr` |
| `float` / `double` | `alloca` + `klee_make_symbolic` + `load` |
| Struct by value | `alloca` struct type + `klee_make_symbolic` + `load` |
| Function pointer | Emit a stub function and pass its address; emit a warning |

---

## 5. Generated Harness Structure

For a patched function `int f(int arg0, char* arg1)` with:

- **Precondition:** `arg0 >= 0`
- **Postcondition:** `return_value in [0, 32]`

**Note on pointer width:** The harness must derive integer widths from the
module's `DataLayout` rather than hardcoding `i64`. For 32-bit firmware targets
(e.g. `ARM:LE:32:Cortex`), the size parameter to `klee_make_symbolic` and the
argument to `klee_assume` should use the target's native pointer-sized integer
(e.g. `i32`). The examples below use `i64` for readability; the implementation
must use `DataLayout::getIntPtrType()` to select the correct width.

The tool generates the following LLVM IR appended to the patched module:

```llvm
; ─── KLEE runtime declarations (added once per module) ───────────────────────
; NOTE: size_t and assume argument widths are derived from DataLayout;
; i64 is shown here for a 64-bit target. For 32-bit targets these become i32.
declare void @klee_make_symbolic(ptr, i64, ptr)
declare void @klee_assume(i64)
declare void @abort() noreturn

@.klee_name_arg0 = private constant [5 x i8] c"arg0\00"
@.klee_name_arg1 = private constant [5 x i8] c"arg1\00"

; ─── Harness driver ──────────────────────────────────────────────────────────
define void @klee_main_f() {
entry:
  ; --- Symbolic inputs ---
  %arg0 = alloca i32, align 4
  call void @klee_make_symbolic(ptr %arg0, i64 4, ptr @.klee_name_arg0)

  %arg1_buf = alloca [64 x i8], align 1       ; pointer arg → fixed-size buffer
  call void @klee_make_symbolic(ptr %arg1_buf, i64 64, ptr @.klee_name_arg1)

  ; --- Preconditions via klee_assume ---
  %arg0_val = load i32, ptr %arg0
  %pre0 = icmp sge i32 %arg0_val, 0
  %pre0_ext = zext i1 %pre0 to i64
  call void @klee_assume(i64 %pre0_ext)

  ; --- Call target ---
  %arg1_ptr = getelementptr [64 x i8], ptr %arg1_buf, i64 0, i64 0
  %result = call i32 @f(i32 %arg0_val, ptr %arg1_ptr)

  ; --- Postconditions: abort on violation (KLEE detects abort as error) ---
  %post_lo = icmp sge i32 %result, 0
  %post_hi = icmp sle i32 %result, 32
  %post_ok = and i1 %post_lo, %post_hi
  br i1 %post_ok, label %done, label %fail

fail:
  call void @abort()
  unreachable

done:
  ret void
}
```

### 5.1 Design Rationale

- **`klee_assume` for preconditions** — constrains the symbolic input space to valid inputs only, eliminating spurious paths that violate caller obligations.
- **`abort()` for postcondition violations** — KLEE treats `abort()` as a detectable error, generating a test case that demonstrates the violation.
- **Fixed-size stack buffers for pointer args** — avoids heap allocation complexity; size is configurable via `-buffer-size` or per-argument YAML override.
- **One driver per target function** — KLEE is invoked independently per entry point.

---

## 6. Predicate-to-IR Translation

This section defines how each predicate kind from `Contract.td` is lowered to KLEE harness IR. Preconditions emit `klee_assume` (constraining symbolic inputs); postconditions emit `if (!cond) abort()` (KLEE reports the violation).

### 6.1 Targets

Targets identify what a predicate applies to. The YAML spec and serialized metadata use different spellings; the parser normalizes both:

| YAML | Serialized (in `!static_contract`) | Operand in IR |
|---|---|---|
| `arg0` .. `argN` | `Arg(0)` .. `Arg(N)` | Loaded value of the Nth symbolic argument |
| `return_value` | `ReturnValue` | `%result` from `call @f(...)` |
| `symbol` + `symbol: "name"` | `Symbol(@name)` | `load @name` from the module |

### 6.2 Values and Constants

String values in `relation.value` and `range.min`/`max` resolve as follows:

- Numeric literals (`"0"`, `"-1"`, `"0x1000"`) — parsed to `ConstantInt` matching the target type.
- `"NULL"` / `"null"` — `ConstantPointerNull` for pointers; zero for integers.
- Named constants (`"USB_MAX_PACKET_SIZE"`) — looked up as a module global; diagnostic + skip if absent. C `#define` values are not available as LLVM symbols and must be pre-resolved in the YAML spec.

### 6.3 Translation by Predicate Kind

Each subsection shows the IR pattern per target. `V` denotes the loaded target value; `PRE` = `klee_assume(cond)`, `POST` = `br i1 cond, %done, %fail` where `%fail` calls `abort()`.

#### `nonnull`

| Target | PRE | POST |
|---|---|---|
| `Arg(N)` ptr | No-op (harness allocates stack buffer; always non-null) | `icmp eq V, null` → abort (pointer-to-pointer output args) |
| `ReturnValue` | N/A | `icmp eq %result, null` → abort |
| `Symbol(@s)` | `klee_assume(load @s != null)` | `icmp eq load @s, null` → abort |

#### `relation`

Maps `relation` field to `icmp` (signed by default; unsigned for pointer-width / unsigned types):

| Relation | `icmp` | | Relation | `icmp` |
|---|---|---|---|---|
| `eq` | `eq` | | `gt` | `sgt` / `ugt` |
| `neq` | `ne` | | `gte` | `sge` / `uge` |
| `lt` | `slt` / `ult` | | `none` | no IR (existence check) |
| `lte` | `sle` / `ule` | | | |

For `Arg(N)`: `PRE = klee_assume(icmp <rel> V, const)`. For `ReturnValue` / `Symbol(@s)`: `POST = if (!(icmp <rel> V, const)) abort()`. `relation: none` emits no IR — it is an existence assertion only.

#### `range`

Emits two `icmp` instructions ANDed together:

```
%lo = icmp sge V, resolved_min
%hi = icmp sle V, resolved_max
%ok = and i1 %lo, %hi
```

`Arg(N)` → PRE (`klee_assume`). `ReturnValue` / `Symbol(@s)` → POST (`abort` on `!%ok`).

#### `alignment`

Checks `(ptr & (align-1)) == 0` using `ptrtoint` + `and` + `icmp eq`:

| Target | PRE | POST |
|---|---|---|
| `Arg(N)` ptr | `klee_assume(...)` | N/A |
| `Symbol(@s)` | `klee_assume(...)` | N/A |
| `ReturnValue` | N/A | N/A (not meaningful) |

#### `expr`

Free-form C-like expression string (e.g., `"size > 0 && size < MAX_LEN"`). Translation depends on complexity:

- **Simple** (`"arg0 != 0"`) — lowered as a `relation` predicate.
- **Compound** (`"a > 0 && a < 100"`) — parsed into `icmp` + `and`/`or` tree.
- **Member access** (`"dev->state == CONFIGURED"`) — emits `getelementptr` + `load`; requires struct type metadata.
- **Unresolvable** — emits a warning comment in output IR; predicate is skipped.

PRE: `klee_assume(compiled_result)`. POST: `if (!compiled_result) abort()`.

### 6.4 Metadata String Format

`serializeStaticContract()` (`tools/patchir-cir2llvm/main.cpp:136–231`) produces:

```text
"preconditions=[{id=\"...\", kind=relation, target=Arg(1), relation=neq, value=0}], postconditions=[{id=..., kind=range, target=ReturnValue, range=[min=0, max=32]}]"
```

Fields per `{...}` block: `id` (always), `kind` (always), `target` (all except `expr`), plus kind-specific fields — `relation`+`value`, `range=[min=..., max=...]`, `align`, `expr`, or `symbol` (when target is `Symbol`). `parseStaticContractString()` in `MetadataParser.hpp` lexes this into `ParsedPredicate` structs.

---

## 7. YAML Override and Configuration (Optional)

A minimal optional section can be added to the existing patch YAML. If absent, the tool uses `contracts:` and `meta_contracts:` as-is.

```yaml
klee_harness:
  - function: "patch__replace__sprintf"    # LLVM function name in the IR
    contract: "sprintf_contract"           # references a contracts: entry by name
    buffer_sizes:                          # per-arg buffer size override (default: 64 bytes)
      1: 256                               # arg index → byte count
```
---

## 8. Runtime Contract Support

Runtime contracts differ from static contracts: `patchir-transform` **compiles** the C contract function and merges it directly into the module. In the patched `.ll` file, `contract__<name>()` is **already fully defined** and called inline inside the patched function body. KLEE executes it automatically as part of symbolic execution — no additional IR synthesis is required.

### 8.1 Example

```llvm
; Inside bl_device__process_entry — present after patchir-transform:
%37 = call i32 @patch__replace__sprintf(...), !static_contract !56
call void @contract__sprintf(i32 %37, i32 32)   ; runtime contract (fully defined)

; Defined in the same module:
define void @contract__sprintf(i32 %0, i32 %1) {
  ...
  call void @__patchestry_assert_fail(...)       ; violation path
}

; Declared but NOT defined — stubs must be injected for KLEE:
declare void @__patchestry_assert_fail(ptr, ptr, i32, ptr)
```

### 8.2 Verification Shim Injection

Before writing the output module, `patchir-klee-harness` scans for declared-but-not-defined verification symbols and injects KLEE-compatible stubs:

| Symbol | Injected Stub Behavior |
|---|---|
| `__patchestry_assert_fail(...)` | `call void @abort(); unreachable` |

These stubs are **only injected if the symbol has no existing definition**, so they never override user-defined stubs. Injection happens once per output module regardless of how many target functions are processed.

### 8.3 Static vs. Runtime Contract Handling Summary

| Aspect | Static Contract | Runtime Contract |
|---|---|---|
| **Source** | `!static_contract` metadata or YAML `contracts:` | `contract__*()` function already defined in module |
| **Tool action** | Synthesize `klee_assume` / `abort` IR predicates | No synthesis required — already in IR |
| **Violation detection** | `if (!post) abort()` inside `klee_main_*` | `__patchestry_assert_fail` → `abort()` shim |
| **Harness structure** | `klee_main_*` driver with symbolic inputs | Same driver — runtime contract executes inside patched fn |

Both contract types are handled in the same `klee_main_<funcname>()` driver. A function may have both: static predicates translated to explicit checks, and runtime contract calls already embedded in its body.

---

## 9. Open Questions and Next Steps

- **Path explosion mitigation.** Functions with deep loop nests or large switch tables may cause KLEE to time out. Investigate `--max-depth`, `--max-time`, and search-heuristic flags to keep exploration tractable for typical firmware patches.
- **Composite struct contracts.** The current predicate set covers scalars, pointers, and ranges. Contracts that refer to individual struct fields (e.g., "field `len` must equal `strlen(buf)`") will require extending it.
- **Multi-function call-chain verification.** The current design generates one driver per function. Verifying contracts across a call chain (caller preconditions imply callee preconditions) would require an interprocedural harness mode.
- **SeaHorn convergence.** Evaluate whether the harness generator can emit both KLEE and SeaHorn drivers from the same contract metadata, reducing duplicate tooling in `analysis/`.