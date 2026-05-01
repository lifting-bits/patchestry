# Firmware Patch Specification

This document outlines the specification for patching firmware binaries using a declarative YAML format. The specification applies patches at the MLIR (Multi-Level Intermediate Representation) level, specifically using Clang IR MLIR representation, allowing for architecture-independent binary modifications.

## Overview

The patch specification is designed to apply patches at specific points in the decompiled MLIR representation:
- Before or after function calls
- At specific operations like memory loads, stores, or other operations
- Replace entire functions or operations
- Based on variable matching and symbolization

By working at the MLIR level, patches can be written once and applied across multiple architectures, as the MLIR abstraction handles the architecture-specific details.

## Matching Types

Patchestry supports two types of matching:

1. **Function-Based Matching**: Matches and instruments entire function calls
2. **Operation-Based Matching**: Matches and instruments specific MLIR operations within functions

### Function-Based vs Operation-Based Matching

| Aspect | Function-Based | Operation-Based |
|--------|----------------|-----------------|
| **Granularity** | Entire function calls | Individual operations (load, store, arithmetic, etc.) |
| **Match Criteria** | Function symbol, arguments | Operation type, function context, variables |
| **Use Cases** | API monitoring, function replacement | Memory access tracking, operation validation |
| **Required Fields** | `match.name` + `match.kind` | `match.name` + `match.kind` |
| **Context** | Caller context | Function containing the operation |

## Specification Format

A patchestry YAML file has a top-level `patches:` key listing the
patches to apply. Each entry names a target match, a mode, and the
patch function to invoke.

```yaml
apiVersion: patchestry.io/v1           # API version

metadata:
  name: "deployment-name"              # Deployment metadata
  description: "Deployment description"
  version: "1.0.0"
  author: "Author Name"
  created: "YYYY-MM-DD"
  organization: "organization-name"
  kind: PatchSpec                      # Optional: PatchSpec (deployment)
                                       #   or PatchLibrary (library)

target:                                # Target binary configuration
  binary: "target_binary.bin"
  arch: "ARCHITECTURE:ENDIANNESS:BITWIDTH:VARIANT"

libraries:                             # External patch and contract libraries
  - "path/to/library.yaml"             # Each file carries patch/contract definitions

patches:                               # Patch configurations
  - name: "..."
    id: "..."
    description: "..."
    match:
      name: "..."
      kind: "..."
      context: ["..."]
    action:
      mode: "..."
      patch: "..."
      arguments:
        - source: "..."
          index: ...
        - source: "..."
          value: ...
      optimization: ["..."]
```

### Matching multiple callees

Use `names:` (list) instead of `name:` (scalar) to match any of several
callees with the same action (OR semantics, equivalent to PatchDSL's
`pattern-either:`):

```yaml
patches:
  - name: "bounded_copy"
    match:
      names: ["strcpy", "strcat", "sprintf"]
      kind: "function"
      context: ["eeprom_write"]
    action:
      mode: "replace"
      patch: "safe_copy"
      arguments:
        - source: "operand"
          index: 0
```

Each name fires independently — if both `strcpy` and `strcat` appear
in the matched context, both are replaced.

For ad-hoc patterns, `name:` also accepts regex with `/pattern/` syntax:

```yaml
    match:
      name: "/str(cpy|cat)|sprintf/"
      kind: "function"
```

### Contracts

Contracts use the `contracts:` top-level key. They are **static only** —
declarative pre/postcondition predicates that attach to the matched op
as MLIR attributes (`contract.static`). They emit no runtime code.
Valid modes: `apply_before` and `apply_after` (both attach the same
attribute; the mode just controls which op the attribute lands on
relative to the match). `apply_at_entrypoint`, `replace`, and `erase`
are patch-only.

```yaml
contracts:
  - name: "usb_msg_nonnull"
    id: "USB-CONTRACT-001"
    description: "Message pointer must be non-null"
    match:
      name: "usbd_ep_write_packet"
      context: ["bl_usb__send_message"]
    action:
      mode: "apply_before"
      contract: "message_nonnull_contract"
```

### Ordering

Entries apply in YAML declaration order: all `patches:` first (top to
bottom), then all `contracts:` (top to bottom). No separate ordering
field is needed — reorder the entries in the file to reorder execution.

---

## Contracts: Static Only

`contracts:` describes **static** contracts — declarative
pre/postcondition predicates that the pass attaches to the matched op
as MLIR attributes (`contract.static`). They carry no executable code
and produce no runtime overhead; they're consumed by downstream
analyzers and verifiers (e.g. `patchir-klee-verifier`).

What used to be called a "runtime contract" — a C/C++ function called
at the matched site to validate a condition — was mechanically the
same as an `apply_before` / `apply_after` / `apply_at_entrypoint`
patch. Those cases have been merged into `patches:`; write the
validator as a patch whose body does the check and asserts (or
traps) on failure.

**Migration**: if you have an older YAML with `type: "RUNTIME"` or a
`code_file` under `contracts:`, move the entry into `patches:` with
the same `code_file` / `function_name` and pick an appropriate patch
mode. The parser rejects runtime contracts with a pointer to this
section.

## Field Descriptions

### Top-Level Fields

| Field | Description | Example |
|-------|-------------|---------|
| `apiVersion` | API version for the patch specification | `"patchestry.io/v1"` |
| `metadata` | Deployment metadata container | See metadata fields below |
| `target` | Target binary configuration | See target fields below |
| `libraries` | External patch and contract library references | See library fields below |
| `patches` | Patch entries to apply (in declaration order) | List of patch entries |
| `contracts` | Contract entries to apply (in declaration order) | List of contract entries |

### Metadata Fields

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Deployment name identifier | `"usb-security-monitoring"` |
| `description` | Deployment description | `"Deploy USB security monitoring"` |
| `version` | Deployment version | `"1.0.0"` |
| `author` | Author or team name | `"Security Team"` |
| `created` | Creation date | `"2025-01-15"` |

### Target Fields

| Field | Description | Example |
|-------|-------------|---------|
| `binary` | Target binary file name | `"firmware.bin"` |
| `arch` | Target architecture specification in format "ARCH:ENDIANNESS:BITWIDTH" | `"ARM:LE:32"` |

### Libraries Fields

`libraries` is a list of paths to external library YAML files. Each
library file defines `patch_definitions`, `contract_definitions`, or
both. Paths are resolved relative to the location of the top-level spec
file.

```yaml
libraries:
  - "patches/my_patches.yaml"
  - "patches/my_contracts.yaml"   # can also hold contract definitions
```

| Field | Description | Example |
|-------|-------------|---------|
| `libraries` (list entry) | Path to a library YAML file containing `patch_definitions:` and/or `contract_definitions:` | `"patches/usb_security_patches.yaml"` |

### Patch / Contract Entry Fields

Each entry under a top-level `patches:` or `contracts:` list carries
identity metadata, a required nested `match:` mapping, and a required
nested `action:` mapping. Any action-level field (`mode`, `patch` /
`contract`, `arguments`, `optimization`) must live inside `action:`;
stray top-level spellings are rejected by the parser.

| Field | Required | Description | Example |
|-------|----------|-------------|---------|
| `name` | Yes | Descriptive entry name (used in logs) | `"usb_endpoint_write_validation_after"` |
| `id` | No | Stable action ID; auto-generated from `name` if omitted | `"USB-PATCH-001"` |
| `description` | No | Human-readable description | `"Validate USB endpoint writes"` |
| `match` | Yes | Nested mapping selecting where to apply — see [Match Fields](#match-fields) | See example |
| `action` | Yes | Nested mapping describing what to do — see [Action Fields](#action-fields) | See example |

### Match Fields

| Field | Description | Example |
|-------|-------------|---------|
| `match.name` | Name pattern to match — a function name for `kind: "function"`, or an MLIR operation name (e.g. `"cir.call"`) for `kind: "operation"` | `"usbd_ep_write_packet"`, `"cir.call"` |
| `match.names` | List of callee names with OR semantics (simplified form; mutually exclusive with `match.name`) | `["strcpy", "strcat"]` |
| `match.kind` | Type of match target (`function` or `operation`) | `"function"` |
| `match.context` | Functions where matches should be applied (list of names or `/regex/` patterns) | `["bl_usb__send_message"]` |
| `match.variable_matches` | Variables used in the operation (for function-based matching) | `name: "/.*password.*/"` |
| `match.argument_matches` | Function arguments to match (for function-based matching) | See below |
| `match.symbol_matches` | Symbols accessed by the operation (for operation-based matching) | `name: "/.*password.*/"` |
| `match.operand_matches` | Operands to match (for operation-based matching) | See below |
| `match.op_kind` | Narrow `cir.binop` / `cir.cmp` matches to a specific kind (see [Op-kind discriminators](#op-kind-discriminators)) | `"mul"` |
| `match.captures` | Bind operand/result values for reuse as `source: "capture"` arguments (see [Named captures](#named-captures)) | See below |

#### Operation-Based Matching

Operation-based matching is supported only for patches, meaning it is not supported for contract matching and insertion. For operation-based matching, set `kind: "operation"` and use `name` to specify the MLIR operation name to match. The following additional fields are available:

| Field | Description | Example |
|-------|-------------|---------|
| `match.name` | **Required** - MLIR operation name to match | `"cir.load"`, `"cir.store"`, `"cir.call"` |
| `match.kind` | **Required** - must be `"operation"` | `"operation"` |
| `match.function_context` | List of functions where operation should be matched | `[{name: "/.*secure.*/"}]` |
| `match.symbol_matches` | Symbols accessed by the operation | `[{name: "/.*secret.*/", type: "!cir.ptr<...>"}]` |
| `match.operand_matches` | Operands to match | `[{index: 0, name: "addr", type: "!cir.ptr<...>"}]` |

#### Function Context Fields

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Function name pattern (supports regex with `/pattern/`) | `"/.*secure.*/`, `"authenticate"` |

#### Symbol Match Fields  (Operation-Based Matching)

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Symbol name pattern (supports regex with `/pattern/`) | `"/.*password.*/`, `"secret_key"` |
| `type` | Symbol type pattern (optional) | `"int32*"` |

#### Operand Match Fields (Operation-Based Matching)

| Field | Description | Required | Example |
|-------|-------------|----------|---------|
| `index` | Position of the operand (0-based) | Yes | `0` (first operand), `1` (second operand) |
| `name` | Name of the operand variable | No | `"addr"`, `"/.*buffer.*/` |
| `type` | Type of the operand | No | `"int32*"` |


Common operand patterns:
- `cir.load`: operand 0 = address to load from
- `cir.store`: operand 0 = value to store, operand 1 = address to store to
- `cir.binop`: operand 0 = left operand, operand 1 = right operand
- `cir.call`: operands = function arguments

#### Argument Match Fields (Function-Based Matching)

| Field | Description | Required | Example |
|-------|-------------|----------|---------|
| `index` | Position of the argument (0-based) | Yes | `1` |
| `name` | Name of the argument | No | `"buff"` |
| `type` | Type of the argument | No | `"void*"` |


#### Variable Match Fields  (Function-Based Matching)

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Variable name pattern (supports regex with `/pattern/`) | `"/.*password.*/`, `"secret_key"` |
| `type` | Variable type pattern (optional) | `"struct struct_anon_struct_4_1_58265f66*"` |

### Action Fields

The action fields live under each entry's nested `action:` mapping
(required sibling of `match:`).

| Field | Required | Description | Example |
|-------|----------|-------------|---------|
| `mode` | Yes | Patching or contract mode to apply. Shared modes: `apply_before`, `apply_after`. Patch-only modes: `apply_at_entrypoint`, `replace`, `erase` | `"apply_before"` |
| `patch` | Under `patches:` (unless `mode: erase`) | Reference to the patch implementation `name:` in a library | `"usb_endpoint_write_validation_after"` |
| `contract` | Under `contracts:` | Reference to the static contract `name:` in a library | `"usb_msg_nonnull"` |
| `arguments` | No | List of arguments to pass to patch function (ignored for contracts) | See [Argument Specification](#argument-specification) |
| `optimization` | No | Optimization flags for this action | `["inline-patches"]` |

### Optimization Flags

The `optimization` field accepts a list of optimization settings on each
entry's `action:` block:

| Flag | Description | Effect |
|------|-------------|--------|
| `"inline-patches"` | Inline patch function calls after insertion | Reduces function call overhead |

> **Note:** `inline-contracts` is deprecated. Contracts are static-only —
> they attach `contract.static` MLIR attributes rather than emitting calls,
> so there is nothing to inline. The flag is still parsed for backward
> compatibility but produces a warning and has no effect.

## Argument Specification

Arguments passed to patch or contract functions can come from different sources and are specified using a structured format that in general supports operands, function arguments, variables, and constants.

### Argument Structure

```yaml
arguments:
  - name: "operand_value"
    source: "operand"
    index: 0
  - name: "constant_size"
    source: "constant"
    value: "1024"
  - name: "local_var"
    source: "variable"
    symbol: "buffer_size"
```

### Argument Fields

| Field | Description | Required | Example |
|-------|-------------|----------|---------|
| `name` | Descriptive name for the argument | Yes | `"left_operand"`, `"store_address"` |
| `source` | Where the argument comes from | Yes | `"operand"`, `"argument"`, `"variable"`, `"symbol"`, `"constant"` |
| `index` | Index for operands/arguments | When `source` is `"operand"` or `"argument"` | `0`, `1`, `2` |
| `symbol` | Symbol name for local variables or module-level globals/functions | When `source` is `"variable"` or `"symbol"` | `"key_size"`, `"bl_spi_mode"` |
| `value` | Literal value for constants | When `source` is `"constant"` | `"1024"`, `"0x1000"` |
| `is_reference` | When `true`, passes a pointer to the value instead of the value itself, allowing the patch function to mutate the caller's variable. Supported for `operand`, `variable`, and `symbol` sources. | No (default: `false`) | `true`, `false` |

### Argument Source Types

| Source | Description | Required Fields | Use Case |
|--------|-------------|-----------------|----------|
| `operand` (alias: `argument`) | Operand or argument of the matched operation by zero-based index; `operand` and `argument` are interchangeable | `index` | Access operands/arguments of matched calls or operations |
| `variable` | Local variable in the enclosing function scope, located by its IR name attribute | `symbol` | Access local variables (alloca'd locals) in scope |
| `symbol` | Module-level global variable or function pointer, located in the module symbol table | `symbol` | Pass a global variable or function pointer to the patch |
| `constant` | Literal constant value | `value` | Pass fixed values to patch functions |
| `return_value` | Return value of function or operation | None | Access return value (`apply_before` / `apply_after` modes only — **not valid for `apply_at_entrypoint`**) |
| `capture` | Named capture bound by `match.captures` — looks up an `mlir::Value` by name from the match site | `name` | Reference the same operand/result in multiple patch arguments, or rebind by name for readability (**not valid for `apply_at_entrypoint`** — captures are bound at the match site, not at function entry) |

### Named captures

The `match:` block can include a `captures:` list that binds an op's
operands and results to names. Patch arguments can then reference those
names via `source: "capture"` — equivalent to PatchDSL's `$A`, `$B` metavars.

```yaml
patches:
  - name: "widen_mul"
    match:
      name: "cir.binop"
      kind: "operation"
      context: ["compute_alloc_size"]
      captures:
        - name: "A"
          operand: 0
        - name: "B"
          operand: 1
        - name: "R"
          result: 0
    action:
      mode: "replace"
      patch: "checked_mul"
      arguments:
        - source: "capture"
          name: "A"
        - source: "capture"
          name: "B"
```

Capture fields:

| Field | Required | Meaning |
|-------|----------|---------|
| `name` | yes | The capture's name, referenced later via `source: "capture"` |
| `operand` | one of | Zero-based operand index to bind |
| `result` | one of | Zero-based result index to bind |
| `type` | no | Optional type constraint (advisory, not yet enforced) |

`operand:` and `result:` are mutually exclusive — exactly one must be set.

### Op-kind discriminators

For operation-kind matching (`kind: "operation"`), the `op_kind:` field
filters by the MLIR enum attribute carried on the op. This allows
targeting a specific arithmetic or comparison kind instead of every
`cir.binop` / `cir.cmp` instance.

```yaml
patches:
  - name: "widen_multiply"
    match:
      name: "cir.binop"
      kind: "operation"
      op_kind: "mul"       # only match Mul binops, not Add/Sub/Div/Rem/...
      context: ["compute_alloc_size"]
    action:
      mode: "replace"
      patch: "checked_mul"
      arguments:
        - source: "operand"
          index: 0
        - source: "operand"
          index: 1
```

Supported values per op type:

| Op | `op_kind` values |
|---|---|
| `cir.binop` | `mul`, `div`, `rem`, `add`, `sub`, `and`, `xor`, `or`, `max` |
| `cir.cmp` | `lt`, `le`, `gt`, `ge`, `eq`, `ne` |

Only `cir.binop` and `cir.cmp` participate in `op_kind:` filtering —
the matcher stringifies their `BinOpKind` / `CmpOpKind` attribute and
compares it to the YAML value. Omitting `op_kind:` matches every
instance of the named op. Setting it on any other op (including
`cir.load`, `cir.cast`, or `cir.shift`) causes the match to fail.

> **`mode: replace` requires `op_kind:` on `cir.binop` / `cir.cmp`.**
> A wildcard replace (no `op_kind:`) would substitute the same
> concrete patch for every arithmetic or comparison kind in scope —
> `add`, `sub`, `mul`, `div`, `and`/`or`/`xor`, or every relational
> operator. Operand types still line up, so the CIR verifier accepts
> the rewritten module, but the semantics are silently wrong. The
> spec parser therefore rejects `mode: replace` paired with a kinded
> generic op and no `op_kind:` filter at load time.
>
> Narrow with `op_kind:` when you want to swap a specific kind (see
> the `"mul"` example above), or use `mode: apply_before` /
> `apply_after` when the intent is observational — counters, trace
> probes, and other kind-agnostic instrumentation are the legitimate
> wildcard use case and stay unrestricted.

> **Shift ops live on `cir.shift`, not `cir.binop`.** Left/right
> shifts are a separate op whose direction is a unit attribute
> (`isShiftleft`), not a `BinOpKind` enumerator, so `op_kind: "shl"`
> / `op_kind: "shr"` on `cir.binop` matches nothing. To target a
> shift, match `name: "cir.shift"` without an `op_kind:` and
> (optionally in the future) filter by direction via a dedicated
> attribute check.

### Argument Examples

#### Function Call Arguments
```yaml
# Validate function arguments before call
arguments:
  - name: "dest_ptr"
    source: "argument"
    index: 0
  - name: "src_ptr"
    source: "argument"
    index: 1
  - name: "max_size"
    source: "constant"
    value: "4096"
```

#### Operation Operands
```yaml
# Check arithmetic overflow
arguments:
  - name: "left_val"
    source: "operand"
    index: 0
  - name: "right_val"
    source: "operand"
    index: 1
  - name: "overflow_limit"
    source: "constant"
    value: "4294967295"
```

#### Module-Level Global Symbol
```yaml
# Pass a module-level global variable to the patch function
# source: "symbol" looks up the name in the module symbol table (cir.GlobalOp or cir.FuncOp)
# Use this for globals and function pointers; use source: "variable" for local alloca'd variables
arguments:
  - name: "global_var"
    source: "symbol"
    symbol: "bl_spi_mode"
```

#### Return Value Handling
```yaml
# Access function return value (apply_before / apply_after modes only —
# both patches and contracts support it there).
# NOTE: return_value is NOT valid for apply_at_entrypoint — the call
# result is only defined at the matched call site, not at the function
# entrypoint. Use variable, symbol, or constant instead. capture is
# also rejected at entrypoint for the same reason.
arguments:
  - name: "result"
    source: "return_value"
  - name: "success_code"
    source: "constant"
    value: "0"
```

#### Mixed Argument Types
```yaml
# Comprehensive validation
arguments:
  - name: "memory_addr"
    source: "operand"
    index: 0
  - name: "buffer_size"
    source: "variable"
    symbol: "allocated_size"
  - name: "validation_level"
    source: "constant"
    value: "2"
  - name: "caller_func"
    source: "argument"
    index: 2
```

## Complete Patch Examples

### Example 1: USB Security Monitoring Deployment

While patches and contracts don't need to be used together, they can; here is an example involving both. Based on the actual specification from `bl_usb__send_message_before_patch.yaml`:

```yaml
apiVersion: patchestry.io/v1

metadata:
  name: "usb-security-monitoring-deployment"
  description: "Deploy USB security monitoring for medical device firmware"
  version: "1.0.0"
  author: "Security Team"
  created: "2025-01-15"

target:
  binary: "medical_device_firmware.bin"
  arch: "ARM:LE:32"

libraries:
  - "patches/usb_security_patches.yaml"

patches:
  - name: "usb_security_replace_patch"
    id: "USB-PATCH-001"
    description: "Replace usbd_ep_write_packet with a validated wrapper"
    match:
      name: "usbd_ep_write_packet"
      kind: "function"
      context: ["bl_usb__send_message"]
      argument_matches:
        - index: 0
          name: "usb_g"
          type: "struct struct_anon_struct_4_1_58265f66*"
    action:
      mode: "replace"
      patch: "usb_security_replace_patch"
      optimization: ["inline-patches"]
      arguments:
        - name: "operand_0"
          source: "operand"
          index: 0
        - name: "variable_2"
          source: "variable"
          symbol: "var1"

contracts:
  - name: "usb_security_contract_before"
    id: "USB-CONTRACT-001"
    description: "Assert allocation size is suitable before the call"
    match:
      name: "usbd_ep_write_packet"
      context: ["bl_usb__send_message"]
    action:
      mode: "apply_before"
      contract: "usb_security_contract"
```

## Operation-Based Matching Examples

### Example 1: Monitor Sensitive Load Operations

```yaml
- name: "sensitive_loads"
  match:
    name: "cir.load"
    kind: "operation"
    context:
      - "/.*secure.*/"   # Functions containing "secure"
      - "authenticate"   # Exact function name
    symbol_matches:
      - name: "/.*password.*/"  # Variables containing "password"
        type: "int32*"
```

### Example 2: Match Store Operations with Specific Operands

```yaml
- name: "validate_stores"
  match:
    name: "cir.store"
    kind: "operation"
    context:
      - "/.*critical.*/"
    operand_matches:
      - index: 0  # The value being stored (first operand)
        name: "user_input"
        type: "char*"
      - index: 1  # The address being stored to (second operand)
        name: "buffer"
        type: "char[256]"
```

### Pattern Matching

Both function context and variable names support regex patterns when enclosed in forward slashes:

- `/pattern/` - Regex pattern matching
- `exact_name` - Exact string matching

Examples:
- `/.*secure.*/` matches functions containing "secure" anywhere
- `/^test_.*/` matches functions starting with "test_"
- `authenticate` matches exactly "authenticate"

## Patch Modes

The specification supports five modes:

- `apply_before`: Apply patch or contract before the matched function or operation
- `apply_after`: Apply patch or contract after the matched function or operation completes
- `apply_at_entrypoint`: Insert a patch at the entry point of the **caller** function (patches only — see [Apply At Entrypoint Mode](#apply-at-entrypoint-mode))
- `replace`: Completely replace the matched function call or operation (patches only)
- `erase`: Delete the matched op without inserting any patch code (patches only — see [Erase Mode](#erase-mode))

### Apply Before Mode

In `apply_before` mode, the patch is applied before the matched function or operation executes.
```yaml
patches:
  - name: "security_precheck"
    id: "SECURITY-001"
    description: "Pre-execution validation"
    match:
      name: "sensitive_call"
      kind: "function"
      context: ["protected_fn"]
    action:
      mode: "apply_before"
      patch: "security_precheck"
      arguments:
        - name: "input_param"
          source: "operand"
          index: 0
        - name: "max_size"
          source: "constant"
          value: "4096"
```

### Apply After Mode

In `apply_after` mode, the patch or contract is applied immediately after the matched function call or operation completes. For patches, this works with both function-based and operation-based matching; for contracts, `apply_after` is currently only supported with function-based (function call) matching.

```yaml
patches:
  - name: "post_execution_logger"
    id: "LOGGING-001"
    description: "Post-execution logging"
    match:
      name: "sensitive_call"
      kind: "function"
      context: ["protected_fn"]
    action:
      mode: "apply_after"
      patch: "post_execution_logger"
      arguments:
        - name: "return_value"
          source: "return_value"
        - name: "execution_time"
          source: "variable"
          symbol: "timer_end"
```

### Replace Mode

In `replace` mode, the matched operation is completely replaced by a
call to the patch function. For function-kind matches the target is
always a `cir.call`. For operation-kind matches, **any op with at
least one result** may be replaced — including `cir.binop`, `cir.cmp`,
`cir.cast`, `cir.load`, `cir.get_member`, `cir.ptr_stride`, and
`cir.unary`. Result-less ops such as `cir.store` cannot be replaced
(there's no value to rewire); use `erase` or `apply_before`/`apply_after`
instead.

The patch function's result types must match (or be castable to) the
matched op's result types — the pass inserts casts where needed. If the
patch returns void but the original op had results, the pass logs an
error and leaves the op unchanged.

```yaml
patches:
  - name: "secure_replacement"
    id: "SECURE-REPLACEMENT-001"
    description: "Secure function replacement"
    match:
      name: "insecure_call"
      kind: "function"
      context: ["caller"]
    action:
      mode: "replace"
      patch: "secure_replacement"
      arguments:
        - name: "original_arg1"
          source: "operand"
          index: 0
        - name: "original_arg2"
          source: "operand"
          index: 1
```

### Erase Mode

In `erase` mode, the matched op is deleted. No patch function is called,
so `action.patch:` is not required.

When the deleted op has results that are used by other ops, each live result
is replaced with a default (zero for integers / null for pointers / false for
bools) so dependent ops remain well-formed. Unsupported result types
abort the erase for that op and log an error.

```yaml
patches:
  - name: "strip_debug_call"
    id: "ERASE-001"
    match:
      name: "debug_log"
      kind: "function"
      context: ["process_request"]
    action:
      mode: "erase"
```

Use ERASE for removing debug/logging calls, stripping unused cleanup
paths, or deleting obsolete instrumentation. For replacing a call with
a different function, use `replace` instead.

### Apply At Entrypoint Mode

`apply_at_entrypoint` is a **patch-only** mode. It inserts the patch call at the beginning of the **caller** function — the function that *contains* the matched call — rather than at the matched call site itself.

> Contracts do not support this mode. Static contracts attach as MLIR attributes on the matched op; there is no call to place, so "entrypoint" has no meaning for them. The earlier runtime-contract flavor that used `apply_at_entrypoint` has been merged into `patches:` — write the entry-block check as a patch.

> **Important**: The name can be misleading. The call is **not** inserted at the beginning of the matched function (the callee). It is inserted at the beginning of the *enclosing* function (the caller) named via `match.context`. Insertion happens after all `cir.alloca` ops and parameter-initialization stores so that all of the caller's parameters are in scope.

**When to use it**: When you want to validate invariants at function entry using the caller's local parameters, before any call inside the function executes.

**How it works** (example):
- `match.name` = `"usbd_ep_write_packet"` — the call to find inside the caller
- `match.context` = `["bl_usb__send_message"]` — the caller whose entry point receives the instrumentation
- The call is inserted at the start of `bl_usb__send_message`, not at the `usbd_ep_write_packet` call site
- Arguments (e.g., `source: "variable"`) are resolved from `bl_usb__send_message`'s local scope

**Argument-source restrictions**: because the inserted call runs at the function entry, SSA values bound at the matched call site are not in scope. The pass rejects:
- `source: "return_value"` — the call result is only defined at the match site
- `source: "capture"` — captures are bound at the match site, not at entry

Use `variable`, `symbol`, `constant`, or `operand` (which maps index N to the caller's Nth block argument) instead.

**Patch example**:
```yaml
patches:
  - name: "entrypoint_message_check"
    id: "ENTRY-PATCH-001"
    description: "Null-check message pointer at function entry"
    match:
      name: "usbd_ep_write_packet"       # call to find inside the caller
      kind: "function"
      context: ["bl_usb__send_message"]  # patch inserts HERE, at this function's entry
    action:
      mode: "apply_at_entrypoint"
      patch: "message_entry_check_patch"
      arguments:
        # source: variable — load a named local/parameter alloca at entry
        - source: "variable"
          symbol: "msg"
```

> **Note**: The call is inserted at the beginning of the **caller** (`bl_usb__send_message` — the function containing the matched call), not at the beginning of the matched function itself (`usbd_ep_write_packet`).

# Contract Library Specification

Contract libraries are separate YAML files that define reusable contracts. They can be referenced by multiple deployment specifications.

### Contract Library Structure

```yaml
apiVersion: patchestry.io/v1

metadata:
  name: "contract-library-name"
  version: "1.0.0"
  description: "Description of contract library"
  author: "Author Name"
  created: "YYYY-MM-DD"
  kind: PatchLibrary               # optional self-doc

contract_definitions:              # was `contracts:` in v1.0
  - name: "contract_name"
    description: "Contract description"
    category: "validation_category"
    severity: "critical|high|medium|low"
    preconditions: [...]
    postconditions: [...]
```

### Contract Library Fields

| Field | Description | Required | Example |
|-------|-------------|----------|---------|
| `apiVersion` | API version | Yes | `"patchestry.io/v1"` |
| `metadata` | Library metadata | Yes | See metadata fields |
| `contract_definitions` | List of contract specifications | Yes | See contract spec fields |

### Contract Specification Fields

Contracts are static-only. Runtime validators live under `patches:`.

| Field | Description | Required | Example |
|-------|-------------|----------|---------|
| `name` | Contract identifier | Yes | `"usb_validation_contract"` |
| `description` | Contract description | No | `"Validates USB parameters"` |
| `category` | Contract category | No | `"validation"`, `"security"` |
| `severity` | Severity level | No | `"critical"`, `"high"`, `"medium"`, `"low"` |
| `preconditions` | Static preconditions (list of predicate specs) | No | See predicate structure |
| `postconditions` | Static postconditions (list of predicate specs) | No | See predicate structure |

### Static Contract Predicates

Static contracts use declarative predicates that are attached as MLIR attributes. Each predicate specifies a condition that must hold.

#### Predicate Structure

```yaml
preconditions:
  - id: "precondition_1"
    description: "Description of the precondition"
    pred:
      kind: "nonnull|relation|alignment|expr|range"
      # Fields depend on kind - see "Predicate Kind Requirements" below
      target: "arg0|arg1|...|return_value|symbol"  # Required for: nonnull, relation, range
      relation: "eq|neq|lt|lte|gt|gte|none"        # Required for: relation
      value: "constant_value"                      # Required for: relation
      symbol: "symbol_name"                        # Optional: descriptive symbol name
      align: "alignment_bytes"                     # Required for: alignment
      expr: "expression_string"                    # Required for: expr
      range:                                       # Required for: range
        min: "min_value"
        max: "max_value"

postconditions:
  - id: "postcondition_1"
    description: "Description of the postcondition"
    pred:
      # Same structure as preconditions
```

#### Static Contract Predicate Fields

| Field | Description | Required | Example |
|-------|-------------|----------|---------|
| `id` | Unique identifier for the condition | Yes | `"precondition_1"` |
| `description` | Human-readable description | No | `"Ensure pointer is non-null"` |
| `pred` | Predicate specification | Yes | See predicate fields below |

#### Predicate Fields

| Field | Description | Required | Valid Values | Example |
|-------|-------------|----------|--------------|---------|
| `kind` | Type of predicate | Yes | `nonnull`, `relation`, `alignment`, `expr`, `range` | `"relation"` |
| `target` | What the predicate applies to | `nonnull`, `relation`, `range` | `arg0`, `arg1`, ..., `return_value`, `symbol` | `"arg0"` |
| `relation` | Comparison relation | `relation` only | `eq`, `neq`, `lt`, `lte`, `gt`, `gte`, `none` | `"neq"` |
| `value` | Constant value for comparison | `relation` only | String representation of value | `"0"`, `"NULL"` |
| `symbol` | Symbol name reference (descriptive) | If target is symbol | Symbol name | `"usb_device"` |
| `align` | Alignment requirement in bytes | `alignment` only | String representation of bytes | `"4"`, `"8"` |
| `expr` | Expression string | `expr` only | Expression string | `"usb_device != NULL"` |
| `range` | Range constraint | `range` only | Range object | See range fields |

#### Range Fields

| Field | Description | Example |
|-------|-------------|---------|
| `min` | Minimum value (inclusive) | `"0"` |
| `max` | Maximum value (inclusive) | `"USB_MAX_PACKET_SIZE"` |

#### Predicate Kinds and Required Fields

Each predicate kind requires specific fields. **Only include the fields listed for each kind** to avoid parser errors:

| Kind | Description | Required Fields | Optional Fields | Example Use Case |
|------|-------------|-----------------|-----------------|------------------|
| `nonnull` | Assert target is not null | `kind`, `target` | `symbol` | Null pointer guard |
| `relation` | Compare target against a value | `kind`, `target` (arg, return_value, or symbol), `relation`, `value` | `symbol` | Bounds checking, comparisons |
| `alignment` | Verify memory alignment | `kind`, `target`, `align` | `symbol` | Memory alignment requirements |
| `expr` | Free-form expression | `kind`, `expr` | `target`, `symbol` | Complex conditions |
| `range` | Verify value is within range | `kind`, `target` (arg, return_value, or symbol), `range` (with `min` and `max`) | `symbol` | Input validation, bounds |

#### Quick Reference: Field Requirements by Predicate Kind

| Predicate Kind | `kind` | `target` | `relation` | `value` | `align` | `expr` | `range` | `symbol` |
|----------------|--------|----------|------------|---------|---------|--------|---------|----------|
| **nonnull**    | ✓      | ✓        | ✗          | ✗       | ✗       | ✗      | ✗       | Optional |
| **relation**   | ✓      | ✓        | ✓          | ✓       | ✗       | ✗      | ✗       | Optional |
| **alignment**  | ✓      | ✓        | ✗          | ✗       | ✓       | ✗      | ✗       | Optional |
| **expr**       | ✓      | Optional | ✗          | ✗       | ✗       | ✓      | ✗       | ✗        |
| **range**      | ✓      | ✓        | ✗          | ✗       | ✗       | ✗      | ✓       | Optional |

**Legend**: ✓ = Required, ✗ = Not allowed/Not used, Optional = May be included

**Important Notes:**
- **`nonnull`**: Only requires `target` (e.g., `arg0`). Asserts the target pointer is not null. No other fields are used.
- **`relation`**: Must specify `target`, `relation` operator, and `value` to compare against. Target can be an argument, return value, or symbol reference.
- **`alignment`**: Must specify `target` and `align` (alignment in bytes). Typically used with pointer arguments.
- **`expr`**: Only requires the `expr` field with a free-form expression string. Target and symbol are optional for context.
- **`range`**: Must specify `target` and a `range` object with both `min` and `max` values.
- **`symbol`**: This field is always optional and serves as a descriptive reference to document what variable or symbol the predicate refers to.

#### Target Specification

The `target` field specifies what the predicate applies to:

- **`arg0`, `arg1`, etc.**: Function arguments (0-indexed)
- **`return_value`**: Return value of the function
- **`symbol`**: A named symbol (requires `symbol` field)

#### Relation Types

For `relation` kind predicates:

| Relation | Operators | Description |
|----------|-----------|-------------|
| `eq` | `==` | Equal to |
| `neq` | `!=` | Not equal to |
| `lt` | `<` | Less than |
| `lte` | `<=` | Less than or equal to |
| `gt` | `>` | Greater than |
| `gte` | `>=` | Greater than or equal to |
| `none` | - | No relation (for existence checks) |

### Static Contract Examples

#### Example 1: Null Pointer Guard (nonnull)

```yaml
preconditions:
  - id: "ptr_nonnull"
    description: "First argument must not be null"
    pred:
      kind: "nonnull"
      target: "arg0"
      symbol: "dest"  # Optional: for documentation
```

**Required fields for `nonnull`**: `kind`, `target`

#### Example 2: Range Validation (range)

```yaml
preconditions:
  - id: "size_range"
    description: "Buffer size must be within valid range"
    pred:
      kind: "range"
      target: "arg1"
      range:
        min: "0"
        max: "USB_MAX_PACKET_SIZE"
      symbol: "buffer_size"  # Optional: for documentation
```

**Required fields for `range`**: `kind`, `target`, `range` (with `min` and `max`)

#### Example 3: Comparison Relation (relation)

```yaml
preconditions:
  - id: "size_positive"
    description: "Size must be greater than zero"
    pred:
      kind: "relation"
      target: "arg1"
      relation: "gt"
      value: "0"
      symbol: "size"  # Optional: for documentation
```

**Required fields for `relation`**: `kind`, `target`, `relation`, `value`

#### Example 4: Memory Alignment (alignment)

```yaml
preconditions:
  - id: "buffer_aligned"
    description: "Buffer must be 4-byte aligned"
    pred:
      kind: "alignment"
      target: "arg0"
      align: "4"
      symbol: "buffer"  # Optional: for documentation
```

**Required fields for `alignment`**: `kind`, `target`, `align`

#### Example 5: Complex Expression (expr)

```yaml
preconditions:
  - id: "device_valid"
    description: "USB device must be in configured state"
    pred:
      kind: "expr"
      expr: "usb_device->state == USB_STATE_CONFIGURED"
```

**Required fields for `expr`**: `kind`, `expr`
**Optional fields**: `target`, `symbol` (for documentation)

#### Example 6: Return Value Check (relation)

```yaml
postconditions:
  - id: "success_return"
    description: "Function must return success code"
    pred:
      kind: "relation"
      target: "return_value"
      relation: "eq"
      value: "0"
```

**Required fields for `relation` on return value**: `kind`, `target` (set to `return_value`), `relation`, `value`

### Complete Static Contract Example

This example demonstrates all predicate kinds with correct field usage:

```yaml
apiVersion: patchestry.io/v1

metadata:
  name: "usb-security-contracts"
  version: "1.0.0"
  description: "USB security contracts for medical devices"
  author: "Security Team"
  created: "2025-01-15"

contracts:
  - name: "usb_endpoint_write_validation"
    description: "Validate USB endpoint write parameters"
    category: "write_validation"

    preconditions:
      # Example 1: range predicate
      - id: "size_range"
        description: "Write size must be within valid range"
        pred:
          kind: "range"
          target: "arg2"
          range:
            min: "1"
            max: "USB_MAX_PACKET_SIZE"

      # Example 2: alignment predicate
      - id: "buffer_aligned"
        description: "Buffer must be properly aligned"
        pred:
          kind: "alignment"
          target: "arg1"
          align: "4"

    postconditions:
      # Example 3: relation predicate on return value
      - id: "return_success"
        description: "Function must return success or error code"
        pred:
          kind: "relation"
          target: "return_value"
          relation: "gte"
          value: "-1"

      # Example 4: expr predicate
      - id: "state_valid"
        description: "Device state must remain valid"
        pred:
          kind: "expr"
          expr: "usb_device->state != USB_STATE_ERROR"
          # Note: target and symbol are optional for expr predicates
```

**Summary of field usage by predicate kind:**
- **nonnull**: `kind` + `target` (`arg<N>`, `return_value`, or `symbol`)
- **relation**: `kind` + `target` + `relation` + `value` (`arg<N>`, `return_value`, or `symbol`)
- **alignment**: `kind` + `target` + `align` (`arg<N>`, `return_value`, or `symbol`)
- **expr**: `kind` + `expr`
- **range**: `kind` + `target` + `range` (with `min`/`max`) (`arg<N>`, `return_value`, or `symbol`)

### Runtime Validators — use `patches:`

What was previously called a "runtime contract" (a C/C++ function
called at the match site to validate a condition) is now expressed
as a plain patch. Write the validator like any other patch
implementation and dispatch it with `apply_before`, `apply_after`,
or `apply_at_entrypoint`:

```yaml
# Patch library entry — same code_file/function_name shape as the old
# runtime-contract entry had.
patches:
  - name: "usb_endpoint_write_validation"
    description: "Runtime validation for USB endpoint write"
    code_file: "patches/patch_usb_validation.c"
    function_name: "patch::before::usb_endpoint_write"
    parameters:
      - name: "usb_device"
        type: "usb_device_t*"
      - name: "buffer"
        type: "const void*"
      - name: "size"
        type: "uint32_t"
```

Use a `contracts:` entry for the *static* predicate you want the
verifier to check (pre/post on the same call site) — the two are
complementary: the patch does the runtime check, the contract
encodes the invariant for static analysis.

## Deployment Architecture

The patch architecture allows for:

1. **Modular Organization**: Group related patches into logical units
2. **External Libraries**: Reference shared patch and contract libraries
3. **Execution Ordering**: Control the order of patch application
4. **Optimization Control**: Fine-tune performance characteristics
5. **Exclusion Patterns**: Exclude specific functions from patching

## Limitations

The current matcher handles single-op C expressions at the CIR level.
The following patterns are **not yet supported**:

### Control-flow and region-carrying ops

Ops that carry regions (bodies or branches) can be matched by name
(`name: "cir.if"`, etc.) but cannot be meaningfully used with REPLACE,
and captures cannot reach into their regions.

| C pattern | CIR representation |
|---|---|
| `if (cond) { … } else { … }` | `cir.if` (two regions) |
| `for`, `while`, `do-while` | `cir.for`, `cir.while`, `cir.do` |
| `switch (x) { cases… }` | `cir.switch` |
| Scoped blocks `{ … }` | `cir.scope` |
| `a ? b : c` | `cir.ternary` |
| `a && b`, `a \|\| b` (short-circuit) | `cir.ternary` / region ops |

### Statement-level and multi-op patterns

Patterns that match more than one op in sequence require dataflow
reachability analysis and are not expressible today:

- `free(p); p = NULL;` — paired-op patterns
- `p = malloc(n); … use(p) …` — resource-tracking patterns
- Patterns anchored by preceding/following ops

### Variadic captures

PatchDSL's variadic metavar (`$...ARGS`) is not implemented. For example,
matching `printf(fmt, $...ARGS)` to capture an arbitrary number of
arguments is not possible. Workaround: match by callee name and use
`apply_before`/`apply_after` without forwarding variadic args.

### Result-less ops with REPLACE

`cir.store`, `cir.return`, and other ops without results cannot be
replaced — there is no value to rewire. Use `erase` together with
`apply_before`/`apply_after` to achieve equivalent rewrites.

### Semantic predicates

PatchDSL-style `where:` predicates (`nonnull(x)`, `tainted(x from src)`,
`bounded(x)`, `aliases(a, b)`, `reaches(a, b)`) require dedicated
analysis passes and are not part of the YAML surface today.

## Future work

These capabilities are planned but deliberately out of scope for the
current surface:

1. **Region-aware matching.** Match the structure of `cir.if` /
   `cir.for` / `cir.while` and bind captures from their condition
   and body regions. Enables rewrites like "wrap the then-branch of
   this if with a pre-check".

2. **Multi-op pattern anchors.** Sequences like
   `free($P); …; free($P)` with reachability between the anchors.
   Requires a dataflow pass that tracks capture identity across
   intervening ops.

3. **Variadic operand capture.** `$...ARGS` binding to the trailing
   operands of a call or the remaining operands of a variadic op.
   Needed for patches that forward an arbitrary number of arguments.

4. **Semantic predicates.** `where:` clauses backed by MLIR
   analyses — nullness, integer range, taint, alias, escape. Each
   predicate needs its own analysis pass; these land independently.

5. **Inline rewrites.** Pattern-to-pattern substitution without
   going through a C patch function, e.g.
   `rewrite: ($R: uint16_t) = (uint16_t)((uint32_t)$A * (uint32_t)$B)`.
   Requires a mini CIR codegen from C fragments.

6. **Cross-scope capture references.** A capture bound at a call
   site used by a patch inserted at the enclosing function's entry
   (currently SSA dominance rejects this). Would require rewriting
   captures to block-argument projections.

Contributions on any of these items are welcome; each is designed to
plug into the existing `MatchConfig` / capture plumbing rather than
requiring a full rewrite of the matcher.
