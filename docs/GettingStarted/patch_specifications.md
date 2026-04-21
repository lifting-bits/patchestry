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
apiVersion: patchestry.io/v1

metadata:
  name: "cwe078-fix"
target:
  binary: "firmware.bin"
  arch: "ARM:LE:32:v7"

libraries:
  - "patches/cwe078_patches.yaml"

patches:
  - name: "sanitize_system"
    id: "CWE-078-001"
    description: "Replace system() with sanitized wrapper"
    match:
      name: "system"
      kind: "function"
      context: ["create_port"]
    mode: "replace"
    patch: "cwe078_sanitized_system"
    arguments:
      - source: "operand"
        index: 0
      - source: "constant"
        value: 512
    optimization: ["inline-patches"]
```

> **Note:** A legacy nested format rooted at `meta_patches:` /
> `meta_contracts:` is still accepted for backward compatibility with
> existing specifications. All new patches should use `patches:` /
> `contracts:`. The two keys are mutually exclusive within one file.

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

Contracts use the `contracts:` top-level key with the same shape:

```yaml
contracts:
  - name: "usb_entry_check"
    id: "ENTRY-CONTRACT-001"
    description: "Null-check message pointer at function entry"
    match:
      name: "usbd_ep_write_packet"
      context: ["bl_usb__send_message"]
    mode: "apply_at_entrypoint"
    contract: "message_entry_check_contract"
    arguments:
      - source: "variable"
        symbol: "msg"
```

### Ordering

Entries apply in YAML declaration order: all `patches:` first (top to
bottom), then all `contracts:` (top to bottom). No separate ordering
field is needed — reorder the entries in the file to reorder execution.

---

## Contract Types

Patchestry supports two types of contracts:

1. **Runtime Contracts**: Implemented as C/C++ functions that are called at runtime to validate conditions
2. **Static Contracts**: Declarative specifications attached as MLIR attributes for static analysis and verification

### Runtime vs Static Contracts

| Aspect | Runtime Contracts | Static Contracts |
|--------|------------------|------------------|
| **Implementation** | C/C++ function code | Declarative predicates in YAML |
| **Verification** | Runtime checks during execution | Static analysis at compile time |
| **Performance** | Runtime overhead | No runtime overhead |
| **Expressiveness** | Full programming language | Limited to supported predicates |
| **Use Cases** | Complex validations, security checks | null checks, range checks, type constraints |


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

`libraries` is a list of paths to external library YAML files. Each library file may define `patches`, `contracts`, or both. Paths are resolved relative to the location of the top-level spec file.

```yaml
libraries:
  - "patches/my_patches.yaml"
  - "patches/my_contracts.yaml"   # can also hold contracts despite the name
```

| Field | Description | Example |
|-------|-------------|---------|
| `libraries` (list entry) | Path to a library YAML file containing `patches:` and/or `contracts:` definitions | `"patches/usb_security_patches.yaml"` |

### Meta-Patch Entry Fields

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Unique identifier for the patch group | `"usb_security_patches"` |
| `description` | Description of the patch group purpose | `"USB security monitoring patches"` |
| `optimization` | List of optimization flags | `["inline-patches", "inline-contracts"]` |
| `patch_actions` | List of individual patch actions | See [patch action fields](#patch-action-fields) below |

### Meta-Contract Entry Fields

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Unique identifier for the contract group | `"usb_control_flow_contracts"` |
| `description` | Description of the contract group purpose | `"USB control flow integrity contracts"` |
| `contract_actions` | List of individual contract actions | See [contract action fields](#contract-action-fields) below |

### Patch Action Fields

| Field | Description | Example |
|-------|-------------|---------|
| `id` | Unique identifier for the patch action | `"USB-PATCH-001"` |
| `description` | Description of what the patch does | `"Add USB security validation"` |
| `match` | List of match criteria | See [match fields](#match-fields) below |
| `action` | List of actions to apply | See [action fields](#action-fields) below |

### Contract Action Fields

| Field | Description | Example |
|-------|-------------|---------|
| `id` | Unique identifier for the contract action | `"USB-CONTRACT-001"` |
| `description` | Description of what the contract does | `"Add USB control flow integrity checking"` |
| `match` | List of match criteria | See [match fields](#match-fields) below |
| `action` | List of actions to apply | See [action fields](#action-fields) below |

### Match Fields

| Field | Description | Example |
|-------|-------------|---------|
| `match.name` | Name pattern to match — a function name for `kind: "function"`, or an MLIR operation name (e.g. `"cir.call"`) for `kind: "operation"` | `"usbd_ep_write_packet"`, `"cir.call"` |
| `match.kind` | Type of match target (`function` or `operation`) | `"function"` |
| `match.function_context` | Functions where matches should be applied | `name: "/.*secure.*/"` |
| `match.variable_matches` | Variables used in the operation (for function-based matching) | `name: "/.*password.*/"` |
| `match.argument_matches` | Function arguments to match (for function-based matching) | See below |
| `match.symbol_matches` | Symbols accessed by the operation (for operation-based matching) | `name: "/.*password.*/"` |
| `match.operand_matches` | Operands to match (for operation-based matching) | See below |

> **Multiple match entries**: Currently, only the first entry under `match:` is evaluated by the implementation. If multiple entries are provided, **only the first one is used** and any additional entries are ignored.

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

| Field | Description | Example |
|-------|-------------|---------|
| `mode` | Patching or contract mode to apply. Patch modes: `apply_before`, `apply_after`, `replace`. Contract-only mode: `apply_at_entrypoint` | `"apply_before"` |
| `patch_id` | Reference to the patch implementation | `"USB-PATCH-001"` |
| `description` | Description of the action being applied | `"Pre-validation security check"` |
| `arguments` | List of arguments to pass to patch function | See [Argument Specification](#argument-specification) |

### Optimization Flags

The `optimization` field accepts a list of optimization settings:

| Flag | Applies to | Description | Effect |
|------|------------|-------------|--------|
| `"inline-patches"` | `meta_patches` | Inline patch function calls after insertion | Reduces function call overhead |
| `"inline-contracts"` | `meta_contracts` | Inline contract function calls after insertion | Reduces contract validation overhead |

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
| `capture` | Named capture bound by `match.captures` — looks up an `mlir::Value` by name from the match site | `name` | Reference the same operand/result in multiple patch arguments, or rebind by name for readability |

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
| `cir.binop` | `mul`, `div`, `rem`, `add`, `sub`, `shl`, `shr`, `and`, `xor`, `or`, `max` |
| `cir.cmp` | `lt`, `gt`, `le`, `ge`, `eq`, `ne` |

Omitting `op_kind:` matches every instance of the named op (pre-existing
behavior). Setting it on an op type without a `kind` attribute (e.g.
`cir.load`) causes the match to fail.

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
# Access function return value (apply_before / apply_after mode only)
# NOTE: return_value is NOT valid for apply_at_entrypoint — the call
# result is only defined at the matched call site, not at the function
# entrypoint.  Use variable, symbol, or constant instead.
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

meta_patches:
  - name: usb_security_meta_patches
    description: "Meta patches for USB security"
    optimization:
      - "inline-patches"
      - "inline-contracts"

    patch_actions:
      - id: "USB-PATCH-001"
        description: "Patch to add USB security validation"
        match:
          - name: "usbd_ep_write_packet"
            kind: "function"
            function_context:
              - name: "bl_usb__send_message"
            argument_matches:
              - index: 0
                name: "usb_g"
                type: "struct struct_anon_struct_4_1_58265f66*"

        action:
          - mode: "replace"
            patch_id: "USB-PATCH-001"
            description: "Pre-validation security patch"
            arguments:
              - name: "operand_0"
                source: "operand"
                index: "0"
              - name: "variable_2"
                source: "variable"
                symbol: "var1"

meta_contracts:
  - name: usb_security_meta_contracts
    description: "Contracts for USB security validation"
    contract_actions:
      - id: "USB-CONTRACTS"
        description: "Assert allocation size is suitable"
        match:
          - name: "usbd_ep_write_packet"
            kind: "function"
            function_context:
              - name: "bl_usb__send_message"
        action:
          - mode: "apply_before"
            contract_id: "USB-CONTRACT-001"
            description: "Pre-validation security contract"
            arguments:
              - name: "operand_0"
                source: "operand"
                index: "0"
```

## Operation-Based Matching Examples

### Example 1: Monitor Sensitive Load Operations

```yaml
- name: "sensitive_loads"
  match:
    - name: "cir.load"
      kind: "operation"
      function_context:
        - name: "/.*secure.*/"  # Functions containing "secure"
        - name: "authenticate"  # Exact function name
      symbol_matches:
        - name: "/.*password.*/"  # Variables containing "password"
          type: "int32*"
```

### Example 2: Match Store Operations with Specific Operands

```yaml
- name: "validate_stores"
  match:
    - name: "cir.store"
      kind: "operation"
      function_context:
        - name: "/.*critical.*/"
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

The specification supports four patch modes and one contract-only mode:

- `apply_before`: Apply patch or contract before the matched function or operation
- `apply_after`: Apply patch or contract after the matched function or operation completes
- `replace`: Completely replace the matched function call or operation (patches only)
- `erase`: Delete the matched op without inserting any patch code (patches only — see [Erase Mode](#erase-mode))
- `apply_at_entrypoint`: Insert a contract at the entry point of the **caller** function (contracts only — see [Apply At Entrypoint Mode](#apply-at-entrypoint-mode))

### Apply Before Mode

In `apply_before` mode, the patch is applied before the matched function or operation executes.
```yaml
action:
  - mode: "apply_before"
    patch_id: "SECURITY-001"
    description: "Pre-execution validation"
    arguments:
      - name: "input_param"
        source: "operand"
        index: "0"
      - name: "max_size"
        source: "constant"
        value: "4096"
```

### Apply After Mode

In `apply_after` mode, the patch or contract is applied immediately after the matched function call or operation completes. For patches, this works with both function-based and operation-based matching; for contracts, `apply_after` is currently only supported with function-based (function call) matching.

```yaml
action:
  - mode: "apply_after"
    patch_id: "LOGGING-001"
    description: "Post-execution logging"
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
action:
  - mode: "replace"
    patch_id: "SECURE-REPLACEMENT-001"
    description: "Secure function replacement"
    arguments:
      - name: "original_arg1"
        source: "operand"
        index: "0"
      - name: "original_arg2"
        source: "operand"
        index: "1"
```

### Erase Mode

In `erase` mode, the matched op is deleted. No patch function is called, so
`patch_id` (nested form) or `patch:` (simplified form) is not required.

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
    mode: "erase"
```

Use ERASE for removing debug/logging calls, stripping unused cleanup
paths, or deleting obsolete instrumentation. For replacing a call with
a different function, use `replace` instead.

### Apply At Entrypoint Mode

`apply_at_entrypoint` is a **contract-only** mode that inserts the contract call at the beginning of the **caller** function — the function that *contains* the matched call — rather than at the matched call site itself.

> **Important**: The name can be misleading. The contract is **not** inserted at the beginning of the matched function (the callee). It is inserted at the beginning of the *enclosing* function (the caller) that was specified in `function_context`. Insertion happens after all `cir.alloca` ops and parameter-initialization stores so that all of the caller's parameters are in scope.

**When to use it**: When you want to validate invariants at function entry using the caller's local parameters, before any call inside the function executes.

**How it works** (example):
- `match.name` = `"usbd_ep_write_packet"` — the call to find inside the caller
- `match.function_context.name` = `"bl_usb__send_message"` — the caller whose entry point receives the contract
- The contract call is inserted at the start of `bl_usb__send_message`, not at the `usbd_ep_write_packet` call site
- Arguments (e.g., `source: "variable"`) are resolved from `bl_usb__send_message`'s local scope

```yaml
contract_actions:
  - id: "ENTRY-CONTRACT-001"
    description: "Null-check message pointer at function entry"

    match:
      - name: "usbd_ep_write_packet"   # call to find inside the caller
        kind: "function"
        function_context:
          - name: "bl_usb__send_message"  # contract inserts HERE, at this function's entry

    action:
      - mode: "apply_at_entrypoint"
        contract_id: "message_entry_check_contract"
        description: "Runtime null-check on message pointer at bl_usb__send_message entry"
        arguments:
          # source: variable — load a named local/parameter alloca at entry
          - name: "msg"
            source: "variable"
            symbol: "msg"
          # source: operand — index 0 maps to the 0th argument of bl_usb__send_message,
          # not the 0th operand of the usbd_ep_write_packet call
          - name: "usb_handle"
            source: "operand"
            index: 0
```

> **Note**: The contract is inserted at the beginning of the **caller** (`bl_usb__send_message` — the function containing the matched call), not at the beginning of the matched function itself (`usbd_ep_write_packet`).

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

contracts:
  - name: "contract_name"
    description: "Contract description"
    category: "validation_category"
    severity: "critical|high|medium|low"
    type: "STATIC|RUNTIME"

    # For STATIC contracts
    preconditions: [...]
    postconditions: [...]

    # For RUNTIME contracts
    function_name: ...
    code_file: ...
```

### Contract Library Fields

| Field | Description | Required | Example |
|-------|-------------|----------|---------|
| `apiVersion` | API version | Yes | `"patchestry.io/v1"` |
| `metadata` | Library metadata | Yes | See metadata fields |
| `contracts` | List of contract specifications | Yes | See contract spec fields |

### Contract Specification Fields

| Field | Description | Required | Type | Example |
|-------|-------------|----------|------|---------|
| `name` | Contract identifier | Yes | All | `"usb_validation_contract"` |
| `description` | Contract description | No | All | `"Validates USB parameters"` |
| `category` | Contract category | No | All | `"validation"`, `"security"` |
| `severity` | Severity level | No | All | `"critical"`, `"high"`, `"medium"`, `"low"` |
| `type` | Contract type | Yes | All | `"STATIC"` or `"RUNTIME"` |
| `preconditions` | Static preconditions | STATIC only | STATIC | List of precondition specs |
| `postconditions` | Static postconditions | STATIC only | STATIC | List of postcondition specs |
| `function_name` | Runtime contract function name | RUNTIME only | RUNTIME | Implementation details |
| `code_file` | Runtime contract implementation file | RUNTIME only | RUNTIME | Implementation details |

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
    type: "STATIC"

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

### Runtime Contract Implementation

For runtime contracts, specify the implementation details:

```yaml
contracts:
  - name: "usb_endpoint_write_contract"
    description: "Runtime validation for USB endpoint write"
    type: "RUNTIME"

    code_file: "contracts/usb_validation.c"
    function_name: "contract::usb_endpoint_write_validation"
    parameters:
      - name: "usb_device"
        type: "usb_device_t*"
        description: "USB device context"
      - name: "buffer"
        type: "const void*"
        description: "Data buffer"
      - name: "size"
        type: "uint32_t"
        description: "Buffer size"
```

## Deployment Architecture

The meta-patch architecture allows for:

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
