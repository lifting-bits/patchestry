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
| **Required Fields** | `match.symbol` | `match.operation` |
| **Context** | Caller context | Function containing the operation |

## Specification Format

The patch specification is a YAML file with the following structure:

```yaml
arch: "ARCHITECTURE:ENDIANNESS:BITWIDTH"  # Architecture specification
patches:
  - name: "PatchName"                     # Unique identifier for the patch
    match:                                # Match criteria
      symbol: "..."                       # Symbol to match
      kind: "..."                         # Kind of match (function, operation)
      # Additional match criteria...
    patch:                                # Patch configuration
      mode: "..."                         # Patch mode (ApplyBefore, ApplyAfter, Replace)
      patch_file: "..."                   # Path to patch implementation file
      patch_function: "..."               # Function in patch file to call
      arguments:                          # Arguments to pass to patch function
        - "..."
    exclude:                              # Exclusion criteria
      - "*"                               # Function name patterns to exclude from matching
```

## Field Descriptions

### Top-Level Fields

| Field | Description | Example |
|-------|-------------|---------|
| `arch` | Target architecture specification in format "ARCH:ENDIANNESS:BITWIDTH" | `"ARM:LE:32"` |
| `patches` | List of patch specifications | `- name: "PatchName"...` |


### Patch Entry Fields

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Unique identifier defining the patch identity | `"CVE-2021-12345"` |
| `match` | Container for match criteria | `symbol: "symbol_name"` |
| `patch` | Container for patch configuration | `mode: "ApplyBefore"` |
| `exclude` | Container for exclusion functions | `- "^func_*"` |

### Match Fields

| Field | Description | Example |
|-------|-------------|---------|
| `match.symbol` | Symbol name to match (for function-based matching) | `"read"` |
| `match.kind` | Type of match target (`function` or `operation`) | `"function"` |
| `match.operation` | Operation type to match (for operation-based matching) | `"cir.load"`, `"cir.store"`, `"cir.binop"` |
| `match.function_context` | Functions where operation matches should be applied | `name: "/.*secure.*/"` |
| `match.variable_matches` | Variables used in the operation (for function-based matching) | `name: "/.*password.*/"` |
| `match.argument_matches` | Function arguments or Opearnds to match (for function-based matching) | See below |
| `match.symbol_matches` | Variables used in the operation (for operation-based matching) | `name: "/.*password.*/"` |
| `match.operand_matches` | Function arguments or Opearnds to match (for operation-based matching) | See below |

#### Operation-Based Matching

For operation-based matching, the following additional fields are available:

| Field | Description | Example |
|-------|-------------|---------|
| `match.operation` | **Required** - MLIR operation name to match | `"cir.load"`, `"cir.store"`, `"cir.call"` |
| `match.function_context` | List of functions where operation should be matched | `[{name: "/.*secure.*/"}]` |
| `match.symbol_matches` | Variables accessed by the operation | `[{name: "/.*secret.*/", type: "!cir.ptr<...>"}]` |
| `match.operand_matches` | Operands matches for the operation | `[{index: 0, name: "addr", type: "!cir.ptr<...>"}]` |

#### Function Context Fields

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Function name pattern (supports regex with `/pattern/`) | `"/.*secure.*/`, `"authenticate"` |
| `type` | Function type pattern (optional) | `"!cir.func<!cir.void ()>"` |

#### Symbol Match Fields  (Operation-Based Matching)

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Symbol name pattern (supports regex with `/pattern/`) | `"/.*password.*/`, `"secret_key"` |
| `type` | Symbol type pattern (optional) | `"!cir.ptr<!cir.int<u, 32>>"` |

#### Operand Match Fields (Operation-Based Matching)

| Field | Description | Example |
|-------|-------------|---------|
| `index` | Position of the operand (0-based) | `0` (first operand), `1` (second operand) |
| `name` | Name of the operand variable | `"addr"`, `"/.*buffer.*/` |
| `type` | Type of the operand | `"!cir.ptr<!cir.int<u, 32>>"` |

Common operand patterns:
- `cir.load`: operand 0 = address to load from
- `cir.store`: operand 0 = value to store, operand 1 = address to store to
- `cir.binop`: operand 0 = left operand, operand 1 = right operand
- `cir.call`: operands = function arguments

#### Argument Match Fields (Function-Based Matching)

| Field | Description | Example |
|-------|-------------|---------|
| `index` | Position of the argument (0-based) | `1` |
| `name` | Name of the argument | `"buff"` |
| `type` | Type of the argument | `"void*"` |

#### Variable Match Fields  (Function-Based Matching)

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Variable name pattern (supports regex with `/pattern/`) | `"/.*password.*/`, `"secret_key"` |
| `type` | Variable type pattern (optional) | `"!cir.ptr<!cir.int<u, 32>>"` |

### Patch Fields

| Field | Description | Example |
|-------|-------------|---------|
| `patch.mode` | patching mode to be applied (`ApplyBefore`, `ApplyAfter`, `Replace`) | `"ApplyBefore"` |
| `patch.patch_file` | Path to the C file containing patch code | `"path/to/patch.c"` |
| `patch.patch_function` | Function in patch file to call | `"patch::before::function_name"` |
| `patch.arguments` | List of arguments to pass to patch function | See [Argument Specification](#argument-specification) |

### Exclude Fields

Exclude is a top-level field in each patch entry that defines patterns to exclude from matching. It allows more precise control over function where patches should not be applied.

| Field | Description | Example |
|-------|-------------|---------|
| `exclude` | List of function name patterns to exclude from matching | `- "^func_*"` |

## Argument Specification

Arguments passed to patch functions can come from different sources and are specified using a structured format that supports operands, function arguments, variables, and constants.

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
| `source` | Where the argument comes from | Yes | `"operand"`, `"argument"`, `"variable"`, `"constant"` |
| `index` | Index for operands/arguments | When `source` is `"operand"` or `"argument"` | `0`, `1`, `2` |
| `symbol` | Symbol name for variables | When `source` is `"variable"` | `"key_size"`, `"current_time"` |
| `value` | Literal value for constants | When `source` is `"constant"` | `"1024"`, `"0x1000"` |

### Argument Source Types

| Source | Description | Required Fields | Use Case |
|--------|-------------|-----------------|----------|
| `operand` | Operation operand by position | `index` | Access operands of matched operations |
| `argument` | Function call argument by position | `index` | Access arguments of matched function calls |
| `variable` | Local variable or symbol by name | `symbol` | Access local variables in scope |
| `constant` | Literal constant value | `value` | Pass fixed values to patch functions |
| `return_value` | Return value of function or operation | None | Access return value (for ApplyAfter mode) |

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

#### Return Value Handling
```yaml
# Access function return value (ApplyAfter mode)
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

### Example 1: Function Call Validation with Structured Arguments

```yaml
arch: "ARM:LE:32"
patches:
  - name: "validate_memcpy_calls"
    match:
      symbol: "memcpy"
      kind: "function"
      argument_matches:
        - index: 0
          name: "dest"
          type: "!cir.ptr<!cir.void>"
        - index: 2
          name: "size"
          type: "!cir.int<u, 32>"
    patch:
      mode: "ApplyBefore"
      patch_file: "patches/bounds_check.c"
      patch_function: "validate_memcpy"
      arguments:
        - name: "dest_ptr"
          source: "argument"
          index: 0
        - name: "copy_size"
          source: "argument"
          index: 2
        - name: "max_allowed"
          source: "constant"
          value: "4096"
        - name: "buffer_limit"
          source: "variable"
          symbol: "max_buffer_size"
    exclude:
      - "test_*"
```

## Operation-Based Matching Examples

### Example 1: Monitor Sensitive Load Operations

```yaml
- name: "sensitive_loads"
  match:
    operation: "cir.load"
    function_context:
      - name: "/.*secure.*/"  # Functions containing "secure"
      - name: "authenticate"  # Exact function name
    symbol_matches:
      - name: "/.*password.*/"  # Variables containing "password"
        type: "!cir.ptr<!cir.int<u, 32>>"
```

### Example 2: Match Store Operations with Specific Operands

```yaml
- name: "validate_stores"
  match:
    operation: "cir.store"
    function_context:
      - name: "/.*critical.*/"
    operand_matches:
      - index: 0  # The value being stored (first operand)
        name: "user_input"
        type: "!cir.ptr<!cir.char>"
      - index: 1  # The address being stored to (second operand)
        name: "buffer"
        type: "!cir.ptr<!cir.array<!cir.char x 256>>"
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

The specification supports three patching modes:

- `ApplyBefore`: Apply patch before the matched function or operation
- `ApplyAfter`: Apply patch after the matched function
- `Replace`: Completely replace the matched function

### ApplyBefore Mode

In `ApplyBefore` mode, the patch is applied before the matched function or operation executes.
```yaml
patch:
  mode: "ApplyBefore"
  patch_file: "path/to/patch.c"
  target_function: "patch::before::function_name"
  arguments:
    - "arg1"
    - "arg2"
```

### ApplyAfter Mode

In `ApplyAfter` mode, the patch is applied after the matched function or operation completes.

```yaml
patch:
  mode: "ApplyAfter"
  patch_file: "path/to/patch.c"
  target_function: "patch::after::function_name"
  arguments:
    - "return_value"
```

### Replace Mode

In `Replace` mode, the matched function or operation is completely replaced by the patch function. The original code is not executed.

```yaml
patch:
  mode: "Replace"
  patch_file: "path/to/patch.c"
  target_function: "patch::replace::function_name"
  arguments:
    - "arg1"
    - "arg2"
```
