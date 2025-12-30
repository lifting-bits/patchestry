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
apiVersion: patchestry.io/v1             # API version

metadata:                                # Deployment metadata
  name: "deployment-name"
  description: "Deployment description"
  version: "1.0.0"
  author: "Author Name"
  created: "YYYY-MM-DD"
  organization: "organization-name"

target:                                  # Target binary configuration
  binary: "target_binary.bin"
  arch: "ARCHITECTURE:ENDIANNESS:BITWIDTH:VARIANT"

libraries:                               # External patch and contract libraries
  patches: "path/to/patches.yaml"
  contracts: "path/to/contracts.yaml"

execution_order:                         # Order of patch/contract execution
  - "meta_patches::meta_patch_name"
  - "meta_contracts::meta_contract_name"

meta_patches:                            # Meta-patch configurations
  - name: ...
    description: "..."
    optimization:                        # Optimization settings
      - "inline-patches"
      - "inline-contracts"
    patch_actions:                       # Individual patch actions
      - id: "PATCH-001"
        description: "..."
        match:                           # Match criteria
          - symbol: "..."
            kind: "..."
            # Additional match criteria...
        action:                          # Patch actions
          - mode: "..."
            patch_id: "..."
            description: "..."
            arguments:                   # Patch arguments
              - name: "..."
                source: "..."
                index: "0"
                is_reference=true

meta_contracts:                          # Meta-contract configurations
  - name: ...
    description: "..."
    contract_actions:                    # Individual contract actions
      - name: "..."
        id: "CONTRACT-001"
        description: "..."
        match:                          # Contract match criteria
          - name: "..."
            kind: "..."
          # Additional match criteria...
        action:                         # Contract actions
          - mode: "..."
            contract_id: "..."
            description: "..."
            arguments:                  # Contract arguments
              - name: "..."
                source: "..."
                index: 0

```

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
| `execution_order` | Order of patch/contract group execution | `- "meta_patches::group_name"` |
| `meta_patches` | Meta-patch group configurations | List of patch groups |
| `meta_contracts` | Meta-contract group configurations | List of contract groups |

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

| Field | Description | Example |
|-------|-------------|---------|
| `patches` | Path to external patch library YAML file | `"patches/security_patches.yaml"` |
| `contracts` | Path to external contract library YAML file | `"contracts/security_contracts.yaml"` |

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
| `match.symbol` | Symbol name to match (for function-based matching) | `"read"` |
| `match.kind` | Type of match target (`function` or `operation`) | `"function"` |
| `match.operation` | Operation type to match (for operation-based matching) | `"cir.load"`, `"cir.store"`, `"cir.binop"` |
| `match.function_context` | Functions where operation matches should be applied | `name: "/.*secure.*/"` |
| `match.variable_matches` | Variables used in the operation (for function-based matching) | `name: "/.*password.*/"` |
| `match.argument_matches` | Function arguments or Operands to match (for function-based matching) | See below |
| `match.symbol_matches` | Variables used in the operation (for operation-based matching) | `name: "/.*password.*/"` |
| `match.operand_matches` | Function arguments or Operands to match (for operation-based matching) | See below |

#### Operation-Based Matching

Operation-based matching is supported only for patches, meaning it is not supported for contract matching and insertion. For operation-based matching for patches, the following additional fields are available:

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

#### Symbol Match Fields  (Operation-Based Matching)

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Symbol name pattern (supports regex with `/pattern/`) | `"/.*password.*/`, `"secret_key"` |
| `type` | Symbol type pattern (optional) | `"int32*"` |

#### Operand Match Fields (Operation-Based Matching)

| Field | Description | Example |
|-------|-------------|---------|
| `index` | Position of the operand (0-based) | `0` (first operand), `1` (second operand) |
| `name` | Name of the operand variable | `"addr"`, `"/.*buffer.*/` |
| `type` | Type of the operand | `"int32*"` |

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
| `type` | Variable type pattern (optional) | `"struct struct_anon_struct_4_1_58265f66*"` |

### Action Fields

| Field | Description | Example |
|-------|-------------|---------|
| `mode` | Patching mode to be applied (`apply_before`, `apply_after`, `replace`) | `"apply_before"` |
| `patch_id` | Reference to the patch implementation | `"USB-PATCH-001"` |
| `description` | Description of the action being applied | `"Pre-validation security check"` |
| `arguments` | List of arguments to pass to patch function | See [Argument Specification](#argument-specification) |

### Optimization Flags

The `optimization` field accepts a list of optimization settings:

| Flag | Description | Effect |
|------|-------------|--------|
| `"inline-patches"` | Inline patch function calls | Reduces function call overhead |
| `"inline-contracts"` | Inline contract function calls | Reduces contract validation overhead |
| `"debug-info"` | Preserve debug information | Maintains debugging symbols |
| `"size-optimize"` | Optimize for binary size | Reduces final binary size |

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
  patches: "patches/usb_security_patches.yaml"
  contracts: "contracts/usb_security_contracts.yaml"

execution_order:
  - "meta_patches::usb_security_meta_patches"
  - "meta_contracts::usb_security_meta_contracts"

meta_patches:
  - name: usb_security_meta_patches
    id: "usb_security_meta_patches"
    description: "Meta patches for USB security"
    optimization:
      - "inline-patches"
      - "inline-contracts"

    patch_actions:
      - id: "USB-PATCH-001"
        description: "Patch to add USB security validation"
        match:
          - symbol: "usbd_ep_write_packet"
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
    id: "usb_security_meta_contracts"
    description: "Contracts for USB security validation"
    contract_actions:
    - id: "USB-CONTRACTS"
      description: "Assert allocation size is suitable"
      match:
        - name: "bl_usb__send_message"
          kind: "function"
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
    operation: "cir.load"
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
    operation: "cir.store"
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

The specification supports three patching modes:

- `apply_before`: Apply patch before the matched function or operation
- `apply_after`: Apply patch after the matched function
- `replace`: Completely replace the matched function

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

In `apply_after` mode, the patch is applied after the matched function or operation completes.

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

In `replace` mode, the matched function or operation is completely replaced by the patch function. The original code is not executed.

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
| `relation` | Compare target against a value | `kind`, `target` (arg, return_value, or symbol), `relation`, `value` | `symbol` | Bounds checking, comparisons |
| `alignment` | Verify memory alignment | `kind`, `target`, `align` | `symbol` | Memory alignment requirements |
| `expr` | Free-form expression | `kind`, `expr` | `target`, `symbol` | Complex conditions |
| `range` | Verify value is within range | `kind`, `target` (arg, return_value, or symbol), `range` (with `min` and `max`) | `symbol` | Input validation, bounds |

#### Quick Reference: Field Requirements by Predicate Kind

| Predicate Kind | `kind` | `target` | `relation` | `value` | `align` | `expr` | `range` | `symbol` |
|----------------|--------|----------|------------|---------|---------|--------|---------|----------|
| **relation**   | ✓      | ✓        | ✓          | ✓       | ✗       | ✗      | ✗       | Optional |
| **alignment**  | ✓      | ✓        | ✗          | ✗       | ✓       | ✗      | ✗       | Optional |
| **expr**       | ✓      | Optional | ✗          | ✗       | ✗       | ✓      | ✗       | ✗        |
| **range**      | ✓      | ✓        | ✗          | ✗       | ✗       | ✗      | ✓       | Optional |

**Legend**: ✓ = Required, ✗ = Not allowed/Not used, Optional = May be included

**Important Notes:**
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

#### Example 1: Range Validation (range)

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

#### Example 2: Comparison Relation (relation)

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

#### Example 3: Memory Alignment (alignment)

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

#### Example 4: Complex Expression (expr)

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

#### Example 5: Return Value Check (relation)

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
    id: "USB-CONTRACT-STATIC-001"
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
- **relation**: `kind` + `target` + `relation` + `value` (`arg<N>`, `return_value`, or `symbol`)
- **alignment**: `kind` + `target` + `align` (`arg<N>`, `return_value`, or `symbol`)
- **expr**: `kind` + `expr`
- **range**: `kind` + `target` + `range` (with `min`/`max`) (`arg<N>`, `return_value`, or `symbol`)

### Runtime Contract Implementation

For runtime contracts, specify the implementation details:

```yaml
contracts:
  - name: "usb_endpoint_write_contract"
    id: "USB-CONTRACT-001"
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
