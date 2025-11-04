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

## Deployment Architecture

The meta-patch architecture allows for:

1. **Modular Organization**: Group related patches into logical units
2. **External Libraries**: Reference shared patch and contract libraries
3. **Execution Ordering**: Control the order of patch application
4. **Optimization Control**: Fine-tune performance characteristics
5. **Exclusion Patterns**: Exclude specific functions from patching
