# Firmware Patch Specification

This document outlines the specification for patching firmware binaries using a declarative YAML format. The specification applies patches at the MLIR (Multi-Level Intermediate Representation) level, specifically using Clang IR MLIR representation, allowing for architecture-independent binary modifications.

## Overview

The patch specification is designed to apply patches at specific points in the decompiled MLIR representation:
- Before or after function calls
- At specific operations like memory loads, stores, or other operations
- Replace entire functions or operations
- Based on variable matching and symbolization

By working at the MLIR level, patches can be written once and applied across multiple architectures, as the MLIR abstraction handles the architecture-specific details.

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
| `match.symbol` | Symbol name to match | `"read"` |
| `match.kind` | Type of match target (`function`, `variable`, or `operation`) | `"function"` |
| `match.operation` | Operation type to match (when kind is "operation") | `"cir.load", "cir.for"` |
| `match.variable_matches` | Variables used in the operation | `name: "puVar8"` |
| `match.argument_matches` | List of function arguments to match | See below |

#### Argument Match Fields

| Field | Description | Example |
|-------|-------------|---------|
| `index` | Position of the argument (0-based) | `1` |
| `name` | Name of the argument | `"buff"` |
| `type` | Type of the argument | `"void*"` |

### Patch Fields

| Field | Description | Example |
|-------|-------------|---------|
| `patch.mode` | patching mode to be applied (`ApplyBefore`, `ApplyAfter`, `Replace`) | `"ApplyBefore"` |
| `patch.patch_file` | Path to the C file containing patch code | `"path/to/patch.c"` |
| `patch.patch_function` | Function in patch file to call | `"patch::before::function_name"` |
| `patch.arguments` | List of arguments to pass to patch function | `["arg1", "arg2"]` |

### Exclude Fields

Exclude is a top-level field in each patch entry that defines patterns to exclude from matching. It allows more precise control over function where patches should not be applied.

| Field | Description | Example |
|-------|-------------|---------|
| `exclude` | List of function name patterns to exclude from matching | `- "^func_*"` |

## Patch Modes

The specification supports three patching modes:

- `ApplyBefore`: Apply patch before the matched function or operation
- `ApplyAfter`: Apply patch after the matched function or operation
- `Replace`: Completely replace the matched function or operation

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
