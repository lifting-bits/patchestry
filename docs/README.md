# Patchestry: Multi-Layered Binary Lifting and Patching Framework

Patchestry aims to make the same impact to binary patching as compilers and high
level languages did to early software development. Its main goal is to enable
developers without extensive knowledge of the deployment platform of the binary
to patch the binary. To do this, the developer has to be confident that what
they're patching is functionally equivalent to what is deployed and also that
the patch they write will integrate into the deployed binary without issue.

Patchestry leverages MLIR as the foundational technology instead of LLVM IR.
MLIR is an emerging compiler development technology allowing for the
specification, transformation, and mixing of IR dialects. The MLIR approach has
significant industry momentum, and has been adopted by companies such as Google
(in TensorFlow) and Meta (ClangIR). With MLIR, the decompilation process could
be stratified into a Tower of IRs (IR dialects). Each IR represents the same
program, but at a different level of abstraction.

MLIR brings a notable advantage by enabling the creation of representations to
streamline communication between diverse state-of-the-art tools. For instance,
ClangIR provides an MLIR dialect that closely mirrors the Clang AST, preserving
high-level C/C++ semantics and enabling precise source-level analysis and
transformation. From there, an LLVM IR dialect can be employed to compile back
to the executable, and MLIR can support LLVM-based contract validation through a
symbolic executor such as KLEE. Moreover, MLIR provides flexibility to
devise our own dialects for representing contracts in specialized logic, such as
SMT. Our compiler stacks empower us to compile C into any of the previously mentioned
representations, promoting seamless interconnection between them.

## Technical Rationale

Our experience on AMP, as well as our performance on other DARPA binary
analysis programs (PACE, CFAR, LOGAN, CGC, Cyber Fast Track), have led us to
four guiding principles that we believe patching solutions for legacy software
must follow in order to be successful.

1. __Fully automated approaches are doomed to failure.__ The process of
decompilation is an inherently intractable problem. However, developers are
often capable of distinguishing between decompilation outcomes deemed 'good' or
'bad', but encoding that kind of heuristic logic into a system invariably yields
unpredictability and unsoundness. Hence, we assert that the involvement of
semi-skilled or skilled human developers is essential in the process. The
best-case scenario is that a developer can use an existing source code patch as
a guide. Given this patch, they can locate the corresponding vulnerable machine
code within a binary using binary function symbol names. The worst-case scenario
involves the ad hoc application of tools (e.g. BinDiff, BSim) and reverse
engineering skills to an opaque binary blob that is without symbols or debugging
information.

2. __Developers must be able to leverage pre-existing software development
experience__ and not have to concern themselves with low level details. They
should be able to operate as if the original source code and build
process/environments were available, and not be expected to have expert
knowledge of every machine code language that may be encountered.

3. __From-scratch development efforts do not scale.__ As much as possible,
pre-existing tooling that already handles the inherent scalability challenges in
(de)compiling code for such a wide variety of platforms should be leveraged. For
example the Ghidra decompiler can decompile over 100 machine code languages to
C, and the Clang compiler can generate machine code for over 20 machine code
languages. Rolling new solutions from scratch is impractical.

4. __There is no one-size-fits-all way of representing code.__ A "complete"
solution to machine code decompilation only exists at the end of a long tail of
special cases. Patchestry aims to provide decompilation to a familiar, C-like
language. Patchestry will not, however, decompile to C or a specific ISO dialect
thereof because some machine code constructs have no equivalents in C, while
others are only loosely equivalent given non-conforming dialect extensions.

## Project Goals

Patchestry accomplishes its goals by integrating various innovative concepts
guided by the four principles described in [technical rationale](#technical-rationale):

### Unified Tooling Integration

Guided by the third principle—recognizing the limitations of from-scratch
development efforts—Patchestry seamlessly integrates existing tooling for
decompilation and recompilation in the binary patching process. Patchestry
advocates a unified tooling integration approach using an MLIR Tower of
Intermediate Representations (IRs) as a mediator between tools. This strategy
enables the incorporation of cutting-edge (de)compiler tools into a cohesive
system, allowing the utilization of specialized tools for each task and ensuring
effectiveness and optimal outcomes across all desired functionalities.

### Incremental Decompilation

Patchestry's innovative approach involves leveraging multiple program
representations simultaneously across various layers of the Tower of IRs. While
state-of-the-art decompilers already offer diverse representations, what sets
the Tower of IRs apart is its capability to create custom user-defined
abstractions (layers) while preserving relationships between these layers. This
modular approach facilitates seamless incremental decompilation and
recompilation processes. This is crucial for effortlessly devising specific
abstractions tailored to unique platforms.

### Unifying Representations for Contracts, Patches, and Software

The Tower of IRs also aligns with the fourth guiding principle: There is no
one-size-fits-all way of representing code. Maintaining multiple
representations simultaneously in the Tower of IRs allows us to establish
meaningful relationships between them and innovate in how we connect tools and
conduct analyses. Additionally, this approach allows us to consolidate all
necessary components for patching within the same representation: patch
description, contract description and the software. This unified strategy
streamlines tooling for analysis and facilitates the recompilation of patched
software, resulting in a single artifact that can undergo desired formal
analyses, such as LLVM-based analysis.

### Declarative Patching and Contracts Description

To address our second guiding principle, which emphasizes the importance of
allowing developers to leverage their existing software development experience,
we mandate that all interactions with patching occur in a language commonly
understood by developers. Patches are written as C functions, and the
meta-programming layer for describing _where_ and _how_ patches and contracts
are applied is specified declaratively in YAML. This separates patch logic
(familiar C code) from patch orchestration (structured YAML configuration),
keeping both accessible to developers without requiring expertise in compiler
internals or custom DSLs. Contracts are declarative predicates expressed as
YAML and attached as MLIR attributes for static verification — runtime
validators that used to be called "runtime contracts" live under `patches:`
now, since mechanically they were just patches.

## Why This Approach

### Why Not Edit Ghidra's Output Directly?

Patchestry's workflow does not allow the developer to modify Ghidra's
decompilation output and then re-compile it. There are two reasons:

1. Ghidra's decompilation is not guaranteed to be syntactically correct or
   compilable. The effort to fix it increases with the complexity of the target
   function(s).
2. Ghidra's heuristic decompilation pipeline has been proven to be unfaithful
   with respect to the execution semantics of the machine code. This could
   result in a developer inadvertently introducing new vulnerabilities during the
   patching process.

Despite this, Ghidra's decompilation is good enough to be a productivity
multiplier for developers trying to locate functions that need patching.

### Why Clang AST?

Patchestry lifts Ghidra's P-Code representation into a Clang AST. This choice
is driven by pragmatic considerations:

- __Recompilation for free.__ The Clang compiler can already target over 20
  machine code languages. By producing a valid Clang AST, Patchestry gets
  recompilation to any Clang-supported target architecture without building a
  custom compiler backend.
- __Function-level granularity.__ Functions are the smallest compilable unit of
  code in compilers like Clang. Function granularity patches also enable
  Patchestry to leverage stronger ABI guarantees: it is only at the entry and
  exit points of a compiled function that higher-level, human-readable types can
  be reliably mapped to low level machine locations (registers, memory).
- __Familiar output.__ The decompiled C output looks approximately similar
  regardless of the platform/architecture, improving developer productivity.
  Developers can read and modify the output using standard C knowledge.
- __Integration with MLIR.__ Clang's CIR (ClangIR) dialect provides a bridge
  into the MLIR ecosystem, enabling instrumentation, patching, and contract
  verification using MLIR-based passes before lowering to LLVM IR.

### Why MLIR?

Patchestry leverages MLIR as the foundational technology for its intermediate
representations instead of LLVM IR directly. MLIR allows for the specification,
transformation, and mixing of IR dialects. With MLIR, the decompilation and
patching process is stratified into a Tower of IRs (IR dialects), where each IR
represents the same program at a different level of abstraction. This enables:

- Creation of an MLIR dialect specifically for P-Code to optimize integration
  with the Ghidra decompiler.
- Use of an LLVM IR dialect to compile back to the executable.
- LLVM-based contract validation through symbolic executors such as KLEE or
  SeaHorn.
- Custom dialects for representing contracts in specialized logic (e.g., SMT).

## Decompilation Pipeline

Patchestry's decompilation pipeline converts binary functions into editable,
recompilable C code through the following stages:

```
Binary --> Ghidra --> P-Code (JSON) --> Clang AST --> C Output
                                           |
                                      CIR (MLIR) --> Instrumentation --> LLVM IR --> Machine Code
```

1. __Ghidra P-Code serialization.__ A Ghidra plugin serializes the decompiled
   P-Code representation of target function(s) to JSON format. This
   serialization captures types, control flow, operations, and variable
   information from Ghidra's analysis database.

2. __Lifting to Clang AST.__ The `patchir-decomp` tool reads the serialized
   P-Code JSON and constructs a Clang AST. This involves type reconstruction,
   control flow structuring (recovering if/else, loops, and switch statements
   from the flat P-Code graph), and mapping P-Code operations to C statements.

3. __C output.__ The Clang AST is emitted as human-readable C code that the
   developer can inspect and edit.

4. __CIR and MLIR lowering.__ The Clang AST is lowered through ClangIR (CIR)
   into the MLIR Tower of IRs. At this level, patches and contracts are applied
   via the instrumentation engine. The result is lowered to LLVM IR for
   recompilation.

## Developer Workflow

Patchestry's technical approach enables the following workflow:

1. A developer is tasked with patching a vulnerability in a program binary
   running on a device.

2. The developer loads the binary into Ghidra and locates the function(s) to
   patch using Ghidra's features, plugins, symbol names, or tools such as
   BinDiff or BSim. Previous binary analysis expertise is not required.

3. The Patchestry Ghidra plugin serializes the target function(s) to P-Code
   JSON. The `patchir-decomp` tool then produces an editable C decompilation
   that is sound and precise with respect to the available information in
   Ghidra's analysis database.

4. The developer edits the decompiled function(s) to patch the vulnerability.
   Alternatively, the developer defines patches declaratively using YAML
   specifications and the meta-patching library (see
   [Patching Interface](#patching-interface)).

5. Contracts are verified. Patchestry generates output compatible with LLVM-based
   analysis tools such as KLEE or SeaHorn to ensure that the patched code
   satisfies developer-defined contracts (see
   [Contracts Interface](#contracts-interface)).

6. Patchestry recompiles the patched function(s) through the MLIR pipeline to
   LLVM IR, then to machine code. The resulting binary patch is packaged for
   insertion into the original binary using a tool such as Patcherex or OFRAK.

7. The developer loads the patched binary onto the device.

## Patching Interface

Patchestry provides a declarative YAML-based interface for specifying patches
and their application. This separates _what_ the patch does (the C code) from
_where_ and _how_ it is applied (the meta-patch configuration).

### Patch Specification

Patches are defined in a YAML library file. Each patch references a C source
file containing the patch implementation:

```yaml
apiVersion: patchestry.io/v1
metadata:
  name: usb-security-patches
  version: "1.0.0"

patches:
  - name: usb_endpoint_write_validation
    id: "USB-PATCH-001"
    description: "Validate USB endpoint write operations"
    category: usb_security
    severity: high
    code_file: "patches/patch_usbd_ep_write_packet.c"
    function_name: "patch::before::usbd_ep_write_packet"
    parameters:
      - name: usb_device
        type: "usb_device_t*"
      - name: buffer
        type: "const void*"
```

### Meta-Patch Configuration

Meta-patches define _where_ patches are applied using match rules and _how_
they are applied using action modes. This enables automated, declarative
patching across the codebase:

```yaml
meta_patches:
  - name: usb_security_meta_patches
    description: "Meta patches for USB security"
    optimization:
      - "inline-patches"
    patch_actions:
      - id: "USB-PATCH-001"
        description: "Pre-validation security check"
        match:
          - name: "usbd_ep_write_packet"
            kind: "operation"
            function_context:
              - name: "bl_usb__send_message"
        action:
          - mode: "apply_before"
            patch_id: "USB-PATCH-001"
            arguments:
              - name: "operand_0"
                source: "operand"
                index: 0
```

The instrumentation engine supports three action modes:

- __`apply_before`__: Insert the patch function call before the matched
  operation.
- __`apply_after`__: Insert the patch function call after the matched operation.
- __`replace`__: Replace the matched operation entirely with the patch function.

Arguments to the patch function can be sourced from:

- __`operand`__: An operand of the matched call or operation by index.
- __`variable`__: A local variable by name.
- __`symbol`__: A global symbol (variable or function) by name.
- __`constant`__: A literal constant value.
- __`return_value`__: The return value of the matched call.

### Configuration File

A top-level YAML configuration file ties together the target binary, patch
libraries, contract libraries, meta-patches, and meta-contracts, along with
an execution order:

```yaml
apiVersion: patchestry.io/v1
metadata:
  name: "usb-security-deployment"
target:
  binary: "firmware.bin"
  arch: "ARM:LE:32"

libraries:
  - "patches/usb_security_patches.yaml"
  - "contracts/usb_security_contracts.yaml"

meta_patches:
  - name: usb_security
    # ... patch actions ...

meta_contracts:
  - name: usb_security
    # ... contract actions ...
```

## Contracts Interface

Patchestry provides contracts as a mechanism for specifying and verifying
correctness properties of patched code. Unlike patches, contracts do not alter
program state. Contracts are **static-only**: they are declarative predicates
(preconditions / postconditions) that the instrumentation pass attaches to the
matched op as MLIR attributes. They emit no runtime code and produce no runtime
overhead; downstream verifiers such as KLEE or SeaHorn consume the attributes
to check that the predicates hold.

The "runtime contract" concept that previously lived here — a C function called
at the matched site to validate a condition — was mechanically the same as a
patch, so those cases have been merged into `patches:`. Write the runtime check
as a patch whose body asserts (or traps) on failure; attach a `contracts:` entry
alongside it to encode the same property for the verifier.

### Contract Specification

Contracts are declarative, static-only predicates expressed as YAML and
attached to the matched op as MLIR attributes for the verifier to consume.
If you need runtime validation (a C function called at the match site),
write it as a patch with `apply_before` / `apply_after` /
`apply_at_entrypoint`.

```yaml
contracts:
  - name: "nonnull_pointer"
    severity: critical
    preconditions:
      - id: "pre-001"
        description: "Pointer argument must not be null"
        pred:
          kind: nonnull
          target: arg0
```

Static contract predicates support:

- __`nonnull`__: Assert that a target is not null.
- __`relation`__: Assert a relational constraint (e.g., `arg0 <= value`).
- __`alignment`__: Assert pointer alignment.
- __`range`__: Assert that a value falls within a min/max range.
- __`expr`__: Assert an arbitrary expression.

### Meta-Contract Configuration

Meta-contracts define where contracts are applied, similar to meta-patches.
Contracts support two modes; both attach the same `contract.static` attribute
and differ only in which op the attribute lands on:

- __`apply_before`__: Attach the predicate to the matched op (or the op
  immediately preceding the match site).
- __`apply_after`__: Attach the predicate to the op immediately following the
  match site.

`apply_at_entrypoint`, `replace`, and `erase` are patch-only. For a check that
needs to run at the caller's entry block, write a patch with
`mode: apply_at_entrypoint`.

## Architecture

The Patchestry design places a strong emphasis on modularity and seamless
developer interaction. The developer plays a key role, providing the binary
pieces to be patched, a patch description, and instructions on how to apply
these patches using the meta-programming framework (meta-patches). Contracts are
similarly specified and applied by instrumentation using the same meta-language.

A significant architectural innovation is the MLIR Tower of IRs, which serves as
the connecting element. This tower facilitates the association of
representations between decompiled programs, such as from P-Code and compilable
and structured representations like LLVM IR. The tower's modularity allows for
the specification of any DSL for the decompiled program, with the only
requirement being the translation of this DSL to a layer of the tower. In our
case, Ghidra's P-Code serves as a suitable starting point layer. This modular
design allows new decompilers to be integrated into Patchestry in the future
while preserving the rest of the architecture.

Utilizing the same representation (MLIR dialects) for both the decompiled binary
and the compiled patched version facilitates seamless instrumentation and
inlining of patches, ultimately producing a patched MLIR (Tower of IRs). The
tower's various abstraction layers enable precise specification of points of
interest, surpassing the limitations of a single representation.

In the verification phase, Patchestry is designed to accommodate various
verification methods. The Tower can produce a customized representation for the
analysis, but it is advisable to stick to the same representation as the
compilation (such as LLVM IR) to prevent errors during translation. Slicing the
codebase into independent parts influenced by the patch makes LLVM-based static
analysis tractable. Since most patches are local and influence only a small part
of the program, dependency analysis can isolate the part of the program that
needs to be verified.

## Example: CVE-2021-22156 Patching

An example of patching the CVE-2021-22156 vulnerability, addressing an integer
overflow within the `calloc()` function of the C standard library. This
vulnerability affects versions of the BlackBerry QNX Software Development
Platform (SDP) up to version 6.5.0SP1, QNX OS for Medical up to version 1.1,
and QNX OS for Safety up to version 1.0.1.

Vulnerable code in which `calloc()` may allocate a zeroed buffer of
insufficient size:

```c
size_t num_elements = get_num_elements();
long *buffer = (long *)calloc(num_elements, sizeof(long));
if (buffer == NULL) {
    /* Handle error condition */
}
```

The desired result after applying the patch:

```c
size_t num_elements = get_num_elements();

/* Patch start */
if (num_elements > SIZE_MAX/sizeof(long)) {
    /* Handle error condition */
}
/* Patch end */

long *buffer = (long *)calloc(num_elements, sizeof(long));
if (buffer == NULL) {
    /* Handle error condition */
    return;
}
```

The patch is written as a C function:

```c
// patches/cve_2021_22156_patch.c
void patch_calloc_overflow(size_t num_elements) {
    if (num_elements > SIZE_MAX/sizeof(long)) {
        /* Handle error condition */
    }
}
```

The meta-patch configuration in YAML describes where and how to apply it:

```yaml
# Patch library
patches:
  - name: calloc_overflow_check
    id: "CVE-2021-22156"
    description: "Integer overflow check before calloc"
    severity: high
    code_file: "patches/cve_2021_22156_patch.c"
    function_name: "patch_calloc_overflow"
    parameters:
      - name: num_elements
        type: "size_t"

# Meta-patch: apply before every call to calloc
meta_patches:
  - name: cve_2021_22156_meta
    patch_actions:
      - id: "CVE-2021-22156-ACTION"
        match:
          - name: "calloc"
            kind: "operation"
        action:
          - mode: "apply_before"
            patch_id: "CVE-2021-22156"
            arguments:
              - name: "num_elements"
                source: "operand"
                index: 0
```

To express the same property declaratively for the verifier, attach a
static contract. The predicate is carried as an MLIR attribute and
produces no runtime overhead. The contract definition lives in a library
file that the deployment spec references via `libraries:`, and the
meta-contract in the deployment spec dispatches it at the match site:

```yaml
# contracts/cve_2021_22156_library.yaml
contracts:
  - name: calloc_bounds_contract
    severity: high
    preconditions:
      - id: "num_elements_bounded"
        description: "calloc count must not overflow a 32-bit size_t"
        pred:
          kind: range
          target: arg0
          # min/max are string-encoded integer literals parsed with
          # std::stoll — C macros and expressions like
          # "SIZE_MAX/sizeof(long)" are not supported. Compute the
          # bound in the spec author's environment and inline the
          # integer.
          range:
            min: "0"
            max: "1073741823"   # (2^32 - 1) / 4, i.e. SIZE_MAX/sizeof(long)
                                # on a 32-bit target with 4-byte long
```

```yaml
# deployment.yaml
libraries:
  - "contracts/cve_2021_22156_library.yaml"

meta_contracts:
  - name: cve_2021_22156_contract_meta
    contract_actions:
      - id: "CVE-2021-22156-CONTRACT"
        match:
          - name: "calloc"
            kind: "function"
            function_context:
              - name: "*"
        action:
          - mode: "apply_before"
            contract_id: "calloc_bounds_contract"
```

> Top-level `contracts:` and `meta_contracts:` are mutually exclusive
> within a single deployment file (see `ConfigurationFile.hpp`), so
> the flat library definition belongs in its own `libraries:` file
> rather than alongside the `meta_contracts:` dispatch above.

If you want a runtime check that traps on violation, write it as a
patch (the old "runtime contract" shape) — same `code_file` /
`function_name` fields, just under `patches:` instead.

A contract is similar to a regression test or a behavioral assertion that an
analysis tool like KLEE or SeaHorn would check.
