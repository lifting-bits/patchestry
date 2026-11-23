---
name: klee-models
description: This skill generates a KLEE model library for an LLVM IR module under test. It extracts the functions the IR declares and does not define, classifies each as libc, compiler runtime, HAL or RTOS API (CMSIS, STM32 HAL, libopencm3, FreeRTOS and similar), function internal to the binary, or import, writes a C model for each from known semantics, vendor conventions or a summary of the function's body read from the original binary, and checks that the library compiles and links with patchir-klee-verifier --model-library. It should be used when the user asks to "generate KLEE models", "write a model library", "model the external functions", "model HAL functions for KLEE", "stub the externals for KLEE", "create models for this IR", "what does this IR need modelled", or passes a .ll or .bc file and wants it run under KLEE or patchir-klee-verifier without unconstrained stubs.
---

# KLEE model library for an IR under test

`patchir-klee-verifier` gives every function the IR calls and does not define
a stub: a symbolic return value and no effect on memory. That is the weakest
model. It is wrong for a function that fills a buffer, sets an out-parameter
or never returns, and too loose for one that only returns a few status
codes. This skill writes better models, as C, in the shape of the
repository's own (`lib/patchestry/klee/models/`), and proves they link.

## Inputs

| Input | Needed | Used for |
| --- | --- | --- |
| LLVM IR, `.ll` or `.bc` | yes | the externals and their signatures |
| the function under test | yes, ask if not given and the module defines more than one | the link check |
| the original binary (ELF) | when available | telling internal functions from imports, addresses, reading bodies |
| a Ghidra P-Code export (`function.json`) | when available | the same, for stripped images |
| vendor headers or firmware source | when available | HAL conventions |

## Outputs

Written next to the input IR unless the user names another place:

- `<stem>_models.c`: one model per external that needs one.
- `<stem>_models.bc`: the compiled library, for `--model-library`.
- `<stem>_models.md`: the report (step 7).

Do not add the file to `lib/patchestry/klee/models/` or its `CMakeLists.txt`
unless the user asks for that.

## Setup

Paths are relative to the repository root. `SKILL` stands for
`.claude/skills/klee-models`.

```bash
export PATCHESTRY_LLVM_PREFIX=<directory with clang, llvm-dis, llvm-link>
```

The fork's tools are where `builds/default/CMakeCache.txt` points
(`grep LLVM_DIR builds/default/CMakeCache.txt`, then the `bin` directory of
that installation). The scripts also take `--llvm-bin DIR` and fall back to
`PATH`. The verifier is
`builds/default/tools/patchir-klee-verifier/<Debug|Release>/patchir-klee-verifier`;
build it with `cmake --build --preset debug -j` if it is missing.

## Workflow

### 1. Extract the externals

```bash
python3 SKILL/scripts/list_externals.py MODULE.ll \
    --models builds/default/lib/patchestry/klee/models/libc_models.bc \
    --models builds/default/lib/patchestry/klee/models/usb_hal_models.bc \
    --output externals.json
```

Each `--models` file marks what it already defines (`modelled_in`). Read
`summary`: `to_model` is the work, and `target_intrinsics` lists `llvm.arm.*`
and similar calls, which no model library can fix (the verifier gives them a
symbolic result). Mention them in the report.

Per external the JSON has the IR signature, `variadic`, `noreturn`,
`call_count`, `callers`, and `constant_arguments`: for each argument
position, the constants passed when every call passes one, otherwise null.
A constant is an integer, `"null"`, `"@global"`, or `"address:0x..."` for a
pointer to a fixed address of the original image. The module does not hold
that memory, so a model must not read through it (rule 6 in
`references/model-patterns.md`).

### 2. Classify

```bash
python3 SKILL/scripts/classify.py externals.json \
    [--binary IMAGE.elf] [--pcode-json function.json]... --output classified.json
```

This adds `class`, `family`, `pattern`, `address` and `source`. Without a
binary or an export, functions the name tables do not know are `unknown`.
Ask the user for the binary when more than a few are, since each one would
otherwise get a signature-only model.

### 3. Decide the source of each model

Work through the externals by `source`, strongest evidence first:

| `source` | What to do |
| --- | --- |
| `existing-model` | nothing. The library named in `modelled_in` has it. Link that library in step 5 |
| `klee` | nothing. KLEE provides the allocator family, and `check_models.py` rejects a model for it |
| `known-semantics` | libc and compiler runtime. Write the model from the C standard or POSIX with the shape in `pattern`. Write real loops for `pure` functions whose result the caller branches or indexes on |
| `convention` | HAL and RTOS. Open `references/hal-families.md`. If the family is under "Verified", use the documented return. If not, find the vendor header in the firmware source and read the declaration and its comment. If there is no header, treat the function as `body` when it has an `address`, else `signature-only` |
| `body` | the function is in the binary. Read it (step 4) and summarize |
| `signature-only` | nothing can be read. Over-approximate: symbolic return of the IR's type with no assumption, and symbolic bytes into every pointer parameter the function could write (not into one a call site passes a constant string or a `@global` constant for) |

A function called from no site and whose address is not taken
(`summary.never_called`) needs no model; name it with `--leave`.

Leaving a function to the verifier's stub is a legitimate choice when the
stub is already right: a `void` function with no pointer parameters, a
logger. Name it with `--leave` in step 5 so the choice is recorded.

### 4. Read a body

Use whichever of these the session allows, in this order.

**Decompiled C, when Docker is available.** One function, as Ghidra prints
it:

```bash
scripts/ghidra/decompile-headless.sh --input IMAGE --c-source \
    --function FUN_000198c4 --output out/FUN_000198c4.c
```

On macOS the output path has to be inside the workspace, with `--ci <abs dir>`.

**An existing lift.** If the function was lifted before, its `.c` or `.ll`
is next to its P-Code JSON. Prefer the `.ll`: the printed `.c` drops
parentheses in pointer arithmetic.

**Disassembly, always available.** Take the start from `address` and the end
from the next function's address or the size in the Ghidra export:

```bash
$PATCHESTRY_LLVM_PREFIX/llvm-objdump -d --no-show-raw-insn \
    --triple=armv7-unknown-linux-gnueabi \
    --start-address=0x198c4 --stop-address=0x19918 IMAGE
```

`--triple` is needed for an image without build attributes, or `movw`/`movt`
print as `<unknown>`. For Thumb code add `--triple=thumbv7-...`. The warning
"no section overlaps the range" on an image without section headers is
harmless: the listing follows it.

Summarize with the five questions under "From a body" in
`references/model-patterns.md`. A summary states only what an instruction or
a line shows. If a question cannot be answered (the result comes from a
callee that was not read), over-approximate that part and say so in the
model's `Assumes:` line.

Reading every callee of a large function is not required. Read the ones
whose result or memory effect the function under test depends on: those
whose return value is used (`result_used` in the calls) and those that take
a pointer to the caller's locals. Give the rest signature-only models.

### 5. Write the models

Follow `references/model-patterns.md`: the rules at its top, then the pattern
that matches. `references/klee-intrinsics.md` lists the only KLEE functions
the stub header declares.

Order the file by class (libc, HAL, internal), one model per function, each
with its header comment (`Source`, `Memory effect`, `Return`, `Assumes`).

### 6. Check

```bash
python3 SKILL/scripts/check_models.py classified.json <stem>_models.c \
    [--leave NAME]... --bitcode <stem>_models.bc --report check.json \
    --verifier builds/default/tools/patchir-klee-verifier/Debug/patchir-klee-verifier \
    --module MODULE.ll --target-function FUNCTION
```

Add the sources of any existing library that `modelled_in` named
(`lib/patchestry/klee/models/usb_hal_models.c`) after `<stem>_models.c`, so
the single output file has them too: the verifier accepts one
`--model-library`.

Fix every error. Its docstring lists the checks; the common ones:

- `type`: a `long` or `size_t` in the model where the IR has `i32`. Use the
  fixed-width type.
- `arity`: the model was written from the header, and the lifted IR has a
  different parameter list. The IR wins.
- `missing`: write the model, or name the function with `--leave`.

Read every warning and either fix it or carry it into the report.
`undefined` for `malloc` is expected when a model allocates.

Exit 0 with `--verifier` means: the library compiles for
`x86_64-unknown-linux-gnu`, every external is modelled, left on purpose,
modelled elsewhere or provided by KLEE, the signatures match the IR, and the
verifier built a harness with it. It does not mean KLEE was run.
`references/verifier-usage.md` has the verifier's options and its rule for
accepting a library.

### 7. Report

Write `<stem>_models.md` and show the table to the user:

| Function | Class | Source | Memory effect | Return | Assumes |
| --- | --- | --- | --- | --- | --- |

Below the table, list:

- every bound used (buffer sizes, loop limits) and where it came from;
- every function left to the verifier's stub, and why that is enough;
- every signature-only model, as the places where the result is weakest;
- target intrinsics from step 1;
- disagreements found while reading: a lifted signature that differs from
  the header or from the body (a `_Bool` return for a function that returns
  2 or 3, dropped parameters);
- the command to use the library:

  ```bash
  patchir-klee-verifier MODULE.ll --target-function FUNCTION \
      --model-library <stem>_models.bc -o harness.bc
  scripts/klee/run-klee.sh --input harness.bc --output ./results
  ```

State plainly what was and was not executed. The check in step 6 runs the
verifier. KLEE runs only in the Docker image (`scripts/klee/run-klee.sh`);
if it was not run, say so.

## Rules that are easy to break

- A model's `Source:` line says `body` only when the body was read in this
  session, with the address range or file named. `classify.py` writing
  `source: body` means a body exists, nothing more.
- No `klee_assume` without a line in the header comment that says what it
  assumes and what justifies it.
- A too-strong model is worse than the stub. When unsure, leave the value
  unconstrained.
- The scripts report; they do not write models. Do not hand-edit their JSON
  to make a check pass.

## Files

- `scripts/list_externals.py`: the externals of a module, as JSON.
- `scripts/classify.py`: class, family, pattern and model source per external.
- `scripts/check_models.py`: compile, compare signatures, try the verifier.
- `references/model-patterns.md`: the model shapes and the rules for all of them.
- `references/hal-families.md`: HAL and RTOS conventions, and which are verified.
- `references/klee-intrinsics.md`: what `klee_stub.h` declares.
- `references/verifier-usage.md`: how the verifier consumes a library.
