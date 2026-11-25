# Verifiable Patching

Patchestry checks contracts on patched LLVM IR before compilation, then
validates the binary rewrite after reinsertion. These workflows address two
questions:

1. Does the patched IR satisfy the specified postconditions under the assumed
   preconditions and harness model?
2. Does the rewritten binary contain the intended code and preserve the
   compared behavior for inputs that both versions accept, without accepting
   inputs rejected by the original?

[IR-level verification](ir_verification.md) describes contract checking with
KLEE and instrumentation for SeaHorn.
[Patch validation](patch_validation.md) describes integrity and static checks,
QEMU execution, and binary comparison with PatchSymDiff. PatchSymDiff executes
the original and rewritten machine code symbolically with angr and uses cozy
to compare the resulting states.

Results apply to individual functions under the specified input bounds and
environment assumptions. Contracts describe the properties required of the
patch; binary comparison complements those checks but does not establish that
the contracts fully capture the vulnerability.

## Checks in the pipeline

```text
Ghidra JSON
    | patchir-decomp
    v
CIR  + YAML patches and contracts
    | patchir-transform
    v
patched CIR (contracts attached as attributes)
    | patchir-cir2llvm
    v
patched LLVM IR (contracts carried as !static_contract metadata)
    |
    |---- IR-level verification -------------------------------------------
    |       patchir-klee-verifier  -> harness -> KLEE
    |       patchir-seahorn-verifier -> instrumented IR for SeaHorn
    |
    | compile, Patcherex2 rewrite of the original ELF
    v
rewritten ELF
    |
    \---- validation after reinsertion ---------------------------------------
            integrity and static checks   (intended IR against the rewritten ELF)
            runtime under QEMU            (rewritten firmware)
            binary comparison             (original ELF against the rewritten ELF)
            evidence                      (one artifact per validation case)
```

## What each check establishes

| | Contract verification | Validation after reinsertion |
| --- | --- | --- |
| Property | Postconditions at a call site, under assumed preconditions | Rewrite integrity, calling convention, expected runtime output, and [P1 to P4](#properties-p1-to-p4) |
| Checked on | Patched LLVM IR | Intended patched IR and the original and rewritten ELF files |
| Method | Symbolic execution of a generated harness with KLEE; instrumentation for SeaHorn (not run in CI) | static checks of the rewritten bytes, a QEMU run, and symbolic comparison of machine code (PatchSymDiff) |
| Configuration | contracts written by the patch author in YAML | input bounds, rejection rules, callee models, and device ranges declared for the case |
| Detail | [IR-level verification](ir_verification.md) | [Patch validation](patch_validation.md) |

A contract check can pass even if the compiled patch corrupts a saved register.
Conversely, preserving the original behavior does not establish that a bug was
fixed. The contract must describe the intended fix, and the binary checks must
cover the behavior the patch needs to preserve.

## Properties P1 to P4

A property is a rule that the comparison checks. P1 through P4 describe how
the patched function may differ from the original when both receive the same
input and start from the same modeled state.

An input is **rejected** when execution meets a rejection rule declared for
the case: an error return value, a reject handler, a write to a declared fault
address, or an omitted call listed in `reject_skips`.
Otherwise, a completed execution is treated as **accepted**. The tool does
not decide whether an input is valid on its own. An execution it cannot
finish or analyze makes the comparison incomplete.

| Property | Rule | Example of a violation |
| --- | --- | --- |
| **P1: Preserve existing rejections** | If the patch accepts an input, the original must also accept it. The patch may reject additional inputs. | The original rejects a malformed request, but the patch processes it. |
| **P2: Preserve memory results** | For an input both versions accept, caller-visible memory must contain the same bytes when the function finishes. This includes globals and output buffers, but excludes temporary memory below the final stack pointer. | A valid request leaves different bytes in an output buffer. |
| **P3: Preserve return values** | For an input both versions accept, they must return the same value. | A valid request returns a different length or status code. |
| **P4: Preserve external calls** | For an input both versions accept, the external calls covered by the model must match in order and arguments. | The patch omits a required call, calls a function twice, or passes a different argument. |

For example, a patch may add a length check that rejects oversized messages.
For messages both versions accept, the memory results, return value, and
modeled external calls must remain unchanged. These properties allow that
restriction without requiring the patched function to reproduce the
original's unsafe behavior on rejected inputs.

The checks apply within the declared input bounds and callee models. In
PatchSymDiff, P4's call records cover stubbed callees. Callees that are executed
are checked through their effects on memory and registers; writes to declared
device ranges are also compared in order. Register checks also cover
preservation of r4 through r11 and sp. Paths with no corresponding inputs in
the other function are reported as `unmatched-path`. See
[Patch validation](patch_validation.md#comparison-properties-p1-to-p4) for the report
fields and implementation limits.

## Verification at IR level

The patch author writes each static contract in the YAML spec (see
[Patch specifications](GettingStarted/patch_specifications.md)).
`patchir-transform` attaches it to the matched operation, and
`patchir-cir2llvm` carries it into LLVM IR as `!static_contract` metadata.
Static contracts add no executable code until a verifier instruments them.

Two tools read that metadata:

- `patchir-klee-verifier` builds a harness around one function: symbolic
  arguments and globals, stubs for external functions, each precondition as an
  assumption, and each postcondition as a check that aborts on failure. KLEE
  explores the harness and reports paths that reach an abort.
- `patchir-seahorn-verifier` rewrites the same metadata into
  `__VERIFIER_assume` and `__VERIFIER_assert` calls for SeaHorn.

These tools check the lifted function after the patch has been applied. They
assume the lift is accurate; they do not inspect the machine code produced
later.

## Validation after reinsertion

Once Patcherex2 has written the patch into the ELF, three versions of the
function exist: the original bytes, the intended patched IR, and the rewritten
bytes. Each check examines a different part of that transformation, which
helps narrow down a failure.

- Integrity checks validate symbol addresses, dependencies, ABI compatibility,
  and patch placement. A separate compilation of the intended IR supplies the
  expected bytes. Static analysis checks stack balance and preservation of
  callee-saved registers in the inserted code, along with declared code-size
  and stack budgets.
- The firmware runtime check boots the rewritten firmware under QEMU and
  compares its output with the expected transcript.
- [PatchSymDiff](../scripts/patch-runtime/patch_symdiff.py) checks P1 to P4
  using the original and rewritten machine code. This allows it to detect
  errors introduced by the Ghidra-to-IR lift. It still uses IR declarations
  for stub signatures and the function's return type.
- [Evidence collection](../scripts/patch-runtime/evidence.py) writes
  `<case>/validation.json`, recording each check's verdict and assumptions and
  aggregating results into integrity, static, runtime, and comparison groups.

The drivers run the applicable stages and stop at the first integrity failure.
The [validation stages](patch_validation.md#validation-stages) describe their
order and link to each script. Contract verification produces separate
results; the binary-validation evidence does not combine KLEE or SeaHorn
results with these checks.

## Verdicts

The binary-validation evidence file records four verdicts. A group or case
receives the weakest verdict among its checks, ordered from `failed` through
`not-run` and `tested` to `proven`.

| Verdict | Meaning |
| --- | --- |
| `proven` | The check established its claim within the specified bounds and assumptions. |
| `tested` | The claim held for the inputs exercised by the test. |
| `failed` | The check found a problem. |
| `not-run` | The check did not run, could not reach a conclusion, or had no declared property to check. |

PatchSymDiff also reports `incomplete` when it cannot finish execution or
requires unsupported models. This can include unconfirmed differences, so it
does not establish that the functions agree. The evidence records incomplete
comparisons and missing reports as `not-run`.

Some firmware cases declare an expected behavioral difference, such as added
UART writes. The comparison must find the declared difference kind and no
others, while QEMU checks the expected output. These comparisons receive
`not-run` in the evidence because detecting an expected difference does not
prove it safe. See [Recorded results](patch_validation.md#recorded-results)
for the firmware and recompilation cases.

The [KLEE gate](ir_verification.md#the-gate-in-ci) reports its own verdicts.
An incomplete run can satisfy a test's `no-errors` expectation without
establishing a proof.

Regression tests include deliberate failures. For example, the KLEE gate must
detect an abort in its control test, and both binary-validation suites include
cases that must be rejected.

## Limitations

- The lifted original is not compared with the patched IR at IR level.
  PatchSymDiff checks for behavioral differences after recompilation.
- The contract checks trust the lift and the harness: stubbed external
  functions return unconstrained values, and pointer arguments point at
  buffers of a fixed size.
- Binary comparison covers one function at a time within the specified input
  bounds. Whole-binary equivalence, timing, and behavior inside stubbed callees
  are outside its scope. Stubs are assumed to leave memory unchanged, and
  memory read before it is written is initialized to zero. Device-write order
  is checked only for declared device ranges. See the
  [comparison limitations](patch_validation.md#limitations) for details.
- KLEE runs the harness retargeted to x86_64, so properties that depend on
  the 32-bit memory layout are outside what it can show (see
  [KLEE integration](klee-integration.md)).
- SeaHorn is not run in CI. The repository builds the instrumentation tool and
  a SeaHorn image, and has no test suite that runs one against the other.

## Related documentation

| Topic | Page |
| --- | --- |
| Writing contracts and patches | [Patch specifications](GettingStarted/patch_specifications.md) |
| Contract verification on LLVM IR | [IR-level verification](ir_verification.md) |
| KLEE image, retargeting, limits and the CI gate | [KLEE integration](klee-integration.md) |
| Stages, properties, and results after reinsertion | [Patch validation](patch_validation.md) |
| Running the firmware suite | [QEMU firmware runtime](GettingStarted/qemu_firmware_runtime.md) |
| Running the recompilation-fidelity suite | [Recompilation fidelity](GettingStarted/recompilation_fidelity.md) |
