# Patch Validation: Strategy, Stages and Comparison with VeriBin

This document describes how Patchestry checks that a patch does no harm, what
each stage of the check does and where it lives in the repository, what has
been run and what has not, and how the approach differs from VeriBin (Wu et
al., NDSS 2025).

## The claim

The validator makes one claim per patched function: the patch only restricts
inputs and changes nothing else. An input the patched function accepts was
accepted by the original, and on every such input the two behave the same.

It does not claim that the vulnerability is fixed. That stays with the
contract checks in `patchir-klee-verifier` (see `klee-integration.md`). Timing,
whole-binary equivalence and the behaviour inside stubbed callees are also out
of scope.

## Strategy

Three versions of the function exist during patching:

1. the lifted original (Ghidra P-Code, lowered through `patchir-decomp` and
   `patchir-cir2llvm` to LLVM IR),
2. the intended patched IR (the output of `patchir-transform`),
3. the bytes that Patcherex2 wrote into the rewritten ELF.

The check is split along those three so that a failure names its cause.

Step A compares 1 with 2. It asks whether the patch design does harm. Declared
reject paths are allowed here: the patch may turn an accepted input into a
rejected one, and nothing else.

Step B compares 2 with 3. It asks whether the patcher and the toolchain applied
what was intended. No reject path and no relaxation is allowed: the two must
be the same function.

Three rules hold across both steps.

Every check reports one of four verdicts, and the weakest one wins.

| Verdict | Meaning |
| --- | --- |
| `proven` | the claim follows from the reports alone, or from a complete symbolic exploration |
| `tested` | the claim held on the inputs that were run, and only on those |
| `failed` | the check found a problem |
| `not-run` | the check did not run, could not decide, or has nothing declared to check |

Nothing counts as passed because nobody looked. A check whose report is absent
is `not-run`.

Everything the result depends on is recorded as a named assumption in the
evidence: that the lift is faithful, which callees were stubbed or modelled,
which exits count as rejection, which address ranges are device registers,
that inputs were sampled.

Every check has seeded harms. A check is only trusted once a deliberately
broken patch has made it fail.

## Stages

The stages run in this order. A case stops at the first integrity failure,
because later stages would compare the wrong things.

| Stage | Question | Where | Verdict it can give |
| --- | --- | --- | --- |
| 1. Bindings | Does every global and function the lifted code refers to exist in the image at the address, size and kind the lift assumed? | `check_bindings.py` | proven, failed |
| 2. Dependencies and ABI | Does the inserted code need anything the image cannot supply (compiler runtime calls, writable state), and was it compiled for the image's ABI (CPU architecture, floating-point unit, float calling convention)? | `patch_dependencies.py`, `check_abi.py` | proven, failed |
| 3. Reinsertion | Did the rewrite stay inside the function, or branch to new code without overwriting a neighbour, splitting an instruction or leaving a stale entry? | `check_trampoline.py` | proven, failed |
| 4. Patch record | Which bytes did Patcherex2 write, where, with which compiler and flags? Do they agree with the rewritten ELF? | `patch_record.py` | proven, failed |
| 5. Applied bytes | Recompile the intended snippets and compare them with the bytes in the rewritten ELF. | `check_applied_bytes.py` | proven, failed |
| 6. Static discipline | Per inserted function: stack balance, saved registers restored, code size and stack depth against declared budgets. | `check_static.py` | proven, failed, not-run when no budget is declared |
| 7. Equivalence (Step A) | P1 to P4 between the lifted original and the patched IR. | `patchir-klee-verifier --compare-original`, `check_equivalence.py` | tested, failed; proven only from KLEE |
| 8. Firmware checks | Device write order, acquire and release balance, hygiene of reject paths. | same harness | tested, failed, not-run when nothing is declared |
| 9. Runtime | Does the rewritten firmware still print the expected transcript under QEMU? | `qemu_firmware_runtime.py` | tested, failed |
| 10. Re-lift (Step B) | Lift the patched function back from the rewritten ELF and compare it with the intended IR. | `check_relift.py`, `relift_cases.py` | tested, failed |
| 11. Evidence | Collect every report into one artifact with verdicts and assumptions. | `evidence.py` | |

All scripts are under `scripts/patch-runtime/`.

### Stages 1 to 6: integrity and static checks

These read the two ELF files and the patch record. They do not execute
anything, and their verdicts are `proven` or `failed`. Stage 5 is the first
half of Step B: it shows that the bytes in the image are the bytes the
compiler produced for the intended code. It trusts the compiler, which is why
stage 10 exists.

### Stage 7: equivalence, properties P1 to P4

The property definitions are VeriBin's.

| Property | Claim |
| --- | --- |
| P1 | An input the patched function accepts, the original accepts. |
| P2 | When both accept, the final non-local memory is byte-equal. |
| P3 | When both accept, the return values are equal. |
| P4 | When both accept, the calls of external functions and their arguments are equal, after the declared rules. |

`patchir-klee-verifier --compare-original` builds a product harness: one LLVM
module that holds both versions, each with its own copy of every global. The
harness makes the arguments, the argument buffers and the writable globals
symbolic and equal, runs the patched version, restores the inputs, runs the
original, and ends the path with a code that says which property differed.

How each property is observed:

- P1: a version rejects when it calls a declared reject handler, returns a
  declared reject value, or skips a declared call. A fault or an abort in the
  original alone also counts.
- P2: paired globals without pointers and the argument buffers are compared
  byte by byte after both runs. Stack memory is not compared.
- P3: the return values are compared; pointers are compared after mapping an
  address in either copy of a global to the same place.
- P4: every external function is a stub that records its name and arguments.
  The two traces are compared. A `pure_callee` may be dropped, duplicated or
  reordered. A `substitution` lets one named callee stand for another. A
  small set of string functions run as models instead of stubs.

Declarations are command-line options today (`--reject-handler`,
`--reject-return`, `--reject-skip`, `--pure-callee`, `--substitution`,
`--arg-buffer`, `--arg-range`). The patch specification does not carry them
yet.

The harness has two runners.

The concrete runner (`check_equivalence.py`) links the harness as bare-metal
ARM code and runs it under `qemu-system-arm -M virt` on generated inputs, 200
by default. It gives `tested`.

KLEE is the symbolic runner. `lib/patchestry/klee/models/compare_models.c`
ends a violating path in `klee_abort` from a function named after the failed
check, and `scripts/klee/klee_gate.py` reads the KLEE output directory and
gives `proven` only for a complete run with no error. This path has not been
exercised: KLEE is not installed on the development host and the project's
image is x86-64 only. Until it runs, no property is `proven`.

### Stage 8: firmware checks

These use the same runs and are `not-run` unless something is declared.

| Check | Declared with | Holds when |
| --- | --- | --- |
| `mmio-order` | `--mmio-region LOW:HIGH` | when both accept, both write the same values to the device registers in the same order |
| `lock-balance` | `--balance ACQUIRE:RELEASE` | the calls balance the same way in both versions, and are balanced when the patched version rejects |
| `reject-hygiene` | `--reject-effect NAME` allows a call | before it rejects, the patched version writes no compared memory and no device register and calls nothing but declared effects |

With a device range declared, every load and store that is not plainly of a
local or a global asks the runner first. Inside a range a write is recorded
and not performed, and a read returns the next of a series of values both
versions share.

### Stage 10: re-lift

`check_relift.py` runs Ghidra on the rewritten ELF, lowers the lifted function
the same way the original was lowered, and compares it with the intended IR
through the stage 7 harness with nothing declared. On firmware it lifts the
whole image so that helper functions the patch inserted are lifted with the
target. On a stripped binary it lifts one function at the trampoline
destination from the reinsertion report.

This is the only check that starts from the bytes and does not share the
compiler or Patcherex2 with the rest of the pipeline. It does share Ghidra,
`patchir-decomp` and the harness. A failure says the bytes, as Ghidra reads
them, do not behave like the intended code. It cannot tell a wrong byte from a
wrong lift.

### Binary-to-binary comparison (prototype)

`check_binary_equivalence.py` compares the function in the original ELF with
the function in the rewritten ELF by symbolic execution of the machine code
of both, in angr. Neither side is lifted by Ghidra or `patchir-decomp`, so
this is the one check where a lifting error shows up as a difference instead
of being shared by both sides. Reinserting a function unchanged and comparing
it this way tests the lift and recompile round trip on its own.

Both sides start from the same symbolic arguments, argument buffers and
writable globals. Calls are not followed: each is recorded and returns a
symbol shared by both sides. Each side is explored path by path, and for
every pair of paths one input can take, the solver is asked for an input that
violates P1, P2, P3, P4 or the device write order. The verdict is `proven`
when no such input exists and every path of both sides was explored, `failed`
with the input when one exists, and `incomplete` when a limit was reached.

The address of a local is compared as "a local" and the address of read-only
data by the bytes there, because the two images lay both out differently.
Callee argument counts come from the declarations in the lifted IR; that is
the only thing taken from the lift.

It is not wired into the suites or the evidence. It needs angr, which the
Patcherex2 environment of the suites already has.

| Case | Result |
| --- | --- |
| firmware `length`, reject value declared | proven, 1 and 2 paths |
| firmware `length`, nothing declared | failed P2 and P3, with inputs |
| firmware `send`, reject handler and device range declared | proven, including the order of both device writes |
| firmware `send`, handler not declared | failed P4 |
| firmware `forward`, skipped call declared | proven |
| firmware `forward`, nothing declared | failed P4 |
| `length` with `adds` changed to `subs` in the inserted code | failed P3 |
| `length` with the limit changed from 16 to 64 | proven: against the original this is still a restricting patch. Only the comparison with the intended code (stage 5, stage 10) sees it |
| `httpd` `FUN_00067c0c` reinserted unchanged | proven, 5 and 5 paths, 25 calls per path |
| the same with one byte changed (a `memset` length, 64 to 32) | failed P4 |
| `httpd` `FUN_0007ae8c` reinserted unchanged, 3-byte buffers | incomplete: 16 and 16 paths agree, but both sides write through an address that depends on the input |
| `cgibin` `FUN_00026534` reinserted unchanged | incomplete: more than 150 paths |

Limits of the prototype: callees are assumed to write no memory; stack and
floating-point arguments are not compared; memory that is read before it is
written is zero; path pairs are compared one by one, so a function with many
paths runs out of time; a writable segment over 1 MB keeps the image's bytes.

### Stage 11: evidence

`evidence.py` writes `<case>/validation.json`: one row per check with its
verdict, the report it came from, the problems found and the facts measured;
a verdict per group (integrity, static, runtime, equivalence, firmware,
cross-check); and the list of assumptions. `collect_reports.py` collects
these into tables.

## What has been run

Two suites drive the stages.

`scripts/test-qemu-firmware-runtime.sh` patches the in-repo Cortex-M3 firmware
`firmwares/serial-qemu` and runs it under QEMU. `scripts/test-identity-reinsertion.sh`
lifts functions from prebuilt ARM binaries, reinserts them unchanged, and
compares seeded variants with the original.

| What | Result |
| --- | --- |
| Firmware cases `length`, `forward`, `send` against their lifted originals | P1, P3, P4 tested on 200 inputs each; `send` also tested on write order, balance and reject hygiene |
| Seeded harms in the firmware suite (8) | each fails the expected check |
| Seeded variants on `httpd` and `cgibin` functions (15) | each gives the expected verdicts |
| Re-lift of `length`, `forward`, `replace`, `httpd` `FUN_0007ae8c` and `FUN_00067c0c` | tested |
| Re-lift with one byte of the inserted code changed (2) | both fail |
| Harness lit tests | 32 pass in Debug and Release |

Known gaps and open findings:

- Nothing is `proven` for P1 to P4. All results are `tested` on sampled
  inputs.
- The re-lift of `send` fails `mmio-order`: Ghidra's decompiler removes the
  first of two stores to a device register as dead. The re-lift cannot confirm
  device write sequences until the lift keeps such stores.
- The re-lift of `cgibin` `FUN_00026534` fails P1. The intended IR faults on
  about a quarter of the inputs where the re-lifted function does not. Not
  diagnosed.
- Both the original and the patched IR come from the same lifter. A lifter
  bug that affects both the same way is invisible to stage 7. Stage 10 covers
  the patched side only. The binary-to-binary prototype removes this for
  the functions it can finish.
- Writes through an address computed from a constant are outside P2 unless
  the range is declared as device registers. A device register nobody
  declared is not seen.
- A callee without a model is a stub that returns shared values. Behaviour
  inside it is not checked, and a function whose callees return pointers can
  be inconclusive on every input.
- The condition of a `substitution` is recorded, not checked.

## How this differs from VeriBin

VeriBin takes an original and a patched binary and decides whether the patch
is Safe to Apply (StA). A patch is StA when it does not enlarge the valid
input space and leaves the output unchanged on valid inputs. The paper splits
this into the four properties used above: P1 non-increasing input space, P2
non-local memory writes, P3 return value, P4 function calls.

Patchestry adopts those definitions and changes how each is established. The
underlying difference is the setting. VeriBin is handed two binaries and must
discover what the patch is. Patchestry produced the patch, so it already
holds the patch specification, the intended IR and the record of what the
patcher wrote. Most of VeriBin's heuristics exist to recover facts that
Patchestry has, and Patchestry gains a second question VeriBin does not ask:
was the patch applied as intended.

| | VeriBin | Patchestry |
| --- | --- | --- |
| Input | original and patched binaries, plus the address of the patched function | original binary, patch specification, intended IR, rewritten binary, patch record |
| Question | is the patch safe to apply | is the patch design harmless (Step A) and was it applied as designed (Step B) |
| Function and block matching | BinDiff | not needed; the patch record names the function and the inserted code |
| Level of comparison | binary, through angr's lifting | LLVM IR lifted through Ghidra, ClangIR and LLVM |
| Engine | under-constrained symbolic execution in angr, SMT queries over path constraints | one product harness; run concretely under QEMU today, KLEE intended for the proof |
| Path handling | enumerate complete paths of each function, pair them into matching path pairs by block similarity, then check per pair | no path pairing; both versions run on the same input inside one harness |
| Loops | each block visited at most twice | concrete runs execute loops fully; KLEE would be bounded by time and harness bounds |
| Error exits | heuristics: call to a non-returning or known error function, a return value judged invalid, a path that is short compared with the others | declared per patch: reject handler, reject return value, skipped call |
| Callees | uninterpreted functions of their arguments; non-constant pointer arguments are treated as outputs | recording stubs with shared return values, models for some string functions, `pure_callees` and `substitutions` declared |
| Compiler offset noise | content-based comparison of constants, offset matching of locals | constants compared by content, paired globals mapped to the same place, locals compared by kind; the IR removes most of the noise |
| When a property fails | adaptive mode asks the analyst a question (five kinds, such as whether a changed value or an extra call is acceptable) and re-checks with the answer | no interaction; relaxations are written down before the run and appear as named assumptions |
| Result | StA or not StA | a verdict per check, per group, and a list of assumptions |
| Beyond P1 to P4 | none | applied-bytes comparison, reinsertion and binding checks, stack and register discipline, device write order, lock balance, reject hygiene, a runtime transcript, re-lift |

Three differences matter most in practice.

Rejection is declared, not inferred. VeriBin's error-path heuristics are the
part its authors name as a source of unsoundness: if every path is classified
as an error path, nothing is compared. Patchestry refuses to guess. With no
reject path declared, any difference is a violation, which is the strict
reading. The cost is that someone has to state how the patch rejects.

The comparison is on the same input, not on matched paths. VeriBin pairs
paths of the two functions and proves implications between their constraints,
which needs the pairing to be right and the paths to be enumerable. The
product harness sidesteps pairing: for one input, both versions run and the
observations are compared. With KLEE this would explore the product of the two
functions' paths; with the concrete runner it samples inputs.

The strength of the result is different today. VeriBin's verdicts come from
SMT queries over symbolic paths, bounded by its loop strategy and the
correctness of its recovered function information. Patchestry's P1 to P4
verdicts are `tested` on a few hundred generated inputs. That is weaker for
any single property. The proof step needs KLEE to run the harness, and it has
not.

What VeriBin does that Patchestry does not:

- It works on any pair of binaries, including patches made by someone else
  with a compiler. Patchestry validates only its own patches.
- It gives a symbolic result without a harness, models or declarations.
- Its adaptive mode lets an analyst accept a semantically equivalent change
  found during the run. Patchestry would report it as a failure until a
  declaration is added and the run repeated.
- It has a published evaluation.

What Patchestry does that VeriBin does not:

- It checks the patcher: that the bytes in the image are the intended code,
  that the rewrite did not damage neighbours, and that the inserted code
  matches the image's ABI and keeps the stack and saved registers intact.
- It checks firmware-specific harm: the order of device register writes,
  interrupt and lock balance, and side effects before a rejection.
- It runs the rewritten firmware.
- It records every assumption with the verdict.

### VeriBin's reported results

From the paper: 4,100 lines of Python using BinDiff, IDA Pro and angr. Three
datasets (MicroPatch Bench, DARPA AMP challenges, FFmpeg patches from
PatchDB), 125 patch pairs. 86 could be analysed; the rest were excluded for
binary-diffing failures, an 8-hour timeout, memory limits, or an AVR target
that angr does not support. On those 86 it reports 93.0% accuracy with no
false positives and 6 false negatives, an average runtime of 1,293.8 seconds
excluding preprocessing, and an average of 1.3 analyst questions per patch in
adaptive mode. On stripped binaries accuracy is 89.4%. The stated limits are
inaccurate function information from the underlying tools, the error-path and
offset heuristics, and path explosion on large functions.

Patchestry has no comparable evaluation. Its corpus is 3 firmware cases, 3
functions from prebuilt ARM binaries, and 25 seeded harms, all of which were
caught.

### Cross-check with VeriBin

The plan lists running unmodified VeriBin on the original and rewritten
binaries as an independent cross-check, because it shares nothing with
Patchestry's lifter. This has not been attempted. Whether VeriBin handles
Cortex-M Thumb images is not known; the paper explains its techniques on
x86-64 and its AMP dataset includes ARM binaries.

## References

- Wu, Wu, Wu, Sharma, Machiry, Bianchi. VeriBin: Adaptive Verification of
  Patches at the Binary Level. NDSS 2025.
  <https://www.ndss-symposium.org/ndss-paper/veribin-adaptive-verification-of-patches-at-the-binary-level/>
- VeriBin source: <https://github.com/purseclab/VeriBin>
- `GettingStarted/identity_reinsertion.md`: the comparison harness, its
  options and the identity suite.
- `GettingStarted/qemu_firmware_runtime.md`: the firmware suite, its reports,
  the seeded harms and the re-lift.
- `klee-integration.md`: the KLEE image and the KLEE gate.
