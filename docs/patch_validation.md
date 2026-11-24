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

1. the bytes of the function in the original ELF,
2. the intended patched IR (the lifted function after `patchir-transform`),
3. the bytes that Patcherex2 wrote into the rewritten ELF.

The check is split along those three so that a failure names its cause.

The integrity and static checks compare 2 with 3. They ask whether the
patcher and the toolchain applied what was intended. No reject path and no
relaxation is allowed: the bytes must be the intended code.

The behavioural check compares 1 with 3, machine code against machine code.
It asks whether the patched function does harm. Declared reject paths are
allowed here: the patch may turn an accepted input into a rejected one, and
nothing else. A comparison of the lifted original with the intended IR, at IR
level, is not part of the validator.

Three rules hold across both.

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
| 7. Runtime | Does the rewritten firmware still print the expected transcript under QEMU? | `qemu_firmware_runtime.py` | tested, failed |
| 8. Binary comparison | Does the function in the rewritten ELF behave like the function in the original ELF, apart from declared rejections? | `patch_symdiff.py` (PatchSymDiff) | proven, failed, incomplete |
| 9. Evidence | Collect every report into one artifact with verdicts and assumptions. | `evidence.py` | |

All scripts are under `scripts/patch-runtime/`.

### Stages 1 to 6: integrity and static checks

These read the two ELF files and the patch record. They do not execute
anything, and their verdicts are `proven` or `failed`. Stage 5 shows that the
bytes in the image are the bytes the compiler produced for the intended code.
It trusts the compiler.

### Properties P1 to P4

The binary comparison answers four properties. The definitions are VeriBin's.
PatchSymDiff reports kinds of difference; the table gives the kind that
violates each property.

| Property | Claim | Kind of difference |
| --- | --- | --- |
| P1 | An input the patched function accepts, the original accepts. | `accepts-rejected-input` |
| P2 | When both accept, the final non-local memory is byte-equal. | `memory` |
| P3 | When both accept, the return values are equal. | `register` (r0, and r1 for a 64-bit result) |
| P4 | When both accept, the calls of external functions and their arguments are equal, after the declared rules. | `calls` for stubbed callees; `device-writes` for what executed callees write to a device |

Two kinds go beyond the four: `register` also covers r4 to r11 and sp, which
a function must preserve, and `unmatched-path` is a path on one side that no
path on the other shares an input with.

### Stage 8: binary comparison

`patch_symdiff.py` compares the function in the original ELF with the
function in the rewritten ELF by symbolic execution of the machine code of
both, in angr, and pairs the final states with cozy. Neither side is lifted
by Ghidra or `patchir-decomp`, so a lifting error shows up as a difference.
Reinserting a function unchanged and comparing it this way tests the lift and
recompile round trip on its own. It is the only comparison engine; the
earlier `check_binary_equivalence.py` was retired in its favour.

Both sides start from the same symbolic arguments and argument buffers.
Globals hold the bytes of the image. r4 to r11 hold one shared symbol each.
Memory read before it is written is zero. `--arg-range INDEX:LOW:HIGH` bounds
an integer argument, and `--arg-suffix INDEX:HEX` sets the bytes that follow
an argument's buffer; without it they are zero.

Rejection is declared: `--reject-handler NAME`, `--reject-return VALUE`,
`--reject-write LOW:HIGH`, `--reject-skip NAME`. With `--mmio-region LOW:HIGH`
the writes to device registers are compared in order.

Callees are executed on both sides unless stubbed. A stub is a record of the
call and an unknown result that is the same on both sides for the same callee
and call index, and it is assumed to write no memory. The suites decide what
to stub:

- The firmware suite stubs nothing, except a callee named in `reject_skips`.
  What a function prints is seen as its writes to the UART.
- The identity-reinsertion suite stubs every function the lifted code is
  bound to, except those the case lists under `follow_callees`. The rest of
  a large image is not what is under comparison, and an import has no code.

The argument and result registers of a stub come from the declarations in
the IR the code was compiled from; that, and the return type of the function
itself, is the only thing taken from the lift. A 64-bit integer is two
registers. A called stub with a parameter or result that is not placed in
core registers (floating point, an aggregate, a stack argument, the variable
arguments of a variadic callee) makes the run `incomplete`.

A stubbed read-only string function (`strncmp`, `strncasecmp`, `strcmp`,
`strcasecmp`, `memcmp`, `strlen`, `strchr`) has its result computed from the
bytes it is given when the path fixes its pointers and length. If one call
is computed on some paths and left unknown on others, the differences found
are listed as `unconfirmed` and the run is `incomplete`, not `failed`: an
unknown result can separate two paths that agree.

The address of a local passed to a stub is not compared, and the address of
read-only data is compared by the bytes there, because the two images lay
both out differently.

The verdict is `proven` when no input shows a difference and every path of
both sides returned, `failed` with the input when one exists, and
`incomplete` when a path stopped at the loop bound or with an error, the time
ran out, or something above was not modelled.

cozy pins an older angr than Patcherex2 uses, so the suites run the script as
a separate process (`patch_symdiff.compare_in_own_environment`). By default
`uv` builds the environment from `scripts/symdiff/requirements.txt`;
`PATCHESTRY_SYMDIFF_PYTHON` names an interpreter that already has cozy.
`scripts/symdiff/` also has a Docker image for running it by hand.

The identity-reinsertion suite runs it for a case that declares its inputs
(`compare` in the cases file: buffers, the bytes after them, argument ranges,
callees to follow, the time limit) and writes `patch_symdiff.json`. The
evidence reads that as the check `binary-comparison`: `proven`, `failed`, or
`not-run` when the comparison is incomplete or the case declares nothing. A
failed comparison fails the case. The firmware suite runs it for every case
that reaches the end, with the rejections each patch declares; a patch that
is meant to make a kind of difference declares that kind, and the comparison
must then find it and nothing else (`not-run` in the evidence).

Results of the two suites:

| Case | Result |
| --- | --- |
| firmware `length`, reject value declared | proven |
| firmware `length`, nothing declared | failed |
| firmware `send`, reject handler and device range declared | proven, including the order of the device writes |
| firmware `forward`, skipped call declared | proven |
| the nine firmware patches that add a line to the transcript | `device-writes` only, as each declares |
| `httpd` `FUN_0007ae8c` reinserted unchanged, 3-byte buffers, offset argument 0 to 8 | proven, 16 and 16 paths |
| `httpd` `FUN_00067c0c` reinserted unchanged | incomplete: it calls `sprintf` and `printf`, whose variable arguments are not compared; nothing differs in what is |
| `cgibin` `FUN_00026534` reinserted unchanged, 4-byte buffer followed by `99;`, `FUN_0002609c` and `FUN_0002619c` followed | proven, 110 and 113 paths, 23 seconds |
| the same with one compared character changed (`;` to `:`) | failed: `register` and `calls` |

Limits: stubs are assumed to write no memory; stack and floating-point
arguments are not compared; memory that is read before it is written is
zero; paths are not joined, so the time grows with the number of paths, which
grows with the size of the input.

### Stage 9: evidence

`evidence.py` writes `<case>/validation.json`: one row per check with its
verdict, the report it came from, the problems found and the facts measured;
a verdict per group (integrity, static, runtime); and the list of
assumptions. `collect_reports.py` collects
these into tables.

## What has been run

Two suites drive the stages.

`scripts/test-qemu-firmware-runtime.sh` patches the in-repo Cortex-M3 firmware
`firmwares/serial-qemu` and runs it under QEMU. `scripts/test-identity-reinsertion.sh`
lifts functions from prebuilt ARM binaries and reinserts them unchanged.
Both run stages 1 to 7 and 9. The first runs stage 8 on the twelve cases that
reach the end: the three restricting patches are proven with their declared
rejections, and the nine that add a line to the transcript differ in
`device-writes` only, as each declares. The second runs stage 8 on the cases
that declare inputs (`httpd` `FUN_0007ae8c` and `FUN_00067c0c`, `cgibin`
`FUN_00026534` with a 4-byte buffer); two are proven and `FUN_00067c0c` is
incomplete because of its variadic callees, and a fourth case with one
constant of `FUN_00026534` changed is rejected by the comparison alone.

Known gaps:

- A patch that declares a property as changed gets no check of that property
  at all. Nine firmware patches declare `device-writes`, so what they print is
  not compared.
- It finishes only on small functions. One of the three functions from the
  prebuilt binaries ends `incomplete`, and another is proven only for 3-byte
  buffers.
- A stubbed callee is assumed to write no memory, and behaviour inside it is
  not checked.
- A device register nobody declared is compared as ordinary memory.
- Stages 1 to 6 compare the bytes with the intended IR and trust the lifter
  for what that IR means. Only stage 8 does not.

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
| Question | is the patch safe to apply | was the patch applied as designed, and does the patched function do harm |
| Function and block matching | BinDiff | not needed; the patch record names the function and the inserted code |
| Level of comparison | binary, through angr's lifting | binary, through angr's lifting, for behaviour; bytes against a recompilation of the intended IR for integrity |
| Engine | under-constrained symbolic execution in angr, SMT queries over path constraints | symbolic execution in angr (through cozy) from a shared symbolic start state, SMT queries per pair of final states |
| Path handling | enumerate complete paths of each function, pair them into matching path pairs by block similarity, then check per pair | enumerate the paths of each function and query every pair one input can take; no pairing heuristic |
| Loops | each block visited at most twice | bounded by an optional loop bound and a time limit; a run that hits one is `incomplete` |
| Error exits | heuristics: call to a non-returning or known error function, a return value judged invalid, a path that is short compared with the others | declared per patch: reject handler, reject return value, skipped call |
| Callees | uninterpreted functions of their arguments; non-constant pointer arguments are treated as outputs | executed, or stubbed: recorded with their arguments, the return value a symbol shared by both sides per callee and call index, assumed to write no memory |
| Compiler offset noise | content-based comparison of constants, offset matching of locals | read-only data compared by content, locals compared by kind |
| When a property fails | adaptive mode asks the analyst a question (five kinds, such as whether a changed value or an extra call is acceptable) and re-checks with the answer | no interaction; relaxations are written down before the run |
| Result | StA or not StA | a verdict per check, per group, and a list of assumptions |
| Beyond P1 to P4 | none | applied-bytes comparison, reinsertion and binding checks, stack and register discipline, device write order, a runtime transcript |

Three differences matter most in practice.

Rejection is declared, not inferred. VeriBin's error-path heuristics are the
part its authors name as a source of unsoundness: if every path is classified
as an error path, nothing is compared. Patchestry refuses to guess. With no
reject path declared, any difference is a violation, which is the strict
reading. The cost is that someone has to state how the patch rejects.

Paths are not matched by similarity. VeriBin pairs paths of the two functions
by block similarity and proves implications between their constraints, which
needs the pairing to be right. The prototype asks the solver about every pair
of paths whose constraints one input can satisfy. It finds those pairs from
inputs: an input of a rewritten path names the original path it follows, and
the search ends when the solver shows the rewritten path has no other input.
That avoids a wrong pairing and costs a few queries per path.

The maturity is different. VeriBin has an evaluation on 86 patches. The
binary comparison here is a prototype that has finished on three firmware
functions and two functions from a prebuilt binary.

What VeriBin does that Patchestry does not:

- It works on any pair of binaries, including patches made by someone else
  with a compiler. Patchestry validates only its own patches.
- It gives a result without declarations.
- Its adaptive mode lets an analyst accept a semantically equivalent change
  found during the run. Patchestry would report it as a failure until a
  declaration is added and the run repeated.
- It has a published evaluation.

What Patchestry does that VeriBin does not:

- It checks the patcher: that the bytes in the image are the intended code,
  that the rewrite did not damage neighbours, and that the inserted code
  matches the image's ABI and keeps the stack and saved registers intact.
- It checks the order of device register writes.
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

Patchestry has no comparable evaluation. Its corpus is 3 firmware cases and
3 functions from prebuilt ARM binaries.

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
- `GettingStarted/identity_reinsertion.md`: the identity suite.
- `GettingStarted/qemu_firmware_runtime.md`: the firmware suite and its
  reports.
- `klee-integration.md`: the KLEE image and the KLEE gate.
