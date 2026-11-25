# Patch Validation

After reinsertion, Patchestry checks the binary rewrite and compares the
original and rewritten functions using PatchSymDiff. PatchSymDiff executes
their machine code symbolically with angr and uses cozy to compare the
resulting states.

This page describes the validation stages, results, and limitations.
[IR-level verification](ir_verification.md) covers contract checking before
compilation, while [Verifiable patching](verifiable_patching.md) explains how
the workflows connect.

## What the comparison establishes

The comparison checks whether a patch rejects additional inputs while
preserving the compared behavior for inputs that both versions accept. It
also checks that the patched function accepts no input rejected by the
original. Results apply to individual functions under the specified input
bounds and environment assumptions.

Contracts describe the properties required of the patch. Binary comparison
complements those checks but does not establish that the contracts fully
capture the vulnerability. Timing, whole-binary equivalence, and behavior
inside stubbed callees remain outside its scope.

## Validation approach

Validation uses three representations of the function:

1. The machine code in the original ELF.
2. The intended patched IR, produced by `patchir-transform`.
3. The machine code written into the rewritten ELF by Patcherex2.

Integrity checks verify that the rewritten ELF contains the intended code at
the expected addresses. Static analysis checks the inserted function's stack
and register use. These requirements apply even when the patch rejects an
input.

Binary comparison checks the original and rewritten machine code, allowing
the patch to reject additional inputs. It does not compare the lifted original
and patched functions at the IR level.

The evidence file records four verdicts. A group or case receives the weakest
verdict among its checks, ordered from `failed` through `not-run` and `tested`
to `proven`.

| Verdict | Meaning |
| --- | --- |
| `proven` | The check established its claim within the specified bounds and assumptions. |
| `tested` | The claim held for the inputs exercised by the test. |
| `failed` | The check found a problem. |
| `not-run` | The check did not run, could not reach a conclusion, or had no declared property to check. |

Missing reports receive `not-run`. The evidence also records input bounds,
callee models, rejection rules, and device address ranges. Regression tests
include deliberately broken patches to verify that the checks detect failures.

This evidence covers validation after reinsertion. KLEE and SeaHorn contract
verification produces separate results; those results are not included in
`validation.json`.

## Validation stages

The drivers run the applicable stages in the following order. A case stops
at the first integrity failure because subsequent checks require a valid
rewrite.

| Stage | Check | Script | Possible verdicts |
| --- | --- | --- | --- |
| 1. Bindings | Verify that referenced globals and functions exist in the image with the expected addresses, sizes, and kinds. | [check_bindings.py](../scripts/patch-runtime/check_bindings.py) | `proven`, `failed` |
| 2. Dependencies and ABI | Check that the image supplies the inserted code's dependencies and that the code matches its architecture, floating-point unit, and calling convention. | [patch_dependencies.py](../scripts/patch-runtime/patch_dependencies.py), [check_abi.py](../scripts/patch-runtime/check_abi.py) | `proven`, `failed` |
| 3. Reinsertion | Check that the rewrite stays within the function or branches to new code without overwriting neighboring code, splitting instructions, or leaving stale entries. | [check_trampoline.py](../scripts/patch-runtime/check_trampoline.py) | `proven`, `failed` |
| 4. Patch record | Record the bytes, addresses, compiler, and flags used by Patcherex2, and check the record against the rewritten ELF. | [patch_record.py](../scripts/patch-runtime/patch_record.py) | `proven`, `failed` |
| 5. Applied bytes | Recompile the intended snippets and compare the resulting bytes with the rewritten ELF. | [check_applied_bytes.py](../scripts/patch-runtime/check_applied_bytes.py) | `proven`, `failed` |
| 6. Static analysis | Check stack and saved-register preservation, along with any declared code-size and stack budgets. | [check_static.py](../scripts/patch-runtime/check_static.py) | `proven`, `failed`, `not-run` |
| 7. Runtime | Run the rewritten firmware under QEMU and check its output against the expected transcript. | [qemu_firmware_runtime.py](../scripts/patch-runtime/qemu_firmware_runtime.py) | `tested`, `failed` |
| 8. Binary comparison | Compare the original and rewritten functions under the declared inputs, rejection rules, and environment assumptions. | [patch_symdiff.py](../scripts/patch-runtime/patch_symdiff.py) | `proven`, `failed`, `incomplete` |
| 9. Evidence | Collect the reports, verdicts, and assumptions into one artifact. | [evidence.py](../scripts/patch-runtime/evidence.py) | Aggregated from the checks above. |

### Stages 1 to 6: integrity and static checks

These stages inspect the ELF files and patch record without executing the
firmware. Stage 5 recompiles the intended IR and compares the resulting bytes
with the rewritten image, relying on the compiler's correctness. Static
analysis reports `not-run` for checks it cannot complete or budgets that have
not been declared.

### Comparison properties: P1 to P4

PatchSymDiff checks four properties: existing input rejections, caller-visible
memory results, return values, and modeled external calls. Each property
compares the original and patched function using the same input and modeled
initial state, within the specified input bounds.

Each case defines rejection through an error return value, a reject handler,
a write to a declared fault address, or an omitted call listed in
`reject_skips`. A completed execution that meets none of these rules is
accepted. An unfinished or unsupported execution makes the comparison
incomplete.

| Property | Requirement | Reported difference |
| --- | --- | --- |
| P1: Preserve existing rejections | The patch must not accept an input that the original rejects. It may reject additional inputs. | `accepts-rejected-input` |
| P2: Preserve memory results | When both functions accept, caller-visible memory, including globals and output buffers, must contain the same final bytes. Temporary memory below the final stack pointer is excluded. | `memory` |
| P3: Preserve return values | When both functions accept, their return values must match. | `register` for r0, and r1 for a 64-bit result |
| P4: Preserve external calls | When both functions accept, modeled external calls must match in order and arguments. | `calls` for stubbed callees |

For example, a patch may add a guard that rejects oversized messages. For
messages that pass the guard, changing an output buffer violates P2, changing
the returned status violates P3, and removing a required external call
violates P4. Removing an existing rejection violates P1.

Calls to stubbed functions are recorded and compared directly. Other callees
are executed, and their effects are compared through memory and registers.
Writes to declared device ranges are compared in order and reported as
`device-writes`. This checks device effects without requiring the same
sequence of internal helper calls.

The comparison also checks behavior beyond these four properties. A
`register` difference can identify a failure to preserve r4 through r11 or sp.
An `unmatched-path` difference identifies a path whose inputs have no matching
path in the other function.

### Stage 8: binary comparison

[patch_symdiff.py](../scripts/patch-runtime/patch_symdiff.py) symbolically
executes the original and rewritten machine code with angr, then pairs and
compares their final states with cozy. Neither side uses the Ghidra or
`patchir-decomp` lift, so the comparison can detect behavioral errors
introduced during lifting. Reinserting a function without changing its IR
also allows this comparison to test the lifting and recompilation process.

Both functions start with the same symbolic arguments and argument buffers.
Globals contain their initial bytes from the ELF image. Registers r4 through
r11 each start with a symbolic value shared by both executions. Memory read
before it is written is initialized to zero. `--arg-range INDEX:LOW:HIGH`
bounds an integer argument, and `--arg-suffix INDEX:HEX` specifies the bytes
after an argument buffer. Those bytes default to zero.

Rejection rules tell PatchSymDiff when a function has rejected an input:

- `--reject-skip NAME`: omitting a call to this function counts as rejection
  when the remaining modeled calls match the original. The function is stubbed.

Separately, `--mmio-region LOW:HIGH` marks device addresses whose writes are
compared in order. A conditional write counts only when it executes.

For a read or write through an input-dependent address, PatchSymDiff explores
up to 256 possible addresses. If more addresses are possible, the comparison
is `incomplete`. Writes through input-dependent addresses are not recorded
as device writes; if such a write can reach a declared device or rejection
range, the comparison is `incomplete`. Unresolved jumps or calls, including
calls through unconstrained input-derived pointers, also make the comparison
incomplete.

Callees execute on both sides unless they are stubbed. A stub records the
call and returns an unknown value shared by both sides for the same callee
and call index. Stubs are assumed to leave memory unchanged. The suites use
different stubbing policies:

- The firmware suite stubs only callees named in `reject_skips`. Printed
  output is observed through writes to the UART.
- The recompilation-fidelity suite stubs functions referenced by the lifted
  code's bindings, except those listed in `follow_callees`. This limits
  execution of the surrounding image and handles imports without available
  code.

Stub argument and result registers are determined from declarations in the
compiled IR. These declarations and the function's return type are the only
information taken from the lift. A 64-bit integer occupies two registers.
Calls to stubs with unsupported parameters or results make the comparison
incomplete. These include floating-point values, aggregates, stack arguments,
and variable arguments to variadic functions.

For supported read-only string functions (`strncmp`, `strncasecmp`, `strcmp`,
`strcasecmp`, `memcmp`, `strlen`, and `strchr`), the stub computes its result
from memory when the path determines the pointers and required length. If a
call's result is computed on some paths but remains unknown on others,
reported differences are marked `unconfirmed` and the comparison is
`incomplete`. An unknown result can otherwise produce an apparent difference
between equivalent paths.

Addresses of local variables passed to stubs are excluded from comparison.
Pointers to read-only data are compared using the bytes they reference,
allowing the two images to use different layouts.

PatchSymDiff reports:

- `proven` when every path on both sides returns and no compared input
  produces a difference.
- `failed` when it finds a confirmed difference, with an input demonstrating
  that difference.
- `incomplete` when execution reaches a loop or time limit, encounters an
  error or unresolved target, exceeds the address exploration limit, or
  requires an unsupported model.

Because cozy pins an older angr version than Patcherex2, the suites run
PatchSymDiff in a separate process through
`patch_symdiff.compare_in_own_environment`. By default, `uv` builds this
environment from
[scripts/symdiff/requirements.txt](../scripts/symdiff/requirements.txt).
`PATCHESTRY_SYMDIFF_PYTHON` can select an interpreter with cozy already
installed. [scripts/symdiff/](../scripts/symdiff/) also provides a Docker image
for manual runs.

The recompilation-fidelity suite runs comparison for cases with a `compare`
configuration. This specifies buffers, buffer suffixes, argument ranges,
callees to follow, and a time limit. Results are written to
`patch_symdiff.json`. The evidence records them under `binary-comparison` as
`proven`, `failed`, or `not-run`. Incomplete comparisons and cases without a
comparison configuration receive `not-run`; failed comparisons fail the case.

The firmware suite compares every case that reaches this stage, using the
patch's declared rejection rules. Cases designed to change behavior must
declare the expected difference kind. The comparison must find that kind
and no others. These cases receive `not-run` for binary comparison in the
evidence because detecting an expected difference does not prove it safe.

### Stage 9: evidence

[evidence.py](../scripts/patch-runtime/evidence.py) writes
`<case>/validation.json`. It records each check's verdict, source report,
findings, measurements, and assumptions. It also assigns verdicts to the
integrity, static, runtime, and comparison groups.
[collect_reports.py](../scripts/patch-runtime/collect_reports.py) converts
these reports into tables.

## Recorded results

Results from the [firmware runtime suite](../scripts/test-qemu-firmware-runtime.sh)
and [recompilation-fidelity suite](../scripts/test-recompilation-fidelity.sh):

| Case | Compare results | Notes |
| --- | --- | --- |
| `serial-qemu` firmware (Cortex-M3) | 3 patches `proven`; 9 produced only declared `device-writes` differences | The 9 cases are recorded as `not-run` for comparison; QEMU checks their expected output. The `length` control without a rejection rule fails as expected. |
| `httpd` — `FUN_0007ae8c` | `proven` | Unchanged reinsertion; 3-byte buffers, offset argument 0–8. |
| `httpd` — `FUN_00067c0c` | `incomplete` | Variable arguments to `sprintf` and `printf` are unsupported. |
| `cgibin` — `FUN_00026534` | `proven` | Unchanged reinsertion; 4-byte buffer followed by `99;`, following `FUN_0002609c` and `FUN_0002619c`. |
| `cgibin` — modified control | `failed`, as expected | Changing `;` to `:` passes insertion checks but produces `register` and `calls` differences. |

## Limitations

Results depend on the input bounds and models used for each case:

- Declaring a behavioral difference does not prove it safe. For the nine
  firmware patches that declare `device-writes`, comparison checks the
  difference kind, while QEMU separately checks the expected transcript.
- Completed comparisons so far cover small functions and bounded inputs.
  One of the three prebuilt-binary cases remains incomplete, and another is
  proven only for 3-byte buffers. Paths are not merged, so larger input
  spaces can increase the number of paths and execution time.
- Stubbed callees are assumed to leave memory unchanged. Their internal
  behavior is not checked, and unsupported arguments or results, including
  stack and floating-point arguments, make the comparison incomplete.
- Memory read before it is written is initialized to zero. Results therefore
  depend on that initial-state assumption.
- Writes to device registers outside declared device ranges are compared as
  ordinary memory, without checking their order as device effects.
- Integrity and static checks rely on the intended IR and do not independently
  validate the lifter's interpretation of the original code. Stage 8 compares
  machine code independently, while still using IR declarations for stub
  signatures and the function's return type.

## Related documentation

- [Recompilation fidelity](GettingStarted/recompilation_fidelity.md): testing
  lifted and recompiled functions from prebuilt binaries.
- [QEMU firmware runtime](GettingStarted/qemu_firmware_runtime.md): running
  patched firmware and inspecting validation reports.
- [KLEE integration](klee-integration.md): the KLEE image and CI gate.
