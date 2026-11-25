# QEMU Firmware Runtime Validation

This suite patches the [serial-qemu firmware](../../firmwares/serial-qemu/)
and runs it on QEMU's `lm3s6965evb` Cortex-M3 board. It uses checked-in
function fixtures, applies YAML/C patches, lowers them to LLVM IR, and
reinserts the compiled code with Patcherex2. It then checks the rewrite,
compares the serial output, and runs PatchSymDiff on the original and
rewritten functions.

The [runner](../../scripts/test-qemu-firmware-runtime.sh) manages dependencies
and invokes [qemu_firmware_runtime.py](../../scripts/patch-runtime/qemu_firmware_runtime.py).
It uses the repository's Patcherex2 fork with LLVM IR patch support.

## Run the suite

Complete the [Patchestry setup](build.md). The suite also needs
`qemu-system-arm`, `uv`, Python 3.11, CMake, Ninja, and an ELF-capable `ld.lld`.
Docker is needed only when refreshing Ghidra fixtures. On macOS, the runner
prepares the Keystone library automatically.

From the repository root, select your installed tools and run:

```sh
export PATCHESTRY_LLVM_PREFIX="${LLVM_INSTALL_PREFIX}/bin"
export PATCHESTRY_LD_LLD="$(command -v ld.lld)"
export PATCHESTRY_PYTHON="$(command -v python3.11)"
export QEMU_SYSTEM_ARM="$(command -v qemu-system-arm)"

scripts/test-qemu-firmware-runtime.sh --build-type Debug
```

`LLVM_INSTALL_PREFIX` must identify the patched LLVM/ClangIR installation.
If a tool is not on `PATH`, set its variable to the executable's full path.
Use `--output-dir` to choose where reports and artifacts are saved.

## Cases

Each case declares its expected outcome. Cases rejected by integrity checks
are not run under QEMU; a rejection counts as a successful test only when it
matches the declared reason.

| Cases | What they exercise | Expected outcome |
| --- | --- | --- |
| `before`, `after`, `replace`, `contract` | Patch placement and runtime contract insertion | Expected serial output; only declared `device-writes` differences |
| `short`, `pointer`, `handler`, `dispatch`, `switch` | A 4-byte target, stored function pointers, indirect calls, helpers, and switch tables | Expected serial output; only declared `device-writes` differences |
| `length`, `forward`, `send` | Patches that reject additional inputs | Binary comparison `proven` under the declared inputs and rejection rules |
| `tiny` | A target too small for the entry branch | Rejected for `span-outside-function` |
| `divide` | A missing compiler runtime helper | Rejected for `unresolved-dependency` |
| `state`, `initialized` | New writable globals in patch code | Rejected for `writable-state` |

PatchSymDiff executes callees unless a rejection rule requires a stub and
compares UART writes in order. The nine cases that add output must produce
exactly the declared `device-writes` difference. This does not prove the
change safe; QEMU separately checks the expected transcript. See
[Patch validation](../patch_validation.md) for the comparison criteria and
recorded results.

## Evidence

The runner writes `summary.md` and `summary.tsv`. Each case retains its
rewritten ELF, intermediate IR, check reports, and `qemu.log` when executed.
`baseline.log` records the unpatched firmware's output.

[evidence.py](../../scripts/patch-runtime/evidence.py) collects the reports
into `<case>/validation.json`, including cases that stop early. It records
verdicts, findings, measurements, assumptions, and source reports for the
integrity, static, runtime, and comparison groups.

Verdicts are `proven`, `tested`, `failed`, or `not-run`. Each group and case
receives its weakest verdict. Missing reports and incomplete comparisons
receive `not-run`. Comparisons with declared behavioral differences also
receive `not-run`, even when the test passes. A passing suite therefore does
not mean every validation check is proven.

For failures, start with the summary and the case's `validation.json`, then
inspect the referenced check report or `qemu.log`.
[collect_reports.py](../../scripts/patch-runtime/collect_reports.py) combines
reports from multiple runs into tables.

## Limitations

- Default fixtures are checked in, with hand-written target descriptions.
  A normal run does not regenerate them from the compiled firmware.
- The ABI check does not run in this suite. The patch compiler targets
  `arm-linux-gnueabihf`, while the firmware uses the soft-float Cortex-M3 ABI,
  so the integrity group remains `not-run`.
- Static analysis cannot follow every instruction sequence. The `switch`
  case explicitly expects `indirect-jump`; this does not count as a proof.
  Code-size and stack budgets are not declared by the suite.
- Patch code can use existing firmware globals, helpers, and read-only data.
  New writable globals and unavailable runtime dependencies are rejected.
- Reinsertion checks do not detect stored pointers or indirect branches into
  the middle of a replaced function. Applied-byte checks trust the compiler.

See [validation stages](../patch_validation.md#validation-stages) for the
individual checks and [comparison limitations](../patch_validation.md#limitations)
for the input bounds and modeling assumptions.

## Refresh fixtures

To regenerate fixtures through Ghidra, start Docker and run:

```sh
scripts/ghidra/build-headless-docker.sh
PATCHESTRY_RUNTIME_REFRESH_GHIDRA=1 scripts/test-qemu-firmware-runtime.sh --build-type Debug
```

This overwrites the selected fixtures. Set `PATCHESTRY_RUNTIME_FIXTURE_DIR`
to use a different fixture directory.
