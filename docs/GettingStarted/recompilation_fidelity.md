# Recompilation of ARM Linux Binaries

This suite lifts functions from prebuilt ARM Linux binaries, recompiles them,
and reinserts them without applying a patch. It checks the rewrite and uses
PatchSymDiff to compare behavior where inputs are declared. This helps
identify lifting, compilation, and reinsertion errors before testing a patch.

Unlike the [QEMU firmware suite](qemu_firmware_runtime.md), it covers stripped,
dynamically linked ARM Linux executables. It does not execute the rewritten
binary.

## Run the suite

Complete the [Patchestry setup](build.md). You also need `uv`, Python 3.11,
and an ELF-capable `ld.lld`. From the repository root, run:

```sh
export PATCHESTRY_LLVM_PREFIX="${LLVM_INSTALL_PREFIX}/bin"
export PATCHESTRY_PYTHON="$(command -v python3.11)"

scripts/test-recompilation-fidelity.sh --binary-root /path/to/test-binaries
```

`LLVM_INSTALL_PREFIX` identifies the patched LLVM/ClangIR installation.
The runner uses its `ld.lld` by default; set `PATCHESTRY_LD_LLD` to use another.
On macOS, run the QEMU suite once to prepare Keystone, or set
`PATCHESTRY_KEYSTONE_LIB_DIR` to a directory containing `libkeystone.dylib`.

Supply the binaries separately. The
[case list](../../test/recompilation-fidelity/test_binaries_cases.json)
selects functions from `httpd` and `cgibin`, with binary paths relative to
`--binary-root`. Each case pins the binary's SHA-256 digest to its P-Code
fixture; a mismatch is rejected.

The [runner](../../scripts/test-recompilation-fidelity.sh) accepts `--cases`
for another case list, `--only` to select a case, and `--output-dir` to choose
where reports and artifacts are saved.

## Checks

The [driver](../../scripts/patch-runtime/recompilation_fidelity.py) checks
bindings, dependencies, ABI compatibility, reinsertion, patch records, applied
bytes, and stack and register preservation. It stops at the first failed
check. Cases with a `compare` configuration also run PatchSymDiff under the
specified input bounds and callee models. See
[Patch validation](../patch_validation.md) for the individual checks.

For Linux binaries, the driver verifies imported functions through their PLT
stubs and relocations, derives compiler ABI settings from the image's build
attributes, and checks ARM-mode entry branches. For stripped functions,
addresses and extents come from the fixture. Rewriting must preserve the
entry point and original loaded segments.

## Recorded results

| Case | Compare results | Notes |
| --- | --- | --- |
| `httpd` — `FUN_0007ae8c` | `proven` | 3-byte buffers; offset argument 0–8. |
| `httpd` — `FUN_00067c0c` | `incomplete` | Variable arguments to `sprintf` and `printf` are unsupported. |
| `httpd` — `formGetUpnpLists` | Not run | No comparison inputs declared. |
| `cgibin` — `FUN_00026534` | `proven` | 4-byte buffer followed by `99;`, argument 2 bounded to 0–4, and two declared callees followed. |
| `cgibin` — modified separator control | `failed`, as expected | Changing `;` to `:` passes insertion checks but produces `register` and `calls` differences. |
| `cgibin` — `FUN_00011708` | Not run | Rejected for `unresolved-dependency`: the lift emits `builtin_strncpy`, which the image does not provide. |

## Reports and limitations

`summary.tsv` records each case's status. Each case retains the lifted IR,
rewritten ELF when produced, check reports, and `validation.json`. The evidence
records verdicts, input bounds, assumptions, and source reports. See
[Evidence](qemu_firmware_runtime.md#evidence) for how to read it.

A passing case means its results match its expectations. It does not mean all
checks are proven:

- Binary comparison is limited to declared inputs and models. Missing or
  incomplete comparisons receive `not-run` in the evidence.
- Runtime checks and undeclared code-size and stack budgets remain `not-run`,
  so the overall evidence verdict can be `not-run` even for a passing case.
- Only fixed-address executables (`ET_EXEC`) are supported. Shared objects
  and position-independent executables (`ET_DYN`) are rejected because their
  runtime load addresses and relocations are not modeled.
- Function extents and instruction modes depend on the fixture and rewriter
  analysis. References outside every image segment are reported as
  `outside-image` and are not validated.
