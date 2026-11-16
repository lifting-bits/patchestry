# Identity Reinsertion on ARM Linux Binaries

`scripts/test-identity-reinsertion.sh` lifts a function from a prebuilt binary, compiles the lifted code and puts it back in place of the original, with no patch applied. It then runs the reinsertion checks on the rewritten binary. Since nothing was meant to change, every difference between the two binaries is the work of the lift and the reinsertion. This is the baseline a real patch is judged against.

The serial-qemu runtime (`qemu_firmware_runtime.md`) covers bare-metal Thumb firmware built in this repository. This run covers what that one cannot: dynamically linked user-space executables, stripped of symbols, with ARM-mode code.

## Running

```sh
PATCHESTRY_LLVM_PREFIX=/path/to/llvm/bin \
scripts/test-identity-reinsertion.sh --binary-root /path/to/challenge/aarno_labs
```

The binaries are not in this repository. `--binary-root` is the directory the `binary` paths in the cases file are relative to. The default cases file, `test/identity-reinsertion/aarno_cases.json`, names functions of the `httpd`, `cgibin` and `libwolfssl` binaries and uses the P-Code fixtures already under `test/patchir-decomp/`. Each case pins the sha256 of the binary its fixture was lifted from; a different binary is an error, not a result.

On macOS the script needs the `libkeystone.dylib` that `scripts/test-qemu-firmware-runtime.sh` builds, so run that once first.

## Checks

Per case, in order. A case ends at the first check that fails, with the status `rejected:<problem>`.

| Check | What it establishes | Script |
| --- | --- | --- |
| `bindings` | every name the lifted code refers to is at the address Ghidra recovered; an import is reached through its own PLT stub | `check_bindings.py` |
| `dependencies` | the code to insert references nothing the binary does not provide and owns no writable state | `patch_dependencies.py` |
| `abi` | the code to insert is compiled for the CPU architecture and floating-point convention the binary declares | `check_abi.py` |
| `reinsertion` | only the function's bytes changed, its entry leads to the new body, nothing else branches into the overwritten bytes | `check_trampoline.py` |
| `link-addresses` | the rewriter linked each reference against the address the binding check confirmed | `patch_record.py` |
| `patch-record` | the record of what was written agrees with the image; entry point and original segments are unchanged | `patch_record.py` |
| `applied-bytes` | the new code equals a fresh compile and link of the same IR; no other loaded byte differs | `check_applied_bytes.py` |
| `rewritten-bindings` | in the rewritten binary every binding still gets the verdict it had | `check_bindings.py` |

## What differs from the bare-metal checks

**Dynamically linked executables.** `check_bindings.py` rejects an image with a program interpreter unless `--allow-dynamic` is given. A fixed-address executable (`ET_EXEC`) still has its own code and data at fixed addresses. A shared object or position-independent executable (`ET_DYN`) stays unsupported.

**Imports.** The lifted code calls an imported function by name. The fixture records the address of its PLT stub. The check decodes the stub in the binary, computes the GOT slot it loads its target from, and requires the jump-slot relocation at that slot to name the same import. Only then is the name bound to the stub address (`plt-stub`). An import with no stub, or with the stub of another import, fails.

**Stripped images.** With no symbol table, a function's address and size come from the Ghidra address range in the fixture, and the names passed to the patch link are the checked addresses.

**ARM-mode code.** `check_trampoline.py` requires Thumb code unless told the mode. The mode here is the one the rewriter's analysis reports for the function. The entry branch is then an ARM `b`, and the applied-bytes check expects that encoding.

**A loaded ELF header.** These binaries load their own ELF header. A rewriter that adds a segment changes the header fields that locate and count the header tables. Those bytes may change; any other byte of the header may not. What the new header says is checked by the record check: same entry point, same original segments.

**ABI.** Patcherex2 compiles for `arm-linux-gnueabihf`, which is ARMv6 with VFP and the hard-float calling convention. The `httpd` and `cgibin` binaries are ARMv7 soft-float. Code built with the default links and loads, and goes wrong the first time a `double` crosses a call. The run reads the binary's build attributes, derives `-march` and `-mfloat-abi` from them, and the `abi` check compares the attributes of the compiled code with the binary's.

## Outputs

Per case under the output directory: `function.json`, the lifted `.c`, `.cir` and `.ll`, `bindings.json`, `dependencies.json`, `abi.json`, `trampoline.json`, `patch_record.json`, `applied_bytes.json`, `bindings.rewritten.json`, `rewritten.elf`, and `validation.json`. `summary.tsv` has one row per case.

`validation.json` lists each check's verdict, the assumptions the verdicts rest on, and what the run does not establish.

## What a pass does not mean

- The reinserted function is not shown to behave like the original. The checks cover where the bytes are and what they link to, not what they compute. The lifted code is larger than the original in every case, and no equivalence check is run.
- The binary is not executed.
- Nothing that needs the patch specification's `validation` block is checked. That block is not designed yet, and `validation.json` records it as `todo`.
- The instruction mode and the function extents are taken from the rewriter's analysis and from Ghidra. A stripped image has no symbol to confirm either.
- An object bound to an address outside every segment gets the verdict `outside-image` and is not judged. `cgibin` `FUN_00026534` has three (`DAT_00000004` to `DAT_00000006`): small integers Ghidra typed as addresses.
