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
| `static` | the new code returns with the stack pointer and the callee-saved registers as it found them, to the address it was called from | `check_static.py` |

## What differs from the bare-metal checks

**Dynamically linked executables.** `check_bindings.py` rejects an image with a program interpreter unless `--allow-dynamic` is given. A fixed-address executable (`ET_EXEC`) still has its own code and data at fixed addresses. A shared object or position-independent executable (`ET_DYN`) stays unsupported.

**Imports.** The lifted code calls an imported function by name. The fixture records the address of its PLT stub. The check decodes the stub in the binary, computes the GOT slot it loads its target from, and requires the jump-slot relocation at that slot to name the same import. Only then is the name bound to the stub address (`plt-stub`). An import with no stub, or with the stub of another import, fails.

**Stripped images.** With no symbol table, a function's address and size come from the Ghidra address range in the fixture, and the names passed to the patch link are the checked addresses.

**ARM-mode code.** `check_trampoline.py` requires Thumb code unless told the mode. The mode here is the one the rewriter's analysis reports for the function. The entry branch is then an ARM `b`, and the applied-bytes check expects that encoding.

**A loaded ELF header.** These binaries load their own ELF header. A rewriter that adds a segment changes the header fields that locate and count the header tables. Those bytes may change; any other byte of the header may not. What the new header says is checked by the record check: same entry point, same original segments.

**ABI.** Patcherex2 compiles for `arm-linux-gnueabihf`, which is ARMv6 with VFP and the hard-float calling convention. The `httpd` and `cgibin` binaries are ARMv7 soft-float. Code built with the default links and loads, and goes wrong the first time a `double` crosses a call. The run reads the binary's build attributes, derives `-march` and `-mfloat-abi` from them, and the `abi` check compares the attributes of the compiled code with the binary's.

## Static checks on these binaries

The `static` check analyzes the new body in the rewritten binary and, for comparison, the original body the compiler of the firmware produced. Neither is trusted: both are read from the disassembly. On the four cases that reach it, both pass. A function it cannot follow ends the case as `rejected:not-run:<reason>`, which is not a pass.

| Function | Original code / frame | New code / frame |
| --- | --- | --- |
| `httpd` `FUN_0007ae8c` | 676 / 32 bytes | 764 / 24 bytes |
| `httpd` `FUN_00067c0c` | 1064 / 504 | 2118 / 696 |
| `httpd` `formGetUpnpLists` | 688 / 15432 | 878 / 15512 |
| `cgibin` `FUN_00026534` | 1156 / 72 | 3562 / 152 |

The sizes are measured, not judged: no budget is declared for these binaries. No copy of half a body or more was found elsewhere in either image.

## Equivalence of a patched version

`scripts/patch-runtime/check_equivalence.py` compares a patched version of a lifted function with the lifted original. `patchir-klee-verifier --compare-original` builds a product harness: one module with both versions, each with its own copy of every global, and a `main` that runs them on the same input. The runner compiles that harness for ARM and runs it under `qemu-system-arm` (machine `virt`, bare metal) on inputs derived from seeds. The modules keep their 32-bit ARM layout.

| Property | Holds when | Violations |
| --- | --- | --- |
| P1 | an input the patched version accepts is one the original accepts | `original-rejects`, `original-faults`, `original-aborts` |
| P3 | when both accept, they return the same value | `return-differs`, `patched-faults`, `patched-aborts` |

A version rejects an input when it calls a declared reject handler (`--reject-handler`) or returns a declared reject value (`--reject-return`). `--arg-buffer INDEX:SIZE` says an argument points at a buffer, which lifted code often passes as an integer; `--arg-range INDEX:LOW:HIGH` bounds an integer argument. These stand in for the patch specification's validation block, which is not designed yet.

A property is `failed` when a violation was seen, `tested` when none was seen and at least one input reached the comparison, and `not-run` otherwise. `tested` covers the inputs that were run. `proven` needs the same harness under KLEE.

The cases file gives `httpd` `FUN_0007ae8c` six patched versions, each its own row in `summary.tsv` and its own evidence. They are C written from the lifted C (`test/identity-reinsertion/patches/httpd_fun_0007ae8c.c`); none is reinserted, so their integrity checks are `not-run`.

| Variant | What it is | P1 | P3 |
| --- | --- | --- | --- |
| `rewritten` | the lifted C written out by hand, no patch | tested | tested |
| `limit-8` | rejects a path token of 8 bytes or more with a null pointer, declared | tested | tested |
| `limit-8-undeclared` | the same patch, reject value not declared | tested | failed |
| `widened` | limit 24 replacing limit 8 | failed | tested |
| `changed-return` | returns the buffer address plus one | tested | failed |
| `printed-c` | the C `-print-tu` printed, compiled as is | tested | failed |

`printed-c` pins a known defect: the printer writes `*(signed char *)param_2 + local_c` where the IR loads from `param_2 + local_c`, so the printed C is not the lifted function. The expectation changes when the printer is fixed.

Limits: memory writes (P2) and calls (P4) are not compared. An external function is a stub that returns the same value at the same call index on both sides; one that returns a pointer ends the path as inconclusive. An input on which a version never returns costs a 5-second timeout, and the run stops after 8 of them.

## Known rejection: `builtin_strncpy`

`cgibin` `FUN_00011708` is rejected with `unresolved-dependency`, and the cases file expects that. The function fills a local buffer with a string constant using inline instructions. Ghidra represents the idiom as a pseudo-call to an intrinsic named `builtin_strncpy`, and `patchir-decomp` emits it as a call to an external function of that name. Nothing in the image provides one, so the code to insert cannot be linked. The rejection is correct. The fix belongs in the lifter: lower Ghidra's `builtin_*` intrinsics to an inline copy. When that lands, the expected status of the case becomes `pass`.

## Known rejection: shared objects

`libwolfssl.so` `FUN_000c4214` is rejected with `unsupported-image` at the first check, and the cases file expects that. The file is a shared object (ELF type `ET_DYN`). Its load base is chosen at run time, while every check here compares absolute addresses and the inserted code is compiled `-fno-pic` against absolute addresses. Relocations the loader applies are not modelled either. The same holds for a position-independent executable. Supporting these images needs position-independent inserted code and relocation-aware checks; whether Patcherex2 can rewrite a shared object is not yet checked. A dynamically linked executable at a fixed address (`httpd`, `cgibin`) is supported.

## Outputs

Per case under the output directory: `function.json`, the lifted `.c`, `.cir` and `.ll`, `bindings.json`, `dependencies.json`, `abi.json`, `trampoline.json`, `patch_record.json`, `applied_bytes.json`, `bindings.rewritten.json`, `static.json`, `rewritten.elf`, and `validation.json`. `summary.tsv` has one row per case.

`validation.json` is the evidence `evidence.py` builds from those reports (see the Evidence section of `qemu_firmware_runtime.md`): a verdict per check (`proven`, `tested`, `failed`, `not-run`), the bounds, the assumptions by name, and the reports with their digests. `case.json` holds what the driver knows and no report does. On the four passing cases `integrity` is `proven` and the calling convention is `proven`; the budgets, the run, P1 to P4, the firmware checks and the re-lift are `not-run`, so the overall verdict is `not-run`. The driver fails a case whose status and evidence disagree.

## What a pass does not mean

- The reinserted function is not shown to behave like the original. The checks cover where the bytes are and what they link to, not what they compute. The lifted code is larger than the original in every case, and no equivalence check is run.
- The binary is not executed.
- Nothing that needs the patch specification's `validation` block is checked. That block is not designed yet; the evidence carries the assumption `validation-block-absent` and those checks are `not-run`.
- The instruction mode and the function extents are taken from the rewriter's analysis and from Ghidra. A stripped image has no symbol to confirm either.
- An object bound to an address outside every segment gets the verdict `outside-image` and is not judged. `cgibin` `FUN_00026534` has three (`DAT_00000004` to `DAT_00000006`): small integers Ghidra typed as addresses.
