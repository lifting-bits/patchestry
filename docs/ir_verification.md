# IR-Level Verification

Patchestry checks contracts on the patched LLVM IR, before compiling and
inserting the patch. This page explains how the contracts become checks for
KLEE or SeaHorn. See [Patch validation](patch_validation.md) for checks on the
rewritten binary and [Verifiable patching](verifiable_patching.md) for an
overview of both workflows.

## What is checked

A contract is a set of preconditions and postconditions attached to a call
site. The verifier assumes the preconditions before the call and checks the
postconditions after it returns. A complete run establishes that the
postconditions hold on the paths allowed by those assumptions and the harness.

Contract verification relies on an accurate lift and the harness model. It
does not establish that the contracts fully capture the vulnerability or
check the machine code that Patcherex2 later writes into the binary.
[Patch validation](patch_validation.md#what-the-comparison-establishes) checks
the rewrite and compares the original and rewritten functions after
reinsertion. Those checks produce separate evidence; a contract result does
not establish a binary-comparison result.

## From YAML to LLVM metadata

Contracts are declared alongside patches in the YAML spec. For example, this
fragment requires the first argument to be non-null (see
[Patch specifications](GettingStarted/patch_specifications.md) for a complete
specification):

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

Static contracts remain metadata until a verifier instruments them:

1. `patchir-transform` matches the call site and attaches the predicates to
   the operation as a `contract.static` attribute of the Contracts dialect.
2. `patchir-cir2llvm` serializes the attribute into `!static_contract`
   metadata on the corresponding LLVM call instruction.

The metadata is a tagged string:

```llvm
%r = call i32 @snprintf(ptr %dest, i32 %dest_size, ptr %format), !static_contract !0

!0 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}, {kind=alignment, target=Arg(0), align=4}, {kind=relation, target=Arg(1), relation=neq, value=0}], postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=32]}]"}
```

Both verifiers parse the tagged string and support these predicates:

| Predicate | Meaning |
| --- | --- |
| `nonnull` | the target is not null |
| `relation` | an argument compared with a constant: `eq`, `neq`, `lt`, `le`, `gt`, `ge` |
| `range` | an argument or the return value lies between `min` and `max` |
| `alignment` | a pointer is a multiple of `align` |

## KLEE

`patchir-klee-verifier` generates a harness for a named function in the patched
module. It performs the following steps:

1. links an optional model library (`--model-library`) that supplies bodies
   for external functions;
2. rewrites abort-like declarations (`abort`, `__assert_fail` and similar) to
   call `klee_abort`;
3. gives every remaining external function a stub that returns a symbolic
   value;
4. instruments each call that carries `!static_contract`: a precondition
   becomes `klee_assume`, a postcondition becomes a branch to `klee_abort`
   when it is false;
5. generates `main()`, which makes the globals and the arguments of the
   target function symbolic and calls it. A pointer argument points at a
   symbolic buffer of `--symbolic-ptr-size` bytes (256 by default);
6. writes the module with an x86_64 triple and data layout, because KLEE
   cannot execute 32-bit targets.

KLEE explores the harness and writes a test case and an `.abort.err` file for
a path that reaches `klee_abort`. The abort may come from a failed
postcondition or an abort already present in the program; inspect the report
to identify the cause.

To lift a function, apply its patch, and generate a harness:

```bash
patchir-decomp -input function.json -emit-cir -output function
patchir-transform function.cir --spec spec.yaml -o patched.cir
patchir-cir2llvm patched.cir -S -o patched.ll
patchir-klee-verifier patched.ll --target-function poke_process -o harness.bc
```

Then run the bitcode in the KLEE image. Use a KLEE image built with a compatible
LLVM version:

```bash
./scripts/klee/run-klee.sh --run-bitcode --input harness.bc --output ./results
```

The end-to-end test `test/patchir-klee-verifier/cwe476_e2e.json` checks harness
generation from Ghidra JSON. It writes textual IR for FileCheck; the example
above writes bitcode for execution.

Options for `patchir-klee-verifier`:

| Option | Meaning |
| --- | --- |
| `--target-function NAME` | the function to verify (required) |
| `-o FILE`, `-S` | output file; `-S` writes LLVM IR text instead of bitcode |
| `--model-library FILE` | LLVM IR or bitcode with models of external functions, linked before stubbing |
| `--symbolic-ptr-size BYTES` | size of the symbolic buffer behind a pointer argument (default 256) |
| `--klee-init-max-depth N` | how deep pointers inside globals are followed when they are initialized (default 2) |
| `--klee-init-array-expand-limit N` | arrays of aggregates longer than this are made symbolic as flat bytes (default 64) |
| `--strict-contracts` | exit non-zero when a predicate does not parse (on by default) |
| `-v` | verbose output |

`--strict-contracts` is enabled by default. If a predicate cannot be parsed,
the tool stops rather than generating a harness that omits part of the
contract.

The image, the retargeting to x86_64 and its limits are described in
[KLEE integration](klee-integration.md).

### The gate in CI

The KLEE step in `.github/workflows/ci.yml` generates a harness for every
`.ll` test in `test/patchir-klee-verifier/`, runs KLEE on it for 30 seconds,
and checks the result with `scripts/klee/klee_gate.py`. A test can declare
either of these expectations:

```llvm
; KLEE-EXPECT: no-errors
; KLEE-EXPECT: errors: abort
```

The gate reads KLEE's output directory and assigns a verdict:

| Verdict | Meaning |
| --- | --- |
| `failed` | at least one path ended in an error |
| `proven` | no error, the run was not cut short, and at least one path was kept |
| `incomplete` | no error, but the time limit was reached |
| `not-run` | the directory is missing or holds no result |

The gate then checks the test's expectation. `no-errors` requires at least one
test case and no error files; it can pass even when the run is incomplete.
`errors:` specifies the exact set of error kinds expected, such as `abort`
in `test000001.abort.err`. CI fails if a result does not match its expectation
or if no test was judged.

`klee_gate_abort.ll` deliberately contains a failing postcondition so the test
can confirm that the gate detects it. Tests without a `KLEE-EXPECT` comment
use the existing criterion of producing at least one test case. If the KLEE
image cannot be pulled, CI skips this step with a warning.

### What a proven run means

A `proven` result applies under the harness's assumptions:

- pointer arguments point at buffers of a fixed size, and nothing is known
  about memory beyond them;
- an external function without a model returns any value and writes nothing;
- preconditions are assumed, not checked. Whether callers establish them is a
  separate question;
- the module was retargeted to x86_64, so the layout of structures that hold
  pointers differs from the firmware's.

## SeaHorn

`patchir-seahorn-verifier` reads the same `!static_contract` metadata and
rewrites the module for SeaHorn: each precondition becomes a call to
`__VERIFIER_assume` before the contracted call, and each postcondition a call
to `__VERIFIER_assert` after it.

```bash
patchir-seahorn-verifier patched.ll -S -o instrumented.ll
```

Its options are `-o FILE`, `-S` (LLVM IR text instead of bitcode) and `-v`.
The tool does not generate a harness or stub external functions. Those must be
accounted for when preparing the module for SeaHorn.

`scripts/seahorn/build_seahorn_docker_image.sh` builds an image with SeaHorn
installed under `/opt/seahorn`, and `.github/workflows/seahorn-image.yml`
publishes it. CI does not run SeaHorn on the instrumented output, and the
repository has no test suite for this workflow. The instrumentation tool is
available, but its end-to-end behavior is not covered by the KLEE tests.

## Tests

`test/patchir-klee-verifier/` holds the harness-generation tests (`.ll`
files, each checked with FileCheck and, in CI, run under KLEE) and the
end-to-end tests that start from Ghidra JSON.

```bash
lit ./builds/default/test/patchir-klee-verifier -D BUILD_TYPE=Debug -v
python3 scripts/klee/test_klee_gate.py
```
