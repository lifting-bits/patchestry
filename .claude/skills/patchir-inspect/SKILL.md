---
name: patchir-inspect
description: >
  This skill should be used when the user asks to verify, debug, or
  generate regression tests for the patchir-decomp structuring pipeline.
  Operates in two modes: --debug verifies that --use-structuring-pass
  produces functionally equivalent C output compared to the goto-based
  baseline and performs JSON ground-truth audit. --test-gen extracts
  P-Code from a binary via Ghidra headless and generates a LIT test
  with FileCheck patterns.


  Trigger the skill when the user asks to "debug patchir output",
  "analyze decompilation issues", "find structuring bugs", "compare
  structuring output", "diff structured vs goto", "check structuring
  quality", "analyze structuring pass", "find structuring issues",
  "test structuring on JSON", "verify structuring correctness", "test
  patchir-decomp on JSON", "debug a fixture", "find missing statements",
  "audit patchir-decomp", "check ClangAST emission", "find emission
  bugs", "generate test", "create LIT test from binary", "generate
  regression test", or invokes "/patchir-inspect [--debug|--test-gen]
  <args>".
---

# patchir-inspect: Structuring Verification & Test Generation Tool

Two modes for the patchir-decomp pipeline:

- **Debug mode** (`--debug`): Verify that `--use-structuring-pass` produces
  functionally equivalent C output compared to the goto-based baseline.
  Audit both outputs against the input P-Code JSON to catch Clang AST
  emission bugs (missing calls, lost operations, dropped blocks).

- **Test-gen mode** (`--test-gen`): Extract P-Code JSON from a binary
  function via Ghidra headless, generate a complete LIT test file with
  FileCheck patterns, and validate it passes.

ARGUMENTS: `[--debug|--test-gen] <args>`

- `--debug <fixture.json>`: run equivalence checks on one JSON file (default mode).
- `--debug --batch`: run on ALL fixtures in `test/patchir-decomp/`.
- `--test-gen --binary <path> --function <name>`: generate LIT test from binary.
- No mode flag: defaults to `--debug`.

## Mode Dispatch

Parse the argument list for `--debug` or `--test-gen`. If neither is present,
default to `--debug`.

- **Debug mode:** Execute Steps 1-3 below (equivalence checks + JSON audit).
- **Test-gen mode:** Execute Step 4 below (binary extraction + LIT generation).

---

## Debug Mode

### Setup

1. Find the project root from the input file path or cwd.
2. Locate `patchir-decomp` binary (`builds/*/tools/patchir-decomp/*/patchir-decomp`)
   and `test/scripts/strip-json-comments.sh`.
3. Strip comment lines if the file starts with `//`:
```bash
bash <project>/test/scripts/strip-json-comments.sh <input.json> > /tmp/patchdbg_clean.json
```

### Step 1: Run Both Paths

```bash
<decomp> -input /tmp/patchdbg_clean.json -emit-cir -print-tu \
    -output /tmp/patchdbg_goto 2>/tmp/patchdbg_goto_err.txt
GOTO_EXIT=$?

<decomp> -input /tmp/patchdbg_clean.json -use-structuring-pass -emit-cir -print-tu \
    -output /tmp/patchdbg_struct 2>/tmp/patchdbg_struct_err.txt
STRUCT_EXIT=$?
```

If the structuring path crashes when the goto path succeeds, report as
**Critical: structuring pass crash** with stderr output.

### Step 2: Functional Equivalence Checks (Goto vs Structured)

Read both `.c` outputs and verify preservation of function signatures,
variable declarations, statements, control flow, call graph, and return values.

**Critical checks (2.5-2.8):**
- **2.5 Condition Preservation:** Count `if (` in both outputs. If structured has
  fewer, triage: check for negation (condition inversion) or merging (`&&`/`||`).
  If no negation/merge match -> **Critical: lost guard**.
- **2.6 Duplicate/Overwritten Assignments:** Two sub-checks:
  - **2.6a Consecutive:** Find consecutive writes to same lvalue in structured
    output without intervening condition, label, or brace. Cross-check goto output
    for guarded single-write -> **Bug: lost guard**.
    Skip typedef/struct/enum lines. Reset on labels (different control paths).
  - **2.6b Cross-scope overwrite:** For each `if (cond) { ... assigns X ... }` block
    in structured output, check if `X = expr;` appears **immediately after** the
    closing `}` (within 3 lines, ignoring blank/label lines). If so, the assignment
    after the if-block unconditionally overwrites values set conditionally inside.
    Cross-check goto output: if the overwriting assignment was goto-only (only
    reached via goto, not fallthrough), this is a **Critical: fallthrough overwrite**.
- **2.7 Dead Code (structuring-introduced):** Scan structured output for code made
  unreachable by the structuring pass. Detect breakless `while(1)` barriers and
  post-return dead code. Exclude label lines (goto targets remain reachable).
- **2.8 Fallthrough into Goto-Only Labels:** For each label `L:` in structured output:
  1. Check the line before `L:` -- if NOT a terminator (`return`, `goto`, `break`,
     `continue`, or `}` of a terminating block) AND NOT a conditional (`if`, `else`),
     then `L` has **unconditional fallthrough**.
  2. Check same label in goto output -- if the line before `L:` IS a terminator,
     then `L` was **goto-only** (never reached by fallthrough).
  3. If structured has fallthrough AND goto had no fallthrough ->
     **Critical: spurious fallthrough changes semantics**.
  Labels inside if/else bodies are NOT fallthrough -- they are conditionally guarded.

For full methodology (checks 2.1-2.10), consult
**`references/structuring-diff-checks.md`**.

### Step 3: JSON Ground-Truth Audit

Parse the input P-Code JSON and cross-reference against BOTH C outputs to
catch Clang AST emission bugs that affect goto and structured paths equally.

#### 3.1 Parse JSON Structure

```python
data = json.load(clean_json)
# Extract: functions (with basic_blocks, operations), globals, types
# Build maps: function_addr->name, global_addr->name
```

**Key JSON fields per operation:**
- `mnemonic`: CALL, STORE, CBRANCH, RETURN, INT_*, COPY, etc.
- `inputs`: operands with `kind` (parameter, global, temporary, local)
- `output`: result destination
- `type`: result type reference

#### 3.2 CALL Coverage Audit

Extract all CALL/CALLIND operations from JSON. Resolve target names via:
1. `inputs[0].global` -> lookup in `data["globals"]` for name
2. `inputs[0].operation` -> trace through COPY/ADDRESS_OF chains to find function address
3. Match function address against `data["functions"]` for name

For each resolved call target, verify `target_name(` appears in both C outputs.

**Flags:**
- `CALL_LOST_GOTO`: call in JSON but missing from goto output -> emission bug
- `CALL_LOST_STRUCT`: call in goto but missing from structured -> structuring bug
- `CALL_LOST_BOTH`: call in JSON but missing from both -> emission bug

#### 3.3 CBRANCH -> Condition Audit

Count CBRANCH operations in JSON. Each CBRANCH should produce a condition
in the C output (`if(`, `while(`, `switch(`, or merged via `&&`/`||`).

Compare: `json_cbranch_count` vs `c_if + c_while + c_switch + merge_delta`.

**Flags:**
- `COND_DEFICIT`: fewer total conditions than CBRANCHes (after accounting for merges)
- Note: some CBRANCHes become `switch` case routing (1 BRANCHIND -> N cases), so
  exact 1:1 mapping is not expected. Flag only large deficits (>30% fewer).

#### 3.4 STORE -> Assignment Audit

Count STORE operations in JSON. Count assignment statements (`lvalue = expr;`)
in C outputs, excluding `==` comparisons and type declarations.

Compare goto vs structured assignment counts. If structured has significantly
fewer assignments than goto -> possible lost STOREs from collapse.

**Flags:**
- `STORE_DEFICIT`: structured has >20% fewer assignments than goto

#### 3.5 RETURN Audit

Count RETURN operations in JSON. Compare with `return` statement count in
C outputs.

**Important:** Skip void functions -- P-Code RETURN with no output value
produces no `return;` in C. Check if the function's JSON has a non-void
return type before flagging.

**Flags:**
- `RET_DEFICIT`: fewer returns in C than non-void RETURNs in JSON

#### 3.6 Global Variable Coverage

Extract all global variable names from `data["globals"]`. For each, check
presence in both C outputs.

**Flags:**
- `GLOBAL_LOST`: global referenced in goto but missing from structured

#### 3.7 Per-Block Reachability

For each basic block with side-effect operations (CALL, STORE, RETURN),
check that its operations appear in the C output. Blocks with only
DECLARE/COPY/BRANCH ops may not have visible C output -- skip those.

**Flags:**
- `BLOCK_LOST`: block with CALL/STORE/RETURN operations has no trace in C output

#### 3.8 Duplicate Assignment Detection (enhanced)

Find consecutive writes to same lvalue in structured output. Reset tracking
on: `if`, `else`, `while`, `switch`, labels (`identifier:`), closing braces.

Cross-check goto output: if the goto output has the same consecutive pattern
-> false positive (pre-existing). If goto has a guard between them -> real bug.

**Flags:**
- `DUP_ASSIGN`: consecutive writes to same lvalue, goto has guard between them

### Batch Mode

With `--batch`, run on all JSON fixtures and present a summary table:

```
| Fixture                   | Status | G(goto) | G(struct) | Elim% | Conds | Fall | JSON | Flags   |
|---------------------------|--------|---------|-----------|-------|-------|------|------|---------|
| bloodview__parse_cli.json | PASS   | 23      | 3         | 86%   | FP:-1 | 0    | OK   |         |
| decode_frame.json         | PASS   | 15      | 5         | 66%   | OK    | 0    | OK   |         |
| decode_basic_field.json   | FLAG   | 97      | 70        | 27%   | OK    | 0    | FLAG | DUP:3   |
```

**Column definitions:**
- `Conds`: condition preservation (OK / FP:-N / LOST:N)
- `Fall`: fallthrough into goto-only labels (0 = clean, N>0 = **Critical**)
- `JSON`: JSON ground-truth audit result (OK / FLAG)
- `Flags`: specific flags from JSON audit (CALL_LOST, DUP_ASSIGN, etc.)

**Batch check 2.8 implementation (Fall column):**
For each fixture with gotos, run the fallthrough check:
1. Extract all labels from structured output (`grep -n '^\s*\w\+:'`)
2. For each label, check line before it -- is it a terminator?
3. Check same label in goto output -- was it goto-only?
4. Count labels that gained spurious fallthrough -> `Fall:N`
5. `Fall:N > 0` is **Critical** -- escalate to single-fixture analysis

**Batch triage:** When `JSON` shows `FLAG` or `Fall` > 0:
1. Run single-fixture mode on that fixture.
2. For each flag, determine: emission bug (both paths) vs structuring bug (struct only).
3. `DUP_ASSIGN` flags: verify by checking if goto output has a guard.
4. `Fall:N` flags: identify which label gained fallthrough and what code executes spuriously.

Print aggregate stats: total/pass/fail/flag, fully structured count, goto elimination rate,
condition-loss count, JSON audit flag count.

### Report Format

| Severity | Examples |
|----------|----------|
| **Critical** | Missing function calls (CALL_LOST), missing switch cases, crash, lost conditions, fallthrough into goto-only label (Fall>0), cross-scope overwrite (2.6b) |
| **Bug** | Lost assignments, duplicate unconditional assignments (DUP_ASSIGN), dead code from structuring |
| **Flag** | JSON audit warnings that need manual triage: COND_DEFICIT, STORE_DEFICIT, BLOCK_LOST |
| **Warning** | Remaining gotos, dead labels, redundant breaks, statement reordering |

End with:
- **VERDICT: PASS** -- no Critical/Bug, no unresolved Flags
- **VERDICT: FLAG** -- no Critical/Bug, but has Flags needing triage
- **VERDICT: FAIL** -- has Critical or Bug issues

---

## Test-Gen Mode

### Usage

```
/patchir-inspect --test-gen --binary /path/to/firmware.elf --function my_function
```

Generates a LIT test file at `test/patchir-decomp/<function_name>.json` from
a real binary function.

### Step 4.1: Ensure Docker Image

Check if the Ghidra headless Docker image is available:
```bash
docker image inspect trailofbits/patchestry-decompilation:latest > /dev/null 2>&1
```

If not found, build it automatically:
```bash
bash scripts/ghidra/build-headless-docker.sh
```

Report: "Building Ghidra headless Docker image (one-time setup)..."

If the build fails, report the error and stop.

### Step 4.2: Extract P-Code JSON

```bash
bash scripts/ghidra/decompile-headless.sh \
  --input <--binary value> --function <--function value> \
  --output /tmp/patchdbg_testgen
```

Produces `/tmp/patchdbg_testgen` (P-Code JSON).

If extraction fails, report error with stderr and stop.

### Step 4.3: Generate Baseline Outputs

Run patchir-decomp on the extracted JSON:
```bash
<decomp> -input /tmp/patchdbg_testgen \
  -use-structuring-pass -emit-cir -emit-llvm -print-tu \
  -output /tmp/patchdbg_testgen_out
```

Produces `.cir`, `.ll`, `.c` files for deriving FileCheck patterns.

### Step 4.4: Generate FileCheck Patterns

From the baseline outputs, extract:

1. **Function signature** (from `.cir`):
   ```
   // FN: cir.func @<function_name>(
   ```

2. **Call targets** (via `gen-call-checks.py`):
   ```bash
   python3 scripts/gen-call-checks.py /tmp/patchdbg_testgen > /tmp/call_checks.txt
   ```
   Produces `// CALL-CHECK-DAG: cir.call @target_name(` lines.

3. **CIR operation patterns** (from `.cir`, optional):
   - Key CIR operations: `cir.if`, `cir.scope`, `cir.return`
   - Type patterns: `cir.cast`, `cir.unary`, `cir.binop`

### Step 4.5: Assemble LIT Test File

Write `test/patchir-decomp/<function_name>.json` with this structure:

```
// RUN: bash %strip-json-comments %s > %t.json
// RUN: %patchir-decomp -input %t.json -use-structuring-pass -emit-cir -emit-llvm -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=FN %s --input-file %t.cir
// FN: cir.func @<function_name>(
// RUN: %gen-call-checks %t.json > %t.call.checks
// RUN: %file-check -check-prefix=CALL-CHECK %t.call.checks --input-file %t.cir
<raw P-Code JSON content>
```

The RUN/CHECK directives follow the existing test pattern
(see `bloodview__parse_cli.json`, `bool_ops.json`).

### Step 4.6: Validate

Run the generated test through LIT to confirm it passes:
```bash
lit builds/default/test/patchir-decomp/<function_name>.json -D BUILD_TYPE=Debug -v
```

If it fails, report the LIT output and keep the generated file for debugging.

### Step 4.7: Report

```
Generated: test/patchir-decomp/<function_name>.json
  Source:  <binary> :: <function_name>
  Blocks:  N basic blocks, M operations
  Checks:  FN (signature), CALL-CHECK (N call targets)
  LIT:     PASS
```

---

## Additional Resources

- **`references/structuring-diff-checks.md`** -- Equivalence checks 2.1-2.10 and metrics
- **`USAGE_PLAN.md`** -- Development workflow recipes for pre-commit gating, regression checking, and test generation
