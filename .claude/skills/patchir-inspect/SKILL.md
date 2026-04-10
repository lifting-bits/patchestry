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

- **`--debug`**: verify `--use-structuring-pass` produces functionally
  equivalent C output compared to the goto-based baseline, and audit
  both outputs against the input P-Code JSON for emission bugs.
- **`--test-gen`**: extract P-Code from a binary function via Ghidra
  headless and generate a complete LIT test with FileCheck patterns.

## Arguments

```
--debug <fixture.json>                        # equivalence checks (default)
--debug --batch                               # run on ALL fixtures in test/patchir-decomp/
--test-gen --binary <path> --function <name>  # generate LIT test from binary
```

When no mode flag is present, default to `--debug`.

## Mode Dispatch

Each mode runs inside its own isolated `general-purpose` subagent via the
Task tool. This keeps the main context clean — raw decomp outputs, batch
tables, and Docker logs never reach the parent conversation.

- `--debug` → spawn the **Debug Agent** (Task prompt defined below)
- `--test-gen` → spawn the **Test-Gen Agent** (Task prompt defined below)

After the subagent returns, relay its verdict and any findings back to
the user. Do not re-run the checks in the parent context.

---

## Debug Agent

When dispatched, spawn a `general-purpose` subagent using the Task tool
with the prompt template below. Substitute `<FIXTURE>` with the target
JSON (or the literal string `--batch` when batch mode is requested).

```
You are running the patchir-inspect debug-mode workflow against <FIXTURE>.

Project root: <cwd>
Binary: builds/*/tools/patchir-decomp/*/patchir-decomp
Strip script: test/scripts/strip-json-comments.sh
Reference methodology: .claude/skills/patchir-inspect/references/structuring-diff-checks.md

Workflow:

1. For each input fixture (one file or every JSON in test/patchir-decomp/
   when --batch), strip // comments with the strip script and run
   patchir-decomp twice — once without --use-structuring-pass (goto
   baseline) and once with it (structured output). Capture stderr.

2. Run the Step 2 functional equivalence checks on both .c outputs:
   - 2.5 Condition preservation (if/while/switch counts, &&/|| merge
     triage)
   - 2.6 Duplicate and cross-scope overwrite detection
   - 2.7 Structuring-introduced dead code
   - 2.8 Fallthrough into goto-only labels
   Full check definitions are in references/structuring-diff-checks.md.

3. Run the Step 3 JSON ground-truth audit on the input P-Code JSON
   vs both C outputs: CALL coverage, CBRANCH/condition, STORE/
   assignment, RETURN, globals, per-block reachability, duplicate
   assignment. Flag definitions are in the reference file.

4. For --batch, emit the summary table with columns Fixture, Status,
   G(goto), G(struct), Elim%, Conds, Fall, JSON, Flags. Aggregate
   stats: total, pass, flag, skip, fully-structured count, goto
   elimination %, condition losses, fallthrough count.

5. Classify findings:
   - Critical: CALL_LOST, missing switch cases, crash, lost conditions,
     Fall>0, 2.6b cross-scope overwrite
   - Bug: DUP_ASSIGN, lost assignments, structuring-introduced dead code
   - Flag: COND_DEFICIT, STORE_DEFICIT, BLOCK_LOST
   - Warning: remaining gotos, dead labels, redundant breaks

6. End with a single VERDICT line:
   - VERDICT: PASS — no Critical/Bug, no unresolved Flags
   - VERDICT: FLAG — Flags present, need triage
   - VERDICT: FAIL — Critical or Bug present

Report concisely. For --batch, include the full table but limit per-
fixture triage narrative to fixtures flagged as non-PASS. For a single
fixture, include any Critical/Bug findings with file:line references.
```

---

## Test-Gen Agent

When dispatched, spawn a `general-purpose` subagent using the Task tool
with the prompt template below. Substitute `<BINARY>` and `<FUNCTION>`
with the user-provided values.

```
You are running the patchir-inspect test-gen workflow for function
"<FUNCTION>" from binary "<BINARY>".

Project root: <cwd>
Decomp binary: builds/*/tools/patchir-decomp/*/patchir-decomp
Ghidra wrapper: scripts/ghidra/decompile-headless.sh
Docker build: scripts/ghidra/build-headless-docker.sh
Call-check generator: scripts/gen-call-checks.py

Workflow:

1. Ensure the Ghidra Docker image exists:
      docker image inspect trailofbits/patchestry-decompilation:latest
   If missing, run scripts/ghidra/build-headless-docker.sh and report
   "Building Ghidra headless Docker image (one-time setup)...". Stop
   on build failure and report stderr.

2. Extract P-Code JSON:
      bash scripts/ghidra/decompile-headless.sh \
        --input <BINARY> \
        --function <FUNCTION> \
        --output /tmp/patchdbg_testgen/<FUNCTION>.json
   Stop on extraction failure and report stderr.

3. Generate baseline outputs from the extracted JSON:
      <decomp> -input <extracted> -use-structuring-pass \
        -emit-cir -emit-llvm -print-tu -output <baseline>

4. Build FileCheck patterns from the baseline:
   - Function signature: grep `cir.func @<FUNCTION>(` from the .cir
     output and emit it as the FN check line.
   - Call targets: run
       python3 scripts/gen-call-checks.py <extracted.json>
     to produce `// CALL-CHECK-DAG: cir.call @<target>(` lines.

5. Assemble test/patchir-decomp/<FUNCTION>.json with this header,
   followed by the raw extracted JSON as-is:
       // RUN: bash %strip-json-comments %s > %t.json
       // RUN: %patchir-decomp -input %t.json -use-structuring-pass -emit-cir -emit-llvm -print-tu -output %t >> /dev/null 2>&1
       // RUN: %file-check -vv -check-prefix=FN %s --input-file %t.cir
       // FN: cir.func @<FUNCTION>(
       // RUN: %gen-call-checks %t.json > %t.call.checks
       // RUN: %file-check -check-prefix=CALL-CHECK %t.call.checks --input-file %t.cir
   Follow the pattern in existing fixtures like bloodview__parse_cli.json
   and bool_ops.json.

6. Validate the generated test:
      lit builds/default/test/patchir-decomp/<FUNCTION>.json \
          -D BUILD_TYPE=Debug -v
   On LIT failure, report the output and leave the file in place.

7. Report:
       Generated: test/patchir-decomp/<FUNCTION>.json
         Source:  <BINARY> :: <FUNCTION>
         Blocks:  N basic blocks, M operations
         Checks:  FN (signature), CALL-CHECK (K call targets)
         LIT:     PASS
```

---

## Additional Resources

### Reference Files

- **`references/structuring-diff-checks.md`** — full methodology for
  checks 2.1–2.10 and 3.1–3.7, flag definitions, triage rules, concrete
  bug examples
- **`references/usage-plan.md`** — development workflow recipes for
  pre-commit gating, regression checking, and test generation
