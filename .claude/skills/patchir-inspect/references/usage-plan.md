# patchir-inspect: Development Usage Plan

How to integrate `/patchir-inspect` into the patchestry development workflow
to catch issues early and prevent regressions.

---

## 1. During Rule Development

**Scenario:** Adding or modifying a CFGStructure rule (e.g., RuleBlockWhileDo).

**Workflow:**
```
1. Make the code change in CFGStructure.cpp
2. Build:  cmake --build builds/default --config Debug --target patchestry_ast
3. Pick a fixture that exercises the rule:
     /patchir-inspect --debug test/patchir-decomp/bloodview__parse_cli.json
4. Fix, rebuild, repeat until PASS
5. Run batch to check for regressions:
     /patchir-inspect --debug --batch
6. Commit only when all fixtures PASS
```

**What it catches:**
- Statement loss from incorrect node absorption
- Polarity inversions (succs[0]/succs[1] confusion)
- Missing break/return in emitted structured blocks

---

## 2. After ClangEmitter Changes

**Scenario:** Modified the C code emission in ClangEmitter (e.g., cleanup passes, goto elimination).

**Workflow:**
```
1. Make the change in ClangEmitter.cpp
2. Build
3. Verify complex fixtures:
     /patchir-inspect --debug test/patchir-decomp/encode_basic_field.json
     /patchir-inspect --debug test/patchir-decomp/decode_basic_field.json
4. Run batch to confirm no regressions:
     /patchir-inspect --debug --batch
5. Commit
```

**What it catches:**
- Statements silently dropped during cleanup passes
- Phantom code introduced by incorrect goto inlining
- Statement reordering that changes semantics

---

## 3. Pre-Commit Regression Gate

**Workflow:**
```
1. Build
2. Run lit tests:  lit ./builds/default/test/patchir-decomp -D BUILD_TYPE=Debug -v
3. Run batch verification:
     /patchir-inspect --debug --batch
4. Only commit if both lit and patchir-inspect pass
```

---

## 4. Debugging a Fixture Failure

**Scenario:** A specific JSON fixture produces wrong output.

**Workflow:**
```
1. Run verification:
     /patchir-inspect --debug test/patchir-decomp/failing_fixture.json
2. On FAIL, the tool automatically performs deep statement audit
   with P-Code cross-reference to pinpoint the lost/phantom statements
3. Fix, rebuild, re-run to confirm
```

---

## 5. Adding New Test Fixtures

**Workflow:**
```
1. Copy the JSON to test/patchir-decomp/new_fixture.json
2. Add // RUN: directives
3. Verify:  /patchir-inspect --debug test/patchir-decomp/new_fixture.json
4. If PASS: commit the fixture
5. If FAIL: investigate whether it's a new pattern the engine doesn't handle
```

---

## 6. Comparing Before/After a Refactor

**Workflow:**
```
1. Run batch BEFORE the refactor:
     /patchir-inspect --debug --batch
     (note: fully structured count, elimination rate)
2. Make the refactor, rebuild
3. Run batch AFTER:
     /patchir-inspect --debug --batch
4. Compare: same pass count, same or better elimination rate
5. Commit only if no regressions
```

---

## 7. Generating LIT Tests from Binaries

**Scenario:** Creating a patchir-decomp regression test from a real binary function.

**Prerequisites:** Docker must be installed. The Ghidra Docker image will be
built automatically on first use if not present.

**Workflow:**
```
1. Generate the LIT test:
     /patchir-inspect --test-gen --binary /path/to/firmware.elf --function parse_cli
2. Review the generated test at test/patchir-decomp/parse_cli.json
3. Run:  lit builds/default/test/patchir-decomp/parse_cli.json -D BUILD_TYPE=Debug -v
4. Commit the fixture if it passes
```

**What it does:**
- Extracts P-Code JSON from the binary via Ghidra headless (Docker)
- Runs patchir-decomp to generate baseline CIR/LLVM outputs
- Generates FileCheck patterns (function signature, call targets)
- Assembles a complete LIT test file with RUN/CHECK directives
- Validates the generated test passes

---

## Quick Reference

| Situation | Command |
|-----------|---------|
| Verify one fixture | `/patchir-inspect --debug <fixture.json>` |
| Regression check (all) | `/patchir-inspect --debug --batch` |
| Pre-commit gate | `lit ... && /patchir-inspect --debug --batch` |
| Generate LIT test | `/patchir-inspect --test-gen --binary <elf> --function <name>` |
