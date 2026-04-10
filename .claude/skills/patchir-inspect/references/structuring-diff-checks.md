# Structuring-Diff: Functional Equivalence Checks

Detailed methodology for verifying that `--use-structuring-pass` produces
functionally equivalent C output compared to the goto-based baseline.

## Check 2.1: Function Signatures (must be identical)

For every function in the goto output, verify the structured output has:
- Same function name
- Same return type
- Same parameter list (names, types, order)
- Same number of functions

**Any difference is a Critical bug.**

## Check 2.2: Variable Declarations (must be identical)

For every local variable declaration in the goto output:
- Same variable name must exist in the structured output
- Same type
- Same initialization value (if present)

For every global/extern declaration:
- Must appear identically in both outputs

**Missing or changed declarations are a Bug.**

## Check 2.3: Statement Preservation (all operations must be present)

Extract all non-control-flow statements from both outputs. These include:
- Assignments: `x = expr;`
- Function calls: `func(args);`
- Return statements: `return expr;`

For each statement in the goto output, verify it appears in the structured
output. The statement may be in a different scope (inside a while/if/switch
instead of after a label) but it MUST exist.

**Methodology:**
1. From the goto output, collect all assignment and call statements (strip
   labels, gotos, and control-flow wrappers).
2. From the structured output, collect the same.
3. Diff the two sets. Statements present in goto but missing in structured
   are **lost code** (Bug). Statements present in structured but missing in
   goto are **phantom code** (Bug).

## Check 2.4: Control Flow Equivalence

The structured output replaces gotos with if/while/switch, but the reachable
paths must be the same:

- **Switch cases:** Count case arms (including default) in both outputs.
  Every case value in the goto path must appear in the structured path.
  Missing cases are a **Critical bug** (lost control flow).

- **Branch conditions:** For each `if (cond)` in the structured output,
  find the corresponding conditional branch in the goto output (same
  condition expression). Verify the then/else bodies contain the same
  statements as the taken/not-taken targets in the goto path.

- **Loop bodies:** For each `while`/`do-while` in the structured output,
  verify all statements from the loop body blocks (identified by back-edge
  targets in the goto path) appear inside the structured loop.

- **Unreachable code:** Check for statements after `break`/`return` in the
  structured output that don't appear in the goto output.

## Check 2.5: Condition Preservation (guards must not be lost)

Count all `if (` conditions in both outputs. If structured has FEWER conditions
than goto, a guard may have been lost — this is a potential **Critical bug**.

**Triage for false positives:**
Not all "lost" conditions are real bugs. These are expected transformations:
- **Condition inversion** (Pass 6): `if (!(x == 0))` → `if (x == 0)` with negated body.
  The original condition disappears, a new negated form appears. Net: count stays same or ±1.
- **Condition merging**: `if (a) { if (b) ... }` → `if (a && b) ...`. Two `if(` become one.
- **Short-circuit boolean**: `if (a) goto L; if (b) goto L;` → `if (a || b) goto L;`.

**How to distinguish real bugs from false positives:**
1. For each "lost" condition, search for its **negation** in the structured output.
   `if (!(x == 0))` lost but `if (x == 0)` gained → false positive (inversion).
2. Check if two conditions merged: `if (a)` and `if (b)` lost but `if (a && b)` or
   `if (a || b)` gained → false positive (merging).
3. If a condition has NO corresponding negation or merge → **real bug: lost guard**.

**Real bug example (from both-succs-in-body loop bug):**
- Goto: `if (bVar3) { buffer[i] = x ^ 32; } else { buffer[i] = x; }`
- Struct: `buffer[i] = x; buffer[i] = x ^ 32;` (guard gone, both execute)
- `if (bVar3)` has no negation or merge in struct → **Critical: lost guard**

## Check 2.6: Duplicate Unconditional Assignments

Detect when both branches of an if-else appear unconditionally (same lvalue
assigned twice without an intervening condition):

```c
// Bug: guard lost — both branches execute
buffer[uVar8] = bVar1;          // from else
buffer[uVar8] = bVar1 ^ 32U;   // from then
```

**Methodology:** Find consecutive assignments to the same lvalue in the
structured output. Cross-check: if the goto output has ONE guarded assignment
to that lvalue, the guard was lost.

**Avoid false positives:** Skip typedef/struct/enum declarations. Only match
actual assignment statements (`identifier = expr;` or `ptr[expr] = expr;`).

## Check 2.7: Dead Code from Structuring

In the goto baseline **all code is reachable** — every basic block is a
goto/label target. The structuring pass replaces gotos with if/while/switch.
If it does this incorrectly, previously-reachable code can become unreachable.
Any dead code in the structured output that was reachable in the goto output
is a structuring bug.

**Detection methodology:**

1. **Breakless `while(1)` barrier:** Find `while (1)` blocks in the structured
   output that contain NO `break` statement anywhere inside their body. Any
   non-label, non-brace code after such a block within the same scope is dead.
   - Parse brace depth to determine "same scope".
   - Label lines (matching `^\s*\w+:`) are excluded — they are goto targets
     and remain reachable even after a control-flow barrier.

2. **Post-return dead code:** Find `return` statements. Any non-label,
   non-brace lines after a `return` before the closing `}` at equal or lower
   indent are dead.
   - Again, exclude label lines (valid goto targets placed after returns).

**Counting:** Report the total number of dead (non-label, non-brace) lines
as `Dead:N` in batch mode.

**Real bug example (degenerate while(1) nesting):**
```c
while (1) {           // infinite, no break
    while (1) {       // 19 layers deep
        while (local_c < 8U) { ... }
    }
}
// DEAD: was reachable via goto in baseline
uVar2 = device__await_revision();
if (uVar2 ^ 1U & 255U == 0U) ...
```

**Severity:** `Dead:N > 0` is a **Bug** — the structuring pass made reachable
code unreachable, changing observable behavior.

## Check 2.8: Call Graph Preservation

Extract all function calls from both outputs. Every call in the goto output
must appear in the structured output with identical arguments. A missing
call is a **Critical bug** (lost side effect).

## Check 2.9: Return Value Preservation

Every `return expr;` in the goto output must have a corresponding `return`
in the structured output with the same expression. Different return values
on different paths must all be preserved.

## Check 2.10: Fallthrough into Goto-Only Labels

In the goto baseline, some labels are **only** reached via `goto` — never by
sequential fallthrough.  The structuring pass may incorrectly place such a
label at a fallthrough position, creating a spurious execution path that
changes observable behavior.

**Detection methodology:**

1. **In the structured output**, for each label `L:`, check the line immediately
   before `L:`. If it is NOT a terminator (`return`, `goto`, `break`, `continue`,
   or closing `}` of a block that itself terminates) AND NOT a conditional
   control-flow line (`if (`, `else`, `else {`), then `L` is reached by
   **unconditional fallthrough** in the structured output.
   Labels inside if/else bodies are NOT fallthrough — they are conditionally
   guarded, matching the original `if (cond) goto L;` pattern.

2. **In the goto baseline**, for the same label `L:`, check the line immediately
   before `L:`. If it IS a terminator (or another `goto L_other`), then `L` was
   **goto-only** in the baseline — it was never reached by fallthrough.

3. If `L` gained a new fallthrough path that didn't exist in the baseline →
   **Critical: spurious fallthrough changes semantics**.

**Why this is critical:**

When a goto-only label becomes a fallthrough target, code that should only
execute on the error/alternate path now also executes on the normal path.
This typically manifests as:
- Return value overwritten (success value clobbered by error value)
- Error handling code running on success path
- Side effects (function calls, writes) executing when they shouldn't

**Real bug example** (`bloodview__parse_cli`):
```c
// Goto baseline:                    // Structured output:
//   success path:                   //   } else {
//     uVar2 = 1U;                   //       uVar2 = 1U;
//     goto stack_check;             //   }
//   error_label:         ←goto-only //   error_label:      ←FALLTHROUGH!
//     uVar2 = 0U;                   //     uVar2 = 0U;    ← overwrites 1→0
//   stack_check:                    //   stack_check:
//     return uVar2;                 //     return uVar2;  ← always returns 0
```
The success path sets `uVar2=1U` then gotos `stack_check`, skipping
`error_label`. After structuring, the `else` branch falls through into
`error_label`, overwriting `uVar2` with `0U`.  The function always returns 0.

**Batch column:** `Fall:N` where N = number of labels with spurious fallthrough.
`Fall:0` means clean.

**Severity:** `Fall:N > 0` is **Critical** — the structuring pass changed
which code executes on which path, altering observable behavior.

## Check 3: JSON Ground-Truth Audit

Parse the input P-Code JSON and cross-reference against both C outputs
to catch Clang AST emission bugs independent of structuring.

### Check 3.1: CALL Coverage

Extract CALL/CALLIND operations from JSON. Resolve targets:
- `inputs[0].global` → `data["globals"][addr]["name"]`
- `inputs[0].operation` → trace COPY/ADDRESS_OF chain → function address
- Match address against `data["functions"][addr]["name"]`

Verify each resolved target appears as `name(` in both C outputs.

**Flags:** `CALL_LOST_GOTO`, `CALL_LOST_STRUCT`, `CALL_LOST_BOTH`

### Check 3.2: CBRANCH → Condition Coverage

Count CBRANCH ops in JSON. Compare with `if( + while( + switch(` in C,
adjusted for condition merges (`&&`/`||` delta). Large deficits (>30%)
are flagged.

**Flag:** `COND_DEFICIT`

### Check 3.3: STORE → Assignment Coverage

Count STORE ops in JSON. Count assignments in C (`lvalue = expr;`
excluding `==` and declarations). Flag if structured has >20% fewer
than goto.

**Flag:** `STORE_DEFICIT`

### Check 3.4: RETURN Audit

Count non-void RETURN ops in JSON (check function return type in
`data["types"]`). Compare with `return` statements in C. Void functions
produce no `return;` — skip these to avoid false positives.

**Flag:** `RET_DEFICIT`

### Check 3.5: Global Variable Coverage

Extract globals from `data["globals"]`. Each referenced global should
appear in both C outputs. Flag if present in goto but missing from
structured.

**Flag:** `GLOBAL_LOST`

### Check 3.6: Per-Block Reachability

For each block with side-effect ops (CALL, STORE, RETURN), verify its
operations appear in C output. Skip declaration-only blocks.

**Flag:** `BLOCK_LOST`

### Check 3.7: Duplicate Assignment (enhanced)

Find consecutive writes to same lvalue in structured output. Reset
tracking on: `if`, `else`, `while`, `switch`, labels, closing braces.
Cross-check goto output — if goto has same pattern, it's pre-existing
(not a structuring bug). If goto has a guard between them, it's a real bug.

**Flag:** `DUP_ASSIGN`

## Quantitative Metrics

Compute and present:

| Metric | JSON | Goto Path | Structured | Delta |
|--------|------|-----------|------------|-------|
| P-Code ops | | — | — | — |
| CALL ops | | calls found | calls found | |
| CBRANCH ops | | if() count | if()+while()+switch() | |
| STORE ops | | assignments | assignments | |
| RETURN ops | | return count | return count | |
| Gotos | — | | | |
| Labels | — | | | |

Also compute:
- Goto elimination rate: `(goto_gotos - struct_gotos) / goto_gotos * 100%`
- Fully structured: `struct_gotos == 0` (yes/no)
- JSON audit: OK / FLAG (with specific flag codes)
