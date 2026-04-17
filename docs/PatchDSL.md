# PatchDSL

A Semgrep-style pattern language for Patchestry, replacing the YAML
`meta_patches` / `match` / `action` spec. Patterns and rewrites are written in
C-shaped syntax, matched and rewritten at the ClangIR (CIR) layer.

Companion documents:
- `docs/GettingStarted/patch_specifications.md` — legacy YAML (still supported
  during migration).

---

## 1. File structure

A `.patch` file is a sequence of top-level items — metadata, imports,
inline patch functions, rules (patches), and contracts (full grammar in §2).

### 1.1 File-level metadata

```
metadata {
  name:        "cwe190-fixes"
  description: "Integer overflow fixes for bloodview"
  version:     "1.0.0"
  author:      "Security Team"
  created:     "2026-04-15"

  target {
    binary: "bloodview.bin"
    arch:   "ARM:LE:32:v7"
  }
}
```

Optional except for `target`, which is **required** when the rule set is
applied to a concrete CIR module. The `target` block mirrors the
`target:` section in legacy YAML (`docs/GettingStarted/patch_specifications.md`)
and pins the rule set to the binary / architecture it was authored against.

`target` fields:

| Field     | Required | Semantics                                                     |
|-----------|----------|---------------------------------------------------------------|
| `binary`  | yes      | Expected source binary name, matched against the CIR module's `patchir.source_binary` attribute. |
| `arch`    | yes      | Ghidra-style `PROCESSOR:ENDIAN:BITWIDTH:VARIANT` triple, matched against `patchir.target_arch`. |

### 1.1.1 Target verification

At rule load time, `patchir-transform` refuses to run a `.patchmod` whose
`metadata.target` is incompatible with the input CIR module. The check is:

1. **Binary name.** `metadata.target.binary` must equal the module's
   `patchir.source_binary` attribute (case-sensitive exact match). A
   mismatch is a hard error — the patcher exits non-zero and applies
   nothing.
2. **Architecture.** `metadata.target.arch` must equal the module's
   `patchir.target_arch`. `arch` is compared component-wise; a `*`
   in any position of the rule-side triple is a wildcard for that
   component (e.g. `"ARM:LE:32:*"` accepts any ARM32 little-endian
   target).
3. **Override.** `patchir-transform --ignore-target-mismatch` downgrades
   both checks to warnings. Intended for cross-binary experimentation;
   never the default.

When `metadata` is omitted entirely, or when the `target` block is
absent, the patcher emits a warning (`rule set has no target pin —
applying blind`) and proceeds. Authors are encouraged to always pin
`target` so an accidentally mis-routed rule set fails loudly instead of
silently corrupting unrelated code.

Verification happens *before* any pattern match is attempted, so the
error points at the rule file, not at a downstream rewrite failure:

```
rules/cwe190.patchmod: error: target mismatch
  rule set target: binary="bloodview.bin", arch="ARM:LE:32:v7"
  module target:   binary="device_query.bin", arch="ARM:LE:32:v7"
  (pass --ignore-target-mismatch to override)
```

### 1.2 Imports

Bring in an external C patch module whose functions can be invoked from
`rewrite:` bodies.

```
import "patches/patch_checked_mul16.c" as mul16
```

Symbols from the imported file are reachable as `mul16::<function>`.

### 1.3 Inline patch functions

For small helpers, skip the separate C file:

```
patch clamp_u16(x: uint32_t) -> uint16_t {
  return x > 0xFFFF ? 0xFFFF : (uint16_t)x;
}
```

Inline patches are compiled and linked exactly like `import`ed ones.

### 1.4 Rules and contracts

Two top-level block kinds carry the DSL's executable intent:

```
rule <name> { <clause>+ <action>+ }        // patches: rewrite / call / insert / remove
contract <name> { <clause>+ <contract-clause>+ }   // static contracts: requires / ensures / invariant / attributes
```

A `rule` *changes* program behavior — patching, hardening, or
instrumenting the binary by inserting, replacing, or removing code.
A `contract` *declares* a static property attached as an MLIR
attribute for verification tooling (KLEE, SARIF exporters, etc.) —
it emits no runtime code. See §5 for the contract surface in full.

> **Note.** Runtime validation calls (null checks, bounds guards,
> assertions inserted before or after a call site) are **rules**, not
> contracts. They insert executable code that hardens the binary at
> runtime, which is exactly what a `rule` with `insert before: call:`
> does. The `contract` block is reserved for *static* metadata that
> verification tools consume without touching the emitted binary.

---

## 2. Grammar (EBNF)

```ebnf
file          = { metadata | import | patch_fn | rule | contract } ;

metadata      = "metadata" "{" { kv | target_block } "}" ;
target_block  = "target" "{" { kv } "}" ;
import        = "import" string [ "as" ident ] ;
patch_fn      = "patch" ident "(" [ params ] ")" [ "->" type ] block ;
params        = param { "," param } ;
param         = ident ":" type ;

rule          = "rule" ident "{"
                    { clause }
                    action
                    { action }
                "}" ;

contract      = "contract" ident "{"
                    { clause | contract_clause }
                "}" ;

contract_clause
              = "requires"   ":" predicate
              | "ensures"    ":" predicate
              | "invariant"  ":" predicate
              | "attributes" ":" "[" ident { "," ident } "]" ;

clause        = pattern_clause
              | scope_clause
              | constraint_clause
              | predicate_clause
              | description_clause ;

pattern_clause
              = "pattern"                ":" code_block
              | "pattern-either"         ":" "[" code_block
                                              { "," code_block } "]" ;

scope_clause  = "pattern-inside"        ":" code_block ;

constraint_clause
              = "capture-pattern"        ":" "{" "var" ":" capture ","
                                              "pattern" ":" code_block "}"
              | "capture-comparison"     ":" "{" "var" ":" capture ","
                                              "cmp" ":" predicate "}"
              | "capture-taint"          ":" "{" "var" ":" capture ","
                                              "from" ":" source "}" ;

predicate_clause
              = "where" ":" predicate ;

description_clause
              = "description" ":" string
              | "id"          ":" string ;

action        = rewrite_action | insert_action ;
rewrite_action
              = "rewrite"         ":" code_block
              | "call"            ":" call_expr
              | "remove"
              | "assert"          ":" predicate ;

insert_action = "insert" position ":" insert_body ;
insert_body   = code_block
              | "call" ":" call_expr ;

position      = "before" | "after" | "at_entry" | "at_exit" ;
call_expr     = ident { "::" ident } "(" [ arg_list ] ")" ;
arg_list      = arg_expr { "," arg_expr } ;
arg_expr      = capture | literal | call_expr ;

code_block    = "|" raw_c_with_metavars               (* YAML-style block *)
              | string ;                              (* short form *)

predicate     = pred_atom
              | predicate "&&" predicate
              | predicate "||" predicate
              | "!" predicate
              | "(" predicate ")"
              | "forall" capture "in" expr ":" predicate
              | "exists" capture "in" expr ":" predicate ;

pred_atom     = "nonnull"    "(" expr ")"
              | "tainted"    "(" expr "from" source ")"
              | "reaches"    "(" expr "," expr ")"
              | "dominates"  "(" expr "," expr ")"
              | "aliases"    "(" expr "," expr ")"
              | "escapes"    "(" expr ")"
              | "sizeof"     "(" expr ")"  relop expr
              | "type"       "(" expr ")"  relop type
              | expr relop expr
              | ident "(" [ args ] ")" ;              (* user predicate *)

source        = "user_input" | "network" | "file"
              | ident ;                               (* user source *)

capture       = "$" ident | "$..." ident ;            (* $X or $...XS *)
ellipsis      = "..." ;                               (* in code_block only *)
```

### 2.1 Pattern vocabulary

Inside `code_block` you may use everything C accepts, plus:

| Token            | Meaning                                                 |
|------------------|---------------------------------------------------------|
| `$X`             | Capture. First use binds, later uses unify.             |
| `$...XS`         | Variadic capture. Captures an argument list.            |
| `...`            | Wildcard. Statement sequence, arg list, or expression.  |
| `$X: T`          | Inline type annotation on a capture.                    |

Positions where `...` makes sense:

```
f(..., $X, ...)           // any call to f with $X somewhere in the args
if (...) { ... }          // any if with any body
{ ...; free($P); ...; }   // free($P) anywhere in the block
$OBJ.$F                   // any field access on $OBJ
```

---

## 3. Clause reference

### 3.1 Matching clauses

| Clause                  | Semantics                                             |
|-------------------------|-------------------------------------------------------|
| `pattern:`              | Required. The shape the rule searches for.            |
| `pattern-either:`       | Disjunction; rule fires if any branch matches.        |

Multiple `pattern:` clauses are ANDed (the same AST region must match
each). Negative constraints belong in `where:` — see §3.4.

### 3.2 Scope clauses

| Clause                  | Semantics                                             |
|-------------------------|-------------------------------------------------------|
| `pattern-inside:`       | Match must lie textually/structurally inside this.    |

Scope clauses lower to `Op::getParentOfType` walks (for structural scopes
like `cir.func`, `cir.loop`, `cir.if`) and CFG-region checks for ordered
scopes (`between X and Y`).

### 3.3 Constraint clauses

| Clause                       | Purpose                                          |
|------------------------------|--------------------------------------------------|
| `capture-pattern:`           | Require a capture to itself match a sub-pattern. |
| `capture-comparison:`        | Require a numeric/ordering predicate on a var.   |
| `capture-taint:`             | Require a capture to be tainted from a source.   |

Static type refinements use the inline `$X: T` annotation in the
pattern itself (see §2.1) rather than a separate clause.

### 3.4 `where:` clauses

`where:` is the escape hatch for semantic predicates the pattern language
cannot express syntactically. Each predicate is backed by a CIR analysis:

| Predicate            | Backing analysis                                |
|----------------------|--------------------------------------------------|
| `nonnull(e)`         | Nullness lattice                                 |
| `e relop n`          | IntegerRangeAnalysis (e.g. `$A * $B <= 0xFFFF`)  |
| `sizeof(e) relop n`  | Type introspection (e.g. `$N <= sizeof($D)`)     |
| `type(e) relop T`    | Type introspection                               |
| `tainted(e from src)`| Taint dataflow                                   |
| `reaches(a, b)`      | Forward dataflow                                 |
| `dominates(a, b)`    | `DominanceInfo` on enclosing region              |
| `aliases(a, b)`      | MLIR alias analysis over CIR                     |
| `escapes(e)`         | Escape analysis                                  |

User-defined predicates are registered C++ callbacks; they receive the
capture environment and return `bool`.

---

## 4. Actions

Each rule has one or more actions. A rule may include any number of
`insert ...` actions, and at most one rewrite-style action
(`rewrite`, `call`, `remove`, or `assert`).

### 4.1 `rewrite:` — inline pattern-to-pattern rewrite

The matched region is replaced by the rewrite pattern, with captures
substituted. The rewrite uses the same pattern grammar as `pattern:`.

```
rule flip_strict_lt {
  pattern: for (...; $I < $N; ...) { ... $ARR[$I + 1] ... }
  rewrite: for (...; $I < $N - 1; ...) { ... $ARR[$I + 1] ... }
}
```

Ellipses on both sides are preserved literally — captured statements
between anchors keep their original position.

### 4.2 `insert before:` / `insert after:`

Adds new statements immediately before or after the match, without
touching the match itself.

```
rule null_after_free {
  pattern: free($P)
  insert after: |
    $P = NULL;
}
```

```
rule guard_before_write {
  pattern: *$P = $V
  where:   !nonnull($P)
  insert before: |
    if ($P == NULL) return;
}
```

### 4.3 `insert at_entry:` / `insert at_exit:`

Inserts at the function-level entry or every exit point of the enclosing
`cir.func`. Useful for contracts-as-code or resource tracking.

```
rule log_entry_exit {
  pattern-inside: $RET $FN(...) { ... }
  pattern:        return $X;
  insert at_entry: |
    trace::enter(#FN);
  insert at_exit: |
    trace::leave(#FN);
}
```

`#FN` is a stringified capture — a compile-time literal, not an SSA
value.

### 4.4 `call:` — invoke an external/inline patch function

Equivalent to `rewrite: $R = <callee>($ARGS...)` when the entire match
result is replaced by a call.

```
import "patches/patch_checked_mul16.c" as mul16

rule cwe190_int_overflow_fix {
  pattern-inside: $RET peek_process(...) { ... }
  pattern:        ($R: uint16_t) = $A * $B
  where:          !($A * $B <= 0xFFFF)
  call:           mul16::patch__replace__int_mul16($A, $B)
}
```

`call:` replaces the matched expression with the call's result, inserting
casts on operands and return value when the callee signature differs from
the capture types.

For `insert before` / `insert after` style call patches, use:

```
rule audit_sprintf {
  pattern: sprintf($D, $F, $...REST)
  insert before:
    call: audit::check_format($D, $F)
}
```

i.e. insert a normal call statement before/after the match.

### 4.5 `remove:` — delete the match

No body. Erases the matched op(s). Combine with `insert` if you need to add
new behavior at the same site.

```
rule drop_redundant_flush {
  pattern: |
    flush($F);
    ...
    flush($F);
  where: !modified_between($F, flush($F), flush($F))
  remove: |
    flush($F);       // second occurrence
}
```

The second occurrence is the one deleted; the ellipsis marks the anchor
pair.

### 4.6 `assert:` — site-local check

Emits a `cir.call @__patchestry_assume` (or runtime assertion, per build
config) at the matched site — useful for preconditions bound to a
*specific* call or op:

```
rule sprintf_precond {
  pattern: sprintf($D, $F, $...REST)
  assert: nonnull($D) && sizeof($D) >= 32 && nonnull($F)
}
```

`assert:` shares the predicate vocabulary with `where:` but is *emitted*
rather than *checked at match time*. It is a runtime action — it
inserts executable code at the matched site, just like `insert before:
call:`. For *static* properties (pre/post-conditions, invariants,
attribute tags consumed by verifiers), use a `contract` block — see §5.

---

## 5. Contracts (static only)

Contracts attach *static* properties to functions or regions without
rewriting their bodies. They lower exclusively to MLIR attributes —
zero runtime cost, consumed by verification tools (KLEE, SARIF
exporters, MLIR dataflow passes).

> **Note.** Runtime validation calls — null checks, bounds guards,
> assertion calls inserted at a call site — are **rules**, not
> contracts. Inserting executable code that hardens the binary at
> runtime is what `rule` with `insert before: call:` / `insert after:
> call:` does. A `contract` block never emits runtime code; it only
> attaches metadata for static analysis.

### 5.1 Block shape

```
contract <name> {
  pattern-inside: <function-or-region scope>

  requires:   <predicate>          // entry precondition
  ensures:    <predicate>          // return postcondition
  invariant:  <predicate>          // loop / region invariant
  attributes: [ <attr>, ... ]      // raw MLIR attrs
}
```

All clauses are optional; a contract block with only `attributes:` is
valid and useful for tagging functions as `pure`, `thread_safe`, etc.
At least one `pattern-inside:` clause is required so the contract has a
scope to attach to.

### 5.2 Clause reference

| Clause        | Scope target                              | Lowering                                          |
|---------------|-------------------------------------------|---------------------------------------------------|
| `requires:`   | enclosing `cir.func`                      | `patchestry.requires = #pred` on the func         |
| `ensures:`    | enclosing `cir.func`                      | `patchestry.ensures = #pred` on the func          |
| `invariant:`  | matched `cir.for` / `cir.while` / region  | `patchestry.invariant = #pred` on the region      |
| `attributes:` | enclosing `cir.func`                      | each attr attached as `patchestry.<attr>`         |

`requires:` / `ensures:` / `invariant:` use the same predicate
vocabulary as `where:` (§3.4) and `assert:` (§4.6).

### 5.3 Rules vs. contracts

| Need                                                    | Use                            |
|---------------------------------------------------------|--------------------------------|
| Runtime null check, bounds guard, or hardening call     | `rule` with `insert before/after: call:` |
| Precondition for a specific call-site (runtime check)   | `assert:` in a `rule`          |
| Function-level entry/exit property (static metadata)    | `requires:` / `ensures:` in a `contract` |
| Loop or region invariant (static metadata)              | `invariant:` in a `contract`   |
| Framework tag (`pure`, `noreturn`, …)                   | `attributes:` in a `contract`  |

### 5.4 Example — `peek_process` safety contract

```
contract peek_process_safety {
  pattern-inside: $RET peek_process(($CNT: uint16_t), ($SZ: uint16_t)) { ... }

  requires:   $CNT > 0 && $CNT * $SZ <= 0xFFFF
  ensures:    @return != NULL
  invariant:  $IDX <= $SZ
  attributes: [pure, no_side_effects]
}
```

The function picks up:

```mlir
cir.func @peek_process(...) attributes {
  patchestry.requires   = #patchestry<"$CNT > 0 && $CNT * $SZ <= 0xFFFF">,
  patchestry.ensures    = #patchestry<"@return != NULL">,
  patchestry.pure       = unit,
  patchestry.no_side_effects = unit
} { ... }
```

A verifier (KLEE, an MLIR dataflow pass, a SARIF exporter) consumes
those attributes directly. No runtime code is emitted.

### 5.5 Postcondition pseudo-capture `@return`

Inside `ensures:` only, `@return` stands for the value being returned at
each `cir.return` site. It is *not* a normal capture (note the `@`
prefix, not `$`) — it refers to a synthesized SSA value resolved
per-return-site, not to a match binding from `pattern:`.

---

## 6. Worked examples — from the test corpus

Examples include both full YAML→PatchDSL ports and PatchDSL-only snippets for
features that are awkward to express in legacy YAML.

### 6.1 Replace: CWE-190 integer overflow (`test/patchir-transform/cwe190_replace.yaml`)

**Legacy:**

```yaml
match:
  - name: "cir.binop"
    kind: "operation"
    function_context:
      - name: "peek_process"
action:
  - mode: "replace"
    patch_id: "cwe190_checked_mul16"
    arguments:
      - { name: "count",      source: "operand", index: 0 }
      - { name: "block_size", source: "operand", index: 1 }
```

**PatchDSL:**

```
import "patches/patch_checked_mul16.c" as mul16

rule cwe190_int_overflow_fix {
  description: "Replace unchecked u16 multiply in peek_process"

  pattern-inside: $RET peek_process(...) { ... }
  pattern:        ($R: uint16_t) = ($A: uint16_t) * ($B: uint16_t)

  where: !($A * $B <= 0xFFFF)

  call:  mul16::patch__replace__int_mul16($A, $B)
}
```

### 6.2 Replace (inline, no C file): CWE-476 null guard

**PatchDSL:**

```
rule cwe476_guard_deref {
  pattern: $P->$F
  where:   !nonnull($P)

  rewrite: ($P ? $P->$F : 0)
}
```

### 6.3 Insert before: audit a call-site

**Legacy (`measurement_update_before_operation.yaml`):**

```yaml
match:
  - name: "cir.call"
    kind: "operation"
    function_context:
      - name: "/.*measurement.*/"
      - name: "/.*update.*/"
    symbol_matches:
      - name: "spo2_lookup"
action:
  - mode: "apply_before"
    patch_id: "measurement_update_before_operation"
    arguments:
      - { name: "function_argument", source: "operand", index: 0 }
```

**PatchDSL:**

```
import "patches/measurement_update_patch.c" as mu

rule audit_spo2_lookup {
  pattern-inside: |
    $RET $FN(...) { ... }
  capture-comparison: { var: $FN,
                        cmp: matches("measurement") && matches("update") }

  pattern: spo2_lookup(($X: float))

  insert before:
    call: mu::measurement_update_before_operation($X)
}
```

### 6.4 Insert after: null-after-free

**PatchDSL:**

```
rule null_after_free {
  pattern-inside: $RET $FN(...) { ... }
  pattern:        free($P)
  where:          !nulled_after($P, free($P))

  insert after: |
    $P = NULL;
}
```

`nulled_after` is a user-defined dataflow predicate that returns true if
a `$P = NULL` store already post-dominates the `free($P)` site. It
prevents double-fixing without a pattern-level negation clause.

### 6.5 Rewrite: double-free

**PatchDSL:**

```
rule cwe415_double_free {
  pattern: |
    free($P);
    ...
    free($P);
  where: aliases($P, $P)           // trivially true, kept for clarity
      && !reassigned_between($P, free($P), free($P))

  rewrite: |
    free($P);
    $P = NULL;
    ...
}
```

The rewrite replaces the *entire* matched region. The ellipsis between
the two frees is preserved; the second `free($P)` is dropped;
`$P = NULL` is inserted after the first.

### 6.6 Replace call: CWE-078 command injection

**PatchDSL:**

```
import "patches/patch_system.c" as patch_sys

rule cwe078_system_taint {
  pattern: system($CMD)
  capture-taint: { var: $CMD, from: user_input }

  call: patch_sys::safe_exec($CMD)
}
```

### 6.7 Assert (contract): sprintf preconditions

**PatchDSL:**

```
rule sprintf_precond {
  pattern: sprintf($D, $F, $...REST)
  assert: nonnull($D) && sizeof($D) >= 32 && nonnull($F)
}
```

### 6.8 Pattern-either: multiple unsafe string variants

**PatchDSL:**

```
rule unsafe_str_copy {
  pattern-either: [
    strcpy($D, $S),
    strcat($D, $S),
    sprintf($D, "%s", $S)
  ]
  where: !(strlen($S) <= sizeof($D))

  call: safe::copy($D, sizeof($D), $S)
}
```

### 6.9 Insert at entry/exit: tracing

**PatchDSL:**

```
rule trace_peek_handlers {
  pattern-inside: $RET peek_process(...) { ... }
  pattern:        $RET peek_process(...) { ... }

  insert at_entry: |
    trace::enter("peek_process");
  insert at_exit: |
    trace::leave("peek_process");
}
```

---

## 7. Execution model

A `.patch` file travels through five phases before the patched `.cir`
is written. Each phase has a well-defined input, output, and failure
mode.

```
  [.patch source] → (1) Load → (2) Compile → (3) Match → (4) Rewrite → (5) Emit → [patched .cir]
                                   │                           │
                             external .c files          contract attrs / asserts
                             compiled to CIR            attached / inserted
                             (cached as .patchmod)
```

### 7.1 Load

1. **Parse.** The `.patch` file is tokenized with capture (`$X`),
   variadic-capture (`$...XS`), and ellipsis (`...`) extensions to a
   C-shaped grammar; the rest is ordinary C syntax.
2. **Verify target.** `metadata.target.binary` / `metadata.target.arch`
   are matched against the input module's `patchir.source_binary` /
   `patchir.target_arch` attributes (see §1.1.1). Mismatches abort the
   pass; no rewrite is attempted.
3. **Classify.** Each top-level block is sorted into one of three
   buckets: **imports**, **rules** (patches), and **contracts**.
4. **Type-check.** Every capture used in an action or predicate must be
   bound by some `pattern:` or `pattern-inside:` clause; every
   predicate must use an atom in §3.4's vocabulary (or a registered
   user predicate); every inline `$X: T` must resolve `T` in the CIR
   type system.

**Failure mode:** parse / target / type errors are hard — the pass
exits non-zero and writes no output.

### 7.2 Compile

1. **External C patches.** Each `import "patches/foo.c" as ns` is
   compiled once (via the project's vendored `clang`) to a CIR module
   and merged into a hidden pattern-library namespace. The DSL does not
   invoke a full C toolchain per call-site — the compiled object is
   cached under `build/<hash>.patchmod`.
2. **Inline `patch fn(...)` helpers** are lowered through the same
   path as external C files, as if they were written to a temporary
   `.c` file and imported.
3. **Rule / contract codegen.** Patterns and action bodies are lowered
   to MLIR rewrite patterns — the precise representation is an
   implementation detail; as an author, assume the DSL compiles
   patterns to "something MLIR's rewrite driver can execute."

**Failure mode:** a C compile error in an imported file is reported
with the original C source span, not the `.patch` site. Pattern codegen
errors point at the offending clause.

### 7.3 Match

For every `cir.func` in the module, the rewrite driver walks each rule
and contract in the order:

1. `pattern-inside:` clauses are evaluated first (cheaply — structural
   ancestor check) and prune most candidates.
2. `pattern:` / `pattern-either:` then walk the remaining operations;
   captures are bound on first use and unified (SSA equality) on later
   uses.
3. `capture-pattern:` / `capture-comparison:` / `capture-taint:`
   refine the candidate set.
4. `where:` predicates are evaluated **last** and **lazily**,
   short-circuiting on `&&` / `||`. This is where the analyses are
   consulted — nullness, taint, integer range, alias, escape,
   dominance, etc. Analyses are requested via the MLIR analysis
   manager and cached per function, so a predicate used across many
   rules pays the analysis cost once.

**Failure mode:** a rule that fails to match is silent — matching is
not an error. Rules that never match across the whole module produce
a *warning* at the end of the pass (`warning: rule foo matched 0
sites`) so unused rules are easy to spot.

### 7.4 Rewrite (rules)

Once a rule's clauses all pass, its single rewrite-style action
(`rewrite:` / `call:` / `remove:`) and any number of `insert …` actions
are applied to the match.

Ordering:

- **Within one rule:** actions apply in source order. `insert` actions
  run before the rewrite-style action, so an inserted before-call can
  read values that the rewrite will later erase.
- **Within one `.patch` file:** rules are considered in declaration
  order. Two rules matching the same operation are applied first-wins;
  the second rule's match is re-evaluated after the first rewrite and
  may simply no longer apply.
- **Across files** (`-dsl a.patchmod -dsl b.patchmod`): files are
  processed in command-line order.
- **Fixed point.** The whole pattern set is driven by
  `applyPatternsAndFoldGreedily`, so a rewrite that exposes a new
  match for another rule gets picked up in a later iteration. Rules
  must be idempotent or rely on `where:` / capture constraints to
  avoid rematching their own output.

**Failure mode:** an emit-time failure (e.g., the rewrite references a
capture that doesn't dominate the insertion point — see §8
Invariants) *rolls back* that rule's changes for the site and emits a
warning. Other rules continue.

### 7.5 Emit (contracts — static only)

Contracts are processed *after* all rules reach a fixed point, so
contracts attach to the final, patched symbols rather than their
pre-patch forms.

Each contract clause lowers to an MLIR attribute on the matched
function or region:

| Clause        | Lowering                                                            |
|---------------|---------------------------------------------------------------------|
| `requires:`   | `patchestry.requires = #pred` on the enclosing `cir.func`          |
| `ensures:`    | `patchestry.ensures = #pred` on the enclosing `cir.func`           |
| `invariant:`  | `patchestry.invariant = #pred` on the `cir.for`/`cir.while`/region |
| `attributes:` | each attr attached as `patchestry.<attr>` on the func              |

No runtime code is emitted by a `contract` block. Runtime hardening
(null checks, bounds guards, assertion calls) belongs in `rule`
blocks with `insert before/after: call:` or `assert:`.

**Failure mode:** an emit-time failure inside a contract (e.g.,
`@return` referenced from `requires:`, which has no return value) is
rejected at type-check (§7.1) rather than at emit, so this phase never
fails at runtime.

### 7.6 Analyses

Predicates in `where:` / `requires:` / `ensures:` / `invariant:` are
backed by the following MLIR analyses, each requested via the analysis
manager and cached per function:

- `DominanceInfo` — `dominates(a, b)`, dominance of rewrite insertion
  points.
- `patchestry.NullnessLattice` — `nonnull(e)` (use `!nonnull(e)` for may-be-null).
- `mlir::IntegerRangeAnalysis` — `e relop n`, `sizeof(e) relop n`.
- `patchestry.TaintAnalysis` — `tainted(e from src)`, `capture-taint:`.
- `mlir::AliasAnalysis` — `aliases(a, b)`.
- `patchestry.EscapeAnalysis` — `escapes(e)`.

A predicate whose analysis cannot prove or refute the property
conservatively returns `⊤` (unknown) — the surrounding `!` or `||`
determines whether unknown counts as fire-the-rule or skip-it.

---

## 8. Invariants

Carried over from `docs/GettingStarted/patch_specifications.md`:

- **SSA dominance.** A captured value referenced in a rewrite body must be
  defined in a block that dominates every insertion point. The compiler
  rejects rules where `insert at_entry:` references a capture from the
  call site.
- **`at_entry` captures (legacy `APPLY_AT_ENTRYPOINT`).** Same rule as legacy:
  `$return_value`-like captures are not visible at function entry; referencing
  them there is a compile-time error.
- **Capture unification is SSA equality.** Two uses of `$P` match
  iff both resolve to the same `Value`. For textual equality across
  separate SSA values (e.g. two `cir.load`s of the same global), use an
  explicit predicate: `aliases($P1, $P2)`.

---

## 9. Build & run

PatchDSL compiles to the same targets as the YAML front-end:

```sh
# Compile a .patch file to a shared pattern module
patchir-dslc rules/cwe190.patch -o rules/cwe190.patchmod

# Apply (drop-in replacement for -spec)
patchir-transform input.cir -dsl rules/cwe190.patchmod -o patched.cir
patchir-cir2llvm -S patched.cir -o patched.ll
```

During the migration window, `-spec` and `-dsl` may be combined; rules
from both sources load into the same `RewritePatternSet`.

Validate without running:

```sh
patchir-dslc rules/cwe190.patch --check      # parse + type-check only
patchir-dslc rules/cwe190.patch --dump-ir    # dump PatternIR
```

---

## 10. Error reporting

Every pattern token carries a source span. Diagnostics point at the
offending column:

```
rules/cwe190.patch:12:21: error: capture $R is used in rewrite: but not
                                  bound by any pattern
  rewrite:  $R = mul16::patch__replace__int_mul16($A, $B)
            ^
```

Semantic errors from analyses (e.g. unreachable `pattern-inside`) surface
at rule-compile time, not silently at match time.
