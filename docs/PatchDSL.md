# PatchDSL

A Semgrep-style pattern language for Patchestry, replacing the YAML
`meta_patches` / `match` / `action` spec. Patterns and fixes are written in
C-shaped syntax, matched and rewritten at the ClangIR (CIR) layer.

Companion documents:
- `docs/GettingStarted/patch_specifications.md` — legacy YAML (still supported
  during migration).
- `docs/SPEC_MIGRATION.md` — status of the YAML → PatchDSL cutover.

---

## 1. Philosophy

1. **Patterns look like real C.** No positional operand indexes, no
   `kind: "operation"` strings. You write the code you want to find.
2. **Metavariables unify by name.** `$P` captured once must be the same SSA
   value everywhere it appears.
3. **One file per rule set.** Match, fix body, inline helpers, and metadata
   live together. External C patches remain available via `import`.
4. **The DSL is a codegen layer.** It lowers to the same
   `InstrumentationPass` backend used today — nothing in the rewrite
   infrastructure changes.

---

## 2. File structure

A `.patch` file is a sequence of top-level items:

```
<file>            ::= { <metadata> | <import> | <patch_fn> | <rule> }
```

### 2.1 File-level metadata

```
metadata {
  name:        "cwe190-fixes"
  description: "Integer overflow fixes for bloodview"
  version:     "1.0.0"
  author:      "Security Team"
  created:     "2026-04-15"
}
```

Optional. Matches the fields present in legacy YAML. Not required for the
DSL to function.

### 2.2 Imports

Bring in an external C patch module whose functions can be invoked from
`fix:` bodies.

```
import "patches/patch_checked_mul16.c" as mul16
```

Symbols from the imported file are reachable as `mul16::<function>`.

### 2.3 Inline patch functions

For small helpers, skip the separate C file:

```
patch clamp_u16(x: uint32_t) -> uint16_t {
  return x > 0xFFFF ? 0xFFFF : (uint16_t)x;
}
```

Inline patches are compiled and linked exactly like `import`ed ones.

### 2.4 Rules

```
rule <name> [@<severity>] {
  <clause>+
  <action>+
}
```

`<severity>` ∈ `low | med | high | crit`. Drives report formatting only; it
does not affect matching.

---

## 3. Grammar (EBNF)

```ebnf
file          = { metadata | import | patch_fn | rule } ;

metadata      = "metadata" "{" { kv } "}" ;
import        = "import" string [ "as" ident ] ;
patch_fn      = "patch" ident "(" [ params ] ")" [ "->" type ] block ;
params        = param { "," param } ;
param         = ident ":" type ;

rule          = "rule" ident [ "@" severity ] "{"
                    { clause }
                    action
                    { action }
                "}" ;

clause        = pattern_clause
              | scope_clause
              | constraint_clause
              | predicate_clause
              | description_clause ;

pattern_clause
              = "pattern"                ":" code_block
              | "pattern-not"            ":" code_block
              | "pattern-either"         ":" "[" code_block
                                              { "," code_block } "]" ;

scope_clause  = "pattern-inside"        ":" code_block
              | "pattern-not-inside"    ":" code_block ;

constraint_clause
              = "metavariable-type"      ":" "{" "var" ":" metavar ","
                                              "type" ":" type "}"
              | "metavariable-pattern"   ":" "{" "var" ":" metavar ","
                                              "pattern" ":" code_block "}"
              | "metavariable-comparison":" "{" "var" ":" metavar ","
                                              "cmp" ":" predicate "}"
              | "metavariable-taint"     ":" "{" "var" ":" metavar ","
                                              "from" ":" source "}" ;

predicate_clause
              = "where" ":" predicate ;

description_clause
              = "description" ":" string
              | "id"          ":" string ;

action        = rewrite_action | insert_action ;
rewrite_action
              = "fix"             ":" code_block
              | "replace"         ":" code_block
              | "call"            ":" call_expr
              | "remove"
              | "assert"          ":" predicate ;

insert_action = "insert" position ":" insert_body ;
insert_body   = code_block
              | "call" ":" call_expr ;

position      = "before" | "after" | "at_entry" | "at_exit" ;
call_expr     = ident { "::" ident } "(" [ arg_list ] ")" ;
arg_list      = arg_expr { "," arg_expr } ;
arg_expr      = metavar | literal | call_expr ;

code_block    = "|" raw_c_with_metavars               (* YAML-style block *)
              | string ;                              (* short form *)

predicate     = pred_atom
              | predicate "&&" predicate
              | predicate "||" predicate
              | "!" predicate
              | "(" predicate ")"
              | "forall" metavar "in" expr ":" predicate
              | "exists" metavar "in" expr ":" predicate ;

pred_atom     = "nonnull"    "(" expr ")"
              | "may_be_null" "(" expr ")"
              | "bounded"    "(" expr [ "," "by" expr ] ")"
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

severity      = "low" | "med" | "high" | "crit" ;

metavar       = "$" ident | "$..." ident ;            (* $X or $...XS *)
ellipsis      = "..." ;                               (* in code_block only *)
```

### 3.1 Pattern vocabulary

Inside `code_block` you may use everything C accepts, plus:

| Token            | Meaning                                                 |
|------------------|---------------------------------------------------------|
| `$X`             | Metavariable. First use binds, later uses unify.        |
| `$...XS`         | Variadic metavariable. Captures an argument list.       |
| `...`            | Wildcard. Statement sequence, arg list, or expression.  |
| `$X: T`          | Inline type annotation on a metavariable.               |

Positions where `...` makes sense:

```
f(..., $X, ...)           // any call to f with $X somewhere in the args
if (...) { ... }          // any if with any body
{ ...; free($P); ...; }   // free($P) anywhere in the block
$OBJ.$F                   // any field access on $OBJ
```

---

## 4. Clause reference

### 4.1 Matching clauses

| Clause                  | Semantics                                             |
|-------------------------|-------------------------------------------------------|
| `pattern:`              | Required. The shape the rule searches for.            |
| `pattern-not:`          | Excludes matches that *also* match this pattern.      |
| `pattern-either:`       | Disjunction; rule fires if any branch matches.        |

Multiple `pattern-not:` clauses may appear; all must fail to match for the
rule to fire. Multiple `pattern:` clauses are ANDed (the same AST region
must match each).

### 4.2 Scope clauses

| Clause                  | Semantics                                             |
|-------------------------|-------------------------------------------------------|
| `pattern-inside:`       | Match must lie textually/structurally inside this.    |
| `pattern-not-inside:`   | Match must *not* lie inside this.                     |

Scope clauses lower to `Op::getParentOfType` walks (for structural scopes
like `cir.func`, `cir.loop`, `cir.if`) and CFG-region checks for ordered
scopes (`between X and Y`).

### 4.3 Constraint clauses

| Clause                       | Purpose                                          |
|------------------------------|--------------------------------------------------|
| `metavariable-type:`         | Refine a metavar by static type.                 |
| `metavariable-pattern:`      | Require a metavar to itself match a sub-pattern. |
| `metavariable-comparison:`   | Require a numeric/ordering predicate on a var.   |
| `metavariable-taint:`        | Require a metavar to be tainted from a source.   |

### 4.4 `where:` clauses

`where:` is the escape hatch for semantic predicates the pattern language
cannot express syntactically. Each predicate is backed by a CIR analysis:

| Predicate            | Backing analysis                                |
|----------------------|--------------------------------------------------|
| `nonnull(e)`         | Nullness lattice                                 |
| `may_be_null(e)`     | Nullness lattice (top / maybe)                   |
| `bounded(e)`         | IntegerRangeAnalysis                             |
| `bounded(e, by=s)`   | IntegerRangeAnalysis vs `s`                      |
| `tainted(e from src)`| Taint dataflow                                   |
| `reaches(a, b)`      | Forward dataflow                                 |
| `dominates(a, b)`    | `DominanceInfo` on enclosing region              |
| `aliases(a, b)`      | MLIR alias analysis over CIR                     |
| `escapes(e)`         | Escape analysis                                  |
| `sizeof(e) relop n`  | Type introspection                               |
| `type(e) relop T`    | Type introspection                               |

User-defined predicates are registered C++ callbacks; they receive the
capture environment and return `bool`.

---

## 5. Actions

Each rule has one or more actions. A rule may include any number of
`insert ...` actions, and at most one rewrite-style action
(`fix`, `replace`, `call`, `remove`, or `assert`).

### 5.1 `fix:` — inline pattern-to-pattern rewrite

The matched region is replaced by the fix pattern, with captured
metavariables substituted. The fix uses the same pattern grammar as
`pattern:`.

```
rule flip_strict_lt {
  pattern: for (...; $I < $N; ...) { ... $ARR[$I + 1] ... }
  fix:     for (...; $I < $N - 1; ...) { ... $ARR[$I + 1] ... }
}
```

Ellipses on both sides are preserved literally — captured statements
between anchors keep their original position.

### 5.2 `replace:` — alias for `fix:`

Identical semantics; use it when "replace" reads more naturally than "fix"
(e.g. style choice for refactorings rather than security patches).

```
rule use_safer_copy {
  pattern: strcpy($D, $S)
  replace: safe_strcpy($D, sizeof($D), $S)
}
```

### 5.3 `insert before:` / `insert after:`

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
  where:   may_be_null($P)
  insert before: |
    if ($P == NULL) return;
}
```

### 5.4 `insert at_entry:` / `insert at_exit:`

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

`#FN` is a stringified metavariable — a compile-time literal, not an SSA
value.

### 5.5 `call:` — invoke an external/inline patch function

Equivalent to `fix: $R = <callee>($ARGS...)` when the entire match result
is replaced by a call. This is the DSL spelling of today's YAML `replace`
with `patch_id`.

```
import "patches/patch_checked_mul16.c" as mul16

rule cwe190_int_overflow_fix @high {
  pattern-inside: $RET peek_process(...) { ... }
  pattern:        $R = $A * $B
  metavariable-type: { var: $R, type: uint16_t }
  where:          !bounded($A) || !bounded($B)
  call:           mul16::patch__replace__int_mul16($A, $B)
}
```

`call:` replaces the matched expression with the call's result, inserting
casts on operands and return value when the callee signature differs from
the capture types. This is exactly what
`InstrumentationPass::prepare_patch_call_arguments` does today, just
driven by named captures instead of `operand: { index: N }`.

For `insert before` / `insert after` style call patches (the
`apply_before` / `apply_after` modes in legacy YAML), use:

```
rule audit_sprintf {
  pattern: sprintf($D, $F, $...REST)
  insert before:
    call: audit::check_format($D, $F)
}
```

i.e. insert a normal call statement before/after the match.

### 5.6 `remove:` — delete the match

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

The second occurrence is the one deleted; the ellipsis marks the anchor
pair.

The second occurrence is the one deleted; the ellipsis marks the anchor
pair.

### 5.7 `assert:` — contract mode

Emits a `cir.call @__patchestry_assume` (or runtime assertion, per build
config) at the insertion point implied by the scope:

```
rule sprintf_precond {
  pattern: sprintf($D, $F, $...REST)
  assert: nonnull($D) && sizeof($D) >= 32 && nonnull($F)
}
```

Contracts share the predicate vocabulary with `where:` but are *emitted*
rather than *checked at match time*. The same DSL source can thus drive
both KLEE harness generation and runtime assertions.

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

rule cwe190_int_overflow_fix @high {
  description: "Replace unchecked u16 multiply in peek_process"

  pattern-inside: $RET peek_process(...) { ... }
  pattern:        $R = $A * $B

  metavariable-type: { var: $R, type: uint16_t }
  metavariable-type: { var: $A, type: uint16_t }
  metavariable-type: { var: $B, type: uint16_t }

  where: !bounded($A) || !bounded($B)

  call:  mul16::patch__replace__int_mul16($A, $B)
}
```

### 6.2 Replace (inline, no C file): CWE-476 null guard

**PatchDSL:**

```
rule cwe476_guard_deref @high {
  pattern:            $P->$F
  pattern-not-inside: if ($P) { ... }
  pattern-not-inside: if ($P != NULL) { ... }
  where:              may_be_null($P)

  fix: ($P ? $P->$F : 0)
}
```

### 6.3 Insert before: audit a call-site

Equivalent to the legacy `apply_before` mode.

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
  metavariable-comparison: { var: $FN,
                             cmp: matches("measurement") && matches("update") }

  pattern: spo2_lookup($X)
  metavariable-type: { var: $X, type: float }

  insert before:
    call: mu::measurement_update_before_operation($X)
}
```

### 6.4 Insert after: null-after-free

**PatchDSL:**

```
rule null_after_free @high {
  pattern-inside: $RET $FN(...) { ... }
  pattern:        free($P)
  pattern-not-inside: |
    free($P);
    ...
    $P = NULL;

  insert after: |
    $P = NULL;
}
```

The `pattern-not-inside` prevents double-fixing: if the code already
NULLs the pointer, skip.

### 6.5 Fix rewrite: double-free

**PatchDSL:**

```
rule cwe415_double_free @crit {
  pattern: |
    free($P);
    ...
    free($P);
  where: aliases($P, $P)           // trivially true, kept for clarity
      && !reassigned_between($P, free($P), free($P))

  fix: |
    free($P);
    $P = NULL;
    ...
}
```

The fix replaces the *entire* matched region. The ellipsis between the
two frees is preserved; the second `free($P)` is dropped; `$P = NULL` is
inserted after the first.

### 6.6 Replace call: CWE-078 command injection

**PatchDSL:**

```
import "patches/patch_system.c" as patch_sys

rule cwe078_system_taint @crit {
  pattern: system($CMD)
  metavariable-taint: { var: $CMD, from: user_input }

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
rule unsafe_str_copy @high {
  pattern-either: [
    strcpy($D, $S),
    strcat($D, $S),
    sprintf($D, "%s", $S)
  ]
  where: !bounded($S, by=sizeof($D))

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

1. **Parse.** The `.patch` file is parsed by a tree-sitter-C grammar
   extended with metavariable / ellipsis tokens.
2. **Lower.** Each pattern becomes a **PatternIR** — a CIR-shaped
   template with named captures and region anchors.
3. **Codegen.** PatternIR lowers to either:
   - PDL bytecode (for rules with no `where:` / taint predicates), or
   - a generated C++ `OpRewritePattern<T>` (when semantic predicates are
     present).
4. **Drive.** Rules are registered into a `RewritePatternSet` and applied
   by the existing `InstrumentationPass` via
   `applyPatternsAndFoldGreedily`.
5. **Emit.** Fix bodies, `call:`, `insert …` actions, and `assert:` emit
   CIR ops directly with the `PatternRewriter`. External patches are
   linked as today.

Analyses (`DominanceInfo`, nullness, taint, alias, integer range) are
requested via MLIR's analysis manager and cached per function.

---

## 8. Invariants

Carried over from `docs/GettingStarted/patch_specifications.md`:

- **SSA dominance.** A captured value referenced in a fix body must be
  defined in a block that dominates every insertion point. The compiler
  rejects rules where `insert at_entry:` references a capture from the
  call site.
- **`at_entry` captures (legacy `APPLY_AT_ENTRYPOINT`).** Same rule as legacy:
  `$return_value`-like captures are not visible at function entry; referencing
  them there is a compile-time error.
- **Metavariable unification is SSA equality.** Two uses of `$P` match
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
rules/cwe190.patch:12:21: error: metavariable $R is used in fix: but not
                                  bound by any pattern
  fix:  $R = mul16::patch__replace__int_mul16($A, $B)
        ^
```

Semantic errors from analyses (e.g. unreachable `pattern-inside`) surface
at rule-compile time, not silently at match time.

---

## 11. Migration notes

The legacy YAML surface continues to parse during the transition. A
`patchir-spec-convert` tool ships alongside `patchir-dslc` and emits a
first-cut `.patch` file from any existing `meta_patches` spec. Hand-edit
the output to add scope constraints, `where:` predicates, and type
refinements — these are the clauses the YAML could not express, and the
main reason to migrate.
