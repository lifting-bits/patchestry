# PatchDSL

Pattern-matching and rewriting language for patching decompiled
binaries at the ClangIR (CIR) level. File extension: `.pdsl`.
Compiled artifact: `.patchmod`. Replaces the YAML patch specification
(`docs/GettingStarted/patch_specifications.md`).

---

## 1. File structure

A `.pdsl` file is a sequence of top-level items: metadata, imports,
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

`target` is required when applying rules to a CIR module.

| Field     | Required | Semantics                                                     |
|-----------|----------|---------------------------------------------------------------|
| `binary`  | yes      | Matched against the CIR module's `patchir.source_binary` attribute. |
| `arch`    | yes      | Ghidra-style `PROCESSOR:ENDIAN:BITWIDTH:VARIANT` triple, matched against `patchir.target_arch`. `*` wildcards any component (e.g. `"ARM:LE:32:*"`). |

Checked before any pattern runs. Mismatch is a hard error.

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
rule <name> { <clause>+ <action>+ }        // patches: rewrite / insert / remove / assert
contract <name> { <clause>+ <contract-clause>+ }   // static contracts: requires / ensures / invariant
```

A `rule` inserts, replaces, or removes code at matched sites.
A `contract` attaches static properties as MLIR attributes for
verification tooling — no runtime code emitted. See §5.

---

## 2. Grammar

```ebnf
/* top-level */

file          = { metadata | import | patch_fn | rule | contract } ;

/* metadata */

metadata      = "metadata" "{" { kv | target_block } "}" ;
target_block  = "target" "{" { kv } "}" ;
kv            = ident ":" ( string | number ) ;

/* imports */

import        = "import" string "as" ident ;

/* inline patch helpers */

patch_fn      = "patch" ident "(" [ params ] ")" "->" type
                "{" raw_c "}" ;
params        = param { "," param } ;
param         = ident ":" type ;

/* rules (patches and runtime contracts) */

rule          = "rule" ident "{"
                    { clause }
                    action
                    { action }
                "}" ;

/* contracts (static only) */

contract      = "contract" ident "{"
                    { clause | contract_clause }
                "}" ;

contract_clause
              = "requires"   ":" predicate
              | "ensures"    ":" predicate
              | "invariant"  ":" predicate ;

/* clauses (shared by rules and contracts) */

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

/* actions */

action        = rewrite_action | insert_action ;

rewrite_action
              = "rewrite"         ":" action_body
              | "remove"          [ ":" code_block ]
              | "assert"          ":" predicate ;

insert_action = "insert" position ":" action_body ;

action_body   = code_block
              | "call" ":" call_expr ;       /* explicit function call */

position      = "before" | "after" | "at_entry" | "at_exit" ;

/* call expressions */

call_expr     = ident { "::" ident } "(" [ arg_list ] ")" ;
arg_list      = arg_expr { "," arg_expr } ;
arg_expr      = "&" arg_expr                 /* address-of / is_reference */
              | capture
              | literal
              | ident ;                      /* bare identifier / global symbol */

/* code blocks */

code_block    = "|" raw_c_with_metavars      /* indented block after | */
              | inline_c ;                   /* rest-of-line C fragment */

/* predicates */
/* Precedence (tightest first): !, &&, ||                      */

predicate     = or_pred ;
or_pred       = and_pred { "||" and_pred } ;
and_pred      = unary_pred { "&&" unary_pred } ;
unary_pred    = "!" unary_pred
              | "(" predicate ")"
              | "forall" capture "in" expr ":" predicate
              | "exists" capture "in" expr ":" predicate
              | pred_atom ;

pred_atom     = "nonnull"    "(" expr ")"
              | "tainted"    "(" expr "from" source ")"
              | "reaches"    "(" expr "," expr ")"
              | "aliases"    "(" expr "," expr ")"
              | "escapes"    "(" expr ")"
              | "sizeof"     "(" expr ")"  relop expr
              | "type"       "(" expr ")"  relop type
              | expr relop expr
              | ident "(" [ arg_list ] ")" ; /* user-defined predicate */

/* expressions */

expr          = capture
              | "@return"                    /* postcondition pseudo-capture */
              | "#" ident                    /* stringified capture */
              | literal
              | ident
              | expr binop expr
              | "(" expr ")" ;

/* terminals */

capture       = "$" ident                    /* $X */
              | "$..." ident ;               /* $...XS, variadic */

literal       = number | string ;
number        = digit { digit }
              | "0x" hex_digit { hex_digit } ;

type          = { "*" } ident ;              /* e.g. *char, uint16_t, *void */

relop         = "<" | "<=" | ">" | ">=" | "==" | "!=" ;
binop         = "+" | "-" | "*" | "/" | "%" ;

source        = ident ;                      /* taint source, user-defined */

ellipsis      = "..." ;                      /* wildcard inside code_block */
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

### 3.1 Matching and scope

`pattern:` is the shape the rule searches for. Multiple `pattern:`
clauses are ANDed; the same AST region must match each. `pattern-either:`
is a disjunction: the rule fires if any branch matches.

`pattern-inside:` restricts the search to a structural scope, typically
a function body (`$RET fn(...) { ... }`).

### 3.2 Constraint clauses

`capture-pattern:` requires a capture to itself match a sub-pattern.
`capture-comparison:` requires a numeric or ordering predicate on a
capture. `capture-taint:` requires a capture to be tainted from a named
source. Static type refinements use the inline `$X: T` annotation in
the pattern itself (see §2.1) rather than a separate clause.

### 3.3 `where:` clauses

`where:` is the escape hatch for semantic predicates the pattern language
cannot express syntactically.

Nullness and type predicates -- `nonnull(e)`, `sizeof(e) relop n`,
`type(e) relop T`, and plain `e relop n` -- use CIR's type system and
integer-range analysis. Dataflow predicates -- `tainted(e from src)`,
`reaches(a, b)`, `aliases(a, b)`, `escapes(e)` -- are backed by MLIR
analysis passes. User-defined predicates are C++ callbacks that receive
the capture environment and return `bool`.

---

## 4. Actions

Each rule has one or more actions. A rule may include any number of
`insert ...` actions, and at most one rewrite-style action
(`rewrite`, `remove`, or `assert`).

`call:` is a body-form indicator inside `rewrite:` and `insert`, not a
standalone action.

### 4.1 `rewrite:`

Inline C:

```
rule flip_strict_lt {
  pattern: for (...; $I < $N; ...) { ... $ARR[$I + 1] ... }
  rewrite: for (...; $I < $N - 1; ...) { ... $ARR[$I + 1] ... }
}
```

`call:` form:

```
import "patches/patch_checked_mul16.c" as mul16 {
  patch patch__replace__int_mul16(count: *char, block_size: *char) -> i32;
}

rule cwe190_int_overflow_fix {
  pattern-inside: $RET peek_process(...) { ... }
  pattern:        ($R: uint16_t) = $A * $B
  where:          !($A * $B <= 0xFFFF)
  rewrite:
    call: mul16::patch__replace__int_mul16($A, $B)
}
```

### 4.2 `insert before:` / `insert after:`

```
rule null_after_free {
  pattern-inside: $RET process_command(...) { ... }
  pattern:        free($P)

  insert after: |
    $P = NULL;
}
```

```
rule guard_before_write {
  pattern-inside: $RET handler(...) { ... }
  pattern:        *$P = $V
  where:          !nonnull($P)

  insert before: |
    if ($P == NULL) return;
}
```

### 4.3 `insert at_entry:` / `insert at_exit:`

```
rule log_entry_exit {
  pattern-inside: $RET $FN(...) { ... }

  insert at_entry: |
    trace::enter(#FN);
  insert at_exit: |
    trace::leave(#FN);
}
```

`#FN` is a stringified capture, a compile-time literal, not an SSA
value.

### 4.4 `remove:`, delete the match

Erases the matched op(s). Combine with `insert` if you need to add
new behavior at the same site.

```
rule strip_debug_prints {
  pattern: debug_printf($...ARGS)

  remove
}
```

### 4.5 `assert:`, site-local check

Emits a `cir.call @__patchestry_assume` (or runtime assertion, per build
config) at the matched site, useful for preconditions bound to a
*specific* call or op:

```
rule sprintf_precond {
  pattern: sprintf($D, $F, $...REST)
  assert: nonnull($D) && sizeof($D) >= 32 && nonnull($F)
}
```

`assert:` shares the predicate vocabulary with `where:` but is *emitted*
rather than *checked at match time*. It is a runtime action; it
inserts executable code at the matched site, just like `insert before:`.
For *static* properties (pre/post-conditions, invariants, attribute
tags consumed by verifiers), use a `contract` block; see §5.

---

## 5. Contracts (static only)

Contracts attach *static* properties to functions or regions without
rewriting their bodies. They lower exclusively to MLIR attributes --
zero runtime cost, consumed by verification tools (KLEE, SARIF
exporters, MLIR dataflow passes). Runtime validation calls (null
checks, bounds guards) are **rules**, not contracts -- if it inserts
executable code, it belongs in a `rule`.

A contract needs a `pattern-inside:` scope and at least one of
`requires:` (entry precondition), `ensures:` (return postcondition),
or `invariant:` (loop/region property). They use the same predicate
vocabulary as `where:` and `assert:`. Inside `ensures:`, `@return`
refers to the value at each `cir.return` site (note the `@` prefix,
not `$` -- it's synthesized per return, not a match binding).

```
contract peek_process_safety {
  pattern-inside: $RET peek_process(($CNT: uint16_t), ($SZ: uint16_t)) { ... }

  requires: $CNT > 0 && $CNT * $SZ <= 0xFFFF
  ensures:  @return != NULL
}
```

The compiler attaches `patchestry.requires` and `patchestry.ensures`
as dialect attributes on the `cir.func`. A verifier consumes them
directly; no runtime code is emitted.

---

## 6. Worked examples

### 6.1 CWE-190: Integer overflow

An unchecked narrow multiply can wrap and bypass a downstream bounds
check. The simplest fix widens the operands before multiplying:

```
rule cwe190_widen_multiply {
  pattern-inside: $RET peek_process(...) { ... }
  pattern:        ($R: uint16_t) = $A * $B
  where:          !($A * $B <= 0xFFFF)

  rewrite: ($R: uint16_t) = (uint16_t)((uint32_t)$A * (uint32_t)$B)
}
```

When a cast isn't enough, say you need saturation arithmetic or an
error return on overflow, delegate to an external C function. The
import declares the signature; `call:` tells the compiler to generate
the call and insert casts where the types don't line up:

```
import "patches/patch_checked_mul16.c" as mul16 {
  patch patch__replace__int_mul16(a: *char, b: *char) -> i32;
}

rule cwe190_checked_multiply {
  pattern-inside: $RET peek_process(...) { ... }
  pattern:        ($R: uint16_t) = $A * $B
  where:          !($A * $B <= 0xFFFF)

  rewrite:
    call: mul16::patch__replace__int_mul16($A, $B)
}
```

### 6.2 CWE-476: Null pointer dereference

```
rule cwe476_guard_deref {
  pattern: $P->$F
  where:   !nonnull($P)

  rewrite: ($P ? $P->$F : 0)
}
```

### 6.3 CWE-415: Double free

Null the pointer after `free()` so a second free on the same path
becomes a no-op. `&$P` passes the pointer by reference so the patch
function can write NULL through it.

```
import "patches/patch_null_after_free.c" as nullify {
  patch patch__after__free(ptr_ref: *void) -> void;
}

rule cwe415_null_after_free {
  pattern-inside: $RET process_command(...) { ... }
  pattern:        free($P)

  insert after:
    call: nullify::patch__after__free(&$P)
}
```

### 6.4 CWE-022: Path traversal

```
rule cwe022_mkdir_path_traversal {
  pattern-inside: $RET init_logger(...) { ... }
  pattern:        mkdir($PATH, $...REST)

  insert before: |
    if (strstr((const char *)$PATH, "..") != NULL) return;
}
```

### 6.5 CWE-121/CWE-122: Buffer overflow

Inline rewrites work well for simple substitutions. `pattern-either:`
catches several unsafe variants in one rule; the rewrite applies to
whichever branch matched:

```
rule cwe121_bounded_strcat {
  pattern-inside: $RET eeprom_write(...) { ... }
  pattern:        strcat($DEST, $SRC)

  rewrite: strncat($DEST, $SRC, sizeof($DEST) - strlen($DEST) - 1)
}

rule cwe120_unsafe_str_ops {
  pattern-either: [
    strcpy($D, $S),
    strcat($D, $S),
    sprintf($D, "%s", $S)
  ]
  where: !(strlen($S) <= sizeof($D))

  rewrite: snprintf($D, sizeof($D), "%s", $S)
}
```

### 6.6 CWE-134: Format string

```
rule cwe134_format_string {
  pattern: printf($FMT)
  capture-taint: { var: $FMT, from: user_input }

  rewrite: printf("%s", $FMT)
}
```

### 6.7 Putting it together: patch + hardening + contract

A real `.pdsl` file often carries several rules and a contract. Rules
apply in file order; the contract attaches static metadata after rules
reach a fixed point.

```
metadata {
  name: "usb-security"
  target { binary: "firmware.bin"  arch: "ARM:LE:32:Cortex" }
}

import "patches/patch_usbd.c" as usb {
  patch patch__before__usbd_ep_write_packet(dev: *void, buf: *void) -> void;
}

import "contracts/usb_validation.c" as usbcheck {
  patch contract__before__test_contract(dev: *void, buf: *void) -> void;
}

rule usb_pre_validation {
  pattern-inside: $RET bl_usb__send_message(...) { ... }
  pattern:        usbd_ep_write_packet($DEV, $EP, $BUF, $LEN)

  insert before:
    call: usb::patch__before__usbd_ep_write_packet($DEV, $BUF)
}

rule usb_runtime_contract {
  pattern-inside: $RET bl_usb__send_message(...) { ... }
  pattern:        usbd_ep_write_packet($DEV, $EP, $BUF, $LEN)

  insert before:
    call: usbcheck::contract__before__test_contract($DEV, $EP)
}

contract usb_endpoint_write_static {
  pattern-inside: $RET bl_usb__send_message(...) { ... }
  pattern:        usbd_ep_write_packet($DEV, $EP, $BUF, $LEN)

  requires: nonnull($DEV) && $LEN >= 0 && $LEN <= 512
  ensures:  @return >= 0 && @return <= 512
}
```

---

## 7. Execution model

1. **Load** — parse `.pdsl`, verify `target` against CIR module, type-check captures.
2. **Compile** — imported `.c` files and inline patches compiled to CIR, cached as `.patchmod`.
3. **Match** — walk each `cir.func`. Evaluate `pattern-inside:` first, then `pattern:`, then `where:` (short-circuit).
4. **Rewrite** — apply actions in declaration order within a file, command-line order across files.
5. **Emit** — contracts attach MLIR attributes after rules reach a fixed point.

Rules that match zero sites produce a warning. Dominance failures
roll back that rule and continue.

---

## 8. Build & run

PatchDSL compiles to the same targets as the YAML front-end:

```sh
# Compile a .pdsl file to a shared pattern module
patchir-dslc rules/cwe190.pdsl -o rules/cwe190.patchmod

# Apply (drop-in replacement for -spec)
patchir-transform input.cir -dsl rules/cwe190.patchmod -o patched.cir
patchir-cir2llvm -S patched.cir -o patched.ll
```

The `-dsl` surface is a superset of `-spec`: every patch expressible in
YAML can be written in PatchDSL, plus inline rewrites, static contracts,
`where:` predicates, and `pattern-either:` that YAML cannot express.
During the migration window, `-spec` (YAML) continues to work unchanged;
new features should be authored in `.pdsl` files via `-dsl`.

Validate without running:

```sh
patchir-dslc rules/cwe190.pdsl --check      # parse + type-check only
```

---

## 9. Migration from YAML

Every YAML spec has a direct `.pdsl` equivalent. The compiler
produces identical CIR output for both paths.

### Before (YAML)

```yaml
apiVersion: patchestry.io/v1
metadata:
  name: "cwe121-strcat-fix"
target:
  binary: "firmware.bin"
  arch: "ARM:LE:32:v7"
libraries:
  - "patches/cwe121_patches.yaml"
meta_patches:
  - name: "bounded_strcat"
    patch_actions:
      - id: "CWE-121-001"
        match:
          - name: "strcat"
            kind: "function"
            function_context:
              - name: "eeprom_write"
        action:
          - mode: "replace"
            patch_id: "cwe121_bounded_strcat"
            arguments:
              - name: "dest"
                source: "operand"
                index: 0
              - name: "src"
                source: "operand"
                index: 1
```

### After (PatchDSL)

```
metadata {
  name: "cwe121-strcat-fix"
  target { binary: "firmware.bin"  arch: "ARM:LE:32:v7" }
}

import "patches/cwe121_bounded_strcat.c" as fix

rule bounded_strcat {
  pattern-inside: $RET eeprom_write(...) { ... }
  pattern:        strcat($DEST, $SRC)

  rewrite:
    call: fix::cwe121_bounded_strcat($DEST, $SRC)
}
```

Both produce byte-identical patched CIR. The `.pdsl` form
eliminates the indirection through `meta_patches` / `patch_actions` /
`match` / `action` layers and names captures directly instead of
by operand index.

---
