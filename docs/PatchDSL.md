# PatchDSL

A pattern-matching and rewriting language for Patchestry. Authors write
patterns and rewrites in C-shaped syntax; the compiler matches them
against the ClangIR (CIR) representation of a decompiled binary and
applies patches, runtime hardening, and static contracts — all without
modifying the original source or re-decompiling.

Replaces the legacy YAML `meta_patches` / `match` / `action` spec
(`docs/GettingStarted/patch_specifications.md`).

---

## 1. File structure

A `.pdsl` file is a sequence of top-level items — metadata, imports,
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
rule <name> { <clause>+ <action>+ }        // patches: rewrite / insert / remove / assert
contract <name> { <clause>+ <contract-clause>+ }   // static contracts: requires / ensures / invariant
```

A `rule` *changes* program behavior — patching, hardening, or
instrumenting the binary by inserting, replacing, or removing code.
A `contract` *declares* a static property attached as an MLIR
attribute for verification tooling (KLEE, SARIF exporters, etc.) —
it emits no runtime code. See §5 for the contract surface in full.

> **Note.** Runtime validation calls (null checks, bounds guards,
> assertions inserted before or after a call site) are **rules**, not
> contracts. They insert executable code that hardens the binary at
> runtime, which is exactly what a `rule` with `insert before:`
> does. The `contract` block is reserved for *static* metadata that
> verification tools consume without touching the emitted binary.

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
              | "dominates"  "(" expr "," expr ")"
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
              | "$..." ident ;               /* $...XS — variadic */

literal       = number | string ;
number        = digit { digit }
              | "0x" hex_digit { hex_digit } ;

type          = { "*" } ident ;              /* e.g. *char, uint16_t, *void */

relop         = "<" | "<=" | ">" | ">=" | "==" | "!=" ;
binop         = "+" | "-" | "*" | "/" | "%" ;

source        = ident ;                      /* taint source — user-defined */

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
(`rewrite`, `remove`, or `assert`).

### 4.1 `rewrite:` — replace the matched region

The matched region is replaced by the rewrite body, with captures
substituted. The body can be either **inline C code** or an explicit
**`call:`** to an imported function.

**Inline C code** (pattern-to-pattern rewrite):

```
rule flip_strict_lt {
  pattern: for (...; $I < $N; ...) { ... $ARR[$I + 1] ... }
  rewrite: for (...; $I < $N - 1; ...) { ... $ARR[$I + 1] ... }
}
```

**`call:` to an imported function** (the compiler generates the call,
inserting casts when the callee signature differs from the capture
types):

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

Ellipses on both sides are preserved literally — captured statements
between anchors keep their original position.

### 4.2 `insert before:` / `insert after:`

Adds new statements immediately before or after the match, without
touching the match itself. The body follows the same rules as
`rewrite:` — inline C or `call:` to an imported function.

```
rule null_after_free {
  pattern: free($P)
  insert after: |
    $P = NULL;
}
```

```
rule audit_sprintf {
  pattern: sprintf($D, $F, $...REST)
  insert before:
    call: audit::check_format($D, $F)
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
`cir.func`. Useful for instrumentation or resource tracking.

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

### 4.4 `remove:` — delete the match

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

### 4.5 `assert:` — site-local check

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
inserts executable code at the matched site, just like `insert before:`.
For *static* properties (pre/post-conditions, invariants, attribute
tags consumed by verifiers), use a `contract` block — see §5.

---

## 5. Contracts (static only)

Contracts attach *static* properties to functions or regions without
rewriting their bodies. They lower exclusively to MLIR attributes —
zero runtime cost, consumed by verification tools (KLEE, SARIF
exporters, MLIR dataflow passes).

> **Note.** Runtime validation calls — null checks, bounds guards,
> assertion calls inserted at a call site — are **rules**, not
> contracts. Inserting executable code that hardens the binary at
> runtime is what `rule` with `insert before:` / `insert after:`
> does. A `contract` block never emits runtime code; it only attaches
> metadata for static analysis.

### 5.1 Block shape

```
contract <name> {
  pattern-inside: <function-or-region scope>

  requires:   <predicate>          // entry precondition
  ensures:    <predicate>          // return postcondition
  invariant:  <predicate>          // loop / region invariant
}
```

All clauses are optional but at least one `requires:` / `ensures:` /
`invariant:` must be present. At least one `pattern-inside:` clause is
required so the contract has a scope to attach to.

### 5.2 Clause reference

| Clause        | Scope target                              | Lowering                                          |
|---------------|-------------------------------------------|---------------------------------------------------|
| `requires:`   | enclosing `cir.func`                      | `patchestry.requires = #pred` on the func         |
| `ensures:`    | enclosing `cir.func`                      | `patchestry.ensures = #pred` on the func          |
| `invariant:`  | matched `cir.for` / `cir.while` / region  | `patchestry.invariant = #pred` on the region      |

`requires:` / `ensures:` / `invariant:` use the same predicate
vocabulary as `where:` (§3.4) and `assert:` (§4.6).

### 5.3 Rules vs. contracts

| Need                                                    | Use                            |
|---------------------------------------------------------|--------------------------------|
| Runtime null check, bounds guard, or hardening call     | `rule` with `insert before/after:` |
| Precondition for a specific call-site (runtime check)   | `assert:` in a `rule`          |
| Function-level entry/exit property (static metadata)    | `requires:` / `ensures:` in a `contract` |
| Loop or region invariant (static metadata)              | `invariant:` in a `contract`   |

### 5.4 Example — `peek_process` safety contract

```
contract peek_process_safety {
  pattern-inside: $RET peek_process(($CNT: uint16_t), ($SZ: uint16_t)) { ... }

  requires:   $CNT > 0 && $CNT * $SZ <= 0xFFFF
  ensures:    @return != NULL
  invariant:  $IDX <= $SZ
}
```

The function picks up:

```mlir
cir.func @peek_process(...) attributes {
  patchestry.requires   = #patchestry<"$CNT > 0 && $CNT * $SZ <= 0xFFFF">,
  patchestry.ensures    = #patchestry<"@return != NULL">
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

## 6. Worked examples — common CWEs for C/C++

Each example shows a complete rule (or rule + contract pair) that
patches a specific vulnerability class. All examples assume the patch
C files exist at the indicated paths.

### 6.1 CWE-190: Integer overflow

An unchecked narrow multiply can wrap and bypass a downstream bounds
check. Widen the operands to 32-bit before multiplying:

```
rule cwe190_widen_multiply {
  description: "Widen u16 multiply to u32 to prevent overflow"

  pattern-inside: $RET peek_process(...) { ... }
  pattern:        ($R: uint16_t) = $A * $B
  where:          !($A * $B <= 0xFFFF)

  rewrite: ($R: uint16_t) = (uint16_t)((uint32_t)$A * (uint32_t)$B)
}
```

When a simple cast is not enough (e.g. saturation or error-return on
overflow), delegate to an external function:

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

Guard a dereference where the pointer may be null. Two variants —
inline rewrite (no external C file) and external call.

**Inline rewrite:**

```
rule cwe476_guard_deref {
  pattern: $P->$F
  where:   !nonnull($P)

  rewrite: ($P ? $P->$F : 0)
}
```

**External call:**

```
import "patches/patch_get_name.c" as nullcheck {
  patch patch__before__get_name(var_ptr: *void) -> void;
}

rule cwe476_null_check_before_call {
  pattern-inside: $RET poke_process(...) { ... }
  pattern:        get_name($OBJ)

  insert before:
    call: nullcheck::patch__before__get_name($OBJ)
}
```

### 6.3 CWE-415: Double free

Null the pointer after the first `free()` so a second `free()` on the
same path becomes a safe no-op.

```
import "patches/patch_null_after_free.c" as nullify {
  patch patch__after__free(ptr_ref: *void) -> void;
}

rule cwe415_null_after_free {
  description: "Null pointer after free to prevent double-free"

  pattern-inside: $RET process_command(...) { ... }
  pattern:        free($P)

  insert after:
    call: nullify::patch__after__free(&$P)
}
```

`&$P` passes the address of the pointer so the patch can null it
(legacy YAML's `is_reference: true`).

### 6.4 CWE-416: Use-after-free (inline rewrite)

Replace a matched double-free pattern with a safe version that NULLs
the pointer between the two frees.

```
rule cwe416_double_free_inline {
  pattern: |
    free($P);
    ...
    free($P);
  where: !reassigned_between($P, free($P), free($P))

  rewrite: |
    free($P);
    $P = NULL;
    ...
}
```

The ellipsis between the two frees is preserved; the second `free($P)`
is dropped; `$P = NULL` is inserted after the first.

### 6.5 CWE-078: OS command injection

Replace `system()` with a sanitized wrapper that rejects shell
metacharacters.

```
import "patches/patch_system.c" as sys {
  patch patch__replace__system(command: *char, max_len: size_t) -> i32;
}

rule cwe078_system_sanitize {
  description: "Replace system() with metacharacter-filtered version"

  pattern-inside: $RET create_port(...) { ... }
  pattern:        system($CMD)

  rewrite:
    call: sys::patch__replace__system($CMD, 512)
}
```

### 6.6 CWE-022: Path traversal

Guard `mkdir()` against directory-traversal sequences with an inline
check. The `strstr` call scans for `../` at runtime; if found, the
function returns early before the filesystem operation.

```
rule cwe022_mkdir_path_traversal {
  description: "Reject ../ sequences before mkdir"

  pattern-inside: $RET init_logger(...) { ... }
  pattern:        mkdir($PATH, $...REST)

  insert before: |
    if (strstr((const char *)$PATH, "..") != NULL) return;
}
```

For more thorough validation (canonicalization, symlink resolution),
delegate to an external function:

```
import "patches/patch_mkdir.c" as mkdirguard {
  patch patch__before__mkdir(path: *void) -> void;
}

rule cwe022_mkdir_external_validation {
  pattern-inside: $RET init_logger(...) { ... }
  pattern:        mkdir($PATH, $...REST)

  insert before:
    call: mkdirguard::patch__before__mkdir($PATH)
}
```

### 6.7 CWE-094: Code injection via `popen()`

Insert an allowlist check before `popen()` to block arbitrary command
execution.

```
import "patches/patch_popen.c" as popenguard {
  patch patch__before__popen(command: *void) -> void;
}

rule cwe094_popen_allowlist {
  pattern-inside: $RET run_diagnostic(...) { ... }
  pattern:        popen($CMD, $...REST)

  insert before:
    call: popenguard::patch__before__popen($CMD)
}
```

### 6.8 CWE-121/CWE-122: Buffer overflow (stack and heap)

Replace unbounded string/memory operations with bounded equivalents.

**Stack — inline rewrite: `strcat` → `strncat` with remaining capacity:**

```
rule cwe121_bounded_strcat {
  pattern-inside: $RET eeprom_write(...) { ... }
  pattern:        strcat($DEST, $SRC)

  rewrite: strncat($DEST, $SRC, sizeof($DEST) - strlen($DEST) - 1)
}
```

**Stack — inline rewrite: `strcpy` → `strncpy` with explicit null-termination:**

```
rule cwe121_bounded_strcpy {
  pattern: strcpy($D, $S)

  rewrite: |
    strncpy($D, $S, sizeof($D) - 1);
    $D[sizeof($D) - 1] = '\0';
}
```

**Heap — external call for complex bounds clamping:**

```
import "patches/patch_memcpy.c" as memguard {
  patch patch__replace__memcpy(dest: *void, src: *void, n: size_t, max: size_t) -> *void;
}

rule cwe122_bounded_memcpy {
  pattern-inside: $RET process_rx(...) { ... }
  pattern:        memcpy($DEST, $SRC, $N)

  rewrite:
    call: memguard::patch__replace__memcpy($DEST, $SRC, $N, 256)
}
```

**Pattern-either — catch multiple unsafe variants in one rule:**

```
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

### 6.9 CWE-134: Uncontrolled format string

Replace `printf($FMT)` (single-argument, user-controlled format) with
`printf("%s", $FMT)` to neutralize format-string attacks.

```
rule cwe134_format_string {
  description: "Neutralize user-controlled format strings"

  pattern: printf($FMT)
  capture-taint: { var: $FMT, from: user_input }

  rewrite: printf("%s", $FMT)
}
```

### 6.10 CWE-125/CWE-787: Out-of-bounds read/write

**Inline fix — off-by-one in a loop bound:**

```
rule cwe787_off_by_one {
  description: "Fix >= to > in circular buffer bounds check"

  pattern-inside: $RET circular_buffer_put(...) { ... }
  pattern:        $IDX >= $SIZE

  rewrite: $IDX > $SIZE
}
```

**Inline fix — missing upper-bound check on array index:**

```
rule cwe125_array_bounds {
  pattern: $ARR[$IDX]
  where:   !($IDX < sizeof($ARR) / sizeof($ARR[0]))

  insert before: |
    if ($IDX >= sizeof($ARR) / sizeof($ARR[0])) return;
}
```

### 6.11 Runtime assertion at a call site

Emit a precondition check for `sprintf` without replacing the call.

```
rule sprintf_precond {
  pattern: sprintf($D, $F, $...REST)
  assert: nonnull($D) && sizeof($D) >= 32 && nonnull($F)
}
```

### 6.12 Static contract — formal pre/post-conditions

Attach verifier-consumable metadata to a function. No runtime code is
emitted — the attributes are consumed by KLEE or a SARIF exporter.

```
contract usb_endpoint_write_safety {
  pattern-inside: $RET usbd_ep_write_packet($DEV, $EP, $BUF, $LEN) { ... }

  requires: nonnull($DEV) && nonnull($BUF) && $LEN >= 0 && $LEN <= 512
  ensures:  @return >= 0 && @return <= 512
}
```

### 6.13 Patch + runtime hardening + static contract in one file

A single `.pdsl` file can carry all three: the patch rule, a runtime
hardening rule, and a static contract. Rules apply in file order;
the contract attaches metadata after rules reach a fixed point.

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

// Rule 1: patch (applied first)
rule usb_pre_validation {
  pattern-inside: $RET bl_usb__send_message(...) { ... }
  pattern:        usbd_ep_write_packet($DEV, $EP, $BUF, $LEN)

  insert before:
    call: usb::patch__before__usbd_ep_write_packet($DEV, $BUF)
}

// Rule 2: runtime hardening call (applied second)
rule usb_runtime_contract {
  pattern-inside: $RET bl_usb__send_message(...) { ... }
  pattern:        usbd_ep_write_packet($DEV, $EP, $BUF, $LEN)

  insert before:
    call: usbcheck::contract__before__test_contract($DEV, $EP)
}

// Static contract: formal preconditions (no runtime code)
contract usb_endpoint_write_static {
  pattern-inside: $RET bl_usb__send_message(...) { ... }
  pattern:        usbd_ep_write_packet($DEV, $EP, $BUF, $LEN)

  requires: nonnull($DEV) && $LEN >= 0 && $LEN <= 512
  ensures:  @return >= 0 && @return <= 512
}
```

### 6.14 Insert at entry/exit: instrumentation

Add tracing calls at the entry and every exit point of a function.

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

A `.pdsl` file travels through five phases before the patched `.cir`
is written. Each phase has a well-defined input, output, and failure
mode.

```
  [.pdsl source] → (1) Load → (2) Compile → (3) Match → (4) Rewrite → (5) Emit → [patched .cir]
                                   │                           │
                             external .c files          contract attrs / asserts
                             compiled to CIR            attached / inserted
                             (cached as .patchmod)
```

### 7.1 Load

1. **Parse.** The `.pdsl` file is tokenized with capture (`$X`),
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
with the original C source span, not the `.pdsl` site. Pattern codegen
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
(`rewrite:` / `remove:`) and any number of `insert …` actions
are applied to the match.

Ordering:

- **Within one rule:** actions apply in source order. `insert` actions
  run before the rewrite-style action, so an inserted before-call can
  read values that the rewrite will later erase.
- **Within one `.pdsl` file:** rules are considered in declaration
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

No runtime code is emitted by a `contract` block. Runtime hardening
(null checks, bounds guards, assertion calls) belongs in `rule`
blocks with `insert before:` / `insert after:` or `assert:`.

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
# Compile a .pdsl file to a shared pattern module
patchir-dslc rules/cwe190.pdsl -o rules/cwe190.patchmod

# Apply (drop-in replacement for -spec)
patchir-transform input.cir -dsl rules/cwe190.patchmod -o patched.cir
patchir-cir2llvm -S patched.cir -o patched.ll
```

During the migration window, `-spec` and `-dsl` may be combined; rules
from both sources load into the same `RewritePatternSet`.

Validate without running:

```sh
patchir-dslc rules/cwe190.pdsl --check      # parse + type-check only
patchir-dslc rules/cwe190.pdsl --dump-ir    # dump PatternIR
```

---

## 10. Error reporting

Every pattern token carries a source span. Diagnostics point at the
offending column:

```
rules/cwe190.pdsl:12:21: error: capture $R is used in rewrite: but not
                                  bound by any pattern
  rewrite:  $R = mul16::patch__replace__int_mul16($A, $B)
            ^
```

Semantic errors from analyses (e.g. unreachable `pattern-inside`) surface
at rule-compile time, not silently at match time.
