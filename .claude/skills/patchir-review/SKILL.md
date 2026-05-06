---
name: patchir-review
description: >
  This skill should be used when the user runs "/patchir-review", asks to
  "review the PR for this repo", "run the repo PR review", "run autonomous 
  PR review", or when .github/workflows/claude-review.yml invokes the skill
  from CI. It performs a repo-specific review of C/C++, Bash, and Python changes
  in a pull request, prioritizes correctness and security over style, optionally
  uses clang-tidy for changed C/C++ files, and posts one inline GitHub comment per
  confirmed finding.
---

# patchir-review: Patchestry PR Review

Review one Patchestry PR diff. Post confirmed findings as inline GitHub
comments; do not print review prose as chat output. Optimize for bugs that can
break firmware lifting, CIR/MLIR transformations, patch application, lowering,
or verification flows.

## Required Inputs

When invoked from `claude-review.yml`, read:

- `REPO` - `${{ github.repository }}`
- `PR NUMBER` - `${{ github.event.pull_request.number || github.event.issue.number }}`

Use those values for every `gh` and `gh api` call. Substitute `<REPO>` and
`<PR>` below.

## Source Access

Treat the local checkout as the base branch, not the PR head. Read PR contents
through GitHub:

```sh
gh pr diff <PR>
gh pr view <PR> --json headRefOid --jq .headRefOid
gh api repos/<REPO>/contents/<path>?ref=<sha>
```

Do not cite local `Read`, `Grep`, or `Glob` output as PR-head source. Use the
API for any file content that supports a finding.

## Scope Check

Review only executable source files:

| Language | Extensions |
|---|---|
| C / C++ | `.c`, `.h`, `.cpp`, `.hpp`, `.cxx`, `.hxx`, `.cc`, `.hh`, `.inc` |
| Bash | `.sh`, `.bash` |
| Python | `.py`, `.pyi`, `.pyx` |

Skip YAML, JSON, Markdown, generated artifacts, binary outputs, and non-executable
test fixtures. Review test code only when the test itself contains executable 
C/C++/Bash/Python logic whose bug can hide or invert coverage.

Run this first:

```sh
gh pr view <PR> --json files \
  --jq '[.files[] | select(.path | test("\\.(c|h|cpp|hpp|cxx|hxx|cc|hh|inc|sh|bash|py|pyi|pyx)$"; "i"))] | length'
```

If the count is zero, post the skip comment and stop:

```sh
gh pr comment <PR> --body "No C/C++/Bash/Python files changed - review skipped."
```

Do not post inline findings or submit a review action when scope is empty.

## Review Procedure

Read the full enclosing function, class method, or script block for every
changed hunk. Trace concrete values through the changed path before flagging. If
a header or public interface changes, inspect affected call sites at PR head.
Fetch supporting call-site source through `gh api` at the PR head SHA. Treat
`AGENTS.md` as authoritative for repository conventions, but only comment on
convention drift when it can cause real maintenance risk.

Classify each finding into exactly one tier.

## clang-tidy Assist

Use `clang-tidy` as optional evidence for changed C/C++ files only when the
current checkout already has a compile database. GitHub review jobs often do not
have a configured build directory, so do not run CMake, build the project, or
generate `compile_commands.json` during review. Prefer existing compile
databases in this order:

```sh
builds/ci/compile_commands.json
builds/default/compile_commands.json
compile_commands.json
```

Run it only on changed implementation/header files that appear in or can be
resolved by an existing compile database. Skip `clang-tidy` if no compile
database, no matching entry, no repository `.clang-tidy`, or no `clang-tidy`
binary is available.

Use the repository `.clang-tidy` config when present:

```sh
clang-tidy <file> -p <build-dir>
```

If no `.clang-tidy` config is present at or above the changed file, skip
`clang-tidy` rather than inventing a separate check profile.

Treat `clang-tidy` output as investigation leads, not automatic findings. Post a
`clang-tidy`-derived finding only after confirming it against the PR-head source
and proving it is a `CRITICAL` or `BUG` issue. Ignore purely stylistic findings,
broad modernization advice, generated-code noise, and warnings outside changed
lines unless the unchanged line becomes wrong because of the PR.

## CRITICAL

Use `CRITICAL` for crash, memory-unsafety, undefined behavior, data corruption,
or command execution risk.

**C / C++:**
- Use-after-free, double-free, dangling reference/view, or returning an address
  or reference tied to a temporary/local object.
- Buffer overflow, out-of-bounds access, unchecked sentinel index
  (`SIZE_MAX`, `-1`, `npos`), or invalid pointer arithmetic.
- Reachable null dereference, including unchecked `dyn_cast`/`dynamic_cast` results.
- Undefined behavior: signed overflow in required arithmetic, invalid shift,
  division by zero, strict-aliasing violation, uninitialized read,
  unsequenced modification, or invalid `const_cast`.
- Iterator or reference invalidation after container mutation.
- Incorrect allocator or deallocator pairing or leaked ownership transfer bugs that
  can produce a double-free or permanent ownership loss.
- Infinite loop or unbounded recursion in CFG traversal, graph search,
  decompilation, transform passes, or Ghidra serialization.
- Unsafe shell execution, non-literal format string, or command construction
  from PR/runtime input.

**Bash:**
- `eval`, backticks, or `bash -c` executed on PR-derived or otherwise
  untrusted input.
- Unquoted variables in paths or commands where word splitting or glob
  expansion can delete, overwrite, or execute the wrong target.
- `rm -rf` or cleanup paths that can resolve to an empty string, `/`,
  the repository root, or a caller-controlled directory.
- Hardcoded temporary paths vulnerable to race conditions or symlink attacks;
  use `mktemp`.
- `curl | bash`, `wget | sh`, or similar direct execution of untrusted
  remote content.

**Python:**
- `eval`, `exec`, unsafe `pickle` usage, unsafe YAML deserialization, or
  command injection through `subprocess`, `os.system`, or shell execution.
- Path traversal during file open, write, extract, move, or delete operations
  using externally controlled paths.
- TOCTOU filesystem checks guarding a security-sensitive operation.
- `assert` used for security, validation, invariant, or bounds checks that
  must remain enforced under `python -O`.

## BUG

Use `BUG` for wrong behavior, missed patching, incorrect verification results,
or test/build logic that can silently report success after a failure.

**Patchestry-specific C / C++:**
- Incorrect CIR/MLIR operation matching, insertion point, dominance, type,
  attribute, address-space, or symbol handling.
- Ghidra P-Code deserialization or serialization mismatches, endian or width
  errors, address truncation, varnode identity mix-ups, or function/block edge
  loss.
- Contract metadata attached to the wrong operation or dropped during lowering.
- Patch specification parsing that accepts invalid YAML semantics or rejects
  valid specifications.
- Diagnostics that report the wrong location, suppress a real error, or allow a
  partially transformed module to continue as success.

**General C / C++:**
- Off-by-one errors, inverted conditions, missing error branches, incorrect
  comparators, operator precedence bugs, or accidental fallthrough.
- Resource leaks on error paths involving file descriptors, raw allocations,
  locks, Ghidra/MLIR/LLVM-owned handles, or temporary files.
- Ignored return values or status codes where failure changes behavior.
- Integer overflow or truncation in size, address, offset, or bit-width
  computations.
- Incorrect move, copy, lifetime, or ownership transfer semantics that change
  behavior.
- Missing virtual `override` only when the intended override is not invoked.

**Bash:**
- Missing `set -euo pipefail` or unchecked command status in scripts that drive
  builds, tests, Ghidra, KLEE, firmware examples, or artifact publishing.
- Pipelines, subshells, or assignment patterns that hide failing commands.
- Word splitting or glob expansion issues involving filenames or paths,
  especially under `scripts/`, `test/`, or workflow helpers.
- Non-portable shell syntax when the shebang is `/bin/sh`.

**Python:**
- Mutable default arguments, mutation during iteration, resource leaks, broad
  exception handling that converts failure into success, or `is` used for value
  comparison.
- Incorrect JSON or YAML schema interpretation, address or width conversion,
  path handling, or subprocess status propagation.
- Test generators or artifact scripts that can produce stale, incomplete, or
  misleading outputs.

## WARNING

Use `WARNING` sparingly for maintainability risk that is likely to cause future
review or debugging cost in Patchestry.

- New public interface, pass option, YAML field, or serialized schema lacks
  nearby validation, diagnostics, or documentation.
- Behavior crosses a documented module boundary in `AGENTS.md` without updating
  the corresponding interface documentation.
- Error messages omit enough context to debug patch specs, operation names,
  addresses, function names, or file paths.
- A changed script assumes a host tool, Docker state, architecture, or build
  directory that conflicts with documented workflows.
- A changed test is too weak to fail for the behavior it claims to cover.

Do not post `WARNING` for magic numbers, TODO wording, minor naming taste,
generic "could be cleaner" comments, or broad style preferences.

## SIMPLIFY

Use `SIMPLIFY` only for a mechanical simplification that removes real review
burden and is clearly safe. Cap at 3 per review.

- Redundant `std::move` on a local return value.
- Redundant null check before `delete`.
- Redundant branch after `return`, `continue`, `break`, or `throw` when removing
  it makes the control flow clearer.
- Duplicate block in changed code that can be extracted without changing
  ownership, lifetime, or diagnostics.
- Shell `if [ $? -eq 0 ]` that can be replaced with `if cmd; then`.

Do not suggest broad modernization (`enum class`, `pathlib`, f-strings,
algorithm rewrites, `std::format`, include cleanup) unless it directly removes
a bug or a concrete maintenance hazard in the changed code.

## Output Rules

- Post one inline comment per finding via
  `mcp__github_inline_comment__create_inline_comment` with `confirmed: true`.
- Use this exact body format:
  `[CRITICAL|BUG|WARNING|SIMPLIFY] <one-sentence description>`
- For `SIMPLIFY`, name the replacement explicitly.
- Post no praise, summary, methodology, notes, nested bullets, or tier
  definitions.
- Do not post uncertain findings. If reproducing or proving the issue requires
  assumptions not supported by PR-head code, skip it.
- Post only on changed lines, or on lines the inline-comment tool accepts for
  the PR diff. If a supporting unchanged line reveals the issue, anchor the
  comment on the changed line that introduced or exposed it.
- Prefer fewer high-confidence comments over a complete checklist dump.

## Final Verdict

Decide on a single final review action after inline comments are posted.
`WARNING` and `SIMPLIFY` do not block approval.

1. If there are zero `CRITICAL` and zero `BUG` findings, submit an approval:

   ```sh
   gh pr review <PR> --approve --body "No blocking issues found."
   ```

   If approval is rejected, fall back to:

   ```sh
   gh pr comment <PR> --body "No blocking issues found. (Self-approval blocked by GitHub.)"
   ```

2. If there is one or more `CRITICAL` or `BUG` finding, do nothing further. The
   inline comments stand on their own.

If a confirmed blocking finding cannot be posted inline, leave a normal PR
comment describing only that blocking issue and do not approve.

Do not submit request-changes or comment-tier reviews. Approval is only for
case 1. Stop after the verdict step.
