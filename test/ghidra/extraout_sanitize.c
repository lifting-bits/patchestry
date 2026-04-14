// Extraout sanitizer LIT test: exercise the --sanitize-extraout flag plumbing
// end-to-end through decompile-headless.sh. The function below is engineered
// to mimic the shape that produces extraout_rN in Ghidra's high p-code:
// the caller holds a pointer in r0/x0, calls a small helper, then
// dereferences the original pointer after the call. Ghidra is often unable
// to prove the helper preserves the argument register and invents an
// `extraout_` local for the post-call value.
//
// The sanitizer must:
//   1. Accept the --sanitize-extraout flag on the command line.
//   2. Run without errors.
//   3. Produce a JSON document that contains no `extraout_` substring
//      anywhere — either because Ghidra never generated one for this
//      code (in which case the assertion is trivially true) or because
//      the sanitizer successfully rewrote it.
//
// Negative control: the same function without the flag must still
// decompile cleanly (no CLI errors, document contains the function name).
//
// This test is aarch64 so Tier 2 (analytical callee walk) is enabled by
// default via the architecture allowlist in PcodeSerializer.
//
// UNSUPPORTED: system-windows

// RUN: %cc-aarch64 %s -g -O1 -c -o %t.o

// Baseline: without the sanitizer flag, the function decompiles and is
// present in the JSON output.
// RUN: %decompile-headless --input %t.o --function helper_caller --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=BASELINE %s --input-file %t
// BASELINE: "name":"{{_?helper_caller}}"

// Sanitizer on: the document still contains the function, and must not
// contain any extraout_ substring.
// RUN: %decompile-headless --input %t.o --function helper_caller --output %t --sanitize-extraout %ci_output_folder
// RUN: %file-check -vv --check-prefix=SANITIZED %s --input-file %t
// SANITIZED: "name":"{{_?helper_caller}}"
// SANITIZED-NOT: extraout_

// Analytical override: explicitly force Tier 2 on (already the default
// for aarch64, but this exercises the flag plumbing).
// RUN: %decompile-headless --input %t.o --function helper_caller --output %t --sanitize-extraout --sanitize-extraout-analytical on %ci_output_folder
// RUN: %file-check -vv --check-prefix=TIER2 %s --input-file %t
// TIER2: "name":"{{_?helper_caller}}"
// TIER2-NOT: extraout_

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;

// A tiny leaf helper that only touches its argument. ARM EABI / AAPCS64
// calling conventions place the first integer arg in x0/r0 and the
// callee is free to clobber it, but in practice the compiler often
// inlines such helpers. Declaring it noinline forces a real call site.
__attribute__((noinline))
static uint32_t to_len(uint8_t tag)
{
    return (uint32_t)tag + 2U;
}

// The caller holds `buf` in x0 across the call to `to_len`. Ghidra often
// flags this as extraout_x0 / extraout_r0 because it cannot prove the
// helper preserves the caller's argument register.
uint32_t helper_caller(uint8_t *buf)
{
    uint32_t len = to_len(*buf);
    return *(buf + len);
}
