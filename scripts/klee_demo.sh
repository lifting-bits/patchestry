#!/usr/bin/env bash
# Usage: demo.sh --binary <bin> --function <fn> --spec <yaml>
#
# Pipeline:
#   binary -> Ghidra headless -> JSON
#   JSON   -> patchir-decomp  -> CIR
#   CIR    -> patchir-transform (with <yaml>) -> patched CIR
#   CIR    -> patchir-cir2llvm  -> LLVM IR
#   lib/patchestry/klee/models/*.c -> clang (-target x86_64-unknown-linux-gnu)
#                                  -> llvm-link -> models.bc
#   LLVM   -> patchir-klee-verifier (+ models.bc) -> harness LL
#   .ll    -> llvm-as -> .bc -> run-klee.sh (Docker)
#   if klee finds zero errors:
#     patche_binary.sh -> apply patched IR to original binary
#
# Env overrides:
#   PATCHEREX_BIN     path to patche_binary.sh (default: Patcherex2/patche_binary.sh)
#   PATCH_FUNCTIONS   function mapping passed to patcherex (default: <fn>,patch__replace__sprintf)
#   PATCHESTRY_ROOT   patchestry checkout (default: /home/akshayk/patchestry)

set -euo pipefail

usage() {
    awk '
        NR == 1 { next }
        /^[^#]/  { exit }
        { sub(/^# ?/, ""); print }
    ' "$0" >&2
    exit "${1:-2}"
}

BIN=""
FN=""
SPEC=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)   BIN="$2"; shift 2 ;;
        --function) FN="$2";  shift 2 ;;
        --spec)     SPEC="$2"; shift 2 ;;
        -h|--help)  usage 0 ;;
        *)          echo "unknown flag: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$BIN"  ]] || { echo "missing --binary"  >&2; usage 1; }
[[ -n "$FN"   ]] || { echo "missing --function" >&2; usage 1; }
[[ -n "$SPEC" ]] || { echo "missing --spec"     >&2; usage 1; }

abspath() {
    local p="$1"
    if command -v readlink >/dev/null && readlink -f / >/dev/null 2>&1; then
        readlink -f -- "$p"
    else
        local d b
        d="$(cd "$(dirname -- "$p")" && pwd)"
        b="$(basename -- "$p")"
        echo "$d/$b"
    fi
}

[[ -f "$BIN"  ]] || { echo "binary not found: $BIN" >&2; exit 1; }
[[ -f "$SPEC" ]] || { echo "spec not found: $SPEC"  >&2; exit 1; }
BIN="$(abspath "$BIN")"
SPEC="$(abspath "$SPEC")"

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${PATCHESTRY_ROOT:-$(cd "$DIR/.." && pwd)}"
OUT="$DIR/output"
mkdir -p "$OUT"

# Per-run label, derived from the spec's basename so successive runs in
# the same output dir do not clobber each other.
LABEL="$(basename "$SPEC" .yaml)"

DECOMP="$ROOT/builds/run/bin/patchir-decomp"
TRANSFORM="$ROOT/builds/run/bin/patchir-transform"
CIR2LLVM="$ROOT/builds/run/bin/patchir-cir2llvm"
VERIFIER="$ROOT/builds/default/tools/patchir-klee-verifier/Release/patchir-klee-verifier"
STRIP="$ROOT/test/scripts/strip-json-comments.sh"
GHIDRA="$ROOT/scripts/ghidra/decompile-headless.sh"
RUN_KLEE="$ROOT/scripts/klee/run-klee.sh"
# Patcherex2 is a sibling repo by convention; if it isn't cloned next to
# patchestry the script auto-fetches the trail-of-forks fork (branch
# patche_support) which carries patche_binary.sh + patche_firmware.py.
# patche_binary.sh accepts an optional function-mapping argument; when
# empty it processes ALL functions in the supplied LLVM IR (modify
# existing, insert new). The cir2llvm output contains exactly the patched
# target plus any inserted patch helper bodies, so the empty default is
# correct for both the safe (patches replace call sites) and degenerate
# (no patches matched) cases. Override via PATCH_FUNCTIONS only when a
# spec produces extra IR symbols you do not want patched.
PATCHEREX="${PATCHEREX_BIN:-$(dirname "$ROOT")/Patcherex2/patche_binary.sh}"
PATCH_FNS="${PATCH_FUNCTIONS:-}"
PATCHEREX_REPO_URL="${PATCHEREX_REPO_URL:-https://github.com/trail-of-forks/Patcherex2.git}"
PATCHEREX_REPO_REF="${PATCHEREX_REPO_REF:-patche_support}"

# Symbolic model sources live under lib/patchestry/klee/models/ alongside
# klee_stub.h. The verifier requires the linked model library to match the
# harness triple + datalayout (KLEE runs on x86_64-unknown-linux-gnu in
# docker), so always cross-compile to that target regardless of host.
MODEL_DIR="$ROOT/lib/patchestry/klee/models"
MODEL_TARGET="x86_64-unknown-linux-gnu"
MODEL_SOURCES=(
    "$MODEL_DIR/libc_models.c"
    "$MODEL_DIR/usb_hal_models.c"
)

cd "$DIR"
export PATCHESTRY_ROOT="$ROOT"

# Banner helper: prints a fat divider so each tool's output is easy to
# pick out when scrolling. The banner echoes the stage number, tool name
# and the artifact it produces; tools then run with full stdout/stderr.
banner() {
    local stage="$1" title="$2" detail="${3:-}"
    printf '\n'
    printf '################################################################\n'
    printf '## [%s] %s\n' "$stage" "$title"
    [[ -n "$detail" ]] && printf '## %s\n' "$detail"
    printf '################################################################\n'
}

banner "1/5" "ghidra :: decompile-headless.sh" \
       "extract $FN from $BIN -> $OUT/extract.json"
"$GHIDRA" --input "$BIN" --function "$FN" --output "$OUT/extract.json"

banner "2/5" "patchir-decomp" \
       "$OUT/in.json -> $OUT/decomp.cir"
bash "$STRIP" "$OUT/extract.json" > "$OUT/in.json"
"$DECOMP" -input "$OUT/in.json" -use-structuring-pass -emit-cir -output "$OUT/decomp"

banner "3/5" "clang :: KLEE model library" \
       "$MODEL_DIR/{libc_models,usb_hal_models}.c -> $OUT/models.bc (target $MODEL_TARGET)"
# Compile each model source to its own bitcode, then llvm-link them into a
# single model library. -include klee_stub.h matches the CMake rule under
# lib/patchestry/klee/models/, so the standalone build here produces the
# same artifacts as the in-tree build.
declare -a MODEL_BC_FILES
for src in "${MODEL_SOURCES[@]}"; do
    [[ -f "$src" ]] || { echo "model source missing: $src" >&2; exit 1; }
    bc="$OUT/$(basename "${src%.c}").bc"
    clang -emit-llvm -c -O0 -ffreestanding \
        -target "$MODEL_TARGET" \
        -include "$MODEL_DIR/klee_stub.h" \
        "$src" -o "$bc"
    MODEL_BC_FILES+=("$bc")
done
MODEL_LIB="$OUT/models.bc"
llvm-link -o "$MODEL_LIB" "${MODEL_BC_FILES[@]}"

banner "4a/5" "patchir-transform" \
       "spec=$(basename "$SPEC") -> $OUT/${LABEL}.cir"
"$TRANSFORM" "$OUT/decomp.cir" --spec "$SPEC" -o "$OUT/${LABEL}.cir"

banner "4b/5" "patchir-cir2llvm" \
       "$OUT/${LABEL}.cir -> $OUT/${LABEL}.ll"
"$CIR2LLVM" -S "$OUT/${LABEL}.cir" -o "$OUT/${LABEL}.ll"

banner "4c/5" "patchir-klee-verifier" \
       "harness -> $OUT/${LABEL}_harness.ll"
"$VERIFIER" "$OUT/${LABEL}.ll" --target-function "$FN" \
    --model-library "$MODEL_LIB" -v -S \
    -o "$OUT/${LABEL}_harness.ll"

banner "4d/5" "llvm-as" \
       "$OUT/${LABEL}_harness.ll -> $OUT/${LABEL}_harness.bc"
llvm-as "$OUT/${LABEL}_harness.ll" -o "$OUT/${LABEL}_harness.bc"

banner "5/5" "klee :: run-klee.sh (docker)" \
       "$OUT/${LABEL}_harness.bc -> $OUT/${LABEL}_klee/"
rm -rf "$OUT/${LABEL}_klee"
"$RUN_KLEE" --run-bitcode --input "$OUT/${LABEL}_harness.bc" \
    --output "$OUT/${LABEL}_klee"

errs=$(find "$OUT/${LABEL}_klee" -name '*.err' 2>/dev/null | wc -l)

# patch_status tracks the outcome of the patcherex stage so the summary
# can report it accurately rather than inferring from `errs` alone.
#   not-run        : not reached
#   ok             : patcherex completed and (if available) verify_patched
#                    accepted the result
#   failed         : patcherex script ran and exited non-zero
#   verify-failed  : patcherex produced a binary but verify_patched.sh failed
#   missing        : PATCHEREX path not found / not executable
#   skipped        : klee reported errors; patching deliberately skipped
patch_status="not-run"
patched_bin=""

if [[ "$errs" -eq 0 ]]; then
    # Auto-clone Patcherex2 if the sibling checkout is missing. We only
    # clone when the parent directory genuinely doesn't exist, to avoid
    # clobbering a hand-managed checkout that might have local edits or
    # be on a different branch. A clone failure is non-fatal; the
    # subsequent -f check downgrades the stage to `missing`. Use -f
    # rather than -x because patche_binary.sh is invoked via `bash` and
    # doesn't need its execute bit set (the upstream tarball ships
    # without it on some platforms).
    patcherex_dir=$(dirname "$PATCHEREX")
    if [[ ! -f "$PATCHEREX" && ! -d "$patcherex_dir" ]]; then
        banner "6a/5" "patcherex2 :: clone" \
               "$PATCHEREX_REPO_URL (branch $PATCHEREX_REPO_REF) -> $patcherex_dir"
        git clone --depth 1 --branch "$PATCHEREX_REPO_REF" \
            "$PATCHEREX_REPO_URL" "$patcherex_dir" \
            || printf 'patcherex clone failed; continuing without patching\n' >&2
    fi

    if [[ -f "$PATCHEREX" ]]; then
        banner "6/5" "patcherex2 :: patche_binary.sh" \
               "0 klee errors -> apply $OUT/${LABEL}.ll to $BIN${PATCH_FNS:+ (functions: $PATCH_FNS)}"
        # Capture exit status without tripping `set -e`. patche_binary.sh
        # treats an empty function-mapping argument as "process all", so
        # only forward $PATCH_FNS when the user explicitly set it.
        if [[ -n "$PATCH_FNS" ]]; then
            bash "$PATCHEREX" "$BIN" "$OUT/${LABEL}.ll" "$PATCH_FNS" \
                && patch_status="ok" || patch_status="failed"
        else
            bash "$PATCHEREX" "$BIN" "$OUT/${LABEL}.ll" \
                && patch_status="ok" || patch_status="failed"
        fi

        # patche_binary.sh writes the patched ELF next to itself in outs/
        # named `<original-basename>_patched`. Verify the pair when the
        # sibling verify_patched.sh script is available; treat verify
        # failure as a soft signal in the summary, not a fatal error.
        if [[ "$patch_status" == "ok" ]]; then
            patched_bin="$patcherex_dir/outs/$(basename "$BIN")_patched"
            verify_sh="$patcherex_dir/verify_patched.sh"
            if [[ -f "$verify_sh" && -f "$patched_bin" ]]; then
                banner "7/5" "patcherex2 :: verify_patched.sh" \
                       "verify $patched_bin against $BIN"
                bash "$verify_sh" "$BIN" "$patched_bin" \
                    || patch_status="verify-failed"
            elif [[ ! -f "$patched_bin" ]]; then
                printf 'note: expected patched binary not found at %s\n' "$patched_bin"
                patched_bin=""
            fi
        fi
    else
        banner "6/5" "patcherex2 SKIPPED" \
               "patche_binary.sh not found at $PATCHEREX (set PATCHEREX_BIN to override)"
        patch_status="missing"
    fi
else
    banner "6/5" "patcherex2 SKIPPED" \
           "klee found $errs error(s); refusing to patch the binary"
    patch_status="skipped"
fi

banner "summary" "$LABEL"
printf 'klee error reports : %s\n' "$errs"
printf 'klee output dir    : %s\n' "$OUT/${LABEL}_klee"
if [[ "$errs" -gt 0 ]]; then
    printf 'errors:\n'
    find "$OUT/${LABEL}_klee" -name '*.err' -printf '  %f\n' 2>/dev/null | sort
fi
case "$patch_status" in
    ok)
        printf 'binary patched     : ok\n'
        [[ -n "$patched_bin" ]] && printf 'patched binary     : %s\n' "$patched_bin"
        ;;
    verify-failed)
        printf 'binary patched     : produced but verify_patched.sh reported issues\n'
        [[ -n "$patched_bin" ]] && printf 'patched binary     : %s\n' "$patched_bin"
        ;;
    failed)
        printf 'binary patched     : FAILED (patcherex returned non-zero)\n'
        ;;
    missing)
        printf 'binary patched     : skipped (patcherex script not found)\n'
        ;;
    skipped)
        printf 'binary patched     : skipped (klee found %s error(s))\n' "$errs"
        ;;
    *)
        printf 'binary patched     : not-run\n'
        ;;
esac
