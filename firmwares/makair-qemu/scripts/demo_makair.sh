#!/usr/bin/env bash
# End-to-end MakAir-QEMU patching demo.
#
# Pipeline:
#   ELF      -> decompile-headless          -> P-Code JSON
#   JSON     -> patchir-decomp              -> CIR
#   CIR      -> patchir-transform           -> patched CIR + contract metadata
#   CIR      -> patchir-cir2llvm            -> LLVM IR
#   LLVM IR  -> patchir-klee-verifier       -> KLEE harness
#   harness  -> llvm-as -> run-klee.sh      -> KLEE state dirs (PASS = zero *.err)
#   patched IR + ELF + Patcherex2           -> patched ELF (gated by --with-patcherex)
#
# Stage flags (cumulative): build < decomp < cir < patch < lower < klee < patche < verify
#
# v0: target functions and patch specs are PLACEHOLDERS — pick `sendBootMessage`
# (a clean small function in srcs/telemetry.cpp) for the decomp/transform/lower
# round-trip; the patch is a no-op pass-through. Real patch shapes (Control-frame
# CRC bounds-check, alarm-trap dispatcher hardening) land in a follow-up once
# the v0 pipeline wiring is verified.
#
# Usage:
#   ./demo_makair.sh                              # run all stages up to KLEE
#   ./demo_makair.sh --stage=lower                # stop before KLEE
#   ./demo_makair.sh --stage=all --with-patcherex # full chain incl. ELF instrumentation
#   ./demo_makair.sh --skip-klee                  # bypass KLEE if local toolchain is unhappy
#
# Env overrides:
#   PATCHIR_BIN_DIR   directory containing patchir-* (default: builds/run/bin/
#                     if present, else builds/default/tools/*/Release/*).
#   PATCHEREX_BIN     path to Patcherex2's patche_binary.sh wrapper (auto-clones
#                     if missing).
#   ELF               path to makair.elf (default: build/makair.elf).
#   TARGET_FUNCTION   function to lift through patchestry (default: sendBootMessage).

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
firmware_dir=$(cd "${script_dir}/.." && pwd)
repo_root=$(cd "${firmware_dir}/../.." && pwd)
out_dir_default="${firmware_dir}/build/demo"

ELF_DEFAULT="${firmware_dir}/build/makair.elf"
ELF="${ELF:-${ELF_DEFAULT}}"
TARGET_FUNCTION="${TARGET_FUNCTION:-sendBootMessage}"

stage_target="klee"
keep_outputs=0
with_patcherex=0
skip_klee=0
out_dir="${out_dir_default}"

for arg in "$@"; do
    case "$arg" in
        --stage=*)         stage_target="${arg#--stage=}" ;;
        --out-dir=*)       out_dir="${arg#--out-dir=}" ;;
        --keep)            keep_outputs=1 ;;
        --with-patcherex)  with_patcherex=1 ;;
        --skip-klee)       skip_klee=1 ;;
        -h|--help)
            sed -n '2,30p' "$0" >&2
            exit 2
            ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

case "$stage_target" in
    build|decomp|cir|patch|lower|klee|patche|verify|all) ;;
    *) echo "invalid --stage=$stage_target" >&2; exit 2 ;;
esac

if [[ "$stage_target" == "all" ]]; then
    stage_target=verify
    [[ "$with_patcherex" -eq 1 ]] || {
        echo "--stage=all implies --with-patcherex; binary patching is required for verify." >&2
        exit 2
    }
fi

# ---- tool discovery (reuses secpump-qemu's resolution scheme) ------------
if [[ -z "${PATCHIR_BIN_DIR:-}" ]]; then
    if [[ -x "${repo_root}/builds/run/bin/patchir-decomp" ]]; then
        PATCHIR_BIN_DIR="${repo_root}/builds/run/bin"
    else
        PATCHIR_BIN_DIR=""
    fi
fi

resolve_tool() {
    local name="$1"
    if [[ -n "${PATCHIR_BIN_DIR}" && -x "${PATCHIR_BIN_DIR}/${name}" ]]; then
        echo "${PATCHIR_BIN_DIR}/${name}"; return
    fi
    local fallback="${repo_root}/builds/default/tools/${name}/Release/${name}"
    if [[ -x "$fallback" ]]; then echo "$fallback"; return; fi
    echo "demo_makair: cannot find ${name} — set PATCHIR_BIN_DIR." >&2
    exit 1
}

PATCHIR_DECOMP=$(resolve_tool patchir-decomp)
PATCHIR_TRANSFORM=$(resolve_tool patchir-transform)
PATCHIR_CIR2LLVM=$(resolve_tool patchir-cir2llvm)
PATCHIR_KLEE=$(resolve_tool patchir-klee-verifier)
DECOMP_HEADLESS="${repo_root}/scripts/ghidra/decompile-headless.sh"
RUN_KLEE="${repo_root}/scripts/klee/run-klee.sh"

banner() { printf '\n=== %s ===\n' "$1" >&2; }

stage_le() {
    local order=(build decomp cir patch lower klee patche verify)
    local x_idx=-1 y_idx=-1
    for i in "${!order[@]}"; do
        [[ "${order[$i]}" == "$1" ]] && x_idx=$i
        [[ "${order[$i]}" == "$2" ]] && y_idx=$i
    done
    [[ $x_idx -le $y_idx ]]
}

if [[ "$keep_outputs" -eq 0 ]]; then
    rm -rf "$out_dir"
fi
mkdir -p "$out_dir"

if stage_le build "$stage_target"; then
    banner "stage 0/8: build firmware"
    if [[ -f "$ELF" ]]; then
        echo "  re-using $(basename "$ELF") (delete to force rebuild)" >&2
    else
        ./build-docker.sh
    fi
fi
[[ -f "$ELF" ]] || { echo "makair.elf missing at $ELF" >&2; exit 1; }

if stage_le decomp "$stage_target"; then
    banner "stage 1/8: decompile $TARGET_FUNCTION"
    bash "$DECOMP_HEADLESS" --input "$ELF" \
        --function "$TARGET_FUNCTION" \
        --output "$out_dir/${TARGET_FUNCTION}.json" >/dev/null
fi

if stage_le cir "$stage_target"; then
    banner "stage 2/8: JSON -> CIR"
    "$PATCHIR_DECOMP" -input "$out_dir/${TARGET_FUNCTION}.json" \
        -emit-cir -output "$out_dir/${TARGET_FUNCTION}"
fi

if stage_le patch "$stage_target"; then
    banner "stage 3/8: apply patch spec (placeholder pass-through)"
    if [[ -f "${repo_root}/test/patchir-transform/makair_${TARGET_FUNCTION}.yaml" ]]; then
        "$PATCHIR_TRANSFORM" "$out_dir/${TARGET_FUNCTION}.cir" \
            --spec "${repo_root}/test/patchir-transform/makair_${TARGET_FUNCTION}.yaml" \
            -o "$out_dir/${TARGET_FUNCTION}_patched.cir"
    else
        echo "  no makair_${TARGET_FUNCTION}.yaml spec yet — passing CIR through unchanged" >&2
        cp "$out_dir/${TARGET_FUNCTION}.cir" "$out_dir/${TARGET_FUNCTION}_patched.cir"
    fi
fi

if stage_le lower "$stage_target"; then
    banner "stage 4/8: CIR -> LLVM IR"
    "$PATCHIR_CIR2LLVM" -S "$out_dir/${TARGET_FUNCTION}_patched.cir" \
                        -o "$out_dir/${TARGET_FUNCTION}_patched.ll"
fi

if stage_le klee "$stage_target"; then
    banner "stage 5/8: build libc model + KLEE harness"
    if [[ ! -f "$out_dir/libc_models.bc" ]]; then
        clang -emit-llvm -c -O0 -ffreestanding \
            -target x86_64-unknown-linux-gnu \
            "${repo_root}/klee-demo/libc_models.c" \
            -o "$out_dir/libc_models.bc"
    fi

    "$PATCHIR_KLEE" "$out_dir/${TARGET_FUNCTION}_patched.ll" \
        --target-function "$TARGET_FUNCTION" \
        --model-library "$out_dir/libc_models.bc" \
        -S -o "$out_dir/${TARGET_FUNCTION}_harness.ll"

    if [[ "$skip_klee" -eq 1 ]]; then
        echo "  --skip-klee: harness generated, skipping symbolic execution." >&2
    elif command -v llvm-as >/dev/null 2>&1; then
        # Inject the klee_make_symbolic marker into the head of the .ll so
        # the entrypoint-heuristic in klee-entrypoint.sh skips uclibc.
        python3 - "$out_dir/${TARGET_FUNCTION}_harness.ll" <<'PYEOF'
import sys
p = sys.argv[1]
with open(p) as f: text = f.read()
marker = '\n@.klee_runtime_marker = private constant [19 x i8] c"klee_make_symbolic\\00"\n'
needle = 'target triple ='
i = text.find(needle)
if i < 0 or marker.strip() in text:
    sys.exit(0)
end = text.find('\n', i) + 1
with open(p, 'w') as f:
    f.write(text[:end] + marker + text[end:])
PYEOF
        llvm-as "$out_dir/${TARGET_FUNCTION}_harness.ll" -o "$out_dir/${TARGET_FUNCTION}_harness.bc"
        rm -rf "$out_dir/klee_out"
        klee_rc=0
        bash "$RUN_KLEE" --run-bitcode \
            --input  "$out_dir/${TARGET_FUNCTION}_harness.bc" \
            --output "$out_dir/klee_out" \
            || klee_rc=$?
        if [[ "$klee_rc" -ne 0 ]]; then
            echo "KLEE exited non-zero (rc=$klee_rc) — interpreter crashed before producing artefacts." >&2
            echo "Re-run with --skip-klee to bypass." >&2
            exit 1
        fi
        errs=$(find "$out_dir/klee_out" -name '*.err' | wc -l | tr -d ' ')
        ktests=$(find "$out_dir/klee_out" -name 'test*.ktest' | wc -l | tr -d ' ')
        if [[ "$errs" -ne 0 ]]; then
            echo "KLEE found $errs error states under $out_dir/klee_out:" >&2
            find "$out_dir/klee_out" -name '*.err' >&2
            exit 1
        fi
        if [[ "$ktests" -eq 0 ]]; then
            echo "KLEE produced no test cases — interpreter likely crashed before exploring any path." >&2
            exit 1
        fi
        echo "  KLEE: $ktests test case(s), 0 error states (PASS)" >&2
    else
        echo "  llvm-as not on PATH; skipping KLEE invocation." >&2
    fi
fi

if stage_le patche "$stage_target"; then
    if [[ "$with_patcherex" -ne 1 ]]; then
        echo "  skipping binary patching (--with-patcherex not set)" >&2
    else
        banner "stage 6/8: apply patched IR to ELF via Patcherex2"
        if [[ -z "${PATCHEREX_BIN:-}" ]]; then
            for cand in \
                "${repo_root}/../Patcherex2/patche_binary.sh" \
                "${HOME}/Patcherex2/patche_binary.sh"; do
                [[ -f "$cand" ]] && PATCHEREX_BIN="$cand" && break
            done
        fi
        if [[ -z "${PATCHEREX_BIN:-}" || ! -f "$PATCHEREX_BIN" ]]; then
            sibling="${repo_root}/../Patcherex2"
            if [[ ! -d "$sibling" ]]; then
                banner "stage 6a: cloning Patcherex2 (patche_support branch)"
                command -v git >/dev/null \
                    || { echo "git not found; install or set PATCHEREX_BIN." >&2; exit 1; }
                git clone --depth=1 --branch patche_support \
                    https://github.com/trail-of-forks/Patcherex2.git "$sibling"
            fi
            PATCHEREX_BIN="${sibling}/patche_binary.sh"
        fi
        bash "$PATCHEREX_BIN" "$ELF" "$out_dir/${TARGET_FUNCTION}_patched.ll" \
            "$TARGET_FUNCTION"

        patcherex_dir=$(cd "$(dirname "$PATCHEREX_BIN")" && pwd)
        patched_src="${patcherex_dir}/outs/$(basename "$ELF")_patched"
        patched_dst="${firmware_dir}/build/makair-patched.elf"
        if [[ -f "$patched_src" ]]; then
            cp "$patched_src" "$patched_dst"
            chmod +x "$patched_dst"
            echo "  patched ELF -> $patched_dst" >&2
        else
            echo "Patcherex2 finished but $patched_src is missing." >&2
            exit 1
        fi
    fi
fi

if stage_le verify "$stage_target"; then
    banner "stage 7/8: validate patched firmware"
    if [[ "$with_patcherex" -eq 1 && -f "${firmware_dir}/build/makair-patched.elf" ]]; then
        python3 "${firmware_dir}/tests/test_patched_protocol.py"
    else
        echo "  skipping (no patched ELF available)" >&2
    fi
fi

echo
echo "demo_makair: stage '$stage_target' completed; artefacts under $out_dir" >&2
