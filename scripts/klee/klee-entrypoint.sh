#!/bin/bash
# Entrypoint for the KLEE Docker container.
#
# Modes:
#   --run-harness   Compile a C harness to LLVM bitcode and run KLEE on it.
#   --run-bitcode   Run KLEE directly on a pre-compiled .bc file.
#
# Options:
#   --input <file>          Input C harness or .bc file (required)
#   --output <dir>          Output directory for KLEE results (default: /work/klee-out)
#   --klee-args <args>      Additional arguments to pass to KLEE
#   --clang-args <args>     Additional arguments to pass to clang (compile mode)
#   --max-time <seconds>    Maximum KLEE execution time (default: 300)
#   --solver <backend>      Solver backend: stp, z3, or stp:z3 (default: stp:z3)

set -euo pipefail

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
INPUT_FILE=""
OUTPUT_DIR="/work/klee-out"
MODE="run-harness"
MAX_TIME=300
SOLVER_BACKEND="stp:z3"
EXTRA_KLEE_ARGS=()
EXTRA_CLANG_ARGS=()

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
show_help() {
    cat <<'USAGE'
Usage: klee-entrypoint.sh [MODE] --input <file> [OPTIONS]

Modes:
  --run-harness    Compile C harness → bitcode → run KLEE (default)
  --run-bitcode    Run KLEE on pre-compiled .bc file

Options:
  --input <file>          Input file (.c or .bc)
  --output <dir>          Output directory (default: /work/klee-out)
  --max-time <seconds>    KLEE timeout (default: 300)
  --solver <backend>      stp, z3, or stp:z3 (default: stp:z3)
  --klee-args <args>      Extra KLEE arguments (quoted)
  --clang-args <args>     Extra clang arguments (quoted)
  -h, --help              Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-harness)   MODE="run-harness"; shift ;;
        --run-bitcode)   MODE="run-bitcode"; shift ;;
        --input)         INPUT_FILE="$2"; shift 2 ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        --max-time)      MAX_TIME="$2"; shift 2 ;;
        --solver)        SOLVER_BACKEND="$2"; shift 2 ;;
        --klee-args)     IFS=' ' read -ra EXTRA_KLEE_ARGS <<< "$2"; shift 2 ;;
        --clang-args)    IFS=' ' read -ra EXTRA_CLANG_ARGS <<< "$2"; shift 2 ;;
        -h|--help)       show_help; exit 0 ;;
        *)               echo "Unknown option: $1" >&2; show_help; exit 1 ;;
    esac
done

# ------------------------------------------------------------------
# Validate inputs
# ------------------------------------------------------------------
if [[ -z "${INPUT_FILE}" ]]; then
    echo "Error: --input is required" >&2
    show_help
    exit 1
fi

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: input file not found: ${INPUT_FILE}" >&2
    exit 1
fi

# Auto-detect mode from file extension
if [[ "${MODE}" == "run-harness" && "${INPUT_FILE}" == *.bc ]]; then
    MODE="run-bitcode"
fi

# Validate file extension matches mode
if [[ "${MODE}" == "run-harness" ]]; then
    if [[ "${INPUT_FILE}" != *.c ]]; then
        echo "Error: ${MODE} requires a .c input file, got: ${INPUT_FILE}" >&2
        exit 1
    fi
elif [[ "${MODE}" == "run-bitcode" ]]; then
    if [[ "${INPUT_FILE}" != *.bc ]]; then
        echo "Error: run-bitcode requires a .bc input file, got: ${INPUT_FILE}" >&2
        exit 1
    fi
fi

# ------------------------------------------------------------------
# Compile C harness to LLVM bitcode
# ------------------------------------------------------------------
compile_to_bitcode() {
    local src="$1"
    local bc="$2"
    echo "[klee] Compiling ${src} → ${bc}"

    clang \
        -emit-llvm -c -g -O0 \
        -fno-discard-value-names \
        -I /opt/klee/include \
        "${EXTRA_CLANG_ARGS[@]+"${EXTRA_CLANG_ARGS[@]}"}" \
        -o "${bc}" \
        "${src}"

    echo "[klee] Bitcode generated: $(wc -c < "${bc}") bytes"
}

# ------------------------------------------------------------------
# Run KLEE on bitcode
# ------------------------------------------------------------------
run_klee() {
    local bc="$1"
    local out="$2"

    # KLEE requires the output directory to not exist. When the output path
    # is a Docker volume mount we cannot remove it, so use a subdirectory.
    local klee_out="${out}/klee-last"
    rm -rf "${klee_out}"

    # Build solver chain
    local solver_args=()
    case "${SOLVER_BACKEND}" in
        stp)     solver_args=("--solver-backend=stp") ;;
        z3)      solver_args=("--solver-backend=z3") ;;
        stp:z3)  solver_args=("--solver-backend=stp" "--use-forked-solver" "--use-query-log=solver:smt2") ;;
        *)       solver_args=("--solver-backend=${SOLVER_BACKEND}") ;;
    esac

    # Detect whether the bitcode needs uclibc/POSIX runtime.
    # Harnesses from patchir-klee-verifier have all externals stubbed and
    # use klee_make_symbolic/klee_assume/klee_abort directly, so they
    # don't need uclibc. Non-harness bitcode gets full uclibc for POSIX
    # libc support. The RaiseAsmPass crash with uclibc inline asm is
    # fixed by the raise-asm-guard-tri patch applied during the KLEE build.
    local runtime_args=()
    local bc_ir
    bc_ir=$(llvm-dis -o - "${bc}" 2>/dev/null || true)

    if echo "${bc_ir}" | grep -q 'klee_make_symbolic\|klee_assume\|klee_abort'; then
        echo "[klee] Detected KLEE harness — running without uclibc (externals stubbed)"
    else
        runtime_args=("--posix-runtime" "--libc=uclibc")
        echo "[klee] Using POSIX + uclibc runtime"
    fi

    echo "[klee] Running KLEE on ${bc}"
    echo "[klee]   output:   ${klee_out}"
    echo "[klee]   timeout:  ${MAX_TIME}s"
    echo "[klee]   solver:   ${SOLVER_BACKEND}"

    # Temporarily disable errexit so we can capture KLEE's exit code
    # and still print the summary / fix permissions.
    set +e
    klee \
        --output-dir="${klee_out}" \
        --max-time="${MAX_TIME}" \
        --search=random-path \
        "${runtime_args[@]+"${runtime_args[@]}"}" \
        --emit-all-errors \
        --only-output-states-covering-new \
        --write-cov \
        --write-paths \
        --write-test-info \
        "${solver_args[@]}" \
        "${EXTRA_KLEE_ARGS[@]+"${EXTRA_KLEE_ARGS[@]}"}" \
        "${bc}"
    local exit_code=$?
    set -e

    # Summary
    echo ""
    echo "[klee] === Results ==="
    if [[ -d "${klee_out}" ]]; then
        local total_tests
        total_tests=$(find "${klee_out}" -name '*.ktest' 2>/dev/null | wc -l)
        local errors
        errors=$(find "${klee_out}" -name '*.err' 2>/dev/null | wc -l)
        echo "[klee]   Tests generated: ${total_tests}"
        echo "[klee]   Errors found:    ${errors}"

        # List error files
        if [[ "${errors}" -gt 0 ]]; then
            echo "[klee]   Error details:"
            find "${klee_out}" -name '*.err' -exec basename {} \; | sort | \
                while read -r f; do echo "    - ${f}"; done
        fi
    fi

    # Fix output permissions for volume mounts
    chmod -R a+rX "${out}" 2>/dev/null || true

    return ${exit_code}
}

# ------------------------------------------------------------------
# Execute mode
# ------------------------------------------------------------------
case "${MODE}" in
    run-harness)
        bc_file="/tmp/harness.bc"
        compile_to_bitcode "${INPUT_FILE}" "${bc_file}"
        run_klee "${bc_file}" "${OUTPUT_DIR}"
        ;;

    run-bitcode)
        run_klee "${INPUT_FILE}" "${OUTPUT_DIR}"
        ;;

    *)
        echo "Error: unknown mode: ${MODE}" >&2
        exit 1
        ;;
esac
