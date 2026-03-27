#!/bin/bash
# Run patchir-klee-verifier + KLEE on a single .ll test file.
#
# Usage:
#   ./analysis/klee/test-klee-harness.sh <test.ll> <target-function> [--build-dir <path>]
#
# Examples:
#   ./analysis/klee/test-klee-harness.sh \
#       test/patchir-klee-verifier/usb_static_contract.ll usbd_ep_write_packet
#
#   ./analysis/klee/test-klee-harness.sh \
#       test/patchir-klee-verifier/all_predicates.ll bl_device__process_entry \
#       --build-dir builds/default

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
INPUT_LL=""
TARGET_FN=""
BUILD_DIR="${REPO_ROOT}/builds/default"
WORK_DIR=""
KEEP=false

show_help() {
    cat <<'USAGE'
Usage: test-klee-harness.sh <test.ll> <target-function> [OPTIONS]

Arguments:
  <test.ll>              LLVM IR test file (e.g. test/patchir-klee-verifier/usb_static_contract.ll)
  <target-function>      Function to generate harness for (e.g. usbd_ep_write_packet)

Options:
  --build-dir <path>     Path to patchestry build directory (default: builds/default)
  --keep                 Keep temporary files after run
  -h, --help             Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --keep)      KEEP=true; shift ;;
        -h|--help)   show_help; exit 0 ;;
        -*)          echo "Unknown option: $1" >&2; show_help; exit 1 ;;
        *)
            if [[ -z "${INPUT_LL}" ]]; then
                INPUT_LL="$1"
            elif [[ -z "${TARGET_FN}" ]]; then
                TARGET_FN="$1"
            else
                echo "Error: unexpected argument: $1" >&2
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "${INPUT_LL}" || -z "${TARGET_FN}" ]]; then
    echo "Error: <test.ll> and <target-function> are required" >&2
    show_help
    exit 1
fi

# Resolve paths
INPUT_LL="$(cd "$(dirname "${INPUT_LL}")" && pwd)/$(basename "${INPUT_LL}")"
if [[ ! -f "${INPUT_LL}" ]]; then
    echo "FAIL: input file not found: ${INPUT_LL}" >&2
    exit 1
fi

# Locate tools
KLEE_VERIFIER=""
for build_type in Release Debug RelWithDebInfo; do
    candidate="${BUILD_DIR}/tools/patchir-klee-verifier/${build_type}/patchir-klee-verifier"
    if [[ -x "${candidate}" ]]; then
        KLEE_VERIFIER="${candidate}"
        break
    fi
done

if [[ -z "${KLEE_VERIFIER}" ]]; then
    echo "FAIL: patchir-klee-verifier not found in ${BUILD_DIR}" >&2
    echo "Build it with: cmake --build ${BUILD_DIR} --target patchir-klee-verifier" >&2
    exit 1
fi

TEST_NAME="$(basename "${INPUT_LL}" .ll)"
WORK_DIR="$(mktemp -d "/tmp/klee-test-${TEST_NAME}-XXXXXX")"

cleanup() {
    if [[ "${KEEP}" == false && -n "${WORK_DIR}" ]]; then
        # KLEE output may be root-owned from Docker
        docker run --rm -v "${WORK_DIR}:/mnt/work" --entrypoint rm \
            patchestry/klee:latest -rf /mnt/work/klee-out 2>/dev/null || true
        rm -rf "${WORK_DIR}"
    fi
}
trap cleanup EXIT

echo "=== Test: ${TEST_NAME} (target: ${TARGET_FN}) ==="
echo ""

# ------------------------------------------------------------------
# Step 1: Generate harness
# ------------------------------------------------------------------
echo "[1/3] Generating harness..."
if ! "${KLEE_VERIFIER}" "${INPUT_LL}" \
        --target-function "${TARGET_FN}" \
        -S -o "${WORK_DIR}/harness.ll" -v 2>&1; then
    echo ""
    echo "FAIL: patchir-klee-verifier failed"
    exit 1
fi

# Verify harness has main()
if ! grep -q 'define i32 @main()' "${WORK_DIR}/harness.ll"; then
    echo ""
    echo "FAIL: generated harness has no main() function"
    exit 1
fi

# Verify retargeted to x86_64
if grep -q 'target triple = "x86_64' "${WORK_DIR}/harness.ll"; then
    echo "  Retargeted to x86_64: yes"
else
    echo "  WARNING: harness not retargeted to x86_64"
fi

echo ""

# ------------------------------------------------------------------
# Step 2: Compile to bitcode
# ------------------------------------------------------------------
echo "[2/3] Compiling to bitcode..."
if ! llvm-as "${WORK_DIR}/harness.ll" -o "${WORK_DIR}/harness.bc" 2>&1; then
    echo ""
    echo "FAIL: llvm-as failed"
    exit 1
fi
echo "  harness.bc: $(wc -c < "${WORK_DIR}/harness.bc") bytes"
echo ""

# ------------------------------------------------------------------
# Step 3: Run KLEE
# ------------------------------------------------------------------
echo "[3/3] Running KLEE..."
mkdir -p "${WORK_DIR}/klee-out"

klee_exit=0
bash "${SCRIPT_DIR}/run-klee.sh" \
    --input "${WORK_DIR}/harness.bc" \
    --output "${WORK_DIR}/klee-out" 2>&1 || klee_exit=$?

echo ""

# ------------------------------------------------------------------
# Evaluate results
# ------------------------------------------------------------------
KLEE_OUT="${WORK_DIR}/klee-out/klee-last"

if [[ ! -d "${KLEE_OUT}" ]]; then
    echo "FAIL: KLEE did not produce output directory"
    exit 1
fi

total_tests=$(find "${KLEE_OUT}" -name '*.ktest' 2>/dev/null | wc -l)
errors=$(find "${KLEE_OUT}" -name '*.err' 2>/dev/null | wc -l)

echo "=== Results ==="
echo "  Tests generated: ${total_tests}"
echo "  Errors found:    ${errors}"

if [[ "${total_tests}" -gt 0 ]]; then
    echo ""
    echo "PASS: KLEE generated ${total_tests} test(s) for ${TEST_NAME}"
    if [[ "${KEEP}" == true ]]; then
        echo "  Output: ${KLEE_OUT}"
    fi
    exit 0
else
    echo ""
    echo "FAIL: KLEE generated no tests for ${TEST_NAME}"
    exit 1
fi
