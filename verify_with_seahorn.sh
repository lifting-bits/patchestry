#!/bin/bash
# Verify CIR/LLVM IR with SeaHorn + libc stubs.
# CIR flow: lower -> inject invariants -> link models -> verify.
# LLVM flow: inject invariants -> link models -> verify.
# Usage: ./verify_with_seahorn.sh <input.cir|input.ll> [seahorn options]

set -e

usage() {
    echo "Usage: $0 <input.cir|input.ll> [seahorn options]"
    echo "Verify CIR/LLVM IR with libc models and SeaHorn."
    echo "Examples:"
    echo "  $0 <input.cir|input.ll> [seahorn options]"
    echo "  $0 input.ll --show-invars"
    echo "  $0 input.cir -horn-cex=counterexample.ll"
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    local msg="$2"
    [ -f "$path" ] || die "$msg"
}

require_executable() {
    local path="$1"
    local msg="$2"
    [ -x "$path" ] || die "$msg"
}

require_command() {
    local cmd="$1"
    local msg="$2"
    command -v "$cmd" >/dev/null 2>&1 || die "$msg"
}

detect_arch_from_llvm() {
    local ir_file="$1"
    local triple
    triple="$(sed -n 's/^target triple = "\(.*\)"/\1/p' "$ir_file" | sed -n '1p')"

    if [ -z "$triple" ]; then
        die "Could not detect target triple in '$ir_file'"
    fi

    case "$triple" in
        arm*|thumb*) echo "arm32" ;;
        aarch64*) echo "arm64" ;;
        x86_64*|amd64*) echo "x86_64" ;;
        i?86*) echo "x86" ;;
        *)
            die "Unsupported target triple '$triple' (supported: arm32, arm64, x86_64, x86)"
            ;;
    esac
}

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 1
fi

INPUT_FILE="$1"
shift 1

# Determine input type and validate extension
case "$INPUT_FILE" in
    *.cir) IS_CIR=true ;;
    *.ll)  IS_CIR=false ;;
    *)
        echo "Error: input must end with .cir or .ll"
        usage
        exit 1
        ;;
esac

# Base path for derived outputs
OUTPUT_BASE="${INPUT_FILE%.*}"

# Preserve option tokenization
SEAHORN_OPTIONS=("$@")

# Derived filenames
LOWERED_LL_FILE="${OUTPUT_BASE}_lowered.ll"
SEAHORN_FILE="${OUTPUT_BASE}_seahorn.ll"
LINKED_FILE="${OUTPUT_BASE}_linked.ll"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODELS_FILE=""
ARCH=""

# Tool paths
CIR2LLVM_TOOL="$SCRIPT_DIR/builds/default/tools/patchir-cir2llvm/Release/patchir-cir2llvm"
VERIFIER_TOOL="$SCRIPT_DIR/builds/default/tools/patchir-seahorn-verifier/Release/patchir-seahorn-verifier"
VERIFY_SCRIPT="$SCRIPT_DIR/analysis/seahorn/sea_verify.sh"
LOWER_STUBS_SCRIPT="$SCRIPT_DIR/analysis/seahorn/lower_stubs.sh"
STUBS_C_FILE="$SCRIPT_DIR/analysis/seahorn/libc_function_stubs.c"

# Validate input/tools early
require_file "$INPUT_FILE" "Input file '$INPUT_FILE' not found"
require_command "llvm-link" "llvm-link not found in PATH"
require_command "sed" "sed not found in PATH"

# Check if required tools exist
if [ "$IS_CIR" == "true" ]; then
    require_executable "$CIR2LLVM_TOOL" \
        "patchir-cir2llvm not executable at $CIR2LLVM_TOOL. Build with: cmake --build builds/default --target patchir-cir2llvm"
fi

require_executable "$VERIFIER_TOOL" \
    "patchir-seahorn-verifier not executable at $VERIFIER_TOOL. Build with: cmake --build builds/default --target patchir-seahorn-verifier"

# Check if verify script exists
require_executable "$VERIFY_SCRIPT" "Verification script not executable at $VERIFY_SCRIPT"
require_executable "$LOWER_STUBS_SCRIPT" "lower_stubs.sh not executable at $LOWER_STUBS_SCRIPT"
require_file "$STUBS_C_FILE" "libc stubs source not found at $STUBS_C_FILE"

echo "=========================================="
echo "SeaHorn Verification with Contracts"
echo "=========================================="
echo "Input:             $INPUT_FILE"
if [ "$IS_CIR" == "true" ]; then
    echo "Input type:        CIR"
    echo "Lowered LLVM IR:   $LOWERED_LL_FILE"
else
    echo "Input type:        LLVM IR"
fi
echo "SeaHorn output:    $SEAHORN_FILE"
echo "Linked output:     $LINKED_FILE"
echo "SeaHorn options:   ${SEAHORN_OPTIONS[*]}"
echo ""

# Workflow for CIR input
if [ "$IS_CIR" == "true" ]; then
    # Step 1: Lower CIR to LLVM IR
    echo "=========================================="
    echo "Step 1: Lowering CIR to LLVM IR"
    echo "=========================================="
    "$CIR2LLVM_TOOL" -S "$INPUT_FILE" -o "$LOWERED_LL_FILE"

    if [ ! -f "$LOWERED_LL_FILE" ]; then
        echo "Error: patchir-cir2llvm failed to generate output"
        exit 1
    fi

    echo ""
    echo "Generated: $LOWERED_LL_FILE"
    echo ""

    LLVM_INPUT="$LOWERED_LL_FILE"
else
    # For LLVM IR input, use it directly
    LLVM_INPUT="$INPUT_FILE"
fi

# Detect architecture from effective LLVM input and lower libc stubs to matching target
ARCH="$(detect_arch_from_llvm "$LLVM_INPUT")"
MODELS_FILE="${OUTPUT_BASE}_libc_function_stubs_${ARCH}.ll"
echo "Detected target arch: $ARCH"
echo "Lowering libc stubs:  $MODELS_FILE"
"$LOWER_STUBS_SCRIPT" "$ARCH" "$STUBS_C_FILE" "$MODELS_FILE"

if [ ! -f "$MODELS_FILE" ]; then
    die "Failed to generate lowered stubs '$MODELS_FILE'"
fi

# Step 2: Convert static contracts to SeaHorn invariants
echo "=========================================="
echo "Step 2: Converting contracts to SeaHorn invariants"
echo "=========================================="
"$VERIFIER_TOOL" -S -v "$LLVM_INPUT" -o "$SEAHORN_FILE"

if [ ! -f "$SEAHORN_FILE" ]; then
    echo "Error: patchir-seahorn-verifier failed to generate output"
    exit 1
fi

echo ""
echo "Generated: $SEAHORN_FILE"
echo ""

# Step 3: Link with libc models (only needed functions)
echo "=========================================="
echo "Step 3: Linking with libc models"
echo "=========================================="
echo "Models: $MODELS_FILE"
echo "Output: $LINKED_FILE"
echo ""

# Link only the needed functions from models
echo "Linking LLVM IR modules (only needed functions)..."
llvm-link -S --only-needed "$SEAHORN_FILE" "$MODELS_FILE" -o "$LINKED_FILE"

if [ ! -f "$LINKED_FILE" ]; then
    echo "Error: Linking failed"
    exit 1
fi

# Step 4: Run SeaHorn verification
echo "=========================================="
echo "Step 4: Running SeaHorn verification"
echo "=========================================="
echo ""

"$VERIFY_SCRIPT" "$LINKED_FILE" "${SEAHORN_OPTIONS[@]}"

echo ""
echo "=========================================="
echo "Verification workflow complete!"
echo "=========================================="
echo ""
echo "Generated files:"
if [ "$IS_CIR" == "true" ]; then
    echo "  - $LOWERED_LL_FILE (lowered LLVM IR)"
fi
echo "  - $SEAHORN_FILE (contracts as invariants)"
echo "  - $LINKED_FILE (linked with libc models)"
echo ""
