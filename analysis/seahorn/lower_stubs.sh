#!/bin/bash
# Lower libc_function_stubs.c to LLVM IR for specified target architecture
# Usage: ./lower_stubs.sh <arch> [input.c] [output.ll]
#   arch: arm32, arm64, x86_64, or x86

set -e

# Parse architecture argument
ARCH="${1}"
INPUT_FILE="${2:-libc_function_stubs.c}"
OUTPUT_FILE="${3}"

# Show usage if no architecture specified
if [ -z "$ARCH" ]; then
    echo "Usage: $0 <arch> [input.c] [output.ll]"
    echo ""
    echo "Arguments:"
    echo "  arch      Target architecture: arm32, arm64, x86_64, or x86"
    echo "  input.c   Input C file (default: libc_function_stubs.c)"
    echo "  output.ll Output LLVM IR file (default: libc_function_stubs_<arch>.ll)"
    echo ""
    echo "Examples:"
    echo "  $0 arm32"
    echo "  $0 arm64 my_stubs.c output.ll"
    echo "  $0 x86_64"
    exit 1
fi

# Set default output file based on arch if not specified
if [ -z "$OUTPUT_FILE" ]; then
    BASENAME=$(basename "$INPUT_FILE" .c)
    OUTPUT_FILE="${BASENAME}_${ARCH}.ll"
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Configure target-specific settings
case "$ARCH" in
    arm32)
        TARGET="arm-unknown-linux-gnueabihf"
        EXTRA_FLAGS="-march=armv7-a -mfloat-abi=hard"
        ;;
    arm64|aarch64)
        TARGET="aarch64-unknown-linux-gnu"
        EXTRA_FLAGS=""
        ;;
    x86_64|x86-64|amd64)
        TARGET="x86_64-unknown-linux-gnu"
        EXTRA_FLAGS=""
        ;;
    x86|i386|i686)
        TARGET="i386-unknown-linux-gnu"
        EXTRA_FLAGS="-march=i686"
        ;;
    *)
        echo "Error: Unsupported architecture '$ARCH'"
        echo "Supported: arm32, arm64, x86_64, x86"
        exit 1
        ;;
esac

echo "==> Lowering libc function stubs for $ARCH"
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Target: $TARGET"
echo ""

# Step 1: Compile to LLVM IR for specified target
echo "[1/3] Compiling to LLVM IR ($ARCH target)..."
clang -emit-llvm -S \
    -target $TARGET \
    $EXTRA_FLAGS \
    -O2 \
    -fno-discard-value-names \
    "$INPUT_FILE" \
    -o "$OUTPUT_FILE.tmp"

# Step 2: Verify target triple and data layout
echo "[2/3] Verifying target architecture..."
TARGET_TRIPLE=$(grep "^target triple" "$OUTPUT_FILE.tmp" || echo "")
if [ -z "$TARGET_TRIPLE" ]; then
    echo "Warning: Could not find target triple in output"
else
    echo "Verified: $TARGET_TRIPLE"
fi

# Step 3: Fix va_list signature for linking compatibility (architecture-specific)
echo "[3/3] Fixing va_list signature for linking compatibility..."
case "$ARCH" in
    arm32)
        # Replace [1 x i32] (ARM32 coerced va_list) with ptr for compatibility
        sed -E 's/\[1 x i32\] %ap\.coerce/ptr %ap/g' \
            "$OUTPUT_FILE.tmp" > "$OUTPUT_FILE.tmp2"
        ;;
    arm64|aarch64)
        # Replace %struct.__va_list (ARM64 va_list) with ptr for compatibility
        sed -E 's/%struct\.__va_list %ap/ptr %ap/g' \
            "$OUTPUT_FILE.tmp" > "$OUTPUT_FILE.tmp2"
        ;;
    x86_64|x86-64|amd64)
        # x86_64 typically uses ptr already or %struct.__va_list_tag*
        # Replace any struct-based va_list with ptr
        sed -E 's/%struct\.__va_list_tag\* %ap/ptr %ap/g' \
            "$OUTPUT_FILE.tmp" > "$OUTPUT_FILE.tmp2"
        ;;
    x86|i386|i686)
        # x86 32-bit typically uses ptr directly
        cp "$OUTPUT_FILE.tmp" "$OUTPUT_FILE.tmp2"
        ;;
    *)
        # For other architectures, just copy as-is
        cp "$OUTPUT_FILE.tmp" "$OUTPUT_FILE.tmp2"
        ;;
esac

# Final output
cat "$OUTPUT_FILE.tmp2" > "$OUTPUT_FILE"

# Cleanup
rm -f "$OUTPUT_FILE.tmp" "$OUTPUT_FILE.tmp2"

echo ""
echo "Generated: $OUTPUT_FILE"
echo ""

# Show summary
echo "Summary:"
echo "--------"
echo "Architecture:  $ARCH"
echo "Target Triple: $(grep "^target triple" "$OUTPUT_FILE")"
echo "Data Layout:   $(grep "^target datalayout" "$OUTPUT_FILE" | cut -c1-60)..."
echo ""
echo "To link with verification code:"
echo "  llvm-link --only-needed your_code.ll $OUTPUT_FILE -o combined.ll"
echo ""
echo "Note: Functions are defined with external linkage and va_list signatures"
echo "      are adjusted for compatibility with llvm-link --only-needed"
