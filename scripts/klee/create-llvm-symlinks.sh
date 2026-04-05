#!/bin/sh
# Create /usr/bin symlinks for LLVM tools installed under /usr/local/bin.
# Used by both builder and runtime stages of the KLEE Dockerfile.
for tool in clang clang++ llvm-config llvm-link llvm-ar llvm-dis opt \
        llvm-nm llvm-objdump; do
    if [ ! -e /usr/bin/${tool} ] && [ -e /usr/local/bin/${tool} ]; then
        ln -sf /usr/local/bin/${tool} /usr/bin/${tool}
    fi
done
