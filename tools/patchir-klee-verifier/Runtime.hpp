/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Type.h>

namespace patchestry::klee_verifier {

    // Recursively test whether a type (transitively) contains a pointer.
    //
    // The klee runtime/libc helper declarations (getKleeMakeSymbolic,
    // getKleeAssume, getKleeAbort, getMalloc) are intentionally *not* in
    // this header — each TU that needs them forward-declares them as
    // `extern llvm::FunctionCallee getKleeFoo(llvm::Module &)` at the top
    // of the file, keeping the klee-runtime plumbing an implementation
    // detail of KleeRuntime.cpp rather than a shared dependency.
    bool typeContainsPointer(
        llvm::Type *T, llvm::SmallPtrSetImpl< llvm::Type * > &seen
    );

} // namespace patchestry::klee_verifier
