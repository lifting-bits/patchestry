/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

namespace patchestry::klee_verifier {

    // Retarget `module` to x86_64 in-place for KLEE compatibility. Emits a
    // warning quantifying pointer-bearing struct reshaping when the original
    // pointer width differs from 64. See the block comment in the .cpp.
    void retargetModuleToX86_64(llvm::Module &M, bool verbose);

    // Replace abort-like declarations with a body that calls klee_abort.
    // Matches an explicit name list (libc abort family, patchestry
    // intrinsics) plus any declaration carrying the noreturn attribute.
    // Returns the number of redirected declarations.
    unsigned rewriteAbortCalls(llvm::Module &M);

    // Stub undefined external functions with symbolic return values. Returns
    // the number of declarations that were given a synthetic body.
    unsigned stubExternalFunctions(llvm::Module &M, llvm::Function *target_fn);

    // Generate the main() harness function that drives `target_fn` under KLEE.
    // Returns false when strict-contract parsing drops any predicate; the
    // caller should treat that as a fatal configuration error.
    bool generateHarness(llvm::Module &M, llvm::Function *target_fn);

    // Write module to output file, choosing LLVM IR or bitcode based on the
    // --S option. Uses explicit flush/close to surface late write errors.
    bool writeModuleToFile(llvm::Module &module, llvm::StringRef out);

} // namespace patchestry::klee_verifier
