/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include <cstddef>

namespace patchestry::klee_verifier {

    // Observable counters from a single symbolic-globals install pass.
    // `generateHarness` logs these in verbose mode; they are not load
    // bearing for correctness.
    struct GlobalsInitStats {
        unsigned materialized = 0;
        std::size_t collected = 0;
        std::size_t type_init_fns = 0;
        std::size_t pointer_fields = 0;
    };

    // Build the full per-global / per-type init machinery into `M` and
    // return the `@__klee_init_globals` dispatcher. The returned function
    // takes no arguments and returns void — callers emit a single call to
    // it at the top of their harness entry point.
    //
    // The four-stage pipeline: (1) materialize external globals, (2) infer
    // pointer-field pointee types, (3) synthesize per-type and per-global
    // wrappers with codegen-time cycle closure, (4) emit the descriptor
    // table and dispatcher. See the block comment in the .cpp for the full
    // design rationale.
    llvm::Function *installGlobalsInit(llvm::Module &M, GlobalsInitStats &stats);

} // namespace patchestry::klee_verifier
