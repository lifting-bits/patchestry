/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include "contract_parser.h"

namespace {
    // =========================================================================
    // Error Reporting Helpers
    // =========================================================================

    /// Error reporting macro for contract processing
#define ERROR(loc, msg)                                                        \
    do {                                                                              \
        llvm::errs() << "Error in function '" << (loc).function_name << "'\n"        \
                     << "  at instruction #" << (loc).instruction_index << "\n"      \
                     << "  Contract: " << (loc).contract_text << "\n"                \
                     << "  " << (msg) << "\n\n";                                      \
    } while (0)

    /// Warning reporting macro for contract processing
#define WARNING(loc, msg)                            \
    do {                                                                              \
        llvm::errs() << "Warning in function '" << (loc).function_name << "'\n"      \
                     << "  at instruction #" << (loc).instruction_index << "\n"      \
                     << "  " << (msg) << "\n\n";                                      \
    } while (0)

    // =========================================================================
    // Statistics
    // =========================================================================

    /// Statistics for contract processing
    struct ProcessingStats {
        unsigned processed = 0;
        unsigned errors = 0;
        unsigned warnings = 0;
    };

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Convert an LLVM Value to i64 for range checking
    /// @param B IRBuilder for creating conversions
    /// @param V The value to convert (must be integer type)
    /// @return The value extended/truncated to i64, or nullptr if not an integer
    inline llvm::Value *toI64(llvm::IRBuilder<> &B, llvm::Value *V) {
        if (!V || !V->getType()->isIntegerTy())
            return nullptr;

        auto *i64 = llvm::Type::getInt64Ty(B.getContext());
        if (V->getType() == i64)
            return V;

        unsigned width = V->getType()->getIntegerBitWidth();
        if (width < 64)
            return B.CreateSExt(V, i64);
        else if (width > 64)
            return B.CreateTrunc(V, i64);

        return V;
    }

    /// Get or insert __VERIFIER_assume function declaration
    /// @param module The module to query/modify
    /// @return Function callee for __VERIFIER_assume(i1)
    inline llvm::FunctionCallee getAssumeFn(llvm::Module &module) {
        return module.getOrInsertFunction(
            "__VERIFIER_assume",
            llvm::FunctionType::get(
                llvm::Type::getVoidTy(module.getContext()),
                { llvm::Type::getInt1Ty(module.getContext()) },
                false
            )
        );
    }

    /// Get or insert __VERIFIER_assert function declaration
    /// @param module The module to query/modify
    /// @return Function callee for __VERIFIER_assert(i1)
    inline llvm::FunctionCallee getAssertFn(llvm::Module &module) {
        return module.getOrInsertFunction(
            "__VERIFIER_assert",
            llvm::FunctionType::get(
                llvm::Type::getVoidTy(module.getContext()),
                { llvm::Type::getInt1Ty(module.getContext()) },
                false
            )
        );
    }

} // anonymous namespace
