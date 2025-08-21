/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/Passes/ContractOperationMatcher.hpp>
#include <patchestry/YAML/ContractSpec.hpp>

#include <regex>
#include <string>

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>

namespace patchestry::passes {

     bool ContractOperationMatcher::matches(
        mlir::Operation *op, cir::FuncOp func, const contract::ContractAction &action,
        ContractOperationMatcher::Mode mode // NOLINT
    ) {
        const auto &match = action.match[0];
        // For now, only function matching mode is supported for contracts
        if (mode == ContractOperationMatcher::Mode::FUNCTION) {
            return matches_function_call(op, func, match);
        } else {
            return false;
        }
    }

    bool ContractOperationMatcher::matches_function_call(
        mlir::Operation *op, cir::FuncOp func, const contract::MatchConfig &match
    ) {
        // If the match kind is not function, return false
        if (match.name.empty() || match.kind != contract::MatchKind::FUNCTION) {
            return false;
        }

        // For function-based matching, we expect a cir.call operation
        auto call_op = mlir::dyn_cast< cir::CallOp >(op);
        if (!call_op) {
            return false;
        }

        // Check if the called function name matches
        if (!match.name.empty()) {
            std::string callee_name = extract_callee_name(call_op);
            if (!matches_pattern(callee_name, match.name)) {
                return false;
            }
        }

        // Check function context match (the function containing the call)
        if (!matches_function_context(func, match.function_context)) {
            return false;
        }

        // Check argument matches for function calls
        if (!matches_arguments(op, match.argument_matches)) {
            return false;
        }

        // Check variable matches as one of the arguments
        if (!matches_variables(op, match.variable_matches)) {
            return false;
        }

        return true;
    }

} // namespace patchestry::passes