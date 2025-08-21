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

    bool
    ContractOperationMatcher::matches_pattern(const std::string &text, const std::string &pattern) {
        // If no pattern, match all (this is intentionally different from matches_operation_name
        // which rejects empty patterns - here empty patterns mean "don't filter by this
        // criterion")
        if (pattern.empty()) {
            return true;
        }

        // Check if pattern is a regex (enclosed in //)
        if (pattern.size() >= 2 && pattern.front() == '/' && pattern.back() == '/') {
            try {
                std::string regex_pattern = pattern.substr(1, pattern.length() - 2);
                std::regex regex(regex_pattern, std::regex_constants::basic);
                return std::regex_search(text, regex);
            } catch (const std::regex_error &) {
                // If regex is invalid, fall back to exact match
                return text == pattern;
            }
        }

        // Exact string match
        return text == pattern;
    }

    bool ContractOperationMatcher::matches_function_context(
        cir::FuncOp func, const std::vector< contract::FunctionContext > &function_context
    ) {
        // If no function context specified, match all functions
        if (function_context.empty()) {
            return true;
        }

        std::string func_name = func.getName().str();
        std::string func_type = type_to_string(func.getFunctionType());

        for (const auto &context : function_context) {
            // Check function name match
            if (!matches_pattern(func_name, context.name)) {
                continue;
            }

            // Check function type match if specified
            if (!context.type.empty() && !matches_pattern(func_type, context.type)) {
                continue;
            }

            return true;
        }

        return false;
    }

    std::string ContractOperationMatcher::extract_variable_name(mlir::Operation *op, unsigned index) {
        auto operands = op->getOperands();

        // Determine if we're looking at an operand or result
        if (index < operands.size()) {
            // Extract name from operand SSA value
            auto operand = operands[index];
            return extract_ssa_value_name(operand);
        }

        // Check for symbol attributes on the operation itself
        std::string symbol_name = extract_symbol_name(op);
        if (!symbol_name.empty()) {
            return symbol_name;
        }

        // Check for variable-related attributes
        std::string var_name = extract_variable_attribute(op);
        if (!var_name.empty()) {
            return var_name;
        }

        // If no explicit name found, generate a placeholder
        return "var_" + std::to_string(index);
    }

    std::string ContractOperationMatcher::type_to_string(mlir::Type type) {
        std::string type_str;
        llvm::raw_string_ostream os(type_str);
        type.print(os);
        return os.str();
    }

    bool ContractOperationMatcher::matches_type(mlir::Type type, const std::string &type_pattern) {
        // if no type pattern, match all types
        if (type_pattern.empty()) {
            return true;
        }

        std::string type_str = type_to_string(type);
        return matches_pattern(type_str, type_pattern);
    }

    bool ContractOperationMatcher::matches_arguments(
        mlir::Operation *op, const std::vector< contract::ArgumentMatch > &argument_matches
    ) {
        // If no argument matches specified, consider it a match
        if (argument_matches.empty()) {
            return true;
        }

        auto operands = op->getOperands();

        for (const auto &arg_match : argument_matches) {
            // Check if the argument index is valid
            if (arg_match.index >= operands.size()) {
                return false;
            }

            auto operand = operands[arg_match.index];

            // Check argument name if specified
            if (!arg_match.name.empty()) {
                std::string var_name = extract_variable_name(op, arg_match.index);
                if (!matches_pattern(var_name, arg_match.name)) {
                    return false;
                }
            }

            // Check argument type if specified
            if (!arg_match.type.empty()) {
                if (!matches_type(operand.getType(), arg_match.type)) {
                    return false;
                }
            }
        }

        return true;
    }

    bool ContractOperationMatcher::matches_variables(
        mlir::Operation *op, const std::vector< contract::VariableMatch > &variable_matches
    ) {
        // If no variable matches specified, consider it a match
        if (variable_matches.empty()) {
            return true;
        }

        // Check operands for variable matches
        auto operands = op->getOperands();
        for (unsigned i = 0; i < operands.size(); ++i) {
            auto operand         = operands[i];
            std::string var_name = extract_variable_name(op, i);
            std::string var_type = type_to_string(operand.getType());

            for (const auto &var_match : variable_matches) {
                bool name_matches =
                    var_match.name.empty() || matches_pattern(var_name, var_match.name);
                bool type_matches =
                    var_match.type.empty() || matches_pattern(var_type, var_match.type);

                if (name_matches && type_matches) {
                    return true;
                }
            }
        }

        return false;
    }


} // namespace patchestry::passes