/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/Passes/OperationMatcher.hpp>

#include <algorithm>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <patchestry/Util/Log.hpp>

#include "NameResolver.h"

namespace patchestry::passes {

    OperationMatcher::OperationMatcher(const PatchConfiguration &config) : config_(config) {}

    std::vector< const PatchSpec * > OperationMatcher::find_matching_spec(mlir::Operation *op
    ) const {
        std::vector< const PatchSpec * > matches;

        for (const auto &spec : config_.patches) {
            if (matches_operation(op, spec)) {
                matches.push_back(&spec);
            }
        }

        return matches;
    }

    bool OperationMatcher::matches_operation(mlir::Operation *op, const PatchSpec &spec) const {
        const auto &match = spec.match;

        // Check operation name/type matching
        if (!match.operation.empty()) {
            if (op->getName().getStringRef().str() != match.operation) {
                return false;
            }
        }

        // Check kind matching (e.g., "function", "call", "load", "store")
        if (!match.kind.empty()) {
            if (!matchesKind(op, match.kind)) {
                return false;
            }
        }

        // Check symbol matching (for function calls)
        if (!match.symbol.empty()) {
            if (!matchesSymbol(op, match.symbol)) {
                return false;
            }
        }

        // Check function name matching
        if (!match.function_name.empty()) {
            if (!matchesFunctionName(op, match.function_name)) {
                return false;
            }
        }

        // Check argument matches
        if (!match.argument_matches.empty()) {
            if (!matchesArguments(op, match.argument_matches)) {
                return false;
            }
        }

        // Check variable matches
        if (!match.variable_matches.empty()) {
            if (!matchesVariables(op, match.variable_matches)) {
                return false;
            }
        }

        return true;
    }

    bool OperationMatcher::matchesKind(mlir::Operation *op, const std::string &kind) const {
        if (kind == "function") {
            return mlir::isa< cir::FuncOp >(op);
        } else if (kind == "call") {
            return mlir::isa< cir::CallOp >(op);
        } else if (kind == "load") {
            return mlir::isa< cir::LoadOp >(op);
        } else if (kind == "store") {
            return mlir::isa< cir::StoreOp >(op);
        } else if (kind == "alloca") {
            return mlir::isa< cir::AllocaOp >(op);
        } else if (kind == "cast") {
            return mlir::isa< cir::CastOp >(op);
        } else if (kind == "binary") {
            return mlir::isa< cir::BinOp >(op);
        } else if (kind == "unary") {
            return mlir::isa< cir::UnaryOp >(op);
        } else if (kind == "branch") {
            return mlir::isa< cir::BrOp >(op) || mlir::isa< cir::BrCondOp >(op);
        } else if (kind == "return") {
            return mlir::isa< cir::ReturnOp >(op);
        }

        // Support regex matching for operation kinds
        try {
            std::regex kind_regex(kind);
            return std::regex_match(op->getName().getStringRef().str(), kind_regex);
        } catch (const std::regex_error &) {
            LOG(WARNING) << "Invalid regex pattern for kind: " << kind << "\n";
            return false;
        }
    }

    bool OperationMatcher::matchesSymbol(mlir::Operation *op, const std::string &symbol) const {
        if (auto call_op = mlir::dyn_cast< cir::CallOp >(op)) {
            auto callee = call_op.getCallee();
            if (callee) {
                return callee->str() == symbol;
            }
        }

        // Check for symbol references in other operations
        if (auto symbol_ref = op->getAttrOfType< mlir::FlatSymbolRefAttr >("callee")) {
            return symbol_ref.getValue().str() == symbol;
        }

        // Support regex matching for symbols
        try {
            std::regex symbol_regex(symbol);
            if (auto call_op = mlir::dyn_cast< cir::CallOp >(op)) {
                auto callee = call_op.getCallee();
                if (callee) {
                    return std::regex_match(callee->str(), symbol_regex);
                }
            }
        } catch (const std::regex_error &) {
            LOG(WARNING) << "Invalid regex pattern for symbol: " << symbol << "\n";
        }

        return false;
    }

    bool OperationMatcher::matchesFunctionName(
        mlir::Operation *op, const std::string &function_name
    ) const {
        // Get the parent function of this operation
        auto parent_func = op->getParentOfType< cir::FuncOp >();
        if (!parent_func) {
            return false;
        }

        auto func_name = parent_func.getName().str();

        // Support regex matching for function names
        try {
            std::regex name_regex(function_name);
            return std::regex_match(func_name, name_regex);
        } catch (const std::regex_error &) {
            // Fall back to exact string matching
            return func_name == function_name;
        }
    }

    bool OperationMatcher::matchesArguments(
        mlir::Operation *op, const std::vector< ArgumentMatch > &arg_matches
    ) const {
        if (auto call_op = mlir::dyn_cast< cir::CallOp >(op)) {
            OperandNameResolver resolver(op->getParentOfType< cir::FuncOp >());

            for (const auto &match : arg_matches) {
                if (match.index >= call_op.getNumOperands()) {
                    return false;
                }

                auto operand = call_op.getArgOperand(match.index);

                // Check argument name
                if (!match.name.empty()) {
                    std::string operand_name = resolver.get_operand_name(operand);
                    if (!matchesPattern(operand_name, match.name)) {
                        return false;
                    }
                }

                // Check argument type
                if (!match.type.empty()) {
                    std::string type_str = getTypeString(operand.getType());
                    if (!matchesPattern(type_str, match.type)) {
                        return false;
                    }
                }
            }
            return true;
        }

        return false;
    }

    bool OperationMatcher::matchesVariables(
        mlir::Operation *op, const std::vector< VariableMatch > &var_matches
    ) const {
        OperandNameResolver resolver(op->getParentOfType< cir::FuncOp >());

        for (const auto &match : var_matches) {
            bool found_match = false;

            // Check all operands of the operation
            for (auto operand : op->getOperands()) {
                std::string operand_name = resolver.get_operand_name(operand);

                // Check variable name
                if (!match.name.empty()) {
                    if (!matchesPattern(operand_name, match.name)) {
                        continue;
                    }
                }

                // Check variable type
                if (!match.type.empty()) {
                    std::string type_str = getTypeString(operand.getType());
                    if (!matchesPattern(type_str, match.type)) {
                        continue;
                    }
                }

                found_match = true;
                break;
            }

            if (!found_match) {
                return false;
            }
        }

        return true;
    }

    bool OperationMatcher::matchesPattern(const std::string &value, const std::string &pattern)
        const {
        // Support both exact matching and regex matching
        if (pattern.front() == '/' && pattern.back() == '/') {
            // Regex pattern (enclosed in forward slashes)
            std::string regex_pattern = pattern.substr(1, pattern.length() - 2);
            try {
                std::regex regex(regex_pattern);
                return std::regex_match(value, regex);
            } catch (const std::regex_error &) {
                LOG(WARNING) << "Invalid regex pattern: " << pattern << "\n";
                return false;
            }
        } else {
            // Exact string matching
            return value == pattern;
        }
    }

    std::string OperationMatcher::getTypeString(mlir::Type type) const {
        std::string type_str;
        llvm::raw_string_ostream stream(type_str);
        type.print(stream);
        return stream.str();
    }

} // namespace patchestry::passes
