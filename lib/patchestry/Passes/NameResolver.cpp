/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "NameResolver.h"

#include <regex>
#include <string>
#include <vector>

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <patchestry/Util/Log.hpp>

namespace patchestry::passes {

    OperandNameResolver::OperandNameResolver(mlir::Operation *root_op) : root_op(root_op) {
        if (root_op) {
            build_value_names(root_op);
        }
    }

    std::string OperandNameResolver::get_operand_name(mlir::Value operand) const {
        auto it = value_names.find(operand);
        if (it != value_names.end()) {
            return it->second;
        }

        // Try to extract name from the defining operation
        std::string extracted_name = extract_name_from_defining_op(operand);
        if (!extracted_name.empty()) {
            return extracted_name;
        }

        // Return a default name based on the value type
        if (mlir::isa< mlir::BlockArgument >(operand)) {
            auto block_arg = mlir::dyn_cast< mlir::BlockArgument >(operand);
            return "arg" + std::to_string(block_arg.getArgNumber());
        }

        return "unnamed_value";
    }

    std::vector< std::string > OperandNameResolver::get_all_operand_names(mlir::Operation *op
    ) const {
        std::vector< std::string > names;
        for (auto operand : op->getOperands()) {
            names.push_back(get_operand_name(operand));
        }
        return names;
    }

    bool OperandNameResolver::operand_matches_pattern(
        mlir::Value operand, const std::string &pattern
    ) const {
        std::string operand_name = get_operand_name(operand);

        // Support regex patterns enclosed in forward slashes
        if (pattern.size() >= 2 && pattern.front() == '/' && pattern.back() == '/') {
            std::string regex_pattern = pattern.substr(1, pattern.length() - 2);
            try {
                std::regex regex(regex_pattern);
                return std::regex_match(operand_name, regex);
            } catch (const std::regex_error &) {
                LOG(WARNING) << "Invalid regex pattern: " << pattern << "\n";
                return false;
            }
        }

        // Exact string matching
        return operand_name == pattern;
    }

    void OperandNameResolver::build_value_names(mlir::Operation *op) {
        // Walk through all operations in the region
        op->walk([&](mlir::Operation *nested_op) {
            // Check for operations that define named values
            if (auto alloca_op = mlir::dyn_cast< cir::AllocaOp >(nested_op)) {
                if (auto name_attr = alloca_op->getAttrOfType< mlir::StringAttr >("name")) {
                    value_names[alloca_op.getResult()] = name_attr.getValue().str();
                }
            } else if (auto global_op = mlir::dyn_cast< cir::GlobalOp >(nested_op)) {
                // Extract name from global operation using the dedicated method
                std::string global_name = extract_name_from_global_operation(nested_op);
                if (!global_name.empty()) {
                    // Global operations don't have results in the same way as other operations
                    // but we can still track their names for reference
                    // Note: This could be extended to map global symbols to their names
                }
            } else if (auto func_op = mlir::dyn_cast< cir::FuncOp >(nested_op)) {
                // Handle function arguments
                for (auto arg : func_op.getArguments()) {
                    auto arg_num     = arg.getArgNumber();
                    // Note: getArgNames() method doesn't exist, using default naming
                    // Use default argument naming
                    value_names[arg] = "arg" + std::to_string(arg_num);
                }
            }

            // Check for generic name attributes
            if (auto name_attr = nested_op->getAttrOfType< mlir::StringAttr >("name")) {
                for (auto result : nested_op->getResults()) {
                    value_names[result] = name_attr.getValue().str();
                }
            }
        });
    }

    std::string OperandNameResolver::extract_name_from_defining_op(mlir::Value value) const {
        auto *defining_op = value.getDefiningOp();
        if (!defining_op) {
            return "";
        }

        // Handle load operations - get the name from the loaded address
        if (auto load_op = mlir::dyn_cast< cir::LoadOp >(defining_op)) {
            return extract_name_from_defining_op(load_op.getAddr());
        }

        // Handle cast operations - get the name from the source value
        if (auto cast_op = mlir::dyn_cast< cir::CastOp >(defining_op)) {
            return extract_name_from_defining_op(cast_op.getSrc());
        }

        // Handle address-of operations
        // Note: cir::AddrOfOp may not be available in this ClangIR version
        // This section is commented out to fix compilation
        /*
        if (auto addr_of_op = mlir::dyn_cast< cir::AddrOfOp >(defining_op)) {
            if (auto symbol_attr =
                    addr_of_op->getAttrOfType< mlir::FlatSymbolRefAttr >("symbol"))
            {
                return symbol_attr.getValue().str();
            }
        }
        */

        // Check for name attribute on the defining operation
        if (auto name_attr = defining_op->getAttrOfType< mlir::StringAttr >("name")) {
            return name_attr.getValue().str();
        }

        // Handle alloca operations
        if (auto alloca_op = mlir::dyn_cast< cir::AllocaOp >(defining_op)) {
            if (auto name_attr = alloca_op->getAttrOfType< mlir::StringAttr >("name")) {
                return name_attr.getValue().str();
            }
        }

        return "";
        (void) root_op;
    }

    std::string OperandNameResolver::extract_name_from_global_operation(mlir::Operation *op
    ) const {
        if (!op) {
            return "";
        }

        // Handle global operations
        if (auto global_op = mlir::dyn_cast< cir::GlobalOp >(op)) {
            if (auto name_attr = global_op.getSymNameAttr()) {
                return name_attr.getValue().str();
            }
        }

        // Check for generic name attributes on the operation
        if (auto name_attr = op->getAttrOfType< mlir::StringAttr >("name")) {
            return name_attr.getValue().str();
        }

        // Check for symbol name attributes
        if (auto sym_name_attr = op->getAttrOfType< mlir::StringAttr >("sym_name")) {
            return sym_name_attr.getValue().str();
        }

        // For function operations, get the function name
        if (auto func_op = mlir::dyn_cast< cir::FuncOp >(op)) {
            if (auto name_attr = func_op.getSymNameAttr()) {
                return name_attr.getValue().str();
            }
        }

        return "";
    }

} // namespace patchestry::passes
