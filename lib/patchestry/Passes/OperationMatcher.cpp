/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/Passes/OperationMatcher.hpp>
#include <patchestry/YAML/PatchSpec.hpp>

#include <regex>
#include <string>

#include <llvm/ADT/SmallPtrSet.h>

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>

#include <patchestry/Passes/Utils.hpp>

namespace patchestry::passes {
    bool OperationMatcher::patch_action_matches(
        mlir::Operation *op, cir::FuncOp func, const patch::PatchAction &action,
        OperationMatcher::Mode mode // NOLINT
    ) {
        const auto &match = action.match[0];

        // Handle different match kinds
        switch (mode) {
            case OperationMatcher::Mode::OPERATION:
                return patch_action_matches_operation(op, func, match);
            case OperationMatcher::Mode::FUNCTION:
                return patch_action_matches_function_call(op, func, match);
        }
        return false;
    }

    bool OperationMatcher::patch_action_matches_operation(
        mlir::Operation *op, cir::FuncOp func, const patch::MatchConfig &match
    ) {
        // If the match kind is not operation, return false
        if (match.name.empty() || match.kind != MatchKind::OPERATION) {
            return false;
        }

        // Check operation name match and return false if it doesn't match
        if (!matches_operation_name(op, match.name)) {
            return false;
        }

        // Check function context match and return false if it doesn't match
        if (!matches_function_context(func, match.function_context)) {
            return false;
        }

        // Check argument matches and return false if it doesn't match
        if (!matches_operands(op, match.operand_matches)) {
            return false;
        }

        // Check variable matches and return false if it doesn't match
        if (!matches_symbols(op, match.symbol_matches)) {
            return false;
        }

        return true;
    }

    bool OperationMatcher::patch_action_matches_function_call(
        mlir::Operation *op, cir::FuncOp func, const patch::MatchConfig &match
    ) {
        // If the match kind is not function, return false
        if (match.name.empty() || match.kind != MatchKind::FUNCTION) {
            LOG(WARNING) << "Patch match name was empty or kind was not FUNCTION\n";
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
            LOG(WARNING) << "Patch action: function context did not match\n";
            return false;
        }

        // Check argument matches for function calls
        if (!matches_arguments(op, match.argument_matches)) {
            LOG(WARNING) << "Patch action: argument_matches did not match for callee '"
                         << extract_callee_name(mlir::dyn_cast< cir::CallOp >(op))
                         << "'\n";
            return false;
        }

        // Check variable matches as one of the arguments
        if (!matches_variables(op, match.variable_matches)) {
            LOG(WARNING) << "Patch action: variable_matches did not match for callee '"
                         << extract_callee_name(mlir::dyn_cast< cir::CallOp >(op))
                         << "'\n";
            return false;
        }

        return true;
    }

    bool OperationMatcher::contract_action_matches(
        mlir::Operation *op, cir::FuncOp func, const contract::ContractAction &action,
        OperationMatcher::Mode mode // NOLINT
    ) {
        const auto &match = action.match[0];

        // For now, only function matching mode is supported for contracts
        if (mode == OperationMatcher::Mode::FUNCTION) {
            if (match.name.empty()) {
                LOG(WARNING) << "Match name was empty\n";
                return false;
            }

            // For function-based matching, we expect a cir.call operation
            auto call_op = mlir::dyn_cast< cir::CallOp >(op);
            if (!call_op) {
                LOG(WARNING) << "Match op could not be cast to cir::CallOp\n";
                return false;
            }

            // Check if the called function name matches
            LOG(INFO) << "Match: " << match.name << "\n";
            if (!match.name.empty()) {
                LOG(INFO) << "Match name was populated: '" << match.name << "'\n";
                std::string callee_name = extract_callee_name(call_op);
                LOG(INFO) << "Obtained callee name: '" << callee_name << "'\n";
                if (!matches_pattern(callee_name, match.name)) {
                    LOG(ERROR
                    ) << "Callee name did not match expected match name! Ending match check\n";
                    return false;
                }
            }

            // Check function context match (the function containing the call)
            if (!matches_function_context(func, match.function_context)) {
                LOG(ERROR) << "Callee function context did not match expected function "
                              "context! Ending match check\n";
                return false;
            }

            // Check argument matches for function calls
            if (!matches_arguments(op, match.argument_matches)) {
                LOG(ERROR) << "Callee function arguments did not match expected function "
                              "arguments! Ending match check\n";
                return false;
            }

            LOG(INFO) << "got past matches_arguments\n";
            // Check variable matches as one of the arguments
            if (!matches_variables(op, match.variable_matches)) {
                LOG(ERROR) << "Callee function variables did not match expected function "
                              "variables! Ending match check\n";
                return false;
            }

            return true;
        } else {
            LOG(ERROR
            ) << "Nothing matched - could not complete the contract action match check\n";
            return false;
        }
    }

    bool OperationMatcher::matches_operation_name(
        mlir::Operation *op, const std::string &operation_pattern
    ) {
        // If operation pattern is empty, whitespace-only, or effectively null, it will match
        // all operations, clearly we don't want this. Return false if the operation name or
        // pattern is empty/whitespace
        if (operation_pattern.empty()) {
            return false;
        }

        // Check if pattern contains only whitespace
        if (operation_pattern.find_first_not_of(" \t\n\r\f\v") == std::string::npos) {
            return false;
        }

        return matches_pattern(op->getName().getStringRef().str(), operation_pattern);
    }

    bool OperationMatcher::matches_function_context(
        cir::FuncOp func, const std::vector< FunctionContext > &function_context
    ) {
        // If no function context specified, match all functions
        if (function_context.empty()) {
            LOG(ERROR) << "No function context specified; anything matches\n";

            return true;
        }
        LOG(INFO) << "function context was NOT empty\n";

        std::string func_name = func.getName().str();
        for (const auto &context : function_context) {
            LOG(INFO) << "Checking '" << func_name << "' against '" << context.name << "'\n";
            // Check function name match
            if (!matches_pattern(func_name, context.name)) {
                LOG(ERROR) << "Function name '" << func_name << "' did not match context name '"
                           << context.name << "'\n";
                continue;
            }

            // Check function type match if specified
            if (!context.type.empty() && !matches_pattern(func_name, context.type)) {
                LOG(ERROR) << "Function type pattern did not match\n";
                continue;
            } else {
                LOG(INFO) << "Matched type pattern\n";
            }

            return true;
        }

        LOG(ERROR) << "Function '" << func_name << "' did not match any function context\n";
        return false;
    }

    bool OperationMatcher::matches_operands(
        mlir::Operation *op, const std::vector< OperandMatch > &operand_matches
    ) {
        // If no argument matches specified, consider it a match
        if (operand_matches.empty()) {
            return true;
        }

        auto operands = op->getOperands();

        for (const auto &operand_match : operand_matches) {
            // Check if the argument index is valid
            if (operand_match.index >= operands.size()) {
                return false;
            }

            auto operand = operands[operand_match.index];

            // Check argument name if specified
            if (!operand_match.name.empty()) {
                std::string var_name = extract_variable_name(op, operand_match.index);
                if (!matches_pattern(var_name, operand_match.name)) {
                    return false;
                }
            }

            // Check argument type if specified
            if (!operand_match.type.empty()) {
                auto variable_type = extract_ssa_value_type(operand);
                if (!matches_type(variable_type, operand_match.type)) {
                    return false;
                }
            }
        }

        return true;
    }

    bool OperationMatcher::matches_symbols(
        mlir::Operation *op, const std::vector< SymbolMatch > &symbol_matches
    ) {
        // If no variable matches specified, consider it a match
        if (symbol_matches.empty()) {
            return true;
        }

        // Check operands for variable matches
        auto operands = op->getOperands();
        for (unsigned i = 0; i < operands.size(); ++i) {
            auto operand         = operands[i];
            std::string var_name = extract_variable_name(op, i);
            std::string var_type = type_to_string(operand.getType());

            for (const auto &var_match : symbol_matches) {
                bool name_matches =
                    var_match.name.empty() || matches_pattern(var_name, var_match.name);
                bool type_matches =
                    var_match.type.empty() || matches_pattern(var_type, var_match.type);

                if (name_matches && type_matches) {
                    return true;
                }
            }
        }

        // Check results for variable matches
        auto results = op->getResults();
        for (unsigned i = 0; i < results.size(); ++i) {
            auto result = results[i];
            std::string var_name =
                extract_variable_name(op, static_cast< unsigned >(operands.size() + i));
            std::string var_type = typing::convertCIRTypesToCTypes(result.getType());

            for (const auto &var_match : symbol_matches) {
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

    bool OperationMatcher::matches_arguments(
        mlir::Operation *op, const std::vector< ArgumentMatch > &argument_matches
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
                    LOG(WARNING) << "argument_matches: operand " << arg_match.index
                                 << " name '" << var_name << "' does not match expected '"
                                 << arg_match.name << "'\n";
                    return false;
                }
            }

            // Check argument type if specified
            if (!arg_match.type.empty()) {
                auto variable_type = extract_ssa_value_type(operand);
                // Note: The type matching here is relaxed and checks for matching types both
                // following the chain of operations or the direct type of the operand; if the
                // match is found with any one, it will go ahead with patching.
                if (!matches_type(variable_type, arg_match.type)
                    && !matches_type(operand.getType(), arg_match.type))
                {
                    LOG(WARNING) << "argument_matches: operand " << arg_match.index
                                 << " type does not match expected '" << arg_match.type
                                 << "'\n";
                    return false;
                }
            }
        }

        return true;
    }

    bool OperationMatcher::matches_variables(
        mlir::Operation *op, const std::vector< VariableMatch > &variable_matches
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
            std::string var_type = typing::convertCIRTypesToCTypes(operand.getType());

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

    std::string OperationMatcher::extract_callee_name(cir::CallOp call_op) {
        // Extract the called function name from the call operation
        if (auto callee = call_op.getCalleeAttr()) {
            return callee.getLeafReference().str();
        }

        // If no direct callee attribute, try to extract from operands
        auto operands = call_op.getOperands();
        if (!operands.empty()) {
            auto first_operand = operands[0];
            std::string name   = extract_ssa_value_name(first_operand);
            if (!name.empty()) {
                return name;
            }
        }

        return "";
    }

    bool
    OperationMatcher::matches_pattern(const std::string &text, const std::string &pattern) {
        // If no pattern, match all (this is intentionally different from matches_operation_name
        // which rejects empty patterns - here empty patterns mean "don't filter by this
        // criterion")
        if (pattern.empty()) {
            LOG(DEBUG) << "No pattern provided; matching anything\n";
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
                LOG(ERROR
                ) << "Pattern regex was invalid, falling back to string equality check\n";
                return text == pattern;
            }
        }

        return text == pattern;
    }

    bool OperationMatcher::matches_type(mlir::Type type, const std::string &type_pattern) {
        // if no type pattern, match all types
        if (type_pattern.empty()) {
            return true;
        }

        // convert mlir types to c types
        std::string type_str = typing::convertCIRTypesToCTypes(type);
        return matches_pattern(type_str, type_pattern);
    }

    std::string OperationMatcher::extract_variable_name(mlir::Operation *op, unsigned index) {
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

    std::string OperationMatcher::extract_ssa_value_name(mlir::Value value) {
        // Check if the value has a location with name information
        // Clangir does not uses NameLoc for associating names with values and it should go
        // to fallback
        auto loc = value.getLoc();
        if (auto name_loc = mlir::dyn_cast< mlir::NameLoc >(loc)) {
            return name_loc.getName().str();
        }

        // Check if the value is defined by an operation with a symbol
        if (auto defining_op = value.getDefiningOp()) {
            std::string symbol_name = extract_symbol_name(defining_op);
            if (!symbol_name.empty()) {
                return symbol_name;
            }

            // Check for variable-related attributes on the defining operation
            std::string var_name = extract_variable_attribute(defining_op);
            if (!var_name.empty()) {
                return var_name;
            }
        }

        // For block arguments, check if they have names
        if (auto block_arg = mlir::dyn_cast< mlir::BlockArgument >(value)) {
            auto block = block_arg.getOwner();
            if (auto parent_op = block->getParentOp()) {
                // Check for function arguments with names
                if (auto func_op = mlir::dyn_cast< cir::FuncOp >(parent_op)) {
                    auto arg_names = func_op->getAttrOfType< mlir::ArrayAttr >("arg_names");
                    if (arg_names && block_arg.getArgNumber() < arg_names.size()) {
                        if (auto name_attr = mlir::dyn_cast< mlir::StringAttr >(
                                arg_names[block_arg.getArgNumber()]
                            ))
                        {
                            return name_attr.getValue().str();
                        }
                    }
                }
            }
        }

        return "";
    }

    static mlir::Type extract_ssa_value_type_impl(
        mlir::Value value, llvm::SmallPtrSetImpl< mlir::Value > &visited
    ) {
        // Guard against cycles and excessive depth: if we have already visited this value,
        // fall back to its direct type rather than recursing forever.
        if (!visited.insert(value).second) {
            return value.getType();
        }

        if (auto defining_op = value.getDefiningOp()) {
            // For cast operations, follow the cast chain to get the original type
            if (auto cast_op = mlir::dyn_cast< cir::CastOp >(defining_op)) {
                return extract_ssa_value_type_impl(cast_op.getSrc(), visited);
            }

            // For load operations, get the pointed-to type.
            // When the address is an alloca, also check whether the value stored
            // into it comes from a global reference (through casts) so that the
            // pre-cast type of the global is returned instead of the alloca's own
            // widened element type (e.g., !cir.ptr<!void> vs the real struct ptr).
            if (auto load_op = mlir::dyn_cast< cir::LoadOp >(defining_op)) {
                auto addr = load_op.getAddr();
                if (auto *addr_def = addr.getDefiningOp()) {
                    if (mlir::isa< cir::AllocaOp >(addr_def)) {
                        for (auto &use : addr.getUses()) {
                            if (auto store_op =
                                    mlir::dyn_cast< cir::StoreOp >(use.getOwner()))
                            {
                                if (store_op.getAddr() != addr) {
                                    continue;
                                }
                                // Walk through casts to find a GetGlobalOp source.
                                mlir::Value val = store_op.getValue();
                                while (auto *vdef = val.getDefiningOp()) {
                                    if (auto cast_op =
                                            mlir::dyn_cast< cir::CastOp >(vdef))
                                    {
                                        val = cast_op.getOperand();
                                        continue;
                                    }
                                    if (mlir::isa< cir::GetGlobalOp >(vdef)) {
                                        return val.getType();
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
                auto ptr_type = addr.getType();
                if (auto cir_ptr_type = mlir::dyn_cast< cir::PointerType >(ptr_type)) {
                    return cir_ptr_type.getPointee();
                }
            }

            // For get member operations, return the member type directly
            if (auto get_member_op = mlir::dyn_cast< cir::GetMemberOp >(defining_op)) {
                return get_member_op.getType();
            }

            // For other operations, follow the first operand
            auto operands = defining_op->getOperands();
            if (!operands.empty()) {
                return extract_ssa_value_type_impl(operands[0], visited);
            }
        }

        // For block arguments that are function parameters, use the block argument type
        if (auto block_arg = mlir::dyn_cast< mlir::BlockArgument >(value)) {
            if (auto parent_op = block_arg.getOwner()->getParentOp()) {
                if (mlir::dyn_cast< cir::FuncOp >(parent_op)) {
                    return block_arg.getType();
                }
            }
        }

        return value.getType();
    }

    mlir::Type OperationMatcher::extract_ssa_value_type(mlir::Value value) {
        llvm::SmallPtrSet< mlir::Value, 8 > visited;
        return extract_ssa_value_type_impl(value, visited);
    }

    std::string OperationMatcher::extract_symbol_name(mlir::Operation *op) {
        // Check for standard symbol attributes
        if (auto symbol_name =
                op->getAttrOfType< mlir::StringAttr >(mlir::SymbolTable::getSymbolAttrName()))
        {
            return symbol_name.getValue().str();
        }

        // Check for symbol reference attributes
        if (auto symbol_ref = op->getAttrOfType< mlir::SymbolRefAttr >("symbol")) {
            return symbol_ref.getLeafReference().str();
        }

        // Check for function symbol in CIR call operations
        if (auto call_op = mlir::dyn_cast< cir::CallOp >(op)) {
            if (auto callee = call_op.getCalleeAttr()) {
                return callee.getLeafReference().str();
            }
        }

        return {};
    }

    std::string OperationMatcher::extract_variable_attribute(mlir::Operation *op) {
        // check for name or symbol_name attributes
        const std::vector< std::string > var_attr_names = { "name", "symbol_name" };

        for (const auto &attr_name : var_attr_names) {
            if (auto str_attr = op->getAttrOfType< mlir::StringAttr >(attr_name)) {
                return str_attr.getValue().str();
            }
        }

        // Check for CIR-specific variable attributes
        if (auto alloca_op = mlir::dyn_cast< cir::AllocaOp >(op)) {
            if (auto name_attr = alloca_op->getAttrOfType< mlir::StringAttr >("name")) {
                return name_attr.getValue().str();
            }
        }

        if (auto global_op = mlir::dyn_cast< cir::GlobalOp >(op)) {
            auto name_attr = global_op->getAttrOfType< mlir::StringAttr >("name");
            if (name_attr) {
                return name_attr.getValue().str();
            }
        }

        if (auto cast_op = mlir::dyn_cast< cir::CastOp >(op)) {
            auto operand             = cast_op.getOperand();
            std::string operand_name = extract_ssa_value_name(operand);
            if (!operand_name.empty()) {
                return operand_name;
            }
        }

        if (auto get_global_op = mlir::dyn_cast< cir::GetGlobalOp >(op)) {
            auto global_var_name = get_global_op.getName().str();
            if (!global_var_name.empty()) {
                return global_var_name;
            }
        }

        if (auto get_member_op = mlir::dyn_cast< cir::GetMemberOp >(op)) {
            auto operand             = get_member_op.getOperand();
            std::string operand_name = extract_ssa_value_name(operand);
            if (!operand_name.empty()) {
                return operand_name;
            }
        }

        // Check for load/store operations that might reference named variables
        if (auto load_op = mlir::dyn_cast< cir::LoadOp >(op)) {
            auto addr = load_op.getAddr();

            // When the address is an alloca, check whether the value stored into it
            // originates from a global variable (through zero or more casts).  If so,
            // return the global name so that argument_matches can identify the original
            // symbol (e.g. `usb_g`) even when the value is routed through a temporary.
            // We intentionally do NOT generalise this to other stored values (e.g. call
            // results) because the alloca name itself is then the right identifier.
            if (auto *addr_def = addr.getDefiningOp()) {
                if (mlir::isa< cir::AllocaOp >(addr_def)) {
                    for (auto &use : addr.getUses()) {
                        if (auto store_op = mlir::dyn_cast< cir::StoreOp >(use.getOwner())) {
                            if (store_op.getAddr() != addr) {
                                continue;
                            }
                            // Walk through casts to find a GetGlobalOp source.
                            mlir::Value val = store_op.getValue();
                            while (auto *def = val.getDefiningOp()) {
                                if (auto cast_op = mlir::dyn_cast< cir::CastOp >(def)) {
                                    val = cast_op.getOperand();
                                    continue;
                                }
                                if (auto get_global = mlir::dyn_cast< cir::GetGlobalOp >(def)) {
                                    return get_global.getName().str();
                                }
                                break;
                            }
                        }
                    }
                }
            }

            // Fall back to the name of the address itself (e.g., an alloca's "name" attr).
            std::string addr_name = extract_ssa_value_name(addr);
            if (!addr_name.empty()) {
                return addr_name;
            }
        }

        if (auto store_op = mlir::dyn_cast< cir::StoreOp >(op)) {
            // Try to extract name from the stored address
            auto addr             = store_op.getAddr();
            std::string addr_name = extract_ssa_value_name(addr);
            if (!addr_name.empty()) {
                return addr_name;
            }
        }

        return "";
    }

    std::string OperationMatcher::type_to_string(mlir::Type type) {
        std::string type_str;
        llvm::raw_string_ostream os(type_str);
        type.print(os);
        return os.str();
    }

} // namespace patchestry::passes
