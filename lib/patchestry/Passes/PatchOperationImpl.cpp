/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <mlir/IR/Builders.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/SymbolTable.h>

#include <clang/CIR/Dialect/IR/CIRDialect.h>

#include <patchestry/Passes/InstrumentationPass.hpp>
#include <patchestry/Util/Log.hpp>
#include <patchestry/YAML/PatchSpec.hpp>

#include "PatchOperationImpl.hpp"

namespace patchestry::passes {

    /**
     * @brief Converts a string to a valid function name by replacing invalid characters.
     */
    static std::string namifyFunction(const std::string &str) {
        std::string result;
        for (char c : str) {
            if ((isalnum(c) != 0) || c == '_') {
                result += c;
            } else {
                result += '_';
            }
        }
        return result;
    }

    void PatchOperationImpl::applyPatchBefore(
        InstrumentationPass &pass, mlir::Operation *target_op, const PatchInformation &patch,
        mlir::ModuleOp patch_module, bool should_inline
    ) {
        if (target_op == nullptr) {
            LOG(ERROR) << "Patch before: Operation is null";
            return;
        }

        const auto &patch_action = patch.patch_action.value();
        const auto &patch_spec   = patch.spec.value();
        (void) patch_action; // Suppress unused warning

        mlir::OpBuilder builder(target_op);
        builder.setInsertionPoint(target_op);
        auto module = target_op->getParentOfType< mlir::ModuleOp >();

        std::string patch_function_name =
            namifyFunction(patch_spec.implementation.function_name);
        auto input_types = llvm::to_vector(target_op->getOperandTypes());

        auto patch_func_from_module =
            patch_module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!patch_func_from_module) {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        // Ensure function declaration exists in the module first
        if (mlir::failed(pass.insert_function_declaration(module, patch_func_from_module))) {
            LOG(ERROR) << "Failed to ensure function declaration for " << patch_function_name
                       << "\n";
            return;
        }

        // check if the patch function is already in the module, if not, merge it
        if (!module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            auto result = pass.merge_module_symbol(module, patch_module, patch_function_name);
            if (mlir::failed(result)) {
                LOG(ERROR) << "Failed to insert symbol into module\n";
                return;
            }
        } else {
            LOG(INFO) << "Patch function " << patch_function_name
                      << " already exists in module, skipping merge\n";
        }

        auto patch_func = module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!patch_func) {
            LOG(ERROR) << "Patch function " << patch_function_name
                       << " not defined. Patching failed...\n";
            return;
        }

        auto symbol_ref =
            mlir::FlatSymbolRefAttr::get(target_op->getContext(), patch_function_name);
        llvm::MapVector< mlir::Value, mlir::Value > function_args_map;
        pass.prepare_patch_call_arguments(
            builder, target_op, patch_func, patch, function_args_map
        );
        llvm::SmallVector< mlir::Value > new_function_args;
        for (auto &[old_arg, new_arg] : function_args_map) {
            new_function_args.push_back(new_arg);
        }

        auto patch_call_op = builder.create< cir::CallOp >(
            target_op->getLoc(), symbol_ref,
            mlir::Type(), // return type is void for all apply before and after patches
            new_function_args
        );

        pass.update_state_after_patch(builder, patch_call_op, target_op, patch, function_args_map);

        // Set appropriate attributes based on operation type
        pass.set_instrumentation_call_attributes(patch_call_op, target_op);

        if (should_inline) {
            pass.inline_worklists.push_back(patch_call_op);
        }
    }

    void PatchOperationImpl::applyPatchAfter(
        InstrumentationPass &pass, mlir::Operation *target_op, const PatchInformation &patch,
        mlir::ModuleOp patch_module, bool inline_patches
    ) {
        if (target_op == nullptr) {
            LOG(ERROR) << "Patch after: Operation is null";
            return;
        }

        const auto &patch_action = patch.patch_action.value();
        const auto &patch_spec   = patch.spec.value();
        (void) patch_action; // Suppress unused warning

        mlir::OpBuilder builder(target_op);
        auto module = target_op->getParentOfType< mlir::ModuleOp >();
        builder.setInsertionPointAfter(target_op);

        std::string patch_function_name =
            namifyFunction(patch_spec.implementation.function_name);
        auto input_types = llvm::to_vector(target_op->getResultTypes());

        auto patch_func_from_module =
            patch_module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!patch_func_from_module) {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        // Ensure function declaration exists in the module first
        if (mlir::failed(pass.insert_function_declaration(module, patch_func_from_module))) {
            LOG(ERROR) << "Failed to ensure function declaration for " << patch_function_name
                       << "\n";
            return;
        }

        // check if the patch function is already in the module, if not, merge it
        if (!module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            auto result = pass.merge_module_symbol(module, patch_module, patch_function_name);
            if (mlir::failed(result)) {
                LOG(ERROR) << "Failed to insert symbol into module\n";
                return;
            }
        } else {
            LOG(INFO) << "Patch function " << patch_function_name
                      << " already exists in module, skipping merge\n";
        }

        auto patch_func = module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!patch_func) {
            LOG(ERROR) << "Patch function " << patch_function_name
                       << " not defined. Patching failed...\n";
            return;
        }

        auto symbol_ref =
            mlir::FlatSymbolRefAttr::get(target_op->getContext(), patch_function_name);
        llvm::MapVector< mlir::Value, mlir::Value > function_args_map;
        pass.prepare_patch_call_arguments(
            builder, target_op, patch_func, patch, function_args_map
        );
        llvm::SmallVector< mlir::Value > function_args;
        for (auto &[old_arg, new_arg] : function_args_map) {
            function_args.push_back(new_arg);
        }
        auto patch_call_op = builder.create< cir::CallOp >(
            target_op->getLoc(), symbol_ref,
            mlir::Type(), // return type is void for all apply before and after patches
            function_args
        );

        pass.update_state_after_patch(builder, patch_call_op, target_op, patch, function_args_map);

        // Set appropriate attributes based on operation type
        pass.set_instrumentation_call_attributes(patch_call_op, target_op);

        if (inline_patches) {
            pass.inline_worklists.push_back(patch_call_op);
        }
    }

    void PatchOperationImpl::replaceCall(
        InstrumentationPass &pass, cir::CallOp call_op, const PatchInformation &patch,
        mlir::ModuleOp patch_module, bool inline_patches
    ) {
        mlir::OpBuilder builder(call_op);
        auto loc    = call_op.getLoc();
        auto *ctx   = call_op->getContext();
        auto module = call_op->getParentOfType< mlir::ModuleOp >();
        assert(module && "Wrap around patch: no module found");

        builder.setInsertionPoint(call_op);

        const auto &patch_spec = patch.spec.value();

        auto callee_name = call_op.getCallee()->str();
        assert(!callee_name.empty() && "Wrap around patch: callee name is empty");

        auto patch_function_name = namifyFunction(patch_spec.implementation.function_name);
        auto result_types        = llvm::to_vector(call_op.getResultTypes());

        auto patch_func_from_module =
            patch_module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!patch_func_from_module) {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        // Ensure function declaration exists in the module first
        if (mlir::failed(pass.insert_function_declaration(module, patch_func_from_module))) {
            LOG(ERROR) << "Failed to ensure function declaration for " << patch_function_name
                       << "\n";
            return;
        }

        // check if the patch function is already in the module, if not, merge it
        if (!module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            auto result = pass.merge_module_symbol(module, patch_module, patch_function_name);
            if (mlir::failed(result)) {
                LOG(ERROR) << "Failed to insert symbol into module\n";
                return;
            }
        } else {
            LOG(INFO) << "Patch function " << patch_function_name
                      << " already exists in module, skipping merge\n";
        }

        auto wrap_func = module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!wrap_func) {
            LOG(ERROR) << "Wrap around patch: patch function "
                       << patch_spec.implementation.function_name
                       << " not defined. Patching failed...\n";
            return;
        }

        auto wrap_func_ref = mlir::FlatSymbolRefAttr::get(ctx, patch_function_name);
        llvm::MapVector< mlir::Value, mlir::Value > function_args_map;
        pass.prepare_patch_call_arguments(builder, call_op, wrap_func, patch, function_args_map);
        llvm::SmallVector< mlir::Value > wrap_call_args;
        for (auto &[old_arg, new_arg] : function_args_map) {
            wrap_call_args.push_back(new_arg);
        }
        auto wrap_function_type = wrap_func.getFunctionType();
        auto wrap_call_op = builder.create< cir::CallOp >(
            loc, wrap_func_ref,
            wrap_function_type ? wrap_function_type.getReturnType() : mlir::Type(),
            wrap_call_args
        );

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(wrap_call_op);
        mlir::DominanceInfo DT(wrap_call_op->getParentOfType< mlir::FunctionOpInterface >());

        // Replace all uses of old call results with new call results
        // Handle type mismatches by inserting casts as needed
        auto call_num_results = call_op.getNumResults();
        auto wrap_num_results = wrap_call_op.getNumResults();

        if (call_num_results > 0) {
            if (wrap_num_results == 0) {
                LOG(ERROR) << "Patch function returns void but original function has "
                           << call_num_results << " result(s)\n";
                // For void replacement, we can't replace the uses, so this is an error
                // But we'll try to continue by removing the old call if it's unused
            } else {
                unsigned result_index = 0;
                for (auto result : wrap_call_op.getResults()) {
                    if (result_index >= call_num_results) {
                        break;
                    }
                    auto original_result = call_op.getResults()[result_index];
                    if (original_result.getType() != result.getType()) {
                        auto new_value = pass.create_cast_if_needed(
                            builder, wrap_call_op, result, original_result.getType()
                        );
                        original_result.replaceAllUsesWith(new_value);
                    } else {
                        // Types match, directly replace uses
                        original_result.replaceAllUsesWith(result);
                    }
                    result_index++;
                }
            }
        }

        // Set appropriate attributes based on operation type
        pass.set_instrumentation_call_attributes(wrap_call_op, call_op);

        // Check if there are any remaining uses before erasing
        if (!call_op->use_empty()) {
            LOG(ERROR) << "Cannot erase call_op, it still has uses. "
                       << "Original results: " << call_num_results
                       << ", Patch results: " << wrap_num_results << "\n";
            return;
        }

        call_op.erase();

        if (inline_patches) {
            pass.inline_worklists.push_back(wrap_call_op);
        }
    }

} // namespace patchestry::passes
