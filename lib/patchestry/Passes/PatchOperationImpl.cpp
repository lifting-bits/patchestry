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

namespace patchestry {
    namespace passes {

        extern std::string namifyFunction(const std::string &str);

        namespace {
            std::string make_log_prefix(llvm::StringRef context_label) {
                if (context_label.empty()) {
                    return {};
                }
                std::string prefix = context_label.str();
                prefix.append(": ");
                return prefix;
            }
        } // namespace

        cir::FuncOp PatchOperationImpl::ensurePatchFunctionAvailable(
            InstrumentationPass &pass, mlir::ModuleOp target_module,
            mlir::ModuleOp patch_module, const std::string &patch_function_name,
            llvm::StringRef context_label
        ) {
            const auto prefix = make_log_prefix(context_label);

            if (!patch_module) {
                LOG(ERROR) << prefix << "patch module is null";
                return {};
            }

            if (patch_function_name.empty()) {
                LOG(ERROR) << prefix << "patch function name is empty";
                return {};
            }

            auto patch_func_from_module =
                patch_module.lookupSymbol< cir::FuncOp >(patch_function_name);
            if (!patch_func_from_module) {
                LOG(ERROR) << prefix << "patch function " << patch_function_name
                           << " not defined in patch module";
                return {};
            }

            if (!target_module) {
                LOG(ERROR) << prefix << "target module is null";
                return {};
            }

            auto patch_func = target_module.lookupSymbol< cir::FuncOp >(patch_function_name);
            if (!patch_func) {
                auto merge_result =
                    pass.merge_module_symbol(target_module, patch_module, patch_function_name);
                if (mlir::failed(merge_result)) {
                    LOG(ERROR) << prefix << "failed to merge patch function "
                               << patch_function_name;
                    return {};
                }

                patch_func = target_module.lookupSymbol< cir::FuncOp >(patch_function_name);
                if (!patch_func) {
                    LOG(ERROR) << prefix << "patch function " << patch_function_name
                               << " missing after merge";
                    return {};
                }
            } else {
                LOG(INFO) << prefix << "patch function " << patch_function_name
                          << " already present, skipping merge";
            }

            return patch_func;
        }

        void PatchOperationImpl::applyBeforePatch(
            InstrumentationPass &pass, mlir::Operation *target_op,
            const PatchInformation &patch, mlir::ModuleOp patch_module, bool should_inline
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

            std::string patch_function_name = namifyFunction(patch_spec.function_name);
            auto patch_func                 = ensurePatchFunctionAvailable(
                pass, module, patch_module, patch_function_name, "Patch before"
            );
            if (!patch_func) {
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

            pass.update_state_after_patch(
                builder, patch_call_op, target_op, patch, function_args_map
            );

            // Set appropriate attributes based on operation type
            pass.set_instrumentation_call_attributes(patch_call_op, target_op);

            if (should_inline) {
                pass.inline_worklists.insert(patch_call_op);
            }
        }

        void PatchOperationImpl::applyAfterPatch(
            InstrumentationPass &pass, mlir::Operation *target_op,
            const PatchInformation &patch, mlir::ModuleOp patch_module, bool inline_patches
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

            std::string patch_function_name = namifyFunction(patch_spec.function_name);
            auto patch_func                 = ensurePatchFunctionAvailable(
                pass, module, patch_module, patch_function_name, "Patch after"
            );
            if (!patch_func) {
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

            pass.update_state_after_patch(
                builder, patch_call_op, target_op, patch, function_args_map
            );

            // Set appropriate attributes based on operation type
            pass.set_instrumentation_call_attributes(patch_call_op, target_op);

            if (inline_patches) {
                pass.inline_worklists.insert(patch_call_op);
            }
        }

        void PatchOperationImpl::replaceCallWithPatch(
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

            auto patch_function_name = namifyFunction(patch_spec.function_name);
            auto wrap_func           = ensurePatchFunctionAvailable(
                pass, module, patch_module, patch_function_name, "Wrap around patch"
            );
            if (!wrap_func) {
                return;
            }

            auto wrap_func_ref = mlir::FlatSymbolRefAttr::get(ctx, patch_function_name);
            llvm::MapVector< mlir::Value, mlir::Value > function_args_map;
            pass.prepare_patch_call_arguments(
                builder, call_op, wrap_func, patch, function_args_map
            );
            llvm::SmallVector< mlir::Value > wrap_call_args;
            for (auto &[old_arg, new_arg] : function_args_map) {
                wrap_call_args.push_back(new_arg);
            }
            auto wrap_function_type = wrap_func.getFunctionType();
            auto wrap_call_op       = builder.create< cir::CallOp >(
                loc, wrap_func_ref,
                wrap_function_type ? wrap_function_type.getReturnType() : mlir::Type(),
                wrap_call_args
            );

            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfter(wrap_call_op);
            mlir::DominanceInfo DT(wrap_call_op->getParentOfType< mlir::FunctionOpInterface >()
            );

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
                pass.inline_worklists.insert(wrap_call_op);
            }
        }

        void PatchOperationImpl::replaceOperationWithPatch(
            InstrumentationPass &pass, mlir::Operation *op,
            const PatchInformation &patch, mlir::ModuleOp patch_module,
            bool inline_patches
        ) {
            if (op->getNumResults() == 0) {
                LOG(ERROR) << "REPLACE mode requires an operation with results, got: "
                           << op->getName().getStringRef().str() << "\n";
                return;
            }

            mlir::OpBuilder builder(op);
            auto loc    = op->getLoc();
            auto *ctx   = op->getContext();
            auto module = op->getParentOfType< mlir::ModuleOp >();
            assert(module && "Replace operation: no module found");

            builder.setInsertionPoint(op);

            const auto &patch_spec = patch.spec.value();

            auto patch_function_name = namifyFunction(patch_spec.function_name);
            auto patch_func          = ensurePatchFunctionAvailable(
                pass, module, patch_module, patch_function_name, "Replace operation"
            );
            if (!patch_func) {
                return;
            }

            auto patch_func_ref = mlir::FlatSymbolRefAttr::get(ctx, patch_function_name);
            llvm::MapVector< mlir::Value, mlir::Value > function_args_map;
            pass.prepare_patch_call_arguments(
                builder, op, patch_func, patch, function_args_map
            );
            llvm::SmallVector< mlir::Value > call_args;
            for (auto &[old_arg, new_arg] : function_args_map) {
                call_args.push_back(new_arg);
            }
            auto patch_func_type = patch_func.getFunctionType();
            auto patch_call_op   = builder.create< cir::CallOp >(
                loc, patch_func_ref,
                patch_func_type ? patch_func_type.getReturnType() : mlir::Type(),
                call_args
            );

            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfter(patch_call_op);

            // Replace all uses of old operation results with new call results
            auto op_num_results   = op->getNumResults();
            auto call_num_results = patch_call_op.getNumResults();

            if (call_num_results == 0) {
                LOG(ERROR) << "Patch function returns void but original operation "
                           << "has " << op_num_results << " result(s)\n";
            } else {
                unsigned result_index = 0;
                for (auto result : patch_call_op.getResults()) {
                    if (result_index >= op_num_results) {
                        break;
                    }
                    auto original_result = op->getResult(result_index);
                    if (original_result.getType() != result.getType()) {
                        auto cast_value = pass.create_cast_if_needed(
                            builder, patch_call_op, result, original_result.getType()
                        );
                        original_result.replaceAllUsesWith(cast_value);
                    } else {
                        original_result.replaceAllUsesWith(result);
                    }
                    result_index++;
                }
            }

            // Set appropriate attributes based on operation type
            pass.set_instrumentation_call_attributes(patch_call_op, op);

            if (!op->use_empty()) {
                LOG(ERROR) << "Cannot erase operation, it still has uses: "
                           << op->getName().getStringRef().str() << "\n";
                return;
            }

            op->erase();

            if (inline_patches) {
                pass.inline_worklists.insert(patch_call_op);
            }
        }

        namespace {
            // Build a zero/default value for the given CIR type. Used by
            // ERASE mode to replace uses of a deleted op's results.
            // Returns nullptr if the type is not handled.
            mlir::Value buildDefaultValue(
                mlir::OpBuilder &builder, mlir::Location loc, mlir::Type type
            ) {
                if (auto int_type = mlir::dyn_cast< cir::IntType >(type)) {
                    auto attr = cir::IntAttr::get(
                        int_type,
                        llvm::APSInt(
                            llvm::APInt(int_type.getWidth(), 0), !int_type.isSigned()
                        )
                    );
                    return builder.create< cir::ConstantOp >(loc, int_type, attr);
                }
                if (auto ptr_type = mlir::dyn_cast< cir::PointerType >(type)) {
                    auto zero_int_type =
                        cir::IntType::get(builder.getContext(), 64, false);
                    auto zero_attr = cir::IntAttr::get(
                        zero_int_type,
                        llvm::APSInt(llvm::APInt(64, 0), true)
                    );
                    auto zero_int =
                        builder.create< cir::ConstantOp >(loc, zero_int_type, zero_attr);
                    return builder.create< cir::CastOp >(
                        loc, ptr_type, cir::CastKind::int_to_ptr, zero_int
                    );
                }
                if (auto bool_type = mlir::dyn_cast< cir::BoolType >(type)) {
                    auto attr = cir::BoolAttr::get(builder.getContext(), bool_type, false);
                    return builder.create< cir::ConstantOp >(loc, bool_type, attr);
                }
                return nullptr;
            }
        } // namespace

        void PatchOperationImpl::eraseOperation(
            InstrumentationPass &pass, mlir::Operation *op
        ) {
            (void) pass;
            if (op == nullptr) {
                LOG(ERROR) << "Erase: operation is null\n";
                return;
            }

            // If the op has live uses, replace each result with a default
            // value (zero / null) of the matching type so dependent ops
            // remain well-formed.
            if (!op->use_empty()) {
                mlir::OpBuilder builder(op);
                builder.setInsertionPoint(op);
                for (auto result : op->getResults()) {
                    if (result.use_empty()) {
                        continue;
                    }
                    auto default_val =
                        buildDefaultValue(builder, op->getLoc(), result.getType());
                    if (!default_val) {
                        LOG(ERROR) << "Erase: cannot build default value for result "
                                      "type of '"
                                   << op->getName().getStringRef().str()
                                   << "', skipping\n";
                        return;
                    }
                    result.replaceAllUsesWith(default_val);
                }
            }

            op->erase();
        }

        void PatchOperationImpl::applyPatchAtEntrypoint(
            InstrumentationPass &pass, cir::CallOp call_op,
            const PatchInformation &patch, mlir::ModuleOp patch_module, bool should_inline
        ) {
            if (call_op == nullptr) {
                LOG(ERROR) << "applyPatchAtEntrypoint: the matched call was null";
                return;
            }

            const auto &patch_spec = patch.spec.value();
            std::string patch_function_name = namifyFunction(patch_spec.function_name);

            // APPLY_AT_ENTRYPOINT inserts the patch at the beginning of the enclosing
            // function (i.e. the function that contains the matched call), not the
            // callee. The matched call identifies which enclosing function to
            // instrument. Argument sources are resolved against the entry block:
            //   - OPERAND index N  → Nth block argument of the enclosing function
            //   - VARIABLE/SYMBOL  → alloca/global already live at the entry block
            //   - CONSTANT         → created inline, always valid
            //   - RETURN_VALUE     → rejected (only defined at the call site)
            //   - CAPTURE          → rejected (only bound at the match site)
            auto enclosing_func = call_op->getParentOfType< cir::FuncOp >();
            if (!enclosing_func) {
                LOG(ERROR) << "applyPatchAtEntrypoint: cannot find enclosing function\n";
                return;
            }
            if (enclosing_func.getBody().empty()) {
                LOG(ERROR) << "applyPatchAtEntrypoint: enclosing function has no body\n";
                return;
            }

            auto module = call_op->getParentOfType< mlir::ModuleOp >();
            assert(module && "applyPatchAtEntrypoint: no module found");

            auto patch_func = ensurePatchFunctionAvailable(
                pass, module, patch_module, patch_function_name, "Patch at entrypoint"
            );
            if (!patch_func) {
                return;
            }

            // Find the insertion point just after all alloca ops and any stores that
            // initialize parameters (i.e. stores whose value is a block argument).
            // This mirrors applyContractAtEntrypoint so that "source: variable" can
            // load a parameter value written by the prologue.
            mlir::Block &entry_block = enclosing_func.getBody().front();
            auto insert_pos          = entry_block.begin();
            while (insert_pos != entry_block.end()
                   && mlir::isa< cir::AllocaOp >(*insert_pos))
            {
                ++insert_pos;
            }
            while (insert_pos != entry_block.end()) {
                if (auto store_op = mlir::dyn_cast< cir::StoreOp >(&*insert_pos)) {
                    if (mlir::isa< mlir::BlockArgument >(store_op->getOperand(0))) {
                        ++insert_pos;
                        continue;
                    }
                }
                break;
            }
            mlir::OpBuilder builder(&entry_block, insert_pos);

            auto symbol_ref = mlir::FlatSymbolRefAttr::get(
                call_op->getContext(), patch_function_name
            );
            llvm::MapVector< mlir::Value, mlir::Value > function_args_map;
            // Pass enclosing_func so that OPERAND sources remap to block arguments and
            // RETURN_VALUE / CAPTURE are rejected — both prevent call-site SSA values
            // from leaking into the entry block (which would violate dominance).
            pass.prepare_patch_call_arguments(
                builder, call_op, patch_func, patch, function_args_map, enclosing_func
            );
            llvm::SmallVector< mlir::Value > call_args;
            for (auto &[old_arg, new_arg] : function_args_map) {
                call_args.push_back(new_arg);
            }

            auto patch_call_op = builder.create< cir::CallOp >(
                enclosing_func->getLoc(), symbol_ref,
                patch_func->getResultTypes().size() != 0
                    ? patch_func->getResultTypes().front()
                    : mlir::Type(),
                call_args
            );

            pass.set_instrumentation_call_attributes(patch_call_op, call_op);

            if (should_inline) {
                pass.inline_worklists.insert(patch_call_op);
            }
        }

    } // namespace passes
} // namespace patchestry
