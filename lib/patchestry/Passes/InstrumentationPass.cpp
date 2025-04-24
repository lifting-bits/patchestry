/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Passes/PatchSpec.hpp"
#include <memory>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <optional>
#include <set>

#include <clang/AST/ASTContext.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <clang/CIR/Dialect/IR/CIROpsEnums.h>
#include <clang/CIR/Dialect/IR/CIRTypes.h>
#include <clang/CIR/LowerToLLVM.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendOptions.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/YAMLTraits.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LLVM.h>

#include <patchestry/Passes/InstrumentationPass.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::passes {

    std::optional< std::string >
    emitModuleAsString(const std::string &filename, const std::string &lang); // NOLINT

    namespace {
        std::string namifyPatchFunction(const std::string &str) {
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

        llvm::Expected< patchestry::passes::PatchSpec >
        parseSpecifications(llvm::StringRef yaml_str) {
            llvm::SourceMgr sm;
            PatchSpec config;
            llvm::errs() << "Parsing YAML: " << yaml_str << "\n";
            llvm::yaml::Input yaml_input(yaml_str, &sm);
            yaml_input >> config;
            if (yaml_input.error()) {
                return llvm::createStringError(
                    llvm::Twine("Failed to parse YAML: ") + yaml_input.error().message()
                );
            }

            return config;
        }

        // Test function to print the parsed config
        void printSpecifications(const PatchSpec &config) {
            llvm::outs() << "Number of patches: " << config.patches.size() << "\n";
            for (const auto &patch : config.patches) {
                llvm::outs() << "  Patch:\n";
                llvm::outs() << "    Function: " << patch.function << "\n";
                llvm::outs() << "    Patch File: " << patch.patch_file << "\n";
                for (const auto &operation : patch.operations) {
                    switch (operation.kind) {
                        case PatchOperationKind::APPLY_BEFORE_PATCH:
                            llvm::outs() << "    Apply Before: " << operation.target << "\n";
                            for (const auto &arg : operation.arguments) {
                                llvm::outs() << "      - " << arg << "\n";
                            }
                            break;
                        case PatchOperationKind::APPLY_AFTER_PATCH:
                            llvm::outs() << "    Apply After: " << operation.target << "\n";
                            for (const auto &arg : operation.arguments) {
                                llvm::outs() << "      - " << arg << "\n";
                            }
                            break;
                        case PatchOperationKind::WRAP_AROUND_PATCH:
                            llvm::outs() << "    Wrap Around: " << operation.target << "\n";
                            break;
                        default:
                            break;
                    }
                }
            }
        }

        cir::CastKind getCastKind(mlir::Type from, mlir::Type to) {
            (void) from;
            (void) to;
            return cir::CastKind::integral;
        }

    } // namespace

    std::unique_ptr< mlir::Pass > createInstrumentationPass(std::string spec_file) {
        return std::make_unique< InstrumentationPass >(spec_file);
    }

    InstrumentationPass::InstrumentationPass(std::string spec) : spec_file(std::move(spec)) {
        auto buffer_or_err = llvm::MemoryBuffer::getFile(spec_file);
        if (!buffer_or_err) {
            LOG(ERROR) << "Error: Failed to read patch specification file: " << spec_file
                       << "\n";
            return;
        }

        auto config_or_err = parseSpecifications(buffer_or_err.get()->getBuffer());
        if (!config_or_err) {
            LOG(ERROR) << "Error: Failed to parse patch specification file: " << spec_file
                       << "\n";
            return;
        }

        config = std::move(config_or_err.get());
        for (auto &patch : config->patches) {
            if (!llvm::sys::fs::exists(patch.patch_file)) {
                LOG(ERROR) << "Patch file " << patch.patch_file << " does not exist\n";
                continue;
            }

            patch.patch_module = emitModuleAsString(patch.patch_file, config->arch);
        }

        // print specifications
        printSpecifications(*config);
    }

    void InstrumentationPass::runOnOperation() {
        mlir::ModuleOp mod = getOperation();
        llvm::SmallVector< cir::FuncOp, 8 > worklist;

        // get the list of functions
        mod.walk([&](cir::FuncOp op) { worklist.push_back(op); });

        // instrument the calls in each function
        for (size_t i = 0; i < worklist.size(); ++i) {
            cir::FuncOp func = worklist[i];
            instrument_function_calls(func);
        }
    }

    void InstrumentationPass::instrument_function_calls(cir::FuncOp func) {
        func.walk([this](cir::CallOp op) {
            if (!config || config->patches.empty()) {
                LOG(ERROR) << "No patch configuration found. Skipping...\n";
                return;
            }

            auto callee_name = op.getCallee()->str();
            assert(!callee_name.empty() && "Callee name is empty");

            std::set< std::string > seen_functions;
            for (const auto &patch : config->patches) {
                if (patch.function != callee_name
                    || seen_functions.find(callee_name) != seen_functions.end())
                {
                    continue;
                }

                seen_functions.emplace(callee_name);
                // Create module from the patch file mlir representation
                auto patch_module = load_patch_module(*op->getContext(), *patch.patch_module);
                if (!patch_module) {
                    LOG(ERROR) << "Failed to load patch module for function: " << callee_name
                               << "\n ";
                    continue;
                }
                for (const auto &operation : patch.operations) {
                    switch (operation.kind) {
                        case PatchOperationKind::APPLY_BEFORE_PATCH:
                            apply_before_patch(op, operation, patch_module.get());
                            break;
                        case PatchOperationKind::APPLY_AFTER_PATCH:
                            apply_after_patch(op, operation, patch_module.get());
                            break;
                        case PatchOperationKind::WRAP_AROUND_PATCH:
                            wrap_around_patch(op, operation, patch_module.get());
                            break;
                        default:
                            break;
                    }
                }
            }
        });
    }

    void InstrumentationPass::prepare_call_arguments(
        mlir::OpBuilder &builder, cir::CallOp op, cir::FuncOp patch_func,
        const PatchOperation &patch, llvm::SmallVector< mlir::Value > &args
    ) {
        if (op.getNumArgOperands() == 0) {
            assert(patch.arguments.empty() && "Patch arguments are not empty");
            assert(
                patch_func.getNumArguments() == 0 && "Patch function arguments are not empty"
            );
            return;
        }

        if (patch_func.getNumArguments() > op.getNumArgOperands()) {
            LOG(ERROR) << "Number of arguments in patch is greater than the number of "
                          "arguments in the call operation\n";
            return;
        }

        // Check if patch function argument is taking return value
        if (patch.arguments.size() == 1 && patch.arguments[0] == "return_value") {
            if (op.getResultTypes().size() == 0) {
                LOG(ERROR) << "Call operation does not have a return value\n";
                return;
            }
            auto patch_argument_type = patch_func.getArguments().front().getType();
            auto call_return_type    = op->getResultTypes().front();
            if (patch_argument_type != call_return_type) {
                auto cast_op = builder.create< cir::CastOp >(
                    op->getLoc(), patch_argument_type,
                    getCastKind(call_return_type, patch_argument_type), op.getResults().front()
                );
                args.append(cast_op->getResults().begin(), cast_op->getResults().end());
                return;
            }
            args.append(op->getResults().begin(), op->getResults().end());
            return;
        }

        auto create_cast = [&](mlir::Value value, mlir::Type type) -> mlir::Value {
            if (value.getType() == type) {
                return value;
            }

            auto cast_op = builder.create< cir::CastOp >(
                op->getLoc(), type, getCastKind(value.getType(), type), value
            );
            return cast_op->getResults().front();
        };
        (void) create_cast;

        // llvm::SmallVector< mlir::Value, 4 > argument_vec;
        for (unsigned i = 0; i < patch.arguments.size(); i++) {
            auto patch_arg_type = patch_func.getArgumentTypes()[i];
            args.push_back(create_cast(op.getArgOperands()[i], patch_arg_type));
        }
    }

    void InstrumentationPass::apply_before_patch(
        cir::CallOp op, const PatchOperation &patch, mlir::ModuleOp patch_module
    ) {
        mlir::OpBuilder builder(op);
        builder.setInsertionPoint(op);
        auto module = op->getParentOfType< mlir::ModuleOp >();

        std::string patch_function_name = namifyPatchFunction(patch.target);
        auto input_types                = llvm::to_vector(op->getOperandTypes());
        if (!patch_module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        auto result = merge_module_symbol(module, patch_module, patch_function_name);
        if (mlir::failed(result)) {
            LOG(ERROR) << "Failed to insert symbol into module\n";
            return;
        }

        auto patch_func = module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!patch_func) {
            LOG(ERROR) << "Patch function " << patch_function_name
                       << " not defined. Patching failed...\n";
            return;
        }

        auto symbol_ref = mlir::FlatSymbolRefAttr::get(op->getContext(), patch_function_name);
        llvm::SmallVector< mlir::Value > function_args;
        prepare_call_arguments(builder, op, patch_func, patch, function_args);
        auto call_op = builder.create< cir::CallOp >(
            op->getLoc(), symbol_ref,
            patch_func->getResultTypes().size() != 0 ? patch_func->getResultTypes().front()
                                                     : mlir::Type(),
            function_args
        );
        call_op->setAttr("extra_attrs", op.getExtraAttrs());
    }

    void InstrumentationPass::apply_after_patch(
        cir::CallOp op, const PatchOperation &patch, mlir::ModuleOp patch_module
    ) {
        mlir::OpBuilder builder(op);
        auto module = op->getParentOfType< mlir::ModuleOp >();
        builder.setInsertionPointAfter(op);

        std::string patch_function_name = namifyPatchFunction(patch.target);
        auto input_types                = llvm::to_vector(op.getResultTypes());
        if (!patch_module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        auto result = merge_module_symbol(module, patch_module, patch_function_name);
        if (mlir::failed(result)) {
            LOG(ERROR) << "Failed to insert symbol into module\n";
            return;
        }

        auto patch_func = module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!patch_func) {
            LOG(ERROR) << "Patch function " << patch_function_name
                       << " not defined. Patching failed...\n";
            return;
        }

        auto symbol_ref = mlir::FlatSymbolRefAttr::get(op->getContext(), patch_function_name);
        llvm::SmallVector< mlir::Value > function_args;
        prepare_call_arguments(builder, op, patch_func, patch, function_args);
        auto call_op = builder.create< cir::CallOp >(
            op->getLoc(), symbol_ref,
            patch_func->getResultTypes().size() != 0 ? patch_func->getResultTypes().front()
                                                     : mlir::Type(),
            function_args
        );
        call_op->setAttr("extra_attrs", op.getExtraAttrs());
    }

    void InstrumentationPass::wrap_around_patch(
        cir::CallOp op, const PatchOperation &patch, mlir::ModuleOp patch_module
    ) {
        mlir::OpBuilder builder(op);
        auto loc    = op.getLoc();
        auto *ctx   = op->getContext();
        auto module = op->getParentOfType< mlir::ModuleOp >();
        assert(module && "Wrap around patch: no module found");

        builder.setInsertionPoint(op);

        auto callee_name = op.getCallee()->str();
        assert(!callee_name.empty() && "Wrap around patch: callee name is empty");

        auto patch_function_name = namifyPatchFunction(patch.target);
        auto result_types        = llvm::to_vector(op.getResultTypes());

        if (!patch_module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        auto result = merge_module_symbol(module, patch_module, patch_function_name);
        if (mlir::failed(result)) {
            LOG(ERROR) << "Failed to insert symbol into module\n";
            return;
        }

        auto wrap_func = module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!wrap_func) {
            LOG(ERROR) << "Wrap around patch: patch function " << patch.target
                       << " not defined. Patching failed...\n";
            return;
        }

        auto wrap_func_ref = mlir::FlatSymbolRefAttr::get(ctx, patch_function_name);
        auto wrap_call_op  = builder.create< cir::CallOp >(
            loc, wrap_func_ref, result_types.size() != 0 ? result_types.front() : mlir::Type(),
            op.getArgOperands()
        );
        wrap_call_op->setAttr("extra_attrs", op.getExtraAttrs());
        op.replaceAllUsesWith(wrap_call_op);
        op.erase();
    }

    mlir::OwningOpRef< mlir::ModuleOp > InstrumentationPass::load_patch_module(
        mlir::MLIRContext &ctx, const std::string &patch_string
    ) {
        return mlir::parseSourceString< mlir::ModuleOp >(patch_string, &ctx);
    }

    mlir::LogicalResult InstrumentationPass::merge_module_symbol(
        mlir::ModuleOp dest, mlir::ModuleOp src, const std::string &symbol_name
    ) {
        mlir::SymbolTable dest_sym_table(dest);
        mlir::SymbolTable src_sym_table(src);
        (void) symbol_name;

        for (auto &op : *src.getBody()) {
            if (auto sym_op = mlir::dyn_cast< mlir::SymbolOpInterface >(op)) {
                std::string sym_name = sym_op.getName().str();
                if (dest_sym_table.lookup(sym_name) != nullptr) {
                    LOG(INFO) << "Symbol " << sym_name
                              << " already exists in destination module, skipping\n";
                    continue;
                }

                // Clone and insert the symbol into the destination module
                dest.push_back(op.clone());
            }
        }

        return mlir::success();
    }

} // namespace patchestry::passes
