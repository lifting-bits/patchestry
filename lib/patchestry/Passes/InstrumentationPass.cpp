/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <memory>
#include <optional>
#include <set>
#include <vector>

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
#include <patchestry/Passes/PatchConfig.hpp>
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

        llvm::Expected< patchestry::passes::PatchConfig >
        parseSpecifications(llvm::StringRef yaml_str) {
            // Create a SourceMgr to hold the YAML string
            llvm::SourceMgr sm;
            llvm::yaml::Input yaml_input(yaml_str, &sm);

            // Parse the YAML into our PatchConfig struct
            PatchConfig config;
            yaml_input >> config;

            // Check for YAML syntax errors
            if (yaml_input.error()) {
                return llvm::createStringError(
                    llvm::Twine("Failed to parse YAML: ") + yaml_input.error().message()
                );
            }

            return config;
        }

        // Test function to print the parsed config
        void printSpecifications(const PatchConfig &config) {
            llvm::outs() << "Number of patches: " << config.specs.size() << "\n";

            for (const auto &patch : config.specs) {
                llvm::outs() << "  Patch:\n";
                llvm::outs() << "    Function: " << patch.function << "\n";
                llvm::outs() << "    Apply Before: " << patch.apply_before.target << "\n";
                llvm::outs() << "    Arguments:\n";

                for (const auto &arg : patch.apply_before.arguments) {
                    llvm::outs() << "      - " << arg << "\n";
                }

                llvm::outs() << "    Apply After: " << patch.apply_after.target << "\n";
                llvm::outs() << "    Return Value: " << patch.apply_after.return_value << "\n";
            }
        }

    } // namespace

    void registerInstrumentationPasses(void) {
        mlir::PassPipelineRegistration< InstrumentationOptions > pipeline(
            "patch-mlir-pipeline", "Instrumentation pass for patching MLIR",
            [](mlir::OpPassManager &pm, const InstrumentationOptions &opts) {
                pm.addPass(std::make_unique< patchestry::passes::InstrumentationPass >(
                    opts.spec_file.getValue()
                ));
            }
        );
    }

    InstrumentationPass::InstrumentationPass(std::string spec) : spec_file(std::move(spec)) {
        if (spec_file.empty()) {
            LOG(ERROR) << "Error: No patch specification file provided\n";
            return;
        }

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
        // Print the parsed config
        printSpecifications(*config_or_err);
        config = std::move(config_or_err.get());
        // Travse through the patches and compile the patch file
        for (auto &patch : config->specs) {
            if (!llvm::sys::fs::exists(patch.patch_file)) {
                LOG(ERROR) << "Patch file " << patch.patch_file << " does not exist\n";
                continue;
            }

            patch.patch_module = emitModuleAsString(patch.patch_file, config->arch);
        }
    }

    void InstrumentationPass::runOnOperation() {
        LOG(ERROR) << "Running instrumentation pass\n";
        mlir::ModuleOp mod = getOperation();
        (void) mod;

        mod.walk([this](cir::FuncOp op) {
            LOG(ERROR) << "Visiting operation: " << op.getSymName() << "\n";
            instrument_function_calls(op);
        });
    }

    void InstrumentationPass::instrument_function_calls(cir::FuncOp func) {
        func.walk([this, func](cir::CallOp op) {
            auto callee_name = op.getCallee()->str();
            if (callee_name.empty()) {
                LOG(ERROR) << "Callee name is empty\n";
                return;
            }

            if (!config || config->specs.empty()) {
                LOG(ERROR) << "No patch configuration found\n";
                return;
            }

            std::set< std::string > seen_functions;
            for (const auto &patch : config->specs) {
                if (patch.function != callee_name) {
                    continue;
                }
                if (seen_functions.find(callee_name) != seen_functions.end()) {
                    LOG(ERROR) << "Function is already patched: " << callee_name
                               << ". Skipping...\n";
                    continue;
                }
                seen_functions.emplace(callee_name);
                if (patch.wrap_around.target.empty()) {
                    if (!patch.apply_before.target.empty()) {
                        apply_before_patch(func, op, patch);
                    }
                    if (!patch.apply_after.target.empty()) {
                        apply_after_patch(func, op, patch);
                    }
                } else {
                    wrap_around_patch(func, op, patch);
                }
            }
        });
    }

    void InstrumentationPass::apply_before_patch(
        cir::FuncOp func, cir::CallOp op, const PatchSpec &patch
    ) {
        mlir::OpBuilder builder(op);
        builder.setInsertionPoint(op);
        auto *ctx   = op->getContext();
        auto module = op->getParentOfType< mlir::ModuleOp >();

        std::string patch_function_name = namifyPatchFunction(patch.apply_before.target);
        auto input_types                = llvm::to_vector(op->getOperandTypes());
        auto maybe_patch_module         = load_patch_module(*ctx, *patch.patch_module);
        if (!maybe_patch_module
            || !maybe_patch_module->lookupSymbol< cir::FuncOp >(patch_function_name))
        {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        auto result =
            merge_module_symbol(module, maybe_patch_module.get(), patch_function_name);
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
        auto call_op    = builder.create< cir::CallOp >(
            op->getLoc(), symbol_ref, mlir::Type(), op.getArgOperands()
        );
        call_op->setAttr("extra_attrs", op.getExtraAttrs());
        (void) func;
    }

    void InstrumentationPass::apply_after_patch(
        cir::FuncOp func, cir::CallOp op, const PatchSpec &patch
    ) {
        mlir::OpBuilder builder(op);
        auto module = op->getParentOfType< mlir::ModuleOp >();
        auto *ctx   = module->getContext();
        builder.setInsertionPointAfter(op);

        std::string patch_function_name = namifyPatchFunction(patch.apply_after.target);
        auto input_types                = llvm::to_vector(op.getResultTypes());
        auto maybe_patch_module         = load_patch_module(*ctx, *patch.patch_module);
        if (!maybe_patch_module
            || !maybe_patch_module->lookupSymbol< cir::FuncOp >(patch_function_name))
        {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        auto result =
            merge_module_symbol(module, maybe_patch_module.get(), patch_function_name);
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
        auto call_op    = builder.create< cir::CallOp >(
            op->getLoc(), symbol_ref, mlir::Type(), op.getResults()
        );
        call_op->setAttr("extra_attrs", op.getExtraAttrs());
        (void) func;
    }

    void InstrumentationPass::wrap_around_patch(
        cir::FuncOp func, cir::CallOp op, const PatchSpec &patch
    ) {
        mlir::OpBuilder builder(op);
        auto loc    = op.getLoc();
        auto *ctx   = op->getContext();
        auto module = op->getParentOfType< mlir::ModuleOp >();
        assert(module && "Wrap around patch: no module found");

        builder.setInsertionPoint(op);

        auto callee_name = op.getCallee()->str();
        assert(!callee_name.empty() && "Wrap around patch: callee name is empty");

        auto patch_function_name = namifyPatchFunction(patch.wrap_around.target);
        auto result_types        = llvm::to_vector(op.getResultTypes());

        auto maybe_patch_module = load_patch_module(*ctx, *patch.patch_module);
        if (!maybe_patch_module
            || !maybe_patch_module->lookupSymbol< cir::FuncOp >(patch_function_name))
        {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        auto result =
            merge_module_symbol(module, maybe_patch_module.get(), patch_function_name);
        if (mlir::failed(result)) {
            LOG(ERROR) << "Failed to insert symbol into module\n";
            return;
        }

        auto wrap_func =
            module.lookupSymbol< cir::FuncOp >(namifyPatchFunction(patch.wrap_around.target));
        if (!wrap_func) {
            LOG(ERROR) << "Wrap around patch: patch function " << patch.wrap_around.target
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
        (void) func;
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

        // Lookup the symbol in the source module
        auto *src_sym = src_sym_table.lookup(symbol_name);
        if (src_sym == nullptr) {
            LOG(ERROR) << "Symbol " << symbol_name << " not found in source module\n";
            return mlir::failure();
        }

        if (dest_sym_table.lookup(symbol_name) != nullptr) {
            LOG(ERROR) << "Symbol " << symbol_name << " already exists in destination module\n";
            return mlir::success();
        }
        // Clone and insert the symbol into the destination module
        auto *cloned_sym = src_sym->clone();
        dest.push_back(cloned_sym);
        return mlir::success();
    }

} // namespace patchestry::passes
