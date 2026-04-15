/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <clang/Frontend/CompilerInstance.h>
#include <llvm/IR/Module.h>
#include <memory>
#include <mlir/Parser/Parser.h>
#include <optional>

#include <clang/AST/ASTContext.h>
#include <clang/AST/DeclBase.h>
#include <clang/AST/DeclGroup.h>
#include <clang/CIR/LowerToLLVM.h>
#include <clang/CIR/Passes.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>

#include "clang/CodeGen/CodeGenAction.h"
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <clang/CodeGen/ModuleBuilder.h>
#ifdef PATCHESTRY_ENABLE_RELLIC
#include <rellic/Decompiler.h>
#endif

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Codegen/PassManager.hpp>
#include <patchestry/Codegen/Serializer.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::codegen {

    std::optional< mlir::ModuleOp > CodeGenerator::lower_ast_to_mlir(clang::ASTContext &ctx) {
        for (const auto &decl : ctx.getTranslationUnitDecl()->noload_decls()) {
            cirdriver->HandleTopLevelDecl(clang::DeclGroupRef(decl));
        }

        cirdriver->emitDeferredDecls();
        cirdriver->verifyModule();

        return std::make_optional(cirdriver->getModule());
    }

    std::unique_ptr< llvm::Module >
    CodeGenerator::lower_ast_to_llvm(clang::ASTContext &ctx, llvm::LLVMContext &llvm_ctx) {
        auto &cg_opts = ci.getCodeGenOpts();
        auto &diags   = ci.getDiagnostics();

        std::unique_ptr< clang::CodeGenerator > cg(clang::CreateLLVMCodeGen(
            diags, ci.getFrontendOpts().OutputFile,
            ci.getFileManager().getVirtualFileSystemPtr(), ci.getHeaderSearchOpts(),
            ci.getPreprocessorOpts(), cg_opts, llvm_ctx
        ));
        cg->Initialize(ctx);
        for (const auto &decl : ctx.getTranslationUnitDecl()->noload_decls()) {
            cg->HandleTopLevelDecl(clang::DeclGroupRef(decl));
        }
        return std::unique_ptr< llvm::Module >(cg->ReleaseModule());
    }

    void
    CodeGenerator::lower_to_ir(clang::ASTContext &actx, const patchestry::Options &options) {
        // Check if diagnostic error is set. If yes, ignore it.
        if (actx.getDiagnostics().hasErrorOccurred()) {
            actx.getDiagnostics().Reset();
        }

#ifdef PATCHESTRY_ENABLE_RELLIC
        // NOTE: We use rellic to improve the generated AST. There are issues running rellic AST
        // passes directly on AST. Temporarily we lower the AST to llvm IR and regenerate AST

        llvm::LLVMContext llvm_ctx;
        auto transform_ast = [&](clang::ASTContext &ctx
                             ) -> std::optional< rellic::DecompilationResult > {
            auto llvm_mod = lower_ast_to_llvm(ctx, llvm_ctx);
            if (!llvm_mod) {
                return {};
            }

            // Decompile llvm IR using rellic
            rellic::DecompilationOptions opts{
                .lower_switches       = false,
                // https://github.com/lifting-bits/rellic/issues/102
                // If there are Phi nodes, it can't be converted back to AST. Enable removing
                // phi node before decompiling.
                .remove_phi_nodes     = true,
                .additional_providers = {},
            };
            auto results = rellic::Decompile(std::move(llvm_mod), std::move(opts));
            if (!results.Succeeded()) {
                LOG(ERROR) << "Failed to decompile LLVM ir for transforming AST"
                           << results.TakeError().message;
                return {};
            }

            return results.TakeValue();
        };

        if (options.use_rellic_transform) {
            auto results = transform_ast(actx);
            if (results && results->ast) {
                emit_cir(results->ast->getASTContext(), options);
                return;
            }
            LOG(WARNING
            ) << "Failed to transform AST using remill; Fallback to default generation";
        }
#else
        if (options.use_rellic_transform) {
            LOG(WARNING
            ) << "use_rellic_transform requested but RELLIC support was disabled at build "
                 "time (PE_USE_RELLIC=OFF); using default generation";
        }
#endif

        emit_cir(actx, options);
    }

    void CodeGenerator::emit_cir(clang::ASTContext &ctx, const patchestry::Options &options) {
        // C pretty-print is now handled by ASTConsumer::HandleTranslationUnit
        // (before codegen) so the .c file is always produced even when CIR
        // lowering encounters a diagnostic error.

        auto maybe_mod = lower_ast_to_mlir(ctx);
        if (!maybe_mod.has_value()) {
            LOG(ERROR) << "Failed to emit mlir module\n";
            return;
        }

        if (options.emit_cir) {
            Serializer::SerializeToFile(*maybe_mod, options.output_file + ".cir");
        }

        if (options.emit_mlir) {
            auto cloned_mod = maybe_mod->clone();
            auto *mctx      = cloned_mod.getContext();
            PassManagerBuilder bld(mctx);
            auto pm = bld.build();
            cir::direct::populateCIRToLLVMPasses(*pm, true);
            auto result = pm->run(cloned_mod);
            if (result.failed()) {
                LOG(ERROR) << "Failed to run conversion passes\n";
                return;
            }
            Serializer::SerializeToFile(cloned_mod, options.output_file + ".mlir");
        }

        if (options.emit_llvm) {
            llvm::LLVMContext lctx;
            auto llvm_mod = cir::direct::lowerDirectlyFromCIRToLLVMIR(*maybe_mod, lctx);
            Serializer::SerializeToFile(llvm_mod.get(), options.output_file + ".ll");
        }

        if (options.emit_asm) {
            LOG_FATAL("Support for lowering to asm not implemented.");
        }
    }

} // namespace patchestry::codegen
