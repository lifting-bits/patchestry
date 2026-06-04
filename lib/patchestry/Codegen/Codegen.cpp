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

#include <algorithm>
#include <vector>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclBase.h>
#include <clang/AST/DeclGroup.h>
#include <clang/AST/Type.h>
#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <clang/CIR/Dialect/IR/CIRTypes.h>
#include <clang/CIR/LowerToLLVM.h>
#include <clang/CIR/Passes.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>

#include <llvm/IR/LLVMContext.h>

#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Codegen/PassManager.hpp>
#include <patchestry/Codegen/Serializer.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::codegen {

    std::optional< mlir::ModuleOp > CodeGenerator::lower_ast_to_mlir(clang::ASTContext &ctx) {
        // Emit declarations first, then definitions in name (= program-address)
        // order. CIRGen caches the first cir.func it lazily creates for a symbol;
        // an unordered emission order let a caller materialize a callee with too
        // few args before the definition, dropping a parameter from localDeclMap
        // and aborting (emitDeclRefLValue: static local). A fixed order removes
        // the run-to-run variance.
        std::vector< clang::Decl * > pre;
        std::vector< clang::Decl * > defs;
        for (auto *decl : ctx.getTranslationUnitDecl()->noload_decls()) {
            auto *fd = llvm::dyn_cast< clang::FunctionDecl >(decl);
            if (fd && fd->doesThisDeclarationHaveABody()) {
                defs.push_back(decl);
            } else {
                pre.push_back(decl);
            }
        }
        std::stable_sort(defs.begin(), defs.end(), [](clang::Decl *a, clang::Decl *b) {
            return llvm::cast< clang::NamedDecl >(a)->getName()
                 < llvm::cast< clang::NamedDecl >(b)->getName();
        });

        for (auto *decl : pre) {
            cirdriver->HandleTopLevelDecl(clang::DeclGroupRef(decl));
        }
        for (auto *decl : defs) {
            cirdriver->HandleTopLevelDecl(clang::DeclGroupRef(decl));
        }

        cirdriver->emitDeferredDecls();

        reconcile_variadic_decls(ctx, cirdriver->getModule());

        cirdriver->verifyModule();

        return std::make_optional(cirdriver->getModule());
    }

    void CodeGenerator::reconcile_variadic_decls(
        clang::ASTContext &ctx, mlir::ModuleOp mod
    ) {
        // CIRGen can emit a variadic libc function (fcntl, printf, ...) as a
        // non-variadic cir.func when it first materializes through a builtin /
        // no-prototype path, making its variadic call sites fail verification.
        // Use the (variadic-correct) Clang FunctionDecl as ground truth and
        // restore the varargs flag on the emitted declaration.
        llvm::StringSet<> variadic_names;
        for (const auto *decl : ctx.getTranslationUnitDecl()->noload_decls()) {
            const auto *fd = llvm::dyn_cast< clang::FunctionDecl >(decl);
            if (fd == nullptr) {
                continue;
            }
            if (const auto *fpt = fd->getType()->getAs< clang::FunctionProtoType >();
                fpt != nullptr && fpt->isVariadic())
            {
                variadic_names.insert(fd->getName());
            }
        }

        mod.walk([&](cir::FuncOp func) {
            auto func_type = func.getFunctionType();
            if (func_type.isVarArg() || !variadic_names.contains(func.getSymName())) {
                return;
            }
            // Widening to varargs only relaxes the operand-count check, so
            // existing call sites stay valid. Preserve inputs and return type.
            func.setFunctionType(cir::FuncType::get(
                func.getContext(), func_type.getInputs(),
                func_type.getOptionalReturnType(), /*varArg=*/true
            ));
            LOG(INFO) << "Restored varargs on cir.func '" << func.getSymName().str()
                      << "' dropped during CIR lowering\n";
        });
    }

    void
    CodeGenerator::lower_to_ir(clang::ASTContext &actx, const patchestry::Options &options) {
        // Check if diagnostic error is set. If yes, ignore it.
        if (actx.getDiagnostics().hasErrorOccurred()) {
            actx.getDiagnostics().Reset();
        }

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
            cir::direct::populateCIRToLLVMPasses(*pm);
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
    }

} // namespace patchestry::codegen
