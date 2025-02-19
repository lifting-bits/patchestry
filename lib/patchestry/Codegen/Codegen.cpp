/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Codegen/Codegen.hpp"
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

#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <optional>
#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Codegen/PassManager.hpp>
#include <patchestry/Codegen/Serializer.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::codegen {

    std::optional< mlir::ModuleOp > CodeGenerator::emit_mlir_module(clang::ASTContext &ctx) {
        for (const auto &decl : ctx.getTranslationUnitDecl()->noload_decls()) {
            cirdriver->HandleTopLevelDecl(clang::DeclGroupRef(decl));
        }

        cirdriver->emitDeferredDecls();
        cirdriver->verifyModule();
        return std::make_optional(cirdriver->getModule());
    }

    void CodeGenerator::emit_cir(clang::ASTContext &actx, const patchestry::Options &options) {
        // Check if diagnostic error is set. If yes, ignore it.
        if (actx.getDiagnostics().hasErrorOccurred()) {
            actx.getDiagnostics().Reset();
        }

        auto maybe_mod = emit_mlir_module(actx);
        if (!maybe_mod.has_value()) {
            LOG(ERROR) << "Failed to emit mlir module\n";
            return;
        }

        if (options.emit_cir) {
            Serializer::serializeToFile(*maybe_mod, options.output_file + ".cir");
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
            Serializer::serializeToFile(cloned_mod, options.output_file + ".mlir");
        }

        if (options.emit_llvm) {
            llvm::LLVMContext lctx;
            auto llvm_mod = cir::direct::lowerDirectlyFromCIRToLLVMIR(*maybe_mod, lctx);
            Serializer::serializeToFile(llvm_mod.get(), options.output_file + ".ll");
        }

        if (options.emit_asm) {
            UNIMPLEMENTED("Support for lowering to asm not implemented"); // NOLINT
        }
    }

} // namespace patchestry::codegen
