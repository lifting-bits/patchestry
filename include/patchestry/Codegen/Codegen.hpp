/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <clang/Frontend/ASTUnit.h>
#include <memory>
#include <string>
#include <vector>

#include <clang/AST/ASTContext.h>
#include <clang/CIR/CIRGenerator.h>
#include <clang/Frontend/CompilerInstance.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/Support/VirtualFileSystem.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include <patchestry/Util/Options.hpp>

namespace clang {
    class ASTUnit;
} // namespace clang

namespace llvm {
    class Module;
    class raw_fd_ostream;
} // namespace llvm

namespace patchestry::codegen {

    using LocationMap = std::vector< std::string >;

    class CodeGenerator
    {
      public:
        explicit CodeGenerator(clang::CompilerInstance &ci) : ci(ci) {
            cirdriver = std::make_shared< cir::CIRGenerator >(
                ci.getDiagnostics(), llvm::vfs::getRealFileSystem(), ci.getCodeGenOpts()
            );
            cirdriver->Initialize(ci.getASTContext());
        }

        CodeGenerator(const CodeGenerator &)                = delete;
        CodeGenerator &operator=(const CodeGenerator &)     = delete;
        CodeGenerator(CodeGenerator &&) noexcept            = delete;
        CodeGenerator &operator=(CodeGenerator &&) noexcept = delete;

        virtual ~CodeGenerator() = default;

        // lower clang AST to CIR representation
        void lower_to_ir(clang::ASTContext &actx, const patchestry::Options &options);

        // Emit CIR representation from ASTContext
        std::optional< mlir::ModuleOp > lower_ast_to_mlir(clang::ASTContext &ctx);

        // Emit LLVM IR representation from ASTContext
        std::unique_ptr< llvm::Module >
        lower_ast_to_llvm(clang::ASTContext &ctx, llvm::LLVMContext &llvm_ctx);

      private:
        void emit_cir(clang::ASTContext &ctx, const patchestry::Options &options);

        void visit_locations(clang::ASTContext &ctx);

        clang::CompilerInstance &ci;
        std::shared_ptr< cir::CIRGenerator > cirdriver;
    };

} // namespace patchestry::codegen
