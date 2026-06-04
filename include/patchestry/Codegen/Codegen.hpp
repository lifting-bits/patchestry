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
#include <clang/Frontend/CompilerInstance.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include <clang/CIR/CIRGenerator.h>

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

      private:
        void emit_cir(clang::ASTContext &ctx, const patchestry::Options &options);

        // Restore the varargs flag on cir.func declarations that ClangIR
        // lowering emitted as non-variadic, using the Clang FunctionDecl as
        // ground truth (works around first-materialization caching in
        // getOrCreateCIRFunction).
        void reconcile_variadic_decls(clang::ASTContext &ctx, mlir::ModuleOp mod);

        void visit_locations(clang::ASTContext &ctx);

        clang::CompilerInstance &ci;
        std::shared_ptr< cir::CIRGenerator > cirdriver;
    };

} // namespace patchestry::codegen
