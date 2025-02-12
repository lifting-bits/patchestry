/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/Support/VirtualFileSystem.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <string>

#include <clang/AST/ASTContext.h>
#include <clang/CIR/CIRGenerator.h>
#include <clang/Frontend/CompilerInstance.h>

#include <patchestry/Util/Options.hpp>
#include <vector>

namespace llvm {
    class raw_fd_ostream;
}

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

        void emit_cir(clang::ASTContext &actx, const patchestry::Options &options);

        void emit_tower(
            clang::ASTContext &actx, const LocationMap &locations,
            const patchestry::Options &options
        );

      private:
        std::optional< mlir::ModuleOp > emit_mlir_module(clang::ASTContext &ctx);

        std::optional< mlir::ModuleOp > emit_after_pipeline(
            clang::ASTContext &ctx, mlir::ModuleOp mod,
            const std::vector< std::string > &pipelines
        );

        void visit_locations(clang::ASTContext &ctx);

        clang::CompilerInstance &ci;
        std::shared_ptr< cir::CIRGenerator > cirdriver;
    };

} // namespace patchestry::codegen
