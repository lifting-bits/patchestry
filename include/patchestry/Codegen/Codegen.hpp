/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>
#include <string>

#include <clang/AST/ASTContext.h>
#include <clang/Basic/CodeGenOptions.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include <clang/CIR/CIRGenerator.h>

namespace patchestry::codegen {

    /// Which lowered forms to write.  Files are named `<output_prefix>.cir`,
    /// `<output_prefix>.mlir` and `<output_prefix>.ll`.
    struct LoweringOptions
    {
        bool emit_cir  = false;
        bool emit_mlir = false;
        bool emit_llvm = false;
        std::string output_prefix;
    };

    /// Lowers one Clang translation unit to ClangIR and writes the selected
    /// outputs.  Independent of where the AST came from: it needs only the
    /// ASTContext (diagnostics are reported through its engine) and the
    /// codegen options, both of which must outlive this object.  The module
    /// returned by `lower_ast_to_mlir` lives in this object's MLIRContext, so
    /// keep the generator alive while the module is in use.
    class CodeGenerator
    {
      public:
        CodeGenerator(clang::ASTContext &ctx, const clang::CodeGenOptions &cg_opts) : ctx(ctx) {
            cirdriver = std::make_shared< cir::CIRGenerator >(
                ctx.getDiagnostics(), llvm::vfs::getRealFileSystem(), cg_opts
            );
            cirdriver->Initialize(ctx);
        }

        CodeGenerator(const CodeGenerator &)                = delete;
        CodeGenerator &operator=(const CodeGenerator &)     = delete;
        CodeGenerator(CodeGenerator &&) noexcept            = delete;
        CodeGenerator &operator=(CodeGenerator &&) noexcept = delete;

        virtual ~CodeGenerator() = default;

        // Emit the CIR module for the bound ASTContext
        std::optional< mlir::ModuleOp > lower_ast_to_mlir();

        // Write the .cir/.mlir/.ll outputs selected by `options` for an
        // already-lowered module.
        void emit_outputs(mlir::ModuleOp module, const LoweringOptions &options);

      private:
        // Restore the varargs flag on cir.func declarations that ClangIR
        // lowering emitted as non-variadic, using the Clang FunctionDecl as
        // ground truth (works around first-materialization caching in
        // getOrCreateCIRFunction).
        void reconcile_variadic_decls(mlir::ModuleOp mod);

        clang::ASTContext &ctx;
        std::shared_ptr< cir::CIRGenerator > cirdriver;
    };

} // namespace patchestry::codegen
