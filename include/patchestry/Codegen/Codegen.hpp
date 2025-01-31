/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include <clang/AST/ASTContext.h>
#include <clang/Frontend/CompilerInstance.h>

#include <vast/Frontend/FrontendAction.hpp>
#include <vast/Frontend/Options.hpp>

#include <patchestry/Util/Options.hpp>

namespace llvm {
    class raw_fd_ostream;
}

namespace patchestry::codegen {

    using LocationMap = std::unordered_map< void *, std::string >;

    class MLIRRegistry
    {
      public:
        explicit MLIRRegistry(mlir::DialectRegistry &registry);
    };

    class CodegenInitializer
    {
      public:
        // Delete copy and move constructors and assignment operators
        CodegenInitializer(const CodegenInitializer &)                = delete;
        CodegenInitializer &operator=(const CodegenInitializer &)     = delete;
        CodegenInitializer(CodegenInitializer &&) noexcept            = delete;
        CodegenInitializer &operator=(CodegenInitializer &&) noexcept = delete;

        // Public static method to access the singleton instance
        static CodegenInitializer &getInstance() {
            static CodegenInitializer instance(0);
            return instance;
        }

        inline mlir::MLIRContext &context() const noexcept { return ctx; }

        ~CodegenInitializer();

      private:
        explicit CodegenInitializer(int /*unused*/);

        // Members
        mlir::DialectRegistry registry;
        MLIRRegistry registry_initializer;
        mutable mlir::MLIRContext ctx;
    };

    class CodeGenerator
    {
      public:
        explicit CodeGenerator(clang::CompilerInstance &ci) : opts(vast::cc::options(ci)) {}

        CodeGenerator(const CodeGenerator &)                = delete;
        CodeGenerator &operator=(const CodeGenerator &)     = delete;
        CodeGenerator(CodeGenerator &&) noexcept            = delete;
        CodeGenerator &operator=(CodeGenerator &&) noexcept = delete;

        virtual ~CodeGenerator() = default;

        void emit_source_ir(
            clang::ASTContext &actx, const LocationMap &locations,
            const patchestry::Options &options
        );

        void emit_tower(
            clang::ASTContext &actx, const LocationMap &locations,
            const patchestry::Options &options
        );

      private:
        std::optional< vast::owning_mlir_module_ref >
        emit_mlir_module(clang::ASTContext &ctx, const LocationMap &locations);

        vast::cc::action_options opts;
    };

} // namespace patchestry::codegen
