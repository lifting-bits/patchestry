/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include "patchestry/Util/Options.hpp"
#include <mlir/IR/MLIRContext.h>
#include <unordered_map>

#include <clang/AST/ASTContext.h>
#include <clang/Frontend/CompilerInstance.h>
#include <mlir/Pass/PassRegistry.h>

#include <vast/Frontend/FrontendAction.hpp>
#include <vast/Frontend/Options.hpp>
#include <vast/Frontend/Targets.hpp>

namespace llvm {
    class raw_fd_ostream;
} // namespace llvm

namespace patchestry::codegen {

    using LocationMap = std::unordered_map< void *, std::string >;

    class MLIRRegistryInitializer
    {
      public:
        explicit MLIRRegistryInitializer(mlir::DialectRegistry &registry);
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
        MLIRRegistryInitializer registry_initializer;
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

        void emit_tower(
            clang::ASTContext &actx, const LocationMap &locations,
            const patchestry::Options &options
        );

        void emit_source_ir(
            clang::ASTContext &actx, const LocationMap &locations,
            const patchestry::Options &options
        );

      private:
        void process_mlir_module(
            clang::ASTContext &actx, vast::cc::target_dialect target, vast::mlir_module mod
        );

        void emit_mlir_after_pipeline(
            clang::ASTContext &actx, vast::mlir_module mod, const patchestry::Options &options
        );

        void emit_llvmir(
            clang::ASTContext &actx, vast::mlir_module mod, const patchestry::Options &options
        );

        void emit_asm(
            clang::ASTContext &actx, vast::mlir_module mod, const patchestry::Options &options
        );

        std::optional< vast::owning_mlir_module_ref >
        emit_mlir(clang::ASTContext &ctx, const LocationMap &locations);

        vast::cc::action_options opts;
    };

} // namespace patchestry::codegen
