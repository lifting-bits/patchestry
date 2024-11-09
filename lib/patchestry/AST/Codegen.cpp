/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <clang/AST/ASTContext.h>
#include <mlir/IR/MLIRContext.h>
#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/AST/Codegen.hpp>

#define VAST_ENABLE_EXCEPTIONS
#include <vast/Util/Warnings.hpp>

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Bytecode/BytecodeWriter.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>
VAST_UNRELAX_WARNINGS

#define GAP_ENABLE_COROUTINES

#include <vast/CodeGen/CodeGenBuilder.hpp>
#include <vast/CodeGen/CodeGenDriver.hpp>
#include <vast/CodeGen/CodeGenMetaGenerator.hpp>
#include <vast/CodeGen/CodeGenPolicy.hpp>
#include <vast/CodeGen/CodeGenVisitorList.hpp>
#include <vast/CodeGen/DataLayout.hpp>
#include <vast/CodeGen/DefaultCodeGenPolicy.hpp>
#include <vast/CodeGen/DefaultVisitor.hpp>
#include <vast/CodeGen/FallthroughVisitor.hpp>
#include <vast/CodeGen/ScopeContext.hpp>
#include <vast/CodeGen/SymbolGenerator.hpp>
#include <vast/CodeGen/TypeCachingProxy.hpp>
#include <vast/CodeGen/UnsupportedVisitor.hpp>

#include <vast/Dialect/Dialects.hpp>
#include <vast/Dialect/Meta/MetaAttributes.hpp>

#include <vast/Frontend/Options.hpp>

#include <vast/Util/Common.hpp>
#include <vast/Util/DataLayout.hpp>

namespace patchestry::ast {

    class MLIRRegistryInitializer
    {
      public:
        explicit MLIRRegistryInitializer(mlir::DialectRegistry &registry) {
            ::vast::registerAllDialects(registry);
            ::mlir::registerAllDialects(registry);
        }
    };

    class MLIRInitializer
    {
      private:
        MLIRInitializer(void) = delete;

        mlir::DialectRegistry registry;
        MLIRRegistryInitializer registry_initializer;
        mutable mlir::MLIRContext context;

      public:
        explicit MLIRInitializer(int);

        inline mlir::MLIRContext &Context(void) const noexcept { return context; }

        ~MLIRInitializer(void);
    };

    MLIRInitializer::MLIRInitializer(int)
        : registry()
        , registry_initializer(registry)
        , context(registry, mlir::MLIRContext::Threading::ENABLED) {
        context.disableMultithreading();
        context.loadAllAvailableDialects();
        context.enableMultithreading();
    }

    MLIRInitializer::~MLIRInitializer(void) { context.disableMultithreading(); }

    static const MLIRInitializer kMLIR(0);

    class MetaGenerator final : public vast::cg::meta_generator
    {
        // MLIR context for generating the MLIR location from source location
        mlir::MLIRContext *const mctx;

        mlir::Location unknown_location;

      public:
        explicit MetaGenerator(mlir::MLIRContext &mctx_)
            : mctx(&mctx_), unknown_location(mlir::UnknownLoc::get(mctx)) {}

        mlir::Location location(const clang::Decl *data) const override {
            return location_impl(data);
        }

        mlir::Location location(const clang::Expr *data) const override {
            return location_impl(data);
        }

        mlir::Location location(const clang::Stmt *data) const override {
            return location_impl(data);
        }

      private:
        template< typename T >
        mlir::Location location_impl(const T *data) const {
            return unknown_location;
        }
    };

    class CodeGenPolicy final : public vast::cg::codegen_policy
    {
      public:
        CodeGenPolicy(void) = default;

        virtual ~CodeGenPolicy(void) = default;

        bool emit_strict_function_return(const vast::cg::clang_function *) const final {
            return false;
        };

        enum vast::cg::missing_return_policy
        get_missing_return_policy(const vast::cg::clang_function *) const final {
            return vast::cg::missing_return_policy::emit_trap;
        }

        bool SkipDeclBody(const void *decl) const { return false; }

        bool skip_function_body(const vast::cg::clang_function *decl) const final {
            return SkipDeclBody(decl);
        }

        bool skip_global_initializer(const vast::cg::clang_var_decl *decl) const final {
            return SkipDeclBody(decl);
        }
    };

    static std::optional< vast::owning_mlir_module_ref > create_module(clang::ASTContext &ctx) {
        auto &mctx = kMLIR.Context();
        auto bld   = vast::cg::mk_codegen_builder(mctx);
        auto mg    = std::make_shared< MetaGenerator >(mctx);
        auto sg =
            std::make_shared< vast::cg::default_symbol_generator >(ctx.createMangleContext());
        auto cp = std::make_shared< CodeGenPolicy >();
        using vast::cg::as_node;
        using vast::cg::as_node_with_list_ref;

        auto visitors = std::make_shared< vast::cg::visitor_list >()
            | as_node< vast::cg::type_caching_proxy >()
            | as_node_with_list_ref< vast::cg::default_visitor >(mctx, ctx, *bld, mg, sg, cp)
            | as_node_with_list_ref< vast::cg::unsup_visitor >(mctx, *bld, mg)
            | as_node< vast::cg::fallthrough_visitor >();

        vast::cg::driver driver(ctx, mctx, std::move(bld), visitors);
        driver.enable_verifier(true);
        driver.emit(const_cast< clang::Decl * >(
            clang::dyn_cast< clang::Decl >(ctx.getTranslationUnitDecl())
        ));

        driver.finalize();
        return std::make_optional(driver.freeze());
    }

    void CodeGenerator::generate_source_ir(clang::ASTContext &ctx, llvm::raw_fd_ostream &os) {
        auto mod   = create_module(ctx);
        auto flags = mlir::OpPrintingFlags();
        flags.enableDebugInfo(true, false);
        (*mod)->print(os, flags);
    }
} // namespace patchestry::ast
