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
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Attributes.h>
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

#include <vast/Frontend/FrontendAction.hpp>
#include <vast/Frontend/Options.hpp>

#include <vast/CodeGen/AttrVisitorProxy.hpp>
#include <vast/CodeGen/DefaultCodeGenPolicy.hpp>
#include <vast/CodeGen/DefaultMetaGenerator.hpp>
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
        mutable mlir::MLIRContext ctx;

      public:
        explicit MLIRInitializer(int);

        inline mlir::MLIRContext &context(void) const noexcept { return ctx; }

        ~MLIRInitializer(void);
    };

    MLIRInitializer::MLIRInitializer(int)
        : registry()
        , registry_initializer(registry)
        , ctx(registry, mlir::MLIRContext::Threading::ENABLED) {
        ctx.disableMultithreading();
        ctx.loadAllAvailableDialects();
        ctx.enableMultithreading();
    }

    MLIRInitializer::~MLIRInitializer(void) { ctx.disableMultithreading(); }

    static const MLIRInitializer kMLIR(0);

    struct MetaGen final : vast::cg::meta_generator
    {
        MetaGen(clang::ASTContext *actx, mlir::MLIRContext *mctx) : actx(actx), mctx(mctx) {}

        mlir::Location location(const clang::Decl *decl) const override {
            return location(decl->getLocation());
        }

        mlir::Location location(const clang::Stmt *stmt) const override {
            return location(stmt->getBeginLoc());
        }

        mlir::Location location(const clang::Expr *expr) const override {
            return location(expr->getExprLoc());
        }

      private:
        mlir::Location location(const clang::SourceLocation &loc) const {
            (void) loc;
            (void) actx;
            auto attr = vast::meta::IdentifierAttr::get(mctx, 0);
            return mlir::FusedLoc::get({}, attr, mctx);
        }

        clang::ASTContext *actx;
        mlir::MLIRContext *mctx;
    };

    namespace {

        std::optional< vast::owning_mlir_module_ref >
        create_module(clang::ASTContext &ctx, vast::cc::action_options &opts) {
            auto &mctx = kMLIR.context();
            auto bld   = vast::cg::mk_codegen_builder(mctx);
            auto mg    = std::make_shared< MetaGen >(&ctx, &mctx);
            auto sg =
                std::make_shared< vast::cg::default_symbol_generator >(ctx.createMangleContext()
                );
            auto cp = std::make_shared< vast::cg::default_policy >(opts);
            using vast::cg::as_node;
            using vast::cg::as_node_with_list_ref;

            auto visitors = std::make_shared< vast::cg::visitor_list >()
                | as_node_with_list_ref< vast::cg::attr_visitor_proxy >()
                | as_node< vast::cg::type_caching_proxy >()
                | as_node_with_list_ref< vast::cg::default_visitor >(
                                mctx, ctx, *bld, mg, sg, cp
                )
                | as_node_with_list_ref< vast::cg::unsup_visitor >(mctx, *bld, mg)
                | as_node< vast::cg::fallthrough_visitor >();

            vast::cg::driver driver(ctx, mctx, std::move(bld), visitors);
            driver.enable_verifier(true);
            for (const auto &decl : ctx.getTranslationUnitDecl()->noload_decls()) {
                driver.emit(clang::dyn_cast< clang::Decl >(decl));
            }

            driver.finalize();
            return std::make_optional(driver.freeze());
        }
    } // namespace

    void CodeGenerator::generate_source_ir(clang::ASTContext &ctx, llvm::raw_fd_ostream &os) {
        auto mod   = create_module(ctx, opts);
        auto flags = mlir::OpPrintingFlags();
        flags.enableDebugInfo(true, false);
        (*mod)->print(os, flags);
    }
} // namespace patchestry::ast
