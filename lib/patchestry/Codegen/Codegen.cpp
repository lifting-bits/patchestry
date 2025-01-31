/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#define VAST_ENABLE_EXCEPTIONS
#include <vast/Util/Warnings.hpp>

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <clang/Tooling/Tooling.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>

#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
VAST_UNRELAX_WARNINGS

#define GAP_ENABLE_COROUTINES

#include <vast/CodeGen/AttrVisitorProxy.hpp>
#include <vast/CodeGen/CodeGenBuilder.hpp>
#include <vast/CodeGen/CodeGenDriver.hpp>
#include <vast/CodeGen/CodeGenMetaGenerator.hpp>
#include <vast/CodeGen/CodeGenPolicy.hpp>
#include <vast/CodeGen/CodeGenVisitorList.hpp>
#include <vast/CodeGen/DataLayout.hpp>
#include <vast/CodeGen/DefaultCodeGenPolicy.hpp>
#include <vast/CodeGen/DefaultMetaGenerator.hpp>
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

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Codegen/MetaGenerator.hpp>
#include <patchestry/Codegen/Serializer.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::codegen {

    std::optional< vast::owning_mlir_module_ref >
    CodeGenerator::emit_mlir_module(clang::ASTContext &ctx, const LocationMap &locations) {
        auto &mctx = CodegenInitializer::getInstance().context();
        auto bld   = vast::cg::mk_codegen_builder(mctx);
        auto mg =
            std::make_shared< patchestry::codegen::MetaGenerator >(&ctx, &mctx, locations);
        auto sg =
            std::make_shared< vast::cg::default_symbol_generator >(ctx.createMangleContext());
        auto cp = std::make_shared< vast::cg::default_policy >(opts);
        using vast::cg::as_node;
        using vast::cg::as_node_with_list_ref;

        auto visitors = std::make_shared< vast::cg::visitor_list >()
            | as_node_with_list_ref< vast::cg::attr_visitor_proxy >()
            | as_node< vast::cg::type_caching_proxy >()
            | as_node_with_list_ref< vast::cg::default_visitor >(mctx, ctx, *bld, mg, sg, cp)
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

    void CodeGenerator::emit_tower(
        clang::ASTContext &actx, const LocationMap &locations,
        const patchestry::Options &options
    ) {
        auto maybe_mod = emit_mlir_module(actx, locations);
        if (!maybe_mod.has_value()) {
            LOG(ERROR) << "Error: Failed to generate mlir module\n";
            return;
        }

        UNIMPLEMENTED("Support for lowering to tower not implemented"); // NOLINT

#ifdef ENABLE_DEBUG
        {
            auto flags = mlir::OpPrintingFlags();
            flags.enableDebugInfo(true, false);
            (*maybe_mod)->print(llvm::outs(), flags);
        }
#endif
#if 0
        mlir::MLIRContext &mctx = CodegenInitializer::getInstance().context();

        PassManagerBuilder bld(&mctx);
        bld.add_passes(options.pipelines);
        auto pm = bld.build();

        vast::tw::location_info_t location_info;
        auto tower = vast::tw::tower(
            CodegenInitializer::getInstance().context(), location_info, std::move(*maybe_mod)
        );
        auto link          = tower.apply(tower.top(), location_info, *pm);
        auto parent        = link->parent();
        auto parent_module = parent.mod;
        auto child         = link->child();

    #ifdef ENABLE_DEBUG
        {
            auto flags = mlir::OpPrintingFlags();
            flags.enableDebugInfo(true, false);
            parent_module->print(llvm::outs(), flags);
        }
    #endif
        Serializer::serializeToFile(parent_module, options.output_file + ".parent");
        Serializer::serializeToFile(child.mod, options.output_file + ".child");
#endif
    }

    void CodeGenerator::emit_source_ir(
        clang::ASTContext &actx, const LocationMap &locations,
        const patchestry::Options &options
    ) {
        auto maybe_mod = emit_mlir_module(actx, locations);
        if (!maybe_mod.has_value()) {
            LOG(ERROR) << "Failed to emit mlir module\n";
            return;
        }

        if (options.emit_mlir) {
            Serializer::serializeToFile(maybe_mod->get(), options.output_file + ".mlir");
        }

        if (options.emit_llvm) {
            UNIMPLEMENTED("Support for lowering to llvm not implemented"); // NOLINT
        }

        if (options.emit_asm) {
            UNIMPLEMENTED("Support for lowering to asm not implemented"); // NOLINT
        }
    }
} // namespace patchestry::codegen
