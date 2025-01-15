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
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>

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
#include <vast/Conversion/Passes.hpp>
#include <vast/Dialect/Dialects.hpp>
#include <vast/Dialect/HighLevel/Passes.hpp>
#include <vast/Dialect/LowLevel/Passes.hpp>
#include <vast/Dialect/Meta/MetaAttributes.hpp>
#include <vast/Frontend/FrontendAction.hpp>
#include <vast/Frontend/Options.hpp>
#include <vast/Frontend/Pipelines.hpp>
#include <vast/Target/LLVMIR/Convert.hpp>
#include <vast/Tower/Tower.hpp>
#include <vast/Util/Common.hpp>
#include <vast/Util/DataLayout.hpp>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Codegen/MetaGenerator.hpp>
#include <patchestry/Codegen/PassInstrumentation.hpp>
#include <patchestry/Codegen/PassManager.hpp>
#include <patchestry/Codegen/Serializer.hpp>

namespace patchestry::codegen {

    std::unique_ptr< llvm::Module >
    translate_to_llvm(vast::mlir_module mod, llvm::LLVMContext &llvm_ctx) {
        if (auto target = mod->getAttr(vast::core::CoreDialect::getTargetTripleAttrName())) {
            auto triple = mlir::cast< mlir::StringAttr >(target);
            mod->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(), triple);
            mod->removeAttr(vast::core::CoreDialect::getTargetTripleAttrName());
        }

        mlir::registerBuiltinDialectTranslation(*mod.getContext());
        mlir::registerLLVMDialectTranslation(*mod.getContext());
        return mlir::translateModuleToLLVMIR(mod, llvm_ctx);
    }

    std::optional< vast::owning_mlir_module_ref >
    CodeGenerator::emit_mlir(clang::ASTContext &ctx, const LocationMap &locations) {
        auto &mctx = CodegenInitializer::getInstance().context();
        auto bld   = vast::cg::mk_codegen_builder(mctx);
        auto mg    = std::make_shared< MetaGen >(&ctx, &mctx, locations);
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
        auto maybe_mod = emit_mlir(actx, locations);
        if (!maybe_mod.has_value()) {
            LOG(ERROR) << "Error: Failed to generate mlir module\n";
            return;
        }

#ifdef ENABLE_DEBUG
        {
            auto flags = mlir::OpPrintingFlags();
            flags.enableDebugInfo(true, false);
            (*maybe_mod)->print(llvm::outs(), flags);
        }
#endif

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
    }

    void CodeGenerator::emit_llvmir(
        clang::ASTContext &actx, vast::mlir_module mod, const patchestry::Options &options
    ) {
        llvm::LLVMContext llvm_context;
        mlir::MLIRContext &mctx = CodegenInitializer::getInstance().context();
        vast::target::llvmir::register_vast_to_llvm_ir(mctx);
        process_mlir_module(actx, vast::cc::target_dialect::llvm, mod);
        auto llvm_mod = translate_to_llvm(mod, llvm_context);

        if (options.output_file.empty()) {
            // emit output file
        }

        /*clang::EmitBackendOutput(
            opts.diags, opts.headers, opts.codegen, opts.target, opts.lang, dl, llvm_mod.get(),
            backend_action, &opts.vfs, std::move(output_stream)
        );*/
    }

    void CodeGenerator::emit_asm(
        clang::ASTContext &actx, vast::mlir_module mod, const patchestry::Options &options
    ) {
        (void) actx;
        (void) mod;
        (void) options;
    }

    void CodeGenerator::process_mlir_module(
        clang::ASTContext &actx, vast::cc::target_dialect target, vast::mlir_module mod
    ) {
        auto &sm          = actx.getSourceManager();
        auto &mctx        = CodegenInitializer::getInstance().context();
        auto main_file_id = sm.getMainFileID();
        auto file_buff    = llvm::MemoryBuffer::getMemBuffer(sm.getBufferOrFake(main_file_id));

        llvm::SourceMgr mlir_sm;
        mlir_sm.AddNewSourceBuffer(std::move(file_buff), llvm::SMLoc());

        mlir::SourceMgrDiagnosticVerifierHandler sm_handler(mlir_sm, &mctx);
        auto file_entry = sm.getFileEntryRefForID(main_file_id);
        if (!file_entry) {
            LOG(ERROR) << "failed to recover file entry ref";
            return;
        }

        auto create_vast_args = [&](void) -> vast::cc::vast_args {
            vast::cc::vast_args vargs;
            vargs.push_back(vast::cc::opt::emit_llvm.data());
            vargs.push_back(vast::cc::opt::print_pipeline.data());
            return vargs;
        };

        auto vast_args = create_vast_args();
        auto pipeline =
            vast::cc::setup_pipeline(vast::cc::pipeline_source::ast, target, mctx, vast_args);
        if (!pipeline) {
            LOG(ERROR) << "Failed to setup pipeline\n";
            return;
        }

        auto core = mod.clone();
        (void) core;

        auto bld = std::make_unique< PassInstrumentation >();
        pipeline->addInstrumentation(std::move(bld));

        auto result = pipeline->run(mod);
        if (result.failed()) {
            LOG(ERROR) << "Failed to run mlir passes\n";
        }
    }

    void CodeGenerator::emit_mlir_after_pipeline(
        clang::ASTContext &actx, vast::mlir_module mod, const patchestry::Options &options
    ) {
        mlir::MLIRContext &mctx = CodegenInitializer::getInstance().context();
        PassManagerBuilder bld(&mctx);
        bld.add_passes(options.pipelines);
        auto pm = bld.build();

        auto instr = std::make_unique< PassInstrumentation >();
        pm->addInstrumentation(std::move(instr));
        std::ignore = pm->run(mod);

        if (!options.output_file.empty()) {
            Serializer::serializeToFile(mod, options.output_file);
        }
        (void) actx;
    }

    void CodeGenerator::emit_source_ir(
        clang::ASTContext &actx, const LocationMap &locations,
        const patchestry::Options &options
    ) {
        auto maybe_mod = emit_mlir(actx, locations);
        if (!maybe_mod.has_value()) {
            LOG(ERROR) << "Failed to emit mlir module\n";
            return;
        }

        if (options.emit_mlir) {
        }

        if (options.emit_llvm) {
            emit_llvmir(actx, (*maybe_mod).get(), options);
        }

        if (options.emit_asm) {
            emit_asm(actx, maybe_mod->get(), options);
        }
    }
} // namespace patchestry::codegen
