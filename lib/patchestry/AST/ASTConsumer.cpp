/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <cassert>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <memory>
#include <unordered_map>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Attrs.inc>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclBase.h>
#include <clang/AST/Expr.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <clang/Basic/AttrKinds.h>
#include <clang/Basic/ExceptionSpecificationType.h>
#include <clang/Basic/LLVM.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/Specifiers.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

#include "rellic/AST/LoopRefine.h"
#include "rellic/AST/MaterializeConds.h"
#include "rellic/AST/NestedCondProp.h"
#include "rellic/AST/NestedScopeCombine.h"
#include "rellic/AST/ReachBasedRefine.h"
#include "rellic/AST/Z3CondSimplify.h"
#include <rellic/AST/ASTPass.h>
#include <rellic/AST/CondBasedRefine.h>
#include <rellic/AST/DeadStmtElim.h>
#include <rellic/AST/ExprCombine.h>
#include <rellic/AST/LocalDeclRenamer.h>
#include <rellic/AST/StructFieldRenamer.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/AST/ASTNormalizationPipeline.hpp>
#include <patchestry/AST/CfgBuilder.hpp>
#include <patchestry/AST/ClangEmitter.hpp>
#include <patchestry/AST/CollapseStructure.hpp>
#include <patchestry/AST/FunctionBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {

    void PcodeASTConsumer::HandleTranslationUnit(clang::ASTContext &ctx) {
        type_builder = std::make_unique< TypeBuilder >(ci.getASTContext());
        if (!get_program().serialized_types.empty()) {
            type_builder->create_types(ctx, get_program().serialized_types);
        }

        if (!get_program().serialized_globals.empty()) {
            create_globals(ctx, get_program().serialized_globals);
        }

        if (!get_program().serialized_functions.empty()) {
            create_functions(
                ctx, get_program().serialized_functions, get_program().serialized_types
            );
        }

        // if (options.use_ghidra_structuring) {
        // Alternative path: rebuild function bodies using Ghidra-style
        // CollapseStructure directly on the CFG, bypassing goto elimination.
        auto cfgs = buildCfgs(ctx);

        // Build a name→ghidra::Function lookup for switch metadata.
        std::unordered_map< std::string, const ghidra::Function * > name_to_ghidra;
        for (const auto &[key, func] : get_program().serialized_functions) {
            name_to_ghidra[func.name] = &func;
        }

            for (auto &cfg : cfgs) {
                if (!cfg.function || !cfg.function->hasBody()) continue;

                // Populate switch metadata from Ghidra's P-Code JSON before
                // CollapseStructure processes the CFG.
                std::string fn_name = cfg.function->getNameAsString();
                auto it = name_to_ghidra.find(fn_name);
                if (it != name_to_ghidra.end()) {
                    populateSwitchMetadata(cfg, *it->second);
                }

                auto *fn = const_cast<clang::FunctionDecl *>(cfg.function);
                SNodeFactory factory;
                SNode *tree = collapseStructure(cfg, factory, ctx);
                emitClangAST(tree, fn, ctx);
                cleanupPrettyPrint(fn, ctx);
            }
            /* } else if (options.enable_goto_elimination && !runASTNormalizationPipeline(ctx,
             options)) { LOG(ERROR) << "Goto elimination pipeline failed.\n"; if
             (options.goto_elimination_strict) { return;
                 }
             }*/

            if (options.print_tu) {
#ifdef ENABLE_DEBUG
            ctx.getTranslationUnitDecl()->dumpColor();
#endif
            }
    }

    void PcodeASTConsumer::set_sema_context(clang::DeclContext *dc) { sema().CurContext = dc; }

    void PcodeASTConsumer::write_to_file(void) {}

    void PcodeASTConsumer::create_functions(
        clang::ASTContext &ctx, FunctionMap &serialized_functions, TypeMap &serialized_types
    ) {
        std::vector< std::shared_ptr< FunctionBuilder > > func_builders;
        for (const auto &[key, function] : serialized_functions) {
            auto builder = std::make_shared< FunctionBuilder >(
                ci, function, *type_builder, function_declarations, global_variable_declarations
            );

            builder->disable_switch_case_inline = options.disable_switch_case_inline;
            builder->initialize_op_builder();
            func_builders.emplace_back(std::move(builder));
        }

        for (auto &builder : func_builders) {
            auto *func_decl = builder->create_definition(ctx);
            (void) func_decl;
        }
        (void) serialized_types;
    }

    void PcodeASTConsumer::create_globals(
        clang::ASTContext &ctx, VariableMap &serialized_variables
    ) {
        for (auto &[key, variable] : serialized_variables) {
            if (variable.name.empty() || variable.type.empty()) {
                continue;
            }

            auto var_type       = type_builder->get_serialized_types().at(variable.type);
            auto location       = sourceLocation(ctx.getSourceManager(), key);
            auto sanitized_name = sanitize_key_to_ident(variable.name);
            auto *var_decl      = clang::VarDecl::Create(
                ctx, ctx.getTranslationUnitDecl(), location, location,
                &ctx.Idents.get(sanitized_name), var_type,
                ctx.getTrivialTypeSourceInfo(var_type), clang::SC_Extern
            );
            var_decl->setDeclContext(ctx.getTranslationUnitDecl());
            ctx.getTranslationUnitDecl()->addDecl(var_decl);
            global_variable_declarations.emplace(variable.key, var_decl);
        }
    }

} // namespace patchestry::ast
