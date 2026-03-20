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
#include <llvm/Support/FileSystem.h>
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
#include <patchestry/AST/CfgBuilder.hpp>
#include <patchestry/AST/CfgDotEmitter.hpp>
#include <patchestry/AST/ClangEmitter.hpp>
#include <patchestry/AST/CfgFoldStructure.hpp>
#include <patchestry/AST/FunctionBuilder.hpp>
#include <patchestry/AST/SNodeDebug.hpp>
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

        if (options.use_structuring_pass) {
            auto cfgs = BuildCfgs(ctx);

            // Compute output directory for DOT files: same dir as output_file,
            // falling back to input_file's directory.
            std::string dot_dir;
            if (options.emit_dot_cfg) {
                const auto &base = options.output_file.empty()
                    ? options.input_file : options.output_file;
                auto slash = base.find_last_of("/\\");
                if (slash != std::string::npos) {
                    dot_dir = base.substr(0, slash + 1);
                }
            }

            // Build a name→ghidra::Function lookup for switch metadata.
            std::unordered_map< std::string, const ghidra::Function * > name_to_ghidra;
            for (const auto &[key, func] : get_program().serialized_functions) {
                name_to_ghidra[func.name] = &func;
            }

            for (auto &cfg : cfgs) {
                if (!cfg.function || !cfg.function->hasBody()) continue;

                // Populate switch metadata from P-Code JSON before
                // CfgFoldStructure processes the CFG.
                std::string fn_name = cfg.function->getName().str();
                auto it = name_to_ghidra.find(fn_name);
                if (it != name_to_ghidra.end()) {
                    PopulateSwitchMetadata(cfg, *it->second);
                }

                if (options.emit_dot_cfg) {
                    std::string dot_filename = dot_dir + fn_name + ".cfg.dot";
                    std::error_code ec;
                    llvm::raw_fd_ostream dot_os(dot_filename, ec, llvm::sys::fs::OF_Text);
                    if (!ec) {
                        EmitCfgDot(cfg, dot_os);
                        LOG(INFO) << "Wrote CFG DOT: " << dot_filename << "\n";
                    }
                }

                auto *fn = const_cast<clang::FunctionDecl *>(cfg.function);
                SNodeFactory factory;
                SNode *tree = CfgFoldStructure(cfg, factory, ctx, options);

                if (options.verify_structuring) {
                    size_t input_count = CountCfgStmts(cfg);
                    size_t output_count = CountSNodeStmts(tree);
                    if (output_count < input_count) {
                        LOG(WARNING) << "STMT DROP: " << fn_name
                                     << ": input=" << input_count
                                     << " output=" << output_count
                                     << " (lost " << (input_count - output_count) << ")\n";
                    } else {
                        LOG(INFO) << "STMT AUDIT OK: " << fn_name
                                  << ": " << output_count << " stmts\n";
                    }
                }

                EmitClangAST(tree, fn, ctx);
                CleanupPrettyPrint(fn, ctx);

                if (options.emit_dot_cfg) {
                    std::string dot_filename = dot_dir + fn_name + ".snode.dot";
                    std::error_code ec;
                    llvm::raw_fd_ostream dot_os(dot_filename, ec, llvm::sys::fs::OF_Text);
                    if (!ec) {
                        EmitDot(tree, dot_os);
                        LOG(INFO) << "Wrote SNode DOT: " << dot_filename << "\n";
                    }
                }
            }
        }

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

            builder->InitializeOpBuilder();
            func_builders.emplace_back(std::move(builder));
        }

        for (auto &builder : func_builders) {
            // Functions without basic blocks (externals/callees) already have a
            // forward declaration created in FunctionBuilder's constructor.
            if (builder->has_basic_blocks()) {
                builder->create_definition(ctx);
            }
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

            auto var_type       = type_builder->GetSerializedTypes().at(variable.type);
            auto location       = SourceLocation(ctx.getSourceManager(), key);
            auto sanitized_name = SanitizeKeyToIdent(variable.name);
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
