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
#include <patchestry/AST/CfgDotEmitter.hpp>
#include <patchestry/AST/ClangEmitter.hpp>
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
            // ---------------------------------------------------------------
            // Pipeline: JSON → CGraph → SNode (goto-based) → Clang AST
            //
            // 1. Create FunctionBuilders (forward declarations + OpBuilder init)
            // 2. For each function with basic blocks:
            //    a. create_definition (FunctionDecl + labels, empty body)
            //    b. BuildCGraph (stmts from JSON, edges from terminals)
            //    c. Emit SNode tree with goto-based control flow
            //    d. EmitClangAST (SNode tree → Clang CompoundStmt body)
            //
            // Functions without basic blocks get forward declarations only
            // (handled by FunctionBuilder's constructor).
            // ---------------------------------------------------------------
            std::vector<std::shared_ptr<FunctionBuilder>> func_builders;
            for (const auto &[key, function] : get_program().serialized_functions) {
                auto builder = std::make_shared<FunctionBuilder>(
                    ci, function, *type_builder, function_declarations,
                    global_variable_declarations
                );
                builder->InitializeOpBuilder();
                func_builders.emplace_back(std::move(builder));
            }

            for (auto &builder : func_builders) {
                if (!builder->has_basic_blocks()) continue;

                auto *fn = builder->create_definition(ctx);
                if (!fn) continue;

                const auto &func = builder->get_function();
                std::string fn_name = func.display_name.empty()
                    ? func.name : func.display_name;

                // Set sema context to function for stmt building
                auto *prev_ctx = builder->enter_function_context(fn);
                CGraph flow_graph = BuildCGraph(*builder, ctx);
                builder->leave_function_context(prev_ctx);
                if (flow_graph.nodes.empty()) {
                    LOG(WARNING) << "BuildCGraph produced empty graph for "
                                 << fn_name << "\n";
                    continue;
                }

                // Optionally emit DOT graph before emission.
                if (options.emit_dot_cfg) {
                    CGraphDotTracer tracer;
                    tracer.fn_name = fn_name;
                    tracer.enabled = true;
                    tracer.Dump(flow_graph, "BuildCGraph", true);
                }

                // Emit the CGraph blocks sequentially with goto-based
                // control flow from terminals.
                // Switch blocks get an SSwitch with goto-to-label cases.
                SNodeFactory factory;
                auto *seq = factory.Make<SSeq>();

                for (auto &node : flow_graph.nodes) {
                    if (node.IsCollapsed()) continue;
                    auto *blk = factory.Make<SBlock>();
                    for (auto *s : node.stmts) blk->AddStmt(s);

                    // Switch block: build SSwitch with goto cases
                    if (!node.switch_cases.empty() && node.branch_cond) {
                        auto *sw = factory.Make<SSwitch>(node.branch_cond);
                        // Use the discriminant's type for case literals to
                        // avoid truncation on targets where uintptr_t > int.
                        // For enum types, use the underlying integer type for
                        // the width since getIntWidth requires a BuiltinType.
                        auto case_type = node.branch_cond->getType();
                        if (case_type->isEnumeralType()) {
                            case_type = case_type->castAs<clang::EnumType>()
                                ->getDecl()->getIntegerType();
                        }
                        unsigned case_width = ctx.getIntWidth(case_type);
                        for (const auto &sc : node.switch_cases) {
                            if (sc.is_default) {
                                // Default arm: goto target label
                                if (sc.succ_index < node.succs.size()) {
                                    auto &tn = flow_graph.Node(node.succs[sc.succ_index]);
                                    std::string lbl = tn.original_label.empty()
                                        ? "block_" + std::to_string(node.succs[sc.succ_index])
                                        : tn.original_label;
                                    sw->SetDefaultBody(factory.Make<SGoto>(
                                        factory.Intern(lbl)));
                                } else {
                                    LOG(WARNING) << "switch default succ_index "
                                                 << sc.succ_index << " out of range (succs="
                                                 << node.succs.size() << ")\n";
                                }
                            } else {
                                auto *val = clang::IntegerLiteral::Create(
                                    ctx, llvm::APInt(case_width, static_cast<uint64_t>(sc.value), true),
                                    case_type, clang::SourceLocation());
                                SNode *body = nullptr;
                                if (sc.succ_index < node.succs.size()) {
                                    auto &tn = flow_graph.Node(node.succs[sc.succ_index]);
                                    std::string lbl = tn.original_label.empty()
                                        ? "block_" + std::to_string(node.succs[sc.succ_index])
                                        : tn.original_label;
                                    body = factory.Make<SGoto>(factory.Intern(lbl));
                                } else {
                                    LOG(WARNING) << "switch case " << sc.value
                                                 << " succ_index " << sc.succ_index
                                                 << " out of range (succs="
                                                 << node.succs.size() << ")\n";
                                }
                                sw->AddCase(val, body);
                            }
                        }
                        auto *sw_seq = factory.Make<SSeq>();
                        if (!blk->Stmts().empty()) sw_seq->AddChild(blk);
                        sw_seq->AddChild(sw);
                        if (!node.label.empty()) {
                            seq->AddChild(factory.Make<SLabel>(
                                factory.Intern(node.label), sw_seq));
                        } else {
                            seq->AddChild(sw_seq);
                        }
                        continue;
                    }

                    // Non-switch: append terminal (goto/if-goto)
                    if (node.terminal) blk->AddStmt(node.terminal);
                    if (!node.label.empty()) {
                        auto *lbl = factory.Make<SLabel>(
                            factory.Intern(node.label), blk);
                        seq->AddChild(lbl);
                    } else {
                        seq->AddChild(blk);
                    }
                }
                EmitClangAST(seq, fn, ctx);

                CleanupPrettyPrint(fn, ctx);
            }
        }

        if (options.print_tu) {
            // Pretty-print the Clang AST as C to <output>.c.
            // This runs before codegen so the C file is always produced
            // even if CIR lowering encounters a diagnostic error.
            if (!options.output_file.empty()) {
                std::error_code ec;
                llvm::raw_fd_ostream out(options.output_file + ".c", ec,
                                         llvm::sys::fs::OF_Text);
                if (!ec) {
                    ctx.getTranslationUnitDecl()->print(
                        out, ctx.getPrintingPolicy(), /*Indentation=*/0);
                } else {
                    LOG(ERROR) << "Failed to write C output: " << ec.message() << "\n";
                }
            } else {
                // No output file — print to stdout
                ctx.getTranslationUnitDecl()->print(
                    llvm::outs(), ctx.getPrintingPolicy(), /*Indentation=*/0);
            }
#ifdef ENABLE_DEBUG
            ctx.getTranslationUnitDecl()->dumpColor();
#endif
        }
    }

    void PcodeASTConsumer::set_sema_context(clang::DeclContext *dc) { sema().CurContext = dc; }

    void PcodeASTConsumer::write_to_file(void) {}

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
