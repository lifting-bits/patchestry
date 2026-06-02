/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInvocation.h>
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
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/ExceptionSpecificationType.h>
#include <clang/Basic/LLVM.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/Specifiers.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/AST/BuildSNodeFromRegion.hpp>
#include <patchestry/AST/CfgDotEmitter.hpp>
#include <patchestry/AST/ClangEmitter.hpp>
#include <patchestry/AST/SNodePostPasses.hpp>
#include <patchestry/AST/FunctionBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {

    namespace {
        // Emit CGraph blocks sequentially with goto-based terminals.
        // Used by --emit-flat-baseline and as the no-region fallback;
        // the post-pass chain drops residual gotos in the latter case.
        void EmitFlatCFG(
            CGraph &flow_graph, SNodeFactory &factory,
            clang::ASTContext &ctx, std::vector< SNode * > &root_body)
        {
            for (auto &node : flow_graph.nodes) {
                if (node.IsCollapsed()) continue;
                std::vector< SNode * > blk_nodes;
                for (auto *s : node.stmts)
                    if (s) blk_nodes.push_back(factory.Make< SStmt >(s));

                if (!node.switch_cases.empty() && node.branch_cond) {
                    auto *sw = factory.Make< SSwitch >(node.branch_cond);
                    // Use discriminant type to avoid truncation when
                    // uintptr_t > int; enums need the underlying integer
                    // type because getIntWidth requires a BuiltinType.
                    auto case_type = node.branch_cond->getType();
                    if (case_type->isEnumeralType()) {
                        case_type = case_type->castAs< clang::EnumType >()
                            ->getDecl()->getIntegerType();
                    }
                    unsigned case_width = ctx.getIntWidth(case_type);
                    for (const auto &sc : node.switch_cases) {
                        if (sc.is_default) {
                            if (sc.succ_index < node.succs.size()) {
                                auto &tn = flow_graph.Node(
                                    node.succs[sc.succ_index]);
                                if (tn.original_label.empty()) {
                                    LOG(FATAL) << "switch default target node "
                                               << node.succs[sc.succ_index]
                                               << " has no original_label — "
                                                  "CGraph builder bug.\n";
                                }
                                sw->SetDefaultBody(factory.Make< SGoto >(
                                    factory.Intern(tn.original_label)));
                            } else {
                                LOG(WARNING)
                                    << "switch default succ_index "
                                    << sc.succ_index
                                    << " out of range (succs="
                                    << node.succs.size() << ")\n";
                            }
                        } else {
                            auto *val = clang::IntegerLiteral::Create(
                                ctx,
                                llvm::APInt(case_width,
                                            static_cast< uint64_t >(sc.value),
                                            true),
                                case_type, VirtualLoc(ctx));
                            SNode *body = nullptr;
                            if (sc.succ_index < node.succs.size()) {
                                auto &tn = flow_graph.Node(
                                    node.succs[sc.succ_index]);
                                if (tn.original_label.empty()) {
                                    LOG(FATAL) << "switch case " << sc.value
                                               << " target node "
                                               << node.succs[sc.succ_index]
                                               << " has no original_label — "
                                                  "CGraph builder bug.\n";
                                }
                                body = factory.Make< SGoto >(
                                    factory.Intern(tn.original_label));
                            } else {
                                LOG(WARNING)
                                    << "switch case " << sc.value
                                    << " succ_index " << sc.succ_index
                                    << " out of range (succs="
                                    << node.succs.size() << ")\n";
                            }
                            sw->AddCase(val, body);
                        }
                    }
                    std::vector< SNode * > sw_seq = std::move(blk_nodes);
                    sw_seq.push_back(sw);
                    if (!node.label.empty()) {
                        root_body.push_back(factory.Make< SLabel >(
                            factory.Intern(node.label),
                            std::move(sw_seq)));
                    } else {
                        for (SNode *s : sw_seq)
                            root_body.push_back(s);
                    }
                    continue;
                }

                if (node.terminal)
                    blk_nodes.push_back(
                        factory.Make< SStmt >(node.terminal));
                if (!node.label.empty()) {
                    root_body.push_back(factory.Make< SLabel >(
                        factory.Intern(node.label),
                        std::move(blk_nodes)));
                } else {
                    for (SNode *s : blk_nodes)
                        root_body.push_back(s);
                }
            }
        }

    } // namespace

    void PcodeASTConsumer::HandleTranslationUnit(clang::ASTContext &ctx) {
        type_builder = std::make_unique< TypeBuilder >(ci.getASTContext());
        if (!get_program().serialized_types.empty()) {
            type_builder->create_types(ctx, get_program().serialized_types);
        }

        if (!get_program().serialized_globals.empty()) {
            create_globals(ctx, get_program().serialized_globals);
        }

        if (!get_program().serialized_functions.empty()) {
            // Pipeline: JSON -> CGraph -> SNode -> Clang AST. Functions
            // without basic blocks get forward declarations only.
            std::vector<std::shared_ptr<FunctionBuilder>> func_builders;
            const auto &program_arch = get_program().arch.value_or(std::string{});
            for (const auto &[key, function] : get_program().serialized_functions) {
                auto builder = std::make_shared<FunctionBuilder>(
                    ci, function, *type_builder, function_declarations,
                    global_variable_declarations, intrinsic_declarations, program_arch
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
                bool seeded_from_ghidra                = false;
                bool snode_cleanup_reached_fixed_point = true;
                // Set when the verifier finds a dangling goto.
                bool unrepaired_region_flow            = false;

                auto *prev_ctx = builder->enter_function_context(fn);
                CGraph flow_graph = BuildCGraph(*builder, ctx);
                builder->leave_function_context(prev_ctx);
                if (flow_graph.nodes.empty()) {
                    LOG(WARNING) << "BuildCGraph produced empty graph for "
                                 << fn_name << "\n";
                    continue;
                }

                if (options.emit_dot_cfg) {
                    CGraphDotTracer tracer;
                    tracer.fn_name = fn_name;
                    tracer.enabled = true;
                    tracer.Dump(flow_graph, "BuildCGraph", true);
                }

                SNodeFactory factory;
                std::vector<SNode *> root_body;
                // Pristine graph for the flat fallback, snapshotted
                // before seeding collapses nodes / clears labels.  Shallow copy
                // is safe: it shares leaf clang::Stmt*, but only one SNode tree
                // is ever emitted and passes never mutate leaf Stmts in place.
                CGraph flat_snapshot;

                if (options.emit_flat_baseline) {
                    EmitFlatCFG(flow_graph, factory, ctx, root_body);
                } else {
                    // Snapshot before seeding mutates flow_graph.
                    flat_snapshot = flow_graph;
                    // Seed from Function.region when present; fall back
                    // to flat-CFG. Post-pass chain runs on either shape.
                    if (func.region.has_value()) {
                        BuildSNodeFromRegion translator(func, flow_graph, factory, ctx);
                        seeded_from_ghidra = translator.TryBuild();
                    }
                    if (seeded_from_ghidra) {
                        // Region translator collapsed every node into entry.
                        for (auto &node : flow_graph.nodes) {
                            if (node.IsCollapsed()) continue;
                            for (SNode *s : node.structured)
                                root_body.push_back(s);
                        }
                    } else {
                        EmitFlatCFG(flow_graph, factory, ctx, root_body);
                    }

                    if (!root_body.empty()) {
                        // Lift raw clang control flow inside SStmt leaves
                        // into first-class SNodes so cleanup passes see it.
                        NormalizeRawControlFlow(root_body, factory, ctx);

                        ConvertGotoToBreakContinue(root_body, factory);
                        ConvertGotoToReturn(root_body, factory, ctx);
                        DuplicateSwitchCaseTargets(root_body, factory);

                        // Cross-region goto resolution: ordering is
                        // load-bearing — see each pass's hpp doc.
                        FoldSwitchLocalCaseTargets(root_body, factory, ctx);
                        DuplicateSmallTerminatingTargets(root_body, factory);
                        SplitAndCloneCrossScopeEntries(root_body, factory, ctx);
                        DuplicateSmallEpilogueTargets(root_body, factory, ctx);
                        DuplicateSwitchFallthroughTargets(root_body, factory, ctx);
                        DuplicateLoopContinueTargets(root_body, factory, ctx);
                        FoldGuardedFallthroughTargets(root_body, factory, ctx);
                        RepairCrossScopeLabelEntries(root_body, factory, ctx);
                        FoldSiblingArmLabelEntries(root_body, factory, ctx);
                        DuplicateStackGuardReturnTargets(root_body, factory, ctx);
                        DuplicateCleanupReturnTargets(root_body, factory, ctx);

                        // Run structural cleanup before goto cleanup so alias
                        // chains and empty shells are gone before InlineResidualGotos
                        // / EliminateGotoToNextLabel observe the tree.
                        SimplifyEmptyControlFlow(root_body, ctx);
                        CollapsePassThroughLabels(root_body, factory);
                        MergeRedundantGotoGuards(root_body, factory, ctx);

                        InlineResidualGotos(root_body, factory);
                        InlineCrossScopeSingleRef(root_body, factory);

                        // Fixed-point loop bounded by kMaxGotoEliminationPasses.
                        // Re-running cross-region resolvers is load-bearing:
                        // cloning one target can expose a fresh cross-region goto.
                        snode_cleanup_reached_fixed_point = false;
                        for (int pass = 0;
                             pass < kMaxGotoEliminationPasses; ++pass) {
                            bool did_absorb = AbsorbFallthroughIntoElse(
                                root_body, factory);
                            bool did_scope = ScopeifyIfGotos(
                                root_body, factory, ctx);
                            bool did_xscope = AbsorbCrossScopeIfGoto(
                                root_body, factory, ctx);
                            bool did_elim = EliminateGotoToNextLabel(
                                root_body, factory, ctx);
                            bool did_fold_sw = FoldSwitchLocalCaseTargets(
                                root_body, factory, ctx);
                            bool did_dup_term =
                                DuplicateSmallTerminatingTargets(
                                    root_body, factory);
                            bool did_split_clone =
                                SplitAndCloneCrossScopeEntries(
                                    root_body, factory, ctx);
                            bool did_dup_epi = DuplicateSmallEpilogueTargets(
                                root_body, factory, ctx);
                            bool did_dup_swf =
                                DuplicateSwitchFallthroughTargets(
                                    root_body, factory, ctx);
                            bool did_dup_lc = DuplicateLoopContinueTargets(
                                root_body, factory, ctx);
                            bool did_fold_gf = FoldGuardedFallthroughTargets(
                                root_body, factory, ctx);
                            bool did_repair = RepairCrossScopeLabelEntries(
                                root_body, factory, ctx);
                            bool did_sibling = FoldSiblingArmLabelEntries(
                                root_body, factory, ctx);
                            bool did_dup_sg =
                                DuplicateStackGuardReturnTargets(
                                    root_body, factory, ctx);
                            bool did_dup_cl = DuplicateCleanupReturnTargets(
                                root_body, factory, ctx);
                            bool did_inline = InlineResidualGotos(
                                root_body, factory);
                            bool did_cross = InlineCrossScopeSingleRef(
                                root_body, factory);
                            if (!did_absorb && !did_scope && !did_xscope
                                && !did_elim && !did_fold_sw && !did_dup_term
                                && !did_split_clone && !did_dup_epi
                                && !did_dup_swf && !did_dup_lc && !did_fold_gf
                                && !did_repair && !did_sibling
                                && !did_dup_sg && !did_dup_cl && !did_inline
                                && !did_cross) {
                                snode_cleanup_reached_fixed_point = true;
                                break;
                            }
                        }
                        if (!snode_cleanup_reached_fixed_point) {
                            // Surfaces pass oscillation; convergence is <=3 iters on corpus.
                            LOG(WARNING)
                                << "goto-elimination cleanup hit cap of "
                                << kMaxGotoEliminationPasses
                                << " passes for " << fn_name
                                << " — possible pass oscillation\n";
                        }

                        RemoveDeadSSeqChildren(root_body);

                        // Final bounded fixed point so any late-exposed region
                        // entry/exit goto patterns still get repaired.
                        FinalizeRegionRepairs(root_body, factory, ctx);

                        // Gate: a dangling goto is a recoverable Diag Error +
                        // flat fallback below, not an abort.  All output modes,
                        // so --print-tu is guarded too.
                        auto verify =
                            VerifyRegionRepairedBeforeLowering(root_body);
                        if (verify.hasFatalErrors()) {
                            unrepaired_region_flow = true;
                            auto diag_id = ci.getDiagnostics().getCustomDiagID(
                                clang::DiagnosticsEngine::Error,
                                "invalid structured-region control flow "
                                "before lowering in function %0: %1");
                            constexpr size_t kMaxReportedVerifierErrors = 5;
                            for (size_t i = 0;
                                 i < verify.diagnostics.size()
                                 && i < kMaxReportedVerifierErrors;
                                 ++i)
                            {
                                ci.getDiagnostics().Report(diag_id)
                                    << fn_name << verify.diagnostics[i];
                            }
                            if (verify.diagnostics.size()
                                > kMaxReportedVerifierErrors)
                            {
                                ci.getDiagnostics().Report(diag_id)
                                    << fn_name
                                    << (std::to_string(
                                            verify.diagnostics.size()
                                            - kMaxReportedVerifierErrors)
                                        + " additional unrepaired "
                                          "region-control-flow issues");
                            }
                        }
                    }
                }

                if (unrepaired_region_flow) {
                    // Salvage the function as goto-based flat C instead of
                    // dropping it.  The Diag Error above already made it loud.
                    LOG(ERROR)
                        << "Falling back to flat goto-based AST for " << fn_name
                        << " (dangling goto, issue #265); see prior diagnostics.\n";

                    std::vector< SNode * > flat_body;
                    EmitFlatCFG(flat_snapshot, factory, ctx, flat_body);

                    // Flat CFG should never dangle; guard anyway.
                    auto flat_verify =
                        VerifyRegionRepairedBeforeLowering(flat_body);
                    if (flat_verify.hasFatalErrors()) {
                        LOG(ERROR)
                            << "Flat fallback for " << fn_name
                            << " still has unresolved gotos; skipping "
                               "Clang-AST emission.\n";
                    } else {
                        EmitClangAST(flat_body, fn, ctx);
                        if (options.clang_ast_cleanup) {
                            CleanupPrettyPrint(fn, ctx);
                        }
                    }
                } else {
                    EmitClangAST(root_body, fn, ctx);

                    if (options.clang_ast_cleanup) {
                        CleanupPrettyPrint(fn, ctx);
                    }
                }
            }
        }

        if (options.print_tu) {
            // Pretty-print before codegen so the C file emits even when
            // CIR lowering later fails.
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
        const auto &fns = get_program().serialized_functions;
        for (auto &[key, variable] : serialized_variables) {
            if (variable.name.empty() || variable.type.empty()) {
                continue;
            }

            // Stale JSON without the #226 producer fix would trip CIRGen's
            // FuncOp assertion deep in vendor code — refuse loudly here.
            if (fns.find(key) != fns.end()) {
                LOG_FATAL("create_globals: global '{0}' at {1} collides with a "
                          "function entry of the same address — pcode.json is "
                          "malformed.  Regenerate it with a PcodeSerializer "
                          "that includes the #226 fix.",
                          variable.name, key);
            }

            auto var_type       = type_builder->GetSerializedType(variable.type);
            if (var_type.isNull()) {
                continue;
            }
            auto location       = SourceLocation(ctx.getSourceManager(), key);
            auto sanitized_name = SanitizeKeyToIdent(variable.name);
            auto *var_decl      = clang::VarDecl::Create(
                ctx, ctx.getTranslationUnitDecl(), location, location,
                &ctx.Idents.get(sanitized_name), var_type,
                ctx.getTrivialTypeSourceInfo(var_type), clang::SC_Extern
            );
            var_decl->setIsUsed();
            var_decl->setDeclContext(ctx.getTranslationUnitDecl());
            ctx.getTranslationUnitDecl()->addDecl(var_decl);
            global_variable_declarations.emplace(variable.key, var_decl);
        }
    }

} // namespace patchestry::ast
