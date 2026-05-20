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
#include <clang/Basic/ExceptionSpecificationType.h>
#include <clang/Basic/LLVM.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/Specifiers.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/AST/CfgDotEmitter.hpp>
#include <patchestry/AST/ClangEmitter.hpp>
#include <patchestry/AST/CFGStructure.hpp>
#include <patchestry/AST/FunctionBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {

    namespace {

        // Emit a bounded list of report diagnostics — caps the number of
        // lines printed to avoid flooding the log on pathological inputs.
        void LogReportDiagnostics(
            const std::vector< std::string > &diagnostics, std::string_view kind
        ) {
            constexpr size_t kMaxReportDiagnostics = 20;
            for (size_t i = 0;
                 i < std::min(diagnostics.size(), kMaxReportDiagnostics); ++i)
            {
                LOG(ERROR) << "  " << diagnostics[i] << "\n";
            }
            if (diagnostics.size() > kMaxReportDiagnostics) {
                LOG(ERROR) << "  ... "
                           << (diagnostics.size() - kMaxReportDiagnostics)
                           << " more " << kind << " diagnostic(s)\n";
            }
        }

        // Log a summary line + diagnostics for a failed SNode-tree
        // structural verification.
        void LogSNodeVerificationFailure(
            std::string_view verifier_name, std::string_view fn_name,
            const SNodeValidationReport &report
        ) {
            LOG(ERROR) << verifier_name << " failed for " << fn_name
                       << " input_blocks=" << report.input_blocks
                       << " emitted_labels=" << report.emitted_labels
                       << " missing_labels=" << report.missing_labels.size()
                       << " extra_labels=" << report.extra_labels.size()
                       << " input_switches=" << report.input_switches
                       << " emitted_switches=" << report.emitted_switches
                       << " missing_switches=" << report.missing_switches.size()
                       << " extra_switches=" << report.extra_switches.size()
                       << " input_gotos=" << report.input_gotos
                       << " emitted_gotos=" << report.emitted_gotos
                       << " dangling_gotos=" << report.dangling_gotos.size()
                       << " diagnostics=" << report.diagnostics.size() << "\n";
            LogReportDiagnostics(report.diagnostics, "SNode");
        }

        // Count SGoto nodes anywhere in an SNode forest.  Used to report the
        // goto-funnel stages (post-collapse vs post-cleanup) in
        // STRUCTURING_IMPROVEMENT_REPORT.
        size_t CountSNodeGotos(SNode *node) {
            if (!node) { return 0; }
            size_t count = (node->Kind() == SNodeKind::kGoto) ? 1u : 0u;
            node->for_each_child([&](SNode *child) {
                count += CountSNodeGotos(child);
            });
            return count;
        }

        size_t CountSNodeGotos(const std::vector< SNode * > &body) {
            size_t count = 0;
            for (SNode *node : body) { count += CountSNodeGotos(node); }
            return count;
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

                SNodeFactory factory;
                // The function body is a sequence of SNodes.
                std::vector<SNode *> root_body;
                bool have_structured = false;
                // Goto-funnel instrumentation (gated by
                // --structuring-improvement-report; default off so a normal
                // structuring run is byte-identical).  pre_cleanup_gotos is
                // the SGoto count straight out of CFGStructure collapse,
                // before any SNode-level cleanup; post_cleanup_gotos is the
                // count after the cleanup schedule and before EmitClangAST.
                size_t pre_cleanup_gotos  = 0;
                size_t post_cleanup_gotos = 0;

                if (options.use_structuring_pass) {
                    // Structured path: run CFGStructure to fold the
                    // CGraph into hierarchical SNodes.
                    CFGStructure cfg_structure(flow_graph, factory, ctx);
                    cfg_structure.StructureAll();

                    // Build the root sequence from the remaining active
                    // (uncollapsed) nodes.  After StructureAll, each
                    // active node carries a ->structured sequence.
                    for (auto &node : flow_graph.nodes) {
                        if (node.IsCollapsed()) continue;
                        for (SNode *s : node.structured)
                            root_body.push_back(s);
                    }

                    // An empty structured result falls through to the
                    // goto-based path below.
                    if (!root_body.empty()) {
                        have_structured = true;

                        // Normalization: lift raw clang control flow
                        // (goto/label/if/switch/compound) embedded in opaque
                        // SStmt leaves into first-class SNodes so the
                        // SNode-layer cleanup passes below can act on them.
                        NormalizeRawControlFlow(root_body, factory, ctx);

                        // Funnel stage 1: SGoto count of the SNode tree
                        // after raw-control-flow normalization and before
                        // any SNode-level cleanup.
                        if (options.structuring_improvement_report) {
                            pre_cleanup_gotos = CountSNodeGotos(root_body);
                        }

                        // Post-pass: replace goto→break/continue for loop labels.
                        ConvertGotoToBreakContinue(root_body, factory);

                        // Post-pass: replace goto→return patterns.
                        ConvertGotoToReturn(root_body, factory, ctx);

                        // Post-pass: duplicate small label targets into
                        // switch case arms that end in `goto L`, making
                        // switches goto-free (including goto-into-switch).
                        DuplicateSwitchCaseTargets(root_body, factory);

                        // Cross-region goto resolution: each pass below
                        // performs one shape of Move/Clone/Hoist on
                        // cross-region (sibling) gotos.  The ordering is
                        // load-bearing — see each pass's hpp doc for the
                        // pattern it matches.
                        FoldSwitchLocalCaseTargets(root_body, factory, ctx);
                        DuplicateSmallTerminatingTargets(root_body, factory);
                        DuplicateSmallEpilogueTargets(root_body, factory, ctx);
                        DuplicateSwitchFallthroughTargets(root_body, factory, ctx);
                        DuplicateLoopContinueTargets(root_body, factory, ctx);
                        FoldGuardedFallthroughTargets(root_body, factory, ctx);
                        RepairCrossScopeLabelEntries(root_body, factory, ctx);
                        DuplicateStackGuardReturnTargets(root_body, factory, ctx);
                        DuplicateCleanupReturnTargets(root_body, factory, ctx);

                        // Structural cleanup: run before the regular goto
                        // cleanup so alias chains and empty shells are gone
                        // before InlineResidualGotos / EliminateGotoToNext-
                        // Label observe the tree.
                        //  - SimplifyEmptyControlFlow folds empty label
                        //    wrappers and drops empty if/else shells.
                        //  - CollapsePassThroughLabels retargets gotos
                        //    through `goto other` pass-through labels.
                        //  - MergeRedundantGotoGuards merges adjacent /
                        //    else-if goto-guards onto the same label.
                        SimplifyEmptyControlFlow(root_body, ctx);
                        CollapsePassThroughLabels(root_body, factory);
                        MergeRedundantGotoGuards(root_body, factory, ctx);

                        // Post-pass: inline residual goto-to-label pairs
                        // where the label is only referenced once.
                        InlineResidualGotos(root_body, factory);

                        // Post-pass: cross-scope single-ref goto inliner.
                        // Moves terminating label bodies into their sole
                        // goto site when no fallthrough reaches the label.
                        InlineCrossScopeSingleRef(root_body, factory);

                        // Post-pass: eliminate gotos to immediately
                        // following labels.  Iterates with the inliners
                        // and the cross-region resolvers for cascading
                        // cleanup, bounded by kMaxGotoEliminationPasses.
                        // Re-running the cross-region resolvers here is
                        // load-bearing: cloning one target can expose a
                        // fresh cross-region goto for another.
                        bool reached_fixed_point = false;
                        for (int pass = 0;
                             pass < kMaxGotoEliminationPasses; ++pass) {
                            bool did_absorb = AbsorbFallthroughIntoElse(
                                root_body, factory);
                            bool did_scope = ScopeifyIfGotos(
                                root_body, factory, ctx);
                            bool did_elim = EliminateGotoToNextLabel(
                                root_body, factory, ctx);
                            bool did_fold_sw = FoldSwitchLocalCaseTargets(
                                root_body, factory, ctx);
                            bool did_dup_term =
                                DuplicateSmallTerminatingTargets(
                                    root_body, factory);
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
                            bool did_dup_sg =
                                DuplicateStackGuardReturnTargets(
                                    root_body, factory, ctx);
                            bool did_dup_cl = DuplicateCleanupReturnTargets(
                                root_body, factory, ctx);
                            bool did_inline = InlineResidualGotos(
                                root_body, factory);
                            bool did_cross = InlineCrossScopeSingleRef(
                                root_body, factory);
                            if (!did_absorb && !did_scope && !did_elim
                                && !did_fold_sw && !did_dup_term
                                && !did_dup_epi && !did_dup_swf && !did_dup_lc
                                && !did_fold_gf && !did_repair && !did_dup_sg
                                && !did_dup_cl && !did_inline && !did_cross) {
                                reached_fixed_point = true;
                                break;
                            }
                        }
                        if (!reached_fixed_point) {
                            // Cap hit without convergence — surface so a
                            // regression introducing oscillation between
                            // these 14 passes is visible.  Empirically the
                            // loop converges in <=3 iterations on the
                            // current corpus.
                            LOG(WARNING)
                                << "goto-elimination cleanup hit cap of "
                                << kMaxGotoEliminationPasses
                                << " passes for " << fn_name
                                << " — possible pass oscillation\n";
                        }

                        // Post-pass: remove unreachable children after
                        // terminating siblings (dead code from block
                        // sequencing).
                        RemoveDeadSSeqChildren(root_body);

                        // Funnel stage 2: SGoto count after the SNode-level
                        // cleanup schedule, before EmitClangAST.
                        if (options.structuring_improvement_report) {
                            post_cleanup_gotos = CountSNodeGotos(root_body);
                        }

                        // Structural verification of the structured SNode
                        // tree after cleanup and before Clang AST emission.
                        // This catches SNode-level damage (dangling gotos,
                        // duplicate labels, ...) introduced after the
                        // JSON->CGraph verifier has already passed.  Gated
                        // behind --verify-no-node-loss so a normal
                        // structuring run is not perturbed.
                        if (options.verify_no_node_loss) {
                            auto snode_report =
                                ValidateSNodeTree(root_body, &flow_graph);
                            if (!snode_report.ok()) {
                                LogSNodeVerificationFailure(
                                    "Structured SNode verification", fn_name,
                                    snode_report);
                                LOG(FATAL)
                                    << "Structured SNode verification failed "
                                       "for "
                                    << fn_name << "\n";
                            }
                        }
                    }
                }

                if (!have_structured) {
                    root_body.clear();
                    // Goto-based path.
                    // Emit the CGraph blocks sequentially with goto-based
                    // control flow from terminals.
                    // Switch blocks get an SSwitch with goto-to-label cases.
                    for (auto &node : flow_graph.nodes) {
                        if (node.IsCollapsed()) continue;
                        // Spill the node's stmts as SStmt siblings.
                        std::vector<SNode *> blk_nodes;
                        for (auto *s : node.stmts)
                            if (s) blk_nodes.push_back(factory.Make<SStmt>(s));

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
                                        auto &tn = flow_graph.Node(
                                            node.succs[sc.succ_index]);
                                        if (tn.original_label.empty()) {
                                            LOG(FATAL) << "switch default target node "
                                                       << node.succs[sc.succ_index]
                                                       << " has no original_label — "
                                                          "CGraph builder bug.\n";
                                        }
                                        sw->SetDefaultBody(factory.Make<SGoto>(
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
                                                    static_cast<uint64_t>(sc.value),
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
                                        body = factory.Make<SGoto>(
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
                            // Pre-switch stmts (if any) precede the switch.
                            std::vector<SNode *> sw_seq = std::move(blk_nodes);
                            sw_seq.push_back(sw);
                            if (!node.label.empty()) {
                                root_body.push_back(factory.Make<SLabel>(
                                    factory.Intern(node.label),
                                    std::move(sw_seq)));
                            } else {
                                for (SNode *s : sw_seq)
                                    root_body.push_back(s);
                            }
                            continue;
                        }

                        // Non-switch: append terminal (goto/if-goto)
                        if (node.terminal)
                            blk_nodes.push_back(
                                factory.Make<SStmt>(node.terminal));
                        if (!node.label.empty()) {
                            root_body.push_back(factory.Make<SLabel>(
                                factory.Intern(node.label),
                                std::move(blk_nodes)));
                        } else {
                            for (SNode *s : blk_nodes)
                                root_body.push_back(s);
                        }
                    }
                }

                // When --verify-no-node-loss is set, also structurally
                // verify the goto-baseline SNode tree (the unstructured
                // path produces root_body directly).  A failure here
                // means the baseline emission — not the structuring
                // engine — is the source of the defect.
                if (options.verify_no_node_loss && !have_structured) {
                    auto baseline_report =
                        ValidateSNodeTree(root_body, &flow_graph);
                    if (!baseline_report.ok()) {
                        LogSNodeVerificationFailure(
                            "Goto baseline SNode verification", fn_name,
                            baseline_report);
                        LOG(FATAL)
                            << "Goto baseline SNode verification failed for "
                            << fn_name << "\n";
                    }
                }

                EmitClangAST(root_body, fn, ctx);

                // Clang-AST post-emission cleanup pipeline.
                // Gated by --clang-ast-cleanup (default on).
                // Toggle to bisect any structuring drift introduced by
                // these post-passes vs the SNode-layer cleanup alone.
                if (options.clang_ast_cleanup) {
                    CleanupPrettyPrint(fn, ctx,
                                       options.structuring_improvement_report,
                                       fn_name);
                }

                // Goto-funnel report: per-function goto-elimination metrics
                // for the structured path.  Gated by
                // --structuring-improvement-report (default off) so default
                // runs and the lit suite are unperturbed.  pre/post cleanup
                // gotos are the SNode funnel stages; the residual counts are
                // read back from a structural scan of the final SNode tree.
                if (options.structuring_improvement_report && have_structured) {
                    auto residual = ValidateSNodeTree(root_body);
                    size_t eliminated = pre_cleanup_gotos >= post_cleanup_gotos
                        ? pre_cleanup_gotos - post_cleanup_gotos
                        : 0;
                    llvm::errs()
                        << "STRUCTURING_IMPROVEMENT_REPORT function=" << fn_name
                        << " pre_cleanup_gotos=" << pre_cleanup_gotos
                        << " post_cleanup_gotos=" << post_cleanup_gotos
                        << " eliminated_gotos=" << eliminated
                        << " residual_gotos=" << residual.emitted_gotos
                        << " emitted_labels=" << residual.emitted_labels
                        << " dangling_gotos=" << residual.dangling_gotos.size()
                        << " duplicate_labels="
                        << residual.duplicate_labels.size() << "\n";
                }
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
