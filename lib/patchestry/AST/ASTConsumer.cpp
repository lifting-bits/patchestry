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

        struct ClangControlFlowStats
        {
            size_t goto_stmts  = 0;
            size_t label_stmts = 0;
        };

        struct SNodeControlFlowStats
        {
            size_t snode_gotos  = 0;
            size_t snode_labels = 0;
            size_t clang_gotos  = 0;
            size_t clang_labels = 0;
        };

        struct StructuringStatsSummary
        {
            size_t functions                    = 0;
            size_t flat_baseline_functions      = 0;
            size_t region_present_functions     = 0;
            size_t region_seeded_functions      = 0;
            size_t flat_fallback_functions      = 0;
            size_t region_fallback_functions    = 0;
            size_t no_region_fallback_functions = 0;
            size_t snode_cleanup_cap_hits       = 0;
            size_t snode_cleanup_iterations     = 0;
            size_t final_region_repair_changes  = 0;
            size_t pre_lowering_verifier_errors = 0;
        };

        struct RegionShapeStats
        {
            size_t total_nodes          = 0;
            size_t max_depth            = 0;
            size_t graph_nodes          = 0;
            size_t list_nodes           = 0;
            size_t plain_nodes          = 0;
            size_t condition_nodes      = 0;
            size_t properif_nodes       = 0;
            size_t ifelse_nodes         = 0;
            size_t ifgoto_nodes         = 0;
            size_t whiledo_nodes        = 0;
            size_t dowhile_nodes        = 0;
            size_t infloop_nodes        = 0;
            size_t switch_nodes         = 0;
            size_t goto_nodes           = 0;
            size_t multigoto_nodes      = 0;
            size_t unknown_nodes        = 0;
            size_t explicit_goto_targets = 0;
            size_t unique_plain_blocks  = 0;
            size_t duplicate_plain_refs = 0;
        };

        struct SNodeGotoPatternStats
        {
            size_t label_defs          = 0;
            size_t goto_refs           = 0;
            size_t unresolved_gotos    = 0;
            size_t single_ref_gotos    = 0;
            size_t multi_ref_gotos     = 0;
            size_t goto_to_next_label  = 0;
        };

        struct ClangGotoPatternStats
        {
            size_t label_defs          = 0;
            size_t goto_refs           = 0;
            size_t unresolved_gotos    = 0;
            size_t single_ref_gotos    = 0;
            size_t multi_ref_gotos     = 0;
            size_t goto_to_next_label  = 0;
        };

        void CountClangControlFlow(const clang::Stmt *stmt, ClangControlFlowStats &stats) {
            if (stmt == nullptr) { return; }
            if (llvm::isa< clang::GotoStmt >(stmt)) { ++stats.goto_stmts; }
            if (llvm::isa< clang::LabelStmt >(stmt)) { ++stats.label_stmts; }
            for (const clang::Stmt *child : stmt->children()) {
                CountClangControlFlow(child, stats);
            }
        }

        void CountSNodeControlFlow(const SNode *node, SNodeControlFlowStats &stats) {
            if (node == nullptr) { return; }
            if (node->dyn_cast< SGoto >() != nullptr) { ++stats.snode_gotos; }
            if (node->dyn_cast< SLabel >() != nullptr) { ++stats.snode_labels; }
            if (auto *stmt = node->dyn_cast< SStmt >()) {
                ClangControlFlowStats clang_stats;
                CountClangControlFlow(stmt->Stmt(), clang_stats);
                stats.clang_gotos  += clang_stats.goto_stmts;
                stats.clang_labels += clang_stats.label_stmts;
            }
            node->for_each_child([&](SNode *child) { CountSNodeControlFlow(child, stats); });
        }

        SNodeControlFlowStats CountSNodeControlFlow(const std::vector< SNode * > &root) {
            SNodeControlFlowStats stats;
            for (const SNode *node : root) { CountSNodeControlFlow(node, stats); }
            return stats;
        }

        ClangControlFlowStats CountClangControlFlow(const clang::Stmt *body) {
            ClangControlFlowStats stats;
            CountClangControlFlow(body, stats);
            return stats;
        }

        void AccumulateRegionShapeStats(
            const ghidra::RegionNode &node, size_t depth, RegionShapeStats &stats,
            std::unordered_map< std::string, size_t > &plain_block_refs
        ) {
            ++stats.total_nodes;
            stats.max_depth = std::max(stats.max_depth, depth);

            if (node.kind == "graph") {
                ++stats.graph_nodes;
            } else if (node.kind == "list") {
                ++stats.list_nodes;
            } else if (node.kind == "plain") {
                ++stats.plain_nodes;
                if (node.block.has_value()) { ++plain_block_refs[*node.block]; }
            } else if (node.kind == "condition") {
                ++stats.condition_nodes;
            } else if (node.kind == "properif") {
                ++stats.properif_nodes;
            } else if (node.kind == "ifelse") {
                ++stats.ifelse_nodes;
            } else if (node.kind == "ifgoto") {
                ++stats.ifgoto_nodes;
            } else if (node.kind == "whiledo") {
                ++stats.whiledo_nodes;
            } else if (node.kind == "dowhile") {
                ++stats.dowhile_nodes;
            } else if (node.kind == "infloop") {
                ++stats.infloop_nodes;
            } else if (node.kind == "switch") {
                ++stats.switch_nodes;
            } else if (node.kind == "goto") {
                ++stats.goto_nodes;
            } else if (node.kind == "multigoto") {
                ++stats.multigoto_nodes;
            } else {
                ++stats.unknown_nodes;
            }

            stats.explicit_goto_targets += node.goto_targets.size();
            for (const auto &child : node.children) {
                AccumulateRegionShapeStats(child, depth + 1, stats, plain_block_refs);
            }
        }

        RegionShapeStats CollectRegionShapeStats(const ghidra::RegionNode &root) {
            RegionShapeStats stats;
            std::unordered_map< std::string, size_t > plain_block_refs;
            AccumulateRegionShapeStats(root, 0, stats, plain_block_refs);
            stats.unique_plain_blocks = plain_block_refs.size();
            for (const auto &[block, count] : plain_block_refs) {
                (void) block;
                if (count > 1) { stats.duplicate_plain_refs += count - 1; }
            }
            return stats;
        }

        void CollectSNodeGotoLabels(
            const SNode *node, std::unordered_map< std::string, size_t > &goto_refs,
            std::unordered_map< std::string, size_t > &label_defs
        ) {
            if (node == nullptr) { return; }
            if (auto *go = node->dyn_cast< SGoto >()) {
                ++goto_refs[std::string(go->Target())];
            }
            if (auto *label = node->dyn_cast< SLabel >()) {
                ++label_defs[std::string(label->Name())];
            }
            node->for_each_child([&](SNode *child) {
                CollectSNodeGotoLabels(child, goto_refs, label_defs);
            });
        }

        void CollectSNodeGotoLabels(
            const std::vector< SNode * > &seq,
            std::unordered_map< std::string, size_t > &goto_refs,
            std::unordered_map< std::string, size_t > &label_defs
        ) {
            for (const SNode *node : seq) {
                CollectSNodeGotoLabels(node, goto_refs, label_defs);
            }
        }

        bool IsGotoToLabel(const SNode *go_node, const SNode *label_node) {
            auto *go = go_node ? go_node->dyn_cast< SGoto >() : nullptr;
            auto *label = label_node ? label_node->dyn_cast< SLabel >() : nullptr;
            return go != nullptr && label != nullptr && go->Target() == label->Name();
        }

        void AnalyzeSNodeGotoPatternSeq(
            const std::vector< SNode * > &seq,
            const std::unordered_map< std::string, size_t > &goto_refs,
            const std::unordered_map< std::string, size_t > &label_defs,
            SNodeGotoPatternStats &stats
        );

        void AnalyzeSNodeGotoPatternNode(
            const SNode *node,
            const std::unordered_map< std::string, size_t > &goto_refs,
            const std::unordered_map< std::string, size_t > &label_defs,
            SNodeGotoPatternStats &stats
        ) {
            if (node == nullptr) { return; }
            if (auto *go = node->dyn_cast< SGoto >()) {
                ++stats.goto_refs;
                auto target = std::string(go->Target());
                if (!label_defs.contains(target)) {
                    ++stats.unresolved_gotos;
                } else if (goto_refs.at(target) == 1) {
                    ++stats.single_ref_gotos;
                } else {
                    ++stats.multi_ref_gotos;
                }
                return;
            }
            if (auto *label = node->dyn_cast< SLabel >()) {
                ++stats.label_defs;
                AnalyzeSNodeGotoPatternSeq(label->BodyList(), goto_refs, label_defs, stats);
                return;
            }
            if (auto *if_stmt = node->dyn_cast< SIfThenElse >()) {
                AnalyzeSNodeGotoPatternSeq(if_stmt->ThenList(), goto_refs, label_defs, stats);
                AnalyzeSNodeGotoPatternSeq(if_stmt->ElseList(), goto_refs, label_defs, stats);
                return;
            }
            if (auto *while_stmt = node->dyn_cast< SWhile >()) {
                AnalyzeSNodeGotoPatternSeq(while_stmt->BodyList(), goto_refs, label_defs, stats);
                return;
            }
            if (auto *do_stmt = node->dyn_cast< SDoWhile >()) {
                AnalyzeSNodeGotoPatternSeq(do_stmt->BodyList(), goto_refs, label_defs, stats);
                return;
            }
            if (auto *for_stmt = node->dyn_cast< SFor >()) {
                AnalyzeSNodeGotoPatternSeq(for_stmt->BodyList(), goto_refs, label_defs, stats);
                return;
            }
            if (auto *sw = node->dyn_cast< SSwitch >()) {
                for (const auto &case_body : sw->Cases()) {
                    AnalyzeSNodeGotoPatternSeq(
                        case_body.body_list, goto_refs, label_defs, stats
                    );
                }
                AnalyzeSNodeGotoPatternSeq(
                    sw->DefaultBodyList(), goto_refs, label_defs, stats
                );
            }
        }

        void AnalyzeSNodeGotoPatternSeq(
            const std::vector< SNode * > &seq,
            const std::unordered_map< std::string, size_t > &goto_refs,
            const std::unordered_map< std::string, size_t > &label_defs,
            SNodeGotoPatternStats &stats
        ) {
            for (size_t i = 0; i < seq.size(); ++i) {
                if (i + 1 < seq.size() && IsGotoToLabel(seq[i], seq[i + 1])) {
                    ++stats.goto_to_next_label;
                }
                AnalyzeSNodeGotoPatternNode(seq[i], goto_refs, label_defs, stats);
            }
        }

        SNodeGotoPatternStats CollectSNodeGotoPatternStats(
            const std::vector< SNode * > &root
        ) {
            std::unordered_map< std::string, size_t > goto_refs;
            std::unordered_map< std::string, size_t > label_defs;
            CollectSNodeGotoLabels(root, goto_refs, label_defs);
            SNodeGotoPatternStats stats;
            AnalyzeSNodeGotoPatternSeq(root, goto_refs, label_defs, stats);
            return stats;
        }

        void CollectClangGotoLabels(
            const clang::Stmt *stmt,
            std::unordered_map< const clang::LabelDecl *, size_t > &goto_refs,
            std::unordered_map< const clang::LabelDecl *, size_t > &label_defs
        ) {
            if (stmt == nullptr) { return; }
            if (auto *go = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                if (go->getLabel() != nullptr) { ++goto_refs[go->getLabel()]; }
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                if (label->getDecl() != nullptr) { ++label_defs[label->getDecl()]; }
            }
            for (const clang::Stmt *child : stmt->children()) {
                CollectClangGotoLabels(child, goto_refs, label_defs);
            }
        }

        bool IsGotoToLabel(const clang::Stmt *go_stmt, const clang::Stmt *label_stmt) {
            auto *go = go_stmt ? llvm::dyn_cast< clang::GotoStmt >(go_stmt) : nullptr;
            auto *label = label_stmt ? llvm::dyn_cast< clang::LabelStmt >(label_stmt) : nullptr;
            return go != nullptr && label != nullptr && go->getLabel() == label->getDecl();
        }

        void AnalyzeClangGotoPatternStmt(
            const clang::Stmt *stmt,
            const std::unordered_map< const clang::LabelDecl *, size_t > &goto_refs,
            const std::unordered_map< const clang::LabelDecl *, size_t > &label_defs,
            ClangGotoPatternStats &stats
        ) {
            if (stmt == nullptr) { return; }
            if (auto *go = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                ++stats.goto_refs;
                const clang::LabelDecl *target = go->getLabel();
                if (target == nullptr || !label_defs.contains(target)) {
                    ++stats.unresolved_gotos;
                } else if (goto_refs.at(target) == 1) {
                    ++stats.single_ref_gotos;
                } else {
                    ++stats.multi_ref_gotos;
                }
                return;
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ++stats.label_defs;
                AnalyzeClangGotoPatternStmt(
                    label->getSubStmt(), goto_refs, label_defs, stats
                );
                return;
            }
            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                auto body = compound->body();
                for (auto it = body.begin(), end = body.end(); it != end; ++it) {
                    auto next = it;
                    ++next;
                    if (next != end && IsGotoToLabel(*it, *next)) {
                        ++stats.goto_to_next_label;
                    }
                    AnalyzeClangGotoPatternStmt(*it, goto_refs, label_defs, stats);
                }
                return;
            }
            for (const clang::Stmt *child : stmt->children()) {
                AnalyzeClangGotoPatternStmt(child, goto_refs, label_defs, stats);
            }
        }

        ClangGotoPatternStats CollectClangGotoPatternStats(const clang::Stmt *body) {
            std::unordered_map< const clang::LabelDecl *, size_t > goto_refs;
            std::unordered_map< const clang::LabelDecl *, size_t > label_defs;
            CollectClangGotoLabels(body, goto_refs, label_defs);
            ClangGotoPatternStats stats;
            AnalyzeClangGotoPatternStmt(body, goto_refs, label_defs, stats);
            return stats;
        }

        void PrintSNodeStructuringStats(
            std::string_view fn_name, std::string_view stage, const SNodeControlFlowStats &stats
        ) {
            llvm::errs() << "[structuring-stats] function=" << fn_name << " stage=" << stage
                         << " snode_gotos=" << stats.snode_gotos
                         << " snode_labels=" << stats.snode_labels
                         << " clang_gotos=" << stats.clang_gotos
                         << " clang_labels=" << stats.clang_labels
                         << " total_gotos=" << (stats.snode_gotos + stats.clang_gotos)
                         << " total_labels=" << (stats.snode_labels + stats.clang_labels)
                         << "\n";
        }

        void PrintClangStructuringStats(
            std::string_view fn_name, std::string_view stage, const ClangControlFlowStats &stats
        ) {
            llvm::errs() << "[structuring-stats] function=" << fn_name << " stage=" << stage
                         << " clang_gotos=" << stats.goto_stmts
                         << " clang_labels=" << stats.label_stmts << "\n";
        }

        void PrintRegionShapeStats(std::string_view fn_name, const RegionShapeStats &stats) {
            llvm::errs() << "[structuring-stats] function=" << fn_name
                         << " stage=ghidra-region"
                         << " region_nodes=" << stats.total_nodes
                         << " max_depth=" << stats.max_depth
                         << " kind_graph=" << stats.graph_nodes
                         << " kind_list=" << stats.list_nodes
                         << " kind_plain=" << stats.plain_nodes
                         << " kind_condition=" << stats.condition_nodes
                         << " kind_properif=" << stats.properif_nodes
                         << " kind_ifelse=" << stats.ifelse_nodes
                         << " kind_ifgoto=" << stats.ifgoto_nodes
                         << " kind_whiledo=" << stats.whiledo_nodes
                         << " kind_dowhile=" << stats.dowhile_nodes
                         << " kind_infloop=" << stats.infloop_nodes
                         << " kind_switch=" << stats.switch_nodes
                         << " kind_goto=" << stats.goto_nodes
                         << " kind_multigoto=" << stats.multigoto_nodes
                         << " kind_unknown=" << stats.unknown_nodes
                         << " unique_plain_blocks=" << stats.unique_plain_blocks
                         << " duplicate_plain_refs=" << stats.duplicate_plain_refs
                         << " explicit_goto_targets=" << stats.explicit_goto_targets
                         << "\n";
        }

        void PrintSNodeGotoPatternStats(
            std::string_view fn_name, std::string_view stage,
            const SNodeGotoPatternStats &stats
        ) {
            llvm::errs() << "[structuring-stats] function=" << fn_name
                         << " stage=" << stage
                         << " labels=" << stats.label_defs
                         << " gotos=" << stats.goto_refs
                         << " unresolved_gotos=" << stats.unresolved_gotos
                         << " single_ref_gotos=" << stats.single_ref_gotos
                         << " multi_ref_gotos=" << stats.multi_ref_gotos
                         << " goto_to_next_label=" << stats.goto_to_next_label
                         << "\n";
        }

        void PrintClangGotoPatternStats(
            std::string_view fn_name, std::string_view stage,
            const ClangGotoPatternStats &stats
        ) {
            llvm::errs() << "[structuring-stats] function=" << fn_name
                         << " stage=" << stage
                         << " labels=" << stats.label_defs
                         << " gotos=" << stats.goto_refs
                         << " unresolved_gotos=" << stats.unresolved_gotos
                         << " single_ref_gotos=" << stats.single_ref_gotos
                         << " multi_ref_gotos=" << stats.multi_ref_gotos
                         << " goto_to_next_label=" << stats.goto_to_next_label
                         << "\n";
        }

        void PrintRegionRepairVerifierStats(
            std::string_view fn_name,
            const RegionRepairVerifierResult &stats
        ) {
            llvm::errs() << "[structuring-stats] function=" << fn_name
                         << " stage=pre-lowering-region-verifier"
                         << " gotos=" << stats.goto_refs
                         << " same_scope_gotos=" << stats.same_scope_gotos
                         << " outward_gotos=" << stats.outward_gotos
                         << " cross_scope_entry_gotos="
                         << stats.cross_scope_entry_gotos
                         << " unresolved_gotos=" << stats.unresolved_gotos
                         << " ok=" << (stats.ok() ? 1 : 0)
                         << "\n";
        }

        void PrintFunctionStructuringSummary(
            std::string_view fn_name, bool flat_baseline, bool had_region,
            bool seeded_from_ghidra, int snode_cleanup_iterations,
            bool snode_cleanup_reached_fixed_point
        ) {
            llvm::errs() << "[structuring-stats] function=" << fn_name
                         << " flat_baseline=" << (flat_baseline ? 1 : 0)
                         << " had_region=" << (had_region ? 1 : 0)
                         << " seeded_from_ghidra=" << (seeded_from_ghidra ? 1 : 0)
                         << " flat_fallback=" << (!flat_baseline && !seeded_from_ghidra ? 1 : 0)
                         << " snode_cleanup_iterations=" << snode_cleanup_iterations
                         << " snode_cleanup_fixed_point="
                         << (snode_cleanup_reached_fixed_point ? 1 : 0) << "\n";
        }

        void PrintStructuringSummary(const StructuringStatsSummary &summary) {
            llvm::errs() << "[structuring-stats] summary"
                         << " functions=" << summary.functions
                         << " flat_baseline_functions=" << summary.flat_baseline_functions
                         << " region_present_functions=" << summary.region_present_functions
                         << " region_seeded_functions=" << summary.region_seeded_functions
                         << " flat_fallback_functions=" << summary.flat_fallback_functions
                         << " region_fallback_functions=" << summary.region_fallback_functions
                         << " no_region_fallback_functions="
                         << summary.no_region_fallback_functions
                         << " snode_cleanup_iterations=" << summary.snode_cleanup_iterations
                         << " snode_cleanup_cap_hits=" << summary.snode_cleanup_cap_hits
                         << " final_region_repair_changes="
                         << summary.final_region_repair_changes
                         << " pre_lowering_verifier_errors="
                         << summary.pre_lowering_verifier_errors
                         << "\n";
        }
    } // namespace

    void PcodeASTConsumer::HandleTranslationUnit(clang::ASTContext &ctx) {
        type_builder = std::make_unique< TypeBuilder >(ci.getASTContext());
        StructuringStatsSummary structuring_stats_summary;
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
                bool had_region                       = func.region.has_value();
                bool seeded_from_ghidra                = false;
                bool snode_cleanup_reached_fixed_point = true;
                int snode_cleanup_iterations           = 0;

                auto *prev_ctx = builder->enter_function_context(fn);
                CGraph flow_graph = BuildCGraph(*builder, ctx);
                builder->leave_function_context(prev_ctx);
                if (flow_graph.nodes.empty()) {
                    LOG(WARNING) << "BuildCGraph produced empty graph for "
                                 << fn_name << "\n";
                    continue;
                }

                if (options.structuring_stats) {
                    ++structuring_stats_summary.functions;
                    if (options.emit_flat_baseline) {
                        ++structuring_stats_summary.flat_baseline_functions;
                    }
                    if (had_region) { ++structuring_stats_summary.region_present_functions; }
                }

                if (options.emit_dot_cfg) {
                    CGraphDotTracer tracer;
                    tracer.fn_name = fn_name;
                    tracer.enabled = true;
                    tracer.Dump(flow_graph, "BuildCGraph", true);
                }

                SNodeFactory factory;
                std::vector<SNode *> root_body;

                if (options.emit_flat_baseline) {
                    EmitFlatCFG(flow_graph, factory, ctx, root_body);
                } else {
                    // Seed from Function.region when present; fall back
                    // to flat-CFG. Post-pass chain runs on either shape.
                    if (func.region.has_value()) {
                        if (options.structuring_stats) {
                            PrintRegionShapeStats(
                                fn_name, CollectRegionShapeStats(*func.region)
                            );
                        }
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
                        if (options.structuring_stats) {
                            PrintSNodeStructuringStats(
                                fn_name, "snode-normalized", CountSNodeControlFlow(root_body)
                            );
                            PrintSNodeGotoPatternStats(
                                fn_name, "snode-normalized-patterns",
                                CollectSNodeGotoPatternStats(root_body)
                            );
                        }

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
                            snode_cleanup_iterations = pass + 1;
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
                        if (options.structuring_stats) {
                            PrintSNodeStructuringStats(
                                fn_name, "snode-final", CountSNodeControlFlow(root_body)
                            );
                            PrintSNodeGotoPatternStats(
                                fn_name, "snode-final-patterns",
                                CollectSNodeGotoPatternStats(root_body)
                            );
                        }

                        // Final bounded fixed point so any late-exposed region
                        // entry/exit goto patterns still get repaired.
                        if (FinalizeRegionRepairs(root_body, factory, ctx)) {
                            ++structuring_stats_summary.final_region_repair_changes;
                            if (options.structuring_stats) {
                                PrintSNodeStructuringStats(
                                    fn_name, "snode-region-repaired",
                                    CountSNodeControlFlow(root_body)
                                );
                                PrintSNodeGotoPatternStats(
                                    fn_name, "snode-region-repaired-patterns",
                                    CollectSNodeGotoPatternStats(root_body)
                                );
                            }
                        }

                        if (options.emit_cir || options.emit_mlir
                            || options.emit_llvm)
                        {
                            auto verify =
                                VerifyRegionRepairedBeforeLowering(root_body);
                            if (options.structuring_stats) {
                                PrintRegionRepairVerifierStats(fn_name, verify);
                            }
                            if (verify.hasFatalErrors()) {
                                ++structuring_stats_summary
                                      .pre_lowering_verifier_errors;
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
                }

                if (options.structuring_stats) {
                    if (!options.emit_flat_baseline) {
                        if (seeded_from_ghidra) {
                            ++structuring_stats_summary.region_seeded_functions;
                        } else {
                            ++structuring_stats_summary.flat_fallback_functions;
                            if (had_region) {
                                ++structuring_stats_summary.region_fallback_functions;
                            } else {
                                ++structuring_stats_summary.no_region_fallback_functions;
                            }
                        }
                    }
                    structuring_stats_summary.snode_cleanup_iterations +=
                        static_cast< size_t >(snode_cleanup_iterations);
                    if (!snode_cleanup_reached_fixed_point) {
                        ++structuring_stats_summary.snode_cleanup_cap_hits;
                    }
                    PrintFunctionStructuringSummary(
                        fn_name, options.emit_flat_baseline, had_region, seeded_from_ghidra,
                        snode_cleanup_iterations, snode_cleanup_reached_fixed_point
                    );
                }

                EmitClangAST(root_body, fn, ctx);
                if (options.structuring_stats) {
                    PrintClangStructuringStats(
                        fn_name, "clang-emitted", CountClangControlFlow(fn->getBody())
                    );
                    PrintClangGotoPatternStats(
                        fn_name, "clang-emitted-patterns",
                        CollectClangGotoPatternStats(fn->getBody())
                    );
                }

                if (options.clang_ast_cleanup) {
                    CleanupPrettyPrint(fn, ctx);
                    if (options.structuring_stats) {
                        PrintClangStructuringStats(
                            fn_name, "clang-cleaned", CountClangControlFlow(fn->getBody())
                        );
                        PrintClangGotoPatternStats(
                            fn_name, "clang-cleaned-patterns",
                            CollectClangGotoPatternStats(fn->getBody())
                        );
                    }
                }
            }
        }

        if (options.structuring_stats) { PrintStructuringSummary(structuring_stats_summary); }

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
