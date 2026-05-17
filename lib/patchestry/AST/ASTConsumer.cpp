/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

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
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/AST/CFGStructure.hpp>
#include <patchestry/AST/CfgDotEmitter.hpp>
#include <patchestry/AST/ClangEmitter.hpp>
#include <patchestry/AST/FunctionBuilder.hpp>
#include <patchestry/AST/SNodeRegion.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {

    namespace {

        std::string_view LabelForEmission(const CNode &node) {
            if (!node.label.empty()) { return node.label; }
            return node.original_label;
        }

        std::vector< SNode * >
        BuildGotoSNodeBody(CGraph &flow_graph, SNodeFactory &factory, clang::ASTContext &ctx) {
            std::vector< SNode * > root_body;
            for (auto &node : flow_graph.nodes) {
                if (node.IsCollapsed()) { continue; }

                std::vector< SNode * > blk_nodes;
                for (auto *stmt : node.stmts) {
                    if (stmt) { blk_nodes.push_back(factory.Make< SStmt >(stmt)); }
                }

                if (!node.switch_cases.empty() && node.branch_cond) {
                    auto *sw       = factory.Make< SSwitch >(node.branch_cond);
                    auto case_type = node.branch_cond->getType();
                    if (case_type->isEnumeralType()) {
                        case_type =
                            case_type->castAs< clang::EnumType >()->getDecl()->getIntegerType();
                    }
                    unsigned case_width = ctx.getIntWidth(case_type);
                    for (const auto &sc : node.switch_cases) {
                        if (sc.succ_index >= node.succs.size()) {
                            LOG(WARNING)
                                << "switch case succ_index " << sc.succ_index
                                << " out of range (succs=" << node.succs.size() << ")\n";
                            continue;
                        }

                        auto &target      = flow_graph.Node(node.succs[sc.succ_index]);
                        auto target_label = LabelForEmission(target);
                        if (target_label.empty()) {
                            LOG(FATAL) << "switch target node " << node.succs[sc.succ_index]
                                       << " has no label — CGraph builder bug.\n";
                        }

                        auto *goto_body = factory.Make< SGoto >(factory.Intern(target_label));
                        if (sc.is_default) {
                            sw->SetDefaultBody(goto_body);
                            continue;
                        }

                        auto *val = clang::IntegerLiteral::Create(
                            ctx,
                            llvm::APInt(case_width, static_cast< uint64_t >(sc.value), true),
                            case_type, VirtualLoc(ctx)
                        );
                        sw->AddCase(val, goto_body);
                    }

                    blk_nodes.push_back(sw);
                } else if (node.terminal) {
                    blk_nodes.push_back(factory.Make< SStmt >(node.terminal));
                }

                auto label = LabelForEmission(node);
                if (!label.empty()) {
                    root_body.push_back(
                        factory.Make< SLabel >(factory.Intern(label), std::move(blk_nodes))
                    );
                } else {
                    for (SNode *s : blk_nodes) { root_body.push_back(s); }
                }
            }
            return root_body;
        }

        struct PayloadRetentionReport
        {
            size_t baseline_payload_stmts = 0;
            size_t retained_payload_stmts = 0;
            std::vector< std::string > missing_payload_stmts;
            std::vector< std::string > duplicated_payload_stmts;
            std::vector< std::string > diagnostics;

            bool ok() const { return diagnostics.empty(); }
        };

        struct SNodeOwnershipReport
        {
            size_t expected_owned_ops = 0;
            size_t retained_owned_ops = 0;
            std::vector< std::string > missing_owned_ops;
            std::vector< std::string > duplicated_owned_ops;
            std::vector< std::string > extra_owned_ops;
            std::vector< std::string > diagnostics;

            bool ok() const { return diagnostics.empty(); }
        };

        std::string DescribeStmt(const clang::Stmt *stmt) {
            if (!stmt) { return "<null-stmt>"; }
            return stmt->getStmtClassName();
        }

        std::string OriginKey(const StmtOrigin &origin) {
            return origin.block_key + "::" + origin.operation_key;
        }

        void AddUniqueOrigin(std::vector< StmtOrigin > &origins, const StmtOrigin &origin) {
            auto same_origin = [&](const StmtOrigin &existing) {
                return existing.block_key == origin.block_key
                    && existing.operation_key == origin.operation_key
                    && existing.kind == origin.kind && existing.primary == origin.primary;
            };
            if (std::find_if(origins.begin(), origins.end(), same_origin) == origins.end()) {
                origins.push_back(origin);
            }
        }

        void CollectRawStmtOrigins(
            const FunctionBuilder &builder, const clang::Stmt *stmt,
            std::vector< StmtOrigin > &origins
        ) {
            std::vector< StmtOrigin > collected;
            builder.CollectStmtOrigins(stmt, collected);
            for (const auto &origin : collected) { AddUniqueOrigin(origins, origin); }
        }

        void AnnotateSNodeOrigins(const FunctionBuilder &builder, SNode *node) {
            if (!node) { return; }

            std::vector< StmtOrigin > origins;
            if (auto *stmt = node->dyn_cast< SStmt >()) {
                CollectRawStmtOrigins(builder, stmt->Stmt(), origins);
            } else if (auto *ite = node->dyn_cast< SIfThenElse >()) {
                CollectRawStmtOrigins(builder, ite->Cond(), origins);
            } else if (auto *while_node = node->dyn_cast< SWhile >()) {
                CollectRawStmtOrigins(builder, while_node->Cond(), origins);
            } else if (auto *do_node = node->dyn_cast< SDoWhile >()) {
                CollectRawStmtOrigins(builder, do_node->Cond(), origins);
            } else if (auto *for_node = node->dyn_cast< SFor >()) {
                CollectRawStmtOrigins(builder, for_node->Init(), origins);
                CollectRawStmtOrigins(builder, for_node->Cond(), origins);
                CollectRawStmtOrigins(builder, for_node->Inc(), origins);
            } else if (auto *switch_node = node->dyn_cast< SSwitch >()) {
                CollectRawStmtOrigins(builder, switch_node->Discriminant(), origins);
            } else if (auto *return_node = node->dyn_cast< SReturn >()) {
                CollectRawStmtOrigins(builder, return_node->Value(), origins);
            }
            node->SetOrigins(std::move(origins));

            node->for_each_child([&](SNode *child) { AnnotateSNodeOrigins(builder, child); });
        }

        void AnnotateSNodeOrigins(
            const FunctionBuilder &builder, const std::vector< SNode * > &root
        ) {
            for (SNode *node : root) { AnnotateSNodeOrigins(builder, node); }
        }

        SNode *CloneSNodeForTransaction(const SNode *node, SNodeFactory &factory);

        std::vector< SNode * >
        CloneSNodeSeqForTransaction(const std::vector< SNode * > &seq, SNodeFactory &factory) {
            std::vector< SNode * > out;
            out.reserve(seq.size());
            for (const SNode *node : seq) {
                if (auto *clone = CloneSNodeForTransaction(node, factory)) {
                    out.push_back(clone);
                }
            }
            return out;
        }

        SNode *CopySNodeTransactionOrigins(SNode *clone, const SNode *src) {
            if (clone && src) { clone->SetOrigins(src->Origins()); }
            return clone;
        }

        SNode *CloneSNodeForTransaction(const SNode *node, SNodeFactory &factory) {
            if (!node) { return nullptr; }

            if (const auto *stmt = node->dyn_cast< SStmt >()) {
                return CopySNodeTransactionOrigins(factory.Make< SStmt >(stmt->Stmt()), node);
            }
            if (const auto *ite = node->dyn_cast< SIfThenElse >()) {
                return CopySNodeTransactionOrigins(
                    factory.Make< SIfThenElse >(
                        ite->Cond(), CloneSNodeSeqForTransaction(ite->ThenList(), factory),
                        CloneSNodeSeqForTransaction(ite->ElseList(), factory)
                    ),
                    node
                );
            }
            if (const auto *while_node = node->dyn_cast< SWhile >()) {
                auto *clone = factory.Make< SWhile >(
                    while_node->Cond(),
                    CloneSNodeSeqForTransaction(while_node->BodyList(), factory)
                );
                clone->SetHeaderLabel(while_node->HeaderLabel());
                clone->SetExitLabel(while_node->ExitLabel());
                return CopySNodeTransactionOrigins(clone, node);
            }
            if (const auto *do_node = node->dyn_cast< SDoWhile >()) {
                auto *clone = factory.Make< SDoWhile >(
                    CloneSNodeSeqForTransaction(do_node->BodyList(), factory), do_node->Cond()
                );
                clone->SetHeaderLabel(do_node->HeaderLabel());
                clone->SetExitLabel(do_node->ExitLabel());
                return CopySNodeTransactionOrigins(clone, node);
            }
            if (const auto *for_node = node->dyn_cast< SFor >()) {
                auto *clone = factory.Make< SFor >(
                    for_node->Init(), for_node->Cond(), for_node->Inc(),
                    CloneSNodeSeqForTransaction(for_node->BodyList(), factory)
                );
                clone->SetHeaderLabel(for_node->HeaderLabel());
                clone->SetExitLabel(for_node->ExitLabel());
                return CopySNodeTransactionOrigins(clone, node);
            }
            if (const auto *switch_node = node->dyn_cast< SSwitch >()) {
                auto *clone = factory.Make< SSwitch >(switch_node->Discriminant());
                for (const auto &case_node : switch_node->Cases()) {
                    clone->AddCase(
                        case_node.value,
                        CloneSNodeSeqForTransaction(case_node.body_list, factory)
                    );
                }
                clone->SetDefaultBody(
                    CloneSNodeSeqForTransaction(switch_node->DefaultBodyList(), factory)
                );
                return CopySNodeTransactionOrigins(clone, node);
            }
            if (const auto *go = node->dyn_cast< SGoto >()) {
                return CopySNodeTransactionOrigins(
                    factory.Make< SGoto >(factory.Intern(go->Target())), node
                );
            }
            if (const auto *label = node->dyn_cast< SLabel >()) {
                return CopySNodeTransactionOrigins(
                    factory.Make< SLabel >(
                        factory.Intern(label->Name()),
                        CloneSNodeSeqForTransaction(label->BodyList(), factory)
                    ),
                    node
                );
            }
            if (const auto *br = node->dyn_cast< SBreak >()) {
                return CopySNodeTransactionOrigins(factory.Make< SBreak >(br->Depth()), node);
            }
            if (node->dyn_cast< SContinue >()) {
                return CopySNodeTransactionOrigins(factory.Make< SContinue >(), node);
            }
            if (const auto *ret = node->dyn_cast< SReturn >()) {
                return CopySNodeTransactionOrigins(factory.Make< SReturn >(ret->Value()), node);
            }

            return nullptr;
        }

        void CountPayloadOrigins(
            const std::vector< StmtOrigin > &origins,
            std::unordered_map< std::string, unsigned > &counts
        ) {
            for (const auto &origin : origins) {
                if (!origin.primary || !IsPayloadCarrierKind(origin.kind)) { continue; }
                ++counts[OriginKey(origin)];
            }
        }

        void CollectSNodeOriginCounts(
            const SNode *node, std::unordered_map< std::string, unsigned > &counts
        ) {
            if (!node) { return; }
            CountPayloadOrigins(node->Origins(), counts);
            node->for_each_child([&](SNode *child) {
                CollectSNodeOriginCounts(child, counts);
            });
        }

        void CollectSNodeOriginCounts(
            const std::vector< SNode * > &root,
            std::unordered_map< std::string, unsigned > &counts
        ) {
            for (const SNode *node : root) { CollectSNodeOriginCounts(node, counts); }
        }

        std::unordered_map< std::string, unsigned > CollectExpectedOwnershipCounts(
            const CGraph &flow_graph, const FunctionBuilder &builder
        ) {
            std::unordered_map< std::string, unsigned > counts;
            for (const auto &node : flow_graph.nodes) {
                for (const clang::Stmt *stmt : node.stmts) {
                    std::vector< StmtOrigin > origins;
                    CollectRawStmtOrigins(builder, stmt, origins);
                    CountPayloadOrigins(origins, counts);
                }
            }
            return counts;
        }

        SNodeOwnershipReport ValidateSNodePayloadOwnership(
            const CGraph &flow_graph, const std::vector< SNode * > &root,
            const FunctionBuilder &builder
        ) {
            SNodeOwnershipReport report;
            auto expected = CollectExpectedOwnershipCounts(flow_graph, builder);

            std::unordered_map< std::string, unsigned > actual;
            CollectSNodeOriginCounts(root, actual);

            for (const auto &[_, count] : expected) { report.expected_owned_ops += count; }
            for (const auto &[_, count] : actual) { report.retained_owned_ops += count; }

            for (const auto &[op_key, expected_count] : expected) {
                unsigned actual_count = 0;
                if (auto it = actual.find(op_key); it != actual.end()) {
                    actual_count = it->second;
                }
                if (actual_count < expected_count) {
                    report.missing_owned_ops.push_back(op_key);
                    report.diagnostics.push_back(
                        "missing SNode-owned payload operation " + op_key + " expected "
                        + std::to_string(expected_count) + " occurrence(s), saw "
                        + std::to_string(actual_count)
                    );
                } else if (actual_count > expected_count) {
                    report.duplicated_owned_ops.push_back(op_key);
                }
            }

            for (const auto &[op_key, actual_count] : actual) {
                if (!expected.contains(op_key)) {
                    report.extra_owned_ops.push_back(
                        op_key + " count=" + std::to_string(actual_count)
                    );
                }
            }

            std::sort(report.missing_owned_ops.begin(), report.missing_owned_ops.end());
            std::sort(report.duplicated_owned_ops.begin(), report.duplicated_owned_ops.end());
            std::sort(report.extra_owned_ops.begin(), report.extra_owned_ops.end());
            return report;
        }

        void CollectClangStmtOccurrences(
            const clang::Stmt *stmt, std::unordered_map< const clang::Stmt *, unsigned > &counts
        ) {
            if (!stmt) { return; }
            ++counts[stmt];

            if (const auto *decl_stmt = llvm::dyn_cast< clang::DeclStmt >(stmt)) {
                for (const clang::Decl *decl : decl_stmt->decls()) {
                    if (const auto *var = llvm::dyn_cast< clang::VarDecl >(decl)) {
                        CollectClangStmtOccurrences(var->getInit(), counts);
                    }
                }
            }

            for (const clang::Stmt *child : stmt->children()) {
                CollectClangStmtOccurrences(child, counts);
            }
        }

        void CollectSNodeStmtOccurrences(
            const SNode *node, std::unordered_map< const clang::Stmt *, unsigned > &counts
        ) {
            if (!node) { return; }
            if (const auto *stmt_node = node->dyn_cast< SStmt >()) {
                CollectClangStmtOccurrences(stmt_node->Stmt(), counts);
                return;
            }
            node->for_each_child([&](SNode *child) {
                CollectSNodeStmtOccurrences(child, counts);
            });
        }

        void CollectSNodeStmtOccurrences(
            const std::vector< SNode * > &root,
            std::unordered_map< const clang::Stmt *, unsigned > &counts
        ) {
            for (const SNode *node : root) { CollectSNodeStmtOccurrences(node, counts); }
        }

        std::unordered_map< const clang::Stmt *, unsigned >
        CollectBaselinePayloadStmts(const CGraph &flow_graph) {
            std::unordered_map< const clang::Stmt *, unsigned > counts;
            for (const auto &node : flow_graph.nodes) {
                for (const clang::Stmt *stmt : node.stmts) {
                    if (stmt) { ++counts[stmt]; }
                }
            }
            return counts;
        }

        PayloadRetentionReport ValidatePayloadRetention(
            const std::unordered_map< const clang::Stmt *, unsigned > &baseline_payload_stmts,
            const std::unordered_map< const clang::Stmt *, unsigned > &retained_payload_stmts,
            std::string_view missing_context
        ) {
            PayloadRetentionReport report;
            for (const auto &[_, count] : baseline_payload_stmts) {
                report.baseline_payload_stmts += count;
            }

            for (const auto &[stmt, expected_count] : baseline_payload_stmts) {
                unsigned actual_count = 0;
                if (auto it = retained_payload_stmts.find(stmt);
                    it != retained_payload_stmts.end())
                {
                    actual_count = it->second;
                }
                report.retained_payload_stmts += actual_count;

                if (actual_count < expected_count) {
                    report.missing_payload_stmts.push_back(DescribeStmt(stmt));
                    report.diagnostics.push_back(
                        std::string("missing ") + std::string(missing_context)
                        + " payload statement " + DescribeStmt(stmt)
                    );
                    continue;
                }
                if (actual_count > expected_count) {
                    report.duplicated_payload_stmts.push_back(DescribeStmt(stmt));
                }
            }

            std::sort(report.missing_payload_stmts.begin(), report.missing_payload_stmts.end());
            std::sort(
                report.duplicated_payload_stmts.begin(), report.duplicated_payload_stmts.end()
            );
            return report;
        }

        PayloadRetentionReport ValidateStructuredPayloadRetention(
            const std::unordered_map< const clang::Stmt *, unsigned > &baseline_payload_stmts,
            const std::vector< SNode * > &structured_body
        ) {
            std::unordered_map< const clang::Stmt *, unsigned > structured_stmts;
            CollectSNodeStmtOccurrences(structured_body, structured_stmts);
            return ValidatePayloadRetention(
                baseline_payload_stmts, structured_stmts, "structured"
            );
        }

        PayloadRetentionReport ValidateClangPayloadRetention(
            const std::unordered_map< const clang::Stmt *, unsigned > &baseline_payload_stmts,
            const clang::FunctionDecl *fn
        ) {
            std::unordered_map< const clang::Stmt *, unsigned > emitted_stmts;
            if (fn && fn->hasBody()) {
                CollectClangStmtOccurrences(fn->getBody(), emitted_stmts);
            }
            return ValidatePayloadRetention(baseline_payload_stmts, emitted_stmts, "emitted");
        }

        void LogPayloadRetentionFailure(
            std::string_view verifier_name, std::string_view fn_name,
            const PayloadRetentionReport &report
        ) {
            LOG(ERROR) << verifier_name << " failed for " << fn_name
                       << " baseline_payload_stmts=" << report.baseline_payload_stmts
                       << " retained_payload_stmts=" << report.retained_payload_stmts
                       << " missing_payload_stmts=" << report.missing_payload_stmts.size()
                       << " duplicated_payload_stmts=" << report.duplicated_payload_stmts.size()
                       << " diagnostics=" << report.diagnostics.size() << "\n";
            constexpr size_t kMaxPayloadDiagnostics = 20;
            for (size_t i = 0; i < std::min(report.diagnostics.size(), kMaxPayloadDiagnostics);
                 ++i)
            {
                LOG(ERROR) << "  " << report.diagnostics[i] << "\n";
            }
            if (report.diagnostics.size() > kMaxPayloadDiagnostics) {
                LOG(ERROR) << "  ... " << (report.diagnostics.size() - kMaxPayloadDiagnostics)
                           << " more payload diagnostic(s)\n";
            }
        }

        void
        LogSNodeOwnershipFailure(std::string_view fn_name, const SNodeOwnershipReport &report) {
            LOG(ERROR) << "SNode payload ownership verification failed for " << fn_name
                       << " expected_owned_ops=" << report.expected_owned_ops
                       << " retained_owned_ops=" << report.retained_owned_ops
                       << " missing_owned_ops=" << report.missing_owned_ops.size()
                       << " duplicated_owned_ops=" << report.duplicated_owned_ops.size()
                       << " extra_owned_ops=" << report.extra_owned_ops.size()
                       << " diagnostics=" << report.diagnostics.size() << "\n";
            constexpr size_t kMaxOwnershipDiagnostics = 20;
            for (size_t i = 0;
                 i < std::min(report.diagnostics.size(), kMaxOwnershipDiagnostics); ++i)
            {
                LOG(ERROR) << "  " << report.diagnostics[i] << "\n";
            }
            if (report.diagnostics.size() > kMaxOwnershipDiagnostics) {
                LOG(ERROR) << "  ... " << (report.diagnostics.size() - kMaxOwnershipDiagnostics)
                           << " more SNode ownership diagnostic(s)\n";
            }
        }

        void
        LogSNodeRegionFailure(std::string_view fn_name, const SRegionValidationReport &report) {
            LOG(ERROR) << "SNode region ownership verification failed for " << fn_name
                       << " regions=" << report.regions << " owned_ops=" << report.owned_ops
                       << " duplicated_owned_ops=" << report.duplicated_owned_ops
                       << " opaque_compound_payloads=" << report.opaque_compound_payloads
                       << " diagnostics=" << report.diagnostics.size() << "\n";
            constexpr size_t kMaxRegionDiagnostics = 20;
            for (size_t i = 0; i < std::min(report.diagnostics.size(), kMaxRegionDiagnostics);
                 ++i)
            {
                LOG(ERROR) << "  " << report.diagnostics[i] << "\n";
            }
            if (report.diagnostics.size() > kMaxRegionDiagnostics) {
                LOG(ERROR) << "  ... " << (report.diagnostics.size() - kMaxRegionDiagnostics)
                           << " more SNode region diagnostic(s)\n";
            }
        }

        void LogSNodeRegionLegalityFailure(
            std::string_view fn_name, const SRegionLegalityReport &report
        ) {
            LOG(ERROR) << "SNode region legality verification failed for " << fn_name
                       << " owned_ops=" << report.owned_ops
                       << " cloned_owned_ops=" << report.cloned_owned_ops
                       << " cross_region_cloned_ops=" << report.cross_region_cloned_ops
                       << " illegal_cloned_ops=" << report.illegal_cloned_ops
                       << " diagnostics=" << report.diagnostics.size() << "\n";
            constexpr size_t kMaxLegalityDiagnostics = 20;
            for (size_t i = 0; i < std::min(report.diagnostics.size(), kMaxLegalityDiagnostics);
                 ++i)
            {
                LOG(ERROR) << "  " << report.diagnostics[i] << "\n";
            }
            if (report.diagnostics.size() > kMaxLegalityDiagnostics) {
                LOG(ERROR) << "  ... " << (report.diagnostics.size() - kMaxLegalityDiagnostics)
                           << " more SNode region legality diagnostic(s)\n";
            }
        }

        bool ValidateSNodeRewriteTransaction(
            const CGraph &flow_graph, const std::vector< SNode * > &root,
            const FunctionBuilder &builder, std::string_view fn_name, std::string_view pass_name
        ) {
            auto ownership_report = ValidateSNodePayloadOwnership(flow_graph, root, builder);
            if (!ownership_report.ok()) {
                LOG(ERROR) << "SNode rewrite transaction failed in " << pass_name << " for "
                           << fn_name << "\n";
                LogSNodeOwnershipFailure(fn_name, ownership_report);
                return false;
            }

            auto region_graph  = BuildSNodeRegionGraph(root);
            auto region_report = ValidateSNodeRegionGraph(region_graph);
            if (!region_report.ok()) {
                LOG(ERROR) << "SNode rewrite transaction failed in " << pass_name << " for "
                           << fn_name << "\n";
                LogSNodeRegionFailure(fn_name, region_report);
                return false;
            }

            return true;
        }

        template< typename RewriteFn >
        bool RunSNodeRewriteTransaction(
            std::vector< SNode * > &root, SNodeFactory &factory, const CGraph &flow_graph,
            const FunctionBuilder &builder, std::string_view fn_name,
            std::string_view pass_name, bool verify, RewriteFn rewrite
        ) {
            if (!verify) { return rewrite(); }

            std::vector< SNode * > snapshot = CloneSNodeSeqForTransaction(root, factory);
            bool changed                    = rewrite();
            if (!changed) { return false; }

            AnnotateSNodeOrigins(builder, root);
            if (ValidateSNodeRewriteTransaction(flow_graph, root, builder, fn_name, pass_name))
            {
                return true;
            }

            LOG(WARNING) << "Rolling back structuring cleanup pass " << pass_name << " for "
                         << fn_name << " after transaction validation failure\n";
            root = std::move(snapshot);
            AnnotateSNodeOrigins(builder, root);
            return false;
        }

        struct SNodeCleanupPass
        {
            std::string_view name;
            SRewriteDecision rewrite_decision = SRewriteDecision::LeaveGoto;
            std::function< bool() > rewrite;
        };

        struct SNodeCleanupScheduleReport
        {
            size_t passes_run                       = 0;
            size_t changed_passes                   = 0;
            size_t changed_profitable_rewrite_passes = 0;

            bool changed() const { return changed_passes != 0; }
        };

        bool IsProfitableRewriteDecision(SRewriteDecision decision) {
            return decision != SRewriteDecision::LeaveGoto;
        }

        template< typename RewriteFn >
        SNodeCleanupPass MakeSNodeCleanupPass(
            std::string_view name, SRewriteDecision rewrite_decision, RewriteFn rewrite
        ) {
            SNodeCleanupPass pass;
            pass.name             = name;
            pass.rewrite_decision = rewrite_decision;
            pass.rewrite          = rewrite;
            return pass;
        }

        template< typename RunCleanupFn >
        SNodeCleanupScheduleReport RunSNodeCleanupSchedule(
            const std::vector< SNodeCleanupPass > &passes, RunCleanupFn &&run_cleanup
        ) {
            SNodeCleanupScheduleReport report;
            for (const auto &pass : passes) {
                ++report.passes_run;
                bool changed = run_cleanup(pass.name, pass.rewrite);
                if (!changed) { continue; }

                ++report.changed_passes;
                if (IsProfitableRewriteDecision(pass.rewrite_decision)) {
                    ++report.changed_profitable_rewrite_passes;
                }
            }
            return report;
        }

        template< typename MakePassesFn, typename RunCleanupFn >
        SNodeCleanupScheduleReport RunSNodeCleanupFixedPoint(
            MakePassesFn &&make_passes, RunCleanupFn &&run_cleanup, int max_iterations
        ) {
            SNodeCleanupScheduleReport total;
            for (int pass = 0; pass < max_iterations; ++pass) {
                auto iteration = RunSNodeCleanupSchedule(make_passes(), run_cleanup);
                total.passes_run += iteration.passes_run;
                total.changed_passes += iteration.changed_passes;
                total.changed_profitable_rewrite_passes +=
                    iteration.changed_profitable_rewrite_passes;
                if (!iteration.changed()) { break; }
            }
            return total;
        }

        struct ControlShapeSummary
        {
            size_t labels    = 0;
            size_t gotos     = 0;
            size_t switches  = 0;
            size_t cases     = 0;
            size_t defaults  = 0;
            size_t breaks    = 0;
            size_t continues = 0;
            size_t returns   = 0;
            size_t fors      = 0;
        };

        struct ControlShapeReport
        {
            ControlShapeSummary expected;
            ControlShapeSummary emitted;
            std::vector< std::string > diagnostics;

            bool ok() const { return diagnostics.empty(); }
        };

        void CollectClangControlShape(const clang::Stmt *stmt, ControlShapeSummary &summary) {
            if (!stmt) { return; }

            if (const auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ++summary.labels;
                CollectClangControlShape(label->getSubStmt(), summary);
                return;
            }
            if (llvm::isa< clang::GotoStmt >(stmt)) {
                ++summary.gotos;
                return;
            }
            if (const auto *switch_stmt = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                ++summary.switches;
                CollectClangControlShape(switch_stmt->getBody(), summary);
                return;
            }
            if (const auto *for_stmt = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                ++summary.fors;
                CollectClangControlShape(for_stmt->getInit(), summary);
                CollectClangControlShape(for_stmt->getCond(), summary);
                CollectClangControlShape(for_stmt->getInc(), summary);
                CollectClangControlShape(for_stmt->getBody(), summary);
                return;
            }
            if (const auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(stmt)) {
                ++summary.cases;
                CollectClangControlShape(case_stmt->getSubStmt(), summary);
                return;
            }
            if (const auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(stmt)) {
                ++summary.defaults;
                CollectClangControlShape(default_stmt->getSubStmt(), summary);
                return;
            }
            if (llvm::isa< clang::BreakStmt >(stmt)) {
                ++summary.breaks;
                return;
            }
            if (llvm::isa< clang::ContinueStmt >(stmt)) {
                ++summary.continues;
                return;
            }
            if (llvm::isa< clang::ReturnStmt >(stmt)) {
                ++summary.returns;
                return;
            }

            if (const auto *decl_stmt = llvm::dyn_cast< clang::DeclStmt >(stmt)) {
                for (const clang::Decl *decl : decl_stmt->decls()) {
                    if (const auto *var = llvm::dyn_cast< clang::VarDecl >(decl)) {
                        CollectClangControlShape(var->getInit(), summary);
                    }
                }
            }

            for (const clang::Stmt *child : stmt->children()) {
                CollectClangControlShape(child, summary);
            }
        }

        void CollectSNodeControlShape(const SNode *node, ControlShapeSummary &summary) {
            if (!node) { return; }

            if (const auto *stmt_node = node->dyn_cast< SStmt >()) {
                CollectClangControlShape(stmt_node->Stmt(), summary);
                return;
            }
            if (node->dyn_cast< SGoto >()) {
                ++summary.gotos;
                return;
            }
            if (const auto *label = node->dyn_cast< SLabel >()) {
                ++summary.labels;
                for (const SNode *child : label->BodyList()) {
                    CollectSNodeControlShape(child, summary);
                }
                return;
            }
            if (const auto *switch_node = node->dyn_cast< SSwitch >()) {
                ++summary.switches;
                summary.cases += switch_node->Cases().size();
                if (!switch_node->DefaultBodyList().empty()) { ++summary.defaults; }
                for (const auto &case_node : switch_node->Cases()) {
                    for (const SNode *child : case_node.body_list) {
                        CollectSNodeControlShape(child, summary);
                    }
                }
                for (const SNode *child : switch_node->DefaultBodyList()) {
                    CollectSNodeControlShape(child, summary);
                }
                return;
            }
            if (const auto *for_node = node->dyn_cast< SFor >()) {
                ++summary.fors;
                CollectClangControlShape(for_node->Init(), summary);
                CollectClangControlShape(for_node->Cond(), summary);
                CollectClangControlShape(for_node->Inc(), summary);
                for (const SNode *child : for_node->BodyList()) {
                    CollectSNodeControlShape(child, summary);
                }
                return;
            }
            if (node->dyn_cast< SBreak >()) {
                ++summary.breaks;
                return;
            }
            if (node->dyn_cast< SContinue >()) {
                ++summary.continues;
                return;
            }
            if (node->dyn_cast< SReturn >()) {
                ++summary.returns;
                return;
            }

            node->for_each_child([&](SNode *child) {
                CollectSNodeControlShape(child, summary);
            });
        }

        ControlShapeSummary CollectSNodeControlShape(const std::vector< SNode * > &root) {
            ControlShapeSummary summary;
            for (const SNode *node : root) { CollectSNodeControlShape(node, summary); }
            return summary;
        }

        ControlShapeSummary CollectClangControlShape(const clang::FunctionDecl *fn) {
            ControlShapeSummary summary;
            if (fn && fn->hasBody()) { CollectClangControlShape(fn->getBody(), summary); }
            return summary;
        }

        ControlShapeReport ValidateClangControlShape(
            const std::vector< SNode * > &root, const clang::FunctionDecl *fn
        ) {
            ControlShapeReport report;
            report.expected = CollectSNodeControlShape(root);
            report.emitted  = CollectClangControlShape(fn);

            auto require_equal = [&](std::string_view field, size_t expected, size_t emitted) {
                if (expected == emitted) { return; }
                report.diagnostics.push_back(
                    std::string("control shape mismatch for ") + std::string(field)
                    + ": expected " + std::to_string(expected) + ", emitted "
                    + std::to_string(emitted)
                );
            };
            auto require_at_most = [&](std::string_view field, size_t expected,
                                       size_t emitted) {
                if (emitted <= expected) { return; }
                report.diagnostics.push_back(
                    std::string("control shape growth for ") + std::string(field)
                    + ": expected at most " + std::to_string(expected) + ", emitted "
                    + std::to_string(emitted)
                );
            };
            auto require_at_least = [&](std::string_view field, size_t expected,
                                        size_t emitted) {
                if (emitted >= expected) { return; }
                report.diagnostics.push_back(
                    std::string("control shape loss for ") + std::string(field)
                    + ": expected at least " + std::to_string(expected) + ", emitted "
                    + std::to_string(emitted)
                );
            };

            // Structuring and pretty-print cleanup intentionally remove labels
            // and gotos.  Growth is suspicious, but reduction is expected.
            require_at_most("labels", report.expected.labels, report.emitted.labels);
            require_at_most("gotos", report.expected.gotos, report.emitted.gotos);
            require_equal("switches", report.expected.switches, report.emitted.switches);
            require_equal("cases", report.expected.cases, report.emitted.cases);
            require_equal("defaults", report.expected.defaults, report.emitted.defaults);

            // ClangEmitter inserts implicit breaks for non-terminating switch
            // cases, and cleanup can duplicate terminating payloads through
            // safe inlining.  For-loop promotion can also erase one redundant
            // terminal continue per newly-emitted for loop: `continue;` at the
            // end of a for body has the same effect as falling off the body.
            // These terminator classes are therefore lower bounds rather than
            // exact-shape fields.
            require_at_least("breaks", report.expected.breaks, report.emitted.breaks);
            size_t promoted_for_allowance = 0;
            if (report.emitted.fors > report.expected.fors) {
                promoted_for_allowance = report.emitted.fors - report.expected.fors;
            }
            require_at_least(
                "continues", report.expected.continues,
                report.emitted.continues + promoted_for_allowance
            );
            require_at_least("returns", report.expected.returns, report.emitted.returns);
            return report;
        }

        void
        LogControlShapeFailure(std::string_view fn_name, const ControlShapeReport &report) {
            LOG(ERROR) << "Clang control shape verification failed for " << fn_name
                       << " expected_labels=" << report.expected.labels
                       << " emitted_labels=" << report.emitted.labels
                       << " expected_gotos=" << report.expected.gotos
                       << " emitted_gotos=" << report.emitted.gotos
                       << " expected_switches=" << report.expected.switches
                       << " emitted_switches=" << report.emitted.switches
                       << " expected_cases=" << report.expected.cases
                       << " emitted_cases=" << report.emitted.cases
                       << " expected_defaults=" << report.expected.defaults
                       << " emitted_defaults=" << report.emitted.defaults
                       << " expected_breaks=" << report.expected.breaks
                       << " emitted_breaks=" << report.emitted.breaks
                       << " expected_continues=" << report.expected.continues
                       << " emitted_continues=" << report.emitted.continues
                       << " expected_fors=" << report.expected.fors
                       << " emitted_fors=" << report.emitted.fors
                       << " expected_returns=" << report.expected.returns
                       << " emitted_returns=" << report.emitted.returns
                       << " diagnostics=" << report.diagnostics.size() << "\n";
            constexpr size_t kMaxControlDiagnostics = 20;
            for (size_t i = 0; i < std::min(report.diagnostics.size(), kMaxControlDiagnostics);
                 ++i)
            {
                LOG(ERROR) << "  " << report.diagnostics[i] << "\n";
            }
            if (report.diagnostics.size() > kMaxControlDiagnostics) {
                LOG(ERROR) << "  ... " << (report.diagnostics.size() - kMaxControlDiagnostics)
                           << " more control-shape diagnostic(s)\n";
            }
        }

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
            constexpr size_t kMaxSNodeDiagnostics = 20;
            for (size_t i = 0; i < std::min(report.diagnostics.size(), kMaxSNodeDiagnostics);
                 ++i)
            {
                LOG(ERROR) << "  " << report.diagnostics[i] << "\n";
            }
            if (report.diagnostics.size() > kMaxSNodeDiagnostics) {
                LOG(ERROR) << "  ... " << (report.diagnostics.size() - kMaxSNodeDiagnostics)
                           << " more SNode diagnostic(s)\n";
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
            std::vector< std::shared_ptr< FunctionBuilder > > func_builders;
            const auto &program_arch = get_program().arch.value_or(std::string{});

            std::vector< std::string > function_keys;
            function_keys.reserve(get_program().serialized_functions.size());
            for (const auto &entry : get_program().serialized_functions) {
                function_keys.push_back(entry.first);
            }
            std::sort(function_keys.begin(), function_keys.end());

            for (const auto &key : function_keys) {
                const auto &function = get_program().serialized_functions.at(key);
                auto builder         = std::make_shared< FunctionBuilder >(
                    ci, function, *type_builder, function_declarations,
                    global_variable_declarations, intrinsic_declarations, program_arch
                );
                builder->InitializeOpBuilder();
                func_builders.emplace_back(std::move(builder));
            }

            for (auto &builder : func_builders) {
                if (!builder->has_basic_blocks()) { continue; }

                auto *fn = builder->create_definition(ctx);
                if (!fn) { continue; }

                const auto &func    = builder->get_function();
                std::string fn_name = func.display_name.empty() ? func.name : func.display_name;

                // Set sema context to function for stmt building
                auto *prev_ctx    = builder->enter_function_context(fn);
                CGraph flow_graph = BuildCGraph(*builder, ctx);
                builder->leave_function_context(prev_ctx);
                if (flow_graph.nodes.empty()) {
                    LOG(WARNING) << "BuildCGraph produced empty graph for " << fn_name << "\n";
                    continue;
                }
                if (options.verify_no_node_loss) {
                    auto cfg_report = ValidateCGraph(flow_graph, &func);
                    if (!cfg_report.ok()) {
                        LOG(ERROR)
                            << "CGraph verification failed for " << fn_name
                            << " nodes=" << cfg_report.node_count
                            << " active=" << cfg_report.active_nodes
                            << " edges=" << cfg_report.edge_count
                            << " input_blocks=" << cfg_report.input_blocks
                            << " emitted_blocks=" << cfg_report.emitted_blocks
                            << " missing_blocks=" << cfg_report.missing_blocks.size()
                            << " extra_blocks=" << cfg_report.extra_blocks.size()
                            << " input_edges=" << cfg_report.input_edges
                            << " emitted_edges=" << cfg_report.emitted_edges
                            << " missing_edges=" << cfg_report.missing_edges.size()
                            << " extra_edges=" << cfg_report.extra_edges.size()
                            << " duplicated_edges=" << cfg_report.duplicated_edges.size()
                            << " input_switches=" << cfg_report.input_switches
                            << " emitted_switches=" << cfg_report.emitted_switches
                            << " missing_switches=" << cfg_report.missing_switches.size()
                            << " extra_switches=" << cfg_report.extra_switches.size()
                            << " input_cases=" << cfg_report.input_cases
                            << " emitted_cases=" << cfg_report.emitted_cases
                            << " missing_cases=" << cfg_report.missing_cases.size()
                            << " extra_cases=" << cfg_report.extra_cases.size()
                            << " duplicated_cases=" << cfg_report.duplicated_cases.size()
                            << " normalized_conditions=" << cfg_report.normalized_conditions
                            << " branch_swaps=" << cfg_report.branch_swaps
                            << " condition_negations=" << cfg_report.condition_negations
                            << " irreducible_regions=" << cfg_report.irreducible_regions
                            << " diagnostics=" << cfg_report.diagnostics.size() << "\n";
                        constexpr size_t kMaxCfgDiagnostics = 20;
                        for (size_t i = 0;
                             i < std::min(cfg_report.diagnostics.size(), kMaxCfgDiagnostics);
                             ++i)
                        {
                            LOG(ERROR) << "  " << cfg_report.diagnostics[i] << "\n";
                        }
                        if (cfg_report.diagnostics.size() > kMaxCfgDiagnostics) {
                            LOG(ERROR) << "  ... "
                                       << (cfg_report.diagnostics.size() - kMaxCfgDiagnostics)
                                       << " more CGraph diagnostic(s)\n";
                        }
                        LOG(FATAL)
                            << "CGraph source verification failed for " << fn_name << "\n";
                    }
                }

                // Optionally emit DOT graph before emission.
                if (options.emit_dot_cfg) {
                    CGraphDotTracer tracer;
                    tracer.fn_name = fn_name;
                    tracer.enabled = true;
                    tracer.Dump(flow_graph, "BuildCGraph", true);
                }

                SNodeFactory factory;
                // The function body is a sequence of SNodes
                // (std::vector) — the SSeq node kind was removed.
                std::vector< SNode * > root_body;
                bool have_structured = false;
                std::unordered_map< const clang::Stmt *, unsigned > baseline_payload_stmts;

                if (options.use_structuring_pass) {
                    if (options.verify_no_node_loss) {
                        SNodeFactory baseline_factory;
                        auto baseline_body =
                            BuildGotoSNodeBody(flow_graph, baseline_factory, ctx);
                        baseline_payload_stmts = CollectBaselinePayloadStmts(flow_graph);
                        auto baseline_report   = ValidateSNodeTree(baseline_body, &flow_graph);
                        if (!baseline_report.ok()) {
                            LogSNodeVerificationFailure(
                                "Goto baseline SNode verification", fn_name, baseline_report
                            );
                            LOG(FATAL) << "Goto baseline source verification failed for "
                                       << fn_name << "\n";
                        }
                    }

                    // Structured path: run CFGStructure to fold the
                    // CGraph into hierarchical SNodes.
                    CFGStructure cfg_structure(flow_graph, factory, ctx);
                    cfg_structure.StructureAll();

                    // Build the root sequence from the remaining active
                    // (uncollapsed) nodes.  After StructureAll, each
                    // active node carries a ->structured sequence.
                    for (auto &node : flow_graph.nodes) {
                        if (node.IsCollapsed()) { continue; }
                        for (SNode *s : node.structured) { root_body.push_back(s); }
                    }

                    // A fully-empty structured result falls through to
                    // the goto-based path below (matches prior behaviour
                    // when MakeSeq returned nullptr).
                    if (!root_body.empty()) {
                        have_structured = true;
                        AnnotateSNodeOrigins(*builder, root_body);
                        auto run_cleanup = [&](std::string_view pass_name, auto rewrite) {
                            return RunSNodeRewriteTransaction(
                                root_body, factory, flow_graph, *builder, fn_name, pass_name,
                                options.verify_no_node_loss, rewrite
                            );
                        };

                        auto make_control_cleanup_passes = [&]() {
                            std::vector< SNodeCleanupPass > passes;
                            passes.push_back(MakeSNodeCleanupPass(
                                "ConvertGotoToBreakContinue", SRewriteDecision::LeaveGoto,
                                [&]() { return ConvertGotoToBreakContinue(root_body, factory); }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "ConvertGotoToReturn", SRewriteDecision::LeaveGoto,
                                [&]() { return ConvertGotoToReturn(root_body, factory, ctx); }
                            ));
                            return passes;
                        };

                        auto make_profitable_rewrite_passes = [&]() {
                            std::vector< SNodeCleanupPass > passes;
                            passes.push_back(MakeSNodeCleanupPass(
                                "DuplicateSwitchCaseTargets", SRewriteDecision::Clone,
                                [&]() { return DuplicateSwitchCaseTargets(root_body, factory); }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "FoldSwitchLocalCaseTargets", SRewriteDecision::Move,
                                [&]() {
                                    return FoldSwitchLocalCaseTargets(
                                        root_body, factory, ctx
                                    );
                                }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "DuplicateSmallTerminatingTargets", SRewriteDecision::Clone,
                                [&]() {
                                    return DuplicateSmallTerminatingTargets(root_body, factory);
                                }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "DuplicateSmallEpilogueTargets", SRewriteDecision::Clone,
                                [&]() {
                                    return DuplicateSmallEpilogueTargets(
                                        root_body, factory, ctx
                                    );
                                }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "DuplicateSwitchFallthroughTargets", SRewriteDecision::Clone,
                                [&]() {
                                    return DuplicateSwitchFallthroughTargets(
                                        root_body, factory, ctx
                                    );
                                }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "DuplicateLoopContinueTargets", SRewriteDecision::Clone,
                                [&]() {
                                    return DuplicateLoopContinueTargets(
                                        root_body, factory, ctx
                                    );
                                }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "FoldGuardedFallthroughTargets", SRewriteDecision::Move,
                                [&]() {
                                    return FoldGuardedFallthroughTargets(
                                        root_body, factory, ctx
                                    );
                                }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "RepairCrossScopeLabelEntries", SRewriteDecision::Clone,
                                [&]() {
                                    return RepairCrossScopeLabelEntries(
                                        root_body, factory, ctx
                                    );
                                }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "DuplicateStackGuardReturnTargets", SRewriteDecision::Clone,
                                [&]() {
                                    return DuplicateStackGuardReturnTargets(
                                        root_body, factory, ctx
                                    );
                                }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "DuplicateCleanupReturnTargets", SRewriteDecision::Clone,
                                [&]() {
                                    return DuplicateCleanupReturnTargets(
                                        root_body, factory, ctx
                                    );
                                }
                            ));
                            return passes;
                        };

                        auto make_structural_cleanup_passes = [&]() {
                            std::vector< SNodeCleanupPass > passes;
                            passes.push_back(MakeSNodeCleanupPass(
                                "SimplifyEmptyControlFlow", SRewriteDecision::LeaveGoto,
                                [&]() { return SimplifyEmptyControlFlow(root_body, ctx); }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "CollapsePassThroughLabels", SRewriteDecision::LeaveGoto,
                                [&]() { return CollapsePassThroughLabels(root_body, factory); }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "MergeRedundantGotoGuards", SRewriteDecision::LeaveGoto,
                                [&]() {
                                    return MergeRedundantGotoGuards(root_body, factory, ctx);
                                }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "InlineResidualGotos", SRewriteDecision::Move,
                                [&]() { return InlineResidualGotos(root_body, factory); }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "InlineCrossScopeSingleRef", SRewriteDecision::Move,
                                [&]() { return InlineCrossScopeSingleRef(root_body, factory); }
                            ));
                            return passes;
                        };

                        auto make_scope_cleanup_passes = [&]() {
                            std::vector< SNodeCleanupPass > passes;
                            passes.push_back(MakeSNodeCleanupPass(
                                "AbsorbFallthroughIntoElse", SRewriteDecision::LeaveGoto,
                                [&]() { return AbsorbFallthroughIntoElse(root_body, factory); }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "ScopeifyIfGotos", SRewriteDecision::LeaveGoto,
                                [&]() { return ScopeifyIfGotos(root_body, factory, ctx); }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "EliminateGotoToNextLabel", SRewriteDecision::LeaveGoto,
                                [&]() {
                                    return EliminateGotoToNextLabel(root_body, factory, ctx);
                                }
                            ));
                            return passes;
                        };

                        auto make_final_cleanup_passes = [&]() {
                            std::vector< SNodeCleanupPass > passes;
                            passes.push_back(MakeSNodeCleanupPass(
                                "RemoveDeadSSeqChildren", SRewriteDecision::LeaveGoto,
                                [&]() { return RemoveDeadSSeqChildren(root_body); }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "RemoveUnreferencedLabels", SRewriteDecision::LeaveGoto,
                                [&]() { return RemoveUnreferencedLabels(root_body, factory); }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "SimplifyEmptyControlFlow", SRewriteDecision::LeaveGoto,
                                [&]() { return SimplifyEmptyControlFlow(root_body, ctx); }
                            ));
                            return passes;
                        };

                        auto make_initial_cleanup_schedule = [&]() {
                            std::vector< SNodeCleanupPass > passes = make_control_cleanup_passes();
                            auto rewrite_passes = make_profitable_rewrite_passes();
                            passes.insert(
                                passes.end(), rewrite_passes.begin(), rewrite_passes.end()
                            );
                            auto cleanup_passes = make_structural_cleanup_passes();
                            passes.insert(
                                passes.end(), cleanup_passes.begin(), cleanup_passes.end()
                            );
                            return passes;
                        };

                        auto make_fixed_point_cleanup_schedule = [&]() {
                            std::vector< SNodeCleanupPass > passes;
                            passes.push_back(MakeSNodeCleanupPass(
                                "SimplifyEmptyControlFlow", SRewriteDecision::LeaveGoto,
                                [&]() { return SimplifyEmptyControlFlow(root_body, ctx); }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "CollapsePassThroughLabels", SRewriteDecision::LeaveGoto,
                                [&]() { return CollapsePassThroughLabels(root_body, factory); }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "MergeRedundantGotoGuards", SRewriteDecision::LeaveGoto,
                                [&]() {
                                    return MergeRedundantGotoGuards(root_body, factory, ctx);
                                }
                            ));
                            auto control_passes = make_control_cleanup_passes();
                            passes.insert(
                                passes.end(), control_passes.begin(), control_passes.end()
                            );
                            auto rewrite_passes = make_profitable_rewrite_passes();
                            passes.insert(
                                passes.end(), rewrite_passes.begin(), rewrite_passes.end()
                            );
                            auto scope_passes = make_scope_cleanup_passes();
                            passes.insert(passes.end(), scope_passes.begin(), scope_passes.end());
                            passes.push_back(MakeSNodeCleanupPass(
                                "InlineResidualGotos", SRewriteDecision::Move,
                                [&]() { return InlineResidualGotos(root_body, factory); }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "InlineCrossScopeSingleRef", SRewriteDecision::Move,
                                [&]() { return InlineCrossScopeSingleRef(root_body, factory); }
                            ));
                            passes.push_back(MakeSNodeCleanupPass(
                                "RemoveUnreferencedLabels", SRewriteDecision::LeaveGoto,
                                [&]() { return RemoveUnreferencedLabels(root_body, factory); }
                            ));
                            return passes;
                        };

                        RunSNodeCleanupSchedule(make_initial_cleanup_schedule(), run_cleanup);
                        RunSNodeCleanupFixedPoint(
                            make_fixed_point_cleanup_schedule, run_cleanup,
                            kMaxGotoEliminationPasses
                        );
                        RunSNodeCleanupSchedule(make_final_cleanup_passes(), run_cleanup);

                        if (options.verify_no_node_loss) {
                            AnnotateSNodeOrigins(*builder, root_body);
                            auto ownership_report =
                                ValidateSNodePayloadOwnership(flow_graph, root_body, *builder);
                            if (!ownership_report.ok()) {
                                LogSNodeOwnershipFailure(fn_name, ownership_report);
                                LOG(FATAL) << "SNode payload ownership verification failed for "
                                           << fn_name << "\n";
                            }

                            auto region_graph  = BuildSNodeRegionGraph(root_body);
                            auto region_report = ValidateSNodeRegionGraph(region_graph);
                            if (!region_report.ok()) {
                                LogSNodeRegionFailure(fn_name, region_report);
                                LOG(FATAL) << "SNode region ownership verification failed for "
                                           << fn_name << "\n";
                            }

                            auto legality_report = ValidateSNodeRegionLegality(region_graph);
                            if (!legality_report.ok()) {
                                LogSNodeRegionLegalityFailure(fn_name, legality_report);
                                LOG(FATAL) << "SNode region legality verification failed for "
                                           << fn_name << "\n";
                            }

                            auto payload_report = ValidateStructuredPayloadRetention(
                                baseline_payload_stmts, root_body
                            );
                            if (!payload_report.ok()) {
                                LogPayloadRetentionFailure(
                                    "Structured payload retention verification", fn_name,
                                    payload_report
                                );
                                LOG(FATAL) << "Structured payload retention "
                                           << "verification failed for " << fn_name << "\n";
                            }
                        }
                    }
                }

                if (!have_structured) {
                    root_body = BuildGotoSNodeBody(flow_graph, factory, ctx);
                }

                if (options.verify_no_node_loss && have_structured) {
                    auto snode_report = ValidateSNodeTree(root_body, &flow_graph);
                    if (!snode_report.ok()) {
                        LogSNodeVerificationFailure(
                            "SNode verification", fn_name, snode_report
                        );
                        LOG(FATAL)
                            << "SNode source verification failed for " << fn_name << "\n";
                    }
                }

                EmitClangAST(root_body, fn, ctx);

                CleanupPrettyPrint(fn, ctx);

                if (options.structuring_improvement_report && have_structured) {
                    auto report = AnalyzeStructuringImprovements(fn);
                    llvm::errs() << "STRUCTURING_IMPROVEMENT_REPORT function=" << fn_name
                                 << " residual_gotos=" << report.residual_gotos
                                 << " emitted_labels=" << report.emitted_labels
                                 << " dangling_gotos=" << report.dangling_gotos
                                 << " layout_fallthrough=" << report.layout_fallthrough
                                 << " pass_through_collapse=" << report.pass_through_collapse
                                 << " terminal_clone_inline=" << report.terminal_clone_inline
                                 << " single_ref_inline_reorder="
                                 << report.single_ref_inline_reorder
                                 << " small_target_clone=" << report.small_target_clone
                                 << " terminal_epilogue=" << report.terminal_epilogue
                                 << " cross_scope_gotos=" << report.cross_scope_gotos
                                 << " residual_hard=" << report.residual_hard << "\n";
                }

                if (options.verify_no_node_loss && have_structured) {
                    auto clang_report = ValidateEmittedClangAST(fn);
                    if (!clang_report.ok()) {
                        LOG(ERROR)
                            << "Clang AST emission verification failed for " << fn_name
                            << " emitted_labels=" << clang_report.emitted_labels
                            << " emitted_gotos=" << clang_report.emitted_gotos
                            << " dangling_gotos=" << clang_report.dangling_gotos
                            << " duplicate_labels=" << clang_report.duplicate_labels
                            << " empty_labels=" << clang_report.empty_labels
                            << " unreachable_statements=" << clang_report.unreachable_statements
                            << " switch_stmts=" << clang_report.switch_stmts
                            << " case_stmts=" << clang_report.case_stmts
                            << " default_stmts=" << clang_report.default_stmts
                            << " break_stmts=" << clang_report.break_stmts
                            << " continue_stmts=" << clang_report.continue_stmts
                            << " diagnostics=" << clang_report.diagnostics.size() << "\n";
                        constexpr size_t kMaxClangDiagnostics = 20;
                        for (size_t i = 0; i
                             < std::min(clang_report.diagnostics.size(), kMaxClangDiagnostics);
                             ++i)
                        {
                            LOG(ERROR) << "  " << clang_report.diagnostics[i] << "\n";
                        }
                        if (clang_report.diagnostics.size() > kMaxClangDiagnostics) {
                            LOG(ERROR)
                                << "  ... "
                                << (clang_report.diagnostics.size() - kMaxClangDiagnostics)
                                << " more Clang AST diagnostic(s)\n";
                        }
                        LOG(FATAL)
                            << "Clang AST emission verification failed for " << fn_name << "\n";
                    }

                    auto control_report = ValidateClangControlShape(root_body, fn);
                    if (!control_report.ok()) {
                        LogControlShapeFailure(fn_name, control_report);
                        LOG(FATAL) << "Clang control shape verification failed for " << fn_name
                                   << "\n";
                    }

                    auto structure_report = AnalyzeStructuringImprovements(fn);
                    if (structure_report.cross_scope_gotos != 0) {
                        LOG(ERROR)
                            << "Clang AST emission verification failed for " << fn_name
                            << " cross_scope_gotos=" << structure_report.cross_scope_gotos
                            << "\n";
                        LOG(FATAL) << "Clang AST contains goto entries into nested structured "
                                      "scopes for "
                                   << fn_name << "\n";
                    }
                }

                if (options.verify_no_node_loss && have_structured) {
                    auto payload_report =
                        ValidateClangPayloadRetention(baseline_payload_stmts, fn);
                    if (!payload_report.ok()) {
                        LogPayloadRetentionFailure(
                            "Clang payload retention verification", fn_name, payload_report
                        );
                        LOG(FATAL) << "Clang payload retention verification failed for "
                                   << fn_name << "\n";
                    }
                }

                if (options.verify_no_node_loss && !builder->VerifyNoNodeLoss(fn)) {
                    LOG(FATAL) << "Source operation node-loss verification "
                               << "failed for " << fn_name << "\n";
                }
            }
        }

        if (options.print_tu) {
            // Pretty-print the Clang AST as C to <output>.c.
            // This runs before codegen so the C file is always produced
            // even if CIR lowering encounters a diagnostic error.
            if (!options.output_file.empty()) {
                std::error_code ec;
                llvm::raw_fd_ostream out(
                    options.output_file + ".c", ec, llvm::sys::fs::OF_Text
                );
                if (!ec) {
                    ctx.getTranslationUnitDecl()->print(
                        out, ctx.getPrintingPolicy(), /*Indentation=*/0
                    );
                } else {
                    LOG(ERROR) << "Failed to write C output: " << ec.message() << "\n";
                }
            } else {
                // No output file — print to stdout
                ctx.getTranslationUnitDecl()->print(
                    llvm::outs(), ctx.getPrintingPolicy(), /*Indentation=*/0
                );
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
        std::vector< std::string > variable_keys;
        variable_keys.reserve(serialized_variables.size());
        for (const auto &entry : serialized_variables) { variable_keys.push_back(entry.first); }
        std::sort(variable_keys.begin(), variable_keys.end());

        for (const auto &key : variable_keys) {
            auto &variable = serialized_variables.at(key);
            if (variable.name.empty() || variable.type.empty()) { continue; }

            // Stale JSON without the #226 producer fix would trip CIRGen's
            // FuncOp assertion deep in vendor code — refuse loudly here.
            if (fns.find(key) != fns.end()) {
                LOG_FATAL(
                    "create_globals: global '{0}' at {1} collides with a "
                    "function entry of the same address — pcode.json is "
                    "malformed.  Regenerate it with a PcodeSerializer "
                    "that includes the #226 fix.",
                    variable.name, key
                );
            }

            auto var_type = type_builder->GetSerializedType(variable.type);
            if (var_type.isNull()) { continue; }
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
