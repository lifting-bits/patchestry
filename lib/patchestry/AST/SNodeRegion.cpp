/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/SNodeRegion.hpp>
#include <patchestry/AST/SourceOrigin.hpp>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <clang/AST/Stmt.h>
#include <llvm/Support/Casting.h>

namespace patchestry::ast {

    const char *SRegionKindName(SRegionKind kind) {
        switch (kind) {
            case SRegionKind::Root:
                return "root";
            case SRegionKind::LabelBody:
                return "label_body";
            case SRegionKind::IfThen:
                return "if_then";
            case SRegionKind::IfElse:
                return "if_else";
            case SRegionKind::WhileBody:
                return "while_body";
            case SRegionKind::DoWhileBody:
                return "do_while_body";
            case SRegionKind::ForBody:
                return "for_body";
            case SRegionKind::SwitchCase:
                return "switch_case";
            case SRegionKind::SwitchDefault:
                return "switch_default";
        }
        return "unknown";
    }

    namespace {

        void
        AddDiagnostic(std::vector< std::string > &diagnostics, const std::string &message) {
            diagnostics.push_back(message);
        }

        void AddPayloadOrigins(
            const SNode &node, std::unordered_map< std::string, unsigned > &counts,
            std::vector< StmtOrigin > &origins
        ) {
            for (const auto &origin : node.Origins()) {
                if (!origin.primary || !IsPayloadCarrierKind(origin.kind)) { continue; }
                ++counts[StmtOriginKey(origin)];
                origins.push_back(origin);
            }
        }

        bool HasOpaqueCompoundPayload(const SNode *node) {
            const auto *stmt_node = node ? node->dyn_cast< SStmt >() : nullptr;
            if (!stmt_node) { return false; }
            return llvm::isa_and_nonnull< clang::CompoundStmt >(stmt_node->Stmt());
        }

        size_t AddRegion(
            SRegionGraph &graph, SRegionKind kind, size_t parent, const SNode *owner,
            std::string name
        ) {
            size_t id = graph.regions.size();
            graph.regions.push_back(
                SRegionNode{
                    id,
                    parent,
                    kind,
                    owner,
                    std::move(name),
                    {},
                    0,
                    0,
                    0,
                    {},
                    {},
                    {},
                }
            );
            if (parent != SRegionNode::kNone && parent < graph.regions.size()) {
                graph.regions[parent].children.push_back(id);
            }
            return id;
        }

        void Accumulate(
            std::unordered_map< std::string, unsigned > &dst,
            const std::unordered_map< std::string, unsigned > &src
        ) {
            for (const auto &[key, count] : src) { dst[key] += count; }
        }

        void BuildRegionForSeq(
            SRegionGraph &graph, const std::vector< SNode * > &seq, SRegionKind kind,
            size_t parent, const SNode *owner, std::string name
        );

        void VisitNode(SRegionGraph &graph, SNode *node, size_t region_id) {
            if (!node || region_id >= graph.regions.size()) { return; }

            auto &region = graph.regions[region_id];
            ++region.direct_snodes;
            ++region.subtree_snodes;
            if (HasOpaqueCompoundPayload(node)) { ++region.opaque_compound_payloads; }
            AddPayloadOrigins(*node, region.direct_owned_ops, region.direct_origins);

            if (auto *label = node->dyn_cast< SLabel >()) {
                BuildRegionForSeq(
                    graph, label->BodyList(), SRegionKind::LabelBody, region_id, node,
                    std::string(label->Name())
                );
                return;
            }

            if (auto *if_node = node->dyn_cast< SIfThenElse >()) {
                BuildRegionForSeq(
                    graph, if_node->ThenList(), SRegionKind::IfThen, region_id, node, "then"
                );
                if (!if_node->ElseList().empty()) {
                    BuildRegionForSeq(
                        graph, if_node->ElseList(), SRegionKind::IfElse, region_id, node, "else"
                    );
                }
                return;
            }

            if (auto *while_node = node->dyn_cast< SWhile >()) {
                BuildRegionForSeq(
                    graph, while_node->BodyList(), SRegionKind::WhileBody, region_id, node,
                    "while"
                );
                return;
            }

            if (auto *do_node = node->dyn_cast< SDoWhile >()) {
                BuildRegionForSeq(
                    graph, do_node->BodyList(), SRegionKind::DoWhileBody, region_id, node,
                    "do_while"
                );
                return;
            }

            if (auto *for_node = node->dyn_cast< SFor >()) {
                BuildRegionForSeq(
                    graph, for_node->BodyList(), SRegionKind::ForBody, region_id, node, "for"
                );
                return;
            }

            if (auto *switch_node = node->dyn_cast< SSwitch >()) {
                size_t case_index = 0;
                for (const auto &case_node : switch_node->Cases()) {
                    BuildRegionForSeq(
                        graph, case_node.body_list, SRegionKind::SwitchCase, region_id, node,
                        "case_" + std::to_string(case_index++)
                    );
                }
                if (!switch_node->DefaultBodyList().empty()) {
                    BuildRegionForSeq(
                        graph, switch_node->DefaultBodyList(), SRegionKind::SwitchDefault,
                        region_id, node, "default"
                    );
                }
            }
        }

        void BuildRegionForSeq(
            SRegionGraph &graph, const std::vector< SNode * > &seq, SRegionKind kind,
            size_t parent, const SNode *owner, std::string name
        ) {
            size_t region_id = AddRegion(graph, kind, parent, owner, std::move(name));
            for (SNode *node : seq) { VisitNode(graph, node, region_id); }
        }

        std::unordered_map< std::string, unsigned >
        ComputeSubtreePayloads(SRegionGraph &graph, size_t region_id) {
            auto &region             = graph.regions[region_id];
            region.subtree_owned_ops = region.direct_owned_ops;
            for (size_t child_id : region.children) {
                auto child_ops = ComputeSubtreePayloads(graph, child_id);
                Accumulate(region.subtree_owned_ops, child_ops);
                region.subtree_snodes += graph.regions[child_id].subtree_snodes;
                region.opaque_compound_payloads +=
                    graph.regions[child_id].opaque_compound_payloads;
            }
            return region.subtree_owned_ops;
        }

        struct SOwnedOpPlacement
        {
            size_t region = SRegionNode::kNone;
            StmtOrigin origin;
        };

        std::string FormatBool(bool value) { return value ? "true" : "false"; }

        std::string FormatRegions(const std::vector< SOwnedOpPlacement > &placements) {
            std::vector< size_t > regions;
            regions.reserve(placements.size());
            for (const auto &placement : placements) { regions.push_back(placement.region); }
            std::sort(regions.begin(), regions.end());
            regions.erase(std::unique(regions.begin(), regions.end()), regions.end());

            std::string result;
            for (size_t i = 0; i < regions.size(); ++i) {
                if (i != 0) { result += ","; }
                result += std::to_string(regions[i]);
            }
            return result;
        }

        bool HasMultipleRegions(const std::vector< SOwnedOpPlacement > &placements) {
            if (placements.empty()) { return false; }
            size_t first_region = placements.front().region;
            return std::any_of(
                placements.begin(), placements.end(), [&](const SOwnedOpPlacement &placement) {
                    return placement.region != first_region;
                }
            );
        }

        const StmtOrigin *
        FirstNonCloneableOrigin(const std::vector< SOwnedOpPlacement > &placements) {
            for (const auto &placement : placements) {
                if (!placement.origin.cloneable) { return &placement.origin; }
            }
            return nullptr;
        }

        std::string DescribeIllegalClone(
            const std::string &op_key, const std::vector< SOwnedOpPlacement > &placements,
            const StmtOrigin &origin
        ) {
            return "illegal cross-region cloned payload operation " + op_key + " kind="
                + PayloadKindName(origin.kind) + " regions=" + FormatRegions(placements)
                + " movable=" + FormatBool(origin.movable) + " cloneable="
                + FormatBool(origin.cloneable) + " may_call=" + FormatBool(origin.may_call)
                + " may_store=" + FormatBool(origin.may_store)
                + " may_volatile=" + FormatBool(origin.may_volatile)
                + " contains_internal_control=" + FormatBool(origin.contains_internal_control);
        }

        const char *SRewriteActionName(SRewriteAction action) {
            switch (action) {
                case SRewriteAction::Move:
                    return "move";
                case SRewriteAction::Clone:
                    return "clone";
            }
            return "unknown";
        }

        bool OriginPermittedByOptions(
            const StmtOrigin &origin, const SRewriteLegalityOptions &options
        ) {
            if (origin.may_call && !options.allow_calls) { return false; }
            if (origin.may_store && !options.allow_stores) { return false; }
            if (origin.may_volatile && !options.allow_volatile) { return false; }
            if (origin.contains_internal_control && !options.allow_internal_control) {
                return false;
            }
            return true;
        }

        bool OriginCanMove(const StmtOrigin &origin, const SRewriteLegalityOptions &options) {
            return origin.movable && OriginPermittedByOptions(origin, options);
        }

        bool OriginCanClone(const StmtOrigin &origin, const SRewriteLegalityOptions &options) {
            return (origin.cloneable || OriginPermittedByOptions(origin, options))
                && origin.movable;
        }

        std::string DescribeRewriteRejection(SRewriteAction action, const StmtOrigin &origin) {
            return std::string("cannot ") + SRewriteActionName(action) + " payload operation "
                + StmtOriginKey(origin) + " kind=" + PayloadKindName(origin.kind) + " movable="
                + FormatBool(origin.movable) + " cloneable=" + FormatBool(origin.cloneable)
                + " may_call=" + FormatBool(origin.may_call)
                + " may_store=" + FormatBool(origin.may_store)
                + " may_volatile=" + FormatBool(origin.may_volatile)
                + " contains_internal_control=" + FormatBool(origin.contains_internal_control);
        }

        void ValidateOriginRewriteLegality(
            const StmtOrigin &origin, SRewriteAction action,
            const SRewriteLegalityOptions &options, SRewriteLegalityReport &report
        ) {
            if (!origin.primary || !IsPayloadCarrierKind(origin.kind)) { return; }

            ++report.payload_origins;
            bool allowed = action == SRewriteAction::Clone ? OriginCanClone(origin, options)
                                                           : OriginCanMove(origin, options);
            if (allowed) { return; }

            ++report.rejected_payload_origins;
            AddDiagnostic(report.diagnostics, DescribeRewriteRejection(action, origin));
        }

        void ValidateNodeRewriteLegality(
            const SNode &node, SRewriteAction action, const SRewriteLegalityOptions &options,
            SRewriteLegalityReport &report
        ) {
            for (const auto &origin : node.Origins()) {
                ValidateOriginRewriteLegality(origin, action, options, report);
            }
            node.for_each_child([&](SNode *child) {
                if (child) { ValidateNodeRewriteLegality(*child, action, options, report); }
            });
        }

    } // namespace

    SRegionGraph BuildSNodeRegionGraph(const std::vector< SNode * > &root) {
        SRegionGraph graph;
        graph.root = AddRegion(graph, SRegionKind::Root, SRegionNode::kNone, nullptr, "root");
        for (SNode *node : root) { VisitNode(graph, node, graph.root); }
        if (!graph.empty()) { ComputeSubtreePayloads(graph, graph.root); }
        return graph;
    }

    SRegionValidationReport ValidateSNodeRegionGraph(const SRegionGraph &graph) {
        SRegionValidationReport report;
        report.regions = graph.regions.size();

        if (graph.empty()) {
            AddDiagnostic(report.diagnostics, "region graph is empty");
            return report;
        }
        if (graph.root >= graph.regions.size()) {
            AddDiagnostic(report.diagnostics, "region graph root is outside region range");
            return report;
        }
        if (graph.regions[graph.root].parent != SRegionNode::kNone) {
            AddDiagnostic(report.diagnostics, "region graph root has a parent");
        }

        std::unordered_map< std::string, unsigned > global_owned_ops;
        std::unordered_set< size_t > child_refs;
        for (const auto &region : graph.regions) {
            if (region.id >= graph.regions.size()) {
                AddDiagnostic(report.diagnostics, "region has id outside region range");
                continue;
            }
            if (graph.regions[region.id].id != region.id) {
                AddDiagnostic(report.diagnostics, "region id does not match storage slot");
            }

            if (region.id != graph.root) {
                if (region.parent >= graph.regions.size()) {
                    AddDiagnostic(
                        report.diagnostics,
                        "region " + std::to_string(region.id) + " has invalid parent "
                            + std::to_string(region.parent)
                    );
                } else {
                    const auto &parent = graph.regions[region.parent];
                    if (std::find(parent.children.begin(), parent.children.end(), region.id)
                        == parent.children.end())
                    {
                        AddDiagnostic(
                            report.diagnostics,
                            "region " + std::to_string(region.id)
                                + " is not listed by its parent"
                        );
                    }
                }
            }

            for (size_t child_id : region.children) {
                if (child_id >= graph.regions.size()) {
                    AddDiagnostic(
                        report.diagnostics,
                        "region " + std::to_string(region.id) + " has invalid child "
                            + std::to_string(child_id)
                    );
                    continue;
                }
                if (graph.regions[child_id].parent != region.id) {
                    AddDiagnostic(
                        report.diagnostics,
                        "region " + std::to_string(region.id) + " child "
                            + std::to_string(child_id) + " has mismatched parent"
                    );
                }
                if (!child_refs.insert(child_id).second) {
                    AddDiagnostic(
                        report.diagnostics,
                        "region " + std::to_string(child_id)
                            + " is referenced by multiple parents"
                    );
                }
            }

            std::unordered_map< std::string, unsigned > recomputed = region.direct_owned_ops;
            size_t recomputed_snodes                               = region.direct_snodes;
            size_t recomputed_opaque                               = 0;
            for (size_t child_id : region.children) {
                if (child_id >= graph.regions.size()) { continue; }
                Accumulate(recomputed, graph.regions[child_id].subtree_owned_ops);
                recomputed_snodes += graph.regions[child_id].subtree_snodes;
                recomputed_opaque += graph.regions[child_id].opaque_compound_payloads;
            }
            if (region.subtree_owned_ops != recomputed) {
                AddDiagnostic(
                    report.diagnostics,
                    "region " + std::to_string(region.id)
                        + " has inconsistent subtree payload ownership"
                );
            }
            if (region.subtree_snodes != recomputed_snodes) {
                AddDiagnostic(
                    report.diagnostics,
                    "region " + std::to_string(region.id)
                        + " has inconsistent subtree SNode count"
                );
            }
            if (region.opaque_compound_payloads < recomputed_opaque) {
                AddDiagnostic(
                    report.diagnostics,
                    "region " + std::to_string(region.id)
                        + " has inconsistent opaque compound payload count"
                );
            }

            Accumulate(global_owned_ops, region.direct_owned_ops);
        }

        for (size_t id = 0; id < graph.regions.size(); ++id) {
            if (id == graph.root) { continue; }
            if (!child_refs.contains(id)) {
                AddDiagnostic(report.diagnostics, "non-root region is unreachable from root");
            }
        }

        for (const auto &[_, count] : global_owned_ops) {
            report.owned_ops += count;
            if (count > 1) { ++report.duplicated_owned_ops; }
        }
        report.opaque_compound_payloads = graph.regions[graph.root].opaque_compound_payloads;
        return report;
    }

    SRegionLegalityReport ValidateSNodeRegionLegality(const SRegionGraph &graph) {
        SRegionLegalityReport report;

        if (graph.empty()) {
            AddDiagnostic(report.diagnostics, "region graph is empty");
            return report;
        }

        std::unordered_map< std::string, std::vector< SOwnedOpPlacement > > placements_by_op;
        for (const auto &region : graph.regions) {
            for (const auto &origin : region.direct_origins) {
                ++report.owned_ops;
                placements_by_op[StmtOriginKey(origin)].push_back(
                    SOwnedOpPlacement{ region.id, origin }
                );
            }
        }

        for (const auto &[op_key, placements] : placements_by_op) {
            if (placements.size() <= 1) { continue; }

            ++report.cloned_owned_ops;
            if (!HasMultipleRegions(placements)) { continue; }

            ++report.cross_region_cloned_ops;
            if (const StmtOrigin *non_cloneable = FirstNonCloneableOrigin(placements)) {
                ++report.illegal_cloned_ops;
                AddDiagnostic(
                    report.diagnostics, DescribeIllegalClone(op_key, placements, *non_cloneable)
                );
            }
        }

        std::sort(report.diagnostics.begin(), report.diagnostics.end());
        return report;
    }

    SRewriteLegalityReport ValidateSNodeRewriteLegality(
        const SNode &node, SRewriteAction action, const SRewriteLegalityOptions &options
    ) {
        SRewriteLegalityReport report;
        report.action = action;
        ValidateNodeRewriteLegality(node, action, options, report);
        std::sort(report.diagnostics.begin(), report.diagnostics.end());
        return report;
    }

    SRewriteLegalityReport ValidateSNodeRewriteLegality(
        const std::vector< SNode * > &seq, SRewriteAction action,
        const SRewriteLegalityOptions &options
    ) {
        SRewriteLegalityReport report;
        report.action = action;
        for (const SNode *node : seq) {
            if (node) { ValidateNodeRewriteLegality(*node, action, options, report); }
        }
        std::sort(report.diagnostics.begin(), report.diagnostics.end());
        return report;
    }

    bool CanCloneSNodePayloads(const SNode &node, const SRewriteLegalityOptions &options) {
        return ValidateSNodeRewriteLegality(node, SRewriteAction::Clone, options).ok();
    }

    bool CanCloneSNodePayloads(
        const std::vector< SNode * > &seq, const SRewriteLegalityOptions &options
    ) {
        return ValidateSNodeRewriteLegality(seq, SRewriteAction::Clone, options).ok();
    }

    bool CanMoveSNodePayloads(const SNode &node, const SRewriteLegalityOptions &options) {
        return ValidateSNodeRewriteLegality(node, SRewriteAction::Move, options).ok();
    }

    bool CanMoveSNodePayloads(
        const std::vector< SNode * > &seq, const SRewriteLegalityOptions &options
    ) {
        return ValidateSNodeRewriteLegality(seq, SRewriteAction::Move, options).ok();
    }

} // namespace patchestry::ast
