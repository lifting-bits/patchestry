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
            const SNode &node, std::unordered_map< std::string, unsigned > &counts
        ) {
            for (const auto &origin : node.Origins()) {
                if (!origin.primary || !IsPayloadCarrierKind(origin.kind)) { continue; }
                ++counts[StmtOriginKey(origin)];
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
            AddPayloadOrigins(*node, region.direct_owned_ops);

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

} // namespace patchestry::ast
