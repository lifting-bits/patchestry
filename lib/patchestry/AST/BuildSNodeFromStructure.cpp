/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/BuildSNodeFromStructure.hpp>

#include <patchestry/AST/CGraph.hpp>
#include <patchestry/AST/SNode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

#include <clang/AST/ASTContext.h>

#include <utility>

namespace patchestry::ast {

    BuildSNodeFromStructure::BuildSNodeFromStructure(
        const ghidra::Function &function, CGraph &graph,
        SNodeFactory &factory, clang::ASTContext &ctx
    )
        : function_(function), graph_(graph), factory_(factory), ctx_(ctx) {}

    bool BuildSNodeFromStructure::TryBuild() {
        if (!function_.structure.has_value() || graph_.nodes.empty()) {
            return false;
        }

        covered_cnodes_.clear();
        auto body = Translate(*function_.structure);
        if (!body.has_value()) {
            return false;
        }

        // Coverage check: every CNode must be reachable from Ghidra's
        // structure tree.  CGraphBuilder and Ghidra may disagree on
        // block boundaries (rare but possible — e.g., unreachable
        // blocks).  Silently dropping uncovered CNodes would lose
        // statements; per loud-failure, we fall back instead.
        if (covered_cnodes_.size() != graph_.nodes.size()) {
            LOG(WARNING) << "Ghidra structure tree covers "
                         << covered_cnodes_.size() << "/" << graph_.nodes.size()
                         << " CNodes for " << function_.name
                         << "; falling back to CFGStructure\n";
            return false;
        }

        // Seed the whole function: collapse every CNode into the entry,
        // with the translated tree as its structured sequence.  The post-
        // pass chain in ASTConsumer walks the SNode tree directly and is
        // agnostic to which CNode owns it.  We put the entry first so
        // IdentifyInternal picks it as the representative.
        std::vector< size_t > ids;
        ids.reserve(graph_.nodes.size());
        ids.push_back(graph_.entry);
        for (const auto &n : graph_.nodes) {
            if (n.id != graph_.entry) {
                ids.push_back(n.id);
            }
        }

        graph_.IdentifyInternal(ids, CNode::BlockType::kSequence, std::move(*body));
        return true;
    }

    BuildSNodeFromStructure::SNodeSeq
    BuildSNodeFromStructure::Translate(const ghidra::StructureNode &node) {
        // Kind vocabulary (PcodeBlock::typeToName) lives on StructureNode
        // — see Ghidra/PcodeOperations.hpp:243.  Phase 1 handles the
        // straight-line trio; everything else falls back per-function.
        if (node.kind == "plain") {
            return TranslatePlain(node);
        }
        if (node.kind == "graph" || node.kind == "list") {
            return TranslateChildSeq(node);
        }
        return std::nullopt;
    }

    BuildSNodeFromStructure::SNodeSeq
    BuildSNodeFromStructure::TranslatePlain(const ghidra::StructureNode &node) {
        if (!node.block.has_value()) {
            LOG(WARNING) << "plain structure node missing block label in "
                         << function_.name << "\n";
            return std::nullopt;
        }
        auto cnode_idx = FindCNode(*node.block);
        if (!cnode_idx.has_value()) {
            // Ghidra named a block we don't have a CNode for.  Treat as
            // failure so the caller falls back to CFGStructure.
            return std::nullopt;
        }
        if (!covered_cnodes_.insert(*cnode_idx).second) {
            // Same CNode named by two plain leaves — Ghidra's tree
            // shouldn't do this for plain blocks.  Bail loudly.
            LOG(WARNING) << "Ghidra structure tree references CNode "
                         << *cnode_idx << " twice in " << function_.name
                         << "; falling back to CFGStructure\n";
            return std::nullopt;
        }
        const auto &cnode = graph_.Node(*cnode_idx);
        std::vector< SNode * > out;
        out.reserve(cnode.stmts.size());
        for (clang::Stmt *stmt : cnode.stmts) {
            if (stmt != nullptr) {
                out.push_back(factory_.Make< SStmt >(stmt));
            }
        }
        return out;
    }

    BuildSNodeFromStructure::SNodeSeq
    BuildSNodeFromStructure::TranslateChildSeq(const ghidra::StructureNode &node) {
        std::vector< SNode * > out;
        for (const auto &child : node.children) {
            auto child_seq = Translate(child);
            if (!child_seq.has_value()) {
                return std::nullopt;
            }
            for (SNode *s : *child_seq) {
                out.push_back(s);
            }
        }
        return out;
    }

    std::optional< size_t >
    BuildSNodeFromStructure::FindCNode(const std::string &block_label) const {
        for (const auto &node : graph_.nodes) {
            if (node.source_key == block_label) {
                return node.id;
            }
        }
        return std::nullopt;
    }

} // namespace patchestry::ast
