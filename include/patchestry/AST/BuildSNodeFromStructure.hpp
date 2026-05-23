/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/CGraph.hpp>
#include <patchestry/AST/SNode.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace clang { class ASTContext; }

namespace patchestry::ghidra {
    struct Function;
    struct StructureNode;
} // namespace patchestry::ghidra

namespace patchestry::ast {

    /// Stage 2 translator: turn Ghidra's structured BlockGraph
    /// (Function::structure) into the same shape CFGStructure produces
    /// — populated per-CNode SNode sequences in CGraph::nodes[*].structured.
    ///
    /// Phases land kind-by-kind.  Any unimplemented kind makes Translate
    /// return nullopt; TryBuild propagates the failure as `false` and the
    /// caller falls back to CFGStructure for that whole function.  Partial
    /// coverage is therefore always safe.
    class BuildSNodeFromStructure {
      public:
        BuildSNodeFromStructure(
            const ghidra::Function &function, CGraph &graph,
            SNodeFactory &factory, clang::ASTContext &ctx
        );

        /// Attempt to seed CGraph nodes from function.structure.
        /// Returns true on success; on false the graph is left untouched
        /// and the caller falls back to CFGStructure.
        bool TryBuild();

      private:
        /// Result of translating a structure subtree: an ordered SNode
        /// sequence.  nullopt = the kind is unimplemented (or a child
        /// failed); empty vector = legitimate empty body (e.g., a `plain`
        /// block whose CNode has no clang statements).
        using SNodeSeq = std::optional< std::vector< SNode * > >;

        /// Dispatch on StructureNode::kind, returning the SNode subtree
        /// (or nullopt for any unimplemented kind).
        SNodeSeq Translate(const ghidra::StructureNode &node);

        /// `plain` leaf: resolve `node.block` to a CNode, wrap each
        /// clang::Stmt in SStmt, return them as a sequence.
        SNodeSeq TranslatePlain(const ghidra::StructureNode &node);

        /// `graph`/`list` interior: translate every child and concatenate
        /// the resulting sequences.  Any child failure fails the whole
        /// translation.
        SNodeSeq TranslateChildSeq(const ghidra::StructureNode &node);

        /// Resolve a Ghidra block label to a CNode index in the graph.
        std::optional< size_t > FindCNode(const std::string &block_label) const;

        const ghidra::Function &function_;
        CGraph &graph_;
        SNodeFactory &factory_;
        // Held for kind handlers that need it (currently unused);
        // remove the attribute when a handler consumes it.
        [[maybe_unused]] clang::ASTContext &ctx_;
        // CNode ids referenced by `plain` leaves during one TryBuild
        // call.  Reset at TryBuild entry.  Used by the coverage check
        // — if Ghidra's structure tree doesn't account for every CNode
        // CGraphBuilder produced, fail loudly so we fall back instead
        // of silently dropping statements from uncovered blocks.
        std::unordered_set< size_t > covered_cnodes_;
    };

} // namespace patchestry::ast
