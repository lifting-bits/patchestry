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
    /// return nullptr; TryBuild propagates the failure as `false` and the
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
        /// Dispatch on StructureNode::kind, returning the SNode subtree
        /// (or nullptr for any unimplemented kind).
        SNode *Translate(const ghidra::StructureNode &node);

        /// Resolve a Ghidra block label to a CNode index in the graph.
        std::optional< size_t > FindCNode(const std::string &block_label) const;

        const ghidra::Function &function_;
        CGraph &graph_;
        // Held for Phase 1+ kind handlers; unused while Translate is a stub.
        [[maybe_unused]] SNodeFactory &factory_;
        [[maybe_unused]] clang::ASTContext &ctx_;
    };

} // namespace patchestry::ast
