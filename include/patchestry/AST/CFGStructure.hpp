/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/CGraph.hpp>
#include <patchestry/AST/SNode.hpp>

#include <list>
#include <unordered_set>
#include <vector>

namespace clang { class ASTContext; }

namespace patchestry::ast {

    /// Structuring works by repeatedly matching topological patterns in the
    /// CGraph and collapsing matched node sets into hierarchical SNode trees
    /// via CGraph::IdentifyInternal.  The algorithm terminates when only a
    /// single active node remains or when no further progress can be made
    /// (remaining nodes are emitted with goto-based control flow).
    class CFGStructure {
      public:
        CFGStructure(CGraph &g, SNodeFactory &factory, clang::ASTContext &ctx);

        /// Run full structuring: loops -> iterative rules -> goto fallback.
        void StructureAll();

      private:
        CGraph &graph_;
        SNodeFactory &factory_;
        clang::ASTContext &ctx_;
        std::list< LoopBody > loop_body_storage_;
        std::vector< LoopBody * > loop_order_;
        std::list< FloatingEdge > likely_goto_;

        // RPO position: rpo_pos_[n] = position of node n in RPO.
        // Lower value = earlier in RPO.  kNone if collapsed/unreachable.
        std::vector< size_t > rpo_pos_;

        // Dominator tree: idom_[n] = immediate dominator of node n.
        // idom_[entry] = entry.  idom_[n] = kNone if unreachable.
        std::vector< size_t > idom_;

        // Post-dominator tree: ipdom_[n] = immediate post-dominator of node n.
        // ipdom_[n] = kNone if no post-dominator (e.g. infinite loop).
        std::vector< size_t > ipdom_;

        // Phase methods
        void RecomputeRPO();
        void ComputeDominatorTree();
        void ComputePostDominatorTree();
        void NormalizeConditionPolarityIPdom();
        void OrderLoops();

        /// Try all collapse rules on every active node.
        /// Returns true if at least one rule fired.
        bool StructureInternal();

        // ---------------------------------------------------------------
        // Collapse rules — each returns true if it matched and fired.
        // ---------------------------------------------------------------

        /// Sequential merge: A->B where B has single predecessor A.
        bool RuleBlockCat(size_t id);

        bool RuleBlockProperIf(size_t id);
        bool RuleBlockIfElse(size_t id);
        bool RuleBlockWhileDo(size_t id);
        bool RuleBlockDoWhile(size_t id);
        bool RuleBlockInfLoop(size_t id);
        bool RuleBlockSwitch(size_t id);

        /// If-then-else where both arms terminate (return, no merge point).
        bool RuleBlockIfReturn(size_t id);

        /// Post-dominator-guided if-then: the merge point is the immediate
        /// post-dominator of A, and the body arm has sole pred = A.
        bool RuleBlockPostDomIf(size_t id);

        /// Check if the only "real" (non-collapsed, non-goto) predecessor
        /// of node_id is expected_pred.
        bool HasSoleRealPredecessor(size_t node_id, size_t expected_pred);

        /// Wrap child SNode with node's prior content (structured or stmts)
        /// and label.  Used by all if/if-else/if-return rules.
        SNode *WrapWithPriorContent(size_t id, SNode *child);

        /// Check if node d is dominated by node root via idom_ chain.
        bool IsDominatedBy(size_t d, size_t root) const;

        /// Collect active nodes dominated by root but not by stop.
        /// Sorted by rpo_pos_.
        std::vector<size_t> CollectDomRegion(size_t root, size_t stop) const;

        // ---------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------

        /// Wrap a single CNode's stmts + label in an SBlock (+ optional SLabel).
        /// When \p include_terminal is false the node's terminal stmt (goto /
        /// if-goto) is omitted — used for non-tail nodes in a sequential merge
        /// where the edge is absorbed by the merge.
        SNode *BuildLeafSNode(size_t id, bool include_terminal = true);

        /// Build an SSeq from multiple node ids (each wrapped via BuildLeafSNode).
        SNode *BuildBodySNode(const std::vector<size_t> &ids);

        /// Build the body SNode for a loop, excluding the header.
        /// Interior node terminals are stripped (edges absorbed by loop).
        /// Conditional interior nodes with exits outside the body get
        /// an if-goto to preserve the exit path.
        SNode *BuildLoopBodySNode(const std::vector<size_t> &body,
                                  size_t header_id,
                                  const std::unordered_set<size_t> &bodyset);

        /// Use TraceDAG to select the least-disruptive edge and mark it
        /// as a goto.  Returns true if an edge was selected and marked.
        bool SelectAndMarkGotoEdge();
    };

    /// Post-structuring cleanup: inline residual goto-to-label pairs.
    ///
    /// Walks the SNode tree looking for SGoto nodes whose target SLabel
    /// appears in the same parent SSeq and is only referenced by that one
    /// goto.  When found, the goto is replaced with the label's body and
    /// the label is removed.  This eliminates trivial gotos left over
    /// from structuring.
    ///
    /// Returns true if any inlining was performed.
    bool InlineResidualGotos(SNode *root, SNodeFactory &factory);

    /// Replace goto-to-break/continue: when a trailing clang::GotoStmt or
    /// SGoto targets an enclosing loop's exit label (→ break) or header
    /// label (→ continue), replace it with the structured equivalent.
    ///
    /// Returns true if any replacement was performed.
    bool ConvertGotoToBreakContinue(SNode *root, SNodeFactory &factory);

    /// Replace goto-to-return: when a trailing clang::GotoStmt in an SBlock
    /// (or an SGoto SNode) targets a label whose body ends with a
    /// clang::ReturnStmt, replace the goto with the label's return stmts.
    /// Dead labels are cleaned up by a subsequent InlineResidualGotos pass.
    ///
    /// Returns true if any replacement was performed.
    bool ConvertGotoToReturn(SNode *root, SNodeFactory &factory);

    /// Remove SLabel nodes from SSeq whose label has zero references
    /// (no SGoto and no clang::GotoStmt targets it).  The label's body
    /// is dropped — it is dead code reachable only via the removed label.
    bool RemoveUnreferencedLabels(SNode *root, SNodeFactory &factory);

} // namespace patchestry::ast
