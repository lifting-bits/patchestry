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
#include <string>
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

        // Node IDs that appear as successors of collapsed nodes.
        // Precomputed per StructureInternal round so RuleBlockCat can
        // check label reachability in O(1) instead of scanning all nodes.
        std::unordered_set< size_t > collapsed_succ_targets_;

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
        void ComputeDominatorTree();
        void ComputePostDominatorTree();
        void NormalizeConditionPolarityIPdom();
        void CanonicalizeTopology();
        void OrderLoops();

        /// Try all collapse rules on every active node.
        /// Returns true if at least one rule fired.
        bool StructureInternal();

        // Collapse rules — each returns true if it matched and fired.

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

        /// True iff any collapsed node's structured SNode tree holds an
        /// SGoto targeting `target_label`.  Live-pred SGotos are NOT
        /// counted (the existing succs/preds graph already encodes
        /// them via HasSoleRealPredecessor).  Used by RuleBlockProperIf
        /// Case 1b/2b to refuse forwarder absorption when an earlier
        /// rule's collapsed tree would dangle after the forwarder's
        /// SLabel is discarded (#252).
        bool TargetHasCollapsedGotoRefs(std::string_view target_label) const;

        /// Wrap child SNode with node's prior content (structured or stmts)
        /// and label.  Used by all if/if-else/if-return rules.
        /// Returns the resulting SNode sequence.
        std::vector< SNode * > WrapWithPriorContent(size_t id, SNode *child);

        /// Check if node d is dominated by node root via idom_ chain.
        bool IsDominatedBy(size_t d, size_t root) const;

        /// Collect active nodes dominated by root but not by stop.
        /// Sorted by rpo_pos_.
        std::vector<size_t> CollectDomRegion(size_t root, size_t stop) const;

        // Helpers

        /// Spill a single CNode's stmts into SStmt siblings, optionally
        /// wrapped in an SLabel if the node has a label.
        /// When \p include_terminal is false the node's terminal stmt (goto /
        /// if-goto) is omitted — used for non-tail nodes in a sequential merge
        /// where the edge is absorbed by the merge.
        /// Returns the resulting SNode sequence (or the node's existing
        /// structured sequence if it was already structured).
        std::vector< SNode * > BuildLeafSNode(size_t id,
                                              bool include_terminal = true);

        /// Build a sequence from multiple node ids (each via BuildLeafSNode).
        std::vector< SNode * > BuildBodySNode(const std::vector<size_t> &ids);

        /// Build the body SNode sequence for a loop, excluding the header.
        /// Interior node terminals are stripped (edges absorbed by loop).
        /// Conditional interior nodes with exits outside the body get
        /// an if-goto to preserve the exit path.
        std::vector< SNode * > BuildLoopBodySNode(
                                  const std::vector<size_t> &body,
                                  size_t header_id,
                                  const std::unordered_set<size_t> &bodyset);

        /// Use TraceDAG to select the least-disruptive edge and mark it
        /// as a goto.  Returns true if an edge was selected and marked.
        bool SelectAndMarkGotoEdge();
    };

    /// Lift raw clang control flow embedded in opaque SStmt leaf SNodes
    /// into first-class SNodes so the SNode-layer cleanup passes can see
    /// it.  Payload shapes lifted (each handled by the shared recursive
    /// helper `NormalizeClangStmt`):
    ///   - `clang::GotoStmt`       -> `SGoto`
    ///   - `clang::LabelStmt`      -> `SLabel` (body normalized recursively)
    ///   - `clang::BreakStmt`      -> `SBreak`
    ///   - `clang::IfStmt`         -> `SIfThenElse` (arms normalized recursively)
    ///   - `clang::SwitchStmt`     -> `SSwitch` when the body matches the
    ///                                 switch-of-CaseStmt/DefaultStmt shape;
    ///                                 each case body normalized recursively,
    ///                                 fallthrough chains preserved.  A
    ///                                 non-matching switch is left as a
    ///                                 single SStmt.
    ///   - `clang::CompoundStmt`   -> decomposed into a sequence of SNodes
    ///                                 ONLY if it transitively contains a
    ///                                 goto or label; otherwise left as a
    ///                                 single opaque SStmt so emission keeps
    ///                                 its brace level.
    ///   - `clang::NullStmt` / null -> dropped.
    /// Any other clang::Stmt shape is left as a plain SStmt — never guess,
    /// never drop.  Goto/label pairing is preserved verbatim: no rename,
    /// no synthesis, no case-value width recompute.
    void NormalizeRawControlFlow(std::vector< SNode * > &root,
                                 SNodeFactory &factory,
                                 clang::ASTContext &ctx);

    /// Eliminate gotos whose target label immediately follows in the
    /// same sibling sequence.  Chases through SLabel→SStmt nesting to
    /// find deeply buried gotos (DeepTrailingStmt pattern).
    bool EliminateGotoToNextLabel(std::vector< SNode * > &root,
                                  SNodeFactory &factory,
                                  clang::ASTContext &ctx);

    /// Absorb unreferenced SLabel siblings after if-then (no else) into
    /// the else branch.  Prevents spurious fallthrough from the false path
    /// into code that was goto-only in the original CFG.
    bool AbsorbFallthroughIntoElse(std::vector< SNode * > &root,
                                   SNodeFactory &factory);

    /// Convert if(cond) goto L; stmts; L: patterns into if(!cond) { stmts }
    /// within a sibling sequence.  Only fires when label L has a single
    /// goto reference and no intermediate SLabel nodes exist between the
    /// goto and its target.  Recurses into all nested SNode bodies.
    bool ScopeifyIfGotos(std::vector< SNode * > &root, SNodeFactory &factory,
                         clang::ASTContext &ctx);

    /// Post-structuring cleanup: inline residual goto-to-label pairs.
    /// When an SGoto's target SLabel is a sibling referenced only by that
    /// goto, replace the goto with the label's body and drop the label.
    /// Returns true if any inlining was performed.
    bool InlineResidualGotos(std::vector< SNode * > &root, SNodeFactory &factory);

    /// Replace goto-to-break/continue: when a trailing clang::GotoStmt or
    /// SGoto targets an enclosing loop's exit label (→ break) or header
    /// label (→ continue), replace it with the structured equivalent.
    ///
    /// Returns true if any replacement was performed.
    bool ConvertGotoToBreakContinue(std::vector< SNode * > &root,
                                    SNodeFactory &factory);

    /// Replace goto-to-return: when an SStmt holding a clang::GotoStmt
    /// (or an SGoto SNode) targets a label whose body ends with a
    /// clang::ReturnStmt, replace the goto with the label's return stmts.
    /// Also handles gotos inside clang::IfStmt arms held by SStmt nodes
    /// — the most common residual pattern.
    /// Dead labels are cleaned up by a subsequent InlineResidualGotos pass.
    ///
    /// Returns true if any replacement was performed.
    bool ConvertGotoToReturn(std::vector< SNode * > &root, SNodeFactory &factory,
                             clang::ASTContext &ctx);

    /// Redirect gotos through labels whose body is only `goto other_label`.
    /// This removes label pass-through chains (alias elimination) and is run
    /// early, before the regular goto cleanup.
    bool CollapsePassThroughLabels(std::vector< SNode * > &root,
                                   SNodeFactory &factory);

    /// Fold empty label wrappers onto the following sibling and simplify
    /// empty if/else shells: drop side-effect-free `if (c) {}` and flip
    /// `if (c) {} else { B }` into `if (!c) { B }`.
    bool SimplifyEmptyControlFlow(std::vector< SNode * > &root,
                                  clang::ASTContext &ctx);

    /// Merge adjacent, nested, and else-if guarded gotos that target the
    /// same label, preserving short-circuit condition order.
    bool MergeRedundantGotoGuards(std::vector< SNode * > &root,
                                  SNodeFactory &factory,
                                  clang::ASTContext &ctx);

    /// Duplicate small, side-effect-contained label targets into switch
    /// case arms that end in `SGoto L`, making the case bodies goto-free.
    /// Handles goto-into-another-switch by recursively cloning the inner
    /// switch.  Refuses to clone subtrees containing labels or loops so
    /// that goto/label pairing stays consistent.  Outbound gotos from
    /// the clone are allowed — they reference an already-live label.
    ///
    /// Returns true if any duplication was performed.
    bool DuplicateSwitchCaseTargets(std::vector< SNode * > &root,
                                    SNodeFactory &factory);

    /// Fold guarded fallthrough-to-join patterns:
    ///
    ///   if (A && B && C) goto body; else goto join;
    /// body:
    ///   ...
    /// join:
    ///
    /// into:
    ///
    ///   if (A && B && C) { ... }
    /// join:
    ///
    /// The pass only moves a single-ref immediately-following label body whose
    /// tail falls through to the next direct label, and only when all other
    /// guard exits already target that join label.
    bool FoldGuardedFallthroughTargets(std::vector< SNode * > &root,
                                       SNodeFactory &factory,
                                       clang::ASTContext &ctx);

    /// Repair one-way gotos that enter a label nested inside a later
    /// structured if/else region by duplicating a small label body at the
    /// outer guard site and moving the skipped region into the guard's else.
    ///
    /// This is intentionally conservative: it only handles single-ref labels
    /// nested under if/else nodes, not loops/switches, and only when the label
    /// is the last child along the nested path so the cloned entry has the
    /// same fallthrough continuation as the original label entry.
    bool RepairCrossScopeLabelEntries(std::vector< SNode * > &root,
                                      SNodeFactory &factory,
                                      clang::ASTContext &ctx);

    /// Move single-ref label targets into switch case arms.
    ///
    /// This complements DuplicateSwitchCaseTargets for switch-local labels that
    /// contain calls and therefore must not be cloned.  The target label must
    /// have exactly one goto reference, no fallthrough predecessor, a bounded
    /// label-free body, and every path through the moved body must terminate.
    bool FoldSwitchLocalCaseTargets(std::vector< SNode * > &root,
                                    SNodeFactory &factory,
                                    clang::ASTContext &ctx);

    /// Duplicate small, terminating, label-free target bodies at ordinary
    /// residual goto sites.  This is the general form of the switch-case target
    /// duplication rule: it clones only bounded assignment/control tails that
    /// always terminate and rejects labels, loops, calls, and break/continue
    /// whose target would depend on the original lexical scope.
    ///
    /// Returns true if any duplication was performed.
    bool DuplicateSmallTerminatingTargets(std::vector< SNode * > &root,
                                          SNodeFactory &factory);

    /// Duplicate small producer->epilogue error tails at residual goto sites.
    /// A direct goto to the epilogue is cloned only when all local variables
    /// read by the epilogue are definitely assigned earlier in the same SNode
    /// sequence.  A goto to a producer label may clone the producer plus one
    /// terminal epilogue hop when the combined body is self-contained.
    bool DuplicateSmallEpilogueTargets(std::vector< SNode * > &root,
                                       SNodeFactory &factory,
                                       clang::ASTContext &ctx);

    /// Duplicate small switch-local fallthrough label tails into case arms.
    /// A clone is emitted as the label body plus an explicit SBreak, and only
    /// when the target is small, label-free, call-free, and has no local
    /// live-ins.  Rewrites are restricted to the same SSwitch so the cloned
    /// break targets the same structured scope as the original fallthrough.
    bool DuplicateSwitchFallthroughTargets(std::vector< SNode * > &root,
                                           SNodeFactory &factory,
                                           clang::ASTContext &ctx);

    /// Duplicate loop-local continue/latch label tails into goto sites.
    /// The pass only clones small, goto-free, label-free tails that end in
    /// continue, and only rewrites gotos inside the same loop body.  Nested
    /// loops are treated as separate continue scopes.
    bool DuplicateLoopContinueTargets(std::vector< SNode * > &root,
                                      SNodeFactory &factory,
                                      clang::ASTContext &ctx);

    /// Duplicate stack-protector return epilogues at residual goto sites.
    /// The cloned body may contain only stack_chk_fail-like calls, must
    /// terminate on every path, and is still gated by local live-in checks
    /// before being placed at a goto site.  Label wrappers are dropped from
    /// the clone so the pass does not duplicate label definitions.
    bool DuplicateStackGuardReturnTargets(std::vector< SNode * > &root,
                                          SNodeFactory &factory,
                                          clang::ASTContext &ctx);

    /// Duplicate small cleanup/logging return tails at residual goto sites.
    /// The pass follows bounded cleanup chains, drops cloned label/goto
    /// wrappers, and only clones direct calls that look like cleanup or
    /// diagnostic calls.  Local live-ins must be definitely available at
    /// the goto site.
    bool DuplicateCleanupReturnTargets(std::vector< SNode * > &root,
                                       SNodeFactory &factory,
                                       clang::ASTContext &ctx);

    /// Cross-scope version of InlineResidualGotos: when a goto's target
    /// label has exactly one reference, the label's body always terminates,
    /// and the label's preceding sibling also terminates (no fallthrough
    /// reaches it), move the body into the goto's slot and drop the label.
    /// Safe because no other goto observes the label, no fallthrough
    /// reaches it, and the moved body terminates.
    /// Returns true if any label was inlined.
    bool InlineCrossScopeSingleRef(std::vector< SNode * > &root,
                                   SNodeFactory &factory);

    /// Remove unreachable siblings that follow a terminating sibling.
    /// Preserves SLabel children (may be goto targets from other scopes).
    /// Bottom-up recursive.
    bool RemoveDeadSSeqChildren(std::vector< SNode * > &root);

} // namespace patchestry::ast
