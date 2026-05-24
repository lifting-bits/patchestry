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
    struct RegionNode;
} // namespace patchestry::ghidra

namespace patchestry::ast {

    /// Stage 2 translator: turn Ghidra's structured region tree
    /// (Function::region) into the same shape CFGStructure produces
    /// — populated per-CNode SNode sequences in CGraph::nodes[*].structured.
    ///
    /// Phases land kind-by-kind.  Any unimplemented kind makes Translate
    /// return nullopt; TryBuild propagates the failure as `false` and the
    /// caller falls back to CFGStructure for that whole function.  Partial
    /// coverage is therefore always safe.
    class BuildSNodeFromRegion {
      public:
        BuildSNodeFromRegion(
            const ghidra::Function &function, CGraph &graph,
            SNodeFactory &factory, clang::ASTContext &ctx
        );

        /// Attempt to seed CGraph nodes from function.region.
        /// Returns true on success; on false the graph is left untouched
        /// and the caller falls back to CFGStructure.
        bool TryBuild();

      private:
        /// Result of translating a region subtree: an ordered SNode
        /// sequence.  nullopt = the kind is unimplemented (or a child
        /// failed); empty vector = legitimate empty body (e.g., a `plain`
        /// block whose CNode has no clang statements).
        using SNodeSeq = std::optional< std::vector< SNode * > >;

        /// Dispatch on RegionNode::kind, returning the SNode subtree
        /// (or nullopt for any unimplemented kind).
        SNodeSeq Translate(const ghidra::RegionNode &node);

        /// `plain` leaf: resolve `node.block` to a CNode, wrap each
        /// clang::Stmt in SStmt, return them as a sequence.
        SNodeSeq TranslatePlain(const ghidra::RegionNode &node);

        /// `graph`/`list` interior: translate every child and concatenate
        /// the resulting sequences.  Any child failure fails the whole
        /// translation.
        SNodeSeq TranslateChildSeq(const ghidra::RegionNode &node);

        /// `switch`: child[0] is the dispatcher block (BRANCHIND), and
        /// children[1..N] are arm bodies in dispatch order.  Discriminant
        /// + case-value labeling come from the dispatcher CNode's
        /// `branch_cond` + `switch_cases` (already populated by
        /// CGraphBuilder from the per-op SwitchCase entries that carry
        /// `is_default` flags recovered from Ghidra's ClangCaseToken markup).
        SNodeSeq TranslateSwitch(const ghidra::RegionNode &node);

        /// Result of resolving the cond head of an if/loop construct.
        /// `pre` is the prefix SNode sequence to emit before the
        /// if/while node (it includes the cond block's own stmts);
        /// `cond_idx` is the CNode whose `branch_cond`/`succs` drive
        /// the condition.  Helper unifies the `plain` and `list`
        /// shapes Ghidra produces for the cond child.
        struct CondHead {
            std::vector< SNode * > pre;
            size_t cond_idx;
        };

        /// Translate the cond head of a properif/ifelse/whiledo.
        /// Accepts kind `plain` (single CBRANCH block) or `list` (one
        /// or more pre-cond blocks ending in a CBRANCH block).
        ///
        /// For `list`, applies an Option-(c) safety check: if any
        /// pre-cond block label is referenced by a goto originating
        /// OUTSIDE the `enclosing` if/loop subtree, returns nullopt.
        /// The downstream post-pass `DuplicateStackGuardReturnTargets`
        /// would otherwise duplicate the SLabel wrap inside the
        /// construct and leave the outside reference dangling.
        std::optional< CondHead > TranslateCondHead(
            const ghidra::RegionNode &cond_struct,
            const ghidra::RegionNode &enclosing);

        /// `properif`: child[0] is the cond block (CBRANCH terminator)
        /// and child[1] is the then-body subtree.  The if condition is
        /// the cond CNode's `branch_cond`, negated if the body entry
        /// matches the not-taken successor.
        SNodeSeq TranslateProperIf(const ghidra::RegionNode &node);

        /// `ifgoto`: child[0] is the cond block; the taken arm exits
        /// the structure with a goto to the cond's taken successor.
        /// Emits SIfThenElse(cond, SGoto(target_label), nullptr).
        SNodeSeq TranslateIfGoto(const ghidra::RegionNode &node);

        /// `ifelse`: child[0] is the cond block, child[1]/child[2] are
        /// the two arms.  Match each arm's first plain-leaf entry
        /// against cond.succs[0/1] to pick which is then vs else
        /// without negating branch_cond.
        SNodeSeq TranslateIfElse(const ghidra::RegionNode &node);

        /// `whiledo`: child[0] = cond block (CBRANCH at end), child[1] =
        /// body subtree.  Build SWhile(cond, body) with polarity
        /// determined by which succ matches the body entry.  Falls back
        /// when the cond block has non-terminal stmts (those would have
        /// to run before each iteration's test — a more complex
        /// transformation not handled in Phase 5).
        SNodeSeq TranslateWhileDo(const ghidra::RegionNode &node);

        /// `dowhile`: child[0] = the loop block (self-loop with stmts
        /// and CBRANCH).  Body = the block's stmts; cond = the block's
        /// branch_cond.  Falls back when the block isn't a self-loop
        /// or when the tree has more than one child.
        SNodeSeq TranslateDoWhile(const ghidra::RegionNode &node);

        /// `infloop`: child[0] = body subtree (no exit condition in the
        /// region tree — break/continue exits are handled by the
        /// post-pass ConvertGotoToBreakContinue).  Build
        /// SWhile(IntegerLiteral(1), body) → emitted as `while(1)`.
        /// This is the issue #259 fix path: Ghidra emits whiledo for
        /// that function, but bloodview region data exists; the
        /// LIT fixture predates region serialization and will fall
        /// back until re-extracted.
        SNodeSeq TranslateInfLoop(const ghidra::RegionNode &node);

        /// `goto`: child[0] is the source block (plain leaf) whose
        /// terminal is an unstructured unconditional branch.  Append
        /// SGoto(target_label) where target = source block's sole
        /// successor; the source block's pre-branch stmts emit as the
        /// prefix.
        SNodeSeq TranslateGoto(const ghidra::RegionNode &node);

        /// Resolve a Ghidra block label to a CNode index in the graph.
        std::optional< size_t > FindCNode(const std::string &block_label) const;

        /// Option-(c) safety check.  Returns true when at least one
        /// pre-cond block label of `cond_struct` (kind `list`) is
        /// referenced by a goto whose source region sits OUTSIDE the
        /// `enclosing` if/loop subtree.  Used by TranslateCondHead to
        /// reject shapes that would crash DuplicateStackGuardReturnTargets
        /// after SLabel duplication.
        bool PreCondLabelsReferencedFromOutside(
            const ghidra::RegionNode &cond_struct,
            const ghidra::RegionNode &enclosing) const;

        /// Recursively check pointer identity: is `needle` the same
        /// node as `subtree` or any of its descendants?  Safe because
        /// the region tree is value-owned and we never reallocate
        /// children during a TryBuild call.
        bool IsAddressInSubtree(
            const ghidra::RegionNode *needle,
            const ghidra::RegionNode &subtree) const;

        /// Walk a region subtree to its first `plain` leaf and return
        /// the block label.  Used to associate a `switch` arm body with
        /// its dispatcher successor.
        std::optional< std::string >
        FirstBlockLabel(const ghidra::RegionNode &node) const;

        const ghidra::Function &function_;
        CGraph &graph_;
        SNodeFactory &factory_;
        clang::ASTContext &ctx_;
        // CNode ids referenced by `plain` leaves during one TryBuild
        // call.  Reset at TryBuild entry.  Used by the coverage check
        // — if Ghidra's region tree doesn't account for every CNode
        // CGraphBuilder produced, fail loudly so we fall back instead
        // of silently dropping statements from uncovered blocks.
        std::unordered_set< size_t > covered_cnodes_;
    };

} // namespace patchestry::ast
