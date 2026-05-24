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
        /// or more pre-cond blocks ending in a CBRANCH block).  Returns
        /// nullopt on any unsupported shape (non-plain cond child of
        /// `list`, unresolvable block label, etc.).
        std::optional< CondHead > TranslateCondHead(
            const ghidra::RegionNode &cond_struct);

        /// `properif`: child[0] is the cond block (CBRANCH terminator)
        /// and child[1] is the then-body subtree.  The if condition is
        /// the cond CNode's `branch_cond`, negated if the body entry
        /// matches the not-taken successor.
        SNodeSeq TranslateProperIf(const ghidra::RegionNode &node);

        /// `ifgoto`: child[0] is the cond head (kind `plain` or
        /// nested `list`); the taken arm exits the structure with a
        /// goto to the cond's taken successor.  Emits
        /// SIfThenElse(cond, SGoto(target_label), nullptr) prefixed
        /// by any pre-cond stmts produced by TranslateCondHead.
        SNodeSeq TranslateIfGoto(const ghidra::RegionNode &node);

        /// `ifelse`: child[0] is the cond block, child[1]/child[2] are
        /// the two arms.  Match each arm's first plain-leaf entry
        /// against cond.succs[0/1] to pick which is then vs else
        /// without negating branch_cond.
        SNodeSeq TranslateIfElse(const ghidra::RegionNode &node);

        /// `whiledo`: child[0] = cond head (kind `plain` or `list`),
        /// child[1] = body subtree.  Fast path (plain cond with empty
        /// pre-test stmts) emits `while (cond) body`; slow path (list
        /// cond or plain cond with non-empty stmts) emits
        /// `while (1) { head.pre; if (!cond) break; body }` so the
        /// pre-cond computation re-executes every iteration.
        SNodeSeq TranslateWhileDo(const ghidra::RegionNode &node);

        /// `dowhile`: child[0] = body subtree (kind `plain` or a
        /// nested `list`).  The CBRANCH-carrying tail is the rightmost
        /// plain leaf; its taken-arm-succ must equal the body's first
        /// leaf (the loop entry).  Single-block self-loop is the
        /// degenerate case where tail == entry.
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
