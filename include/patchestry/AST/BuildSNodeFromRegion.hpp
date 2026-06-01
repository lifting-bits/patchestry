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
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace clang {
    class ASTContext;
    class Expr;
} // namespace clang

namespace patchestry::ghidra {
    struct Function;
    struct RegionNode;
} // namespace patchestry::ghidra

namespace patchestry::ast {

    /// Stage 2 translator: turn Ghidra's structured region tree
    /// (Function::region) into the same shape CFGStructure produces.
    /// Any unimplemented kind makes Translate return nullopt and the
    /// caller falls back to CFGStructure for that whole function.
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
        /// nullopt = unimplemented kind or child failure; empty vector =
        /// legitimate empty body.
        using SNodeSeq = std::optional< std::vector< SNode * > >;

        SNodeSeq Translate(const ghidra::RegionNode &node);

        SNodeSeq TranslatePlain(const ghidra::RegionNode &node);

        SNodeSeq TranslateChildSeq(const ghidra::RegionNode &node);

        SNodeSeq TranslateSwitch(const ghidra::RegionNode &node);

        /// Fallback for `switch` whose dispatcher is a `multigoto`:
        /// emit each plain child with its raw terminal so
        /// NormalizeRawControlFlow can lift them.
        SNodeSeq TranslateMultigotoDispatch(const ghidra::RegionNode &mg);

        /// Cond head of an if/loop. `pre` runs before the if/while
        /// (re-executed each iteration for loops); `cond_idx` is the
        /// tail CNode (representative for compound conditions);
        /// `succs` are the {false, true} exits.
        struct CondHead {
            std::vector< SNode * > pre;
            size_t cond_idx;
            size_t entry_idx;
            clang::Expr *branch_cond = nullptr;
            std::vector< size_t > succs;
            std::string original_label;
        };

        /// Cond head for properif/ifelse/whiledo. Accepts `plain`
        /// (single CBRANCH) or `list` (pre-cond blocks ending in CBRANCH);
        /// nullopt on any unsupported shape.
        std::optional< CondHead > TranslateCondHead(
            const ghidra::RegionNode &cond_struct);

        /// Translate a Ghidra BlockCondition into a short-circuit cond head.
        std::optional< CondHead > TranslateConditionHead(
            const ghidra::RegionNode &cond_struct);

        SNodeSeq TranslateProperIf(const ghidra::RegionNode &node);

        /// Emits SIfThenElse(cond, SGoto(target_label), nullptr) +
        /// pre-cond stmts. Explicit goto target (when present) picks
        /// polarity; older JSON falls back to successor-order heuristic.
        SNodeSeq TranslateIfGoto(const ghidra::RegionNode &node);

        SNodeSeq TranslateIfElse(const ghidra::RegionNode &node);

        /// Fast path (plain cond, empty pre-test) emits `while (cond) body`;
        /// slow path emits `while(1) { pre; if (!cond) break; body }` so
        /// pre-cond computation re-runs every iteration.
        SNodeSeq TranslateWhileDo(const ghidra::RegionNode &node);

        /// CBRANCH-carrying tail is the rightmost plain leaf; its taken
        /// arm must equal the body entry. Self-loop is the tail==entry case.
        SNodeSeq TranslateDoWhile(const ghidra::RegionNode &node);

        /// Emits `while(1) body`; break/continue exits are recovered later
        /// by ConvertGotoToBreakContinue.
        SNodeSeq TranslateInfLoop(const ghidra::RegionNode &node);

        SNodeSeq TranslateGoto(const ghidra::RegionNode &node);

        std::optional< size_t > FindCNode(const std::string &block_label) const;

        /// Populate `shared_plain_blocks_` with CNode ids named by more
        /// than one `plain` leaf — Ghidra's region tree is DAG-shaped so
        /// shared tails would duplicate side effects without this prescan.
        void PrescanSharedPlainBlocks();

        std::optional< std::string >
        FirstBlockLabel(const ghidra::RegionNode &node) const;

        /// When `node` is the body of an enclosing `properif`, return the
        /// CNode id of the properif's fallthrough sibling. Used by
        /// TranslateIfGoto to detect the `properif(cond, ifgoto)` shape
        /// where succs[1] is in-region fallthrough and the goto sits on succs[0].
        std::optional< size_t >
        ProperIfFallthroughBlock(const ghidra::RegionNode &node) const;

        /// True when `idx` is an enclosing loop header. Used to map
        /// switch case targets jumping to the header into `continue`.
        bool IsActiveLoopHeader(size_t idx) const;

        /// True when `idx` is the innermost break target. Innermost matters
        /// for switch-in-loop: C `break` exits the switch, not the loop.
        bool IsInnermostBreakTarget(size_t idx) const;

        /// Build the most structured exit for a goto target under the
        /// current loop/switch scopes; falls back to SGoto when neither
        /// break nor continue applies.
        SNode *MakeStructuredExitNode(
            std::optional< int > goto_type,
            std::optional< size_t > target_idx,
            std::string_view target_label
        );

        struct RegionBoundary {
            std::optional< size_t > entry;
            std::unordered_set< size_t > blocks;
            std::vector< std::pair< size_t, size_t > > external_entries;
            std::vector< std::pair< size_t, size_t > > exit_edges;
            std::unordered_set< size_t > exit_targets;
        };

        /// CFG boundary for a region subtree. Verifier/repair aid only;
        /// multi-exit and irreducible-entry regions stay as explicit gotos.
        std::optional< RegionBoundary >
        AnalyzeRegionBoundary(const ghidra::RegionNode &node) const;

        bool CheckRegionBoundaryResolvable(const ghidra::RegionNode &node) const;

        const ghidra::Function &function_;
        CGraph &graph_;
        SNodeFactory &factory_;
        clang::ASTContext &ctx_;
        // Coverage check: CNodes named by `plain` leaves during a TryBuild;
        // uncovered CNodes fail loudly instead of silently dropping stmts.
        std::unordered_set< size_t > covered_cnodes_;
        std::unordered_set< size_t > shared_plain_blocks_;
        std::vector< size_t > active_loop_headers_;
        // Innermost-first; loops push their exit, switches push the merge.
        std::vector< size_t > active_break_targets_;
    };

} // namespace patchestry::ast
