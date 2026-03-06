/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/CfgBuilder.hpp>
#include <patchestry/AST/SNode.hpp>

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include <clang/AST/Expr.h>

namespace patchestry::ast {

    // -----------------------------------------------------------------------
    // detail:: namespace — internal but testable collapse graph types.
    // -----------------------------------------------------------------------
    namespace detail {

        /// A node in the collapsing graph.  Starts as a 1:1 mirror of CfgBlock.
        /// As rules fire, nodes get absorbed into StructuredNode wrappers and
        /// removed from the active set.
        struct CNode {
            size_t id;                          // original CfgBlock index
            std::vector<size_t> succs;          // outgoing edges (by CNode id)
            std::vector<size_t> preds;          // incoming edges (by CNode id)

            // Edge properties (indexed same as succs)
            std::vector<uint32_t> edge_flags;

            // The SNode produced when this node is collapsed (null = leaf)
            SNode *structured = nullptr;

            // Leaf payload: statements from the original CfgBlock
            std::vector<clang::Stmt *> stmts;
            clang::Expr *branch_cond = nullptr;
            bool is_conditional = false;

            // Switch case metadata (copied from CfgBlock; non-empty for switch blocks)
            std::vector< SwitchCaseEntry > switch_cases;

            // Set when absorbed into a parent structured node
            bool collapsed = false;

            // Flags for the collapse algorithm
            bool mark = false;
            int visit_count = 0;

            enum EdgeFlag : uint32_t {
                F_GOTO     = 1u << 0,
                F_BACK     = 1u << 1,
                F_LOOP_EXIT = 1u << 2,
            };

            bool isGotoOut(size_t i) const {
                return i < edge_flags.size() && (edge_flags[i] & F_GOTO);
            }
            bool isBackEdge(size_t i) const {
                return i < edge_flags.size() && (edge_flags[i] & F_BACK);
            }
            bool isDecisionOut(size_t i) const {
                return !isGotoOut(i) && !isBackEdge(i);
            }
            bool isSwitchOut() const { return succs.size() > 2; }

            size_t sizeIn() const { return preds.size(); }
            size_t sizeOut() const { return succs.size(); }

            void setGoto(size_t i) {
                if (i < edge_flags.size()) edge_flags[i] |= F_GOTO;
            }
        };

        /// The collapsing graph — lightweight mirror of the Cfg that supports
        /// collapsing blocks into structured nodes.
        struct CGraph {
            std::vector<CNode> nodes;
            size_t entry = 0;

            /// Active (uncollapsed) node ids
            std::vector<size_t> activeIds() const {
                std::vector<size_t> result;
                for (auto &n : nodes) {
                    if (!n.collapsed) result.push_back(n.id);
                }
                return result;
            }

            size_t activeCount() const {
                size_t c = 0;
                for (auto &n : nodes) {
                    if (!n.collapsed) ++c;
                }
                return c;
            }

            CNode &node(size_t id) { return nodes[id]; }
            const CNode &node(size_t id) const { return nodes[id]; }

            /// Remove an edge from the active graph
            void removeEdge(size_t from, size_t to) {
                auto &s = nodes[from].succs;
                auto &f = nodes[from].edge_flags;
                for (size_t i = 0; i < s.size(); ++i) {
                    if (s[i] == to) {
                        s.erase(s.begin() + static_cast<ptrdiff_t>(i));
                        f.erase(f.begin() + static_cast<ptrdiff_t>(i));
                        break;
                    }
                }
                auto &p = nodes[to].preds;
                p.erase(std::remove(p.begin(), p.end(), from), p.end());
            }

            /// Replace a set of nodes with a single new structured node.
            /// Returns the representative node id.
            size_t collapseNodes(const std::vector<size_t> &ids, SNode *snode);
        };

        /// Build the collapse graph from a Cfg.
        CGraph buildCGraph(const Cfg &cfg);

        /// Detect back-edges using DFS and mark them in the graph.
        void markBackEdges(CGraph &g);

    } // namespace detail

    /// Build a structured SNode tree directly from a Cfg using Ghidra-style
    /// iterative pattern-matching and collapse.
    ///
    /// This is an alternative to the SNodeBuilder + normalization-passes pipeline.
    /// Instead of building a flat goto-heavy SNode tree and then eliminating gotos,
    /// this algorithm structures the CFG *before* generating the SNode tree —
    /// producing better results for complex control flow.
    ///
    /// Algorithm (adapted from Ghidra's CollapseStructure):
    ///   1. Detect loops via back-edges (Tarjan-based).
    ///   2. Collapse logical AND/OR conditions.
    ///   3. Iteratively pattern-match and collapse:
    ///      - Sequential blocks → SSeq
    ///      - If / if-else → SIfThenElse
    ///      - While-do / do-while → SWhile / SDoWhile
    ///      - Infinite loops → SWhile(true, body)
    ///      - Switch statements → SSwitch
    ///   4. When stuck, select an edge to mark as goto (DAG heuristic).
    ///   5. Repeat until fully collapsed.
    ///   6. Post-transforms: while→for, break/continue insertion.
    SNode *collapseStructure(const Cfg &cfg, SNodeFactory &factory,
                             clang::ASTContext &ctx);

} // namespace patchestry::ast
