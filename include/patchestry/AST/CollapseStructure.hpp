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
#include <limits>
#include <list>
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
            static constexpr size_t NONE = std::numeric_limits<size_t>::max();

            size_t id;                          // original CfgBlock index
            std::vector<size_t> succs;          // outgoing edges (by CNode id)
            std::vector<size_t> preds;          // incoming edges (by CNode id)

            // Edge properties (indexed same as succs)
            std::vector<uint32_t> edge_flags;

            // The SNode produced when this node is collapsed (null = leaf)
            SNode *structured = nullptr;

            // Leaf payload: statements from the original CfgBlock
            std::string label;                      // original CfgBlock label (for SLabel wrapping)
            std::vector<clang::Stmt *> stmts;
            clang::Expr *branch_cond = nullptr;
            bool is_conditional = false;

            // Switch case metadata (copied from CfgBlock; non-empty for switch blocks)
            std::vector< SwitchCaseEntry > switch_cases;

            // Set when absorbed into a parent structured node
            bool collapsed = false;
            size_t collapsed_into = NONE;  // representative node after collapse

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
            bool isLoopExit(size_t i) const {
                return i < edge_flags.size() && (edge_flags[i] & F_LOOP_EXIT);
            }
            bool isDecisionOut(size_t i) const {
                return !isGotoOut(i) && !isBackEdge(i);
            }
            /// Edges to trace in TraceDAG: not back-edges, not loop-exit edges.
            bool isLoopDAGOut(size_t i) const {
                return !isBackEdge(i) && !isLoopExit(i);
            }
            bool isSwitchOut() const { return succs.size() > 2; }

            size_t sizeIn() const { return preds.size(); }
            size_t sizeOut() const { return succs.size(); }

            void setGoto(size_t i) {
                if (i < edge_flags.size()) edge_flags[i] |= F_GOTO;
            }
            void setLoopExit(size_t i) {
                if (i < edge_flags.size()) edge_flags[i] |= F_LOOP_EXIT;
            }
            void clearLoopExit(size_t i) {
                if (i < edge_flags.size()) edge_flags[i] &= ~static_cast<uint32_t>(F_LOOP_EXIT);
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

        /// A detected natural loop: header, back-edge tails, nesting info.
        struct LoopBody {
            size_t head;                          // loop header CNode id
            std::vector<size_t> tails;            // CNode ids with back-edges to head
            int depth = 0;                        // nesting depth (deeper = higher number)
            int unique_count = 0;                 // count of head+tail nodes before reachability
            size_t exit_block = NONE;             // official exit CNode id, or NONE
            LoopBody *immed_container = nullptr;  // immediately containing loop

            static constexpr size_t NONE = std::numeric_limits<size_t>::max();

            explicit LoopBody(size_t h) : head(h) {}

            void addTail(size_t t) { tails.push_back(t); }

            /// Core body computation: backward reachability from tails to head.
            /// Populates `body` with all CNode ids in the loop. Sets CNode::mark.
            /// Caller MUST call clearMarks() after using the body.
            void findBase(CGraph &g, std::vector<size_t> &body) const;

            /// Set immed_container based on containment within other loops.
            void labelContainments(const CGraph &g, const std::vector<size_t> &body,
                                   const std::vector<LoopBody *> &looporder);

            /// Exit detection, tail ordering, body extension, exit edge labeling.
            /// (Declared here, implemented in Plan 02)
            void findExit(const CGraph &g, const std::vector<size_t> &body);
            void orderTails(const CGraph &g);
            void extend(CGraph &g, std::vector<size_t> &body) const;
            void labelExitEdges(CGraph &g, const std::vector<size_t> &body) const;

            /// Merge LoopBody records that share the same head.
            static void mergeIdenticalHeads(std::vector<LoopBody *> &looporder,
                                            std::list<LoopBody> &storage);

            /// Mark edges leaving the loop body as F_LOOP_EXIT.
            void setExitMarks(CGraph &g, const std::vector<size_t> &body) const;

            /// Clear F_LOOP_EXIT marks set by setExitMarks.
            void clearExitMarks(CGraph &g, const std::vector<size_t> &body) const;

            /// Check if this loop's head is still active (not collapsed).
            bool update(const CGraph &g) const;

            /// Sort innermost-first (higher depth = processed first).
            bool operator<(const LoopBody &other) const {
                return depth > other.depth;
            }
        };

        /// A lazily-resolved edge reference that survives collapse operations.
        struct FloatingEdge {
            size_t top_id;     // source CNode id
            size_t bottom_id;  // destination CNode id

            FloatingEdge(size_t t, size_t b) : top_id(t), bottom_id(b) {}

            /// Resolve to current edge in graph. Returns {source_id, edge_index},
            /// or {CNode::NONE, 0} if the edge no longer exists.
            std::pair<size_t, size_t> getCurrentEdge(const CGraph &g) const;
        };

        /// TraceDAG: traces DAG paths through the CGraph to identify
        /// the least-disruptive edges to mark as gotos.
        class TraceDAG {
            struct BlockTrace;

            struct BranchPoint {
                BranchPoint *parent = nullptr;
                int pathout = -1;
                size_t top_id;                     // CNode id of branch point
                std::vector<BlockTrace *> paths;
                int depth = 0;
                bool ismark = false;

                void markPath();
                int distance(BranchPoint *op2);
            };

            struct BlockTrace {
                enum : uint32_t { f_active = 1, f_terminal = 2 };
                uint32_t flags = 0;
                BranchPoint *top;
                int pathout;
                size_t bottom_id;      // CNode id (or NONE for root traces)
                size_t dest_id;        // destination CNode id
                int edgelump = 1;
                std::list<BlockTrace *>::iterator activeiter;
                BranchPoint *derivedbp = nullptr;

                bool isActive() const { return flags & f_active; }
                bool isTerminal() const { return flags & f_terminal; }
            };

            struct BadEdgeScore {
                size_t exitproto_id;
                BlockTrace *trace;
                int distance = -1;
                int terminal = 0;
                int siblingedge = 0;

                bool compareFinal(const BadEdgeScore &op2) const;
                bool operator<(const BadEdgeScore &op2) const;
            };

            std::list<FloatingEdge> &likelygoto;
            std::vector<size_t> rootlist;
            std::vector<BranchPoint *> branchlist;
            int activecount = 0;
            std::list<BlockTrace *> activetrace;
            std::list<BlockTrace *>::iterator current_activeiter;
            size_t finishblock_id = CNode::NONE;

            void removeTrace(BlockTrace *trace);
            void processExitConflict(std::list<BadEdgeScore>::iterator start,
                                     std::list<BadEdgeScore>::iterator end);
            BlockTrace *selectBadEdge();
            void insertActive(BlockTrace *trace);
            void removeActive(BlockTrace *trace);
            bool checkOpen(const CGraph &g, BlockTrace *trace);
            std::list<BlockTrace *>::iterator openBranch(CGraph &g,
                                                         BlockTrace *parent);
            bool checkRetirement(BlockTrace *trace, size_t &exitblock_id);
            std::list<BlockTrace *>::iterator retireBranch(BranchPoint *bp,
                                                           size_t exitblock_id);
            void clearVisitCount(CGraph &g);

        public:
            TraceDAG(std::list<FloatingEdge> &lg) : likelygoto(lg) {}
            ~TraceDAG();

            void addRoot(size_t root_id) { rootlist.push_back(root_id); }
            void setFinishBlock(size_t id) { finishblock_id = id; }
            void initialize();
            void pushBranches(CGraph &g);
        };

        /// Clear CNode::mark for all nodes in body vector.
        void clearMarks(CGraph &g, const std::vector<size_t> &body);

        /// Scan back-edges and create LoopBody records.
        void labelLoops(CGraph &g, std::list<LoopBody> &loopbody,
                        std::vector<LoopBody *> &looporder);

        /// Discover all loops, compute bodies/nesting/exits, and order innermost-first.
        /// Called once from collapseStructure() after markBackEdges().
        void orderLoopBodies(CGraph &g, std::list<LoopBody> &loopbody);

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
