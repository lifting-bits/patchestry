/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/CfgBuilder.hpp>
#include <patchestry/AST/SNode.hpp>
#include <patchestry/Util/Options.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <list>
#include <string>
#include <unordered_set>
#include <utility>
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
            static constexpr size_t kNone = std::numeric_limits<size_t>::max();

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
            size_t collapsed_into = kNone;  // representative node after collapse

            // Flags for the collapse algorithm
            bool mark = false;
            int visit_count = 0;

            enum EdgeFlag : uint32_t {
                kGoto     = 1u << 0,
                kBack     = 1u << 1,
                kLoopExit = 1u << 2,
            };

            bool IsGotoOut(size_t i) const {
                return i < edge_flags.size() && (edge_flags[i] & kGoto);
            }
            bool IsBackEdge(size_t i) const {
                return i < edge_flags.size() && (edge_flags[i] & kBack);
            }
            bool IsLoopExit(size_t i) const {
                return i < edge_flags.size() && (edge_flags[i] & kLoopExit);
            }
            bool IsDecisionOut(size_t i) const {
                return !IsGotoOut(i) && !IsBackEdge(i);
            }
            /// Edges to trace in TraceDAG: not back-edges, not loop-exit edges.
            bool IsLoopDagOut(size_t i) const {
                return !IsBackEdge(i) && !IsLoopExit(i);
            }
            bool IsSwitchOut() const { return succs.size() > 2; }

            size_t SizeIn() const { return preds.size(); }
            size_t SizeOut() const { return succs.size(); }

            void SetGoto(size_t i) {
                if (i < edge_flags.size()) edge_flags[i] |= kGoto;
            }
            void SetLoopExit(size_t i) {
                if (i < edge_flags.size()) edge_flags[i] |= kLoopExit;
            }
            void ClearLoopExit(size_t i) {
                if (i < edge_flags.size()) edge_flags[i] &= ~static_cast<uint32_t>(kLoopExit);
            }
        };

        /// The collapsing graph — lightweight mirror of the Cfg that supports
        /// collapsing blocks into structured nodes.
        struct CGraph {
            std::vector<CNode> nodes;
            size_t entry = 0;

            /// Active (uncollapsed) node ids
            std::vector<size_t> ActiveIds() const {
                std::vector<size_t> result;
                for (auto &n : nodes) {
                    if (!n.collapsed) result.push_back(n.id);
                }
                return result;
            }

            size_t ActiveCount() const {
                size_t c = 0;
                for (auto &n : nodes) {
                    if (!n.collapsed) ++c;
                }
                return c;
            }

            CNode &Node(size_t id) { return nodes[id]; }
            const CNode &Node(size_t id) const { return nodes[id]; }

            /// Remove an edge from the active graph
            void RemoveEdge(size_t from, size_t to) {
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
            size_t CollapseNodes(const std::vector<size_t> &ids, SNode *snode);
        };

        /// Build the collapse graph from a Cfg.
        CGraph BuildCGraph(const Cfg &cfg);

        /// Detect back-edges using DFS and mark them in the graph.
        void MarkBackEdges(CGraph &g);

        /// A detected natural loop: header, back-edge tails, nesting info.
        struct LoopBody {
            size_t head;                          // loop header CNode id
            std::vector<size_t> tails;            // CNode ids with back-edges to head
            int depth = 0;                        // nesting depth (deeper = higher number)
            int unique_count = 0;                 // count of head+tail nodes before reachability
            size_t exit_block = kNone;            // official exit CNode id, or kNone
            LoopBody *immed_container = nullptr;  // immediately containing loop

            static constexpr size_t kNone = std::numeric_limits<size_t>::max();

            explicit LoopBody(size_t h) : head(h) {}

            void AddTail(size_t t) { tails.push_back(t); }

            /// Core body computation: backward reachability from tails to head.
            /// Populates `body` with all CNode ids in the loop. Sets CNode::mark.
            /// Caller MUST call ClearMarks() after using the body.
            void FindBase(CGraph &g, std::vector<size_t> &body) const;

            /// Set immed_container based on containment within other loops.
            void LabelContainments(const CGraph &g, const std::vector<size_t> &body,
                                   const std::vector<LoopBody *> &looporder);

            /// Exit detection, tail ordering, body extension, exit edge labeling.
            void FindExit(const CGraph &g, const std::vector<size_t> &body);
            void OrderTails(const CGraph &g);
            void Extend(CGraph &g, std::vector<size_t> &body) const;
            void LabelExitEdges(CGraph &g, const std::vector<size_t> &body) const;

            /// Merge LoopBody records that share the same head.
            static void MergeIdenticalHeads(std::vector<LoopBody *> &looporder,
                                            std::list<LoopBody> &storage);

            /// Mark edges leaving the loop body as kLoopExit.
            void SetExitMarks(CGraph &g, const std::vector<size_t> &body) const;

            /// Clear kLoopExit marks set by SetExitMarks.
            void ClearExitMarks(CGraph &g, const std::vector<size_t> &body) const;

            /// Check if this loop's head is still active (not collapsed).
            bool Update(const CGraph &g) const;

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
            /// or {CNode::kNone, 0} if the edge no longer exists.
            std::pair<size_t, size_t> GetCurrentEdge(const CGraph &g) const;
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

                void MarkPath();
                int Distance(BranchPoint *op2);
            };

            struct BlockTrace {
                enum : uint32_t { kActive = 1, kTerminal = 2 };
                uint32_t flags = 0;
                BranchPoint *top;
                int pathout;
                size_t bottom_id;      // CNode id (or kNone for root traces)
                size_t dest_id;        // destination CNode id
                int edgelump = 1;
                std::list<BlockTrace *>::iterator activeiter;
                BranchPoint *derivedbp = nullptr;

                bool IsActive() const { return flags & kActive; }
                bool IsTerminal() const { return flags & kTerminal; }
            };

            struct BadEdgeScore {
                size_t exitproto_id;
                BlockTrace *trace;
                int distance = -1;
                int terminal = 0;
                int siblingedge = 0;

                bool CompareFinal(const BadEdgeScore &op2) const;
                bool operator<(const BadEdgeScore &op2) const;
            };

            std::list<FloatingEdge> &likelygoto_;
            std::vector<size_t> rootlist_;
            std::vector<BranchPoint *> branchlist_;
            int activecount_ = 0;
            std::list<BlockTrace *> activetrace_;
            std::list<BlockTrace *>::iterator current_activeiter_;
            size_t finishblock_id_ = CNode::kNone;

            void RemoveTrace(BlockTrace *trace);
            void ProcessExitConflict(std::list<BadEdgeScore>::iterator start,
                                     std::list<BadEdgeScore>::iterator end);
            BlockTrace *SelectBadEdge();
            void InsertActive(BlockTrace *trace);
            void RemoveActive(BlockTrace *trace);
            bool CheckOpen(CGraph &g, BlockTrace *trace);
            std::list<BlockTrace *>::iterator OpenBranch(CGraph &g,
                                                         BlockTrace *parent);
            bool CheckRetirement(BlockTrace *trace, size_t &exitblock_id);
            std::list<BlockTrace *>::iterator RetireBranch(BranchPoint *bp,
                                                           size_t exitblock_id);
            void ClearVisitCount(CGraph &g);

        public:
            TraceDAG(std::list<FloatingEdge> &lg) : likelygoto_(lg) {}
            ~TraceDAG();

            void AddRoot(size_t root_id) { rootlist_.push_back(root_id); }
            void SetFinishBlock(size_t id) { finishblock_id_ = id; }
            void Initialize();
            void PushBranches(CGraph &g);
        };

        /// Clear CNode::mark for all nodes in body vector.
        void ClearMarks(CGraph &g, const std::vector<size_t> &body);

        /// Scan back-edges and create LoopBody records.
        void LabelLoops(CGraph &g, std::list<LoopBody> &loopbody,
                        std::vector<LoopBody *> &looporder);

        /// Discover all loops, compute bodies/nesting/exits, and order innermost-first.
        /// Called once from CfgFoldStructure() after MarkBackEdges().
        void OrderLoopBodies(CGraph &g, std::list<LoopBody> &loopbody);

    } // namespace detail

    /// Build a structured SNode tree from a Cfg using iterative
    /// pattern-matching and graph collapse.
    ///
    /// The algorithm structures the CFG *before* generating the SNode tree,
    /// producing better results for complex control flow than a flat
    /// goto-heavy tree followed by goto elimination.
    ///
    /// Pipeline:
    ///   1. Detect loops via back-edges (Tarjan-based).
    ///   2. Resolve logical AND/OR condition chains.
    ///   3. Iteratively fold graph patterns:
    ///      - Sequential blocks → SSeq
    ///      - If / if-else → SIfThenElse
    ///      - While-do / do-while → SWhile / SDoWhile
    ///      - Infinite loops → SWhile(true, body)
    ///      - Switch statements → SSwitch
    ///   4. When stuck, resolve by selecting an edge to mark as goto (DAG heuristic).
    ///   5. Repeat until fully collapsed.
    ///   6. Refine: while→for, break/continue insertion, dead label removal.
    SNode *CfgFoldStructure(const Cfg &cfg, SNodeFactory &factory,
                             clang::ASTContext &ctx,
                             const patchestry::Options &options = {});

} // namespace patchestry::ast
