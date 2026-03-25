/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/SNode.hpp>

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

    /// A case entry for switch blocks — carries P-Code case metadata.
    struct SwitchCaseEntry {
        int64_t value;            // case constant value
        size_t succ_index;        // index into CNode::succs[] for this case target
        bool has_exit = false;    // whether P-Code marked this case as having a break/exit
        bool is_default = false;  // true for the default/fallback arm
    };

    // -----------------------------------------------------------------------
    // CGraph / CNode — CFG representation for structuring.
    //
    // Built directly from P-Code JSON and structured in-place.
    // CNode  ≡ basic block + topology + collapse state
    // CGraph ≡ container of CNodes
    // -----------------------------------------------------------------------

    /// A node in the graph.  Starts as a leaf (one basic block's content).
    /// As structuring rules fire, nodes get absorbed into SNode wrappers and
    /// removed from the active set.
    struct CNode {
        static constexpr size_t kNone = std::numeric_limits<size_t>::max();

        size_t id;                          // node index in CGraph::nodes
        std::vector<size_t> succs;          // outgoing edges (by CNode id)
        std::vector<size_t> preds;          // incoming edges (by CNode id)

        // Edge properties (indexed same as succs)
        std::vector<uint32_t> edge_flags;

        // The SNode produced when this node is collapsed (null = leaf)
        SNode *structured = nullptr;

        // Leaf payload: statements from the original basic block
        std::string label;                      // mutable label (cleared after SLabel wrapping)
        std::string original_label;             // immutable — set once, never cleared
        std::vector<clang::Stmt *> stmts;
        clang::Expr *branch_cond = nullptr;
        bool is_conditional = false;

        /// Terminal control-flow stmt (goto/if-goto/switch) popped by
        /// edge construction.  Stored separately from content stmts
        /// (structural: operations and edges are decoupled).
        /// Used to reconstruct gotos when fold rules collapse blocks
        /// with unhandled external exits.
        clang::Stmt *terminal = nullptr;

        // Switch case metadata (non-empty for switch blocks)
        std::vector<SwitchCaseEntry> switch_cases;

        // Set when absorbed into a parent structured node
        size_t collapsed_into = kNone;  // representative node after collapse

        bool IsCollapsed() const { return collapsed_into != kNone; }

        // Hierarchical block type
        enum class BlockType : uint8_t {
            kBasic,      // leaf block
            kSequence,   // sequential chain
            kIf,         // if / if-else
            kWhile,      // while loop
            kDoWhile,    // do-while loop
            kInfLoop,    // infinite loop
            kSwitch,     // switch statement
            kGoto,       // goto wrapper
        };
        BlockType block_type = BlockType::kBasic;
        std::vector<size_t> children;  // child node ids absorbed by this block

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
        /// A switch block has either >2 successors OR explicit switch_cases
        /// metadata (multiple cases can target the same successor, so a 2-case
        /// switch with 2 targets still counts).
        bool IsSwitchOut() const {
            return succs.size() > 2 || !switch_cases.empty();
        }

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

    /// The flow graph — single graph type used for both CFG representation
    /// and in-place structuring.  Replaces the Cfg→CGraph two-step pipeline.
    struct CGraph {
        std::vector<CNode> nodes;
        size_t entry = 0;

        /// Active (uncollapsed) node ids
        std::vector<size_t> ActiveIds() const {
            std::vector<size_t> result;
            for (auto &n : nodes) {
                if (!n.IsCollapsed()) result.push_back(n.id);
            }
            return result;
        }

        size_t ActiveCount() const {
            size_t c = 0;
            for (auto &n : nodes) {
                if (!n.IsCollapsed()) ++c;
            }
            return c;
        }

        CNode &Node(size_t id) { assert(id < nodes.size()); return nodes[id]; }
        const CNode &Node(size_t id) const { assert(id < nodes.size()); return nodes[id]; }

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

        /// Absorb a set of nodes into a hierarchical structured block.
        /// The first id becomes the representative.  Children are marked
        /// collapsed but their stmts/labels remain accessible.
        /// Returns the representative node id.
        size_t IdentifyInternal(const std::vector<size_t> &ids,
                                CNode::BlockType type, SNode *snode);
    };

    class FunctionBuilder;

    /// Build CGraph directly from P-Code JSON via FunctionBuilder.
    /// This is the structural path: JSON → CGraph (no intermediate Clang AST gotos).
    CGraph BuildCGraph(FunctionBuilder &builder, clang::ASTContext &ctx);

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
        void FindBase(CGraph &g, std::vector<size_t> &body) const;

        /// Set immed_container based on containment within other loops.
        void LabelContainments(const CGraph &g, const std::vector<size_t> &body,
                               const std::vector<LoopBody *> &looporder);

        /// Exit detection, tail ordering, body extension, exit edge labeling.
        void FindExit(CGraph &g, const std::vector<size_t> &body);
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
            size_t top_id;
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
            size_t bottom_id;
            size_t dest_id;
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
    void OrderLoopBodies(CGraph &g, std::list<LoopBody> &loopbody);

} // namespace patchestry::ast
