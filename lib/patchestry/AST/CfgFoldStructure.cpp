/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/CfgFoldStructure.hpp>
#include <patchestry/AST/DomTree.hpp>
#include <patchestry/AST/LoopInfo.hpp>
#include <patchestry/Util/Log.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>

namespace patchestry::ast {

    // -----------------------------------------------------------------------
    // detail:: namespace — CGraph method definitions and graph builders
    // -----------------------------------------------------------------------
    namespace detail {

        // ---------------------------------------------------------------
        // CollapseNodes — merge a set of CNode ids into a single
        // representative node, rewiring all external edges.
        //
        //  BEFORE  (collapsing {A, B} into rep=A):
        //
        //      P ──→ A ──→ B ──→ S        Q ──→ B
        //            │           │
        //            └───→ X    (internal edges dropped)
        //
        //  AFTER:
        //
        //      P ──→ rep ──→ S             Q ──→ rep
        //
        //  Steps:
        //    1. Collect ext_preds  — predecessors outside the set
        //    2. Collect ext_succs  — successors outside the set (deduped)
        //    3. Mark non-rep nodes as collapsed
        //    4. Rewire: external successors drop old pred refs,
        //       external predecessors redirect succ refs → rep
        //    5. Dedup predecessor succs (handles shared-pred case,
        //       e.g. P→A and P→B both become P→rep)
        //    6. Install ext_preds / ext_succs on rep
        //    7. Ensure rep appears in each successor's pred list
        // ---------------------------------------------------------------
        size_t CGraph::CollapseNodes(const std::vector< size_t > &ids, SNode *snode) {
            // --- Step 0: choose representative ---
            size_t rep = ids[0];
            nodes[rep].structured = snode;

            std::unordered_set< size_t > idset(ids.begin(), ids.end());

            // --- Step 1: collect external predecessors ---
            std::vector< size_t > ext_preds;
            for (size_t nid : ids) {
                for (size_t p : nodes[nid].preds) {
                    if (idset.count(p) == 0) ext_preds.push_back(p);
                }
            }

            // --- Step 2: collect external successors (first-seen flags win) ---
            std::vector< size_t > ext_succs;
            std::vector< uint32_t > ext_succ_flags;
            for (size_t nid : ids) {
                for (size_t i = 0; i < nodes[nid].succs.size(); ++i) {
                    size_t s = nodes[nid].succs[i];
                    if (idset.count(s) == 0) {
                        if (std::find(ext_succs.begin(), ext_succs.end(), s) == ext_succs.end()) {
                            ext_succs.push_back(s);
                            ext_succ_flags.push_back(nodes[nid].edge_flags[i]);
                        }
                    }
                }
            }

            // --- Step 3: mark non-rep nodes as collapsed ---
            for (size_t nid : ids) {
                if (nid != rep) {
                    nodes[nid].collapsed = true;
                    nodes[nid].collapsed_into = rep;
                }
            }

            // --- Step 4: rewire edges ---
            //  4a. For each external successor, remove old pred references
            //      to any collapsed node.
            //  4b. For each external predecessor, redirect succ entries
            //      that pointed to a collapsed node → rep.
            for (size_t nid : ids) {
                for (size_t s : nodes[nid].succs) {
                    if (idset.count(s) == 0) {
                        auto &p = nodes[s].preds;
                        p.erase(std::remove(p.begin(), p.end(), nid), p.end());
                    }
                }
                for (size_t p : nodes[nid].preds) {
                    if (idset.count(p) == 0) {
                        auto &ss = nodes[p].succs;
                        for (size_t i = 0; i < ss.size(); ++i) {
                            if (ss[i] == nid) {
                                ss[i] = rep;
                            }
                        }
                    }
                }
            }

            // --- Step 5: deduplicate predecessor succs ---
            //
            //  Shared-predecessor case:
            //
            //      BEFORE redirect       AFTER redirect (broken)
            //       P.succs = [A, B]      P.succs = [rep, rep]  ← dup!
            //
            //  We compact each predecessor's succs/edge_flags in place,
            //  keeping only the first occurrence of each target.
            for (size_t p : ext_preds) {
                auto &ss = nodes[p].succs;
                auto &sf = nodes[p].edge_flags;
                std::unordered_set< size_t > seen;
                size_t write = 0;
                for (size_t i = 0; i < ss.size(); ++i) {
                    if (seen.insert(ss[i]).second) {
                        ss[write] = ss[i];
                        sf[write] = sf[i];
                        ++write;
                    }
                }
                ss.resize(write);
                sf.resize(write);
            }

            // --- Step 6: install edges on representative ---
            nodes[rep].succs = ext_succs;
            nodes[rep].edge_flags = ext_succ_flags;

            std::sort(ext_preds.begin(), ext_preds.end());
            ext_preds.erase(std::unique(ext_preds.begin(), ext_preds.end()), ext_preds.end());
            nodes[rep].preds = ext_preds;

            // --- Step 7: ensure rep is listed in each successor's preds ---
            for (size_t s : ext_succs) {
                auto &p = nodes[s].preds;
                if (std::find(p.begin(), p.end(), rep) == p.end()) {
                    p.push_back(rep);
                }
            }

            nodes[rep].is_conditional = !ext_succs.empty() && ext_succs.size() == 2;
            nodes[rep].stmts.clear();
            nodes[rep].label.clear();  // label already embedded via LeafFromNode in rule

            // Preserve branch_cond from the collapsed node that owns the
            // conditional split producing the 2 external successors.
            // Without this, FoldIfElse/FoldIfThen see a null
            // condition and synthesize a spurious `if(1)`.
            nodes[rep].branch_cond = nullptr;
            if (nodes[rep].is_conditional) {
                // Walk collapsed nodes in reverse (last node most likely
                // contributed the outgoing conditional edges).
                for (auto it = ids.rbegin(); it != ids.rend(); ++it) {
                    if (nodes[*it].branch_cond) {
                        nodes[rep].branch_cond = nodes[*it].branch_cond;
                        break;
                    }
                }
            }

            // Preserve switch metadata from collapsed nodes so FoldSwitch
            // can still fire on the representative after chaining.
            nodes[rep].switch_cases.clear();
            for (size_t nid : ids) {
                if (nid != rep && !nodes[nid].switch_cases.empty()) {
                    nodes[rep].switch_cases = std::move(nodes[nid].switch_cases);
                    // Also preserve branch_cond for switch discriminant.
                    if (!nodes[rep].branch_cond) {
                        nodes[rep].branch_cond = nodes[nid].branch_cond;
                    }
                    break;
                }
            }

            return rep;
        }

        // Build the collapse graph from the Cfg
        CGraph buildCGraph(const Cfg &cfg) {
            CGraph g;
            g.entry = cfg.entry;
            g.nodes.resize(cfg.blocks.size());

            for (size_t i = 0; i < cfg.blocks.size(); ++i) {
                auto &cb = cfg.blocks[i];
                auto &cn = g.nodes[i];
                cn.id = i;
                cn.label = cb.label;
                cn.stmts = cb.stmts;
                cn.branch_cond = cb.branch_cond;
                cn.is_conditional = cb.is_conditional;
                cn.succs = cb.succs;
                cn.switch_cases = cb.switch_cases;
                cn.edge_flags.resize(cb.succs.size(), 0);

                // Assert Ghidra convention: 2-successor conditional blocks have
                // succs[0]=false/fallthrough, succs[1]=true/taken
                assert((!cb.is_conditional || cb.succs.size() != 2 ||
                        (cb.succs[0] == cb.fallthrough_succ && cb.succs[1] == cb.taken_succ))
                       && "CfgBlock edge polarity must be false-first (Ghidra convention)");
            }

            // Build predecessor lists
            for (size_t i = 0; i < g.nodes.size(); ++i) {
                for (size_t s : g.nodes[i].succs) {
                    g.nodes[s].preds.push_back(i);
                }
            }

            return g;
        }

        // Detect back-edges using DFS
        void MarkBackEdges(CGraph &g) {
            enum Color { WHITE, GRAY, BLACK };
            std::vector<Color> color(g.nodes.size(), WHITE);

            std::function<void(size_t)> dfs = [&](size_t u) {
                color[u] = GRAY;
                auto &nd = g.Node(u);
                for (size_t i = 0; i < nd.succs.size(); ++i) {
                    size_t v = nd.succs[i];
                    if (color[v] == GRAY) {
                        nd.edge_flags[i] |= CNode::kBack;
                    } else if (color[v] == WHITE) {
                        dfs(v);
                    }
                }
                color[u] = BLACK;
            };

            dfs(g.entry);
        }

        // -------------------------------------------------------------------
        // LoopBody core methods
        // -------------------------------------------------------------------

        void ClearMarks(CGraph &g, const std::vector< size_t > &body) {
            for (size_t id : body) {
                g.Node(id).mark = false;
            }
        }

        void LoopBody::FindBase(CGraph &g, std::vector< size_t > &body) const {
            body.clear();

            // Mark head, add to body.
            g.Node(head).mark = true;
            body.push_back(head);

            // Mark each tail (skip if already marked, e.g. head == tail).
            for (size_t t : tails) {
                if (!g.Node(t).mark) {
                    g.Node(t).mark = true;
                    body.push_back(t);
                }
            }

            // BFS backward from tails: iterate body starting at index 1
            // (index 0 is head -- we don't go backward from head).
            for (size_t idx = 1; idx < body.size(); ++idx) {
                auto &nd = g.Node(body[idx]);
                for (size_t p : nd.preds) {
                    if (g.Node(p).mark) {
                        continue;
                    }
                    if (g.Node(p).collapsed) {
                        continue;
                    }

                    // Check if the incoming edge from p to body[idx] is a goto edge.
                    // Find the succ index in p that points to body[idx].
                    bool is_goto = false;
                    auto &pn     = g.Node(p);
                    for (size_t si = 0; si < pn.succs.size(); ++si) {
                        if (pn.succs[si] == body[idx]) {
                            if (pn.IsGotoOut(si)) {
                                is_goto = true;
                            }
                            break;
                        }
                    }
                    if (is_goto) continue;

                    g.Node(p).mark = true;
                    body.push_back(p);
                }
            }
        }

        // ---------------------------------------------------------------
        // LabelContainments — set immed_container for nested loops.
        //
        // Precondition: the caller has marked every CNode in `this`
        // loop's body (node.mark == true).
        //
        // For each other loop `lb` in `looporder`, if lb's head is
        // marked then `lb` is nested inside `this` loop.  We want
        // lb->immed_container to point to the SMALLEST (tightest)
        // enclosing loop.  The decision rule:
        //
        //   - If lb has no container yet, adopt `this`.
        //   - If lb already has a container C whose head is also
        //     marked, then C is inside `this` loop too — C is a
        //     tighter fit, so keep C.
        //   - Otherwise C is not inside `this`, which means `this`
        //     is a better (tighter) container — replace C with `this`.
        // ---------------------------------------------------------------
        void LoopBody::LabelContainments(
            const CGraph &g, const std::vector< size_t > & /*body*/,
            const std::vector< LoopBody * > &looporder
        ) {
            for (LoopBody *lb : looporder) {
                if (lb == this) continue;
                if (!g.Node(lb->head).mark) {
                    continue;
                }

                if (lb->immed_container == nullptr) {
                    lb->immed_container = this;
                } else if (!g.Node(lb->immed_container->head).mark) {
                    lb->immed_container = this;
                }
            }
        }

        void LoopBody::MergeIdenticalHeads(
            std::vector< LoopBody * > &looporder, std::list< LoopBody > &storage
        ) {
            // Sort by head for merge detection.
            std::sort(looporder.begin(), looporder.end(),
                [](const LoopBody *a, const LoopBody *b) { return a->head < b->head; });

            size_t write = 0;
            for (size_t read = 0; read < looporder.size(); ++read) {
                if (write > 0 && looporder[write - 1]->head == looporder[read]->head) {
                    // Merge tails into the previous entry.
                    auto *dst = looporder[write - 1];
                    auto *src = looporder[read];
                    for (size_t t : src->tails) {
                        dst->AddTail(t);
                    }
                    // Erase src from storage.
                    for (auto it = storage.begin(); it != storage.end(); ++it) {
                        if (&(*it) == src) {
                            storage.erase(it);
                            break;
                        }
                    }
                } else {
                    looporder[write++] = looporder[read];
                }
            }
            looporder.resize(write);
        }

        void LabelLoops(
            CGraph &g, std::list< LoopBody > &loopbody, std::vector< LoopBody * > &looporder
        ) {
            // Scan all edges for back-edges. Create one LoopBody per back-edge.
            for (auto &n : g.nodes) {
                if (n.collapsed) continue;
                for (size_t i = 0; i < n.succs.size(); ++i) {
                    if (n.IsBackEdge(i)) {
                        size_t hd = n.succs[i];
                        loopbody.emplace_back(hd);
                        loopbody.back().AddTail(n.id);
                        looporder.push_back(&loopbody.back());
                    }
                }
            }

            // Merge loops sharing the same head.
            LoopBody::MergeIdenticalHeads(looporder, loopbody);

            // For each loop: findBase, labelContainments, clearMarks.
            for (LoopBody *lb : looporder) {
                std::vector<size_t> body;
                lb->FindBase(g, body);
                lb->unique_count = static_cast<int>(body.size());
                lb->LabelContainments(g, body, looporder);
                ClearMarks(g, body);
            }

            // Compute depth from immed_container chains.
            for (LoopBody *lb : looporder) {
                int d = 0;
                for (LoopBody *c = lb->immed_container; c != nullptr; c = c->immed_container) {
                    ++d;
                }
                lb->depth = d;
            }

            // Sort innermost-first (higher depth first) for processing order.
            std::sort(looporder.begin(), looporder.end(),
                [](const LoopBody *a, const LoopBody *b) { return *a < *b; });
        }

        // -------------------------------------------------------------------
        // LoopBody exit detection, tail ordering, extension, exit labeling
        // -------------------------------------------------------------------

        void LoopBody::FindExit(const CGraph &g, const std::vector< size_t > &body) {
            // Select a single exit block for the loop.
            // Priority: tail exits first, then head exits, then body exits.
            // A candidate exit is an unmarked successor reached via a non-goto edge.

            std::vector<size_t> candidates;

            // Scan tails for exits.
            for (size_t t : tails) {
                const auto &tn = g.Node(t);
                for (size_t i = 0; i < tn.succs.size(); ++i) {
                    size_t s = tn.succs[i];
                    if (!g.Node(s).mark && !tn.IsGotoOut(i) && !tn.IsBackEdge(i)) {
                        if (!immed_container) {
                            // No container -- take first candidate immediately.
                            exit_block = s;
                            return;
                        }
                        candidates.push_back(s);
                    }
                }
            }

            // If container exists, also scan head and remaining body nodes.
            if (immed_container) {
                // Scan head (body[0]).
                const auto &hd = g.Node(body[0]);
                for (size_t i = 0; i < hd.succs.size(); ++i) {
                    size_t s = hd.succs[i];
                    if (!g.Node(s).mark && !hd.IsGotoOut(i) && !hd.IsBackEdge(i)) {
                        candidates.push_back(s);
                    }
                }
                // Scan body nodes beyond the unique set (index >= unique_count).
                for (size_t idx = static_cast<size_t>(unique_count); idx < body.size(); ++idx) {
                    const auto &bn = g.Node(body[idx]);
                    for (size_t i = 0; i < bn.succs.size(); ++i) {
                        size_t s = bn.succs[i];
                        if (!g.Node(s).mark && !bn.IsGotoOut(i) && !bn.IsBackEdge(i)) {
                            candidates.push_back(s);
                        }
                    }
                }
            }

            // Pick exit: for v1, take the first candidate.
            // NOTE: Full container-aware exit selection (preferring candidates inside
            // the container's body) can be added later if needed.
            if (!candidates.empty()) {
                exit_block = candidates[0];
            } else {
                exit_block = kNone;
            }
        }

        void LoopBody::OrderTails(const CGraph &g) {
            // Swap the tail that directly reaches exit_block to tails[0].
            if (tails.size() <= 1 || exit_block == kNone) {
                return;
            }

            for (size_t ti = 0; ti < tails.size(); ++ti) {
                const auto &tn = g.Node(tails[ti]);
                for (size_t s : tn.succs) {
                    if (s == exit_block) {
                        // Found a tail that reaches exit_block directly.
                        if (ti != 0) {
                            std::swap(tails[0], tails[ti]);
                        }
                        return;
                    }
                }
            }
            // No tail reaches exit_block directly -- leave order unchanged.
        }

        void LoopBody::Extend(CGraph &g, std::vector< size_t > &body) const {
            // Add dominated-only blocks: blocks whose ALL predecessors are in the body.
            // Uses visit_count to track how many predecessors are body members.
            // Fixpoint iteration until no new nodes are added.

            std::vector<size_t> touched;  // track nodes with non-zero visit_count

            bool added = true;
            while (added) {
                added = false;

                // For each body node, increment visit_count of non-body, non-collapsed successors.
                for (size_t bid : body) {
                    const auto &bn = g.Node(bid);
                    for (size_t i = 0; i < bn.succs.size(); ++i) {
                        size_t s = bn.succs[i];
                        if (g.Node(s).mark) {
                            continue; // already in body
                        }
                        if (g.Node(s).collapsed) {
                            continue; // absorbed
                        }
                        if (bn.IsGotoOut(i)) {
                            continue; // don't extend through gotos
                        }
                        if (s == exit_block) continue;        // don't extend into exit
                        if (g.Node(s).visit_count == 0) {
                            touched.push_back(s);
                        }
                        g.Node(s).visit_count++;
                    }
                }

                // Check which visited nodes have ALL predecessors in body.
                for (size_t s : touched) {
                    auto &sn = g.Node(s);
                    if (sn.mark) continue;  // already added
                    if (sn.visit_count == static_cast< int >(sn.SizeIn())) {
                        // All preds are in body -- include this node.
                        sn.mark = true;
                        body.push_back(s);
                        added = true;
                    }
                }

                // Reset visit_count for next iteration.
                for (size_t s : touched) {
                    g.Node(s).visit_count = 0;
                }
                touched.clear();
            }
        }

        void LoopBody::LabelExitEdges(CGraph &g, const std::vector< size_t > &body) const {
            // Mark kLoopExit on all edges leaving the loop body.
            for (size_t bid : body) {
                auto &bn = g.Node(bid);
                for (size_t i = 0; i < bn.succs.size(); ++i) {
                    size_t s = bn.succs[i];
                    if (!g.Node(s).mark && !bn.IsGotoOut(i)) {
                        bn.edge_flags[i] |= CNode::kLoopExit;
                    }
                }
            }
        }

        // -------------------------------------------------------------------
        // orderLoopBodies: orchestrate full loop detection pipeline
        // -------------------------------------------------------------------

        void OrderLoopBodies(CGraph &g, std::list< LoopBody > &loopbody) {
            // 1. labelLoops: create LoopBody per back-edge, merge, compute containment/depth
            std::vector<LoopBody *> looporder;
            LabelLoops(g, loopbody, looporder);
            if (loopbody.empty()) return;

            // 2. For each loop (innermost-first): findExit, orderTails, extend, labelExitEdges
            for (auto &lb : loopbody) {
                std::vector<size_t> body;
                lb.FindBase(g, body);
                lb.FindExit(g, body);
                lb.OrderTails(g);
                lb.Extend(g, body);
                lb.LabelExitEdges(g, body);
                ClearMarks(g, body);
            }
        }

        // -------------------------------------------------------------------
        // LoopBody: exit mark management and update
        // -------------------------------------------------------------------

        void LoopBody::SetExitMarks(CGraph &g, const std::vector< size_t > &body) const {
            std::unordered_set<size_t> bodyset(body.begin(), body.end());
            for (size_t nid : body) {
                auto &n = g.Node(nid);
                for (size_t i = 0; i < n.succs.size(); ++i) {
                    if (bodyset.count(n.succs[i]) == 0) {
                        n.SetLoopExit(i);
                    }
                }
            }
        }

        void LoopBody::ClearExitMarks(CGraph &g, const std::vector< size_t > &body) const {
            for (size_t nid : body) {
                auto &n = g.Node(nid);
                for (size_t i = 0; i < n.succs.size(); ++i) {
                    n.ClearLoopExit(i);
                }
            }
        }

        bool LoopBody::Update(const CGraph &g) const { return !g.Node(head).collapsed; }

        // -------------------------------------------------------------------
        // FloatingEdge
        // -------------------------------------------------------------------

        std::pair< size_t, size_t > FloatingEdge::GetCurrentEdge(const CGraph &g) const {
            // Gap 3: walk up collapse hierarchy (like Ghidra's getParent walk)
            size_t top = top_id;
            while (top < g.nodes.size() && g.Node(top).collapsed) {
                size_t next = g.Node(top).collapsed_into;
                if (next == CNode::kNone || next == top) {
                    break;
                }
                top = next;
            }
            size_t bot = bottom_id;
            while (bot < g.nodes.size() && g.Node(bot).collapsed) {
                size_t next = g.Node(bot).collapsed_into;
                if (next == CNode::kNone || next == bot) {
                    break;
                }
                bot = next;
            }

            if (top >= g.nodes.size() || bot >= g.nodes.size())
                return { CNode::kNone, 0 };
            if (g.Node(top).collapsed || g.Node(bot).collapsed) {
                return { CNode::kNone, 0 };
            }
            if (top == bot) {
                return { CNode::kNone, 0 };
            }

            const auto &succs = g.Node(top).succs;
            for (size_t i = 0; i < succs.size(); ++i) {
                if (succs[i] == bot)
                    return {top, i};
            }
            return { CNode::kNone, 0 };
        }

        // -------------------------------------------------------------------
        // TraceDAG implementation
        // -------------------------------------------------------------------

        TraceDAG::~TraceDAG() {
            for (auto *bp : branchlist_) {
                for (auto *bt : bp->paths) {
                    delete bt;
                }
                delete bp;
            }
        }

        void TraceDAG::BranchPoint::MarkPath() {
            ismark = true;
            if (parent != nullptr && !parent->ismark)
                parent->MarkPath();
        }

        int TraceDAG::BranchPoint::Distance(BranchPoint *op2) {
            // Clear marks
            for (auto *cur = this; cur != nullptr; cur = cur->parent)
                cur->ismark = false;
            for (auto *cur = op2; cur != nullptr; cur = cur->parent)
                cur->ismark = false;

            // Mark path from this to root
            MarkPath();

            // Walk from op2 to find common ancestor
            int dist = 0;
            auto *cur = op2;
            while (cur != nullptr && !cur->ismark) {
                ++dist;
                cur = cur->parent;
            }
            if (cur == nullptr) return dist;

            // Add distance from this to common ancestor
            auto *cur2 = this;
            while (cur2 != cur) {
                ++dist;
                cur2 = cur2->parent;
            }
            return dist;
        }

        void TraceDAG::InsertActive(BlockTrace *trace) {
            trace->flags |= BlockTrace::kActive;
            activetrace_.push_back(trace);
            trace->activeiter = std::prev(activetrace_.end());
            ++activecount_;
        }

        void TraceDAG::RemoveActive(BlockTrace *trace) {
            if (trace->IsActive()) {
                activetrace_.erase(trace->activeiter);
                trace->flags &= ~BlockTrace::kActive;
                --activecount_;
            }
        }

        void TraceDAG::RemoveTrace(BlockTrace *trace) {
            RemoveActive(trace);
            // If this trace derived a BranchPoint, remove its traces too
            if (trace->derivedbp != nullptr) {
                for (auto *bt : trace->derivedbp->paths) {
                    if (bt != trace)
                        RemoveTrace(bt);
                }
            }
        }

        void TraceDAG::Initialize() {
            for (size_t root_id : rootlist_) {
                auto *bp = new BranchPoint();
                bp->top_id = root_id;
                bp->depth = 0;
                branchlist_.push_back(bp);

                auto *bt = new BlockTrace();
                bt->top = bp;
                bt->pathout = 0;
                bt->bottom_id = CNode::kNone;
                bt->dest_id = root_id;
                bp->paths.push_back(bt);
                InsertActive(bt);
            }
        }

        bool TraceDAG::CheckOpen(CGraph &g, BlockTrace *trace) {
            size_t dest = trace->dest_id;
            if (dest == CNode::kNone || dest >= g.nodes.size()) {
                return true; // terminal
            }

            auto &n = g.Node(dest);
            if (n.collapsed) {
                trace->flags |= BlockTrace::kTerminal;
                return true;
            }

            if (dest == finishblock_id_) {
                trace->flags |= BlockTrace::kTerminal;
                return true;
            }

            // Count DAG-eligible out-edges
            size_t dag_out_count = 0;
            size_t single_succ   = CNode::kNone;
            for (size_t i = 0; i < n.succs.size(); ++i) {
                if (n.IsLoopDagOut(i)) {
                    ++dag_out_count;
                    single_succ = n.succs[i];
                }
            }

            if (dag_out_count == 0) {
                trace->flags |= BlockTrace::kTerminal;
                return true;
            }

            if (dag_out_count == 1) {
                // Linear trace: advance to successor.
                // Increment visit_count on the successor so that the
                // predecessor-readiness check in PushBranches (which gates
                // OpenBranch on visit_count >= dag_preds) sees this arrival.
                // Without this, traces that converge via linear advance never
                // satisfy the readiness check, causing an infinite loop.
                trace->bottom_id = dest;
                trace->dest_id = single_succ;
                if (single_succ < g.nodes.size() && !g.Node(single_succ).collapsed) {
                    g.Node(single_succ).visit_count += 1;
                }
                return true;
            }

            // Multiple out-edges: needs branching
            return false;
        }

        // ---------------------------------------------------------------
        // OpenBranch — replace a single active trace with child traces
        // for each outgoing edge of the branch node.
        //
        //  BEFORE (activetrace_):
        //    ... → [parent] → ...
        //
        //  AFTER:
        //    ... → [child_0] → [child_1] → ... → [next]
        //
        //  `parent` is removed from the active list and a new
        //  BranchPoint is created.  Each non-loop-internal successor
        //  gets a child BlockTrace inserted into the active list,
        //  unless it was already visited (edge-lump case).
        //
        //  Returns an iterator to the element that followed parent
        //  in activetrace_ (i.e. the resumption point for
        //  PushBranches's outer loop).  We must capture this BEFORE
        //  erasing parent — see RetireBranch for the same pattern.
        // ---------------------------------------------------------------
        std::list< TraceDAG::BlockTrace * >::iterator
        TraceDAG::OpenBranch(CGraph &g, BlockTrace *parent) {
            size_t branch_id = parent->dest_id;
            const auto &n    = g.Node(branch_id);

            auto *bp = new BranchPoint();
            bp->parent = parent->top;
            bp->pathout = parent->pathout;
            bp->top_id = branch_id;
            bp->depth = parent->top->depth + 1;
            branchlist_.push_back(bp);
            parent->derivedbp = bp;

            // Capture the next iterator BEFORE erasing parent.
            // RemoveActive erases parent->activeiter, which is also
            // current_activeiter_ in the caller — reading it after
            // erase would be undefined behaviour.
            auto next_iter = std::next(parent->activeiter);
            RemoveActive(parent);

            int pathindex = 0;
            for (size_t i = 0; i < n.succs.size(); ++i) {
                if (!n.IsLoopDagOut(i)) {
                    continue;
                }

                size_t succ_id = n.succs[i];
                auto &succ_node = g.Node(succ_id);

                auto *bt = new BlockTrace();
                bt->top = bp;
                bt->pathout = pathindex++;
                bt->bottom_id = branch_id;
                bt->dest_id = succ_id;
                bp->paths.push_back(bt);

                if (!succ_node.collapsed && succ_node.visit_count > 0) {
                    for (auto *existing : activetrace_) {
                        if (existing->dest_id == succ_id || existing->bottom_id == succ_id) {
                            existing->edgelump += 1;
                            bt->flags          |= BlockTrace::kTerminal;
                            break;
                        }
                    }
                    if (!bt->IsTerminal()) {
                        InsertActive(bt);
                    }
                } else {
                    InsertActive(bt);
                }

                if (!succ_node.collapsed) {
                    succ_node.visit_count += 1;
                }
            }

            return next_iter;
        }

        bool TraceDAG::CheckRetirement(BlockTrace *trace, size_t &exitblock_id) {
            // Gap 2+6: Only first sibling triggers retirement (matches Ghidra)
            if (trace->pathout != 0) return false;
            BranchPoint *bp = trace->top;
            if (bp == nullptr) return false;

            if (bp->depth == 0) {
                // Root BranchPoint: all paths must be active AND terminal
                for (auto *bt : bp->paths) {
                    if (!bt->IsActive()) {
                        return false;
                    }
                    if (!bt->IsTerminal()) {
                        return false;
                    }
                }
                exitblock_id = CNode::kNone;
                return true;
            }

            // Non-root: all must be active; non-terminal paths must converge
            // to the SAME destination. Different exits = DON'T retire (forces
            // selectBadEdge to fire).
            size_t outblock = CNode::kNone;
            for (auto *bt : bp->paths) {
                if (!bt->IsActive()) {
                    return false;
                }
                if (bt->IsTerminal()) {
                    continue;
                }
                if (outblock == bt->dest_id) continue;
                if (outblock != CNode::kNone) {
                    return false; // divergent exits
                }
                outblock = bt->dest_id;
            }
            exitblock_id = outblock;
            return true;
        }

        std::list< TraceDAG::BlockTrace * >::iterator
        TraceDAG::RetireBranch(BranchPoint *bp, size_t exitblock_id) {
            // Remove all traces from this BranchPoint
            std::list< BlockTrace * >::iterator next_iter = current_activeiter_;
            for (auto *bt : bp->paths) {
                if (bt->IsActive()) {
                    auto it = bt->activeiter;
                    if (it == next_iter) ++next_iter;
                    RemoveActive(bt);
                }
            }

            // If bp has a parent BranchPoint, update the parent trace
            if (bp->parent != nullptr) {
                // Find the parent trace that derived this BP
                for (auto *pt : bp->parent->paths) {
                    if (pt->derivedbp == bp) {
                        pt->derivedbp = nullptr;
                        if (exitblock_id != CNode::kNone) {
                            pt->dest_id = exitblock_id;
                            InsertActive(pt);
                        } else {
                            pt->flags |= BlockTrace::kTerminal;
                        }
                        break;
                    }
                }
            }

            return next_iter;
        }

        bool TraceDAG::BadEdgeScore::CompareFinal(const BadEdgeScore &op2) const {
            // Gap 1: Match Ghidra's scoring (blockaction.cc:617-630)
            // "worse" = more likely to be the bad (goto) edge.
            // Bigger sibling count = MORE structural = LESS likely bad.
            if (siblingedge != op2.siblingedge)
                return (op2.siblingedge < siblingedge);
            if (terminal != op2.terminal)
                return (terminal < op2.terminal);
            if (distance != op2.distance)
                return (distance < op2.distance);
            // Less depth = less likely to be bad
            if (trace->top && op2.trace->top)
                return (trace->top->depth < op2.trace->top->depth);
            return false;
        }

        bool TraceDAG::BadEdgeScore::operator<(const BadEdgeScore &op2) const {
            // Sort by exitproto_id for grouping
            return exitproto_id < op2.exitproto_id;
        }

        void TraceDAG::ProcessExitConflict(
            std::list< BadEdgeScore >::iterator start, std::list< BadEdgeScore >::iterator end
        ) {
            // Count traces in this group
            int count = 0;
            for (auto it = start; it != end; ++it) ++count;
            if (count <= 1) return;

            // For each pair, compute distance between their BranchPoints
            for (auto it = start; it != end; ++it) {
                it->siblingedge = count - 1;
                for (auto jt = start; jt != end; ++jt) {
                    if (it == jt) continue;
                    int d = it->trace->top->Distance(jt->trace->top);
                    if (it->distance < 0 || d < it->distance) {
                        it->distance = d;
                    }
                }
            }
        }

        TraceDAG::BlockTrace *TraceDAG::SelectBadEdge() {
            // Build score list from non-terminal active traces (Ghidra skips terminals)
            std::list<BadEdgeScore> scores;
            for (auto *bt : activetrace_) {
                if (bt->IsTerminal()) {
                    continue;
                }
                BadEdgeScore score;
                score.exitproto_id = bt->dest_id;
                score.trace = bt;
                score.terminal = 0;  // non-terminal traces only (terminals filtered above)
                scores.push_back(score);
            }

            if (scores.empty()) return nullptr;

            // Sort by exitproto_id to group by destination
            scores.sort();

            // Process each group of same-destination traces
            auto group_start = scores.begin();
            while (group_start != scores.end()) {
                auto group_end = group_start;
                while (group_end != scores.end() &&
                       group_end->exitproto_id == group_start->exitproto_id) {
                    ++group_end;
                }
                ProcessExitConflict(group_start, group_end);
                group_start = group_end;
            }

            // Find the worst edge
            BlockTrace *worst = nullptr;
            BadEdgeScore worst_score;
            for (auto &s : scores) {
                if (worst == nullptr || s.CompareFinal(worst_score)) {
                    worst = s.trace;
                    worst_score = s;
                }
            }

            return worst;
        }

        // ---------------------------------------------------------------
        // PushBranches — build the trace DAG by iteratively advancing,
        // branching, or retiring active traces.
        //
        //  Each iteration of the inner loop picks one active trace
        //  and attempts, in order:
        //
        //    1. Retire  — all sibling paths of a BranchPoint reached
        //                 the same destination; collapse back to parent.
        //                 Resets missedcount (progress made).
        //    2. Advance — single-successor or newly terminal node;
        //                 slide the trace forward or mark it done.
        //                 Resets missedcount (progress made).
        //       Stall   — trace was already terminal before CheckOpen;
        //                 no progress, increments missedcount.
        //    3. Branch  — multi-successor node with all in-edges visited;
        //                 replace trace with per-successor child traces.
        //                 Resets missedcount (progress made).
        //       Skip    — multi-successor node waiting for in-edges;
        //                 increments missedcount.
        //
        //  When missedcount >= activecount_, every active trace was
        //  skipped without progress → select the worst "bad edge" and
        //  record it as a likely goto to break the cycle.
        // ---------------------------------------------------------------
        void TraceDAG::PushBranches(CGraph &g) {
            ClearVisitCount(g);

            for (size_t root_id : rootlist_) {
                if (root_id < g.nodes.size() && !g.Node(root_id).collapsed) {
                    g.Node(root_id).visit_count = 1;
                }
            }

            // Safety bound: the algorithm should converge in O(edges) iterations
            // now that CheckOpen increments visit_count.  The bound catches any
            // remaining pathological CFGs that defeat convergence.
            const size_t max_outer = g.nodes.size() * g.nodes.size() + 512;
            size_t outer_iter      = 0;
            while (activecount_ > 0) {
                if (++outer_iter > max_outer) {
                    LOG(WARNING) << "PushBranches: iteration limit reached (" << max_outer
                                 << " on " << g.nodes.size() << " nodes), bailing out\n";
                    break;
                }
                int missedcount = 0;
                current_activeiter_ = activetrace_.begin();

                while (current_activeiter_ != activetrace_.end()) {
                    BlockTrace *bt = *current_activeiter_;

                    if (missedcount >= activecount_) {
                        // Stuck: select bad edge
                        BlockTrace *bad = SelectBadEdge();
                        if (bad == nullptr) {
                            ClearVisitCount(g);
                            return;
                        }
                        if (bad->bottom_id != CNode::kNone && bad->dest_id != CNode::kNone) {
                            likelygoto_.emplace_back(bad->bottom_id, bad->dest_id);
                        }
                        RemoveTrace(bad);
                        missedcount = 0;
                        current_activeiter_ = activetrace_.begin();
                        continue;
                    }

                    // Try retirement (only first sibling via pathout==0 check
                    // inside checkRetirement)
                    {
                        size_t exit_id = CNode::kNone;
                        if (CheckRetirement(bt, exit_id)) {
                            current_activeiter_ = RetireBranch(bt->top, exit_id);
                            missedcount = 0;
                            continue;
                        }
                    }

                    {
                        bool was_terminal = bt->IsTerminal();
                        if (CheckOpen(g, bt)) {
                            ++current_activeiter_;
                            if (was_terminal) {
                                // Already terminal before CheckOpen — no progress.
                                ++missedcount;
                            } else {
                                // Trace advanced (linear) or newly terminated.
                                missedcount = 0;
                            }
                            continue;
                        }
                    }

                    // CheckOpen returned false → multi-edge node needs branching.
                    // Only open the branch if all in-edges have been visited
                    // (visit_count >= DAG-eligible pred count).  Otherwise the
                    // node isn't ready — skip it and bump missedcount so the
                    // stuck-detection guard can eventually fire.
                    {
                        const auto &dest_node = g.Node(bt->dest_id);
                        size_t dag_preds      = 0;
                        for (size_t p : dest_node.preds) {
                            if (!g.Node(p).collapsed) {
                                ++dag_preds;
                            }
                        }
                        if (dag_preds > 1
                            && dest_node.visit_count < static_cast< int >(dag_preds))
                        {
                            ++current_activeiter_;
                            ++missedcount;
                            continue;
                        }
                    }

                    current_activeiter_ = OpenBranch(g, bt);
                    missedcount = 0;
                }
            }

            ClearVisitCount(g);
        }

        void TraceDAG::ClearVisitCount(CGraph &g) {
            for (auto &n : g.nodes) {
                n.visit_count = 0;
            }
        }

    } // namespace detail

    // -----------------------------------------------------------------------
    // Anonymous namespace — rule functions and internal helpers
    // -----------------------------------------------------------------------
    namespace {

        using detail::CNode;
        using detail::CGraph;

        // ---------------------------------------------------------------
        // Condition negation helper
        // ---------------------------------------------------------------

        static clang::Expr *NegateCond(clang::Expr *cond, clang::ASTContext &ctx) {
            // Double negation elimination
            if (auto *uo = llvm::dyn_cast<clang::UnaryOperator>(cond)) {
                if (uo->getOpcode() == clang::UO_LNot) {
                    return uo->getSubExpr();
                }
            }
            return clang::UnaryOperator::Create(
                ctx, cond, clang::UO_LNot, ctx.IntTy,
                clang::VK_PRValue, clang::OK_Ordinary,
                clang::SourceLocation(), false,
                clang::FPOptionsOverride());
        }

        // ---------------------------------------------------------------
        // SNode construction helpers
        // ---------------------------------------------------------------

        SNode *LeafFromNode(CNode &n, SNodeFactory &factory) {
            SNode *result;
            if (n.structured) {
                result = n.structured;
            } else {
                auto *block = factory.Make<SBlock>();
                for (auto *s : n.stmts) block->AddStmt(s);
                result = block;
            }
            // Wrap with SLabel if this CNode carries a label from the original CFG.
            // Clear the label after wrapping to prevent double-wrapping if
            // LeafFromNode is called again on the same node.
            if (!n.label.empty()) {
                result = factory.Make<SLabel>(factory.Intern(n.label), result);
                n.label.clear();
            }
            return result;
        }

        // ---------------------------------------------------------------
        // Pattern-matching collapse rules (adapted from Ghidra)
        // ---------------------------------------------------------------

        // Rule: Sequential blocks (A->B chain)
        bool FoldSequence(CGraph &g, size_t id, SNodeFactory &factory) {
            auto &bl = g.Node(id);
            if (bl.collapsed || bl.SizeOut() != 1) {
                return false;
            }
            if (bl.IsSwitchOut()) {
                return false;
            }
            if (!bl.IsDecisionOut(0)) {
                return false;
            }

            // Start-of-chain guard: don't fire mid-chain (prevents nested SSeq)
            if (bl.SizeIn() == 1) {
                auto &pred = g.Node(bl.preds[0]);
                if (!pred.collapsed && pred.SizeOut() == 1) {
                    return false;
                }
            }

            size_t next_id = bl.succs[0];
            if (next_id == id) return false;  // no self-loop
            auto &next = g.Node(next_id);
            if (next.collapsed) return false;
            if (next.SizeIn() != 1) {
                return false;
            }
            // Don't chain into a switch node — FoldSwitch needs it intact.
            if (next.IsSwitchOut()) {
                return false;
            }

            // Build a sequence
            auto *seq = factory.Make<SSeq>();
            seq->AddChild(LeafFromNode(bl, factory));

            // Extend the chain
            std::vector<size_t> chain = {id, next_id};
            size_t cur = next_id;
            while (g.Node(cur).SizeOut() == 1 && g.Node(cur).IsDecisionOut(0)) {
                size_t nxt = g.Node(cur).succs[0];
                if (nxt == id) break;
                auto &nxtNode = g.Node(nxt);
                if (nxtNode.collapsed || nxtNode.SizeIn() != 1) {
                    break;
                }
                if (nxtNode.IsSwitchOut()) {
                    break;
                }
                chain.push_back(nxt);
                cur = nxt;
            }

            for (size_t i = 1; i < chain.size(); ++i) {
                seq->AddChild(LeafFromNode(g.Node(chain[i]), factory));
            }

            g.CollapseNodes(chain, seq);
            return true;
        }

        // Rule: If without else (proper if)
        bool FoldIfThen(CGraph &g, size_t id, SNodeFactory &factory,
                               clang::ASTContext &ctx) {
            auto &bl = g.Node(id);
            if (bl.collapsed || bl.SizeOut() != 2) {
                return false;
            }
            if (bl.IsSwitchOut()) {
                return false;
            }
            if (bl.IsGotoOut(0) || bl.IsGotoOut(1)) {
                return false;
            }

            for (size_t i = 0; i < 2; ++i) {
                size_t clause_id = bl.succs[i];
                if (clause_id == id) continue;
                auto &clause = g.Node(clause_id);
                if (clause.collapsed) continue;
                if (clause.SizeIn() != 1 || clause.SizeOut() != 1) {
                    continue;
                }
                if (!bl.IsDecisionOut(i)) {
                    continue;
                }
                if (clause.IsGotoOut(0)) {
                    continue;
                }

                size_t exit_id = clause.succs[0];
                if (exit_id != bl.succs[1 - i]) continue;

                // Build the if
                clang::Expr *cond = bl.branch_cond;
                if (!cond) {
                    // Create a placeholder true literal
                    cond = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                }

                // If clause is on false branch (i==0), negate condition
                // succs[0]=false branch per Ghidra convention
                if (i == 0) {
                    cond = NegateCond(cond, ctx);
                }

                SNode *clause_body = LeafFromNode(clause, factory);
                SNode *if_node = factory.Make<SIfThenElse>(cond, clause_body, nullptr);

                // Prepend head block's accumulated content so it is not lost.
                SNode *head_content = LeafFromNode(bl, factory);
                bool has_head = false;
                if (head_content) {
                    if (head_content->Kind() == SNodeKind::BLOCK) {
                        has_head = !head_content->as<SBlock>()->Stmts().empty();
                    } else {
                        has_head = true;
                    }
                }
                if (has_head) {
                    auto *seq = factory.Make<SSeq>();
                    seq->AddChild(head_content);
                    seq->AddChild(if_node);
                    if_node = seq;
                }

                if (!bl.label.empty()) {
                    if_node = factory.Make<SLabel>(factory.Intern(bl.label), if_node);
                    bl.label.clear();
                }

                g.CollapseNodes({ id, clause_id }, if_node);
                return true;
            }
            return false;
        }

        // Rule: If-else
        bool FoldIfElse(CGraph &g, size_t id, SNodeFactory &factory,
                             clang::ASTContext &ctx) {
            auto &bl = g.Node(id);
            if (bl.collapsed || bl.SizeOut() != 2) {
                return false;
            }
            if (bl.IsSwitchOut()) {
                return false;
            }
            if (!bl.IsDecisionOut(0) || !bl.IsDecisionOut(1)) {
                return false;
            }

            size_t tc_id = bl.succs[1];  // true clause (Ghidra: out[1])
            size_t fc_id = bl.succs[0];  // false clause (Ghidra: out[0])
            auto &tc     = g.Node(tc_id);
            auto &fc     = g.Node(fc_id);

            if (tc.collapsed || fc.collapsed) return false;
            if (tc.SizeIn() != 1 || fc.SizeIn() != 1) {
                return false;
            }
            if (tc.SizeOut() != 1 || fc.SizeOut() != 1) {
                return false;
            }
            if (tc.succs[0] != fc.succs[0]) return false;  // must exit to same block
            if (tc.succs[0] == id) return false;  // no loops
            if (tc.IsGotoOut(0) || fc.IsGotoOut(0)) {
                return false;
            }

            clang::Expr *cond = bl.branch_cond;
            if (!cond) {
                cond = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
            }

            auto *then_body = LeafFromNode(tc, factory);
            auto *else_body = LeafFromNode(fc, factory);
            SNode *if_node = factory.Make<SIfThenElse>(cond, then_body, else_body);

            // Prepend the head block's accumulated content (from prior
            // collapses) so it is not lost.  LeafFromNode returns the
            // structured SNode tree built by earlier rules.
            SNode *head_content = LeafFromNode(bl, factory);
            bool has_content = false;
            if (head_content) {
                if (head_content->Kind() == SNodeKind::BLOCK) {
                    has_content = !head_content->as<SBlock>()->Stmts().empty();
                } else {
                    has_content = true;
                }
            }
            if (has_content) {
                auto *seq = factory.Make<SSeq>();
                seq->AddChild(head_content);
                seq->AddChild(if_node);
                if_node = seq;
            }

            if (!bl.label.empty()) {
                if_node = factory.Make<SLabel>(factory.Intern(bl.label), if_node);
                bl.label.clear();
            }

            g.CollapseNodes({ id, tc_id, fc_id }, if_node);
            return true;
        }

        // Rule: While-do loop
        bool FoldWhileLoop(CGraph &g, size_t id, SNodeFactory &factory,
                              clang::ASTContext &ctx) {
            auto &bl = g.Node(id);
            if (bl.collapsed || bl.SizeOut() != 2) {
                return false;
            }
            if (bl.IsSwitchOut()) {
                return false;
            }
            if (bl.IsGotoOut(0) || bl.IsGotoOut(1)) {
                return false;
            }

            for (size_t i = 0; i < 2; ++i) {
                size_t clause_id = bl.succs[i];
                if (clause_id == id) continue;
                auto &clause = g.Node(clause_id);
                if (clause.collapsed) continue;
                if (clause.SizeIn() != 1 || clause.SizeOut() != 1) {
                    continue;
                }
                if (clause.succs[0] != id) continue;  // must loop back

                clang::Expr *cond = bl.branch_cond;
                if (!cond) {
                    cond = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                }

                // Negate condition when body is on false branch
                if (i == 0) {
                    cond = NegateCond(cond, ctx);
                }

                SNode *clause_body = LeafFromNode(clause, factory);

                // If the header block has statements (e.g. getopt_long()),
                // they execute each iteration BEFORE the condition check.
                // Emit as: while(true) { header_stmts; if(exit_cond) break; body; }
                SNode *while_node;
                if (!bl.stmts.empty()) {
                    auto *true_lit = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());

                    // Exit condition: the original condition on the exit branch
                    clang::Expr *exit_cond = bl.branch_cond;
                    if (!exit_cond) {
                        exit_cond = clang::IntegerLiteral::Create(
                            ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                    }
                    // When body is on false branch (i==0), exit is on true branch
                    // → exit_cond is already the exit condition (no negation)
                    // When body is on true branch (i==1), exit is on false branch
                    // → negate to get exit condition
                    if (i == 1) {
                        exit_cond = NegateCond(exit_cond, ctx);
                    }

                    auto *break_node = factory.Make<SBreak>();
                    auto *if_break = factory.Make<SIfThenElse>(exit_cond, break_node, nullptr);

                    SNode *header_block = factory.Make<SBlock>();
                    for (auto *s : bl.stmts) static_cast<SBlock *>(header_block)->AddStmt(s);
                    if (!bl.label.empty()) {
                        header_block = factory.Make<SLabel>(factory.Intern(bl.label), header_block);
                    }

                    auto *seq = factory.Make<SSeq>();
                    seq->AddChild(header_block);
                    seq->AddChild(if_break);
                    seq->AddChild(clause_body);
                    while_node = factory.Make<SWhile>(true_lit, seq);
                } else {
                    while_node = factory.Make<SWhile>(cond, clause_body);
                    if (!bl.label.empty()) {
                        while_node = factory.Make<SLabel>(factory.Intern(bl.label), while_node);
                        bl.label.clear();
                    }
                }

                g.CollapseNodes({ id, clause_id }, while_node);
                return true;
            }
            return false;
        }

        // Rule: Do-while loop (single block looping to itself)
        bool FoldDoWhileLoop(CGraph &g, size_t id, SNodeFactory &factory,
                              clang::ASTContext &ctx) {
            auto &bl = g.Node(id);
            if (bl.collapsed || bl.SizeOut() != 2) {
                return false;
            }
            if (bl.IsSwitchOut()) {
                return false;
            }
            if (bl.IsGotoOut(0) || bl.IsGotoOut(1)) {
                return false;
            }

            for (size_t i = 0; i < 2; ++i) {
                if (bl.succs[i] != id) continue;  // must loop to self

                clang::Expr *cond = bl.branch_cond;
                if (!cond) {
                    cond = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                }

                // Negate condition when self-loop is on false branch
                if (i == 0) {
                    cond = NegateCond(cond, ctx);
                }

                // Save label before LeafFromNode clears it — for loops,
                // the label should wrap the entire loop (not be inside the body)
                std::string saved_label = bl.label;
                bl.label.clear();
                auto *body = LeafFromNode(bl, factory);
                SNode *dowhile_node = factory.Make<SDoWhile>(body, cond);
                if (!saved_label.empty()) {
                    dowhile_node = factory.Make<SLabel>(factory.Intern(saved_label), dowhile_node);
                }

                // Use collapseNodes to handle edge cleanup uniformly
                g.CollapseNodes({ id }, dowhile_node);
                g.Node(id).is_conditional = false;
                g.Node(id).branch_cond    = nullptr;
                return true;
            }
            return false;
        }

        // Rule: Infinite loop (single out to self)
        bool FoldInfiniteLoop(CGraph &g, size_t id, SNodeFactory &factory,
                              clang::ASTContext &ctx) {
            auto &bl = g.Node(id);
            if (bl.collapsed || bl.SizeOut() != 1) {
                return false;
            }
            if (bl.IsGotoOut(0)) {
                return false;
            }
            if (bl.succs[0] != id) return false;  // must loop to self

            std::string saved_label = bl.label;
            bl.label.clear();
            auto *body = LeafFromNode(bl, factory);
            auto *true_lit = clang::IntegerLiteral::Create(
                ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
            SNode *loop = factory.Make<SWhile>(true_lit, body);
            if (!saved_label.empty()) {
                loop = factory.Make<SLabel>(factory.Intern(saved_label), loop);
            }

            // Use collapseNodes to handle edge cleanup uniformly
            g.CollapseNodes({ id }, loop);
            return true;
        }

        // Rule: If with no exit (clause has zero out edges)
        bool FoldIfForcedGoto(CGraph &g, size_t id, SNodeFactory &factory,
                               clang::ASTContext &ctx) {
            auto &bl = g.Node(id);
            if (bl.collapsed || bl.SizeOut() != 2) {
                return false;
            }
            if (bl.IsSwitchOut()) {
                return false;
            }
            if (bl.IsGotoOut(0) || bl.IsGotoOut(1)) {
                return false;
            }

            for (size_t i = 0; i < 2; ++i) {
                size_t clause_id = bl.succs[i];
                if (clause_id == id) continue;
                auto &clause = g.Node(clause_id);
                if (clause.collapsed) continue;
                if (clause.SizeIn() != 1 || clause.SizeOut() != 0) {
                    continue;
                }
                if (!bl.IsDecisionOut(i)) {
                    continue;
                }

                clang::Expr *cond = bl.branch_cond;
                if (!cond) {
                    cond = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                }

                // If clause is on false branch (i==0), negate condition
                if (i == 0) {
                    cond = NegateCond(cond, ctx);
                }

                auto *clause_body = LeafFromNode(clause, factory);
                SNode *if_node = factory.Make<SIfThenElse>(cond, clause_body, nullptr);

                // Prepend the head block's accumulated content (from prior
                // collapses) so it is not lost.
                SNode *head_content = LeafFromNode(bl, factory);
                bool has_head = false;
                if (head_content) {
                    if (head_content->Kind() == SNodeKind::BLOCK) {
                        has_head = !head_content->as<SBlock>()->Stmts().empty();
                    } else {
                        has_head = true;
                    }
                }
                if (has_head) {
                    auto *seq = factory.Make<SSeq>();
                    seq->AddChild(head_content);
                    seq->AddChild(if_node);
                    if_node = seq;
                }

                if (!bl.label.empty()) {
                    if_node = factory.Make<SLabel>(factory.Intern(bl.label), if_node);
                    bl.label.clear();
                }

                g.CollapseNodes({ id, clause_id }, if_node);
                return true;
            }
            return false;
        }

        // Rule: Switch statement
        bool FoldSwitch(CGraph &g, size_t id, SNodeFactory &factory,
                             clang::ASTContext &ctx) {
            auto &bl = g.Node(id);
            if (bl.collapsed || !bl.IsSwitchOut()) {
                return false;
            }
            // Must have switch_cases metadata to fire.
            if (bl.switch_cases.empty()) return false;

            // Find exit block: look for a successor with sizeIn > 1 or sizeOut > 1
            size_t exit_id = std::numeric_limits<size_t>::max();
            for (size_t s : bl.succs) {
                auto &sn = g.Node(s);
                if (sn.collapsed) continue;
                if (s == id || sn.SizeIn() > 1 || sn.SizeOut() > 1) {
                    exit_id = s;
                    break;
                }
            }

            // Validate: each case must have sizeIn==1.
            // Cases may exit to exit_id (break), have no exit (tail), or exit
            // to a different target (goto — e.g., loop back-edge).
            for (size_t s : bl.succs) {
                if (s == exit_id) continue;
                auto &sn = g.Node(s);
                if (sn.collapsed) return false;
                if (sn.SizeIn() != 1) {
                    return false;
                }
                if (sn.SizeOut() > 1) {
                    return false;
                }
            }

            // Build the switch SNode from branch_cond and successor nodes.
            clang::Expr *disc = bl.branch_cond;
            if (!disc) {
                disc = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(32, 0), ctx.IntTy, clang::SourceLocation());
            }
            auto *sw = factory.Make<SSwitch>(disc);

            // Build a map from succ_index to case values (multiple cases
            // can target the same successor).
            std::unordered_map<size_t, std::vector<int64_t>> succ_to_values;
            for (const auto &entry : bl.switch_cases) {
                succ_to_values[entry.succ_index].push_back(entry.value);
            }

            // Build case arms.  When a successor has multiple case values
            // (e.g. case 1: case 2: case 3: body), only the LAST value
            // carries the body; preceding values get nullptr (fallthrough).
            unsigned iw = ctx.getIntWidth(ctx.IntTy);
            for (size_t si = 0; si < bl.succs.size(); ++si) {
                size_t s = bl.succs[si];
                if (s == exit_id) continue;
                auto it = succ_to_values.find(si);
                if (it == succ_to_values.end()) continue;

                const auto &vals = it->second;
                auto *body = LeafFromNode(g.Node(s), factory);
                for (size_t vi = 0; vi < vals.size(); ++vi) {
                    auto *case_val = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(iw, static_cast<uint64_t>(vals[vi]), true),
                        ctx.IntTy, clang::SourceLocation());
                    bool is_last = (vi + 1 == vals.size());
                    sw->AddCase(case_val, is_last ? body : nullptr);
                }
            }
            // Prepend the switch block's own stmts (ops before the switch)
            // and strip the original SwitchStmt (CfgFoldStructure rebuilds it).
            SNode *sw_node = nullptr;
            if (!bl.stmts.empty()) {
                auto *seq = factory.Make<SSeq>();
                auto *block = factory.Make<SBlock>();
                for (auto *s : bl.stmts) {
                    // Skip the original SwitchStmt/CompoundStmt containing it
                    if (llvm::isa<clang::SwitchStmt>(s)) continue;
                    if (auto *cs = llvm::dyn_cast<clang::CompoundStmt>(s)) {
                        bool has_sw = false;
                        for (auto *child : cs->body())
                            if (llvm::isa<clang::SwitchStmt>(child)) { has_sw = true; break; }
                        if (has_sw) continue;
                    }
                    block->AddStmt(s);
                }
                if (!block->Stmts().empty()) seq->AddChild(block);
                seq->AddChild(sw);
                sw_node = seq;
            } else {
                sw_node = sw;
            }

            // Collapse only case successor nodes (those with case values).
            // Successors without case values (fallback paths) stay as
            // separate blocks so they can be collapsed by later rules.
            std::vector<size_t> collapse_ids = {id};
            for (size_t si = 0; si < bl.succs.size(); ++si) {
                size_t s = bl.succs[si];
                if (s == exit_id) continue;
                if (succ_to_values.find(si) == succ_to_values.end()) continue;
                collapse_ids.push_back(s);
            }

            if (!bl.label.empty()) {
                sw_node = factory.Make<SLabel>(factory.Intern(bl.label), sw_node);
                bl.label.clear();
            }
            size_t rep                 = g.CollapseNodes(collapse_ids, sw_node);
            // The collapsed node has no real branch condition — prevent
            // FoldGoto from emitting a spurious if(1) goto.
            g.Node(rep).is_conditional = false;
            g.Node(rep).switch_cases.clear();
            return true;
        }

        // Rule: Mark goto edges
        bool FoldGoto(CGraph &g, size_t id, SNodeFactory &factory,
                           clang::ASTContext &ctx) {
            auto &bl = g.Node(id);
            if (bl.collapsed) return false;

            for (size_t i = 0; i < bl.succs.size(); ++i) {
                if (!bl.IsGotoOut(i)) {
                    continue;
                }

                // Use the target CNode's actual label if available, otherwise
                // fall back to synthetic "block_N" label.
                auto &target_node        = g.Node(bl.succs[i]);
                std::string target_label = target_node.label.empty()
                    ? "block_" + std::to_string(bl.succs[i])
                    : target_node.label;
                auto *goto_node = factory.Make<SGoto>(factory.Intern(target_label));

                // Conditional block with one goto edge: emit if(cond) goto
                // and keep the non-goto successor edge.
                if (bl.is_conditional && bl.succs.size() == 2) {
                    clang::Expr *cond = bl.branch_cond;
                    if (!cond) {
                        cond = clang::IntegerLiteral::Create(
                            ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                    }
                    // If goto is on the false branch (i==0), the goto fires
                    // when cond is false → emit if(!cond) goto
                    // If goto is on the true branch (i==1), emit if(cond) goto
                    if (i == 0) {
                        cond = NegateCond(cond, ctx);
                    }

                    auto *if_goto = factory.Make<SIfThenElse>(cond, goto_node, nullptr);

                    auto *seq = factory.Make<SSeq>();
                    if (!bl.stmts.empty() || bl.structured) {
                        // LeafFromNode embeds the label around stmts/structured
                        seq->AddChild(LeafFromNode(bl, factory));
                    }
                    seq->AddChild(if_goto);

                    SNode *result = seq;
                    // If LeafFromNode wasn't called, wrap with label now
                    if (bl.stmts.empty() && !bl.structured && !bl.label.empty()) {
                        result = factory.Make<SLabel>(factory.Intern(bl.label), result);
                    }
                    bl.structured = result;
                    bl.label.clear();
                    bl.is_conditional = false;
                    bl.branch_cond = nullptr;
                    g.RemoveEdge(id, bl.succs[i]);
                    return true;
                }

                // Unconditional goto: wrap stmts + goto
                auto *body = LeafFromNode(bl, factory);
                auto *seq = factory.Make<SSeq>();
                seq->AddChild(body);
                seq->AddChild(goto_node);

                bl.structured = seq;
                bl.label.clear();  // label embedded via LeafFromNode
                g.RemoveEdge(id, bl.succs[i]);
                return true;
            }
            return false;
        }

        // Gap 4: Switch case fallthrough detection (late-stage fallback)
        bool FoldCaseFallthrough(CGraph &g, size_t id) {
            auto &bl = g.Node(id);
            if (bl.collapsed || !bl.IsSwitchOut()) {
                return false;
            }

            std::vector<size_t> fallthru;
            int nonfallthru = 0;

            for (size_t i = 0; i < bl.succs.size(); ++i) {
                size_t case_id = bl.succs[i];
                if (case_id == id) return false;
                auto &casebl = g.Node(case_id);

                if (casebl.SizeIn() > 2 || casebl.SizeOut() > 1) {
                    nonfallthru++;
                } else if (casebl.SizeOut() == 1) {
                    size_t target_id = casebl.succs[0];
                    auto &target     = g.Node(target_id);
                    if (target.SizeIn() == 2 && target.SizeOut() <= 1) {
                        bool other_is_switch = false;
                        for (size_t p : target.preds) {
                            if (p != case_id && p == id) other_is_switch = true;
                        }
                        if (other_is_switch) fallthru.push_back(case_id);
                    }
                }
                if (nonfallthru > 1) return false;
            }

            if (fallthru.empty()) return false;

            for (size_t fid : fallthru) {
                g.Node(fid).SetGoto(0);
            }
            return true;
        }

        // ---------------------------------------------------------------
        // AND/OR condition collapsing (ResolveConditionChain)
        // ---------------------------------------------------------------

        // Detects chained if-gotos and collapses them into compound conditions.
        // OR pattern (i==0): bl->false leads to orblock, both reach same clauseblock => BO_LOr
        // AND pattern (i==1): bl->true leads to orblock, both reach same clauseblock => BO_LAnd
        static bool ResolveConditionChain(CGraph &g, size_t id, SNodeFactory &/*factory*/,
                                 clang::ASTContext &ctx) {
            auto &bl = g.Node(id);
            if (bl.collapsed || bl.SizeOut() != 2) {
                return false;
            }
            if (bl.IsSwitchOut()) {
                return false;
            }
            if (bl.IsGotoOut(0) || bl.IsGotoOut(1)) {
                return false;
            }

            for (size_t i = 0; i < 2; ++i) {
                size_t or_id = bl.succs[i];
                if (or_id == id) continue;
                auto &orblock = g.Node(or_id);
                if (orblock.collapsed) continue;
                if (orblock.SizeIn() != 1 || orblock.SizeOut() != 2) {
                    continue;
                }
                if (orblock.IsSwitchOut()) {
                    continue;
                }
                if (bl.IsBackEdge(i)) {
                    continue;
                }
                // Don't collapse if orblock has side-effectful statements
                // that would be lost (e.g. function calls between conditions).
                if (!orblock.stmts.empty()) continue;

                size_t clause_id = bl.succs[1 - i];
                if (clause_id == id || clause_id == or_id) continue;

                // Find which orblock successor matches clause_id
                size_t j = 2;
                for (size_t jj = 0; jj < 2; ++jj) {
                    if (orblock.succs[jj] == clause_id) { j = jj; break; }
                }
                if (j == 2) continue;
                if (orblock.succs[1 - j] == id) continue; // no looping back

                // Determine opcode:
                // succs[0]=false, succs[1]=true (Ghidra convention)
                // i==0: bl's FALSE branch leads to orblock => OR pattern
                // i==1: bl's TRUE branch leads to orblock => AND pattern
                bool is_or = (i == 0);
                clang::BinaryOperatorKind op = is_or ? clang::BO_LOr : clang::BO_LAnd;

                clang::Expr *cond_a = bl.branch_cond;
                clang::Expr *cond_b = orblock.branch_cond;
                if (!cond_a || !cond_b) continue;

                // Condition normalization
                if (is_or) {
                    // OR: bl's false leads to orblock, bl's true leads to clauseblock
                    // If j==0, orblock's false leads to clauseblock -- negate cond_b
                    if (j == 0) {
                        cond_b = NegateCond(cond_b, ctx);
                    }
                } else {
                    // AND: bl's true leads to orblock, bl's false leads to clauseblock
                    // If j==1, orblock's true leads to clause -- negate cond_b
                    if (j == 1) {
                        cond_b = NegateCond(cond_b, ctx);
                    }
                }

                auto *compound = clang::BinaryOperator::Create(
                    ctx, cond_a, cond_b, op, ctx.IntTy,
                    clang::VK_PRValue, clang::OK_Ordinary,
                    clang::SourceLocation(), clang::FPOptionsOverride());

                // Rewire: bl absorbs orblock
                bl.branch_cond = compound;
                size_t new_other = orblock.succs[1 - j];
                bl.succs[i] = new_other;

                // Update pred list of new_other: remove or_id, add id if not present
                auto &np = g.Node(new_other).preds;
                np.erase(std::remove(np.begin(), np.end(), or_id), np.end());
                if (std::find(np.begin(), np.end(), id) == np.end()) {
                    np.push_back(id);
                }

                // Remove orblock from clause_id's preds
                auto &cp = g.Node(clause_id).preds;
                cp.erase(std::remove(cp.begin(), cp.end(), or_id), cp.end());

                // Mark orblock collapsed
                orblock.collapsed = true;
                orblock.succs.clear();
                orblock.preds.clear();
                return true;
            }
            return false;
        }

        // ---------------------------------------------------------------
        // AND/OR condition collapsing orchestrator
        // ---------------------------------------------------------------

        static void ResolveAllConditionChains(CGraph &g, SNodeFactory &factory,
                                        clang::ASTContext &ctx) {
            // Iteratively apply ResolveConditionChain until no more changes.
            // This collapses all AND/OR chains before the main collapse loop.
            bool changed = true;
            while (changed) {
                changed = false;
                for (auto &n : g.nodes) {
                    if (n.collapsed) continue;
                    if (ResolveConditionChain(g, n.id, factory, ctx)) {
                        changed = true;
                        break; // restart iteration since graph changed
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // TraceDAG-based goto selection (replaces selectAndMarkGoto)
        // ---------------------------------------------------------------

        /// ResolveDisconnectedRoots: final fallback for unreachable/disconnected regions.
        /// Finds active nodes with no predecessors (other than entry) and adds
        /// a FloatingEdge for an incoming edge to reconnect them via goto.
        static void
        ResolveDisconnectedRoots(CGraph &g, std::list< detail::FloatingEdge > &likelygoto_) {
            for (auto &n : g.nodes) {
                if (n.collapsed) continue;
                if (n.id == g.entry) continue;
                if (n.SizeIn() != 0) {
                    continue;
                }

                // Extra root: find any active predecessor that has an edge to it
                for (auto &pred : g.nodes) {
                    if (pred.collapsed) continue;
                    for (size_t i = 0; i < pred.succs.size(); ++i) {
                        if (pred.succs[i] == n.id && !pred.IsGotoOut(i)) {
                            likelygoto_.emplace_back(pred.id, n.id);
                            goto found_pred;
                        }
                    }
                }
                found_pred:;
            }
        }

        /// ResolveLoopBodyTracing: run TraceDAG loop-scoped (innermost-first),
        /// then fall back to full-graph TraceDAG.
        static bool ResolveLoopBodyTracing(
            CGraph &g, std::list< detail::LoopBody > &loopbody,
            std::list< detail::FloatingEdge > &likelygoto_
        ) {
            // Try each loop body (innermost-first)
            for (auto &loop : loopbody) {
                if (!loop.Update(g)) {
                    continue;
                }

                std::vector<size_t> body;
                loop.FindBase(g, body);
                loop.SetExitMarks(g, body);

                detail::TraceDAG dag(likelygoto_);
                dag.AddRoot(loop.head);
                if (loop.exit_block != detail::LoopBody::kNone
                    && loop.exit_block < g.nodes.size() && !g.Node(loop.exit_block).collapsed)
                {
                    dag.SetFinishBlock(loop.exit_block);
                }
                dag.Initialize();
                dag.PushBranches(g);

                loop.ClearExitMarks(g, body);
                detail::ClearMarks(g, body);

                if (!likelygoto_.empty()) {
                    return true;
                }

                // Post-TraceDAG heuristic: if TraceDAG found nothing within
                // the loop body, look for "extra loop exits" — edges from
                // non-header body nodes to outside the body.  These must
                // become gotos for the body to collapse into a chainable
                // single-successor structure required by FoldWhileLoop.
                // Prefer the deepest (last-in-body-order) edge first.
                {
                    std::unordered_set<size_t> bodyset(body.begin(), body.end());
                    for (auto rit = body.rbegin(); rit != body.rend(); ++rit) {
                        size_t bid = *rit;
                        if (bid == loop.head) continue;
                        auto &bn = g.Node(bid);
                        for (size_t i = 0; i < bn.succs.size(); ++i) {
                            if (bn.IsGotoOut(i) || bn.IsBackEdge(i)) {
                                continue;
                            }
                            if (bodyset.count(bn.succs[i]) == 0) {
                                likelygoto_.emplace_back(bid, bn.succs[i]);
                            }
                        }
                        if (!likelygoto_.empty()) {
                            return true;
                        }
                    }
                }
            }

            // Fall back to full-graph TraceDAG
            {
                detail::TraceDAG dag(likelygoto_);

                // Add active root nodes (entry or nodes with no predecessors)
                bool has_root = false;
                for (auto &n : g.nodes) {
                    if (n.collapsed) continue;
                    if (n.id == g.entry || n.SizeIn() == 0) {
                        dag.AddRoot(n.id);
                        has_root = true;
                    }
                }
                if (!has_root) return false;

                dag.Initialize();
                dag.PushBranches(g);
                return !likelygoto_.empty();
            }
        }

        /// ResolveGotoSelection: pick the least-disruptive edge to mark as goto using TraceDAG.
        /// Replaces the old selectAndMarkGoto heuristic.
        static bool ResolveGotoSelection(CGraph &g, std::list<detail::LoopBody> &loopbody) {
            std::list< detail::FloatingEdge > likelygoto_;

            if (!ResolveLoopBodyTracing(g, loopbody, likelygoto_)) {
                // TraceDAG found nothing; try ResolveDisconnectedRoots as final fallback
                ResolveDisconnectedRoots(g, likelygoto_);
            }

            // Iterate likelygoto_ and mark the first valid edge as goto
            for (auto &fe : likelygoto_) {
                auto [src, edge_idx] = fe.GetCurrentEdge(g);
                if (src != CNode::kNone) {
                    g.Node(src).SetGoto(edge_idx);
                    return true;
                }
            }

            return false;
        }

        // ---------------------------------------------------------------
        // Guard-chain absorption — mark guard→fallback edges as goto so
        // FoldGoto + FoldSequence chain the guards into the switch,
        // reducing the fallback block's sizeIn and unblocking FoldSwitch.
        // ---------------------------------------------------------------

        /// Detect a linear chain of conditional guards leading into a switch
        /// block, where all guards share a common "bail" target that is also
        /// the switch's fallback successor.
        ///
        /// Pattern:
        ///   G1(cond) → G2, T     G1 guards: if(cond) goto T; else fall to G2
        ///   G2(cond) → S,  T     G2 guards: if(cond) goto T; else fall to S
        ///   S(switch) → cases..., T   switch fallback → T
        ///
        /// T has sizeIn > 1 because G1, G2, and S all have edges to it.
        /// FoldSwitch rejects this because T.sizeIn != 1.
        ///
        /// Fix: mark each guard's edge to T as kGoto. Then:
        ///   FoldGoto converts guards to "if(cond) goto T" (sizeOut→1)
        ///   FoldSequence chains guards into the switch predecessor
        ///   T.sizeIn drops → FoldSwitch can fire
        static bool ResolveSwitchGuards(CGraph &g) {
            bool changed = false;

            for (auto &sw : g.nodes) {
                if (sw.collapsed || !sw.IsSwitchOut()) {
                    continue;
                }
                if (sw.SizeIn() != 1) {
                    continue; // switch needs unique predecessor
                }

                // Find the fallback target: a successor of the switch with sizeIn > 1.
                // This is the block that both the guard chain and the switch's
                // fallback edge point to.
                size_t fallback_id = CNode::kNone;
                for (size_t s : sw.succs) {
                    if (g.Node(s).SizeIn() > 1 && !g.Node(s).collapsed) {
                        // Prefer a non-conditional single-exit block as fallback
                        // (the exit/continue block, not the loop header).
                        auto &candidate = g.Node(s);
                        if (!candidate.is_conditional && candidate.SizeOut() <= 1) {
                            fallback_id = s;
                            break;
                        }
                    }
                }
                if (fallback_id == CNode::kNone) {
                    continue;
                }

                // Walk backwards from the switch through the unique predecessor
                // chain, collecting guard blocks that have one edge to the
                // fallback target.
                std::vector<size_t> guards;
                size_t cur = sw.preds[0];
                while (true) {
                    auto &gn = g.Node(cur);
                    if (gn.collapsed) break;
                    if (!gn.is_conditional || gn.SizeOut() != 2) {
                        break;
                    }
                    if (gn.SizeIn() < 1) {
                        break;
                    }

                    // One successor must be the next block in the chain (or the
                    // switch), and the other must be the fallback target.
                    bool has_fallback_edge = false;
                    for (size_t i = 0; i < 2; ++i) {
                        if (gn.succs[i] == fallback_id && !gn.IsGotoOut(i)) {
                            has_fallback_edge = true;
                        }
                    }
                    if (!has_fallback_edge) break;

                    guards.push_back(cur);

                    // Continue walking if this guard has a unique predecessor
                    if (gn.SizeIn() != 1) {
                        break;
                    }
                    cur = gn.preds[0];
                }

                if (guards.empty()) continue;

                // Mark each guard's edge to the fallback as kGoto
                for (size_t gid : guards) {
                    auto &gn = g.Node(gid);
                    for (size_t i = 0; i < gn.succs.size(); ++i) {
                        if (gn.succs[i] == fallback_id && !gn.IsGotoOut(i)) {
                            gn.SetGoto(i);
                            LOG(INFO) << "ResolveSwitchGuards: marked guard "
                                      << gid << " edge to fallback " << fallback_id
                                      << " as goto (for switch " << sw.id << ")\n";
                        }
                    }
                }
                changed = true;
            }

            return changed;
        }

        // ---------------------------------------------------------------
        // Control-equivalence hoisting — unblock FoldIfThen /
        // FoldIfElse by duplicating or absorbing small shared blocks.
        // ---------------------------------------------------------------

        /// Clause splitting: when a conditional block's clause has sizeIn > 1
        /// (shared with other predecessors), duplicate it so the conditional
        /// can fire FoldIfThen.
        ///
        /// Pattern: cond→clause→other, cond→other  (clause.sizeIn > 1)
        /// After:   cond→clause_copy→other, cond→other  (clause_copy.sizeIn == 1)
        static bool ResolveClauseSplit(CGraph &g) {
            for (auto &bl : g.nodes) {
                if (bl.collapsed || bl.SizeOut() != 2 || !bl.is_conditional) {
                    continue;
                }
                if (bl.IsGotoOut(0) || bl.IsGotoOut(1)) {
                    continue;
                }

                for (size_t i = 0; i < 2; ++i) {
                    size_t clause_id = bl.succs[i];
                    if (clause_id == bl.id) continue;
                    auto &clause = g.Node(clause_id);
                    if (clause.collapsed) continue;

                    // Clause must exit to the other successor of cond
                    if (clause.SizeOut() != 1) {
                        continue;
                    }
                    if (clause.succs[0] != bl.succs[1 - i]) continue;

                    // Already unique — FoldIfThen should handle it
                    if (clause.SizeIn() == 1) {
                        continue;
                    }

                    // Don't duplicate conditionals, large blocks, labels,
                    // or blocks with back-edge predecessors
                    if (clause.is_conditional) continue;
                    if (clause.stmts.size() > 16) continue;
                    if (!clause.label.empty()) continue;
                    if (clause.IsGotoOut(0)) {
                        continue;
                    }

                    bool has_back_pred = false;
                    for (size_t p : clause.preds) {
                        auto &pn = g.Node(p);
                        for (size_t ei = 0; ei < pn.succs.size(); ++ei) {
                            if (pn.succs[ei] == clause_id && pn.IsBackEdge(ei)) {
                                has_back_pred = true;
                                break;
                            }
                        }
                        if (has_back_pred) break;
                    }
                    if (has_back_pred) continue;

                    // --- Duplicate clause as a new node ---
                    // Capture ids before push_back — the vector realloc
                    // invalidates bl and clause references.
                    size_t cond_id   = bl.id;
                    size_t clause_exit = clause.succs[0];

                    CNode copy;
                    copy.stmts = clause.stmts;
                    copy.branch_cond = clause.branch_cond;
                    copy.is_conditional = clause.is_conditional;
                    copy.structured = clause.structured;

                    size_t copy_id = g.nodes.size();
                    copy.id = copy_id;
                    copy.preds = {cond_id};
                    copy.succs = {clause_exit};
                    copy.edge_flags = {clause.edge_flags[0]};

                    g.nodes.push_back(std::move(copy));
                    // bl, clause now dangling — only use ids + g.Node()

                    g.Node(cond_id).succs[i] = copy_id;

                    auto &cpreds = g.Node(clause_id).preds;
                    cpreds.erase(std::remove(cpreds.begin(), cpreds.end(), cond_id),
                                 cpreds.end());

                    g.Node(clause_exit).preds.push_back(copy_id);

                    LOG(INFO) << "ResolveClauseSplit: duplicated node " << clause_id
                              << " as " << copy_id << " for cond " << cond_id << "\n";
                    return true;
                }
            }
            return false;
        }

        /// Join absorption: when both branches of a conditional exit to
        /// different blocks, but one branch's successor is a small shared
        /// block that could be absorbed to create a common exit point.
        ///
        /// Pattern: cond→A→J, cond→B→K where A,B have sizeIn=1 sizeOut=1,
        ///          J has sizeIn>1, is small, non-conditional, no label
        ///          and J.succs matches K (or J.succs[0] == K)
        /// After:   absorb J's stmts into A, redirect A→K → FoldIfElse fires
        static bool ResolveJoinAbsorb(CGraph &g) {
            for (auto &bl : g.nodes) {
                if (bl.collapsed || bl.SizeOut() != 2 || !bl.is_conditional) {
                    continue;
                }
                if (bl.IsGotoOut(0) || bl.IsGotoOut(1)) {
                    continue;
                }

                size_t a_id = bl.succs[1];  // true branch
                size_t b_id = bl.succs[0];  // false branch

                // Try both orientations: absorb into A or absorb into B
                for (int orient = 0; orient < 2; ++orient) {
                    if (orient == 1) std::swap(a_id, b_id);

                    auto &a = g.Node(a_id);
                    auto &b = g.Node(b_id);
                    if (a.collapsed || b.collapsed) continue;
                    if (a.SizeIn() != 1 || a.SizeOut() != 1) {
                        continue;
                    }
                    if (b.SizeIn() != 1 || b.SizeOut() != 1) {
                        continue;
                    }
                    if (a.IsGotoOut(0) || b.IsGotoOut(0)) {
                        continue;
                    }

                    size_t j_id = a.succs[0];  // A's successor (candidate for absorption)
                    size_t k_id = b.succs[0];  // B's successor

                    if (j_id == k_id) continue;  // already same exit → ifElse should fire
                    auto &j = g.Node(j_id);
                    if (j.collapsed) continue;
                    if (j.SizeIn() <= 1) {
                        continue; // not shared
                    }
                    if (j.SizeOut() > 1) {
                        continue; // must have <=1 exit
                    }
                    if (j.is_conditional) continue;
                    if (!j.label.empty()) continue;
                    if (j.stmts.size() > 16) continue;

                    // J must exit to K (so after absorption, A→K matches B→K)
                    if (j.SizeOut() == 1 && j.succs[0] != k_id) {
                        continue;
                    }
                    // If J has no successors, B must also have no successors
                    if (j.SizeOut() == 0 && b.SizeOut() != 0) {
                        continue;
                    }

                    // Absorb: append J's stmts to A, redirect A past J
                    auto &an = g.Node(a_id);
                    for (auto *s : g.Node(j_id).stmts) {
                        an.stmts.push_back(s);
                    }

                    // Remove edge A→J
                    g.RemoveEdge(a_id, j_id);

                    if (j.SizeOut() == 1) {
                        // Add edge A→K
                        an.succs.push_back(k_id);
                        an.edge_flags.push_back(0);
                        g.Node(k_id).preds.push_back(a_id);
                    }

                    LOG(INFO) << "ResolveJoinAbsorb: absorbed node " << j_id
                              << " into " << a_id << " for cond " << bl.id << "\n";
                    return true;
                }
            }
            return false;
        }

        /// Try control-equivalence hoisting transforms to unblock the main
        /// collapse rules before falling back to goto selection.
        static bool ResolveControlEquivHoist(CGraph &g) {
            if (ResolveClauseSplit(g)) return true;
            if (ResolveJoinAbsorb(g)) return true;
            return false;
        }

        // ---------------------------------------------------------------
        // Main collapse loop
        // ---------------------------------------------------------------

        size_t FoldMainLoop(CGraph &g, SNodeFactory &factory, clang::ASTContext &ctx) {
            bool change;
            size_t isolated_count;

            do {
                do {
                    change = false;
                    isolated_count = 0;
                    for (auto &n : g.nodes) {
                        if (n.collapsed) continue;
                        if (n.SizeIn() == 0 && n.SizeOut() == 0) {
                            isolated_count++;
                            continue;
                        }

                        if (FoldGoto(g, n.id, factory, ctx)) { change = true; continue; }
                        if (FoldSequence(g, n.id, factory)) { change = true; continue; }
                        if (FoldIfThen(g, n.id, factory, ctx)) { change = true; continue; }
                        if (FoldIfElse(g, n.id, factory, ctx)) { change = true; continue; }
                        if (FoldWhileLoop(g, n.id, factory, ctx)) { change = true; continue; }
                        if (FoldDoWhileLoop(g, n.id, factory, ctx)) { change = true; continue; }
                        if (FoldInfiniteLoop(g, n.id, factory, ctx)) { change = true; continue; }
                        if (FoldSwitch(g, n.id, factory, ctx)) { change = true; continue; }
                    }
                } while (change);

                // Try IfNoExit as fallback (Ghidra applies this only when stuck)
                change = false;
                for (auto &n : g.nodes) {
                    if (n.collapsed) continue;
                    if (FoldIfForcedGoto(g, n.id, factory, ctx)) {
                        change = true;
                        break;
                    }
                    if (FoldCaseFallthrough(g, n.id)) {
                        change = true;
                        break;
                    }
                }
            } while (change);

            return isolated_count;
        }

        // ---------------------------------------------------------------
        // Post-collapse transforms: RefineBreakContinue, RefineWhileToFor, RefineDeadLabels
        // ---------------------------------------------------------------

        // RefineBreakContinue: convert loop-exit gotos to SBreak and loop-header gotos
        // to SContinue. Walks the SNode tree, tracking the current loop context
        // via exit/header label names.
        void RefineBreakContinue(SNode *node, std::string_view loop_exit_label,
                        std::string_view loop_header_label, SNodeFactory &factory) {
            if (!node) return;

            if (auto *seq = node->dyn_cast<SSeq>()) {
                // Recurse into children. When entering a loop child, we need to
                // determine the new loop exit/header labels from sibling context.
                for (size_t i = 0; i < seq->Size(); ++i) {
                    auto *child = (*seq)[i];

                    // For SWhile/SDoWhile, compute new loop context labels
                    if (auto *w = child->dyn_cast<SWhile>()) {
                        // New loop_exit_label: label of the next sibling (if SLabel)
                        std::string_view new_exit;
                        if (i + 1 < seq->Size()) {
                            if (auto *lbl = (*seq)[i + 1]->dyn_cast<SLabel>()) {
                                new_exit = lbl->Name();
                            }
                        }
                        if (new_exit.empty()) new_exit = loop_exit_label;

                        // New loop_header_label: if prev sibling is SLabel wrapping
                        // this while, use that label. Also check if the while body's
                        // first child is an SLabel (header-stmts-inside-loop pattern).
                        std::string_view new_header;
                        if (i > 0) {
                            if (auto *lbl = (*seq)[i - 1]->dyn_cast<SLabel>()) {
                                new_header = lbl->Name();
                            }
                        }
                        if (new_header.empty()) {
                            if (auto *body_seq = w->Body() ? w->Body()->dyn_cast<SSeq>() : nullptr) {
                                if (body_seq->Size() > 0) {
                                    if (auto *lbl = (*body_seq)[0]->dyn_cast<SLabel>()) {
                                        new_header = lbl->Name();
                                    }
                                }
                            }
                        }

                        RefineBreakContinue(w->Body(), new_exit, new_header, factory);
                        continue;
                    }

                    if (auto *dw = child->dyn_cast<SDoWhile>()) {
                        std::string_view new_exit;
                        if (i + 1 < seq->Size()) {
                            if (auto *lbl = (*seq)[i + 1]->dyn_cast<SLabel>()) {
                                new_exit = lbl->Name();
                            }
                        }
                        if (new_exit.empty()) new_exit = loop_exit_label;

                        std::string_view new_header;
                        if (i > 0) {
                            if (auto *lbl = (*seq)[i - 1]->dyn_cast<SLabel>()) {
                                new_header = lbl->Name();
                            }
                        }

                        RefineBreakContinue(dw->Body(), new_exit, new_header, factory);
                        continue;
                    }

                    // For non-loop children, recurse with current loop context
                    RefineBreakContinue(child, loop_exit_label, loop_header_label, factory);
                }

                // After recursing, check if last child is SGoto targeting loop exit/header
                if (seq->Size() > 0) {
                    auto *last = (*seq)[seq->Size() - 1];
                    if (auto *g = last->dyn_cast<SGoto>()) {
                        if (!loop_exit_label.empty() && g->Target() == loop_exit_label) {
                            seq->ReplaceChild(seq->Size() - 1, factory.Make<SBreak>());
                        } else if (!loop_header_label.empty() && g->Target() == loop_header_label) {
                            seq->ReplaceChild(seq->Size() - 1, factory.Make<SContinue>());
                        }
                    }
                }
            }
            else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                RefineBreakContinue(ite->ThenBranch(), loop_exit_label, loop_header_label, factory);
                RefineBreakContinue(ite->ElseBranch(), loop_exit_label, loop_header_label, factory);
            }
            else if (auto *sw = node->dyn_cast<SSwitch>()) {
                // Switch does not change loop context
                for (auto &c : sw->Cases()) {
                    RefineBreakContinue(c.body, loop_exit_label, loop_header_label, factory);
                }
                RefineBreakContinue(sw->DefaultBody(), loop_exit_label, loop_header_label, factory);
            }
            else if (auto *lbl = node->dyn_cast<SLabel>()) {
                RefineBreakContinue(lbl->Body(), loop_exit_label, loop_header_label, factory);
            }
            else if (auto *f = node->dyn_cast<SFor>()) {
                // SFor has its own loop context
                RefineBreakContinue(f->Body(), loop_exit_label, loop_header_label, factory);
            }
            // SBlock, SGoto, SBreak, SContinue, SReturn: leaf nodes, nothing to do
        }

        // --- RefineWhileToFor helpers ---

        bool IsAssignOrDecl(clang::Stmt *s) {
            if (!s) return false;
            if (auto *bo = llvm::dyn_cast<clang::BinaryOperator>(s)) {
                return bo->getOpcode() == clang::BO_Assign;
            }
            return llvm::isa<clang::DeclStmt>(s);
        }

        bool IsIncrement(clang::Stmt *s) {
            if (!s) return false;
            if (auto *uo = llvm::dyn_cast<clang::UnaryOperator>(s)) {
                auto op = uo->getOpcode();
                return op == clang::UO_PreInc || op == clang::UO_PostInc
                    || op == clang::UO_PreDec || op == clang::UO_PostDec;
            }
            if (auto *bo = llvm::dyn_cast<clang::BinaryOperator>(s)) {
                auto op = bo->getOpcode();
                return op == clang::BO_Assign || op == clang::BO_AddAssign
                    || op == clang::BO_SubAssign;
            }
            return false;
        }

        // Extract the DeclRefExpr from an assignment or decl statement
        clang::DeclRefExpr *getAssignTarget(clang::Stmt *s) {
            if (!s) return nullptr;
            if (auto *bo = llvm::dyn_cast<clang::BinaryOperator>(s)) {
                return llvm::dyn_cast<clang::DeclRefExpr>(bo->getLHS()->IgnoreParenCasts());
            }
            if (auto *uo = llvm::dyn_cast<clang::UnaryOperator>(s)) {
                return llvm::dyn_cast<clang::DeclRefExpr>(uo->getSubExpr()->IgnoreParenCasts());
            }
            if (auto *ds = llvm::dyn_cast<clang::DeclStmt>(s)) {
                if (ds->isSingleDecl()) {
                    if (auto *vd = llvm::dyn_cast<clang::VarDecl>(ds->getSingleDecl())) {
                        (void)vd;
                        // DeclStmt doesn't directly produce a DeclRefExpr;
                        // we can't easily manufacture one without an ASTContext.
                        // Return nullptr — SameVariable will handle this via VarDecl matching.
                        return nullptr;
                    }
                }
            }
            return nullptr;
        }

        // Extract the VarDecl referenced by a statement (works for assign, unary, decl)
        clang::VarDecl *getReferencedVar(clang::Stmt *s) {
            if (auto *dre = getAssignTarget(s)) {
                return llvm::dyn_cast<clang::VarDecl>(dre->getDecl());
            }
            if (auto *ds = llvm::dyn_cast<clang::DeclStmt>(s)) {
                if (ds->isSingleDecl()) {
                    return llvm::dyn_cast<clang::VarDecl>(ds->getSingleDecl());
                }
            }
            return nullptr;
        }

        // Check if an expression tree contains a reference to the given VarDecl
        bool ContainsVarRef(clang::Stmt *s, clang::VarDecl *vd) {
            if (!s || !vd) return false;
            if (auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(s)) {
                return dre->getDecl() == vd;
            }
            for (auto *child : s->children()) {
                if (ContainsVarRef(child, vd)) return true;
            }
            return false;
        }

        // Check that init, cond, and inc all reference the same variable
        bool SameVariable(clang::Stmt *init, clang::Expr *cond, clang::Expr *inc) {
            auto *init_var = getReferencedVar(init);
            if (!init_var) return false;
            auto *inc_var = getReferencedVar(inc);
            if (!inc_var) return false;
            if (init_var != inc_var) return false;
            return ContainsVarRef(cond, init_var);
        }

        // RefineWhileToFor: convert init/while(cond)/inc patterns to SFor
        void RefineWhileToFor(SNode *node, SNodeFactory &factory, clang::ASTContext &ctx) {
            if (!node) return;

            if (auto *seq = node->dyn_cast<SSeq>()) {
                // Scan for pattern: SBlock(init), SWhile(cond, SSeq(..., SBlock(inc)))
                for (size_t i = 0; i < seq->Size(); ++i) {
                    auto *w = (*seq)[i]->dyn_cast<SWhile>();
                    if (!w) continue;

                    // Check for init before the while
                    clang::Stmt *init_stmt = nullptr;
                    bool has_init = false;
                    if (i > 0) {
                        auto *prev = (*seq)[i - 1]->dyn_cast<SBlock>();
                        if (prev && prev->Size() == 1 && IsAssignOrDecl(prev->Stmts()[0])) {
                            init_stmt = prev->Stmts()[0];
                            has_init = true;
                        }
                    }

                    // Check for inc at end of while body
                    clang::Expr *inc_expr = nullptr;
                    auto *body_seq = w->Body() ? w->Body()->dyn_cast<SSeq>() : nullptr;
                    if (body_seq && body_seq->Size() > 0) {
                        auto *last = (*body_seq)[body_seq->Size() - 1]->dyn_cast<SBlock>();
                        if (last && last->Size() == 1 && IsIncrement(last->Stmts()[0])) {
                            inc_expr = llvm::dyn_cast<clang::Expr>(last->Stmts()[0]);
                        }
                    }

                    if (has_init && inc_expr && SameVariable(init_stmt, w->Cond(), inc_expr)) {
                        // Remove inc from body
                        body_seq->RemoveChild(body_seq->Size() - 1);
                        // Build SFor
                        auto *for_node = factory.Make<SFor>(init_stmt, w->Cond(), inc_expr, w->Body());
                        // Replace [i-1, i+1) with the for node
                        seq->ReplaceRange(i - 1, i + 1, for_node);
                        --i; // adjust for removed element
                    }
                }

                // Recurse into remaining children
                for (size_t i = 0; i < seq->Size(); ++i) {
                    RefineWhileToFor((*seq)[i], factory, ctx);
                }
            }
            else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                RefineWhileToFor(ite->ThenBranch(), factory, ctx);
                RefineWhileToFor(ite->ElseBranch(), factory, ctx);
            }
            else if (auto *w = node->dyn_cast<SWhile>()) {
                RefineWhileToFor(w->Body(), factory, ctx);
            }
            else if (auto *dw = node->dyn_cast<SDoWhile>()) {
                RefineWhileToFor(dw->Body(), factory, ctx);
            }
            else if (auto *f = node->dyn_cast<SFor>()) {
                RefineWhileToFor(f->Body(), factory, ctx);
            }
            else if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases()) {
                    RefineWhileToFor(c.body, factory, ctx);
                }
                RefineWhileToFor(sw->DefaultBody(), factory, ctx);
            }
            else if (auto *lbl = node->dyn_cast<SLabel>()) {
                RefineWhileToFor(lbl->Body(), factory, ctx);
            }
        }

        // --- RefineDeadLabels: dead label removal ---

        // Collect all SGoto target names in the tree
        void CollectGotoTargets(SNode *node, std::unordered_set<std::string> &targets) {
            if (!node) return;

            if (auto *g = node->dyn_cast<SGoto>()) {
                targets.emplace(g->Target());
            }
            else if (auto *seq = node->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < seq->Size(); ++i) {
                    CollectGotoTargets((*seq)[i], targets);
                }
            }
            else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                CollectGotoTargets(ite->ThenBranch(), targets);
                CollectGotoTargets(ite->ElseBranch(), targets);
            }
            else if (auto *w = node->dyn_cast<SWhile>()) {
                CollectGotoTargets(w->Body(), targets);
            }
            else if (auto *dw = node->dyn_cast<SDoWhile>()) {
                CollectGotoTargets(dw->Body(), targets);
            }
            else if (auto *f = node->dyn_cast<SFor>()) {
                CollectGotoTargets(f->Body(), targets);
            }
            else if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases()) {
                    CollectGotoTargets(c.body, targets);
                }
                CollectGotoTargets(sw->DefaultBody(), targets);
            }
            else if (auto *lbl = node->dyn_cast<SLabel>()) {
                CollectGotoTargets(lbl->Body(), targets);
            }
            else if (auto *blk = node->dyn_cast<SBlock>()) {
                // Scan raw Clang AST stmts for GotoStmt targets
                // (e.g. from OperationStmt switch case inlining)
                std::function<void(clang::Stmt *)> scanStmt = [&](clang::Stmt *s) {
                    if (!s) return;
                    if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(s)) {
                        targets.emplace(gs->getLabel()->getName().str());
                    }
                    for (auto *child : s->children()) scanStmt(child);
                };
                for (auto *s : blk->Stmts()) scanStmt(s);
            }
        }

        // Remove SLabel nodes whose name is not in the goto target set.
        // If the SLabel has a body, replace the label with its body.
        // If no body, remove entirely.
        void RemoveDeadLabels(SNode *node,
                              const std::unordered_set<std::string> &targets) {
            if (!node) return;

            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < seq->Size(); ) {
                    auto *child = (*seq)[i];
                    if (auto *lbl = child->dyn_cast<SLabel>()) {
                        if (targets.find(std::string(lbl->Name())) == targets.end()) {
                            // Dead label -- replace with body or remove
                            if (lbl->Body()) {
                                seq->ReplaceChild(i, lbl->Body());
                                // Don't increment — re-check the replacement
                            } else {
                                seq->RemoveChild(i);
                            }
                            continue;
                        }
                    }
                    // Recurse into child
                    RemoveDeadLabels(child, targets);
                    ++i;
                }
            }
            else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                RemoveDeadLabels(ite->ThenBranch(), targets);
                RemoveDeadLabels(ite->ElseBranch(), targets);
            }
            else if (auto *w = node->dyn_cast<SWhile>()) {
                RemoveDeadLabels(w->Body(), targets);
            }
            else if (auto *dw = node->dyn_cast<SDoWhile>()) {
                RemoveDeadLabels(dw->Body(), targets);
            }
            else if (auto *f = node->dyn_cast<SFor>()) {
                RemoveDeadLabels(f->Body(), targets);
            }
            else if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases()) {
                    RemoveDeadLabels(c.body, targets);
                }
                RemoveDeadLabels(sw->DefaultBody(), targets);
            }
            else if (auto *lbl = node->dyn_cast<SLabel>()) {
                RemoveDeadLabels(lbl->Body(), targets);
            }
        }

        // RefineDeadLabels: remove dead labels left after RefineBreakContinue converts
        // gotos to break/continue. For v1, this is dead label removal only
        // (full bump-up optimization deferred).
        void RefineDeadLabels(SNode *root) {
            std::unordered_set<std::string> targets;
            CollectGotoTargets(root, targets);
            RemoveDeadLabels(root, targets);
        }

    } // anonymous namespace

    // ---------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------

    SNode *CfgFoldStructure(const Cfg &cfg, SNodeFactory &factory,
                             clang::ASTContext &ctx) {
        if (cfg.blocks.empty()) {
            return factory.Make<SSeq>();
        }

        // 1. Build the collapse graph
        detail::CGraph g = detail::buildCGraph(cfg);

        // 2. Mark back-edges
        detail::MarkBackEdges(g);

        // 2b. Discover loops, compute bodies/nesting/exits, order innermost-first
        std::list<detail::LoopBody> loopbody;
        detail::OrderLoopBodies(g, loopbody);
        LOG(INFO) << "CfgFoldStructure: found " << loopbody.size() << " loop(s)\n";

        // 2c. Pre-pass: for non-conditional 2-successor blocks where one
        // edge is a back-edge and the other is a forward exit, remove the
        // exit edge (the goto is inside the stmts) and clear kBack so the
        // block can be chained into the loop body.
        {
            bool changed = false;
            for (auto &bl : g.nodes) {
                if (bl.collapsed || bl.SizeOut() != 2 || bl.is_conditional) {
                    continue;
                }
                int back_idx = -1;
                for (size_t i = 0; i < 2; ++i) {
                    if (bl.IsBackEdge(i)) {
                        back_idx = static_cast< int >(i);
                    }
                }
                if (back_idx < 0) continue;

                size_t exit_idx = 1 - static_cast<size_t>(back_idx);
                size_t exit_id = bl.succs[exit_idx];
                g.RemoveEdge(bl.id, exit_id);
                for (size_t i = 0; i < bl.edge_flags.size(); ++i)
                    bl.edge_flags[i] &= ~CNode::kBack;
                changed = true;
            }
            // Re-discover loops with updated graph so TraceDAG is consistent.
            if (changed) {
                loopbody.clear();
                detail::MarkBackEdges(g);
                detail::OrderLoopBodies(g, loopbody);
            }
        }

        // 2d. Collapse AND/OR conditions before main collapse loop
        ResolveAllConditionChains(g, factory, ctx);
        LOG(INFO) << "CfgFoldStructure: condition collapsing complete\n";

        // 2e. Absorb switch guard chains — mark guard→fallback edges as goto
        // so FoldGoto + FoldSequence chain guards into the switch block,
        // reducing the fallback block's sizeIn for FoldSwitch.
        ResolveSwitchGuards(g);

        // 3. Main collapse loop
        size_t isolated = FoldMainLoop(g, factory, ctx);

        // 3b. Try control-equivalence hoisting before falling back to gotos.
        // This duplicates or absorbs small shared blocks to unblock
        // FoldIfThen / FoldIfElse.
        while (isolated < g.ActiveCount()) {
            if (!ResolveControlEquivHoist(g)) break;
            isolated = FoldMainLoop(g, factory, ctx);
        }

        // 4. When stuck, select gotos via TraceDAG and retry
        size_t max_iterations = g.nodes.size() * 4;  // safety bound
        size_t iter = 0;
        while (isolated < g.ActiveCount() && iter < max_iterations) {
            if (!ResolveGotoSelection(g, loopbody)) {
                LOG(WARNING) << "CfgFoldStructure: could not select goto, "
                             << g.ActiveCount() - isolated << " blocks remaining\n";
                break;
            }
            isolated = FoldMainLoop(g, factory, ctx);
            ++iter;
        }

        // 5. Collect the final structured tree
        // Find the root (entry node or its collapsed representative)
        SNode *root = nullptr;
        for (auto &n : g.nodes) {
            if (n.collapsed) continue;
            // LeafFromNode handles both structured and leaf nodes,
            // and wraps with SLabel when the CNode carries a label.
            auto *block = LeafFromNode(n, factory);
            if (!root) {
                root = block;
            } else {
                auto *seq = root->dyn_cast<SSeq>();
                if (!seq) {
                    seq = factory.Make<SSeq>();
                    seq->AddChild(root);
                    root = seq;
                }
                seq->AddChild(block);
            }
        }

        if (!root) root = factory.Make<SSeq>();

        // 6. Post-collapse transforms (order matters per research)
        RefineBreakContinue(root, "", "", factory);    // 1st: gotos -> break/continue
        RefineWhileToFor(root, factory, ctx);       // 2nd: while -> for patterns
        RefineDeadLabels(root);                // 3rd: clean up dead labels

        return root;
    }

} // namespace patchestry::ast
