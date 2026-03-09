/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/CollapseStructure.hpp>
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

        size_t CGraph::collapseNodes(const std::vector<size_t> &ids, SNode *snode) {
            // Pick the first id as the representative
            size_t rep = ids[0];
            nodes[rep].structured = snode;

            // Collect external in/out edges
            std::unordered_set<size_t> idset(ids.begin(), ids.end());
            std::vector<size_t> ext_preds, ext_succs;
            std::vector<uint32_t> ext_succ_flags;

            for (size_t nid : ids) {
                for (size_t p : nodes[nid].preds) {
                    if (idset.count(p) == 0) ext_preds.push_back(p);
                }
            }
            // Collect exit edges from ALL collapsed nodes (not just last)
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

            // Mark all as collapsed except rep (Gap 7: track collapsed_into)
            for (size_t nid : ids) {
                if (nid != rep) {
                    nodes[nid].collapsed = true;
                    nodes[nid].collapsed_into = rep;
                }
            }

            // Replace edges: remove old, add new through rep
            for (size_t nid : ids) {
                // Remove all edges involving collapsed nodes
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
                                ss[i] = rep; // redirect to rep
                            }
                        }
                    }
                }
            }

            // Set rep's edges to the external edges
            nodes[rep].succs = ext_succs;
            nodes[rep].edge_flags = ext_succ_flags;

            // Deduplicate preds
            std::sort(ext_preds.begin(), ext_preds.end());
            ext_preds.erase(std::unique(ext_preds.begin(), ext_preds.end()), ext_preds.end());
            nodes[rep].preds = ext_preds;

            // Add rep to succs' pred lists
            for (size_t s : ext_succs) {
                auto &p = nodes[s].preds;
                if (std::find(p.begin(), p.end(), rep) == p.end()) {
                    p.push_back(rep);
                }
            }

            nodes[rep].is_conditional = !ext_succs.empty() && ext_succs.size() == 2;
            nodes[rep].stmts.clear();
            nodes[rep].label.clear();  // label already embedded via leafFromNode in rule
            nodes[rep].branch_cond = nullptr;

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
        void markBackEdges(CGraph &g) {
            enum Color { WHITE, GRAY, BLACK };
            std::vector<Color> color(g.nodes.size(), WHITE);

            std::function<void(size_t)> dfs = [&](size_t u) {
                color[u] = GRAY;
                auto &nd = g.node(u);
                for (size_t i = 0; i < nd.succs.size(); ++i) {
                    size_t v = nd.succs[i];
                    if (color[v] == GRAY) {
                        nd.edge_flags[i] |= CNode::F_BACK;
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

        void clearMarks(CGraph &g, const std::vector<size_t> &body) {
            for (size_t id : body) {
                g.node(id).mark = false;
            }
        }

        void LoopBody::findBase(CGraph &g, std::vector<size_t> &body) const {
            body.clear();

            // Mark head, add to body.
            g.node(head).mark = true;
            body.push_back(head);

            // Mark each tail (skip if already marked, e.g. head == tail).
            for (size_t t : tails) {
                if (!g.node(t).mark) {
                    g.node(t).mark = true;
                    body.push_back(t);
                }
            }

            // BFS backward from tails: iterate body starting at index 1
            // (index 0 is head -- we don't go backward from head).
            for (size_t idx = 1; idx < body.size(); ++idx) {
                auto &nd = g.node(body[idx]);
                for (size_t p : nd.preds) {
                    if (g.node(p).mark) continue;
                    if (g.node(p).collapsed) continue;

                    // Check if the incoming edge from p to body[idx] is a goto edge.
                    // Find the succ index in p that points to body[idx].
                    bool is_goto = false;
                    auto &pn = g.node(p);
                    for (size_t si = 0; si < pn.succs.size(); ++si) {
                        if (pn.succs[si] == body[idx]) {
                            if (pn.isGotoOut(si)) {
                                is_goto = true;
                            }
                            break;
                        }
                    }
                    if (is_goto) continue;

                    g.node(p).mark = true;
                    body.push_back(p);
                }
            }
        }

        void LoopBody::labelContainments(
            const CGraph &g, const std::vector<size_t> & /*body*/,
            const std::vector<LoopBody *> &looporder
        ) {
            // For each other loop in looporder: if that loop's head is marked
            // (i.e., inside this loop's body), then that loop is nested in this.
            // Update its immed_container if this is a tighter (smaller) container.
            for (LoopBody *lb : looporder) {
                if (lb == this) continue;
                if (!g.node(lb->head).mark) continue;

                // lb is nested inside this loop.
                if (lb->immed_container == nullptr
                    || lb->immed_container->unique_count > unique_count) {
                    // `this` is a tighter container (fewer unique head+tail nodes).
                    // But wait -- smaller unique_count means tighter is wrong.
                    // A tighter container has FEWER body nodes. Use unique_count as proxy:
                    // Actually, we want the INNERMOST container. If lb already has a
                    // container C, and C is smaller than `this`, keep C. Otherwise use this.
                    // Smaller unique_count ≈ smaller loop ≈ tighter. But `this` is the
                    // OUTER loop (lb's head is in our body). So we want the smallest
                    // outer loop. Replace only if this is smaller than current container.
                }
                // Correction: `this` contains `lb`. We want lb->immed_container to be
                // the SMALLEST loop that contains lb. So replace if this loop is smaller
                // than lb's current immed_container, OR if lb has no container yet.
                if (lb->immed_container == nullptr) {
                    lb->immed_container = this;
                } else {
                    // Replace if `this` is a smaller container (fewer unique nodes).
                    // Wait -- unique_count is head+tails before reachability. Not great
                    // as a size proxy. But Ghidra uses the approach: the last loop to
                    // set the container is the tightest, because loops are processed
                    // outermost-first (by body size). Since we don't have body size here,
                    // use unique_count as a rough proxy. Actually the body vector IS
                    // available via marks. Let's just not over-think: use unique_count.
                    // Tighter = smaller body ≈ larger unique_count (fewer extra nodes).
                    // Actually no -- just check if the current container's head is in our
                    // body (meaning we're OUTSIDE it).
                    // Simplest correct approach: if the current immed_container's head is
                    // also marked (in our body), then we are OUTER -- don't replace.
                    // If not marked, current container is not inside us, which is
                    // inconsistent, so replace.
                    if (g.node(lb->immed_container->head).mark) {
                        // Current container is also inside this loop -- it's tighter. Keep it.
                    } else {
                        // Current container is outside this loop -- replace with us.
                        lb->immed_container = this;
                    }
                }
            }
        }

        void LoopBody::mergeIdenticalHeads(
            std::vector<LoopBody *> &looporder,
            std::list<LoopBody> &storage
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
                        dst->addTail(t);
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

        void labelLoops(
            CGraph &g,
            std::list<LoopBody> &loopbody,
            std::vector<LoopBody *> &looporder
        ) {
            // Scan all edges for back-edges. Create one LoopBody per back-edge.
            for (auto &n : g.nodes) {
                if (n.collapsed) continue;
                for (size_t i = 0; i < n.succs.size(); ++i) {
                    if (n.isBackEdge(i)) {
                        size_t hd = n.succs[i];
                        loopbody.emplace_back(hd);
                        loopbody.back().addTail(n.id);
                        looporder.push_back(&loopbody.back());
                    }
                }
            }

            // Merge loops sharing the same head.
            LoopBody::mergeIdenticalHeads(looporder, loopbody);

            // For each loop: findBase, labelContainments, clearMarks.
            for (LoopBody *lb : looporder) {
                std::vector<size_t> body;
                lb->findBase(g, body);
                lb->unique_count = static_cast<int>(body.size());
                lb->labelContainments(g, body, looporder);
                clearMarks(g, body);
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

        void LoopBody::findExit(const CGraph &g, const std::vector<size_t> &body) {
            // Select a single exit block for the loop.
            // Priority: tail exits first, then head exits, then body exits.
            // A candidate exit is an unmarked successor reached via a non-goto edge.

            std::vector<size_t> candidates;

            // Scan tails for exits.
            for (size_t t : tails) {
                const auto &tn = g.node(t);
                for (size_t i = 0; i < tn.succs.size(); ++i) {
                    size_t s = tn.succs[i];
                    if (!g.node(s).mark && !tn.isGotoOut(i) && !tn.isBackEdge(i)) {
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
                const auto &hd = g.node(body[0]);
                for (size_t i = 0; i < hd.succs.size(); ++i) {
                    size_t s = hd.succs[i];
                    if (!g.node(s).mark && !hd.isGotoOut(i) && !hd.isBackEdge(i)) {
                        candidates.push_back(s);
                    }
                }
                // Scan body nodes beyond the unique set (index >= unique_count).
                for (size_t idx = static_cast<size_t>(unique_count); idx < body.size(); ++idx) {
                    const auto &bn = g.node(body[idx]);
                    for (size_t i = 0; i < bn.succs.size(); ++i) {
                        size_t s = bn.succs[i];
                        if (!g.node(s).mark && !bn.isGotoOut(i) && !bn.isBackEdge(i)) {
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
                exit_block = NONE;
            }
        }

        void LoopBody::orderTails(const CGraph &g) {
            // Swap the tail that directly reaches exit_block to tails[0].
            if (tails.size() <= 1 || exit_block == NONE) return;

            for (size_t ti = 0; ti < tails.size(); ++ti) {
                const auto &tn = g.node(tails[ti]);
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

        void LoopBody::extend(CGraph &g, std::vector<size_t> &body) const {
            // Add dominated-only blocks: blocks whose ALL predecessors are in the body.
            // Uses visit_count to track how many predecessors are body members.
            // Fixpoint iteration until no new nodes are added.

            std::vector<size_t> touched;  // track nodes with non-zero visit_count

            bool added = true;
            while (added) {
                added = false;

                // For each body node, increment visit_count of non-body, non-collapsed successors.
                for (size_t bid : body) {
                    const auto &bn = g.node(bid);
                    for (size_t i = 0; i < bn.succs.size(); ++i) {
                        size_t s = bn.succs[i];
                        if (g.node(s).mark) continue;       // already in body
                        if (g.node(s).collapsed) continue;   // absorbed
                        if (bn.isGotoOut(i)) continue;       // don't extend through gotos
                        if (s == exit_block) continue;        // don't extend into exit
                        if (g.node(s).visit_count == 0) {
                            touched.push_back(s);
                        }
                        g.node(s).visit_count++;
                    }
                }

                // Check which visited nodes have ALL predecessors in body.
                for (size_t s : touched) {
                    auto &sn = g.node(s);
                    if (sn.mark) continue;  // already added
                    if (sn.visit_count == static_cast<int>(sn.sizeIn())) {
                        // All preds are in body -- include this node.
                        sn.mark = true;
                        body.push_back(s);
                        added = true;
                    }
                }

                // Reset visit_count for next iteration.
                for (size_t s : touched) {
                    g.node(s).visit_count = 0;
                }
                touched.clear();
            }
        }

        void LoopBody::labelExitEdges(CGraph &g, const std::vector<size_t> &body) const {
            // Mark F_LOOP_EXIT on all edges leaving the loop body.
            for (size_t bid : body) {
                auto &bn = g.node(bid);
                for (size_t i = 0; i < bn.succs.size(); ++i) {
                    size_t s = bn.succs[i];
                    if (!g.node(s).mark && !bn.isGotoOut(i)) {
                        bn.edge_flags[i] |= CNode::F_LOOP_EXIT;
                    }
                }
            }
        }

        // -------------------------------------------------------------------
        // orderLoopBodies: orchestrate full loop detection pipeline
        // -------------------------------------------------------------------

        void orderLoopBodies(CGraph &g, std::list<LoopBody> &loopbody) {
            // 1. labelLoops: create LoopBody per back-edge, merge, compute containment/depth
            std::vector<LoopBody *> looporder;
            labelLoops(g, loopbody, looporder);
            if (loopbody.empty()) return;

            // 2. For each loop (innermost-first): findExit, orderTails, extend, labelExitEdges
            for (auto &lb : loopbody) {
                std::vector<size_t> body;
                lb.findBase(g, body);
                lb.findExit(g, body);
                lb.orderTails(g);
                lb.extend(g, body);
                lb.labelExitEdges(g, body);
                clearMarks(g, body);
            }
        }

        // -------------------------------------------------------------------
        // LoopBody: exit mark management and update
        // -------------------------------------------------------------------

        void LoopBody::setExitMarks(CGraph &g, const std::vector<size_t> &body) const {
            std::unordered_set<size_t> bodyset(body.begin(), body.end());
            for (size_t nid : body) {
                auto &n = g.node(nid);
                for (size_t i = 0; i < n.succs.size(); ++i) {
                    if (bodyset.count(n.succs[i]) == 0) {
                        n.setLoopExit(i);
                    }
                }
            }
        }

        void LoopBody::clearExitMarks(CGraph &g, const std::vector<size_t> &body) const {
            for (size_t nid : body) {
                auto &n = g.node(nid);
                for (size_t i = 0; i < n.succs.size(); ++i) {
                    n.clearLoopExit(i);
                }
            }
        }

        bool LoopBody::update(const CGraph &g) const {
            return !g.node(head).collapsed;
        }

        // -------------------------------------------------------------------
        // FloatingEdge
        // -------------------------------------------------------------------

        std::pair<size_t, size_t> FloatingEdge::getCurrentEdge(const CGraph &g) const {
            // Gap 3: walk up collapse hierarchy (like Ghidra's getParent walk)
            size_t top = top_id;
            while (top < g.nodes.size() && g.node(top).collapsed) {
                size_t next = g.node(top).collapsed_into;
                if (next == CNode::NONE || next == top) break;
                top = next;
            }
            size_t bot = bottom_id;
            while (bot < g.nodes.size() && g.node(bot).collapsed) {
                size_t next = g.node(bot).collapsed_into;
                if (next == CNode::NONE || next == bot) break;
                bot = next;
            }

            if (top >= g.nodes.size() || bot >= g.nodes.size())
                return {CNode::NONE, 0};
            if (g.node(top).collapsed || g.node(bot).collapsed)
                return {CNode::NONE, 0};
            if (top == bot) return {CNode::NONE, 0};

            const auto &succs = g.node(top).succs;
            for (size_t i = 0; i < succs.size(); ++i) {
                if (succs[i] == bot)
                    return {top, i};
            }
            return {CNode::NONE, 0};
        }

        // -------------------------------------------------------------------
        // TraceDAG implementation
        // -------------------------------------------------------------------

        TraceDAG::~TraceDAG() {
            for (auto *bp : branchlist) {
                for (auto *bt : bp->paths) {
                    delete bt;
                }
                delete bp;
            }
        }

        void TraceDAG::BranchPoint::markPath() {
            ismark = true;
            if (parent != nullptr && !parent->ismark)
                parent->markPath();
        }

        int TraceDAG::BranchPoint::distance(BranchPoint *op2) {
            // Clear marks
            for (auto *cur = this; cur != nullptr; cur = cur->parent)
                cur->ismark = false;
            for (auto *cur = op2; cur != nullptr; cur = cur->parent)
                cur->ismark = false;

            // Mark path from this to root
            markPath();

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

        void TraceDAG::insertActive(BlockTrace *trace) {
            trace->flags |= BlockTrace::f_active;
            activetrace.push_back(trace);
            trace->activeiter = std::prev(activetrace.end());
            ++activecount;
        }

        void TraceDAG::removeActive(BlockTrace *trace) {
            if (trace->isActive()) {
                activetrace.erase(trace->activeiter);
                trace->flags &= ~BlockTrace::f_active;
                --activecount;
            }
        }

        void TraceDAG::removeTrace(BlockTrace *trace) {
            removeActive(trace);
            // If this trace derived a BranchPoint, remove its traces too
            if (trace->derivedbp != nullptr) {
                for (auto *bt : trace->derivedbp->paths) {
                    if (bt != trace)
                        removeTrace(bt);
                }
            }
        }

        void TraceDAG::initialize() {
            for (size_t root_id : rootlist) {
                auto *bp = new BranchPoint();
                bp->top_id = root_id;
                bp->depth = 0;
                branchlist.push_back(bp);

                auto *bt = new BlockTrace();
                bt->top = bp;
                bt->pathout = 0;
                bt->bottom_id = CNode::NONE;
                bt->dest_id = root_id;
                bp->paths.push_back(bt);
                insertActive(bt);
            }
        }

        bool TraceDAG::checkOpen(const CGraph &g, BlockTrace *trace) {
            size_t dest = trace->dest_id;
            if (dest == CNode::NONE || dest >= g.nodes.size())
                return true;  // terminal

            const auto &n = g.node(dest);
            if (n.collapsed) {
                trace->flags |= BlockTrace::f_terminal;
                return true;
            }

            if (dest == finishblock_id) {
                trace->flags |= BlockTrace::f_terminal;
                return true;
            }

            // Count DAG-eligible out-edges
            size_t dag_out_count = 0;
            size_t single_succ = CNode::NONE;
            for (size_t i = 0; i < n.succs.size(); ++i) {
                if (n.isLoopDAGOut(i)) {
                    ++dag_out_count;
                    single_succ = n.succs[i];
                }
            }

            if (dag_out_count == 0) {
                trace->flags |= BlockTrace::f_terminal;
                return true;
            }

            if (dag_out_count == 1) {
                // Linear trace: advance
                trace->bottom_id = dest;
                trace->dest_id = single_succ;
                return true;
            }

            // Multiple out-edges: needs branching
            return false;
        }

        std::list<TraceDAG::BlockTrace *>::iterator
        TraceDAG::openBranch(CGraph &g, BlockTrace *parent) {
            size_t branch_id = parent->dest_id;
            const auto &n = g.node(branch_id);

            auto *bp = new BranchPoint();
            bp->parent = parent->top;
            bp->pathout = parent->pathout;
            bp->top_id = branch_id;
            bp->depth = parent->top->depth + 1;
            branchlist.push_back(bp);
            parent->derivedbp = bp;

            // Remove parent from active (it's now represented by children)
            removeActive(parent);

            int pathindex = 0;
            for (size_t i = 0; i < n.succs.size(); ++i) {
                if (!n.isLoopDAGOut(i)) continue;

                size_t succ_id = n.succs[i];
                auto &succ_node = g.node(succ_id);

                auto *bt = new BlockTrace();
                bt->top = bp;
                bt->pathout = pathindex++;
                bt->bottom_id = branch_id;
                bt->dest_id = succ_id;
                bp->paths.push_back(bt);

                // Check if already visited (edge lump)
                if (!succ_node.collapsed && succ_node.visit_count > 0) {
                    // Find the existing trace that visits this node
                    for (auto *existing : activetrace) {
                        if (existing->dest_id == succ_id || existing->bottom_id == succ_id) {
                            existing->edgelump += 1;
                            bt->flags |= BlockTrace::f_terminal;
                            break;
                        }
                    }
                    if (!bt->isTerminal()) {
                        // No existing active trace found, just mark visited
                        insertActive(bt);
                    }
                } else {
                    insertActive(bt);
                }

                if (!succ_node.collapsed) {
                    succ_node.visit_count += 1;
                }
            }

            return current_activeiter;
        }

        bool TraceDAG::checkRetirement(BlockTrace *trace, size_t &exitblock_id) {
            // Gap 2+6: Only first sibling triggers retirement (matches Ghidra)
            if (trace->pathout != 0) return false;
            BranchPoint *bp = trace->top;
            if (bp == nullptr) return false;

            if (bp->depth == 0) {
                // Root BranchPoint: all paths must be active AND terminal
                for (auto *bt : bp->paths) {
                    if (!bt->isActive()) return false;
                    if (!bt->isTerminal()) return false;
                }
                exitblock_id = CNode::NONE;
                return true;
            }

            // Non-root: all must be active; non-terminal paths must converge
            // to the SAME destination. Different exits = DON'T retire (forces
            // selectBadEdge to fire).
            size_t outblock = CNode::NONE;
            for (auto *bt : bp->paths) {
                if (!bt->isActive()) return false;
                if (bt->isTerminal()) continue;
                if (outblock == bt->dest_id) continue;
                if (outblock != CNode::NONE) return false;  // divergent exits
                outblock = bt->dest_id;
            }
            exitblock_id = outblock;
            return true;
        }

        std::list<TraceDAG::BlockTrace *>::iterator
        TraceDAG::retireBranch(BranchPoint *bp, size_t exitblock_id) {
            // Remove all traces from this BranchPoint
            std::list<BlockTrace *>::iterator next_iter = current_activeiter;
            for (auto *bt : bp->paths) {
                if (bt->isActive()) {
                    auto it = bt->activeiter;
                    if (it == next_iter) ++next_iter;
                    removeActive(bt);
                }
            }

            // If bp has a parent BranchPoint, update the parent trace
            if (bp->parent != nullptr) {
                // Find the parent trace that derived this BP
                for (auto *pt : bp->parent->paths) {
                    if (pt->derivedbp == bp) {
                        pt->derivedbp = nullptr;
                        if (exitblock_id != CNode::NONE) {
                            pt->dest_id = exitblock_id;
                            insertActive(pt);
                        } else {
                            pt->flags |= BlockTrace::f_terminal;
                        }
                        break;
                    }
                }
            }

            return next_iter;
        }

        bool TraceDAG::BadEdgeScore::compareFinal(const BadEdgeScore &op2) const {
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

        void TraceDAG::processExitConflict(
            std::list<BadEdgeScore>::iterator start,
            std::list<BadEdgeScore>::iterator end)
        {
            // Count traces in this group
            int count = 0;
            for (auto it = start; it != end; ++it) ++count;
            if (count <= 1) return;

            // For each pair, compute distance between their BranchPoints
            for (auto it = start; it != end; ++it) {
                it->siblingedge = count - 1;
                for (auto jt = start; jt != end; ++jt) {
                    if (it == jt) continue;
                    int d = it->trace->top->distance(jt->trace->top);
                    if (it->distance < 0 || d < it->distance) {
                        it->distance = d;
                    }
                }
            }
        }

        TraceDAG::BlockTrace *TraceDAG::selectBadEdge() {
            // Build score list from non-terminal active traces (Ghidra skips terminals)
            std::list<BadEdgeScore> scores;
            for (auto *bt : activetrace) {
                if (bt->isTerminal()) continue;
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
                processExitConflict(group_start, group_end);
                group_start = group_end;
            }

            // Find the worst edge
            BlockTrace *worst = nullptr;
            BadEdgeScore worst_score;
            for (auto &s : scores) {
                if (worst == nullptr || s.compareFinal(worst_score)) {
                    worst = s.trace;
                    worst_score = s;
                }
            }

            return worst;
        }

        void TraceDAG::pushBranches(CGraph &g) {
            // Gap 6: Rewrite to match Ghidra's missedcount-based loop
            clearVisitCount(g);

            for (size_t root_id : rootlist) {
                if (root_id < g.nodes.size() && !g.node(root_id).collapsed)
                    g.node(root_id).visit_count = 1;
            }

            while (activecount > 0) {
                int missedcount = 0;
                current_activeiter = activetrace.begin();

                while (current_activeiter != activetrace.end()) {
                    BlockTrace *bt = *current_activeiter;

                    if (missedcount >= activecount) {
                        // Stuck: select bad edge
                        BlockTrace *bad = selectBadEdge();
                        if (bad == nullptr) { clearVisitCount(g); return; }
                        if (bad->bottom_id != CNode::NONE && bad->dest_id != CNode::NONE)
                            likelygoto.emplace_back(bad->bottom_id, bad->dest_id);
                        removeTrace(bad);
                        missedcount = 0;
                        current_activeiter = activetrace.begin();
                        continue;
                    }

                    // Try retirement (only first sibling via pathout==0 check
                    // inside checkRetirement)
                    {
                        size_t exit_id = CNode::NONE;
                        if (checkRetirement(bt, exit_id)) {
                            current_activeiter = retireBranch(bt->top, exit_id);
                            missedcount = 0;
                            continue;
                        }
                    }

                    if (checkOpen(g, bt)) {
                        // Trace advanced or terminated
                        ++current_activeiter;
                        missedcount = 0;
                        continue;
                    }

                    // Needs branching
                    current_activeiter = openBranch(g, bt);
                    missedcount = 0;
                }
            }

            clearVisitCount(g);
        }

        void TraceDAG::clearVisitCount(CGraph &g) {
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

        static clang::Expr *negateCond(clang::Expr *cond, clang::ASTContext &ctx) {
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

        SNode *leafFromNode(CNode &n, SNodeFactory &factory) {
            SNode *result;
            if (n.structured) {
                result = n.structured;
            } else {
                auto *block = factory.make<SBlock>();
                for (auto *s : n.stmts) block->addStmt(s);
                result = block;
            }
            // Wrap with SLabel if this CNode carries a label from the original CFG.
            // Clear the label after wrapping to prevent double-wrapping if
            // leafFromNode is called again on the same node.
            if (!n.label.empty()) {
                result = factory.make<SLabel>(factory.intern(n.label), result);
                n.label.clear();
            }
            return result;
        }

        // ---------------------------------------------------------------
        // Pattern-matching collapse rules (adapted from Ghidra)
        // ---------------------------------------------------------------

        // Rule: Sequential blocks (A->B chain)
        bool ruleBlockCat(CGraph &g, size_t id, SNodeFactory &factory) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 1) return false;
            if (bl.isSwitchOut()) return false;
            if (!bl.isDecisionOut(0)) return false;

            // Start-of-chain guard: don't fire mid-chain (prevents nested SSeq)
            if (bl.sizeIn() == 1) {
                auto &pred = g.node(bl.preds[0]);
                if (!pred.collapsed && pred.sizeOut() == 1) {
                    return false;
                }
            }

            size_t next_id = bl.succs[0];
            if (next_id == id) return false;  // no self-loop
            auto &next = g.node(next_id);
            if (next.collapsed) return false;
            if (next.sizeIn() != 1) return false;

            // Build a sequence
            auto *seq = factory.make<SSeq>();
            seq->addChild(leafFromNode(bl, factory));

            // Extend the chain
            std::vector<size_t> chain = {id, next_id};
            size_t cur = next_id;
            while (g.node(cur).sizeOut() == 1 && g.node(cur).isDecisionOut(0)) {
                size_t nxt = g.node(cur).succs[0];
                if (nxt == id) break;
                auto &nxtNode = g.node(nxt);
                if (nxtNode.collapsed || nxtNode.sizeIn() != 1) break;
                if (nxtNode.isSwitchOut()) break;
                chain.push_back(nxt);
                cur = nxt;
            }

            for (size_t i = 1; i < chain.size(); ++i) {
                seq->addChild(leafFromNode(g.node(chain[i]), factory));
            }

            g.collapseNodes(chain, seq);
            return true;
        }

        // Rule: If without else (proper if)
        bool ruleBlockProperIf(CGraph &g, size_t id, SNodeFactory &factory,
                               clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 2) return false;
            if (bl.isSwitchOut()) return false;
            if (bl.isGotoOut(0) || bl.isGotoOut(1)) return false;

            for (size_t i = 0; i < 2; ++i) {
                size_t clause_id = bl.succs[i];
                if (clause_id == id) continue;
                auto &clause = g.node(clause_id);
                if (clause.collapsed) continue;
                if (clause.sizeIn() != 1 || clause.sizeOut() != 1) continue;
                if (!bl.isDecisionOut(i)) continue;
                if (clause.isGotoOut(0)) continue;

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
                    cond = negateCond(cond, ctx);
                }

                SNode *clause_body = leafFromNode(clause, factory);
                SNode *if_node = factory.make<SIfThenElse>(cond, clause_body, nullptr);
                if (!bl.label.empty()) {
                    if_node = factory.make<SLabel>(factory.intern(bl.label), if_node);
                    bl.label.clear();
                }

                g.collapseNodes({id, clause_id}, if_node);
                return true;
            }
            return false;
        }

        // Rule: If-else
        bool ruleBlockIfElse(CGraph &g, size_t id, SNodeFactory &factory,
                             clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 2) return false;
            if (bl.isSwitchOut()) return false;
            if (!bl.isDecisionOut(0) || !bl.isDecisionOut(1)) return false;

            size_t tc_id = bl.succs[1];  // true clause (Ghidra: out[1])
            size_t fc_id = bl.succs[0];  // false clause (Ghidra: out[0])
            auto &tc = g.node(tc_id);
            auto &fc = g.node(fc_id);

            if (tc.collapsed || fc.collapsed) return false;
            if (tc.sizeIn() != 1 || fc.sizeIn() != 1) return false;
            if (tc.sizeOut() != 1 || fc.sizeOut() != 1) return false;
            if (tc.succs[0] != fc.succs[0]) return false;  // must exit to same block
            if (tc.succs[0] == id) return false;  // no loops
            if (tc.isGotoOut(0) || fc.isGotoOut(0)) return false;

            clang::Expr *cond = bl.branch_cond;
            if (!cond) {
                cond = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
            }

            auto *then_body = leafFromNode(tc, factory);
            auto *else_body = leafFromNode(fc, factory);
            SNode *if_node = factory.make<SIfThenElse>(cond, then_body, else_body);
            if (!bl.label.empty()) {
                if_node = factory.make<SLabel>(factory.intern(bl.label), if_node);
                bl.label.clear();
            }

            g.collapseNodes({id, tc_id, fc_id}, if_node);
            return true;
        }

        // Rule: While-do loop
        bool ruleBlockWhileDo(CGraph &g, size_t id, SNodeFactory &factory,
                              clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 2) return false;
            if (bl.isSwitchOut()) return false;
            if (bl.isGotoOut(0) || bl.isGotoOut(1)) return false;

            for (size_t i = 0; i < 2; ++i) {
                size_t clause_id = bl.succs[i];
                if (clause_id == id) continue;
                auto &clause = g.node(clause_id);
                if (clause.collapsed) continue;
                if (clause.sizeIn() != 1 || clause.sizeOut() != 1) continue;
                if (clause.succs[0] != id) continue;  // must loop back

                clang::Expr *cond = bl.branch_cond;
                if (!cond) {
                    cond = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                }

                // Negate condition when body is on false branch
                if (i == 0) {
                    cond = negateCond(cond, ctx);
                }

                SNode *clause_body = leafFromNode(clause, factory);

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
                        exit_cond = negateCond(exit_cond, ctx);
                    }

                    auto *break_node = factory.make<SBreak>();
                    auto *if_break = factory.make<SIfThenElse>(exit_cond, break_node, nullptr);

                    SNode *header_block = factory.make<SBlock>();
                    for (auto *s : bl.stmts) static_cast<SBlock *>(header_block)->addStmt(s);
                    if (!bl.label.empty()) {
                        header_block = factory.make<SLabel>(factory.intern(bl.label), header_block);
                    }

                    auto *seq = factory.make<SSeq>();
                    seq->addChild(header_block);
                    seq->addChild(if_break);
                    seq->addChild(clause_body);
                    while_node = factory.make<SWhile>(true_lit, seq);
                } else {
                    while_node = factory.make<SWhile>(cond, clause_body);
                    if (!bl.label.empty()) {
                        while_node = factory.make<SLabel>(factory.intern(bl.label), while_node);
                        bl.label.clear();
                    }
                }

                g.collapseNodes({id, clause_id}, while_node);
                return true;
            }
            return false;
        }

        // Rule: Do-while loop (single block looping to itself)
        bool ruleBlockDoWhile(CGraph &g, size_t id, SNodeFactory &factory,
                              clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 2) return false;
            if (bl.isSwitchOut()) return false;
            if (bl.isGotoOut(0) || bl.isGotoOut(1)) return false;

            for (size_t i = 0; i < 2; ++i) {
                if (bl.succs[i] != id) continue;  // must loop to self

                clang::Expr *cond = bl.branch_cond;
                if (!cond) {
                    cond = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                }

                // Negate condition when self-loop is on false branch
                if (i == 0) {
                    cond = negateCond(cond, ctx);
                }

                // Save label before leafFromNode clears it — for loops,
                // the label should wrap the entire loop (not be inside the body)
                std::string saved_label = bl.label;
                bl.label.clear();
                auto *body = leafFromNode(bl, factory);
                SNode *dowhile_node = factory.make<SDoWhile>(body, cond);
                if (!saved_label.empty()) {
                    dowhile_node = factory.make<SLabel>(factory.intern(saved_label), dowhile_node);
                }

                // Use collapseNodes to handle edge cleanup uniformly
                g.collapseNodes({id}, dowhile_node);
                g.node(id).is_conditional = false;
                g.node(id).branch_cond = nullptr;
                return true;
            }
            return false;
        }

        // Rule: Infinite loop (single out to self)
        bool ruleBlockInfLoop(CGraph &g, size_t id, SNodeFactory &factory,
                              clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 1) return false;
            if (bl.isGotoOut(0)) return false;
            if (bl.succs[0] != id) return false;  // must loop to self

            std::string saved_label = bl.label;
            bl.label.clear();
            auto *body = leafFromNode(bl, factory);
            auto *true_lit = clang::IntegerLiteral::Create(
                ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
            SNode *loop = factory.make<SWhile>(true_lit, body);
            if (!saved_label.empty()) {
                loop = factory.make<SLabel>(factory.intern(saved_label), loop);
            }

            // Use collapseNodes to handle edge cleanup uniformly
            g.collapseNodes({id}, loop);
            return true;
        }

        // Rule: If with no exit (clause has zero out edges)
        bool ruleBlockIfNoExit(CGraph &g, size_t id, SNodeFactory &factory,
                               clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 2) return false;
            if (bl.isSwitchOut()) return false;
            if (bl.isGotoOut(0) || bl.isGotoOut(1)) return false;

            for (size_t i = 0; i < 2; ++i) {
                size_t clause_id = bl.succs[i];
                if (clause_id == id) continue;
                auto &clause = g.node(clause_id);
                if (clause.collapsed) continue;
                if (clause.sizeIn() != 1 || clause.sizeOut() != 0) continue;
                if (!bl.isDecisionOut(i)) continue;

                clang::Expr *cond = bl.branch_cond;
                if (!cond) {
                    cond = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                }

                auto *clause_body = leafFromNode(clause, factory);
                SNode *if_node = factory.make<SIfThenElse>(cond, clause_body, nullptr);
                if (!bl.label.empty()) {
                    if_node = factory.make<SLabel>(factory.intern(bl.label), if_node);
                    bl.label.clear();
                }

                g.collapseNodes({id, clause_id}, if_node);
                return true;
            }
            return false;
        }

        // Rule: Switch statement
        bool ruleBlockSwitch(CGraph &g, size_t id, SNodeFactory &factory,
                             clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || !bl.isSwitchOut()) return false;
            // Must have switch_cases metadata to fire.
            if (bl.switch_cases.empty()) return false;

            // Find exit block: look for a successor with sizeIn > 1 or sizeOut > 1
            size_t exit_id = std::numeric_limits<size_t>::max();
            for (size_t s : bl.succs) {
                auto &sn = g.node(s);
                if (sn.collapsed) continue;
                if (s == id || sn.sizeIn() > 1 || sn.sizeOut() > 1) {
                    exit_id = s;
                    break;
                }
            }

            // Validate: each case must have sizeIn==1.
            // Cases may exit to exit_id (break), have no exit (tail), or exit
            // to a different target (goto — e.g., loop back-edge).
            for (size_t s : bl.succs) {
                if (s == exit_id) continue;
                auto &sn = g.node(s);
                if (sn.collapsed) return false;
                if (sn.sizeIn() != 1) return false;
                if (sn.sizeOut() > 1) return false;
            }

            // Build the switch SNode from branch_cond and successor nodes.
            clang::Expr *disc = bl.branch_cond;
            if (!disc) {
                disc = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(32, 0), ctx.IntTy, clang::SourceLocation());
            }
            auto *sw = factory.make<SSwitch>(disc);

            // Build a map from succ_index to case values (multiple cases
            // can target the same successor).
            std::unordered_map<size_t, std::vector<int64_t>> succ_to_values;
            for (const auto &entry : bl.switch_cases) {
                succ_to_values[entry.succ_index].push_back(entry.value);
            }

            unsigned iw = ctx.getIntWidth(ctx.IntTy);
            for (size_t si = 0; si < bl.succs.size(); ++si) {
                size_t s = bl.succs[si];
                if (s == exit_id) continue;
                auto it = succ_to_values.find(si);
                if (it != succ_to_values.end()) {
                    auto *body = leafFromNode(g.node(s), factory);
                    for (int64_t val : it->second) {
                        auto *case_val = clang::IntegerLiteral::Create(
                            ctx, llvm::APInt(iw, static_cast<uint64_t>(val), true),
                            ctx.IntTy, clang::SourceLocation());
                        sw->addCase(case_val, body);
                    }
                } else {
                    // Successor with no case value — not a case arm; skip it
                    // (it is the fallback/exit path, not a switch case).
                }
            }
            // Prepend the switch block's own stmts (ops before the switch)
            // and strip the original SwitchStmt (CollapseStructure rebuilds it).
            SNode *sw_node = nullptr;
            if (!bl.stmts.empty()) {
                auto *seq = factory.make<SSeq>();
                auto *block = factory.make<SBlock>();
                for (auto *s : bl.stmts) {
                    // Skip the original SwitchStmt/CompoundStmt containing it
                    if (llvm::isa<clang::SwitchStmt>(s)) continue;
                    if (auto *cs = llvm::dyn_cast<clang::CompoundStmt>(s)) {
                        bool has_sw = false;
                        for (auto *child : cs->body())
                            if (llvm::isa<clang::SwitchStmt>(child)) { has_sw = true; break; }
                        if (has_sw) continue;
                    }
                    block->addStmt(s);
                }
                if (!block->stmts().empty()) seq->addChild(block);
                seq->addChild(sw);
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
                sw_node = factory.make<SLabel>(factory.intern(bl.label), sw_node);
                bl.label.clear();
            }
            size_t rep = g.collapseNodes(collapse_ids, sw_node);
            // The collapsed node has no real branch condition — prevent
            // ruleBlockGoto from emitting a spurious if(1) goto.
            g.node(rep).is_conditional = false;
            g.node(rep).switch_cases.clear();
            return true;
        }

        // Rule: Mark goto edges
        bool ruleBlockGoto(CGraph &g, size_t id, SNodeFactory &factory,
                           clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed) return false;

            for (size_t i = 0; i < bl.succs.size(); ++i) {
                if (!bl.isGotoOut(i)) continue;

                // Use the target CNode's actual label if available, otherwise
                // fall back to synthetic "block_N" label.
                auto &target_node = g.node(bl.succs[i]);
                std::string target_label = target_node.label.empty()
                    ? "block_" + std::to_string(bl.succs[i])
                    : target_node.label;
                auto *goto_node = factory.make<SGoto>(factory.intern(target_label));

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
                        cond = negateCond(cond, ctx);
                    }

                    auto *if_goto = factory.make<SIfThenElse>(cond, goto_node, nullptr);

                    auto *seq = factory.make<SSeq>();
                    if (!bl.stmts.empty() || bl.structured) {
                        // leafFromNode embeds the label around stmts/structured
                        seq->addChild(leafFromNode(bl, factory));
                    }
                    seq->addChild(if_goto);

                    SNode *result = seq;
                    // If leafFromNode wasn't called, wrap with label now
                    if (bl.stmts.empty() && !bl.structured && !bl.label.empty()) {
                        result = factory.make<SLabel>(factory.intern(bl.label), result);
                    }
                    bl.structured = result;
                    bl.label.clear();
                    bl.is_conditional = false;
                    bl.branch_cond = nullptr;
                    g.removeEdge(id, bl.succs[i]);
                    return true;
                }

                // Unconditional goto: wrap stmts + goto
                auto *body = leafFromNode(bl, factory);
                auto *seq = factory.make<SSeq>();
                seq->addChild(body);
                seq->addChild(goto_node);

                bl.structured = seq;
                bl.label.clear();  // label embedded via leafFromNode
                g.removeEdge(id, bl.succs[i]);
                return true;
            }
            return false;
        }

        // Gap 4: Switch case fallthrough detection (late-stage fallback)
        bool ruleCaseFallthru(CGraph &g, size_t id) {
            auto &bl = g.node(id);
            if (bl.collapsed || !bl.isSwitchOut()) return false;

            std::vector<size_t> fallthru;
            int nonfallthru = 0;

            for (size_t i = 0; i < bl.succs.size(); ++i) {
                size_t case_id = bl.succs[i];
                if (case_id == id) return false;
                auto &casebl = g.node(case_id);

                if (casebl.sizeIn() > 2 || casebl.sizeOut() > 1) {
                    nonfallthru++;
                } else if (casebl.sizeOut() == 1) {
                    size_t target_id = casebl.succs[0];
                    auto &target = g.node(target_id);
                    if (target.sizeIn() == 2 && target.sizeOut() <= 1) {
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
                g.node(fid).setGoto(0);
            }
            return true;
        }

        // ---------------------------------------------------------------
        // AND/OR condition collapsing (ruleBlockOr)
        // ---------------------------------------------------------------

        // Detects chained if-gotos and collapses them into compound conditions.
        // OR pattern (i==0): bl->false leads to orblock, both reach same clauseblock => BO_LOr
        // AND pattern (i==1): bl->true leads to orblock, both reach same clauseblock => BO_LAnd
        static bool ruleBlockOr(CGraph &g, size_t id, SNodeFactory &/*factory*/,
                                 clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 2) return false;
            if (bl.isSwitchOut()) return false;
            if (bl.isGotoOut(0) || bl.isGotoOut(1)) return false;

            for (size_t i = 0; i < 2; ++i) {
                size_t or_id = bl.succs[i];
                if (or_id == id) continue;
                auto &orblock = g.node(or_id);
                if (orblock.collapsed) continue;
                if (orblock.sizeIn() != 1 || orblock.sizeOut() != 2) continue;
                if (orblock.isSwitchOut()) continue;
                if (bl.isBackEdge(i)) continue;

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
                        cond_b = negateCond(cond_b, ctx);
                    }
                } else {
                    // AND: bl's true leads to orblock, bl's false leads to clauseblock
                    // If j==1, orblock's true leads to clause -- negate cond_b
                    if (j == 1) {
                        cond_b = negateCond(cond_b, ctx);
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
                auto &np = g.node(new_other).preds;
                np.erase(std::remove(np.begin(), np.end(), or_id), np.end());
                if (std::find(np.begin(), np.end(), id) == np.end()) {
                    np.push_back(id);
                }

                // Remove orblock from clause_id's preds
                auto &cp = g.node(clause_id).preds;
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

        static void collapseConditions(CGraph &g, SNodeFactory &factory,
                                        clang::ASTContext &ctx) {
            // Iteratively apply ruleBlockOr until no more changes.
            // This collapses all AND/OR chains before the main collapse loop.
            bool changed = true;
            while (changed) {
                changed = false;
                for (auto &n : g.nodes) {
                    if (n.collapsed) continue;
                    if (ruleBlockOr(g, n.id, factory, ctx)) {
                        changed = true;
                        break; // restart iteration since graph changed
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // TraceDAG-based goto selection (replaces selectAndMarkGoto)
        // ---------------------------------------------------------------

        /// clipExtraRoots: final fallback for unreachable/disconnected regions.
        /// Finds active nodes with no predecessors (other than entry) and adds
        /// a FloatingEdge for an incoming edge to reconnect them via goto.
        static void clipExtraRoots(CGraph &g, std::list<detail::FloatingEdge> &likelygoto) {
            for (auto &n : g.nodes) {
                if (n.collapsed) continue;
                if (n.id == g.entry) continue;
                if (n.sizeIn() != 0) continue;

                // Extra root: find any active predecessor that has an edge to it
                for (auto &pred : g.nodes) {
                    if (pred.collapsed) continue;
                    for (size_t i = 0; i < pred.succs.size(); ++i) {
                        if (pred.succs[i] == n.id && !pred.isGotoOut(i)) {
                            likelygoto.emplace_back(pred.id, n.id);
                            goto found_pred;
                        }
                    }
                }
                found_pred:;
            }
        }

        /// updateLoopBody: run TraceDAG loop-scoped (innermost-first),
        /// then fall back to full-graph TraceDAG.
        static bool updateLoopBody(CGraph &g,
                                   std::list<detail::LoopBody> &loopbody,
                                   std::list<detail::FloatingEdge> &likelygoto) {
            // Try each loop body (innermost-first)
            for (auto &loop : loopbody) {
                if (!loop.update(g)) continue;

                std::vector<size_t> body;
                loop.findBase(g, body);
                loop.setExitMarks(g, body);

                detail::TraceDAG dag(likelygoto);
                dag.addRoot(loop.head);
                if (loop.exit_block != detail::LoopBody::NONE &&
                    loop.exit_block < g.nodes.size() &&
                    !g.node(loop.exit_block).collapsed) {
                    dag.setFinishBlock(loop.exit_block);
                }
                dag.initialize();
                dag.pushBranches(g);

                loop.clearExitMarks(g, body);
                detail::clearMarks(g, body);

                if (!likelygoto.empty()) return true;

                // Post-TraceDAG heuristic: if TraceDAG found nothing within
                // the loop body, look for "extra loop exits" — edges from
                // non-header body nodes to outside the body.  These must
                // become gotos for the body to collapse into a chainable
                // single-successor structure required by ruleBlockWhileDo.
                // Prefer the deepest (last-in-body-order) edge first.
                {
                    std::unordered_set<size_t> bodyset(body.begin(), body.end());
                    for (auto rit = body.rbegin(); rit != body.rend(); ++rit) {
                        size_t bid = *rit;
                        if (bid == loop.head) continue;
                        auto &bn = g.node(bid);
                        for (size_t i = 0; i < bn.succs.size(); ++i) {
                            if (bn.isGotoOut(i) || bn.isBackEdge(i)) continue;
                            if (bodyset.count(bn.succs[i]) == 0) {
                                likelygoto.emplace_back(bid, bn.succs[i]);
                            }
                        }
                        if (!likelygoto.empty()) return true;
                    }
                }
            }

            // Fall back to full-graph TraceDAG
            {
                detail::TraceDAG dag(likelygoto);

                // Add active root nodes (entry or nodes with no predecessors)
                bool has_root = false;
                for (auto &n : g.nodes) {
                    if (n.collapsed) continue;
                    if (n.id == g.entry || n.sizeIn() == 0) {
                        dag.addRoot(n.id);
                        has_root = true;
                    }
                }
                if (!has_root) return false;

                dag.initialize();
                dag.pushBranches(g);
                return !likelygoto.empty();
            }
        }

        /// selectGoto: pick the least-disruptive edge to mark as goto using TraceDAG.
        /// Replaces the old selectAndMarkGoto heuristic.
        static bool selectGoto(CGraph &g, std::list<detail::LoopBody> &loopbody) {
            std::list<detail::FloatingEdge> likelygoto;

            if (!updateLoopBody(g, loopbody, likelygoto)) {
                // TraceDAG found nothing; try clipExtraRoots as final fallback
                clipExtraRoots(g, likelygoto);
            }

            // Iterate likelygoto and mark the first valid edge as goto
            for (auto &fe : likelygoto) {
                auto [src, edge_idx] = fe.getCurrentEdge(g);
                if (src != CNode::NONE) {
                    g.node(src).setGoto(edge_idx);
                    return true;
                }
            }

            return false;
        }

        // ---------------------------------------------------------------
        // Guard-chain absorption — mark guard→fallback edges as goto so
        // ruleBlockGoto + ruleBlockCat chain the guards into the switch,
        // reducing the fallback block's sizeIn and unblocking ruleBlockSwitch.
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
        /// ruleBlockSwitch rejects this because T.sizeIn != 1.
        ///
        /// Fix: mark each guard's edge to T as F_GOTO. Then:
        ///   ruleBlockGoto converts guards to "if(cond) goto T" (sizeOut→1)
        ///   ruleBlockCat chains guards into the switch predecessor
        ///   T.sizeIn drops → ruleBlockSwitch can fire
        static bool absorbSwitchGuards(CGraph &g) {
            bool changed = false;

            for (auto &sw : g.nodes) {
                if (sw.collapsed || !sw.isSwitchOut()) continue;
                if (sw.sizeIn() != 1) continue;  // switch needs unique predecessor

                // Find the fallback target: a successor of the switch with sizeIn > 1.
                // This is the block that both the guard chain and the switch's
                // fallback edge point to.
                size_t fallback_id = CNode::NONE;
                for (size_t s : sw.succs) {
                    if (g.node(s).sizeIn() > 1 && !g.node(s).collapsed) {
                        // Prefer a non-conditional single-exit block as fallback
                        // (the exit/continue block, not the loop header).
                        auto &candidate = g.node(s);
                        if (!candidate.is_conditional && candidate.sizeOut() <= 1) {
                            fallback_id = s;
                            break;
                        }
                    }
                }
                if (fallback_id == CNode::NONE) continue;

                // Walk backwards from the switch through the unique predecessor
                // chain, collecting guard blocks that have one edge to the
                // fallback target.
                std::vector<size_t> guards;
                size_t cur = sw.preds[0];
                while (true) {
                    auto &gn = g.node(cur);
                    if (gn.collapsed) break;
                    if (!gn.is_conditional || gn.sizeOut() != 2) break;
                    if (gn.sizeIn() < 1) break;

                    // One successor must be the next block in the chain (or the
                    // switch), and the other must be the fallback target.
                    bool has_fallback_edge = false;
                    for (size_t i = 0; i < 2; ++i) {
                        if (gn.succs[i] == fallback_id && !gn.isGotoOut(i)) {
                            has_fallback_edge = true;
                        }
                    }
                    if (!has_fallback_edge) break;

                    guards.push_back(cur);

                    // Continue walking if this guard has a unique predecessor
                    if (gn.sizeIn() != 1) break;
                    cur = gn.preds[0];
                }

                if (guards.empty()) continue;

                // Mark each guard's edge to the fallback as F_GOTO
                for (size_t gid : guards) {
                    auto &gn = g.node(gid);
                    for (size_t i = 0; i < gn.succs.size(); ++i) {
                        if (gn.succs[i] == fallback_id && !gn.isGotoOut(i)) {
                            gn.setGoto(i);
                            LOG(INFO) << "absorbSwitchGuards: marked guard "
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
        // Control-equivalence hoisting — unblock ruleBlockProperIf /
        // ruleBlockIfElse by duplicating or absorbing small shared blocks.
        // ---------------------------------------------------------------

        /// Clause splitting: when a conditional block's clause has sizeIn > 1
        /// (shared with other predecessors), duplicate it so the conditional
        /// can fire ruleBlockProperIf.
        ///
        /// Pattern: cond→clause→other, cond→other  (clause.sizeIn > 1)
        /// After:   cond→clause_copy→other, cond→other  (clause_copy.sizeIn == 1)
        static bool tryClauseSplit(CGraph &g) {
            for (auto &bl : g.nodes) {
                if (bl.collapsed || bl.sizeOut() != 2 || !bl.is_conditional)
                    continue;
                if (bl.isGotoOut(0) || bl.isGotoOut(1))
                    continue;

                for (size_t i = 0; i < 2; ++i) {
                    size_t clause_id = bl.succs[i];
                    if (clause_id == bl.id) continue;
                    auto &clause = g.node(clause_id);
                    if (clause.collapsed) continue;

                    // Clause must exit to the other successor of cond
                    if (clause.sizeOut() != 1) continue;
                    if (clause.succs[0] != bl.succs[1 - i]) continue;

                    // Already unique — ruleBlockProperIf should handle it
                    if (clause.sizeIn() == 1) continue;

                    // Don't duplicate conditionals, large blocks, labels,
                    // or blocks with back-edge predecessors
                    if (clause.is_conditional) continue;
                    if (clause.stmts.size() > 16) continue;
                    if (!clause.label.empty()) continue;
                    if (clause.isGotoOut(0)) continue;

                    bool has_back_pred = false;
                    for (size_t p : clause.preds) {
                        auto &pn = g.node(p);
                        for (size_t ei = 0; ei < pn.succs.size(); ++ei) {
                            if (pn.succs[ei] == clause_id && pn.isBackEdge(ei)) {
                                has_back_pred = true;
                                break;
                            }
                        }
                        if (has_back_pred) break;
                    }
                    if (has_back_pred) continue;

                    // --- Duplicate clause as a new node ---
                    // Copy data BEFORE push_back (vector may realloc)
                    CNode copy;
                    copy.stmts = clause.stmts;
                    copy.branch_cond = clause.branch_cond;
                    copy.is_conditional = clause.is_conditional;
                    copy.structured = clause.structured;

                    size_t copy_id = g.nodes.size();
                    copy.id = copy_id;
                    copy.preds = {bl.id};
                    copy.succs = {clause.succs[0]};
                    copy.edge_flags = {clause.edge_flags[0]};

                    g.nodes.push_back(std::move(copy));
                    // References invalidated — use g.node() from here

                    // Redirect cond's edge from clause to copy
                    g.node(bl.id).succs[i] = copy_id;

                    // Remove cond from clause's pred list
                    auto &cpreds = g.node(clause_id).preds;
                    cpreds.erase(std::remove(cpreds.begin(), cpreds.end(), bl.id),
                                 cpreds.end());

                    // Add copy to exit block's pred list
                    size_t exit_id = g.node(copy_id).succs[0];
                    g.node(exit_id).preds.push_back(copy_id);

                    LOG(INFO) << "tryClauseSplit: duplicated node " << clause_id
                              << " as " << copy_id << " for cond " << bl.id << "\n";
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
        /// After:   absorb J's stmts into A, redirect A→K → ruleBlockIfElse fires
        static bool tryJoinAbsorb(CGraph &g) {
            for (auto &bl : g.nodes) {
                if (bl.collapsed || bl.sizeOut() != 2 || !bl.is_conditional)
                    continue;
                if (bl.isGotoOut(0) || bl.isGotoOut(1))
                    continue;

                size_t a_id = bl.succs[1];  // true branch
                size_t b_id = bl.succs[0];  // false branch

                // Try both orientations: absorb into A or absorb into B
                for (int orient = 0; orient < 2; ++orient) {
                    if (orient == 1) std::swap(a_id, b_id);

                    auto &a = g.node(a_id);
                    auto &b = g.node(b_id);
                    if (a.collapsed || b.collapsed) continue;
                    if (a.sizeIn() != 1 || a.sizeOut() != 1) continue;
                    if (b.sizeIn() != 1 || b.sizeOut() != 1) continue;
                    if (a.isGotoOut(0) || b.isGotoOut(0)) continue;

                    size_t j_id = a.succs[0];  // A's successor (candidate for absorption)
                    size_t k_id = b.succs[0];  // B's successor

                    if (j_id == k_id) continue;  // already same exit → ifElse should fire
                    auto &j = g.node(j_id);
                    if (j.collapsed) continue;
                    if (j.sizeIn() <= 1) continue;  // not shared
                    if (j.sizeOut() > 1) continue;   // must have <=1 exit
                    if (j.is_conditional) continue;
                    if (!j.label.empty()) continue;
                    if (j.stmts.size() > 16) continue;

                    // J must exit to K (so after absorption, A→K matches B→K)
                    if (j.sizeOut() == 1 && j.succs[0] != k_id) continue;
                    // If J has no successors, B must also have no successors
                    if (j.sizeOut() == 0 && b.sizeOut() != 0) continue;

                    // Absorb: append J's stmts to A, redirect A past J
                    auto &an = g.node(a_id);
                    for (auto *s : g.node(j_id).stmts) {
                        an.stmts.push_back(s);
                    }

                    // Remove edge A→J
                    g.removeEdge(a_id, j_id);

                    if (j.sizeOut() == 1) {
                        // Add edge A→K
                        an.succs.push_back(k_id);
                        an.edge_flags.push_back(0);
                        g.node(k_id).preds.push_back(a_id);
                    }

                    LOG(INFO) << "tryJoinAbsorb: absorbed node " << j_id
                              << " into " << a_id << " for cond " << bl.id << "\n";
                    return true;
                }
            }
            return false;
        }

        /// Try control-equivalence hoisting transforms to unblock the main
        /// collapse rules before falling back to goto selection.
        static bool tryControlEquivHoist(CGraph &g) {
            if (tryClauseSplit(g)) return true;
            if (tryJoinAbsorb(g)) return true;
            return false;
        }

        // ---------------------------------------------------------------
        // Main collapse loop
        // ---------------------------------------------------------------

        size_t collapseInternal(CGraph &g, SNodeFactory &factory, clang::ASTContext &ctx) {
            bool change;
            size_t isolated_count;

            do {
                do {
                    change = false;
                    isolated_count = 0;
                    for (auto &n : g.nodes) {
                        if (n.collapsed) continue;
                        if (n.sizeIn() == 0 && n.sizeOut() == 0) {
                            isolated_count++;
                            continue;
                        }

                        if (ruleBlockGoto(g, n.id, factory, ctx)) { change = true; continue; }
                        if (ruleBlockCat(g, n.id, factory)) { change = true; continue; }
                        if (ruleBlockProperIf(g, n.id, factory, ctx)) { change = true; continue; }
                        if (ruleBlockIfElse(g, n.id, factory, ctx)) { change = true; continue; }
                        if (ruleBlockWhileDo(g, n.id, factory, ctx)) { change = true; continue; }
                        if (ruleBlockDoWhile(g, n.id, factory, ctx)) { change = true; continue; }
                        if (ruleBlockInfLoop(g, n.id, factory, ctx)) { change = true; continue; }
                        if (ruleBlockSwitch(g, n.id, factory, ctx)) { change = true; continue; }
                    }
                } while (change);

                // Try IfNoExit as fallback (Ghidra applies this only when stuck)
                change = false;
                for (auto &n : g.nodes) {
                    if (n.collapsed) continue;
                    if (ruleBlockIfNoExit(g, n.id, factory, ctx)) {
                        change = true;
                        break;
                    }
                    if (ruleCaseFallthru(g, n.id)) {
                        change = true;
                        break;
                    }
                }
            } while (change);

            return isolated_count;
        }

        // ---------------------------------------------------------------
        // Post-collapse transforms: scopeBreak, whileToFor, markLabelBumpUp
        // ---------------------------------------------------------------

        // scopeBreak: convert loop-exit gotos to SBreak and loop-header gotos
        // to SContinue. Walks the SNode tree, tracking the current loop context
        // via exit/header label names.
        void scopeBreak(SNode *node, std::string_view loop_exit_label,
                        std::string_view loop_header_label, SNodeFactory &factory) {
            if (!node) return;

            if (auto *seq = node->dyn_cast<SSeq>()) {
                // Recurse into children. When entering a loop child, we need to
                // determine the new loop exit/header labels from sibling context.
                for (size_t i = 0; i < seq->size(); ++i) {
                    auto *child = (*seq)[i];

                    // For SWhile/SDoWhile, compute new loop context labels
                    if (auto *w = child->dyn_cast<SWhile>()) {
                        // New loop_exit_label: label of the next sibling (if SLabel)
                        std::string_view new_exit;
                        if (i + 1 < seq->size()) {
                            if (auto *lbl = (*seq)[i + 1]->dyn_cast<SLabel>()) {
                                new_exit = lbl->name();
                            }
                        }
                        if (new_exit.empty()) new_exit = loop_exit_label;

                        // New loop_header_label: if prev sibling is SLabel wrapping
                        // this while, use that label. Also check if the while body's
                        // first child is an SLabel (header-stmts-inside-loop pattern).
                        std::string_view new_header;
                        if (i > 0) {
                            if (auto *lbl = (*seq)[i - 1]->dyn_cast<SLabel>()) {
                                new_header = lbl->name();
                            }
                        }
                        if (new_header.empty()) {
                            if (auto *body_seq = w->body() ? w->body()->dyn_cast<SSeq>() : nullptr) {
                                if (body_seq->size() > 0) {
                                    if (auto *lbl = (*body_seq)[0]->dyn_cast<SLabel>()) {
                                        new_header = lbl->name();
                                    }
                                }
                            }
                        }

                        scopeBreak(w->body(), new_exit, new_header, factory);
                        continue;
                    }

                    if (auto *dw = child->dyn_cast<SDoWhile>()) {
                        std::string_view new_exit;
                        if (i + 1 < seq->size()) {
                            if (auto *lbl = (*seq)[i + 1]->dyn_cast<SLabel>()) {
                                new_exit = lbl->name();
                            }
                        }
                        if (new_exit.empty()) new_exit = loop_exit_label;

                        std::string_view new_header;
                        if (i > 0) {
                            if (auto *lbl = (*seq)[i - 1]->dyn_cast<SLabel>()) {
                                new_header = lbl->name();
                            }
                        }

                        scopeBreak(dw->body(), new_exit, new_header, factory);
                        continue;
                    }

                    // For non-loop children, recurse with current loop context
                    scopeBreak(child, loop_exit_label, loop_header_label, factory);
                }

                // After recursing, check if last child is SGoto targeting loop exit/header
                if (seq->size() > 0) {
                    auto *last = (*seq)[seq->size() - 1];
                    if (auto *g = last->dyn_cast<SGoto>()) {
                        if (!loop_exit_label.empty() && g->target() == loop_exit_label) {
                            seq->replaceChild(seq->size() - 1, factory.make<SBreak>());
                        } else if (!loop_header_label.empty() && g->target() == loop_header_label) {
                            seq->replaceChild(seq->size() - 1, factory.make<SContinue>());
                        }
                    }
                }
            }
            else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                scopeBreak(ite->thenBranch(), loop_exit_label, loop_header_label, factory);
                scopeBreak(ite->elseBranch(), loop_exit_label, loop_header_label, factory);
            }
            else if (auto *sw = node->dyn_cast<SSwitch>()) {
                // Switch does not change loop context
                for (auto &c : sw->cases()) {
                    scopeBreak(c.body, loop_exit_label, loop_header_label, factory);
                }
                scopeBreak(sw->defaultBody(), loop_exit_label, loop_header_label, factory);
            }
            else if (auto *lbl = node->dyn_cast<SLabel>()) {
                scopeBreak(lbl->body(), loop_exit_label, loop_header_label, factory);
            }
            else if (auto *f = node->dyn_cast<SFor>()) {
                // SFor has its own loop context
                scopeBreak(f->body(), loop_exit_label, loop_header_label, factory);
            }
            // SBlock, SGoto, SBreak, SContinue, SReturn: leaf nodes, nothing to do
        }

        // --- whileToFor helpers ---

        bool isAssignOrDecl(clang::Stmt *s) {
            if (!s) return false;
            if (auto *bo = llvm::dyn_cast<clang::BinaryOperator>(s)) {
                return bo->getOpcode() == clang::BO_Assign;
            }
            return llvm::isa<clang::DeclStmt>(s);
        }

        bool isIncrement(clang::Stmt *s) {
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
                        // Return nullptr — sameVariable will handle this via VarDecl matching.
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
        bool containsVarRef(clang::Stmt *s, clang::VarDecl *vd) {
            if (!s || !vd) return false;
            if (auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(s)) {
                return dre->getDecl() == vd;
            }
            for (auto *child : s->children()) {
                if (containsVarRef(child, vd)) return true;
            }
            return false;
        }

        // Check that init, cond, and inc all reference the same variable
        bool sameVariable(clang::Stmt *init, clang::Expr *cond, clang::Expr *inc) {
            auto *init_var = getReferencedVar(init);
            if (!init_var) return false;
            auto *inc_var = getReferencedVar(inc);
            if (!inc_var) return false;
            if (init_var != inc_var) return false;
            return containsVarRef(cond, init_var);
        }

        // whileToFor: convert init/while(cond)/inc patterns to SFor
        void whileToFor(SNode *node, SNodeFactory &factory, clang::ASTContext &ctx) {
            if (!node) return;

            if (auto *seq = node->dyn_cast<SSeq>()) {
                // Scan for pattern: SBlock(init), SWhile(cond, SSeq(..., SBlock(inc)))
                for (size_t i = 0; i < seq->size(); ++i) {
                    auto *w = (*seq)[i]->dyn_cast<SWhile>();
                    if (!w) continue;

                    // Check for init before the while
                    clang::Stmt *init_stmt = nullptr;
                    bool has_init = false;
                    if (i > 0) {
                        auto *prev = (*seq)[i - 1]->dyn_cast<SBlock>();
                        if (prev && prev->size() == 1 && isAssignOrDecl(prev->stmts()[0])) {
                            init_stmt = prev->stmts()[0];
                            has_init = true;
                        }
                    }

                    // Check for inc at end of while body
                    clang::Expr *inc_expr = nullptr;
                    auto *body_seq = w->body() ? w->body()->dyn_cast<SSeq>() : nullptr;
                    if (body_seq && body_seq->size() > 0) {
                        auto *last = (*body_seq)[body_seq->size() - 1]->dyn_cast<SBlock>();
                        if (last && last->size() == 1 && isIncrement(last->stmts()[0])) {
                            inc_expr = llvm::dyn_cast<clang::Expr>(last->stmts()[0]);
                        }
                    }

                    if (has_init && inc_expr && sameVariable(init_stmt, w->cond(), inc_expr)) {
                        // Remove inc from body
                        body_seq->removeChild(body_seq->size() - 1);
                        // Build SFor
                        auto *for_node = factory.make<SFor>(init_stmt, w->cond(), inc_expr, w->body());
                        // Replace [i-1, i+1) with the for node
                        seq->replaceRange(i - 1, i + 1, for_node);
                        --i; // adjust for removed element
                    }
                }

                // Recurse into remaining children
                for (size_t i = 0; i < seq->size(); ++i) {
                    whileToFor((*seq)[i], factory, ctx);
                }
            }
            else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                whileToFor(ite->thenBranch(), factory, ctx);
                whileToFor(ite->elseBranch(), factory, ctx);
            }
            else if (auto *w = node->dyn_cast<SWhile>()) {
                whileToFor(w->body(), factory, ctx);
            }
            else if (auto *dw = node->dyn_cast<SDoWhile>()) {
                whileToFor(dw->body(), factory, ctx);
            }
            else if (auto *f = node->dyn_cast<SFor>()) {
                whileToFor(f->body(), factory, ctx);
            }
            else if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->cases()) {
                    whileToFor(c.body, factory, ctx);
                }
                whileToFor(sw->defaultBody(), factory, ctx);
            }
            else if (auto *lbl = node->dyn_cast<SLabel>()) {
                whileToFor(lbl->body(), factory, ctx);
            }
        }

        // --- markLabelBumpUp: dead label removal ---

        // Collect all SGoto target names in the tree
        void collectGotoTargets(SNode *node, std::unordered_set<std::string> &targets) {
            if (!node) return;

            if (auto *g = node->dyn_cast<SGoto>()) {
                targets.emplace(g->target());
            }
            else if (auto *seq = node->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < seq->size(); ++i) {
                    collectGotoTargets((*seq)[i], targets);
                }
            }
            else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                collectGotoTargets(ite->thenBranch(), targets);
                collectGotoTargets(ite->elseBranch(), targets);
            }
            else if (auto *w = node->dyn_cast<SWhile>()) {
                collectGotoTargets(w->body(), targets);
            }
            else if (auto *dw = node->dyn_cast<SDoWhile>()) {
                collectGotoTargets(dw->body(), targets);
            }
            else if (auto *f = node->dyn_cast<SFor>()) {
                collectGotoTargets(f->body(), targets);
            }
            else if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->cases()) {
                    collectGotoTargets(c.body, targets);
                }
                collectGotoTargets(sw->defaultBody(), targets);
            }
            else if (auto *lbl = node->dyn_cast<SLabel>()) {
                collectGotoTargets(lbl->body(), targets);
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
                for (auto *s : blk->stmts()) scanStmt(s);
            }
        }

        // Remove SLabel nodes whose name is not in the goto target set.
        // If the SLabel has a body, replace the label with its body.
        // If no body, remove entirely.
        void removeDeadLabels(SNode *node,
                              const std::unordered_set<std::string> &targets) {
            if (!node) return;

            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < seq->size(); ) {
                    auto *child = (*seq)[i];
                    if (auto *lbl = child->dyn_cast<SLabel>()) {
                        if (targets.find(std::string(lbl->name())) == targets.end()) {
                            // Dead label -- replace with body or remove
                            if (lbl->body()) {
                                seq->replaceChild(i, lbl->body());
                                // Don't increment — re-check the replacement
                            } else {
                                seq->removeChild(i);
                            }
                            continue;
                        }
                    }
                    // Recurse into child
                    removeDeadLabels(child, targets);
                    ++i;
                }
            }
            else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                removeDeadLabels(ite->thenBranch(), targets);
                removeDeadLabels(ite->elseBranch(), targets);
            }
            else if (auto *w = node->dyn_cast<SWhile>()) {
                removeDeadLabels(w->body(), targets);
            }
            else if (auto *dw = node->dyn_cast<SDoWhile>()) {
                removeDeadLabels(dw->body(), targets);
            }
            else if (auto *f = node->dyn_cast<SFor>()) {
                removeDeadLabels(f->body(), targets);
            }
            else if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->cases()) {
                    removeDeadLabels(c.body, targets);
                }
                removeDeadLabels(sw->defaultBody(), targets);
            }
            else if (auto *lbl = node->dyn_cast<SLabel>()) {
                removeDeadLabels(lbl->body(), targets);
            }
        }

        // markLabelBumpUp: remove dead labels left after scopeBreak converts
        // gotos to break/continue. For v1, this is dead label removal only
        // (full bump-up optimization deferred).
        void markLabelBumpUp(SNode *root) {
            std::unordered_set<std::string> targets;
            collectGotoTargets(root, targets);
            removeDeadLabels(root, targets);
        }

    } // anonymous namespace

    // ---------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------

    SNode *collapseStructure(const Cfg &cfg, SNodeFactory &factory,
                             clang::ASTContext &ctx) {
        if (cfg.blocks.empty()) {
            return factory.make<SSeq>();
        }

        // 1. Build the collapse graph
        detail::CGraph g = detail::buildCGraph(cfg);

        // 2. Mark back-edges
        detail::markBackEdges(g);

        // 2b. Discover loops, compute bodies/nesting/exits, order innermost-first
        std::list<detail::LoopBody> loopbody;
        detail::orderLoopBodies(g, loopbody);
        LOG(INFO) << "CollapseStructure: found " << loopbody.size() << " loop(s)\n";

        // 2c. Pre-pass: for non-conditional 2-successor blocks where one
        // edge is a back-edge and the other is a forward exit, remove the
        // exit edge (the goto is inside the stmts) and clear F_BACK so the
        // block can be chained into the loop body.
        {
            bool changed = false;
            for (auto &bl : g.nodes) {
                if (bl.collapsed || bl.sizeOut() != 2 || bl.is_conditional)
                    continue;
                int back_idx = -1;
                for (size_t i = 0; i < 2; ++i) {
                    if (bl.isBackEdge(i))
                        back_idx = static_cast<int>(i);
                }
                if (back_idx < 0) continue;

                size_t exit_idx = 1 - static_cast<size_t>(back_idx);
                size_t exit_id = bl.succs[exit_idx];
                g.removeEdge(bl.id, exit_id);
                for (size_t i = 0; i < bl.edge_flags.size(); ++i)
                    bl.edge_flags[i] &= ~CNode::F_BACK;
                changed = true;
            }
            // Re-discover loops with updated graph so TraceDAG is consistent.
            if (changed) {
                loopbody.clear();
                detail::markBackEdges(g);
                detail::orderLoopBodies(g, loopbody);
            }
        }

        // 2d. Collapse AND/OR conditions before main collapse loop
        collapseConditions(g, factory, ctx);
        LOG(INFO) << "CollapseStructure: condition collapsing complete\n";

        // 2e. Absorb switch guard chains — mark guard→fallback edges as goto
        // so ruleBlockGoto + ruleBlockCat chain guards into the switch block,
        // reducing the fallback block's sizeIn for ruleBlockSwitch.
        absorbSwitchGuards(g);

        // 3. Main collapse loop
        size_t isolated = collapseInternal(g, factory, ctx);

        // 3b. Try control-equivalence hoisting before falling back to gotos.
        // This duplicates or absorbs small shared blocks to unblock
        // ruleBlockProperIf / ruleBlockIfElse.
        while (isolated < g.activeCount()) {
            if (!tryControlEquivHoist(g)) break;
            isolated = collapseInternal(g, factory, ctx);
        }

        // 4. When stuck, select gotos via TraceDAG and retry
        size_t max_iterations = g.nodes.size() * 4;  // safety bound
        size_t iter = 0;
        while (isolated < g.activeCount() && iter < max_iterations) {
            if (!selectGoto(g, loopbody)) {
                LOG(WARNING) << "CollapseStructure: could not select goto, "
                             << g.activeCount() - isolated << " blocks remaining\n";
                break;
            }
            isolated = collapseInternal(g, factory, ctx);
            ++iter;
        }

        // 5. Collect the final structured tree
        // Find the root (entry node or its collapsed representative)
        SNode *root = nullptr;
        for (auto &n : g.nodes) {
            if (n.collapsed) continue;
            // leafFromNode handles both structured and leaf nodes,
            // and wraps with SLabel when the CNode carries a label.
            auto *block = leafFromNode(n, factory);
            if (!root) {
                root = block;
            } else {
                auto *seq = root->dyn_cast<SSeq>();
                if (!seq) {
                    seq = factory.make<SSeq>();
                    seq->addChild(root);
                    root = seq;
                }
                seq->addChild(block);
            }
        }

        if (!root) root = factory.make<SSeq>();

        // 6. Post-collapse transforms (order matters per research)
        scopeBreak(root, "", "", factory);    // 1st: gotos -> break/continue
        whileToFor(root, factory, ctx);       // 2nd: while -> for patterns
        markLabelBumpUp(root);                // 3rd: clean up dead labels

        return root;
    }

} // namespace patchestry::ast
