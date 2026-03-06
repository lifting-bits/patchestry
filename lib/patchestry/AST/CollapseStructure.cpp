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
#include <clang/AST/Expr.h>

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

            // Mark all as collapsed except rep
            for (size_t nid : ids) {
                if (nid != rep) nodes[nid].collapsed = true;
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
            if (top_id >= g.nodes.size() || bottom_id >= g.nodes.size())
                return {CNode::NONE, 0};
            if (g.node(top_id).collapsed || g.node(bottom_id).collapsed)
                return {CNode::NONE, 0};
            const auto &succs = g.node(top_id).succs;
            for (size_t i = 0; i < succs.size(); ++i) {
                if (succs[i] == bottom_id)
                    return {top_id, i};
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
            BranchPoint *bp = trace->top;
            if (bp == nullptr) return false;

            // Check if all paths from this BranchPoint are terminal or converge
            size_t common_exit = CNode::NONE;
            bool all_done = true;

            for (auto *bt : bp->paths) {
                if (bt->isActive() && !bt->isTerminal()) {
                    all_done = false;
                    break;
                }
                if (bt->isTerminal()) {
                    // Terminal traces converge at their dest
                    size_t exit_cand = bt->dest_id;
                    if (common_exit == CNode::NONE) {
                        common_exit = exit_cand;
                    } else if (common_exit != exit_cand) {
                        // Different exits -- check if both are terminal (acceptable)
                        // In Ghidra, terminals with different exits still allow retirement
                    }
                } else if (!bt->isActive()) {
                    // Already retired trace
                    continue;
                }
            }

            if (!all_done) return false;

            // All paths from this BranchPoint are done
            exitblock_id = common_exit;
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
            // "worse" means more likely to be a goto candidate
            // 1. Having an exit conflict is worse than not having one
            if ((siblingedge > 0) != (op2.siblingedge > 0))
                return siblingedge > 0;
            // 2. Higher siblingedge count is worse
            if (siblingedge != op2.siblingedge)
                return siblingedge > op2.siblingedge;
            // 3. Terminal traces are worse (prefer removing terminals)
            if (terminal != op2.terminal)
                return terminal > op2.terminal;
            // 4. Lower distance is worse (closer = more disruptive)
            if (distance != op2.distance)
                return distance < op2.distance;
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
            // Build score list from all active traces
            std::list<BadEdgeScore> scores;
            for (auto *bt : activetrace) {
                BadEdgeScore score;
                score.exitproto_id = bt->dest_id;
                score.trace = bt;
                score.terminal = bt->isTerminal() ? 1 : 0;
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
            clearVisitCount(g);

            // Mark initial visit counts for roots
            for (size_t root_id : rootlist) {
                if (root_id < g.nodes.size() && !g.node(root_id).collapsed)
                    g.node(root_id).visit_count = 1;
            }

            int max_iter = static_cast<int>(g.nodes.size()) * 10;
            int iter = 0;

            while (activecount > 0 && iter < max_iter) {
                ++iter;
                bool progress = false;

                current_activeiter = activetrace.begin();
                while (current_activeiter != activetrace.end()) {
                    BlockTrace *bt = *current_activeiter;

                    if (checkOpen(g, bt)) {
                        // Trace advanced or terminated
                        ++current_activeiter;
                        progress = true;

                        // After advancing, try retiring BranchPoints
                        for (auto *bp : branchlist) {
                            size_t exit_id = CNode::NONE;
                            if (checkRetirement(bp->paths.empty() ? nullptr : bp->paths[0],
                                                exit_id)) {
                                current_activeiter = retireBranch(bp, exit_id);
                                progress = true;
                                break;
                            }
                        }
                        continue;
                    }

                    // Needs branching
                    current_activeiter = openBranch(g, bt);
                    progress = true;
                }

                if (!progress && activecount > 0) {
                    // Stuck: select bad edge
                    BlockTrace *bad = selectBadEdge();
                    if (bad == nullptr) break;

                    // Add the bad edge to likelygoto
                    size_t src = bad->bottom_id;
                    size_t dst = bad->dest_id;
                    if (src != CNode::NONE && dst != CNode::NONE) {
                        likelygoto.emplace_back(src, dst);
                    }

                    // Remove the trace
                    removeTrace(bad);
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

        SNode *leafFromNode(const CNode &n, SNodeFactory &factory) {
            if (n.structured) return n.structured;
            auto *block = factory.make<SBlock>();
            for (auto *s : n.stmts) block->addStmt(s);
            return block;
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
                auto *if_node = factory.make<SIfThenElse>(cond, clause_body, nullptr);

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
            auto *if_node = factory.make<SIfThenElse>(cond, then_body, else_body);

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

                auto *body = leafFromNode(clause, factory);
                auto *while_node = factory.make<SWhile>(cond, body);

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

                auto *body = leafFromNode(bl, factory);
                auto *dowhile_node = factory.make<SDoWhile>(body, cond);

                // Remove the self-edge, keep the exit edge
                size_t exit_id = bl.succs[1 - i];
                g.removeEdge(id, id);
                bl.structured = dowhile_node;
                bl.succs = {exit_id};
                bl.edge_flags = {0};
                bl.is_conditional = false;
                bl.branch_cond = nullptr;
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

            auto *body = leafFromNode(bl, factory);
            auto *true_lit = clang::IntegerLiteral::Create(
                ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
            auto *loop = factory.make<SWhile>(true_lit, body);

            g.removeEdge(id, id);
            bl.structured = loop;
            bl.succs.clear();
            bl.edge_flags.clear();
            bl.preds.erase(std::remove(bl.preds.begin(), bl.preds.end(), id), bl.preds.end());
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
                auto *if_node = factory.make<SIfThenElse>(cond, clause_body, nullptr);

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

            // Validate: each case must have sizeIn==1 and either exit to exit_id or have no exit
            for (size_t s : bl.succs) {
                if (s == exit_id) continue;
                auto &sn = g.node(s);
                if (sn.collapsed) return false;
                if (sn.sizeIn() != 1) return false;
                if (sn.sizeOut() == 1) {
                    if (exit_id != std::numeric_limits<size_t>::max() && sn.succs[0] != exit_id)
                        return false;
                    if (sn.isGotoOut(0)) return false;
                } else if (sn.sizeOut() > 1) {
                    return false;
                }
            }

            // Build the switch SNode
            clang::Expr *disc = bl.branch_cond;
            if (!disc) {
                disc = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(32, 0), ctx.IntTy, clang::SourceLocation());
            }
            auto *sw = factory.make<SSwitch>(disc);

            // Build a map from succ_index to case value for this switch block
            std::unordered_map<size_t, int64_t> succ_to_value;
            for (const auto &entry : bl.switch_cases) {
                succ_to_value[entry.succ_index] = entry.value;
            }

            std::vector<size_t> collapse_ids = {id};
            for (size_t si = 0; si < bl.succs.size(); ++si) {
                size_t s = bl.succs[si];
                if (s == exit_id) continue;
                collapse_ids.push_back(s);
                clang::Expr *case_val = nullptr;
                auto it = succ_to_value.find(si);
                if (it != succ_to_value.end()) {
                    case_val = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(64, static_cast<uint64_t>(it->second), true),
                        ctx.LongTy, clang::SourceLocation());
                }
                sw->addCase(case_val, leafFromNode(g.node(s), factory));
            }

            g.collapseNodes(collapse_ids, sw);
            return true;
        }

        // Rule: Mark goto edges
        bool ruleBlockGoto(CGraph &g, size_t id, SNodeFactory &factory) {
            auto &bl = g.node(id);
            if (bl.collapsed) return false;

            for (size_t i = 0; i < bl.succs.size(); ++i) {
                if (bl.isGotoOut(i)) {
                    // Wrap in a goto SNode
                    auto *body = leafFromNode(bl, factory);
                    auto *seq = factory.make<SSeq>();
                    seq->addChild(body);

                    std::string target_label = "block_" + std::to_string(bl.succs[i]);
                    seq->addChild(factory.make<SGoto>(factory.intern(target_label)));

                    bl.structured = seq;
                    // Remove the goto edge
                    g.removeEdge(id, bl.succs[i]);
                    return true;
                }
            }
            return false;
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

                        if (ruleBlockGoto(g, n.id, factory)) { change = true; continue; }
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
                }
            } while (change);

            return isolated_count;
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

        // 2c. Collapse AND/OR conditions before main collapse loop
        collapseConditions(g, factory, ctx);
        LOG(INFO) << "CollapseStructure: condition collapsing complete\n";

        // 3. Main collapse loop
        size_t isolated = collapseInternal(g, factory, ctx);

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
            if (n.structured) {
                if (!root) {
                    root = n.structured;
                } else {
                    // Multiple uncollapsed nodes -- wrap in sequence
                    auto *seq = root->dyn_cast<SSeq>();
                    if (!seq) {
                        seq = factory.make<SSeq>();
                        seq->addChild(root);
                        root = seq;
                    }
                    seq->addChild(n.structured);
                }
            } else {
                // Uncollapsed leaf -- add as block
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
        }

        if (!root) root = factory.make<SSeq>();
        return root;
    }

} // namespace patchestry::ast
