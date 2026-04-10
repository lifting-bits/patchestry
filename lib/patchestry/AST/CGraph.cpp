/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/CGraph.hpp>
#include <patchestry/Util/Log.hpp>

#include <algorithm>
#include <cassert>
#include <limits>
#include <list>
#include <unordered_set>
#include <vector>

namespace patchestry::ast {

    // ---------------------------------------------------------------
    // IdentifyInternal — absorb component nodes into a hierarchical
    // structured block (structural).
    // ---------------------------------------------------------------
    size_t CGraph::IdentifyInternal(const std::vector<size_t> &ids,
                                        CNode::BlockType type, SNode *snode) {
        if (ids.empty()) return CNode::kNone;
        size_t rep = ids[0];
        nodes[rep].structured = snode;
        nodes[rep].block_type = type;

        std::unordered_set<size_t> idset(ids.begin(), ids.end());

        // Collect external predecessors
        std::vector<size_t> ext_preds;
        for (size_t nid : ids) {
            for (size_t p : nodes[nid].preds) {
                if (idset.count(p) == 0) ext_preds.push_back(p);
            }
        }

        // Collect external successors (first-seen flags win)
        std::vector<size_t> ext_succs;
        std::vector<uint32_t> ext_succ_flags;
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

        // Mark non-rep nodes as collapsed, record as children
        nodes[rep].children.clear();
        for (size_t nid : ids) {
            if (nid != rep) {
                nodes[nid].collapsed_into = rep;
                nodes[rep].children.push_back(nid);
            }
        }

        // Rewire edges
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

        // Deduplicate predecessor succs
        for (size_t p : ext_preds) {
            auto &ss = nodes[p].succs;
            auto &sf = nodes[p].edge_flags;
            std::unordered_set<size_t> seen;
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

        // ---------------------------------------------------------------
        // Deduplicate convergent ext_succs.
        //
        // When two collapsed nodes independently exit to different
        // external blocks that lie on the same sequential path (e.g.,
        // switch-default→N17 and loop-exit→N14→...→N17), both appear
        // as ext_succs.  The representative isn't a true conditional —
        // the two exits are sequential, not alternatives.
        //
        // Detection: for each pair (A, B) in ext_succs, walk from A
        // through single-in/single-out blocks.  If B is reachable, A→B
        // is sequential — drop B from ext_succs (A will eventually
        // reach it via the graph).
        // ---------------------------------------------------------------
        if (ext_succs.size() == 2) {
            // Check if `target` is reachable from `from` through a
            // bounded forward walk.  Follows single-successor chains
            // and also handles conditionals where ALL branches converge
            // to the target (if-then-else diamond → same exit).
            auto reaches = [&](size_t from, size_t target, size_t limit = 20) -> bool {
                size_t cur = from;
                for (size_t step = 0; step < limit; ++step) {
                    auto &cn = nodes[cur];
                    if (cn.IsCollapsed()) break;
                    for (size_t s : cn.succs) {
                        if (s == target) return true;
                    }
                    if (cn.succs.size() == 1) {
                        cur = cn.succs[0];
                        if (cur == from) break;
                        continue;
                    }
                    // Conditional: check if ALL successors reach the
                    // target within a short walk (diamond pattern).
                    if (cn.succs.size() == 2) {
                        bool all_reach = true;
                        for (size_t s : cn.succs) {
                            bool found = (s == target);
                            if (!found) {
                                // One-hop check from each branch
                                auto &sn = nodes[s];
                                if (!sn.IsCollapsed()) {
                                    for (size_t ss : sn.succs) {
                                        if (ss == target) { found = true; break; }
                                    }
                                }
                            }
                            if (!found) { all_reach = false; break; }
                        }
                        if (all_reach) return true;
                    }
                    break;
                }
                return false;
            };

            if (reaches(ext_succs[0], ext_succs[1])) {
                // A reaches B — keep only A
                auto &bp = nodes[ext_succs[1]].preds;
                bp.erase(std::remove(bp.begin(), bp.end(), rep), bp.end());
                for (size_t nid : ids) {
                    bp.erase(std::remove(bp.begin(), bp.end(), nid), bp.end());
                }
                ext_succs.erase(ext_succs.begin() + 1);
                ext_succ_flags.erase(ext_succ_flags.begin() + 1);
            } else if (reaches(ext_succs[1], ext_succs[0])) {
                // B reaches A — keep only B
                auto &ap = nodes[ext_succs[0]].preds;
                ap.erase(std::remove(ap.begin(), ap.end(), rep), ap.end());
                for (size_t nid : ids) {
                    ap.erase(std::remove(ap.begin(), ap.end(), nid), ap.end());
                }
                ext_succs.erase(ext_succs.begin());
                ext_succ_flags.erase(ext_succ_flags.begin());
            } else {
                // Check if both reach a common descendant — if so,
                // they're fan-out paths to a shared merge point.
                // Find common target by checking 1-hop successors.
                auto first_succ = [&](size_t nid) -> size_t {
                    auto &n = nodes[nid];
                    if (n.IsCollapsed() || n.succs.empty()) return CNode::kNone;
                    if (n.succs.size() == 1) return n.succs[0];
                    // For conditionals, check if both branches go to same target
                    if (n.succs.size() == 2) {
                        auto &s0 = nodes[n.succs[0]];
                        auto &s1 = nodes[n.succs[1]];
                        if (!s0.IsCollapsed() && s0.succs.size() == 1
                            && !s1.IsCollapsed() && s1.succs.size() == 1
                            && s0.succs[0] == s1.succs[0]) {
                            return s0.succs[0];
                        }
                    }
                    return CNode::kNone;
                };

                size_t dest0 = first_succ(ext_succs[0]);
                size_t dest1 = first_succ(ext_succs[1]);
                if (dest0 != CNode::kNone && dest0 == dest1) {
                    // Both converge — keep only the first (it will
                    // reach the common merge sequentially).
                    auto &bp = nodes[ext_succs[1]].preds;
                    bp.erase(std::remove(bp.begin(), bp.end(), rep), bp.end());
                    for (size_t nid : ids) {
                        bp.erase(std::remove(bp.begin(), bp.end(), nid), bp.end());
                    }
                    ext_succs.erase(ext_succs.begin() + 1);
                    ext_succ_flags.erase(ext_succ_flags.begin() + 1);
                }
            }
        }

        // Install edges on representative
        nodes[rep].succs = ext_succs;
        nodes[rep].edge_flags = ext_succ_flags;

        std::sort(ext_preds.begin(), ext_preds.end());
        ext_preds.erase(std::unique(ext_preds.begin(), ext_preds.end()), ext_preds.end());
        nodes[rep].preds = ext_preds;

        // Ensure rep is listed in each successor's preds
        for (size_t s : ext_succs) {
            auto &p = nodes[s].preds;
            if (std::find(p.begin(), p.end(), rep) == p.end()) {
                p.push_back(rep);
            }
        }

        nodes[rep].is_conditional = !ext_succs.empty() && ext_succs.size() == 2;

        nodes[rep].stmts.clear();
        nodes[rep].label.clear();

        // Preserve branch_cond from the collapsed node that owns the conditional split
        nodes[rep].branch_cond = nullptr;
        if (nodes[rep].is_conditional) {
            for (auto it = ids.rbegin(); it != ids.rend(); ++it) {
                if (nodes[*it].branch_cond) {
                    nodes[rep].branch_cond = nodes[*it].branch_cond;
                    break;
                }
            }
        }

        // Preserve switch metadata from collapsed nodes
        nodes[rep].switch_cases.clear();
        for (size_t nid : ids) {
            if (nid != rep && !nodes[nid].switch_cases.empty()) {
                nodes[rep].switch_cases = std::move(nodes[nid].switch_cases);
                if (!nodes[rep].branch_cond) {
                    nodes[rep].branch_cond = nodes[nid].branch_cond;
                }
                break;
            }
        }

        return rep;
    }



    // Detect back-edges using iterative DFS
    void MarkBackEdges(CGraph &g) {
        // Clear previous back-edge marks so stale flags from earlier
        // calls don't accumulate after topology changes.
        for (auto &n : g.nodes) {
            for (auto &f : n.edge_flags)
                f &= ~CNode::kBack;
        }

        enum Color { WHITE, GRAY, BLACK };
        std::vector<Color> color(g.nodes.size(), WHITE);

        struct Frame { size_t u; size_t i; };
        std::vector<Frame> stack;
        stack.push_back({g.entry, 0});
        color[g.entry] = GRAY;

        while (!stack.empty()) {
            // Note: u and i are references into stack.back() and must not be
            // read after push_back (which may reallocate the stack vector).
            auto &[u, i] = stack.back();
            auto &nd = g.Node(u);
            if (i < nd.succs.size()) {
                size_t v = nd.succs[i];
                ++i;
                // Skip collapsed nodes — not part of the active graph.
                if (g.Node(v).IsCollapsed()) continue;
                if (color[v] == GRAY) {
                    nd.edge_flags[i - 1] |= CNode::kBack;
                } else if (color[v] == WHITE) {
                    color[v] = GRAY;
                    stack.push_back({v, 0});
                }
            } else {
                color[u] = BLACK;
                stack.pop_back();
            }
        }
    }

    // -------------------------------------------------------------------
    // LoopBody core methods
    // -------------------------------------------------------------------

    void ClearMarks(CGraph &g, const std::vector<size_t> &body) {
        for (size_t id : body) {
            g.Node(id).mark = false;
        }
    }

    void LoopBody::FindBase(CGraph &g, std::vector<size_t> &body) const {
        body.clear();

        g.Node(head).mark = true;
        body.push_back(head);

        for (size_t t : tails) {
            if (!g.Node(t).mark) {
                g.Node(t).mark = true;
                body.push_back(t);
            }
        }

        // BFS backward from tails
        for (size_t idx = 1; idx < body.size(); ++idx) {
            auto &nd = g.Node(body[idx]);
            for (size_t p : nd.preds) {
                if (g.Node(p).mark) continue;
                if (g.Node(p).IsCollapsed()) continue;

                bool is_goto = false;
                auto &pn = g.Node(p);
                for (size_t si = 0; si < pn.succs.size(); ++si) {
                    if (pn.succs[si] == body[idx]) {
                        if (pn.IsGotoOut(si)) is_goto = true;
                        break;
                    }
                }
                if (is_goto) continue;

                g.Node(p).mark = true;
                body.push_back(p);
            }
        }
    }

    void LoopBody::LabelContainments(
        const CGraph &g, const std::vector<size_t> & /*body*/,
        const std::vector<LoopBody *> &looporder
    ) {
        for (LoopBody *lb : looporder) {
            if (lb == this) continue;
            if (!g.Node(lb->head).mark) continue;

            if (lb->immed_container == nullptr) {
                lb->immed_container = this;
            } else if (!g.Node(lb->immed_container->head).mark) {
                lb->immed_container = this;
            }
        }
    }

    void LoopBody::MergeIdenticalHeads(
        std::vector<LoopBody *> &looporder, std::list<LoopBody> &storage
    ) {
        std::sort(looporder.begin(), looporder.end(),
            [](const LoopBody *a, const LoopBody *b) { return a->head < b->head; });

        size_t write = 0;
        for (size_t read = 0; read < looporder.size(); ++read) {
            if (write > 0 && looporder[write - 1]->head == looporder[read]->head) {
                auto *dst = looporder[write - 1];
                auto *src = looporder[read];
                for (size_t t : src->tails) {
                    dst->AddTail(t);
                }
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
        CGraph &g, std::list<LoopBody> &loopbody, std::vector<LoopBody *> &looporder
    ) {
        for (auto &n : g.nodes) {
            if (n.IsCollapsed()) continue;
            for (size_t i = 0; i < n.succs.size(); ++i) {
                if (n.IsBackEdge(i)) {
                    size_t hd = n.succs[i];
                    loopbody.emplace_back(hd);
                    loopbody.back().AddTail(n.id);
                    looporder.push_back(&loopbody.back());
                }
            }
        }

        LoopBody::MergeIdenticalHeads(looporder, loopbody);

        for (LoopBody *lb : looporder) {
            std::vector<size_t> body;
            lb->FindBase(g, body);
            lb->unique_count = static_cast<int>(body.size());
            lb->LabelContainments(g, body, looporder);
            ClearMarks(g, body);
        }

        for (LoopBody *lb : looporder) {
            int d = 0;
            for (LoopBody *c = lb->immed_container; c != nullptr; c = c->immed_container) {
                ++d;
            }
            lb->depth = d;
        }

        std::sort(looporder.begin(), looporder.end(),
            [](const LoopBody *a, const LoopBody *b) { return *a < *b; });
    }

    // -------------------------------------------------------------------
    // LoopBody exit detection, tail ordering, extension, exit labeling
    // -------------------------------------------------------------------

    void LoopBody::FindExit(CGraph &g, const std::vector<size_t> &body) {
        std::vector<size_t> candidates;

        // Phase 1: scan tails for exits
        for (size_t t : tails) {
            const auto &tn = g.Node(t);
            for (size_t i = 0; i < tn.succs.size(); ++i) {
                size_t s = tn.succs[i];
                if (!g.Node(s).mark && !tn.IsGotoOut(i) && !tn.IsBackEdge(i)) {
                    if (!immed_container) {
                        exit_block = s;
                        return;
                    }
                    candidates.push_back(s);
                }
            }
        }

        // Phase 2: scan head and middle body nodes
        {
            const auto &hd = g.Node(body[0]);
            for (size_t i = 0; i < hd.succs.size(); ++i) {
                size_t s = hd.succs[i];
                if (!g.Node(s).mark && !hd.IsGotoOut(i) && !hd.IsBackEdge(i)) {
                    if (!immed_container) {
                        exit_block = s;
                        return;
                    }
                    candidates.push_back(s);
                }
            }
            for (size_t idx = static_cast<size_t>(unique_count); idx < body.size(); ++idx) {
                const auto &bn = g.Node(body[idx]);
                for (size_t i = 0; i < bn.succs.size(); ++i) {
                    size_t s = bn.succs[i];
                    if (!g.Node(s).mark && !bn.IsGotoOut(i) && !bn.IsBackEdge(i)) {
                        if (!immed_container) {
                            exit_block = s;
                            return;
                        }
                        candidates.push_back(s);
                    }
                }
            }
        }

        if (candidates.empty()) {
            exit_block = kNone;
            return;
        }

        if (!immed_container) {
            exit_block = candidates[0];
            return;
        }

        // Phase 3: Container filtering (structural)
        std::vector<size_t> container_body;
        {
            g.Node(immed_container->head).visit_count = 1;
            container_body.push_back(immed_container->head);
            for (size_t t : immed_container->tails) {
                if (g.Node(t).visit_count == 0) {
                    g.Node(t).visit_count = 1;
                    container_body.push_back(t);
                }
            }
            for (size_t idx = 1; idx < container_body.size(); ++idx) {
                auto &nd = g.Node(container_body[idx]);
                for (size_t p : nd.preds) {
                    if (g.Node(p).visit_count != 0) continue;
                    if (g.Node(p).IsCollapsed()) continue;
                    bool is_goto = false;
                    auto &pn = g.Node(p);
                    for (size_t si = 0; si < pn.succs.size(); ++si) {
                        if (pn.succs[si] == container_body[idx]) {
                            if (pn.IsGotoOut(si)) is_goto = true;
                            break;
                        }
                    }
                    if (is_goto) continue;
                    g.Node(p).visit_count = 1;
                    container_body.push_back(p);
                }
            }
        }

        exit_block = kNone;
        for (size_t c : candidates) {
            if (g.Node(c).visit_count != 0) {
                exit_block = c;
                break;
            }
        }

        if (exit_block == kNone && !candidates.empty()) {
            exit_block = candidates[0];
        }

        for (size_t nid : container_body) {
            g.Node(nid).visit_count = 0;
        }
    }

    void LoopBody::OrderTails(const CGraph &g) {
        if (tails.size() <= 1 || exit_block == kNone) return;

        for (size_t ti = 0; ti < tails.size(); ++ti) {
            const auto &tn = g.Node(tails[ti]);
            for (size_t s : tn.succs) {
                if (s == exit_block) {
                    if (ti != 0) std::swap(tails[0], tails[ti]);
                    return;
                }
            }
        }
    }

    void LoopBody::Extend(CGraph &g, std::vector<size_t> &body) const {
        std::vector<size_t> trial;

        size_t idx = 0;
        while (idx < body.size()) {
            auto &bn = g.Node(body[idx]);
            ++idx;
            for (size_t i = 0; i < bn.succs.size(); ++i) {
                if (bn.IsGotoOut(i)) continue;
                size_t s = bn.succs[i];
                auto &sn = g.Node(s);
                if (sn.mark) continue;
                if (sn.IsCollapsed()) continue;
                if (s == exit_block) continue;

                if (sn.visit_count == 0) trial.push_back(s);
                sn.visit_count++;

                if (sn.visit_count == static_cast<int>(sn.SizeIn())) {
                    sn.mark = true;
                    body.push_back(s);
                }
            }
        }

        for (size_t s : trial) {
            g.Node(s).visit_count = 0;
        }
    }

    void LoopBody::LabelExitEdges(CGraph &g, const std::vector<size_t> &body) const {
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
    // OrderLoopBodies: orchestrate full loop detection pipeline
    // -------------------------------------------------------------------

    void OrderLoopBodies(CGraph &g, std::list<LoopBody> &loopbody) {
        std::vector<LoopBody *> looporder;
        LabelLoops(g, loopbody, looporder);
        if (loopbody.empty()) return;

        for (LoopBody *lb : looporder) {
            std::vector<size_t> body;
            lb->FindBase(g, body);
            lb->FindExit(g, body);
            lb->OrderTails(g);
            lb->Extend(g, body);
            lb->LabelExitEdges(g, body);
            ClearMarks(g, body);
        }
    }

    // -------------------------------------------------------------------
    // LoopBody: exit mark management and update
    // -------------------------------------------------------------------

    void LoopBody::SetExitMarks(CGraph &g, const std::vector<size_t> &body) const {
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

    void LoopBody::ClearExitMarks(CGraph &g, const std::vector<size_t> &body) const {
        for (size_t nid : body) {
            auto &n = g.Node(nid);
            for (size_t i = 0; i < n.succs.size(); ++i) {
                n.ClearLoopExit(i);
            }
        }
    }

    bool LoopBody::Update(const CGraph &g) const { return !g.Node(head).IsCollapsed(); }

    // -------------------------------------------------------------------
    // FloatingEdge
    // -------------------------------------------------------------------

    std::pair<size_t, size_t> FloatingEdge::GetCurrentEdge(const CGraph &g) const {
        size_t top = top_id;
        while (top < g.nodes.size() && g.Node(top).IsCollapsed()) {
            size_t next = g.Node(top).collapsed_into;
            if (next == CNode::kNone || next == top) break;
            top = next;
        }
        size_t bot = bottom_id;
        while (bot < g.nodes.size() && g.Node(bot).IsCollapsed()) {
            size_t next = g.Node(bot).collapsed_into;
            if (next == CNode::kNone || next == bot) break;
            bot = next;
        }

        if (top >= g.nodes.size() || bot >= g.nodes.size())
            return {CNode::kNone, 0};
        if (g.Node(top).IsCollapsed() || g.Node(bot).IsCollapsed())
            return {CNode::kNone, 0};
        if (top == bot)
            return {CNode::kNone, 0};

        const auto &succs = g.Node(top).succs;
        for (size_t i = 0; i < succs.size(); ++i) {
            if (succs[i] == bot) return {top, i};
        }
        return {CNode::kNone, 0};
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
        for (auto *cur = this; cur != nullptr; cur = cur->parent)
            cur->ismark = false;
        for (auto *cur = op2; cur != nullptr; cur = cur->parent)
            cur->ismark = false;

        MarkPath();

        int dist = 0;
        auto *cur = op2;
        while (cur != nullptr && !cur->ismark) {
            ++dist;
            cur = cur->parent;
        }
        if (cur == nullptr) return dist;

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
            return true;
        }

        auto &n = g.Node(dest);
        if (n.IsCollapsed()) {
            trace->flags |= BlockTrace::kTerminal;
            return true;
        }

        if (dest == finishblock_id_) {
            trace->flags |= BlockTrace::kTerminal;
            return true;
        }

        size_t dag_out_count = 0;
        size_t single_succ = CNode::kNone;
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
            trace->bottom_id = dest;
            trace->dest_id = single_succ;
            if (single_succ < g.nodes.size() && !g.Node(single_succ).IsCollapsed()) {
                g.Node(single_succ).visit_count += 1;
            }
            return true;
        }

        return false;
    }

    std::list<TraceDAG::BlockTrace *>::iterator
    TraceDAG::OpenBranch(CGraph &g, BlockTrace *parent) {
        size_t branch_id = parent->dest_id;
        const auto &n = g.Node(branch_id);

        auto *bp = new BranchPoint();
        bp->parent = parent->top;
        bp->pathout = parent->pathout;
        bp->top_id = branch_id;
        bp->depth = parent->top->depth + 1;
        branchlist_.push_back(bp);
        parent->derivedbp = bp;

        auto next_iter = std::next(parent->activeiter);
        RemoveActive(parent);

        int pathindex = 0;
        for (size_t i = 0; i < n.succs.size(); ++i) {
            if (!n.IsLoopDagOut(i)) continue;

            size_t succ_id = n.succs[i];
            auto &succ_node = g.Node(succ_id);

            auto *bt = new BlockTrace();
            bt->top = bp;
            bt->pathout = pathindex++;
            bt->bottom_id = branch_id;
            bt->dest_id = succ_id;
            bp->paths.push_back(bt);

            if (!succ_node.IsCollapsed() && succ_node.visit_count > 0) {
                for (auto *existing : activetrace_) {
                    if (existing->dest_id == succ_id || existing->bottom_id == succ_id) {
                        existing->edgelump += 1;
                        bt->flags |= BlockTrace::kTerminal;
                        break;
                    }
                }
                if (!bt->IsTerminal()) {
                    InsertActive(bt);
                }
            } else {
                InsertActive(bt);
            }

            if (!succ_node.IsCollapsed()) {
                succ_node.visit_count += 1;
            }
        }

        return next_iter;
    }

    bool TraceDAG::CheckRetirement(BlockTrace *trace, size_t &exitblock_id) {
        if (trace->pathout != 0) return false;
        BranchPoint *bp = trace->top;
        if (bp == nullptr) return false;

        if (bp->depth == 0) {
            for (auto *bt : bp->paths) {
                if (!bt->IsActive()) return false;
                if (!bt->IsTerminal()) return false;
            }
            exitblock_id = CNode::kNone;
            return true;
        }

        size_t outblock = CNode::kNone;
        for (auto *bt : bp->paths) {
            if (!bt->IsActive()) return false;
            if (bt->IsTerminal()) continue;
            if (outblock == bt->dest_id) continue;
            if (outblock != CNode::kNone) return false;
            outblock = bt->dest_id;
        }
        exitblock_id = outblock;
        return true;
    }

    std::list<TraceDAG::BlockTrace *>::iterator
    TraceDAG::RetireBranch(BranchPoint *bp, size_t exitblock_id) {
        std::list<BlockTrace *>::iterator next_iter = current_activeiter_;
        for (auto *bt : bp->paths) {
            if (bt->IsActive()) {
                auto it = bt->activeiter;
                if (it == next_iter) ++next_iter;
                RemoveActive(bt);
            }
        }

        if (bp->parent != nullptr) {
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
        if (siblingedge != op2.siblingedge)
            return (op2.siblingedge < siblingedge);
        if (terminal != op2.terminal)
            return (terminal < op2.terminal);
        if (distance != op2.distance)
            return (distance < op2.distance);
        if (trace->top && op2.trace->top)
            return (trace->top->depth < op2.trace->top->depth);
        return false;
    }

    bool TraceDAG::BadEdgeScore::operator<(const BadEdgeScore &op2) const {
        return exitproto_id < op2.exitproto_id;
    }

    void TraceDAG::ProcessExitConflict(
        std::list<BadEdgeScore>::iterator start, std::list<BadEdgeScore>::iterator end
    ) {
        int count = 0;
        for (auto it = start; it != end; ++it) ++count;
        if (count <= 1) return;

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
        std::list<BadEdgeScore> scores;
        for (auto *bt : activetrace_) {
            if (bt->IsTerminal()) continue;
            BadEdgeScore score;
            score.exitproto_id = bt->dest_id;
            score.trace = bt;
            score.terminal = 0;
            scores.push_back(score);
        }

        if (scores.empty()) return nullptr;

        scores.sort();

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

    void TraceDAG::PushBranches(CGraph &g) {
        ClearVisitCount(g);

        for (size_t root_id : rootlist_) {
            if (root_id < g.nodes.size() && !g.Node(root_id).IsCollapsed()) {
                g.Node(root_id).visit_count = 1;
            }
        }

        // Single-loop structure matching Ghidra's pushBranches():
        // one while(activecount>0) with wrap-around, no nested loops.
        // Convergence is detected by missed_count >= activecount_ —
        // when all active traces are stuck, selectBadEdge removes the
        // worst trace (reducing activecount_) and retries.
        //
        // Safety bound: scale by edges (N*E*4) rather than nodes squared
        // to handle switch-heavy graphs where edge count dominates.
        // This should never trigger in correct operation but prevents
        // hangs from bugs in trace logic.
        size_t total_edges = 0;
        for (auto &n : g.nodes) {
            if (!n.IsCollapsed()) {
                total_edges += n.succs.size();
            }
        }
        // Saturate to avoid overflow on huge graphs.
        constexpr size_t kLimit = std::numeric_limits< size_t >::max() / 8;
        size_t n               = g.nodes.size() + 1;
        size_t e               = total_edges + 1;
        const size_t max_outer = (n <= kLimit / e) ? n * e * 4 + 512 : kLimit;
        size_t outer_iter      = 0;
        int missed_count        = 0;
        current_activeiter_    = activetrace_.begin();

        while (activecount_ > 0 && outer_iter < max_outer) {
            ++outer_iter;
            if (current_activeiter_ == activetrace_.end()) {
                if (activetrace_.empty()) {
                    break;
                }
                current_activeiter_ = activetrace_.begin();
            }

            BlockTrace *bt = *current_activeiter_;

            if (missed_count >= activecount_) {
                BlockTrace *bad = SelectBadEdge();
                if (bad == nullptr) {
                    ClearVisitCount(g);
                    return;
                }
                if (bad->bottom_id != CNode::kNone && bad->dest_id != CNode::kNone) {
                    likelygoto_.emplace_back(bad->bottom_id, bad->dest_id);
                }
                RemoveTrace(bad);
                missed_count         = 0;
                current_activeiter_ = activetrace_.begin();
                continue;
            }

            {
                size_t exit_id = CNode::kNone;
                if (CheckRetirement(bt, exit_id)) {
                    current_activeiter_ = RetireBranch(bt->top, exit_id);
                    missed_count         = 0;
                    continue;
                }
            }

            {
                bool was_terminal = bt->IsTerminal();
                if (CheckOpen(g, bt)) {
                    ++current_activeiter_;
                    if (was_terminal) {
                        ++missed_count;
                    } else {
                        missed_count = 0;
                    }
                    continue;
                }
            }

            {
                const auto &dest_node = g.Node(bt->dest_id);
                size_t dag_preds      = 0;
                for (size_t p : dest_node.preds) {
                    if (!g.Node(p).IsCollapsed()) {
                        ++dag_preds;
                    }
                }
                if (dag_preds > 1 && dest_node.visit_count < static_cast< int >(dag_preds))
                {
                    ++current_activeiter_;
                    ++missed_count;
                    continue;
                }
            }

            current_activeiter_ = OpenBranch(g, bt);
            missed_count         = 0;
        }

        if (outer_iter >= max_outer) {
            LOG(WARNING) << "PushBranches: safety limit reached (" << max_outer
                         << " iterations on " << g.nodes.size() << " nodes, " << total_edges
                         << " edges)\n";
        }

        ClearVisitCount(g);
    }

    void TraceDAG::ClearVisitCount(CGraph &g) {
        for (auto &n : g.nodes) {
            n.visit_count = 0;
        }
    }

} // namespace patchestry::ast
