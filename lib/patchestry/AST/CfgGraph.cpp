/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "CfgFoldInternal.hpp"

namespace patchestry::ast {

    namespace detail {

        // ---------------------------------------------------------------
        // IdentifyInternal — absorb component nodes into a hierarchical
        // structured block (Ghidra-style).  Children's stmts/labels
        // remain accessible; only external edges are rewired.
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
        size_t CGraph::IdentifyInternal(const std::vector< size_t > &ids,
                                         CNode::BlockType type, SNode *snode) {
            // --- Step 0: choose representative ---
            size_t rep = ids[0];
            nodes[rep].structured = snode;
            nodes[rep].block_type = type;

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

            // --- Step 3: mark non-rep nodes as collapsed, record as children ---
            nodes[rep].children.clear();
            for (size_t nid : ids) {
                if (nid != rep) {
                    nodes[nid].collapsed = true;
                    nodes[nid].collapsed_into = rep;
                    nodes[rep].children.push_back(nid);
                    // Children's stmts/labels/original_label are NOT cleared —
                    // they remain accessible through the hierarchy for goto
                    // resolution and debugging.
                }
            }

            // --- Step 4: rewire edges ---
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
            nodes[rep].branch_cond = nullptr;
            if (nodes[rep].is_conditional) {
                for (auto it = ids.rbegin(); it != ids.rend(); ++it) {
                    if (nodes[*it].branch_cond) {
                        nodes[rep].branch_cond = nodes[*it].branch_cond;
                        break;
                    }
                }
            }

            // Preserve switch metadata from collapsed nodes.
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

        // Build the collapse graph from the Cfg
        CGraph BuildCGraph(const Cfg &cfg) {
            CGraph g;
            g.entry = cfg.entry;
            g.nodes.resize(cfg.blocks.size());

            for (size_t i = 0; i < cfg.blocks.size(); ++i) {
                auto &cb = cfg.blocks[i];
                auto &cn = g.nodes[i];
                cn.id = i;
                cn.label = cb.label;
                cn.original_label = cb.label;  // immutable copy for goto resolution
                cn.stmts = cb.stmts;
                cn.branch_cond = cb.branch_cond;
                cn.is_conditional = cb.is_conditional;
                cn.terminal = cb.terminal;
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
            // Select exit block (Ghidra-style priority search).
            // Priority: tail exits first, then head, then body.
            // No container: take first candidate immediately.
            // Has container: collect all candidates, filter to those
            // reachable within the container's body.

            std::vector<size_t> candidates;

            // Phase 1: scan tails for exits.
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

            // Phase 2: scan head and middle body nodes.
            // Head is body[0]. Middle nodes start at unique_count.
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
                // Should not reach here (no-container paths return above)
                exit_block = candidates[0];
                return;
            }

            // Phase 3: Container filtering (Ghidra-style).
            // Compute the container's body using visit_count as temporary
            // membership flag (separate from mark, which is in use by this loop).
            // A candidate is valid if it's inside the container's body.
            //
            // Compute container body via forward BFS from container head,
            // bounded by container's back-edge targets.
            std::vector<size_t> container_body;
            {
                // BFS backward from container tails to container head
                // (same as FindBase but using visit_count instead of mark).
                auto &cg = const_cast<CGraph &>(g);
                cg.Node(immed_container->head).visit_count = 1;
                container_body.push_back(immed_container->head);
                for (size_t t : immed_container->tails) {
                    if (cg.Node(t).visit_count == 0) {
                        cg.Node(t).visit_count = 1;
                        container_body.push_back(t);
                    }
                }
                for (size_t idx = 1; idx < container_body.size(); ++idx) {
                    auto &nd = cg.Node(container_body[idx]);
                    for (size_t p : nd.preds) {
                        if (cg.Node(p).visit_count != 0) continue;
                        if (cg.Node(p).collapsed) continue;
                        bool is_goto = false;
                        auto &pn = cg.Node(p);
                        for (size_t si = 0; si < pn.succs.size(); ++si) {
                            if (pn.succs[si] == container_body[idx]) {
                                if (pn.IsGotoOut(si)) is_goto = true;
                                break;
                            }
                        }
                        if (is_goto) continue;
                        cg.Node(p).visit_count = 1;
                        container_body.push_back(p);
                    }
                }
            }

            // Select first candidate that is inside container body.
            exit_block = kNone;
            for (size_t c : candidates) {
                if (g.Node(c).visit_count != 0) {
                    exit_block = c;
                    break;
                }
            }

            // If no candidate inside container, fall back to first candidate.
            if (exit_block == kNone && !candidates.empty()) {
                exit_block = candidates[0];
            }

            // Clean up visit_count.
            for (size_t nid : container_body) {
                const_cast<CGraph &>(g).Node(nid).visit_count = 0;
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
            // Single-pass extension (Ghidra-style): iterate the body vector
            // which grows as new nodes are absorbed.  For each body node's
            // successors, increment visit_count.  When visit_count equals
            // SizeIn, all predecessors are in the body — absorb the node.
            //
            // This is O(edges) vs the previous O(nodes²) fixpoint approach.

            std::vector<size_t> trial; // nodes with non-zero visit_count (for cleanup)

            size_t idx = 0;
            while (idx < body.size()) {
                auto &bn = g.Node(body[idx]);
                ++idx;
                for (size_t i = 0; i < bn.succs.size(); ++i) {
                    if (bn.IsGotoOut(i)) continue;        // don't extend through gotos
                    size_t s = bn.succs[i];
                    auto &sn = g.Node(s);
                    if (sn.mark) continue;                 // already in body
                    if (sn.collapsed) continue;
                    if (s == exit_block) continue;          // don't extend into exit

                    if (sn.visit_count == 0)
                        trial.push_back(s);                 // track for cleanup
                    sn.visit_count++;

                    if (sn.visit_count == static_cast<int>(sn.SizeIn())) {
                        // All predecessors are in body — absorb
                        sn.mark = true;
                        body.push_back(s);
                    }
                }
            }

            // Clean up visit_count
            for (size_t s : trial) {
                g.Node(s).visit_count = 0;
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
            // 1. labelLoops: create LoopBody per back-edge, merge, compute
            //    containment/depth, sort innermost-first.
            std::vector<LoopBody *> looporder;
            LabelLoops(g, loopbody, looporder);
            if (loopbody.empty()) return;

            // 2. Process loops in innermost-first order (Ghidra-style).
            //    This ensures inner loop exits are identified before the
            //    outer loop's Extend absorbs surrounding blocks.
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
            const size_t max_outer = g.nodes.size() * g.nodes.size() * 8 + 512;
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

} // namespace patchestry::ast
