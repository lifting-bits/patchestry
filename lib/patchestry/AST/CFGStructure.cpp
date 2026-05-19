/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/CFGStructure.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Util/Log.hpp>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <unordered_set>

namespace patchestry::ast {

    // Count how many SGoto nodes and clang::GotoStmt nodes reference
    // each label name anywhere in the SNode tree.  Shared by
    // InlineResidualGotos, AbsorbFallthroughIntoElse, and ScopeifyIfGotos.
    static void CountGotoRefs(SNode *node,
                              std::unordered_map<std::string_view, int> &refs) {
        if (!node) return;

        // Leaf cases for_each_child can't express: SGoto's target label
        // and clang::GotoStmt embedded in an SStmt's raw Stmt.
        if (auto *g = node->dyn_cast<SGoto>()) {
            refs[g->Target()]++;
            return;
        }
        if (auto *st = node->dyn_cast<SStmt>()) {
            std::function< void(clang::Stmt *) > walk =
                [&](clang::Stmt *s) {
                if (!s) return;
                if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
                    refs[gs->getLabel()->getName()]++;
                    return;
                }
                for (auto *child : s->children()) walk(child);
            };
            walk(st->Stmt());
            return;
        }

        // All other SNode kinds: recurse uniformly via the visitor API.
        node->for_each_child([&](SNode *c) { CountGotoRefs(c, refs); });
    }

    // Count goto refs across a sequence (vector) of SNodes.
    static void CountGotoRefs(const std::vector< SNode * > &seq,
                              std::unordered_map<std::string_view, int> &refs) {
        for (auto *c : seq) CountGotoRefs(c, refs);
    }

    // Append every element of `src` onto `dst`.
    static void SeqAppend(std::vector< SNode * > &dst,
                          const std::vector< SNode * > &src) {
        dst.insert(dst.end(), src.begin(), src.end());
    }

    // Spill raw clang::Stmt* into individual SStmt SNodes appended to
    // `out`.  Null statements are dropped.
    static void AppendStmts(SNodeFactory &factory,
                            std::vector< SNode * > &out,
                            const std::vector< clang::Stmt * > &stmts) {
        for (auto *s : stmts)
            if (s) out.push_back(factory.Make< SStmt >(s));
    }

    // Invoke fn(std::vector<SNode*>&) on each body-vector slot held by
    // `node` (SLabel/loop bodies, if-then/else arms, switch case lists).
    // Leaf kinds carry no body-vectors.
    template< typename Fn >
    static void ForEachBodyList(SNode *node, Fn &&fn) {
        if (!node) return;
        switch (node->Kind()) {
            case SNodeKind::kLabel:
                fn(node->as< SLabel >()->BodyList());
                break;
            case SNodeKind::kWhile:
                fn(node->as< SWhile >()->BodyList());
                break;
            case SNodeKind::kDoWhile:
                fn(node->as< SDoWhile >()->BodyList());
                break;
            case SNodeKind::kFor:
                fn(node->as< SFor >()->BodyList());
                break;
            case SNodeKind::kIfThenElse: {
                auto *ite = node->as< SIfThenElse >();
                fn(ite->ThenList());
                fn(ite->ElseList());
                break;
            }
            case SNodeKind::kSwitch: {
                auto *sw = node->as< SSwitch >();
                for (auto &c : sw->Cases()) fn(c.body_list);
                fn(sw->DefaultBodyList());
                break;
            }
            default:
                break;
        }
    }

    // Run `worker` on every body-vector in the tree rooted at the
    // sequence `seq`, post-order: deepest body-vectors first, then
    // `seq` itself.  `worker` returns true if it mutated the list.
    //
    // With `stop_after_change`, the traversal unwinds immediately once
    // `worker` reports a mutation — no suspended ancestor loop resumes
    // over a vector the worker may have mutated.  Required for workers
    // that mutate a non-local sequence (e.g. cross-scope inlining).
    static bool ForEachSeqPostOrder(
        std::vector< SNode * > &seq,
        const std::function< bool(std::vector< SNode * > &) > &worker,
        bool stop_after_change = false) {
        bool changed = false;
        for (SNode *child : seq) {
            bool stop = false;
            ForEachBodyList(child, [&](std::vector< SNode * > &body) {
                if (stop) return;
                if (ForEachSeqPostOrder(body, worker, stop_after_change)) {
                    changed = true;
                    if (stop_after_change) stop = true;
                }
            });
            if (stop) return true;
        }
        if (worker(seq)) changed = true;
        return changed;
    }

    CFGStructure::CFGStructure(CGraph &g, SNodeFactory &factory,
                                         clang::ASTContext &ctx)
        : graph_(g), factory_(factory), ctx_(ctx) {}

    // ---------------------------------------------------------------
    // MergeConditionalForwarders — pre-pass
    // ---------------------------------------------------------------
    //
    // Absorb conditional successor nodes into their unconditional
    // predecessors when the successor has a sole predecessor.
    //
    // Pattern:
    //   Block A: unconditional, exactly 1 succ → B, not a goto edge
    //   Block B: conditional, sole pred = A, 2 succs
    //
    // Result: A absorbs B's stmts, becomes conditional with B's
    // branch_cond, succs, edge_flags, and terminal.  B is collapsed.

    static void MergeConditionalForwarders(CGraph &g) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto &a : g.nodes) {
                if (a.IsCollapsed()) continue;
                if (a.is_conditional) continue;
                if (a.succs.size() != 1) continue;
                if (a.IsGotoOut(0)) continue;

                size_t b_id = a.succs[0];
                auto &b = g.Node(b_id);
                if (b.IsCollapsed()) continue;
                if (!b.is_conditional) continue;
                if (b.succs.size() != 2) continue;

                // B must have sole predecessor A
                bool sole_pred = true;
                for (size_t p : b.preds) {
                    if (p != a.id && !g.Node(p).IsCollapsed()) {
                        sole_pred = false;
                        break;
                    }
                }
                if (!sole_pred) continue;

                // Absorb B's stmts into A (B may or may not have stmts)
                for (auto *s : b.stmts) {
                    a.stmts.push_back(s);
                }
                b.stmts.clear();

                // Transfer B's conditional state to A
                a.branch_cond = b.branch_cond;
                a.is_conditional = true;
                a.terminal = b.terminal;

                // Replace A's single succ with B's two succs
                a.succs = b.succs;
                a.edge_flags = b.edge_flags;

                // Update B's successors: replace B with A in their pred lists
                for (size_t s : b.succs) {
                    auto &sp = g.Node(s).preds;
                    for (auto &p : sp) {
                        if (p == b_id) { p = a.id; break; }
                    }
                }

                // Collapse B into A
                b.collapsed_into = a.id;
                a.children.push_back(b_id);
                b.succs.clear();
                b.preds.clear();
                b.edge_flags.clear();

                changed = true;
                break;  // restart scan — topology changed
            }
        }
    }

    // After a rule emits an explicit `if(cond) goto X` SNode for one
    // arm of a conditional, the graph edge rep→X is redundant — the
    // jump is now encoded in the SNode, not the topology.  Drop it so
    // downstream rules treat the rep as a plain fallthrough node
    // instead of re-structuring it as a two-way conditional (which
    // duplicates the guard and can strand the fallthrough block).
    static void ConsumeGotoEdge(CGraph &g, size_t rep, size_t target) {
        auto &n = g.Node(rep);
        bool present = false;
        for (size_t s : n.succs) {
            if (s == target) { present = true; break; }
        }
        if (!present) return;
        g.RemoveEdge(rep, target);
        n.is_conditional = n.succs.size() >= 2;
    }

    // ---------------------------------------------------------------
    // ---------------------------------------------------------------
    // ComputeDominatorTree
    // ---------------------------------------------------------------
    //
    // Cooper-Harvey-Kennedy iterative dominator algorithm.
    //
    // RPO positions: CGraph nodes are allocated in RPO order by
    // BuildCGraph (compute_rpo_from_function assigns index i to
    // rpo[i]).  MergeConditionalForwarders preserves this ordering
    // — it only collapses B into a preceding A.  We use node.id
    // directly as RPO position for active nodes and mark collapsed
    // nodes with kNone so the intersect function skips them.

    void CFGStructure::ComputeDominatorTree() {
        const size_t n = graph_.nodes.size();
        constexpr size_t kNone = CNode::kNone;
        idom_.assign(n, kNone);

        // Initialize RPO positions from node indices (already RPO-ordered).
        rpo_pos_.assign(n, kNone);
        for (size_t i = 0; i < n; ++i) {
            if (!graph_.nodes[i].IsCollapsed())
                rpo_pos_[i] = i;
        }

        size_t entry = graph_.entry;
        if (entry >= n || graph_.Node(entry).IsCollapsed()) return;

        idom_[entry] = entry;

        // Build RPO-ordered list of active node ids (excluding entry).
        std::vector<size_t> rpo_nodes;
        for (auto &node : graph_.nodes) {
            if (node.IsCollapsed()) continue;
            if (node.id == entry) continue;
            if (rpo_pos_[node.id] == kNone) continue;
            rpo_nodes.push_back(node.id);
        }
        std::sort(rpo_nodes.begin(), rpo_nodes.end(),
                  [this](size_t a, size_t b) {
                      return rpo_pos_[a] < rpo_pos_[b];
                  });

        // Intersect: walk up the dominator tree using RPO positions.
        auto intersect = [this](size_t a, size_t b) -> size_t {
            while (a != b) {
                while (rpo_pos_[a] > rpo_pos_[b]) {
                    size_t up = idom_[a];
                    if (up == kNone || up == a) return a;
                    a = up;
                }
                while (rpo_pos_[b] > rpo_pos_[a]) {
                    size_t up = idom_[b];
                    if (up == kNone || up == b) return b;
                    b = up;
                }
            }
            return a;
        };

        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t nid : rpo_nodes) {
                auto &node = graph_.Node(nid);

                // Find first predecessor with a defined idom
                size_t new_idom = kNone;
                for (size_t p : node.preds) {
                    if (graph_.Node(p).IsCollapsed()) continue;
                    if (idom_[p] == kNone) continue;
                    new_idom = p;
                    break;
                }
                if (new_idom == kNone) continue;

                // Intersect with remaining defined predecessors
                for (size_t p : node.preds) {
                    if (p == new_idom) continue;
                    if (graph_.Node(p).IsCollapsed()) continue;
                    if (idom_[p] == kNone) continue;
                    new_idom = intersect(new_idom, p);
                }

                if (idom_[nid] != new_idom) {
                    idom_[nid] = new_idom;
                    changed = true;
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // ComputePostDominatorTree — pre-pass Phase 6
    // ---------------------------------------------------------------
    //
    // Same Cooper-Harvey-Kennedy algorithm as Phase 5, but on the
    // reversed CFG.  A virtual exit node is added: all terminal
    // nodes (no active successors) are treated as predecessors of
    // the virtual exit in the reversed graph.

    void CFGStructure::ComputePostDominatorTree() {
        const size_t n = graph_.nodes.size();
        constexpr size_t kNone = CNode::kNone;
        const size_t virt_exit = n;  // virtual node id

        ipdom_.assign(n, kNone);

        // Identify terminal nodes (active, no active successors).
        std::vector<size_t> terminals;
        for (auto &node : graph_.nodes) {
            if (node.IsCollapsed()) continue;
            bool has_active_succ = false;
            for (size_t s : node.succs) {
                if (!graph_.Node(s).IsCollapsed()) {
                    has_active_succ = true;
                    break;
                }
            }
            if (!has_active_succ) terminals.push_back(node.id);
        }
        if (terminals.empty()) return;

        // Compute reverse-RPO: DFS from virtual exit following
        // reversed edges (succs in original = preds in reversed).
        std::vector<size_t> rev_rpo_pos(n + 1, kNone);  // +1 for virt_exit
        {
            std::vector<size_t> post_order;
            std::vector<bool> visited(n + 1, false);

            struct Frame {
                size_t id;
                size_t child_idx;
            };
            std::vector<Frame> stack;

            // Start from virtual exit.  Its "successors" in the reversed
            // graph are the terminal nodes' reversed edges — i.e., the
            // terminal nodes themselves.
            visited[virt_exit] = true;
            stack.push_back({virt_exit, 0});

            while (!stack.empty()) {
                auto &top = stack.back();

                // Children: for virtual exit → terminals;
                //           for real nodes → preds in original graph.
                const auto *children_ptr = (top.id == virt_exit)
                    ? &terminals
                    : &graph_.Node(top.id).preds;
                auto &children = *children_ptr;

                if (top.child_idx < children.size()) {
                    size_t child = children[top.child_idx];
                    ++top.child_idx;
                    if (child < n && !visited[child] &&
                        !graph_.Node(child).IsCollapsed()) {
                        visited[child] = true;
                        stack.push_back({child, 0});
                    }
                } else {
                    post_order.push_back(top.id);
                    stack.pop_back();
                }
            }

            size_t pos = 0;
            for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
                rev_rpo_pos[*it] = pos++;
            }
        }

        // Build reverse-RPO ordered list of active nodes (excl virt_exit).
        std::vector<size_t> rev_rpo_nodes;
        for (auto &node : graph_.nodes) {
            if (node.IsCollapsed()) continue;
            if (rev_rpo_pos[node.id] == kNone) continue;
            rev_rpo_nodes.push_back(node.id);
        }
        std::sort(rev_rpo_nodes.begin(), rev_rpo_nodes.end(),
                  [&rev_rpo_pos](size_t a, size_t b) {
                      return rev_rpo_pos[a] < rev_rpo_pos[b];
                  });

        // Dominator algorithm on reversed graph.
        // "entry" = virt_exit.  "preds of n" = succs of n in original
        // graph + (if n is terminal) virt_exit.
        //
        // We store idom for real nodes only.  Use a separate slot for
        // virt_exit's idom: ipdom_virt = virt_exit (it dominates itself).
        std::vector<size_t> rev_idom(n + 1, kNone);
        rev_idom[virt_exit] = virt_exit;

        // For terminals, virt_exit is their sole "predecessor" in the
        // reversed graph — set their idom to virt_exit as initial seed.
        for (size_t t : terminals) {
            rev_idom[t] = virt_exit;
        }

        auto intersect = [&rev_rpo_pos, &rev_idom](size_t a, size_t b) -> size_t {
            while (a != b) {
                while (rev_rpo_pos[a] > rev_rpo_pos[b]) {
                    size_t up = rev_idom[a];
                    if (up == kNone || up == a) return a;
                    a = up;
                }
                while (rev_rpo_pos[b] > rev_rpo_pos[a]) {
                    size_t up = rev_idom[b];
                    if (up == kNone || up == b) return b;
                    b = up;
                }
            }
            return a;
        };

        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t nid : rev_rpo_nodes) {
                // "Predecessors" of nid in reversed graph = succs in original.
                auto &node = graph_.Node(nid);

                size_t new_idom = kNone;

                // Check if this is a terminal → virt_exit is a "pred"
                bool is_terminal = std::find(terminals.begin(), terminals.end(), nid)
                                   != terminals.end();
                if (is_terminal && rev_idom[virt_exit] != kNone) {
                    new_idom = virt_exit;
                }

                // Succs in original graph = preds in reversed graph
                for (size_t s : node.succs) {
                    if (graph_.Node(s).IsCollapsed()) continue;
                    if (rev_idom[s] == kNone) continue;
                    if (new_idom == kNone) {
                        new_idom = s;
                    } else {
                        new_idom = intersect(new_idom, s);
                    }
                }
                if (new_idom == kNone) continue;

                if (rev_idom[nid] != new_idom) {
                    rev_idom[nid] = new_idom;
                    changed = true;
                }
            }
        }

        // Copy results into ipdom_, mapping virt_exit to kNone.
        for (size_t i = 0; i < n; ++i) {
            ipdom_[i] = (rev_idom[i] == virt_exit) ? kNone : rev_idom[i];
        }
    }

    // ---------------------------------------------------------------
    // NormalizeConditionPolarityIPdom — pre-pass 5
    // ---------------------------------------------------------------
    //
    // Refine conditional polarity using the post-dominator tree.
    // If ipdom[A] == succs[1] (taken), the merge point is on the
    // wrong arm — swap succs and negate the condition so that
    // succs[0] (not-taken) = merge, succs[1] (taken) = body.

    void CFGStructure::NormalizeConditionPolarityIPdom() {
        constexpr size_t kNone = CNode::kNone;

        for (auto &a : graph_.nodes) {
            if (a.IsCollapsed()) continue;
            if (!a.is_conditional) continue;
            if (a.succs.size() != 2) continue;
            if (!a.branch_cond) continue;

            size_t ipd = ipdom_[a.id];
            if (ipd == kNone) continue;

            // Already normalized: merge on not-taken
            if (ipd == a.succs[0]) continue;

            // Merge on taken — swap to put it on not-taken
            if (ipd == a.succs[1]) {
                std::swap(a.succs[0], a.succs[1]);
                std::swap(a.edge_flags[0], a.edge_flags[1]);
                a.branch_cond = NegateExpr(ctx_, a.branch_cond);

                if (auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(a.terminal)) {
                    auto loc = ifs->getIfLoc();
                    a.terminal = clang::IfStmt::Create(
                        ctx_, loc, clang::IfStatementKind::Ordinary,
                        nullptr, nullptr,
                        a.branch_cond, a.branch_cond->getBeginLoc(),
                        ifs->getElse() ? ifs->getElse()->getBeginLoc() : loc,
                        ifs->getElse() ? ifs->getElse() : ifs->getThen(),
                        ifs->getThen() ? ifs->getThen()->getBeginLoc() : loc,
                        ifs->getThen());
                }
            }
            // Neither succ is ipdom — leave as-is (Phase 2 RPO heuristic)
        }
    }

    // ---------------------------------------------------------------
    // StructureAll — main driver
    // ---------------------------------------------------------------

    void CFGStructure::StructureAll() {
        if (graph_.nodes.empty()) return;

        // Pre-pass 1: merge conditional forwarders into predecessors.
        MergeConditionalForwarders(graph_);

        // Pre-pass 2: compute immediate dominators.
        // RPO positions are derived from node indices (already RPO-ordered
        // by BuildCGraph).  MergeConditionalForwarders preserves this
        // ordering, so no separate RPO recomputation is needed.
        ComputeDominatorTree();

        // Pre-pass 3: compute immediate post-dominators.
        ComputePostDominatorTree();

        // Pre-pass 4: identify back-edges and discover loops.
        // Runs after dominator computation so loop body membership
        // uses final topology.  MergeConditionalForwarders does not
        // read back-edge flags, so no earlier MarkBackEdges is needed.
        MarkBackEdges(graph_);
        OrderLoops();

        // Pre-pass 5: refine polarity using ipdom.
        // Runs AFTER loop detection so loop body membership is stable.
        // Loop rules handle both polarities dynamically (s1_in_body check).
        NormalizeConditionPolarityIPdom();

        // Hard termination bound: each successful rule reduces active
        // count, so 2 * initial count is generous.
        const size_t max_iterations = graph_.ActiveCount() * 2 + 1;
        size_t iterations = 0;
        size_t stall_count = 0;
        size_t last_active = graph_.ActiveCount();
        // Stop if SelectAndMarkGotoEdge fires repeatedly without
        // reducing the active node count — further goto edges won't
        // enable new rules and each call burns O(N×E) in PushBranches.
        constexpr size_t kMaxStall = 20;

        while (graph_.ActiveCount() > 1 && iterations < max_iterations) {
            if (StructureInternal()) {
                ++iterations;
                size_t cur = graph_.ActiveCount();
                if (cur < last_active) { last_active = cur; stall_count = 0; }
                continue;
            }
            // No structural rule fired — use TraceDAG to select the
            // least-disruptive edge and mark it as a goto, then retry.
            if (SelectAndMarkGotoEdge()) {
                ++iterations;
                size_t cur = graph_.ActiveCount();
                if (cur < last_active) {
                    last_active = cur;
                    stall_count = 0;
                } else if (++stall_count >= kMaxStall) {
                    break;
                }
                continue;
            }
            // TraceDAG couldn't find an edge to cut either — done.
            break;
        }

        if (iterations >= max_iterations && graph_.ActiveCount() > 1) {
            LOG(WARNING) << "CFGStructure: hit iteration bound ("
                         << max_iterations << ") with "
                         << graph_.ActiveCount()
                         << " active nodes remaining"
                         << " — output will contain residual gotos\n";
        }

        // Wrap any remaining uncollapsed leaf nodes so they have an
        // SNode sequence for the emitter to consume.
        for (auto &node : graph_.nodes) {
            if (!node.IsCollapsed() && node.structured.empty()) {
                node.structured = BuildLeafSNode(node.id);
            }
        }

        // Insert explicit gotos where an active node's successor is
        // NOT the next active node in iteration order.  Without this,
        // the emitter's sequential layout creates spurious fallthrough
        // into goto-only labels placed between the node and its
        // successor (e.g., error labels between if-else and merge point).
        {
            // Build ordered list of active node IDs.
            std::vector<size_t> active_order;
            for (auto &node : graph_.nodes) {
                if (!node.IsCollapsed()) active_order.push_back(node.id);
            }

            for (size_t idx = 0; idx < active_order.size(); ++idx) {
                auto &node = graph_.Node(active_order[idx]);
                if (node.structured.empty()) continue;
                if (node.succs.size() != 1) continue;
                // Skip if the sole successor IS the next active node
                // (fallthrough is correct).
                size_t succ = node.succs[0];
                if (idx + 1 < active_order.size()
                    && active_order[idx + 1] == succ)
                    continue;
                // Successor is NOT next — need explicit goto.
                auto &sn = graph_.Node(succ);
                if (sn.original_label.empty()) continue;
                // Append an explicit SGoto to the node's structured seq.
                node.structured.push_back(factory_.Make<SGoto>(
                    factory_.Intern(sn.original_label)));
            }
        }

        // Ensure labels are preserved on active representative nodes.
        // When IdentifyInternal collapses nodes into a representative,
        // the rule-supplied SNode may not include the representative's
        // own label.  Wrap it now so goto targets remain valid.  Check
        // recursively: the label might be inside a child node.
        auto has_label = [](const std::vector<SNode *> &snodes,
                            std::string_view name) -> bool {
            std::function<bool(const SNode *)> check = [&](const SNode *n) -> bool {
                if (!n) return false;
                if (auto *lbl = n->dyn_cast<SLabel>())
                    if (lbl->Name() == name) return true;
                bool found = false;
                n->for_each_child([&](const SNode *c) {
                    if (!found && check(c)) found = true;
                });
                return found;
            };
            for (const auto *n : snodes)
                if (check(n)) return true;
            return false;
        };

        for (auto &node : graph_.nodes) {
            if (node.IsCollapsed()) continue;
            if (node.original_label.empty()) continue;
            if (node.structured.empty()) continue;
            if (has_label(node.structured, node.original_label)) continue;
            node.structured = { factory_.Make<SLabel>(
                factory_.Intern(node.original_label), node.structured) };
        }
    }

    // ---------------------------------------------------------------
    // OrderLoops
    // ---------------------------------------------------------------

    void CFGStructure::OrderLoops() {
        OrderLoopBodies(graph_, loop_body_storage_);
        for (auto &lb : loop_body_storage_) { loop_order_.push_back(&lb); }
    }

    // ---------------------------------------------------------------
    // StructureInternal — one round of rule matching
    // ---------------------------------------------------------------

    bool CFGStructure::StructureInternal() {
        auto active = graph_.ActiveIds();

        // Build set of node IDs referenced as successors of collapsed
        // nodes.  Used by RuleBlockCat to avoid O(N) label scans.
        collapsed_succ_targets_.clear();
        for (auto &n : graph_.nodes) {
            if (!n.IsCollapsed()) continue;
            for (size_t s : n.succs)
                collapsed_succ_targets_.insert(s);
        }

        // Switch rules must fire before ANY other rule modifies the
        // graph.  RuleBlockCat on predecessor chains can dedup the
        // switch node's succs, invalidating succ_index values in
        // switch_cases metadata.  Process all switches first.
        for (size_t id : active) {
            if (graph_.Node(id).IsCollapsed()) continue;
            if (RuleBlockSwitch(id)) return true;
        }

        for (size_t id : active) {
            if (graph_.Node(id).IsCollapsed()) continue;
            if (RuleBlockCat(id)) return true;
            if (RuleBlockIfElse(id)) return true;
            if (RuleBlockIfReturn(id)) return true;
            // ProperIf: body.succs[0] == merge (direct edge to merge).
            //   Also merges conditional forwarders into &&/|| conditions.
            // PostDomIf: body.succs[0] != merge (body eventually reaches
            //   merge via ipdom/BFS).  Defers when body's successor is a
            //   loop header or has other active preds — lets loop/other
            //   rules collapse the successor first.
            // Non-overlapping: ProperIf skips when succ != merge,
            // PostDomIf skips when succ == merge.
            if (RuleBlockProperIf(id)) return true;
            if (RuleBlockPostDomIf(id)) return true;
            if (RuleBlockWhileDo(id)) return true;
            if (RuleBlockDoWhile(id)) return true;
            if (RuleBlockInfLoop(id)) return true;
        }
        return false;
    }

    // ---------------------------------------------------------------
    // RuleBlockCat — sequential merge
    //
    // Pattern: Node A has exactly 1 successor B, and B has exactly 1
    //          predecessor A.  A is not a conditional or switch.
    //          Neither is already collapsed.
    // Action:  Merge into a sequence, collapse via IdentifyInternal.
    // ---------------------------------------------------------------

    bool CFGStructure::RuleBlockCat(size_t id) {
        auto &a = graph_.Node(id);
        if (a.IsCollapsed()) return false;
        if (a.succs.size() != 1) return false;
        if (a.is_conditional) return false;
        if (a.IsSwitchOut()) return false;

        size_t b_id = a.succs[0];
        auto &b = graph_.Node(b_id);
        if (b.IsCollapsed()) return false;
        if (b.preds.size() != 1) return false;

        // Don't merge if B is conditional or a switch — absorbing it
        // would destroy a diamond/switch pattern that RuleBlockIfElse
        // or RuleBlockSwitch should match instead.
        if (b.is_conditional) return false;
        if (b.IsSwitchOut()) return false;

        // Don't merge across a goto edge.
        if (a.IsGotoOut(0)) return false;

        // Don't merge if B has a label that may still be referenced
        // by collapsed nodes or active goto edges.  After collapse,
        // b.preds may show 1 active pred, but collapsed nodes' SNode
        // trees can still hold clang::GotoStmt to B's label.
        // Merging B would consume the label, creating dangling goto
        // references.
        //
        // collapsed_succ_targets_ (precomputed per StructureInternal
        // round) gives O(1) lookup for the collapsed-node check.
        // Active goto edges are checked via B's pred list.
        if (!b.original_label.empty()) {
            // Collapsed node references B — label needed.
            if (collapsed_succ_targets_.count(b_id))
                return false;

            // Active node with goto edge to B — label needed.
            for (size_t p : b.preds) {
                if (p == id) continue;
                auto &pn = graph_.Node(p);
                if (pn.IsCollapsed()) continue;
                for (size_t i = 0; i < pn.succs.size(); ++i) {
                    if (pn.succs[i] == b_id && pn.IsGotoOut(i))
                        return false;
                }
            }
        }

        // Build the merged sequence: A's content followed by B's.
        std::vector< SNode * > seq = BuildLeafSNode(id, /*include_terminal=*/false);
        SeqAppend(seq, BuildLeafSNode(b_id));

        // Collapse {A, B} into the representative node.
        graph_.IdentifyInternal({id, b_id}, CNode::BlockType::kSequence,
                                std::move(seq));

        return true;
    }

    // ---------------------------------------------------------------
    // Helper: check if the only "real" predecessor of `node_id` is
    // `expected_pred`.  A predecessor is "real" if it is not collapsed
    // and the edge to node_id is not marked as a goto.  Goto-edge
    // predecessors and collapsed predecessors are ignored.
    //
    // This relaxes the strict single-predecessor check for if/else
    // rules: a node may have multiple predecessors in the graph, but
    // if all the extra ones are goto edges or collapsed nodes, the
    // node is effectively single-predecessor for structuring purposes.
    // ---------------------------------------------------------------

    bool CFGStructure::HasSoleRealPredecessor(size_t node_id,
                                                    size_t expected_pred) {
        auto &node = graph_.Node(node_id);
        for (size_t p : node.preds) {
            if (p == expected_pred) continue;
            auto &pn = graph_.Node(p);
            if (pn.IsCollapsed()) continue;

            // Check if the edge p→node_id is a goto edge.
            bool is_goto = false;
            for (size_t i = 0; i < pn.succs.size(); ++i) {
                if (pn.succs[i] == node_id && pn.IsGotoOut(i)) {
                    is_goto = true;
                    break;
                }
            }
            if (is_goto) continue;

            // Found a real non-goto, non-collapsed predecessor that
            // isn't expected_pred — node has multiple real predecessors.
            return false;
        }
        return true;
    }

    // ---------------------------------------------------------------
    // WrapWithPriorContent — shared helper for all if/if-else rules
    // ---------------------------------------------------------------

    std::vector< SNode * > CFGStructure::WrapWithPriorContent(
        size_t id, SNode *child) {
        auto &a = graph_.Node(id);
        // Prefix: a's prior content — either a pre-structured sequence
        // from an earlier rule, or a.stmts spilled as SStmt siblings.
        std::vector< SNode * > result;
        if (!a.structured.empty()) {
            result = a.structured;
        } else {
            AppendStmts(factory_, result, a.stmts);
        }
        if (child) result.push_back(child);
        if (!a.original_label.empty()) {
            // Skip the wrap if `result` already exposes the label at
            // its head; loop rules emit SLabel(name, SWhile(...)) and
            // a second wrap would produce duplicate LabelStmts.
            bool head_has_label = false;
            if (!result.empty()) {
                if (auto *lbl = result[0]->dyn_cast<SLabel>())
                    head_has_label = (lbl->Name() == a.original_label);
            }
            if (!head_has_label) {
                return { factory_.Make<SLabel>(
                    factory_.Intern(a.original_label), result) };
            }
        }
        return result;
    }

    // ---------------------------------------------------------------
    // RuleBlockProperIf — if-then (no else)
    //
    // Pattern: Node A is conditional with 2 successors (T, F).
    //          T has exactly 1 predecessor (A) and exactly 1
    //          successor that equals F.  F is the merge point.
    //          Neither T nor F is collapsed.
    // Action:  Create SIfThenElse(cond, then_body, nullptr).
    //          Collapse {A, T} with exit to F.
    //
    // Also handles the symmetric case: F is the body and T is the
    // merge — the condition is negated so the body is always the
    // then-branch.
    // ---------------------------------------------------------------

    bool CFGStructure::RuleBlockProperIf(size_t id) {
        auto &a = graph_.Node(id);
        if (a.IsCollapsed()) return false;
        if (!a.is_conditional) return false;
        if (a.succs.size() != 2) return false;
        if (!a.branch_cond) return false;

        size_t f_id = a.succs[0];  // not-taken (cond false)
        size_t t_id = a.succs[1];  // taken (cond true)

        auto &t = graph_.Node(t_id);
        auto &f = graph_.Node(f_id);
        if (t.IsCollapsed() || f.IsCollapsed()) return false;

        // Use ipdom to determine orientation.  After Phase 7 normalization,
        // ipdom[A] == succs[0] (merge on not-taken) for most nodes, so
        // Case 1 (body=taken) should fire.  Skip Case 2 when ipdom confirms.
        constexpr size_t kNone = CNode::kNone;
        size_t ipd = (!ipdom_.empty() && id < ipdom_.size()) ? ipdom_[id] : kNone;
        bool skip_case1 = (ipd != kNone && ipd == t_id);  // merge on taken → body on not-taken
        bool skip_case2 = (ipd != kNone && ipd == f_id);  // merge on not-taken → body on taken

        // Case 1: T is the body, F is the merge.
        // T has A as sole real pred, 1 succ (F).
        if (!skip_case1
            && HasSoleRealPredecessor(t_id, id) && t.succs.size() == 1
            && t.succs[0] == f_id && !t.is_conditional)
        {
            auto then_body = BuildLeafSNode(t_id, /*include_terminal=*/false);
            auto *if_node = factory_.Make<SIfThenElse>(
                a.branch_cond, then_body, std::vector< SNode * >{});

            std::vector< SNode * > result = WrapWithPriorContent(id, if_node);

            graph_.IdentifyInternal({id, t_id}, CNode::BlockType::kIf, result);
            return true;
        }

        // Case 1b: T is a conditional forwarder (no stmts, just a branch)
        // with one arm going to F (merge).  Two emission forms:
        //   (a) Disjunctive targeting merge (F):
        //         if (!a.cond || inner_cond) goto F.label;
        //       Used when F's label is "independently referenced" -- either
        //       a collapsed SNode tree already emits a goto to it, or some
        //       other live graph node will also reach it.  Guarantees the
        //       label survives cleanup so the guard isn't dropped.
        //   (b) Conjunctive targeting non-merge (goto_target):
        //         if (a.cond && inner_cond) goto goto_target.label;
        //       Used when the merge is only referenced by this collapse
        //       path.  Relies on ClangEmitterCleanup::ScopeifyIfGotos to
        //       scope the body; fails if the target becomes adjacent but
        //       safe for the majority of short-range skip patterns (see
        //       cwe22_init_logger pre-fix state).
        // Guard: must not have a structured SNode -- collapsed nodes have
        // stmts cleared by IdentifyInternal but carry content in structured.
        if (!skip_case1
            && HasSoleRealPredecessor(t_id, id) && t.is_conditional
            && t.stmts.empty() && t.structured.empty()
            && t.succs.size() == 2 && t.branch_cond)
        {
            bool t_s0_is_merge = (t.succs[0] == f_id);
            bool t_s1_is_merge = (t.succs[1] == f_id);
            if (t_s0_is_merge || t_s1_is_merge) {
                // Is the merge (F) independently labeled?  Any pred
                // other than A (this node) and T (the forwarder we are
                // collapsing) implies at least one other route to F,
                // which will keep F's label live through cleanup.
                // Count collapsed preds too: their edges still existed
                // in the original graph, and the structured SNode trees
                // that replaced them typically emit gotos to F.  A
                // previous version of this check used the snapshot
                // `collapsed_succ_targets_` + filtered live preds, but
                // that snapshot is populated once per StructureInternal
                // round and goes stale as rules fire within the round.
                auto merge_is_safe_goto_target = [&]() {
                    for (size_t p : f.preds) {
                        if (p == id || p == t_id) continue;
                        return true;
                    }
                    return false;
                };

                if (!f.original_label.empty() && merge_is_safe_goto_target()) {
                    // Form (a): disjunctive, goto F (merge).
                    // A.cond FALSE -> F directly, so outer = !a.cond.
                    // If T.succs[1] (taken) is F, t.cond TRUE -> F, so
                    // inner = t.cond; else inner = !t.cond.
                    // Clone the raw branch_cond pointers so this merged
                    // condition owns an independent Expr tree — the same
                    // a.branch_cond / t.branch_cond pointers may already
                    // be referenced by other SNode constructions.
                    clang::Expr *outer_cond = NegateExpr(ctx_, CloneExpr(ctx_, a.branch_cond));
                    clang::Expr *inner_cond = t_s1_is_merge
                        ? CloneExpr(ctx_, t.branch_cond)
                        : NegateExpr(ctx_, CloneExpr(ctx_, t.branch_cond));
                    auto *merged_cond = clang::BinaryOperator::Create(
                        ctx_,
                        EnsureRValue(ctx_, outer_cond),
                        EnsureRValue(ctx_, inner_cond),
                        clang::BO_LOr, ctx_.BoolTy, clang::VK_PRValue,
                        clang::OK_Ordinary, VirtualLoc(ctx_),
                        clang::FPOptionsOverride());

                    auto *if_goto = factory_.Make<SIfThenElse>(
                        merged_cond,
                        factory_.Make<SGoto>(factory_.Intern(f.original_label)),
                        nullptr);

                    std::vector< SNode * > result = WrapWithPriorContent(id, if_goto);
                    graph_.IdentifyInternal({id, t_id}, CNode::BlockType::kIf, result);
                    // The guard jumps to f via an explicit goto; the
                    // fallthrough is the non-merge arm.
                    ConsumeGotoEdge(graph_, id, f_id);
                    return true;
                }

                // Form (b): original conjunctive, goto non-merge.
                size_t goto_target = t_s0_is_merge ? t.succs[1] : t.succs[0];
                auto &target_node = graph_.Node(goto_target);
                if (!target_node.original_label.empty()) {
                    // succs[1] = taken (cond true).  If taken goes to merge,
                    // the goto fires when cond is false — negate.  Clone
                    // the raw branch_cond pointers so this merged condition
                    // owns an independent Expr tree.
                    clang::Expr *outer_cond = CloneExpr(ctx_, a.branch_cond);
                    clang::Expr *inner_cond = t_s1_is_merge
                        ? NegateExpr(ctx_, CloneExpr(ctx_, t.branch_cond))
                        : CloneExpr(ctx_, t.branch_cond);
                    auto *merged_cond = clang::BinaryOperator::Create(
                        ctx_,
                        EnsureRValue(ctx_, outer_cond),
                        EnsureRValue(ctx_, inner_cond),
                        clang::BO_LAnd, ctx_.BoolTy, clang::VK_PRValue,
                        clang::OK_Ordinary, VirtualLoc(ctx_),
                        clang::FPOptionsOverride());

                    auto *if_goto = factory_.Make<SIfThenElse>(
                        merged_cond,
                        factory_.Make<SGoto>(factory_.Intern(target_node.original_label)),
                        nullptr);

                    std::vector< SNode * > result = WrapWithPriorContent(id, if_goto);
                    graph_.IdentifyInternal({id, t_id}, CNode::BlockType::kIf, result);
                    // The guard jumps to goto_target via an explicit
                    // goto; the fallthrough is the merge arm.
                    ConsumeGotoEdge(graph_, id, goto_target);
                    return true;
                }
            }
        }

        // Case 2: F is the body, T is the merge.
        // F has A as sole real pred, 1 succ (T).
        // Skipped when ipdom confirms merge is on not-taken (Case 1 orientation).
        if (!skip_case2
            && HasSoleRealPredecessor(f_id, id) && f.succs.size() == 1
            && f.succs[0] == t_id && !f.is_conditional)
        {
            auto then_body = BuildLeafSNode(f_id, /*include_terminal=*/false);
            // The condition is for the taken branch (T = merge).
            // The body executes when the condition is false — negate.
            auto *if_node = factory_.Make<SIfThenElse>(
                NegateExpr(ctx_, a.branch_cond), then_body,
                std::vector< SNode * >{});

            std::vector< SNode * > result = WrapWithPriorContent(id, if_node);

            graph_.IdentifyInternal({id, f_id}, CNode::BlockType::kIf, result);
            return true;
        }

        // Case 2b: F is a conditional forwarder (no stmts, just a branch)
        // with one arm going to T (merge).  Mirror of Case 1b with T as
        // the merge.  Two emission forms:
        //   (a) Disjunctive targeting merge (T):
        //         if (a.cond || inner_cond) goto T.label;
        //       (A.cond TRUE -> T directly, so outer = a.cond, no negate.)
        //   (b) Conjunctive targeting non-merge (goto_target):
        //         if (!a.cond && inner_cond) goto goto_target.label;
        // See Case 1b for rationale.  Guard: must not have structured SNode.
        if (!skip_case2
            && HasSoleRealPredecessor(f_id, id) && f.is_conditional
            && f.stmts.empty() && f.structured.empty()
            && f.succs.size() == 2 && f.branch_cond)
        {
            bool f_s0_is_merge = (f.succs[0] == t_id);
            bool f_s1_is_merge = (f.succs[1] == t_id);
            if (f_s0_is_merge || f_s1_is_merge) {
                // Is the merge (T) independently labeled?  See Case 1b
                // for rationale on counting collapsed preds too.
                auto merge_is_safe_goto_target = [&]() {
                    for (size_t p : t.preds) {
                        if (p == id || p == f_id) continue;
                        return true;
                    }
                    return false;
                };

                if (!t.original_label.empty() && merge_is_safe_goto_target()) {
                    // Form (a): disjunctive, goto T (merge).
                    // A.cond TRUE -> T directly, no negation on outer.
                    // If F.succs[1] (taken) is T, f.cond TRUE -> T, so
                    // inner = f.cond; else inner = !f.cond.
                    // Clone raw branch_cond pointers — see Case 1b.
                    clang::Expr *outer_cond = CloneExpr(ctx_, a.branch_cond);
                    clang::Expr *inner_cond = f_s1_is_merge
                        ? CloneExpr(ctx_, f.branch_cond)
                        : NegateExpr(ctx_, CloneExpr(ctx_, f.branch_cond));
                    auto *merged_cond = clang::BinaryOperator::Create(
                        ctx_,
                        EnsureRValue(ctx_, outer_cond),
                        EnsureRValue(ctx_, inner_cond),
                        clang::BO_LOr, ctx_.BoolTy, clang::VK_PRValue,
                        clang::OK_Ordinary, VirtualLoc(ctx_),
                        clang::FPOptionsOverride());

                    auto *if_goto = factory_.Make<SIfThenElse>(
                        merged_cond,
                        factory_.Make<SGoto>(factory_.Intern(t.original_label)),
                        nullptr);

                    std::vector< SNode * > result = WrapWithPriorContent(id, if_goto);
                    graph_.IdentifyInternal({id, f_id}, CNode::BlockType::kIf, result);
                    // The guard jumps to t via an explicit goto; the
                    // fallthrough is the non-merge arm.
                    ConsumeGotoEdge(graph_, id, t_id);
                    return true;
                }

                // Form (b): original conjunctive, goto non-merge.
                size_t goto_target = f_s0_is_merge ? f.succs[1] : f.succs[0];
                auto &target_node = graph_.Node(goto_target);
                if (!target_node.original_label.empty()) {
                    // Outer condition: F is the not-taken arm, so body
                    // executes when c1 is false — negate outer.  Clone
                    // raw branch_cond pointers — see Case 1b.
                    clang::Expr *outer_cond = NegateExpr(ctx_, CloneExpr(ctx_, a.branch_cond));
                    clang::Expr *inner_cond = f_s1_is_merge
                        ? NegateExpr(ctx_, CloneExpr(ctx_, f.branch_cond))
                        : CloneExpr(ctx_, f.branch_cond);
                    auto *merged_cond = clang::BinaryOperator::Create(
                        ctx_,
                        EnsureRValue(ctx_, outer_cond),
                        EnsureRValue(ctx_, inner_cond),
                        clang::BO_LAnd, ctx_.BoolTy, clang::VK_PRValue,
                        clang::OK_Ordinary, VirtualLoc(ctx_),
                        clang::FPOptionsOverride());

                    auto *if_goto = factory_.Make<SIfThenElse>(
                        merged_cond,
                        factory_.Make<SGoto>(factory_.Intern(target_node.original_label)),
                        nullptr);

                    std::vector< SNode * > result = WrapWithPriorContent(id, if_goto);
                    graph_.IdentifyInternal({id, f_id}, CNode::BlockType::kIf, result);
                    // The guard jumps to goto_target via an explicit
                    // goto; the fallthrough is the merge arm.
                    ConsumeGotoEdge(graph_, id, goto_target);
                    return true;
                }
            }
        }

        return false;
    }

    // ---------------------------------------------------------------
    // RuleBlockPostDomIf — post-dominator-guided if-then
    //
    // Pattern: Node A is conditional with 2 successors.
    //          One arm (body) has A as sole real pred and is not conditional.
    //          The other arm (merge) is reachable from the body arm
    //          (i.e., the body eventually reaches the merge).
    //          The merge may have multiple preds (shared).
    //
    // This is a relaxation of RuleBlockProperIf that doesn't require
    // the body arm's sole succ to be the merge.  Instead, it checks
    // that the body arm eventually reaches the merge — i.e., the
    // merge is the post-dominator of the conditional.
    //
    // Only fires when the body arm has a single succ (simple body),
    // to avoid absorbing complex multi-exit bodies.
    // ---------------------------------------------------------------

    bool CFGStructure::RuleBlockPostDomIf(size_t id) {
        auto &a = graph_.Node(id);
        if (a.IsCollapsed()) return false;
        if (!a.is_conditional) return false;
        if (a.succs.size() != 2) return false;
        if (!a.branch_cond) return false;

        size_t f_id = a.succs[0];  // not-taken
        size_t t_id = a.succs[1];  // taken

        auto &t = graph_.Node(t_id);
        auto &f = graph_.Node(f_id);
        if (t.IsCollapsed() || f.IsCollapsed()) return false;

        // Use ipdom to identify the merge point (O(1) vs BFS).
        constexpr size_t kNone = CNode::kNone;
        size_t ipd = (!ipdom_.empty()) ? ipdom_[id] : kNone;

        // Try both orientations: T=body/F=merge and F=body/T=merge.
        for (int orient = 0; orient < 2; ++orient) {
            size_t body_id = orient == 0 ? t_id : f_id;
            size_t merge_id = orient == 0 ? f_id : t_id;
            auto &body = graph_.Node(body_id);
            auto &merge = graph_.Node(merge_id);

            // Body must have sole real pred = A.
            if (!HasSoleRealPredecessor(body_id, id)) continue;

            // Body must not be conditional or switch.
            if (body.is_conditional || body.IsSwitchOut()) continue;

            // Body must have exactly 1 successor.
            if (body.succs.size() != 1) continue;

            // Body's succ must NOT be the merge (RuleBlockProperIf handles that).
            if (body.succs[0] == merge_id) continue;

            // Defer when the body's successor is a loop header or has
            // other active non-goto predecessors.  This lets loop/switch/
            // if/cat rules collapse the successor first, so RuleBlockCat
            // can merge body + collapsed successor into a complete
            // if-body on a later pass.
            {
                size_t bsucc = body.succs[0];

                // Check if bsucc is a loop header — always defer so
                // loop rules structure it first.
                bool is_loop_header = false;
                for (auto *lb : loop_order_) {
                    if (lb->head == bsucc && !graph_.Node(bsucc).IsCollapsed()) {
                        is_loop_header = true;
                        break;
                    }
                }
                if (is_loop_header) continue;

                // Also defer if bsucc has other active non-goto preds.
                auto &bs = graph_.Node(bsucc);
                bool has_other_active_pred = false;
                for (size_t p : bs.preds) {
                    if (p == body_id) continue;
                    if (graph_.Node(p).IsCollapsed()) continue;
                    auto &pn = graph_.Node(p);
                    bool all_pure_goto = true;
                    for (size_t si = 0; si < pn.succs.size(); ++si) {
                        if (pn.succs[si] != bsucc) continue;
                        if (!pn.IsGotoOut(si)) {
                            all_pure_goto = false;
                            break;
                        }
                    }
                    if (!all_pure_goto) {
                        has_other_active_pred = true;
                        break;
                    }
                }
                if (has_other_active_pred) continue;
            }

            // Verify body eventually reaches merge.
            // Fast path: ipdom_[A] == merge_id (O(1)).
            // Slow path: BFS reachability when ipdom doesn't match
            // directly (merge may be a non-immediate post-dominator).
            bool reaches_merge = false;
            if (ipd == merge_id) {
                reaches_merge = true;
            } else {
                std::vector<size_t> worklist = {body.succs[0]};
                std::unordered_set<size_t> visited;
                while (!worklist.empty() && visited.size() < graph_.nodes.size()) {
                    size_t cur = worklist.back();
                    worklist.pop_back();
                    if (cur == merge_id) { reaches_merge = true; break; }
                    if (!visited.insert(cur).second) continue;
                    auto &cn = graph_.Node(cur);
                    if (cn.IsCollapsed()) continue;
                    for (size_t s : cn.succs) worklist.push_back(s);
                }
            }
            if (!reaches_merge) continue;

            // Merge must not be collapsed.
            if (merge.IsCollapsed()) continue;

            // Build if-then: body is the then-branch.
            auto then_body = BuildLeafSNode(body_id, /*include_terminal=*/false);

            // Negate condition if body is the not-taken arm (orient==1).
            clang::Expr *cond = orient == 0
                ? a.branch_cond
                : NegateExpr(ctx_, a.branch_cond);

            auto *if_node = factory_.Make<SIfThenElse>(
                cond, then_body, std::vector< SNode * >{});

            std::vector< SNode * > result = WrapWithPriorContent(id, if_node);

            // Collapse {A, body} with exit to merge.
            graph_.IdentifyInternal(
                {id, body_id}, CNode::BlockType::kIf, result);
            return true;
        }

        return false;
    }

    // ---------------------------------------------------------------
    // RuleBlockIfElse — if-then-else (diamond)
    //
    // Pattern: Node A is conditional with 2 successors (T, F).
    //          Both T and F have exactly 1 predecessor (A).
    //          Both T and F have exactly 1 successor, and those
    //          successors are the same merge point M.
    //          M is not collapsed.
    // Action:  Create SIfThenElse(cond, then_body, else_body).
    //          Collapse {A, T, F} with exit to M.
    // ---------------------------------------------------------------

    /// Helper: check if node d is dominated by node root via idom_ chain.
    bool CFGStructure::IsDominatedBy(size_t d, size_t root) const {
        constexpr size_t kNone = CNode::kNone;
        while (d != root) {
            if (d == kNone) return false;
            size_t up = idom_[d];
            if (up == d) return false;  // reached entry without finding root
            d = up;
        }
        return true;
    }

    /// Helper: collect all active nodes dominated by `root` but not
    /// dominated by `stop` (and not `stop` itself).  Returns them
    /// sorted by rpo_pos_.
    std::vector<size_t> CFGStructure::CollectDomRegion(
            size_t root, size_t stop) const {
        std::vector<size_t> region;
        for (auto &n : graph_.nodes) {
            if (n.IsCollapsed()) continue;
            if (n.id == stop) continue;
            if (IsDominatedBy(n.id, root) && !IsDominatedBy(n.id, stop)) {
                region.push_back(n.id);
            }
        }
        std::sort(region.begin(), region.end(),
                  [this](size_t a, size_t b) {
                      return rpo_pos_[a] < rpo_pos_[b];
                  });
        return region;
    }

    bool CFGStructure::RuleBlockIfElse(size_t id) {
        auto &a = graph_.Node(id);
        if (a.IsCollapsed()) return false;
        if (!a.is_conditional) return false;
        if (a.succs.size() != 2) return false;
        if (!a.branch_cond) return false;

        size_t f_id = a.succs[0];
        size_t t_id = a.succs[1];

        auto &t = graph_.Node(t_id);
        auto &f = graph_.Node(f_id);
        if (t.IsCollapsed() || f.IsCollapsed()) return false;

        // --- Path 1: standard diamond (both arms single-succ to same merge) ---
        if (t.succs.size() == 1 && f.succs.size() == 1 &&
            t.succs[0] == f.succs[0] &&
            !graph_.Node(t.succs[0]).IsCollapsed() &&
            !t.is_conditional && !f.is_conditional &&
            !t.IsSwitchOut() && !f.IsSwitchOut()) {

            bool t_sole = HasSoleRealPredecessor(t_id, id);
            bool f_sole = HasSoleRealPredecessor(f_id, id);

            if (t_sole && f_sole) {
                auto then_body = BuildLeafSNode(t_id, /*include_terminal=*/false);
                auto else_body = BuildLeafSNode(f_id, /*include_terminal=*/false);
                auto *if_node = factory_.Make<SIfThenElse>(
                    a.branch_cond, then_body, else_body);

                std::vector< SNode * > result = WrapWithPriorContent(id, if_node);

                graph_.IdentifyInternal(
                    {id, t_id, f_id}, CNode::BlockType::kIf, result);
                return true;
            }

            // Relaxed: one arm has sole pred, the other is shared.
            // Only safe when the shared arm has NO stmts (pure routing
            // node) — otherwise its side effects would execute on both
            // branches instead of only the original one.
            size_t sole_id;
            bool sole_is_taken;
            if (t_sole && !f_sole && f.stmts.empty() && f.structured.empty()) {
                sole_id = t_id; sole_is_taken = true;
            } else if (f_sole && !t_sole && t.stmts.empty() && t.structured.empty()) {
                sole_id = f_id; sole_is_taken = false;
            } else {
                return false;
            }

            {
                auto sole_body = BuildLeafSNode(sole_id, /*include_terminal=*/false);
                clang::Expr *cond = sole_is_taken
                    ? a.branch_cond
                    : NegateExpr(ctx_, a.branch_cond);
                auto *if_node = factory_.Make<SIfThenElse>(
                    cond, sole_body, std::vector< SNode * >{});

                std::vector< SNode * > result = WrapWithPriorContent(id, if_node);

                graph_.IdentifyInternal(
                    {id, sole_id}, CNode::BlockType::kIf, result);
                return true;
            }
        }

        return false;
    }

    // ---------------------------------------------------------------
    // RuleBlockIfReturn — if-then-else where both arms terminate
    //
    // Pattern: Node A is conditional with 2 successors (T, F).
    //          Both T and F have exactly 1 predecessor (A).
    //          Both T and F have NO successors (they end with return
    //          or are dead-ends).  No merge point is needed.
    // Action:  Create SIfThenElse(cond, then_body, else_body).
    //          Collapse {A, T, F} — representative has no successors.
    // ---------------------------------------------------------------

    bool CFGStructure::RuleBlockIfReturn(size_t id) {
        auto &a = graph_.Node(id);
        if (a.IsCollapsed()) return false;
        if (!a.is_conditional) return false;
        if (a.succs.size() != 2) return false;
        if (!a.branch_cond) return false;

        size_t f_id = a.succs[0];  // not-taken (cond false)
        size_t t_id = a.succs[1];  // taken (cond true)

        auto &t = graph_.Node(t_id);
        auto &f = graph_.Node(f_id);
        if (t.IsCollapsed() || f.IsCollapsed()) return false;

        // Both must have A as sole real predecessor (ignoring goto
        // edges and collapsed nodes).
        if (!HasSoleRealPredecessor(t_id, id)) return false;
        if (!HasSoleRealPredecessor(f_id, id)) return false;

        // Both must have NO successors (terminate with return).
        if (!t.succs.empty() || !f.succs.empty()) return false;

        // Don't match if either branch is itself conditional or switch.
        if (t.is_conditional || f.is_conditional) return false;
        if (t.IsSwitchOut() || f.IsSwitchOut()) return false;

        // Build the if-then-else SNode.
        // Both arms include their terminal (return stmt).
        auto then_body = BuildLeafSNode(t_id, /*include_terminal=*/true);
        auto else_body = BuildLeafSNode(f_id, /*include_terminal=*/true);
        auto *if_node = factory_.Make<SIfThenElse>(
            a.branch_cond, then_body, else_body);

        // A's prior content goes before the if: use a.structured
        std::vector< SNode * > result = WrapWithPriorContent(id, if_node);

        graph_.IdentifyInternal(
            {id, t_id, f_id}, CNode::BlockType::kIf, result);
        return true;
    }

    // ---------------------------------------------------------------
    // Loop helper: find the LoopBody whose head matches `id`.
    // Returns nullptr if none found.
    // ---------------------------------------------------------------

    /// Resolve collapsed body nodes to their active representatives.
    /// Collapsed nodes are replaced by their representative (following
    /// the collapsed_into chain).  Duplicates and the header are removed.
    static void ResolveBodyToRepresentatives(
        const CGraph &g, std::vector<size_t> &body, size_t /*header_id*/
    ) {
        std::unordered_set<size_t> seen;
        std::vector<size_t> resolved;
        for (size_t nid : body) {
            auto &nd = g.Node(nid);
            if (!nd.IsCollapsed()) {
                // Active node — keep as-is.
                if (seen.insert(nid).second) resolved.push_back(nid);
            } else if (!nd.structured.empty()) {
                // Collapsed but has structured content from a prior rule.
                // Keep the original id so BuildLoopBodySNode can use
                // its structured SNode.  Don't resolve to representative
                // (which may be outside the loop body).
                if (seen.insert(nid).second) resolved.push_back(nid);
            } else {
                // Collapsed without structured content — resolve to rep.
                size_t cur = nid;
                size_t steps = 0;
                while (g.Node(cur).IsCollapsed() && steps < g.nodes.size()) {
                    size_t next = g.Node(cur).collapsed_into;
                    if (next == cur || next == CNode::kNone) break;
                    cur = next;
                    ++steps;
                }
                if (seen.insert(cur).second) resolved.push_back(cur);
            }
        }
        body = std::move(resolved);
    }

    static LoopBody *FindLoopForHead(std::list<LoopBody> &storage, size_t id) {
        for (auto &lb : storage) {
            if (lb.head == id) return &lb;
        }
        return nullptr;
    }

    // ---------------------------------------------------------------
    // Loop helper: build the body SNode for a loop.
    //
    // Collects body nodes (excluding the header) in RPO-ish order,
    // strips terminals from interior nodes.
    // ---------------------------------------------------------------

    std::vector< SNode * > CFGStructure::BuildLoopBodySNode(
        const std::vector<size_t> &body, size_t header_id,
        const std::unordered_set<size_t> &bodyset
    ) {
        // Collect body nodes excluding header, sorted by node id.
        // Node ids are RPO indices from CGraph construction, so sorting
        // by id produces the correct topological order.
        //
        // Include active nodes AND collapsed nodes whose representative
        // has a structured SNode.  Use the representative's id if active,
        // otherwise keep the original (for its structured SNode).
        std::unordered_set<size_t> seen;
        std::vector<size_t> interior;
        for (size_t nid : body) {
            if (nid == header_id) continue;
            auto &nd = graph_.Node(nid);
            if (!nd.IsCollapsed()) {
                if (seen.insert(nid).second) interior.push_back(nid);
            } else if (!nd.structured.empty()) {
                // Collapsed but has structured content — include it.
                if (seen.insert(nid).second) interior.push_back(nid);
            }
        }
        std::sort(interior.begin(), interior.end());

        if (interior.empty()) {
            return {}; // empty body
        }

        // Helper: for a conditional interior node, if one successor
        // exits the loop (not in bodyset), emit if (exit_cond) goto label;
        // to preserve the exit path that would be lost when the terminal
        // is stripped.
        // next_rpo_id: node ID of the next interior block in RPO order,
        // or SIZE_MAX if this is the last block.  Used to decide whether
        // stripping a terminal goto produces correct fallthrough.
        auto build_node = [&](size_t nid,
                              size_t next_rpo_id) -> std::vector< SNode * > {
            auto &nd = graph_.Node(nid);

            // Collapsed nodes with pre-built structured content: return
            // it directly — don't try to access succs/preds (stale).
            if (nd.IsCollapsed() && !nd.structured.empty())
                return nd.structured;

            std::vector< SNode * > leaf =
                BuildLeafSNode(nid, /*include_terminal=*/false);

            // Non-conditional nodes: check if the sole successor is
            // the next RPO block.  If not, emit an explicit goto to
            // preserve the control-flow edge.
            if (!nd.is_conditional || !nd.branch_cond || nd.succs.size() != 2) {
                if (nd.succs.size() == 1) {
                    size_t target = nd.succs[0];
                    if (bodyset.count(target) > 0
                        && target != next_rpo_id
                        && !nd.IsGotoOut(0)) {
                        auto &tn = graph_.Node(target);
                        if (!tn.original_label.empty()) {
                            leaf.push_back(factory_.Make<SGoto>(
                                factory_.Intern(tn.original_label)));
                        }
                    }
                }
                return leaf;
            }

            // Check if one successor exits the loop body.
            // CGraph convention: succs[0] = not-taken (cond false),
            //                    succs[1] = taken (cond true).
            size_t s0 = nd.succs[0];  // not-taken (cond false)
            size_t s1 = nd.succs[1];  // taken (cond true)
            bool s0_in = bodyset.count(s0) > 0;
            bool s1_in = bodyset.count(s1) > 0;

            // Both outside — should not happen in a valid loop body.
            if (!s0_in && !s1_in) return leaf;

            // Both inside: preserve the conditional as
            //   if(branch_cond) goto taken_label;
            // The not-taken path falls through to the next RPO block.
            if (s0_in && s1_in) {
                auto &s1_node = graph_.Node(s1);
                if (!s1_node.original_label.empty()) {
                    auto *taken_goto = factory_.Make<SGoto>(
                        factory_.Intern(s1_node.original_label));
                    SNode *else_branch = nullptr;
                    // If not-taken is NOT the next RPO block, add else-goto.
                    if (s0 != next_rpo_id && next_rpo_id != SIZE_MAX) {
                        auto &s0_node = graph_.Node(s0);
                        if (!s0_node.original_label.empty()) {
                            else_branch = factory_.Make<SGoto>(
                                factory_.Intern(s0_node.original_label));
                        }
                    }
                    auto *if_goto = factory_.Make<SIfThenElse>(
                        nd.branch_cond, taken_goto, else_branch);

                    leaf.push_back(if_goto);
                    return leaf;
                }
                return leaf;
            }

            // One exits: build if (exit_cond) goto exit_label;
            size_t exit_id = s0_in ? s1 : s0;
            auto &exit_node = graph_.Node(exit_id);

            // If the exit target has no label (e.g., entry block), emit
            // if(exit_cond) break instead of a dangling goto.  This
            // preserves the exit path — without it the loop would have
            // no conditional exit for this branch.
            const bool use_break = exit_node.original_label.empty();
            const std::string &lbl = exit_node.original_label;

            // Exit condition: the branch arm that leaves the body.
            // succs[1] = taken (cond true).  If taken exits, exit_cond = branch_cond.
            // succs[0] = not-taken (cond false). If not-taken exits, exit_cond = !cond.
            clang::Expr *exit_cond = s0_in
                ? nd.branch_cond                     // s1 (taken) exits → exit when true
                : NegateExpr(ctx_, nd.branch_cond);  // s0 (not-taken) exits → exit when false

            SNode *exit_stmt = use_break
                ? static_cast<SNode *>(factory_.Make<SBreak>())
                : static_cast<SNode *>(factory_.Make<SGoto>(factory_.Intern(lbl)));
            auto *if_goto = factory_.Make<SIfThenElse>(
                exit_cond, exit_stmt, nullptr);

            leaf.push_back(if_goto);
            return leaf;
        };

        if (interior.size() == 1) {
            return build_node(interior[0], SIZE_MAX);
        }

        // Build all nodes, then merge consecutive if-gotos that target
        // the same label into a single if (cond1 || cond2) goto label;
        std::vector<SNode *> children;
        for (size_t idx = 0; idx < interior.size(); ++idx) {
            size_t next = (idx + 1 < interior.size())
                ? interior[idx + 1] : SIZE_MAX;
            SeqAppend(children, build_node(interior[idx], next));
        }

        // Merge pass: look for SIfThenElse(cond, SGoto(L), null) nodes
        // targeting the same label and combine with ||.  In the spilled
        // model empty blocks no longer exist, so adjacent if-gotos merge
        // directly.
        for (size_t i = 0; i < children.size(); ++i) {
            auto *ite1 = children[i]->dyn_cast<SIfThenElse>();
            if (!ite1 || ite1->ElseBranch() || !ite1->ThenBranch())
                continue;
            auto *g1 = ite1->ThenBranch()->dyn_cast<SGoto>();
            if (!g1) continue;

            // The next sibling is the merge candidate.
            size_t j = i + 1;
            if (j >= children.size()) continue;

            auto *ite2 = children[j]->dyn_cast<SIfThenElse>();
            if (!ite2 || ite2->ElseBranch() || !ite2->ThenBranch())
                continue;
            auto *g2 = ite2->ThenBranch()->dyn_cast<SGoto>();
            if (!g2 || g1->Target() != g2->Target()) continue;

            // Merge: if (c1 || c2) goto L;
            auto *merged_cond = clang::BinaryOperator::Create(
                ctx_,
                EnsureRValue(ctx_, ite1->Cond()),
                EnsureRValue(ctx_, ite2->Cond()),
                clang::BO_LOr, ctx_.BoolTy, clang::VK_PRValue,
                clang::OK_Ordinary, ite1->Cond()->getExprLoc(),
                clang::FPOptionsOverride());
            auto *merged = factory_.Make<SIfThenElse>(
                merged_cond,
                factory_.Make<SGoto>(g1->Target()),
                nullptr);
            children[i] = merged;
            // Remove the merged-in sibling j.
            children.erase(children.begin() + static_cast<ptrdiff_t>(i + 1),
                           children.begin() + static_cast<ptrdiff_t>(j + 1));
            --i;  // retry from same position (might merge 3+)
        }

        return factory_.MakeSeq(std::move(children));
    }

    // ---------------------------------------------------------------
    // RuleBlockWhileDo — while loop
    //
    // Pattern: Node H is the head of a detected LoopBody.
    //          H is conditional (2 successors).  One successor is
    //          inside the loop body, the other is the exit.
    // Action:  Create SWhile(H.branch_cond, body).
    //          Collapse all body nodes via IdentifyInternal.
    // ---------------------------------------------------------------

    bool CFGStructure::RuleBlockWhileDo(size_t id) {
        auto &h = graph_.Node(id);
        if (h.IsCollapsed()) return false;
        if (!h.is_conditional) return false;
        if (h.succs.size() != 2) return false;
        if (!h.branch_cond) return false;

        LoopBody *lb = FindLoopForHead(loop_body_storage_, id);
        if (!lb) return false;

        // Compute the body.  Filter out nodes collapsed by prior rules
        // (e.g., switch targets absorbed by RuleBlockSwitch) — their
        // representatives are still active and carry the structured SNode.
        std::vector<size_t> body;
        lb->FindBase(graph_, body);
        if (body.empty()) return false;
        ResolveBodyToRepresentatives(graph_, body, id);

        lb->FindExit(graph_, body);

        // Determine which successor is the body entry and which is exit.
        std::unordered_set<size_t> bodyset(body.begin(), body.end());
        size_t s0 = h.succs[0];
        size_t s1 = h.succs[1];
        bool s0_in_body = bodyset.count(s0) > 0;
        bool s1_in_body = bodyset.count(s1) > 0;

        // Neither successor in body — not a while-do.
        if (!s0_in_body && !s1_in_body) {
            ClearMarks(graph_, body);
            return false;
        }

        // Both successors in body: the header doesn't directly control
        // loop exit — all exits are from interior nodes.  Build as
        // while(1) { header_stmts; if(cond) goto taken; body; }
        if (s0_in_body && s1_in_body) {
            std::vector< SNode * > loop_body_snode =
                BuildLoopBodySNode(body, id, bodyset);

            std::vector<SNode *> inner_children;
            if (!h.structured.empty()) {
                SeqAppend(inner_children, h.structured);
            } else {
                AppendStmts(factory_, inner_children, h.stmts);
            }

            // Header conditional: if(branch_cond) goto taken_label;
            auto &s1_node = graph_.Node(s1);
            if (!s1_node.original_label.empty()) {
                auto *taken_goto = factory_.Make<SGoto>(
                    factory_.Intern(s1_node.original_label));
                inner_children.push_back(factory_.Make<SIfThenElse>(
                    h.branch_cond, taken_goto, nullptr));
            }

            SeqAppend(inner_children, loop_body_snode);

            auto *while_node = factory_.Make<SWhile>(
                nullptr, std::move(inner_children));
            if (!h.original_label.empty())
                while_node->SetHeaderLabel(factory_.Intern(h.original_label));
            if (lb->exit_block != LoopBody::kNone) {
                auto &exit_node = graph_.Node(lb->exit_block);
                if (!exit_node.original_label.empty())
                    while_node->SetExitLabel(
                        factory_.Intern(exit_node.original_label));
            }

            SNode *result = while_node;
            if (!h.original_label.empty()) {
                result = factory_.Make<SLabel>(
                    factory_.Intern(h.original_label), result);
            }

            ClearMarks(graph_, body);
            graph_.IdentifyInternal(body, CNode::BlockType::kWhile, {result});
            return true;
        }

        // Build the while loop SNode.
        // CGraph: succs[0] = not-taken (cond false), succs[1] = taken (cond true).
        // continue_cond: true when body-entry arm is followed.
        // exit_cond: true when the non-body arm is followed.
        clang::Expr *continue_cond = s1_in_body
            ? h.branch_cond                     // body on taken → continue when true
            : NegateExpr(ctx_, h.branch_cond);  // body on not-taken → continue when false
        clang::Expr *exit_cond = s1_in_body
            ? NegateExpr(ctx_, h.branch_cond)   // exit on not-taken → exit when false
            : h.branch_cond;                     // exit on taken → exit when true

        std::vector< SNode * > loop_body_snode =
            BuildLoopBodySNode(body, id, bodyset);
        SNode *result = nullptr;

        if (!h.structured.empty() || !h.stmts.empty()) {
            // Header has computation stmts (or prior structured content)
            // that must re-execute each iteration.  Emit as:
            //   while(1) { header_content; if (exit_cond) break; body; }

            // 1. Header content (re-execute each iteration).
            std::vector< SNode * > inner;
            if (!h.structured.empty()) {
                SeqAppend(inner, h.structured);
            } else {
                AppendStmts(factory_, inner, h.stmts);
            }

            // 2. Exit test: if (exit_cond) break;
            inner.push_back(factory_.Make<SIfThenElse>(
                exit_cond, factory_.Make<SBreak>(), nullptr));

            // 3. Loop body.
            SeqAppend(inner, loop_body_snode);

            // while(1) — nullptr condition → emitter synthesizes true.
            result = factory_.Make<SWhile>(nullptr, std::move(inner));
        } else {
            // Pure condition header (no side-effectful stmts).
            // Emit as: while(continue_cond) { body; }
            result = factory_.Make<SWhile>(continue_cond,
                                           std::move(loop_body_snode));
        }

        // Set loop scope labels for break/continue resolution.
        // result is always SWhile here (both branches above create SWhile).
        auto *while_node = result->as<SWhile>();
        if (!h.original_label.empty())
            while_node->SetHeaderLabel(factory_.Intern(h.original_label));
        size_t exit_id = s1_in_body ? s0 : s1;
        auto &exit_node = graph_.Node(exit_id);
        if (!exit_node.original_label.empty())
            while_node->SetExitLabel(factory_.Intern(exit_node.original_label));

        // Preserve header label.
        if (!h.original_label.empty()) {
            result = factory_.Make<SLabel>(
                factory_.Intern(h.original_label), result);
        }

        ClearMarks(graph_, body);
        graph_.IdentifyInternal(body, CNode::BlockType::kWhile, {result});
        return true;
    }

    // ---------------------------------------------------------------
    // RuleBlockDoWhile — do-while loop
    //
    // Pattern: Node H is the head of a detected LoopBody.
    //          H is NOT conditional (unconditional entry to body).
    //          The loop has a single tail T that IS conditional.
    //          T's back-edge goes to H, other edge is the exit.
    // Action:  Create SDoWhile(body, T.branch_cond).
    // ---------------------------------------------------------------

    bool CFGStructure::RuleBlockDoWhile(size_t id) {
        auto &h = graph_.Node(id);
        if (h.IsCollapsed()) return false;
        // For do-while, header should NOT be conditional — the test
        // is at the bottom.  (If header is conditional, WhileDo should
        // have matched instead.)
        if (h.is_conditional) return false;

        LoopBody *lb = FindLoopForHead(loop_body_storage_, id);
        if (!lb) return false;

        // Must have exactly one tail, and it must be conditional.
        if (lb->tails.size() != 1) return false;
        size_t tail_id = lb->tails[0];
        auto &tail = graph_.Node(tail_id);
        if (tail.IsCollapsed()) return false;
        if (!tail.is_conditional) return false;
        if (tail.succs.size() != 2) return false;
        if (!tail.branch_cond) return false;

        // One of tail's successors must be the header (back-edge).
        bool s0_is_header = (tail.succs[0] == id);
        bool s1_is_header = (tail.succs[1] == id);
        if (!s0_is_header && !s1_is_header) return false;

        // Compute body — filter collapsed nodes (see RuleBlockWhileDo).
        std::vector<size_t> body;
        lb->FindBase(graph_, body);
        if (body.empty()) return false;
        ResolveBodyToRepresentatives(graph_, body, id);

        lb->FindExit(graph_, body);

        // Build do-while: body excludes the tail's branch condition.
        // The tail's stmts (before the branch) are part of the body.
        std::unordered_set<size_t> bodyset(body.begin(), body.end());
        std::vector< SNode * > loop_body = BuildLoopBodySNode(body, id, bodyset);

        // Include header's content in the body (executes each iteration).
        std::vector< SNode * > full_body;
        if (!h.structured.empty() || !h.stmts.empty()) {
            if (!h.structured.empty()) {
                SeqAppend(full_body, h.structured);
            } else {
                AppendStmts(factory_, full_body, h.stmts);
            }
        }
        SeqAppend(full_body, loop_body);

        // CGraph: succs[0] = not-taken (cond false), succs[1] = taken (cond true).
        // If the back-edge to header is on taken (s1), continue cond = branch_cond.
        // If the back-edge is on not-taken (s0), continue cond = !branch_cond.
        clang::Expr *dowhile_cond = s1_is_header
            ? tail.branch_cond
            : NegateExpr(ctx_, tail.branch_cond);
        auto *dowhile_node = factory_.Make<SDoWhile>(
            std::move(full_body), dowhile_cond);

        // Set loop scope labels for break/continue resolution.
        if (!h.original_label.empty())
            dowhile_node->SetHeaderLabel(factory_.Intern(h.original_label));
        size_t exit_id = s1_is_header ? tail.succs[0] : tail.succs[1];
        auto &exit_node = graph_.Node(exit_id);
        if (!exit_node.original_label.empty())
            dowhile_node->SetExitLabel(factory_.Intern(exit_node.original_label));

        SNode *result = dowhile_node;

        // Preserve header label.
        if (!h.original_label.empty()) {
            result = factory_.Make<SLabel>(
                factory_.Intern(h.original_label), result);
        }

        ClearMarks(graph_, body);
        graph_.IdentifyInternal(body, CNode::BlockType::kDoWhile, {result});
        return true;
    }

    // ---------------------------------------------------------------
    // RuleBlockInfLoop — infinite loop (no conditional exit)
    //
    // Pattern: Node H is the head of a detected LoopBody.
    //          The exit_block is kNone (no exit found).
    //          All paths loop back to H.
    // Action:  Create SWhile(true, body) — infinite loop.
    // ---------------------------------------------------------------

    bool CFGStructure::RuleBlockInfLoop(size_t id) {
        auto &h = graph_.Node(id);
        if (h.IsCollapsed()) return false;

        LoopBody *lb = FindLoopForHead(loop_body_storage_, id);
        if (!lb) return false;

        // Compute body — filter collapsed nodes (see RuleBlockWhileDo).
        std::vector<size_t> body;
        lb->FindBase(graph_, body);
        if (body.empty()) return false;
        ResolveBodyToRepresentatives(graph_, body, id);

        // Reject degenerate single-node body (just the header itself)
        // when the header was already structured as a loop by a prior
        // rule.  Re-wrapping a while/do-while in while(1) creates
        // nested degenerate wrappers with unreachable post-loop code.
        // Also handles spurious loops caused by goto-edge markings.
        if (body.size() == 1 && body[0] == id && !h.structured.empty()) {
            ClearMarks(graph_, body);
            return false;
        }

        lb->FindExit(graph_, body);

        // Only match if there is no exit — truly infinite.
        if (lb->exit_block != LoopBody::kNone) {
            ClearMarks(graph_, body);
            return false;
        }

        // Also verify: no body node has an exit edge to outside.
        std::unordered_set<size_t> bodyset(body.begin(), body.end());
        for (size_t nid : body) {
            auto &n = graph_.Node(nid);
            for (size_t si = 0; si < n.succs.size(); ++si) {
                if (!n.IsBackEdge(si) && !n.IsGotoOut(si)
                    && bodyset.count(n.succs[si]) == 0)
                {
                    // Has an exit edge — not truly infinite.
                    ClearMarks(graph_, body);
                    return false;
                }
            }
        }

        // Build infinite loop: while(1) { body }
        // Pass nullptr as the condition — the emitter synthesizes a
        // true literal (IntegerLiteral 1) for null SWhile conditions.
        std::vector< SNode * > loop_body = BuildLoopBodySNode(body, id, bodyset);

        // Include header content in body.
        std::vector< SNode * > full_body;
        if (!h.structured.empty() || !h.stmts.empty()) {
            if (!h.structured.empty()) {
                SeqAppend(full_body, h.structured);
            } else {
                AppendStmts(factory_, full_body, h.stmts);
            }
        }
        SeqAppend(full_body, loop_body);

        auto *inf_while = factory_.Make<SWhile>(
            nullptr, std::move(full_body));

        // Set header label for continue resolution (no exit label for inf loops).
        if (!h.original_label.empty())
            inf_while->SetHeaderLabel(factory_.Intern(h.original_label));

        SNode *result = inf_while;
        if (!h.original_label.empty()) {
            result = factory_.Make<SLabel>(
                factory_.Intern(h.original_label), result);
        }

        ClearMarks(graph_, body);
        graph_.IdentifyInternal(body, CNode::BlockType::kWhile, {result});
        return true;
    }

    // ---------------------------------------------------------------
    // RuleBlockSwitch — switch statement
    //
    // Pattern: Node A has IsSwitchOut() (non-empty switch_cases or
    //          >2 successors) and a branch_cond (discriminant).
    // Action:  Create SSwitch with SCase entries from switch_cases
    //          metadata.  Case bodies are built from target blocks.
    //          Collapse A + case body nodes via IdentifyInternal.
    // ---------------------------------------------------------------

    bool CFGStructure::RuleBlockSwitch(size_t id) {
        auto &a = graph_.Node(id);
        if (a.IsCollapsed()) return false;
        if (!a.IsSwitchOut()) return false;
        if (a.switch_cases.empty()) return false;
        if (!a.branch_cond) return false;

        // Determine discriminant type and width for case literals.
        auto case_type = a.branch_cond->getType();
        if (case_type->isEnumeralType()) {
            case_type = case_type->castAs<clang::EnumType>()
                ->getDecl()->getIntegerType();
        }
        unsigned case_width = ctx_.getIntWidth(case_type);

        // Collect the set of nodes to collapse: start with A.
        std::vector<size_t> collapse_ids;
        collapse_ids.push_back(id);

        // Build the SSwitch.
        auto *sw = factory_.Make<SSwitch>(a.branch_cond);

        for (const auto &sc : a.switch_cases) {
            if (sc.succ_index >= a.succs.size()) {
                LOG(WARNING) << "RuleBlockSwitch: case "
                             << (sc.is_default ? "default" : std::to_string(sc.value))
                             << " succ_index " << sc.succ_index
                             << " out of range (succs=" << a.succs.size() << ")\n";
                continue;
            }
            size_t target = a.succs[sc.succ_index];
            auto &tn = graph_.Node(target);

            // If the target was collapsed by a prior rule, emit as goto
            // rather than silently dropping the case arm.
            if (tn.IsCollapsed()) {
                if (tn.original_label.empty()) {
                    LOG(WARNING) << "RuleBlockSwitch: collapsed target node "
                                 << target << " lost its label — skipping case\n";
                    continue;
                }
                const std::string &lbl = tn.original_label;
                SNode *case_body = factory_.Make<SGoto>(factory_.Intern(lbl));
                if (sc.is_default) {
                    sw->SetDefaultBody(case_body);
                } else {
                    auto *val = clang::IntegerLiteral::Create(
                        ctx_,
                        llvm::APInt(case_width, static_cast<uint64_t>(sc.value), true),
                        case_type, VirtualLoc(ctx_));
                    sw->AddCase(val, case_body);
                }
                continue;
            }

            // Build case body from the target block.
            std::vector< SNode * > case_body;

            // Absorb the target block if all its predecessors come from
            // the switch node (or nodes already in the collapse set).
            // In loops, switch targets often have preds > 1 because the
            // switch and a loop back-edge both feed them — but the
            // back-edge pred was already collapsed into the switch's
            // representative, so all preds effectively come from the
            // switch.  Shared targets (reached by non-switch preds)
            // stay as gotos.
            bool absorb = true;
            for (size_t p : tn.preds) {
                if (p != id && std::find(collapse_ids.begin(),
                        collapse_ids.end(), p) == collapse_ids.end()
                    && !graph_.Node(p).IsCollapsed())
                {
                    absorb = false;
                    break;
                }
            }

            if (absorb) {
                // Pure dispatcher: target has no stmts, just a
                // conditional branch to two successors.  Instead of
                // emitting the raw terminal (if/goto), build an
                // SIfThenElse that absorbs reachable successors,
                // eliminating gotos where possible.
                bool pure_dispatcher = tn.stmts.empty()
                    && tn.structured.empty() && tn.terminal
                    && tn.is_conditional && tn.succs.size() == 2
                    && tn.branch_cond;

                if (pure_dispatcher) {
                    // CGraph: succs[0] = not-taken, succs[1] = taken.
                    size_t f_id = tn.succs[0];
                    size_t t_id = tn.succs[1];

                    // Helper: are all of sid's active predecessors
                    // the dispatcher (target) or already in collapse_ids?
                    auto preds_from_switch = [&](size_t sid) -> bool {
                        auto &sn = graph_.Node(sid);
                        if (sn.IsCollapsed()) return false;
                        for (size_t p : sn.preds) {
                            if (p == target) continue;
                            if (std::find(collapse_ids.begin(),
                                    collapse_ids.end(), p)
                                != collapse_ids.end()) continue;
                            if (graph_.Node(p).IsCollapsed()) continue;
                            return false;
                        }
                        return true;
                    };

                    // Would absorbing sid orphan any of its successors?
                    // A successor is orphaned if its only active
                    // predecessors are sid or nodes already being
                    // collapsed — after collapse it would have no
                    // entry path and no label to reach it.
                    auto would_orphan = [&](size_t sid) -> bool {
                        auto &sn = graph_.Node(sid);
                        for (size_t succ_id : sn.succs) {
                            auto &succ = graph_.Node(succ_id);
                            if (succ.IsCollapsed()) continue;
                            // Already being absorbed — not orphaned.
                            if (std::find(collapse_ids.begin(),
                                    collapse_ids.end(), succ_id)
                                != collapse_ids.end()) continue;
                            bool all_preds_gone = true;
                            for (size_t p : succ.preds) {
                                if (p == sid) continue;
                                if (graph_.Node(p).IsCollapsed()) continue;
                                if (std::find(collapse_ids.begin(),
                                        collapse_ids.end(), p)
                                    != collapse_ids.end()) continue;
                                // Has a live predecessor outside the
                                // collapse set — won't be orphaned.
                                all_preds_gone = false;
                                break;
                            }
                            if (all_preds_gone) return true;
                        }
                        return false;
                    };

                    // Can we safely absorb sid?  Must have all preds
                    // from the switch AND must not orphan its successors
                    // (partial absorption creates dangling dead code).
                    auto can_absorb_succ = [&](size_t sid) -> bool {
                        return preds_from_switch(sid) && !would_orphan(sid);
                    };

                    // Build then/else bodies: absorb or emit goto.
                    auto build_branch =
                        [&](size_t sid) -> std::vector< SNode * > {
                        if (can_absorb_succ(sid)) {
                            auto &sn = graph_.Node(sid);
                            bool nested_disp = sn.stmts.empty()
                                && sn.structured.empty() && sn.terminal;
                            std::vector< SNode * > body = BuildLeafSNode(sid,
                                /*include_terminal=*/nested_disp);
                            if (std::find(collapse_ids.begin(),
                                    collapse_ids.end(), sid)
                                == collapse_ids.end())
                                collapse_ids.push_back(sid);
                            return body;
                        }
                        auto &sn = graph_.Node(sid);
                        if (!sn.original_label.empty())
                            return { factory_.Make<SGoto>(
                                factory_.Intern(sn.original_label)) };
                        return {};
                    };

                    std::vector< SNode * > then_body = build_branch(t_id);
                    std::vector< SNode * > else_body = build_branch(f_id);
                    case_body = { factory_.Make<SIfThenElse>(
                        tn.branch_cond, then_body, else_body) };

                    // Wrap with dispatcher's label if present.
                    if (!tn.original_label.empty()) {
                        case_body = { factory_.Make<SLabel>(
                            factory_.Intern(tn.original_label),
                            case_body) };
                    }
                } else {
                    // Keep terminal if the target has a successor outside
                    // the collapse set — the goto/if-goto is needed to
                    // maintain the edge to the non-absorbed node (e.g. a
                    // shared guard block with multiple predecessors).
                    bool has_external_succ = false;
                    for (size_t s : tn.succs) {
                        if (graph_.Node(s).IsCollapsed()) continue;
                        if (std::find(collapse_ids.begin(),
                                collapse_ids.end(), s)
                            == collapse_ids.end() && s != id) {
                            has_external_succ = true;
                            break;
                        }
                    }
                    case_body = BuildLeafSNode(target,
                        /*include_terminal=*/has_external_succ);
                }

                // Track for collapse.
                if (std::find(collapse_ids.begin(), collapse_ids.end(), target)
                    == collapse_ids.end())
                {
                    collapse_ids.push_back(target);
                }
            } else {
                // Shared target or fallthrough — emit goto.
                if (tn.original_label.empty()) {
                    LOG(WARNING) << "RuleBlockSwitch: target node "
                                 << target << " missing label for goto"
                                 << " — emitting empty case body\n";
                    case_body = {};
                } else {
                    const std::string &lbl = tn.original_label;
                    case_body = { factory_.Make<SGoto>(factory_.Intern(lbl)) };
                }
            }

            if (sc.is_default) {
                sw->SetDefaultBody(case_body);
            } else {
                auto *val = clang::IntegerLiteral::Create(
                    ctx_,
                    llvm::APInt(case_width, static_cast<uint64_t>(sc.value), true),
                    case_type, VirtualLoc(ctx_));
                sw->AddCase(val, case_body);
            }
        }

        // A's pre-switch stmts go before the switch.
        std::vector< SNode * > result;
        AppendStmts(factory_, result, a.stmts);
        result.push_back(sw);

        // Preserve A's label.
        if (!a.original_label.empty()) {
            result = { factory_.Make<SLabel>(
                factory_.Intern(a.original_label), result) };
        }

        graph_.IdentifyInternal(
            collapse_ids, CNode::BlockType::kSwitch, std::move(result));
        return true;
    }

    // ---------------------------------------------------------------
    // SelectAndMarkGotoEdge — use TraceDAG to find the least-disruptive
    // edge and mark it as a goto.
    //
    // This is the fallback when no structural rule can fire.  The
    // selected edge is marked kGoto so structural rules can skip it
    // and RuleBlockCat won't merge across it.  The edge is NOT removed
    // from the graph — it stays as a goto in the emitted output.
    //
    // Returns true if an edge was selected.
    // ---------------------------------------------------------------

    bool CFGStructure::SelectAndMarkGotoEdge() {
        auto active = graph_.ActiveIds();
        if (active.empty()) return false;

        // Use TraceDAG to identify likely-goto edges.
        likely_goto_.clear();
        TraceDAG dag(likely_goto_);

        // Add all active nodes with outgoing edges as roots.
        for (size_t nid : active) {
            auto &n = graph_.Node(nid);
            if (n.succs.empty()) continue;
            dag.AddRoot(nid);
        }

        if (likely_goto_.empty()) {
            dag.Initialize();
            dag.PushBranches(graph_);
        }

        if (likely_goto_.empty()) {
            // TraceDAG couldn't find any candidate.  Fall back to
            // scored selection: prefer edges that don't target the
            // immediate post-dominator (preserve merge structure),
            // penalize intra-loop edges, and prefer targets with
            // more predecessors (merge points that already need labels).
            constexpr size_t kNone = CNode::kNone;

            auto is_intra_loop = [&](size_t src, size_t dest) -> bool {
                for (auto *lb : loop_order_) {
                    if (graph_.Node(lb->head).IsCollapsed()) continue;
                    std::vector<size_t> body;
                    lb->FindBase(graph_, body);
                    bool src_in = false, dest_in = false;
                    for (size_t b : body) {
                        if (b == src) src_in = true;
                        if (b == dest) dest_in = true;
                    }
                    if (src_in && dest_in) return true;
                }
                return false;
            };

            size_t best_src = kNone;
            size_t best_idx = 0;
            int best_score = -1;

            for (size_t nid : active) {
                auto &n = graph_.Node(nid);
                if (n.succs.size() <= 1) continue;
                size_t ipd = (nid < ipdom_.size()) ? ipdom_[nid] : kNone;
                for (size_t i = 0; i < n.succs.size(); ++i) {
                    if (n.IsGotoOut(i) || n.IsBackEdge(i)) continue;
                    size_t dest = n.succs[i];
                    int score = 0;
                    if (dest != ipd) score += 10;
                    if (is_intra_loop(nid, dest)) score -= 20;
                    score += static_cast<int>(graph_.Node(dest).preds.size());
                    if (score > best_score) {
                        best_score = score;
                        best_src = nid;
                        best_idx = i;
                    }
                }
            }
            if (best_src != kNone) {
                graph_.Node(best_src).SetGoto(best_idx);
                return true;
            }
            return false;
        }

        // Select from TraceDAG candidates.  Prefer edges that don't
        // target the source's immediate post-dominator.
        {
            constexpr size_t kNone = CNode::kNone;
            FloatingEdge *best = nullptr;
            int best_score = -1;
            for (auto &fe : likely_goto_) {
                auto [s, ei] = fe.GetCurrentEdge(graph_);
                if (s == kNone) continue;
                int score = 0;
                size_t dest = graph_.Node(s).succs[ei];
                size_t ipd = (s < ipdom_.size()) ? ipdom_[s] : kNone;
                if (dest != ipd) score += 10;
                if (score > best_score) {
                    best_score = score;
                    best = &fe;
                }
            }
            if (best) {
                auto [s, ei] = best->GetCurrentEdge(graph_);
                graph_.Node(s).SetGoto(ei);
                return true;
            }
        }
        return false;
    }

    // ---------------------------------------------------------------
    // BuildLeafSNode — spill a CNode's stmts into SStmt siblings,
    //                  optionally wrapped in an SLabel if the node
    //                  has a label.
    // ---------------------------------------------------------------

    std::vector< SNode * > CFGStructure::BuildLeafSNode(
        size_t id, bool include_terminal) {
        auto &node = graph_.Node(id);

        // If this node was already structured (e.g., by a prior rule),
        // return its existing structured sequence.
        if (!node.structured.empty()) return node.structured;

        // Terminal (goto/if-goto) is appended so the emitter can
        // reconstruct control flow for unstructured remainders.
        // Skipped for non-tail nodes in a sequential merge where the
        // edge is absorbed — the terminal would be a dead goto.
        const bool has_content = !node.stmts.empty()
            || (include_terminal && node.terminal);

        // An empty unlabeled leaf contributes nothing.
        if (!has_content && node.original_label.empty()) {
            return {};
        }

        // An empty labeled leaf keeps its SLabel (it is still a goto
        // target) but with an empty body, rendered as `label: ;`.
        if (!has_content) {
            return { factory_.Make<SLabel>(
                factory_.Intern(node.original_label),
                std::vector< SNode * >{}) };
        }

        std::vector< SNode * > body;
        AppendStmts(factory_, body, node.stmts);
        if (include_terminal && node.terminal) {
            body.push_back(factory_.Make<SStmt>(node.terminal));
        }

        if (!node.original_label.empty()) {
            return { factory_.Make<SLabel>(
                factory_.Intern(node.original_label), std::move(body)) };
        }

        return body;
    }

    // ---------------------------------------------------------------
    // BuildBodySNode — sequence of leaf SNodes from a list of node ids.
    // ---------------------------------------------------------------

    std::vector< SNode * > CFGStructure::BuildBodySNode(
        const std::vector<size_t> &ids) {
        std::vector<SNode *> children;
        for (size_t nid : ids) SeqAppend(children, BuildLeafSNode(nid));
        return factory_.MakeSeq(std::move(children));
    }

    // ---------------------------------------------------------------
    // InlineResidualGotos — post-structuring cleanup.  Replaces an
    // SGoto whose target SLabel is a sibling referenced only by that
    // goto with the label's body, and removes the label.
    // ---------------------------------------------------------------

    namespace {

        // Try to inline gotos in a single sequence.  Returns true if changed.
        bool InlineGotosInSeq(
            std::vector< SNode * > &seq, SNodeFactory & /*factory*/,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            bool changed = false;

            // Build index: label name → position in this sequence.
            std::unordered_map<std::string_view, size_t> label_pos;
            for (size_t i = 0; i < seq.size(); ++i) {
                if (auto *lbl = seq[i]->dyn_cast<SLabel>()) {
                    label_pos[lbl->Name()] = i;
                }
            }

            // Scan children for SGoto nodes that can be inlined.
            // Only inline when the label is the immediate next sibling
            // (forward goto, no skipped code).  This prevents changing
            // control-flow semantics by executing code the goto would
            // have jumped over.  (In the spilled model empty placeholder
            // blocks no longer exist, so immediate adjacency is the only
            // "nothing skipped" case.)
            for (size_t i = 0; i < seq.size(); ++i) {
                auto *g = seq[i]->dyn_cast<SGoto>();
                if (!g) continue;

                auto target = g->Target();
                auto lp = label_pos.find(target);
                if (lp == label_pos.end()) continue;

                // Check: label is only referenced by this one goto.
                auto rc = refs.find(target);
                if (rc == refs.end() || rc->second != 1) continue;

                size_t label_idx = lp->second;

                // Backward gotos: skip — inlining would re-order execution.
                if (label_idx <= i) continue;

                // The goto must skip no real code: the label must be
                // the immediate next sibling.
                if (label_idx != i + 1) continue;

                auto *lbl = seq[label_idx]->as<SLabel>();

                // Splice the label's body (a vector) into the goto's
                // slot, dropping the goto and the label node itself.
                std::vector< SNode * > repl = lbl->BodyList();

                seq.erase(seq.begin() + static_cast<ptrdiff_t>(i),
                          seq.begin() + static_cast<ptrdiff_t>(label_idx) + 1);
                seq.insert(seq.begin() + static_cast<ptrdiff_t>(i),
                           repl.begin(), repl.end());

                changed = true;
                // Rebuild label_pos since indices shifted.
                label_pos.clear();
                for (size_t j = 0; j < seq.size(); ++j) {
                    if (auto *l = seq[j]->dyn_cast<SLabel>()) {
                        label_pos[l->Name()] = j;
                    }
                }
            }

            return changed;
        }

    } // anonymous namespace

    // ---------------------------------------------------------------
    // EliminateGotoToNextLabel — SNode post-pass
    //
    // Walk a sibling sequence.  For each child followed by an SLabel,
    // chase through nesting (SLabel body → last child → SStmt) to find
    // the deepest trailing stmt.  If it's a goto (SGoto or
    // clang::GotoStmt) targeting the next label, or an IfStmt with one
    // arm being such a goto, eliminate it.
    // ---------------------------------------------------------------

    namespace {

        /// Chase through SLabel/SStmt to find the deepest trailing
        /// SNode or clang::Stmt.  Returns {leaf_snode, clang_stmt_or_null}.
        /// The leaf_snode is the SNode containing the trailing stmt.
        struct TrailingInfo {
            SNode *container = nullptr;  // innermost SNode (SStmt, SGoto, etc.)
            clang::Stmt *stmt = nullptr; // if container is SStmt, its stmt
        };

        TrailingInfo DeepTrailingSNode(SNode *node) {
            if (!node) return {};
            if (auto *lbl = node->dyn_cast<SLabel>()) {
                auto &body = lbl->BodyList();
                if (body.empty()) return {};
                return DeepTrailingSNode(body.back());
            }
            if (auto *st = node->dyn_cast<SStmt>()) {
                return {st, st->Stmt()};
            }
            // SGoto, SIfThenElse, etc. — the node itself is the trailing
            return {node, nullptr};
        }

        /// Get goto target name from an SGoto SNode.
        std::string_view SNodeGotoTarget(SNode *node) {
            if (auto *g = node->dyn_cast<SGoto>()) return g->Target();
            return {};
        }

        /// Get goto target name from a clang::GotoStmt.
        std::string ClangGotoTarget(clang::Stmt *s) {
            if (auto *gs = llvm::dyn_cast_or_null<clang::GotoStmt>(s))
                return gs->getLabel()->getName().str();
            return {};
        }

        // Forward declaration — defined in InlineCrossScopeSingleRef namespace.
        bool SNodeAlwaysTerminates(SNode *node);

        bool EliminateInSeq(std::vector< SNode * > &children,
                            SNodeFactory &factory,
                            clang::ASTContext &ctx) {
            bool changed = false;
            bool local_changed = true;
            // Cap restarts to avoid O(N²) on large sequences (e.g., 100+
            // uncollapsed leaf nodes from a partially-structured graph).
            int restart_budget = 50;

            while (local_changed && restart_budget-- > 0) {
                local_changed = false;
                for (size_t i = 0; i + 1 < children.size(); ++i) {
                    // Find the next SLabel sibling, possibly skipping
                    // dead code after a terminating child.
                    SLabel *nxt_label = nullptr;
                    for (size_t k = i + 1; k < children.size() && !nxt_label; ++k) {
                        nxt_label = children[k]->dyn_cast<SLabel>();
                        if (nxt_label) break;
                        // Only skip over siblings that are unreachable
                        // (preceded by a terminating node).
                        if (!SNodeAlwaysTerminates(children[k - 1]))
                            break;
                    }
                    if (!nxt_label) continue;
                    auto next_name = nxt_label->Name();

                    auto info = DeepTrailingSNode(children[i]);
                    if (!info.container) continue;

                    // --- Check SGoto SNode ---
                    auto snode_tgt = SNodeGotoTarget(info.container);
                    if (!snode_tgt.empty() && snode_tgt == next_name) {
                        // Remove the trailing SGoto.  Direct child →
                        // erase from this sequence; otherwise chase the
                        // SLabel body-vectors down to its slot.
                        if (info.container == children[i]) {
                            children.erase(
                                children.begin() + static_cast<ptrdiff_t>(i));
                        } else {
                            std::function<bool(SNode *)> remove_trailing;
                            remove_trailing = [&](SNode *n) -> bool {
                                auto *l = n->dyn_cast<SLabel>();
                                if (!l) return false;
                                auto &body = l->BodyList();
                                if (body.empty()) return false;
                                if (body.back() == info.container) {
                                    body.pop_back();
                                    return true;
                                }
                                return remove_trailing(body.back());
                            };
                            remove_trailing(children[i]);
                        }
                        local_changed = true; changed = true;
                        break;
                    }

                    // --- Check clang::GotoStmt / IfStmt held by an SStmt ---
                    if (info.stmt) {
                        auto *st = info.container->dyn_cast<SStmt>();

                        // Erase the trailing SStmt container, whether it
                        // is a direct child or nested at the tail of an
                        // SLabel body chain.
                        auto erase_container = [&]() {
                            if (info.container == children[i]) {
                                children.erase(
                                    children.begin()
                                    + static_cast<ptrdiff_t>(i));
                                return;
                            }
                            std::function<bool(SNode *)> rec;
                            rec = [&](SNode *n) -> bool {
                                auto *l = n->dyn_cast<SLabel>();
                                if (!l) return false;
                                auto &body = l->BodyList();
                                if (body.empty()) return false;
                                if (body.back() == info.container) {
                                    body.pop_back();
                                    return true;
                                }
                                return rec(body.back());
                            };
                            rec(children[i]);
                        };

                        auto clang_tgt = ClangGotoTarget(info.stmt);
                        if (st && !clang_tgt.empty() && clang_tgt == next_name) {
                            // goto L; L: → drop the trailing goto SStmt
                            erase_container();
                            local_changed = true; changed = true;
                            break;
                        }

                        // --- Check clang::IfStmt with goto arm ---
                        if (auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(info.stmt)) {
                            auto else_tgt = ClangGotoTarget(ifs->getElse());
                            auto then_tgt = ClangGotoTarget(ifs->getThen());

                            if (st && !else_tgt.empty() && else_tgt == next_name) {
                                // else goto L; L: → drop else arm
                                auto loc = ifs->getIfLoc();
                                auto *new_if = clang::IfStmt::Create(
                                    ctx, loc, clang::IfStatementKind::Ordinary,
                                    nullptr, nullptr,
                                    ifs->getCond(), loc, loc,
                                    ifs->getThen(), loc, nullptr);
                                st->SetStmt(new_if);
                                local_changed = true; changed = true;
                                break;
                            }

                            if (st && !then_tgt.empty() && then_tgt == next_name
                                && !ifs->getElse()) {
                                // if(c) goto L; L: → nop (remove the if-stmt)
                                erase_container();
                                local_changed = true; changed = true;
                                break;
                            }

                            if (st && !then_tgt.empty() && then_tgt == next_name
                                && ifs->getElse()) {
                                // if(c) goto L; else S; L: → if(!c) S
                                auto *neg = NegateExpr(ctx, ifs->getCond());
                                auto loc = ifs->getIfLoc();
                                auto *new_if = clang::IfStmt::Create(
                                    ctx, loc, clang::IfStatementKind::Ordinary,
                                    nullptr, nullptr,
                                    neg, loc, loc,
                                    ifs->getElse(), loc, nullptr);
                                st->SetStmt(new_if);
                                local_changed = true; changed = true;
                                break;
                            }
                        }
                    }

                    // --- Check SIfThenElse with SGoto arm ---
                    if (auto *ite = info.container->dyn_cast<SIfThenElse>()) {
                        auto else_tgt = ite->ElseBranch()
                            ? SNodeGotoTarget(ite->ElseBranch()) : std::string_view{};
                        auto then_tgt = ite->ThenBranch()
                            ? SNodeGotoTarget(ite->ThenBranch()) : std::string_view{};

                        if (!else_tgt.empty() && else_tgt == next_name) {
                            ite->SetElseBranch(nullptr);
                            local_changed = true; changed = true;
                            break;
                        }
                        if (!then_tgt.empty() && then_tgt == next_name
                            && !ite->ElseBranch()) {
                            // if(c) goto L; L: → nop (remove the if-then-goto)
                            if (info.container == children[i]) {
                                children.erase(
                                    children.begin()
                                    + static_cast<ptrdiff_t>(i));
                            } else {
                                // Nested — chase the SLabel body-vectors
                                // and remove the trailing SIfThenElse.
                                std::function<bool(SNode *)> remove_trailing;
                                remove_trailing = [&](SNode *n) -> bool {
                                    auto *l = n->dyn_cast<SLabel>();
                                    if (!l) return false;
                                    auto &body = l->BodyList();
                                    if (body.empty()) return false;
                                    if (body.back() == info.container) {
                                        body.pop_back();
                                        return true;
                                    }
                                    return remove_trailing(body.back());
                                };
                                remove_trailing(children[i]);
                            }
                            local_changed = true; changed = true;
                            break;
                        }
                        if (!then_tgt.empty() && then_tgt == next_name
                            && ite->ElseBranch() && ite->Cond()) {
                            // if(c) goto L; else { S... }; L: → if(!c) { S... }
                            // The whole else list becomes the new then list —
                            // ElseBranch() alone would drop all but the first
                            // sibling in the spilled (SStmt) model.
                            auto *neg = NegateExpr(ctx, ite->Cond());
                            auto *new_ite = factory.Make<SIfThenElse>(
                                neg, ite->ElseList(), std::vector< SNode * >{});
                            // Replace in parent
                            if (info.container == children[i]) {
                                children[i] = new_ite;
                            }
                            // TODO: handle deeply nested SIfThenElse
                            local_changed = true; changed = true;
                            break;
                        }
                    }
                }
            }
            return changed;
        }

    } // anonymous namespace (EliminateGotoToNextLabel helpers)

    bool EliminateGotoToNextLabel(std::vector< SNode * > &root,
                                  SNodeFactory &factory,
                                  clang::ASTContext &ctx) {
        return ForEachSeqPostOrder(
            root, [&](std::vector< SNode * > &seq) {
                return EliminateInSeq(seq, factory, ctx);
            });
    }

    bool InlineResidualGotos(std::vector< SNode * > &root, SNodeFactory &factory) {
        // Count all goto references globally.  Keys are string_views
        // into interned strings owned by SNodeFactory (stable lifetime).
        std::unordered_map<std::string_view, int> refs;
        CountGotoRefs(root, refs);

        // Iteratively inline until no more changes (inlining may expose
        // new single-ref gotos).  Bound by the initial goto count — each
        // pass inlines at least one, so we can't need more passes than
        // there are gotos.
        bool any_changed = false;
        size_t max_passes = std::min(refs.size() + 1, size_t{20});
        for (size_t pass = 0; pass < max_passes; ++pass) {
            bool did = ForEachSeqPostOrder(
                root, [&](std::vector< SNode * > &seq) {
                    return InlineGotosInSeq(seq, factory, refs);
                });
            if (!did)
                break;
            any_changed = true;
            // Recount after mutations.
            refs.clear();
            CountGotoRefs(root, refs);
        }
        return any_changed;
    }

    // ---------------------------------------------------------------
    // InlineCrossScopeSingleRef — cross-scope single-ref label inliner
    //
    // Extends InlineResidualGotos to the case where the goto and its
    // target label live in *different* sibling sequences.  When a label has
    // exactly one goto reference, its body always terminates, and no
    // fallthrough can reach the label, the body is moved (not cloned)
    // into the goto's slot and the label node is deleted.
    // ---------------------------------------------------------------

    namespace {

        /// Check if a clang::Stmt always terminates control flow
        /// (return, break, continue, goto, or an IfStmt / CompoundStmt
        /// whose every path ends in one of those).
        bool ClangStmtIsTerminator(clang::Stmt *s) {
            if (!s) return false;
            if (llvm::isa< clang::ReturnStmt >(s)
                || llvm::isa< clang::BreakStmt >(s)
                || llvm::isa< clang::ContinueStmt >(s)
                || llvm::isa< clang::GotoStmt >(s))
                return true;
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                if (cs->body_empty()) return false;
                return ClangStmtIsTerminator(cs->body_back());
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
                if (!ifs->getThen() || !ifs->getElse()) return false;
                return ClangStmtIsTerminator(ifs->getThen())
                    && ClangStmtIsTerminator(ifs->getElse());
            }
            return false;
        }

        bool SNodeAlwaysTerminates(SNode *node);

        /// A sequence always terminates iff its last element does.
        bool SeqAlwaysTerminates(const std::vector< SNode * > &seq) {
            if (seq.empty()) return false;
            return SNodeAlwaysTerminates(seq.back());
        }

        /// Return true if this SNode always terminates control flow —
        /// every execution path through the node exits via return,
        /// break, continue, throw, or an unconditional goto.  Such a
        /// node never "falls off the end".
        bool SNodeAlwaysTerminates(SNode *node) {
            if (!node) return false;

            if (node->dyn_cast<SReturn>()) return true;
            if (node->dyn_cast<SBreak>()) return true;
            if (node->dyn_cast<SContinue>()) return true;
            if (node->dyn_cast<SGoto>()) return true;

            if (auto *st = node->dyn_cast<SStmt>()) {
                return ClangStmtIsTerminator(st->Stmt());
            }
            if (auto *lbl = node->dyn_cast<SLabel>()) {
                return SeqAlwaysTerminates(lbl->BodyList());
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                // Both branches must terminate (else fallthrough
                // when one arm is missing).
                if (ite->ThenList().empty() || ite->ElseList().empty())
                    return false;
                return SeqAlwaysTerminates(ite->ThenList())
                    && SeqAlwaysTerminates(ite->ElseList());
            }
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                // Every case + default must terminate, and default
                // must be present (otherwise unmatched values fall
                // out of the switch).
                if (sw->DefaultBodyList().empty()) return false;
                if (!SeqAlwaysTerminates(sw->DefaultBodyList()))
                    return false;
                for (auto &c : sw->Cases()) {
                    if (!SeqAlwaysTerminates(c.body_list))
                        return false;
                }
                return true;
            }
            // Loops may iterate zero times → conservative: not
            // guaranteed to terminate via fallthrough.
            return false;
        }

        /// Recursively check whether a subtree contains any SLabel.
        /// Moving a subtree containing a label would displace the
        /// label and invalidate any goto references still pointing at
        /// the original lexical position.
        bool SubtreeHasLabel(SNode *node) {
            if (!node) return false;
            if (node->dyn_cast<SLabel>()) return true;

            // An SStmt may hold a clang::LabelStmt — check too.
            // (SStmt has no SNode children, so we return early.)
            if (auto *st = node->dyn_cast<SStmt>()) {
                return st->Stmt()
                    && llvm::isa<clang::LabelStmt>(st->Stmt());
            }

            // Uniform recursion via the visitor API: any descendant
            // SLabel triggers the early return above on the next call.
            bool found = false;
            node->for_each_child([&](SNode *c) {
                if (!found && SubtreeHasLabel(c)) found = true;
            });
            return found;
        }

        /// Locate an SLabel by name, returning the sibling sequence it
        /// lives in and its index.  Only labels in a genuine sibling
        /// sequence qualify — this is load-bearing: inlining checks need
        /// a real sibling list, and moving a body containing break/
        /// continue across loop boundaries would change its target loop.
        struct LabelLoc {
            std::vector< SNode * > *parent = nullptr;
            size_t idx = 0;
        };

        bool FindLabel(std::vector< SNode * > &seq, std::string_view name,
                       LabelLoc &out) {
            for (size_t i = 0; i < seq.size(); ++i) {
                if (auto *lbl = seq[i]->dyn_cast<SLabel>()) {
                    if (lbl->Name() == name) {
                        out.parent = &seq;
                        out.idx = i;
                        return true;
                    }
                }
            }
            bool found = false;
            for (SNode *c : seq) {
                ForEachBodyList(c, [&](std::vector< SNode * > &body) {
                    if (!found && FindLabel(body, name, out)) found = true;
                });
            }
            return found;
        }

        /// Scan one sequence for an SGoto whose target label can be
        /// inlined (single ref, terminating label-free body, no
        /// fallthrough into the label).  On the first match the
        /// label's body-vector is spliced into the goto's slot and
        /// the label node removed.  Returns true on success.
        bool CrossScopeInlineInSeq(
            std::vector< SNode * > &seq,
            std::vector< SNode * > &root,
            std::unordered_map<std::string_view, int> &refs
        ) {
            for (size_t i = 0; i < seq.size(); ++i) {
                auto *g = seq[i]->dyn_cast<SGoto>();
                if (!g) continue;

                auto target = g->Target();
                auto it = refs.find(target);
                if (it == refs.end() || it->second != 1) continue;

                LabelLoc loc;
                if (!FindLabel(root, target, loc)) continue;

                auto *lbl = (*loc.parent)[loc.idx]->as<SLabel>();
                std::vector< SNode * > &body = lbl->BodyList();
                if (!SeqAlwaysTerminates(body)) continue;

                bool has_label = false;
                for (SNode *b : body)
                    if (SubtreeHasLabel(b)) { has_label = true; break; }
                if (has_label) continue;

                // No fallthrough may reach the label in its original
                // position.  Require a preceding terminating sibling.
                if (loc.idx == 0) continue;
                if (!SNodeAlwaysTerminates((*loc.parent)[loc.idx - 1]))
                    continue;

                // Detach the label first, then re-locate the goto
                // (its index may shift if the label sat in the same
                // sequence at a lower position) and splice the body in.
                std::vector< SNode * > spliced = body;
                loc.parent->erase(
                    loc.parent->begin() + static_cast<ptrdiff_t>(loc.idx));

                size_t gi = seq.size();
                for (size_t k = 0; k < seq.size(); ++k)
                    if (seq[k] == g) { gi = k; break; }
                if (gi == seq.size()) {
                    refs[target] = 0;
                    return true;
                }
                seq.erase(seq.begin() + static_cast<ptrdiff_t>(gi));
                seq.insert(seq.begin() + static_cast<ptrdiff_t>(gi),
                           spliced.begin(), spliced.end());
                refs[target] = 0;
                return true;
            }
            return false;
        }

    } // anonymous namespace

    bool InlineCrossScopeSingleRef(std::vector< SNode * > &root,
                                   SNodeFactory & /*factory*/) {
        std::unordered_map<std::string_view, int> refs;
        CountGotoRefs(root, refs);

        bool any_changed = false;
        // Fixed-point iteration — each pass eliminates at least one
        // goto, so the initial ref count bounds the loop.
        size_t max_passes = std::min(refs.size() + 1, size_t{20});
        for (size_t p = 0; p < max_passes; ++p) {
            // stop_after_change: CrossScopeInlineInSeq erases the target
            // label from whatever sequence holds it — possibly a strict
            // ancestor of the goto's sequence.  Unwinding immediately
            // keeps a suspended ancestor traversal from resuming over
            // the mutated vector (iterator invalidation / UB).
            bool changed = ForEachSeqPostOrder(
                root,
                [&](std::vector< SNode * > &seq) {
                    return CrossScopeInlineInSeq(seq, root, refs);
                },
                /*stop_after_change=*/true);
            if (!changed) break;
            any_changed = true;
            refs.clear();
            CountGotoRefs(root, refs);
        }
        return any_changed;
    }

    // AbsorbFallthroughIntoElse: if SIfThenElse(cond, body, null) is
    // followed by an SLabel with 0 goto refs and the then-body always
    // terminates, move the label body into the else.  Without the
    // terminates guard, both paths reach the label and absorbing it
    // diverts the then-path past the trailing continuation.

    namespace {

        // Forward declaration — defined in InlineCrossScopeSingleRef namespace.
        bool SNodeAlwaysTerminates(SNode *node);

        /// Chase through SLabel nesting to find the deepest trailing
        /// SIfThenElse that has no else branch.
        SIfThenElse *DeepTrailingIfThen(SNode *node) {
            if (!node) return nullptr;
            if (auto *ite = node->dyn_cast<SIfThenElse>())
                return ite->ElseList().empty() ? ite : nullptr;
            if (auto *lbl = node->dyn_cast<SLabel>()) {
                auto &body = lbl->BodyList();
                return body.empty() ? nullptr
                                    : DeepTrailingIfThen(body.back());
            }
            return nullptr;
        }

        bool AbsorbInSeq(std::vector< SNode * > &children,
                         const std::unordered_map<std::string_view, int> &refs) {
            bool changed = false;
            for (size_t i = 0; i + 1 < children.size(); ++i) {
                // Chase into SLabel nesting to find a deeply buried
                // trailing if-then (no else).
                auto *ite = DeepTrailingIfThen(children[i]);
                if (!ite) continue;

                // Next sibling must be an SLabel.
                auto *lbl = children[i + 1]->dyn_cast<SLabel>();
                if (!lbl) continue;

                auto rc = refs.find(lbl->Name());
                if (rc != refs.end() && rc->second > 0) continue;
                // The whole then-sequence must terminate — ThenBranch()
                // only exposes then_[0], so a multi-statement then that
                // ends in a return would otherwise be missed.
                if (ite->ThenList().empty()
                    || !SNodeAlwaysTerminates(ite->ThenList().back()))
                    continue;

                ite->SetElseBranch(lbl->BodyList());

                // Remove the absorbed SLabel.
                children.erase(
                    children.begin() + static_cast<ptrdiff_t>(i) + 1);
                changed = true;
                // Don't break — continue scanning for more opportunities.
            }
            return changed;
        }

    } // anonymous namespace

    bool AbsorbFallthroughIntoElse(std::vector< SNode * > &root,
                                   SNodeFactory & /*factory*/) {
        std::unordered_map<std::string_view, int> refs;
        CountGotoRefs(root, refs);
        return ForEachSeqPostOrder(
            root, [&](std::vector< SNode * > &seq) {
                return AbsorbInSeq(seq, refs);
            });
    }

    // ---------------------------------------------------------------
    // ScopeifyIfGotos — convert if(cond) goto L; stmts; L: → if(!cond){stmts}
    //
    // Scan siblings for: children[i] = SIfThenElse(cond, SGoto("L"), null)
    // followed by intermediate children, then children[j] = SLabel("L", body).
    // When label "L" has exactly one goto reference and no intermediate
    // children contain SLabel nodes, negate the condition, wrap the
    // intermediates in a scoped if(!cond){...}, and inline the label body.
    // ---------------------------------------------------------------

    namespace {

        bool ScopeifyHasLabel(SNode *node) {
            if (!node) return false;
            if (node->dyn_cast<SLabel>()) return true;
            // Note: this scan deliberately does NOT inspect clang::LabelStmt
            // inside an SStmt (unlike SubtreeHasLabel).  Preserving that
            // narrower scope; the SStmt leaf has no SNode children so
            // the visitor simply returns false for it.
            bool found = false;
            node->for_each_child([&](SNode *c) {
                if (!found && ScopeifyHasLabel(c)) found = true;
            });
            return found;
        }



        bool ScopeifyInSeq(std::vector< SNode * > &children,
                           SNodeFactory &factory,
                           clang::ASTContext &ctx,
                           const std::unordered_map<std::string_view, int> &refs) {
            bool any_change = false;
            bool changed = true;

            while (changed) {
                changed = false;
                for (size_t i = 0; i < children.size(); ++i) {
                    // Match: SIfThenElse(cond, SGoto("L"), null)
                    auto *ite = children[i]->dyn_cast<SIfThenElse>();
                    if (!ite || !ite->Cond() || ite->ElseBranch()) continue;
                    auto *goto_node = ite->ThenBranch()
                        ? ite->ThenBranch()->dyn_cast<SGoto>() : nullptr;
                    if (!goto_node) continue;

                    auto target = goto_node->Target();

                    // Find target SLabel in same sequence (forward only)
                    size_t label_idx = children.size();
                    for (size_t j = i + 1; j < children.size(); ++j) {
                        if (auto *lbl = children[j]->dyn_cast<SLabel>()) {
                            if (lbl->Name() == target) {
                                label_idx = j;
                                break;
                            }
                        }
                    }
                    if (label_idx >= children.size()) continue;
                    // Adjacent — handled by EliminateGotoToNextLabel
                    if (label_idx == i + 1) continue;

                    // Single reference only
                    auto rc = refs.find(target);
                    if (rc == refs.end() || rc->second != 1) continue;

                    // No intermediate labels
                    bool has_label = false;
                    for (size_t j = i + 1; j < label_idx; ++j) {
                        if (ScopeifyHasLabel(children[j])) {
                            has_label = true;
                            break;
                        }
                    }
                    if (has_label) continue;

                    // Build scoped body from intermediates.  MakeSeq
                    // normalizes (drops nulls).
                    std::vector<SNode *> scoped_children;
                    scoped_children.reserve(label_idx - i - 1);
                    for (size_t j = i + 1; j < label_idx; ++j)
                        scoped_children.push_back(children[j]);
                    std::vector< SNode * > scoped_body =
                        factory.MakeSeq(std::move(scoped_children));

                    // Negate condition
                    auto *neg = NegateExpr(ctx, ite->Cond());
                    auto *new_if = factory.Make<SIfThenElse>(
                        neg, scoped_body, std::vector< SNode * >{});

                    // Build the replacement: new_if followed by the
                    // label body's children.
                    auto *lbl = children[label_idx]->as<SLabel>();
                    std::vector<SNode *> replacements;
                    replacements.push_back(new_if);
                    for (SNode *c : lbl->BodyList())
                        replacements.push_back(c);

                    // Replace range [i, label_idx+1) with `replacements`.
                    children.erase(
                        children.begin() + static_cast<ptrdiff_t>(i),
                        children.begin()
                            + static_cast<ptrdiff_t>(label_idx) + 1);
                    children.insert(
                        children.begin() + static_cast<ptrdiff_t>(i),
                        replacements.begin(), replacements.end());

                    changed = true;
                    any_change = true;
                    break; // restart scan
                }
            }
            return any_change;
        }

    } // anonymous namespace

    bool ScopeifyIfGotos(std::vector< SNode * > &root, SNodeFactory &factory,
                         clang::ASTContext &ctx) {
        bool any_changed = false;
        // Recount refs after each pass — ScopeifyInSeq removes gotos,
        // which can make previously multi-ref labels single-ref.
        for (int pass = 0; pass < 8; ++pass) {
            std::unordered_map<std::string_view, int> refs;
            CountGotoRefs(root, refs);
            bool did = ForEachSeqPostOrder(
                root, [&](std::vector< SNode * > &seq) {
                    return ScopeifyInSeq(seq, factory, ctx, refs);
                });
            if (!did)
                break;
            any_changed = true;
        }
        return any_changed;
    }

    // ---------------------------------------------------------------
    // RemoveDeadSSeqChildren — strip unreachable siblings.  Bottom-up:
    // when child[i] always terminates, remove siblings [i+1..end) that
    // are not SLabel nodes.  SLabels are preserved — they may be goto
    // targets from other scopes.
    // ---------------------------------------------------------------

    namespace {

        /// Check if an SNode contains any SLabel (directly or nested).
        /// Scans all body slots via the visitor API; over-reporting is
        /// safe for the dead-code removal caller.
        bool ContainsLabel(SNode *node) {
            if (!node) return false;
            if (node->dyn_cast<SLabel>()) return true;
            bool found = false;
            node->for_each_child([&](SNode *c) {
                if (!found && ContainsLabel(c)) found = true;
            });
            return found;
        }

        /// Strict terminator check for dead-code removal.  Unlike a
        /// "tail terminates" check, this only accepts nodes that stay
        /// terminating under post-passes that move content between
        /// siblings:
        ///   - primitive terminators (SReturn/SBreak/SContinue/SGoto)
        ///   - SStmt whose clang::Stmt is itself terminal
        ///   - SLabel wrapping any of the above
        ///   - SIfThenElse where BOTH branches are strong terminators
        /// Sequences / loops / switch are excluded: their internal flow
        /// can be mutated in ways a forward "tail terminates" check
        /// doesn't capture (cf. pb_decode_inner CALL_LOST regression).
        bool IsStrongTerminator(SNode *node) {
            if (!node) return false;
            if (node->dyn_cast<SReturn>()) return true;
            if (node->dyn_cast<SBreak>()) return true;
            if (node->dyn_cast<SContinue>()) return true;
            if (node->dyn_cast<SGoto>()) return true;
            if (auto *st = node->dyn_cast<SStmt>()) {
                return ClangStmtIsTerminator(st->Stmt());
            }
            if (auto *lbl = node->dyn_cast<SLabel>()) {
                auto &body = lbl->BodyList();
                return !body.empty() && IsStrongTerminator(body.back());
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                // Both arms must be strong terminators.  Absorb only
                // targets if-then-without-else, so this shape is stable.
                return !ite->ThenList().empty() && !ite->ElseList().empty()
                    && IsStrongTerminator(ite->ThenList().back())
                    && IsStrongTerminator(ite->ElseList().back());
            }
            // SWhile / SDoWhile / SFor / SSwitch:
            // DO NOT treat as terminators here, even if their tail
            // terminates.  AbsorbFallthroughIntoElse et al. can mutate
            // their internals in ways that invalidate the simple
            // "last child terminates" heuristic for parent-level
            // dead-code analysis.
            return false;
        }

        bool RemoveDeadInSeq(std::vector< SNode * > &children) {
            bool changed = false;
            // Dead statements after a terminator are now dead SStmt
            // siblings — the loop below trims them at the sibling level.

            for (size_t i = 0; i + 1 < children.size(); ++i) {
                if (!IsStrongTerminator(children[i])) continue;
                // children[i] always terminates, so the contiguous run
                // of label-free siblings immediately after it is
                // unreachable.  Stop at the first label-bearing sibling:
                // a label is a goto re-entry point, so it and everything
                // after it can still be reached and must be kept.
                while (i + 1 < children.size()
                       && !ContainsLabel(children[i + 1])) {
                    children.erase(
                        children.begin() + static_cast<ptrdiff_t>(i) + 1);
                    changed = true;
                }
                break; // only process first terminator
            }
            return changed;
        }

    } // anonymous namespace

    bool RemoveDeadSSeqChildren(std::vector< SNode * > &root) {
        return ForEachSeqPostOrder(
            root, [](std::vector< SNode * > &seq) {
                return RemoveDeadInSeq(seq);
            });
    }

    // ---------------------------------------------------------------
    // ConvertGotoToBreakContinue — replace gotos to loop exit/header
    //
    // Walks the SNode tree with a scope stack of enclosing loops.
    // For each SStmt holding a clang::GotoStmt:
    //   - target matches loop ExitLabel → replace with SBreak
    //   - target matches loop HeaderLabel → replace with SContinue
    // Also handles SGoto SNodes from SelectAndMarkGotoEdge.
    // ---------------------------------------------------------------

    namespace {

        struct LoopScope {
            std::string_view exit_label;
            std::string_view header_label;
        };

        /// Convert gotos to break/continue across one sequence and its
        /// nested body-vectors.  The scope stack tracks enclosing loops;
        /// a loop pushes its exit/header labels before its body is
        /// processed.  An SGoto / trailing clang::GotoStmt targeting a
        /// loop label becomes SBreak / SContinue.
        bool ConvertGotosInSeq(
            std::vector< SNode * > &seq, SNodeFactory &factory,
            std::vector<LoopScope> &scopes
        ) {
            bool changed = false;

            // Recurse into each child's body-vectors, pushing a loop
            // scope around loop bodies.
            for (SNode *child : seq) {
                bool is_loop = false;
                std::string_view exit_l, header_l;
                if (auto *w = child->dyn_cast<SWhile>()) {
                    is_loop = true;
                    exit_l = w->ExitLabel();
                    header_l = w->HeaderLabel();
                } else if (auto *dw = child->dyn_cast<SDoWhile>()) {
                    is_loop = true;
                    exit_l = dw->ExitLabel();
                    header_l = dw->HeaderLabel();
                } else if (auto *f = child->dyn_cast<SFor>()) {
                    is_loop = true;
                    exit_l = f->ExitLabel();
                    header_l = f->HeaderLabel();
                }
                if (is_loop) scopes.push_back({exit_l, header_l});
                ForEachBodyList(child, [&](std::vector< SNode * > &body) {
                    if (ConvertGotosInSeq(body, factory, scopes))
                        changed = true;
                });
                if (is_loop) scopes.pop_back();
            }

            // Replace SGoto / trailing-GotoStmt children that target an
            // enclosing loop's exit or header label.
            for (size_t i = 0; i < seq.size(); ++i) {
                if (auto *g = seq[i]->dyn_cast<SGoto>()) {
                    std::string_view target = g->Target();
                    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
                        if (!it->exit_label.empty()
                            && target == it->exit_label) {
                            seq[i] = factory.Make<SBreak>();
                            changed = true;
                            break;
                        }
                        if (!it->header_label.empty()
                            && target == it->header_label) {
                            seq[i] = factory.Make<SContinue>();
                            changed = true;
                            break;
                        }
                    }
                    continue;
                }

                auto *st = seq[i]->dyn_cast<SStmt>();
                if (!st) continue;
                auto *goto_stmt =
                    llvm::dyn_cast_or_null<clang::GotoStmt>(st->Stmt());
                if (!goto_stmt || !goto_stmt->getLabel()) continue;

                std::string_view target = goto_stmt->getLabel()->getName();
                for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
                    if (!it->exit_label.empty() && target == it->exit_label) {
                        seq[i] = factory.Make<SBreak>();
                        changed = true;
                        break;
                    }
                    if (!it->header_label.empty()
                        && target == it->header_label) {
                        seq[i] = factory.Make<SContinue>();
                        changed = true;
                        break;
                    }
                }
            }
            return changed;
        }

    } // anonymous namespace

    bool ConvertGotoToBreakContinue(std::vector< SNode * > &root,
                                    SNodeFactory &factory) {
        std::vector<LoopScope> scopes;
        return ConvertGotosInSeq(root, factory, scopes);
    }

    // ---------------------------------------------------------------
    // ConvertGotoToReturn — replace goto-to-return patterns
    //
    // When an SGoto targets a label whose body is a single SReturn
    // (or a sequence whose last child is SReturn with no other control
    // flow), replace the goto with a cloned SReturn.
    // ---------------------------------------------------------------

    namespace {

        /// True if a clang::Stmt tree contains a CallExpr.
        bool StmtTreeHasCall(const clang::Stmt *s) {
            if (!s) return false;
            if (llvm::isa<clang::CallExpr>(s)) return true;
            for (const auto *c : s->children())
                if (StmtTreeHasCall(c)) return true;
            return false;
        }

        /// True if any stmt in the vector contains a CallExpr.
        bool StmtVecHasCall(const std::vector<clang::Stmt *> &v) {
            for (auto *s : v)
                if (StmtTreeHasCall(s)) return true;
            return false;
        }

        /// Check whether the maximal trailing run of SStmt siblings in
        /// `body` forms a safe, return-terminating block that can be
        /// duplicated at goto sites, collecting its statements into
        /// `out`.  The run must:
        ///   - be non-empty
        ///   - end with clang::ReturnStmt
        ///   - have ≤ max_stmts statements
        ///   - contain no GotoStmt or LabelStmt (no new label references)
        ///   - contain no CallExpr — duplicating a call inflates the
        ///     output with extra call sites that do not exist in the
        ///     input P-Code; such a body stays shared behind its goto
        bool ExtractReturnTail(const std::vector<SNode *> &body,
                               std::vector<clang::Stmt *> &out,
                               size_t max_stmts = 8) {
            out.clear();
            size_t start = body.size();
            while (start > 0 && body[start - 1]->dyn_cast<SStmt>())
                --start;
            if (start == body.size()) return false; // no trailing SStmt run
            for (size_t j = start; j < body.size(); ++j)
                out.push_back(body[j]->as<SStmt>()->Stmt());
            if (out.size() > max_stmts) { out.clear(); return false; }
            if (out.empty()
                || !llvm::isa<clang::ReturnStmt>(out.back())) {
                out.clear();
                return false;
            }
            for (auto *s : out) {
                if (!s || llvm::isa<clang::GotoStmt>(s)
                    || llvm::isa<clang::LabelStmt>(s)
                    || StmtTreeHasCall(s)) {
                    out.clear();
                    return false;
                }
            }
            return true;
        }

        /// Info about a label's position in the SNode tree.  Only the
        /// owning sequence is recorded — the label's index within it is
        /// re-derived on demand, since goto-to-return splicing shifts
        /// sibling positions during the pass.
        struct LabelEntry {
            SLabel *label;
            std::vector<SNode *> *parent_seq;  // body-vector holding the label
        };

        /// Build global label→LabelEntry map.
        void CollectLabels(std::vector<SNode *> &seq,
                           std::unordered_map<std::string_view, LabelEntry> &labels) {
            for (size_t i = 0; i < seq.size(); ++i) {
                if (auto *lbl = seq[i]->dyn_cast<SLabel>())
                    labels[lbl->Name()] = {lbl, &seq};
                ForEachBodyList(seq[i], [&](std::vector<SNode *> &body) {
                    CollectLabels(body, labels);
                });
            }
        }

        /// Collect the fallthrough tail stmts starting from a label's
        /// position in its parent sequence.  Follows consecutive SStmt
        /// siblings and all-SStmt-bodied SLabel siblings, collecting
        /// their stmts.  Returns true if the tail ends with ReturnStmt
        /// and total stmts ≤ max_stmts.  All stmts must be safe (no
        /// GotoStmt/LabelStmt except the label markers).
        bool CollectReturnTail(
            const LabelEntry &entry,
            std::vector<clang::Stmt *> &out,
            size_t max_stmts = 6
        ) {
            if (!entry.parent_seq) return false;
            auto &seq = *entry.parent_seq;

            // Re-derive the label's current index — splicing earlier in
            // this pass may have shifted it.
            size_t start = seq.size();
            for (size_t k = 0; k < seq.size(); ++k)
                if (seq[k] == entry.label) { start = k; break; }
            if (start == seq.size()) return false;

            out.clear();
            auto collect = [&](clang::Stmt *s) -> bool {
                if (llvm::isa<clang::GotoStmt>(s)
                    || llvm::isa<clang::LabelStmt>(s))
                    return false; // unsafe stmt — hard fail
                out.push_back(s);
                return out.size() <= max_stmts;
            };

            for (size_t j = start; j < seq.size(); ++j) {
                auto *child = seq[j];

                if (auto *st = child->dyn_cast<SStmt>()) {
                    if (!collect(st->Stmt())) return false;
                } else if (auto *lbl = child->dyn_cast<SLabel>()) {
                    // Only a label whose body is a non-empty run of
                    // SStmt nodes contributes (matches the prior
                    // single-statement-body restriction).
                    auto &b = lbl->BodyList();
                    bool all_stmt = !b.empty();
                    for (auto *bc : b)
                        if (!bc->dyn_cast<SStmt>()) { all_stmt = false; break; }
                    if (!all_stmt) break;
                    for (auto *bc : b)
                        if (!collect(bc->as<SStmt>()->Stmt())) return false;
                } else {
                    break; // non-label/non-stmt sibling — stop
                }
            }

            // Must end with ReturnStmt.
            if (out.empty() || !llvm::isa<clang::ReturnStmt>(out.back()))
                return false;
            return true;
        }

        // Forward declaration for chain resolution.
        bool ResolveGotoChain(
            std::string_view start_target,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            clang::ASTContext &ctx,
            std::vector<clang::Stmt *> &out,
            size_t max_stmts = 16,
            size_t max_hops = 8);

        /// Build a clang::ReturnStmt from an SReturn.
        clang::ReturnStmt *MakeReturn(clang::ASTContext &ctx, SReturn *sr) {
            return clang::ReturnStmt::Create(ctx, VirtualLoc(ctx), sr->Value(), nullptr);
        }

        bool FlattenTerminatingSeq(
            const std::vector<SNode *> &body, clang::ASTContext &ctx,
            std::vector<clang::Stmt *> &out, size_t max_stmts);

        /// Try to flatten a terminating SNode body into clang::Stmts.
        /// Handles SStmt holding a return, SReturn, SLabel (unwrap),
        /// and SIfThenElse where both arms terminate.
        bool FlattenTerminatingBody(
            SNode *body, clang::ASTContext &ctx,
            std::vector<clang::Stmt *> &out,
            size_t max_stmts
        ) {
            if (!body) return false;

            if (auto *sr = body->dyn_cast<SReturn>()) {
                out.push_back(MakeReturn(ctx, sr));
                return out.size() <= max_stmts;
            }
            // SLabel: unwrap and flatten the inner body sequence.
            if (auto *lbl = body->dyn_cast<SLabel>()) {
                return FlattenTerminatingSeq(
                    lbl->BodyList(), ctx, out, max_stmts);
            }
            if (auto *st = body->dyn_cast<SStmt>()) {
                auto *s = st->Stmt();
                if (!s || !llvm::isa<clang::ReturnStmt>(s)
                    || StmtTreeHasCall(s))
                    return false;
                out.push_back(s);
                return out.size() <= max_stmts;
            }
            // SIfThenElse where both arms terminate.  An empty else
            // falls through, so it cannot be flattened as a terminating
            // body — hard-fail rather than drop the fallthrough path
            // (the else is filled later by AbsorbFallthroughIntoElse).
            if (auto *ite = body->dyn_cast<SIfThenElse>()) {
                if (!ite->Cond() || ite->ThenList().empty()
                    || ite->ElseList().empty())
                    return false;
                std::vector<clang::Stmt *> then_stmts, else_stmts;
                if (!FlattenTerminatingSeq(
                        ite->ThenList(), ctx, then_stmts, max_stmts))
                    return false;
                clang::Stmt *then_body = nullptr;
                if (then_stmts.size() == 1) {
                    then_body = then_stmts[0];
                } else {
                    auto loc = VirtualLoc(ctx);
                    then_body = clang::CompoundStmt::Create(
                        ctx, then_stmts, clang::FPOptionsOverride(),
                        loc, loc);
                }
                clang::Stmt *else_body = nullptr;
                {
                    if (!FlattenTerminatingSeq(
                            ite->ElseList(), ctx, else_stmts, max_stmts))
                        return false;
                    if (else_stmts.size() == 1) {
                        else_body = else_stmts[0];
                    } else {
                        auto loc = VirtualLoc(ctx);
                        else_body = clang::CompoundStmt::Create(
                            ctx, else_stmts, clang::FPOptionsOverride(),
                            loc, loc);
                    }
                }
                auto loc = VirtualLoc(ctx);
                auto *new_if = clang::IfStmt::Create(
                    ctx, loc, clang::IfStatementKind::Ordinary,
                    nullptr, nullptr, ite->Cond(), loc, loc,
                    then_body, loc, else_body);
                out.push_back(new_if);
                return out.size() <= max_stmts;
            }
            return false;
        }

        /// Flatten a terminating sequence: SStmt prefix children
        /// followed by a terminating last element.
        bool FlattenTerminatingSeq(
            const std::vector<SNode *> &body, clang::ASTContext &ctx,
            std::vector<clang::Stmt *> &out, size_t max_stmts
        ) {
            if (body.empty()) return false;
            for (size_t i = 0; i + 1 < body.size(); ++i) {
                auto *child = body[i]->dyn_cast<SStmt>();
                if (!child) return false;
                auto *s = child->Stmt();
                if (llvm::isa<clang::GotoStmt>(s)
                    || llvm::isa<clang::LabelStmt>(s))
                    return false;
                out.push_back(s);
                if (out.size() > max_stmts) return false;
            }
            return FlattenTerminatingBody(body.back(), ctx, out, max_stmts);
        }

        /// Follow a goto chain through labels collecting stmts until
        /// a return-terminating body is reached.  Each link in the
        /// chain is a label whose body ends in a goto to the next
        /// link.  Returns true if a return-terminated sequence was
        /// assembled within the budget.
        ///
        /// Example chain:  goto L1 → L1:{s1; goto L2} → L2:{s2; return v;}
        /// Result:          out = {s1, s2, return v;}
        bool ResolveGotoChain(
            std::string_view start_target,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            clang::ASTContext &ctx,
            std::vector<clang::Stmt *> &out,
            size_t max_stmts,
            size_t max_hops
        ) {
            out.clear();
            auto target = start_target;

            for (size_t hop = 0; hop < max_hops; ++hop) {
                auto it = labels.find(target);
                if (it == labels.end()) return false;

                auto &body = it->second.label->BodyList();
                if (body.empty()) return false;

                // Try direct: body is return-terminating (no goto inside).
                std::vector<clang::Stmt *> direct;
                if (ExtractReturnTail(body, direct, max_stmts)) {
                    for (auto *s : direct) out.push_back(s);
                    return out.size() <= max_stmts && !StmtVecHasCall(out);
                }

                // Try fallthrough tail from label's sequence position.
                std::vector<clang::Stmt *> tail;
                if (CollectReturnTail(it->second, tail)) {
                    for (auto *s : tail) out.push_back(s);
                    return out.size() <= max_stmts;
                }

                // Try flattening complex terminating body (SIfThenElse etc.)
                std::vector<clang::Stmt *> flat;
                if (FlattenTerminatingSeq(body, ctx, flat, max_stmts)) {
                    for (auto *s : flat) out.push_back(s);
                    return out.size() <= max_stmts;
                }

                // Body must end in a goto (passthrough).  Collect the
                // non-label stmts from the prefix SStmt siblings, then
                // follow the chain to the next label.
                std::string_view next_target;
                for (size_t ci = 0; ci + 1 < body.size(); ++ci) {
                    auto *child_st = body[ci]->dyn_cast<SStmt>();
                    if (!child_st) return false;
                    auto *s = child_st->Stmt();
                    if (llvm::isa<clang::LabelStmt>(s))
                        return false;
                    out.push_back(s);
                    if (out.size() > max_stmts) return false;
                }
                SNode *last = body.back();
                if (auto *sg = last->dyn_cast<SGoto>()) {
                    next_target = sg->Target();
                } else if (auto *st = last->dyn_cast<SStmt>()) {
                    auto *gs2 = llvm::dyn_cast_or_null<clang::GotoStmt>(
                        st->Stmt());
                    if (!gs2 || !gs2->getLabel()) return false;
                    next_target = gs2->getLabel()->getName();
                } else {
                    return false;
                }

                target = next_target;
            }
            return false; // exceeded max_hops
        }

        /// Try to resolve a GotoStmt target into a vector of replacement
        /// stmts (the label's return-terminating body).  Follows goto
        /// chains when the direct body ends in another goto.
        bool ResolveGotoReturnBody(
            clang::GotoStmt *gs,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            clang::ASTContext &ctx,
            std::vector<clang::Stmt *> &out
        ) {
            if (!gs || !gs->getLabel()) return false;
            return ResolveGotoChain(gs->getLabel()->getName(), labels, ctx, out);
        }

        /// Extract the GotoStmt from a clang::Stmt that is either a bare
        /// GotoStmt or a CompoundStmt whose last stmt is a GotoStmt.
        /// Returns {GotoStmt*, is_compound, stmts_before_goto}.
        struct IfArmGoto {
            clang::GotoStmt *gs = nullptr;
            clang::CompoundStmt *compound = nullptr;  // non-null if goto is inside compound
        };

        IfArmGoto ExtractIfArmGoto(clang::Stmt *arm) {
            if (!arm) return {};
            if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(arm))
                return {gs, nullptr};
            if (auto *cs = llvm::dyn_cast<clang::CompoundStmt>(arm)) {
                if (cs->body_empty()) return {};
                if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(cs->body_back()))
                    return {gs, cs};
            }
            return {};
        }

        /// Build a CompoundStmt containing `prefix` stmts followed by
        /// `suffix` stmts.  Used to replace an IfStmt arm that had a
        /// trailing goto: the prefix is the non-goto stmts that preceded
        /// the goto, the suffix is the label body stmts.
        clang::CompoundStmt *BuildReplacementArm(
            clang::ASTContext &ctx,
            clang::CompoundStmt *original_compound,
            const std::vector<clang::Stmt *> &label_body
        ) {
            std::vector<clang::Stmt *> stmts;
            if (original_compound) {
                // Copy all stmts except the trailing GotoStmt.
                for (auto *s : original_compound->body()) {
                    if (s == original_compound->body_back()) break;
                    stmts.push_back(s);
                }
            }
            for (auto *s : label_body)
                stmts.push_back(s);
            auto loc = VirtualLoc(ctx);
            return clang::CompoundStmt::Create(
                ctx, stmts, clang::FPOptionsOverride(), loc, loc);
        }

        /// For a clang::IfStmt whose then/else arm is a GotoStmt
        /// targeting a return-terminating label, replace the goto arm
        /// with the label's body wrapped in a CompoundStmt.  Handles
        /// bare gotos and CompoundStmt-wrapped gotos.
        bool TryReplaceIfGuardedGoto(
            clang::IfStmt *ifs,
            clang::ASTContext &ctx,
            const std::unordered_map<std::string_view, LabelEntry> &labels
        ) {
            bool changed = false;

            // Check then-arm.
            auto then_info = ExtractIfArmGoto(ifs->getThen());
            if (then_info.gs) {
                std::vector<clang::Stmt *> body;
                if (ResolveGotoReturnBody(then_info.gs, labels, ctx, body)) {
                    ifs->setThen(BuildReplacementArm(
                        ctx, then_info.compound, body));
                    changed = true;
                }
            }

            // Check else-arm.
            auto else_info = ExtractIfArmGoto(ifs->getElse());
            if (else_info.gs) {
                std::vector<clang::Stmt *> body;
                if (ResolveGotoReturnBody(else_info.gs, labels, ctx, body)) {
                    ifs->setElse(BuildReplacementArm(
                        ctx, else_info.compound, body));
                    changed = true;
                }
            }
            return changed;
        }

        /// Resolve a goto `target` into the return-terminating stmts
        /// that should replace the goto.  Tries the label body's
        /// trailing SStmt run, then the fallthrough tail across sibling
        /// labels, then multi-hop goto-chain resolution.
        bool ResolveReturnStmts(
            std::string_view target,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            clang::ASTContext &ctx,
            std::vector<clang::Stmt *> &out
        ) {
            auto it = labels.find(target);
            if (it == labels.end()) return false;

            out.clear();
            if (ExtractReturnTail(it->second.label->BodyList(), out))
                return true;
            out.clear();
            if (CollectReturnTail(it->second, out))
                return true;
            out.clear();
            if (ResolveGotoChain(target, labels, ctx, out))
                return true;
            out.clear();
            return false;
        }

        /// Replace goto-to-return across one sequence and its nested
        /// body-vectors.  An SStmt holding a clang::IfStmt with a goto
        /// arm gets the arm rewritten in place; an SStmt holding a
        /// GotoStmt — or a bare SGoto — that targets a return-terminating
        /// label is replaced by the label's return body, spilled as
        /// SStmt siblings.
        bool ReplaceGotoInSeq(
            std::vector<SNode *> &seq, SNodeFactory &factory,
            clang::ASTContext &ctx,
            const std::unordered_map<std::string_view, LabelEntry> &labels
        ) {
            bool changed = false;

            // Recurse into nested body-vectors first.
            for (SNode *child : seq) {
                ForEachBodyList(child, [&](std::vector<SNode *> &body) {
                    if (ReplaceGotoInSeq(body, factory, ctx, labels))
                        changed = true;
                });
            }

            // Rewrite if-guarded goto arms in place (mutates the
            // clang::IfStmt; does not alter the sibling list).
            for (SNode *child : seq) {
                if (auto *st = child->dyn_cast<SStmt>())
                    if (auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(
                            st->Stmt()))
                        if (TryReplaceIfGuardedGoto(ifs, ctx, labels))
                            changed = true;
            }

            // Replace goto-holding siblings (SStmt(GotoStmt) or bare
            // SGoto) with the resolved return body.
            for (size_t i = 0; i < seq.size(); ++i) {
                std::string_view target;
                if (auto *st = seq[i]->dyn_cast<SStmt>()) {
                    if (auto *gs = llvm::dyn_cast_or_null<clang::GotoStmt>(
                            st->Stmt()))
                        if (gs->getLabel())
                            target = gs->getLabel()->getName();
                } else if (auto *g = seq[i]->dyn_cast<SGoto>()) {
                    target = g->Target();
                }
                if (target.empty()) continue;

                std::vector<clang::Stmt *> body;
                if (!ResolveReturnStmts(target, labels, ctx, body))
                    continue;

                std::vector<SNode *> repl;
                AppendStmts(factory, repl, body);
                if (repl.empty()) continue;

                seq.erase(seq.begin() + static_cast<ptrdiff_t>(i));
                seq.insert(seq.begin() + static_cast<ptrdiff_t>(i),
                           repl.begin(), repl.end());
                i += repl.size() - 1; // skip the just-inserted siblings
                changed = true;
            }
            return changed;
        }

    } // anonymous namespace

    bool ConvertGotoToReturn(std::vector<SNode *> &root, SNodeFactory &factory,
                             clang::ASTContext &ctx) {
        std::unordered_map<std::string_view, LabelEntry> labels;
        CollectLabels(root, labels);

        // Single pass: chain resolution (ResolveGotoChain) already
        // follows multi-hop goto chains in one shot, so iterating
        // here would duplicate epilogue bodies exponentially.
        return ReplaceGotoInSeq(root, factory, ctx, labels);
    }

    // ---------------------------------------------------------------
    // RemoveUnreferencedLabels — drop SLabel nodes with zero refs.
    //
    // After ConvertGotoToReturn inlines return bodies at goto sites,
    // the original labels may have zero remaining references.  Their
    // bodies are dead code (only reachable via the now-removed label).
    // This pass removes such SLabel+body from the SNode tree.
    // ---------------------------------------------------------------

    namespace {

        // Collect ALL label references: SGoto targets + clang::GotoStmt
        // targets inside SStmt stmts.
        void CountAllGotoRefs(
            const SNode *node,
            std::unordered_set<std::string_view> &refs
        ) {
            if (!node) return;
            if (auto *g = node->dyn_cast<SGoto>()) {
                refs.insert(g->Target());
                return;
            }
            if (auto *st = node->dyn_cast<SStmt>()) {
                // Recursively walk the clang::Stmt tree for GotoStmt.
                std::function<void(clang::Stmt *)> walk =
                    [&](clang::Stmt *s) {
                    if (!s) return;
                    if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(s)) {
                        refs.insert(gs->getLabel()->getName());
                        return;
                    }
                    for (auto *child : s->children()) walk(child);
                };
                walk(st->Stmt());
                return;
            }
            // All other kinds: recurse uniformly via the visitor API.
            node->for_each_child([&](SNode *c) { CountAllGotoRefs(c, refs); });
        }

        void CountAllGotoRefs(
            const std::vector<SNode *> &seq,
            std::unordered_set<std::string_view> &refs
        ) {
            for (auto *c : seq) CountAllGotoRefs(c, refs);
        }

        // Remove unreferenced SLabel children from a sequence.
        bool RemoveDeadLabelsInSeq(
            std::vector<SNode *> &seq,
            const std::unordered_set<std::string_view> &refs
        ) {
            bool changed = false;
            for (size_t i = 0; i < seq.size(); ) {
                auto *lbl = seq[i]->dyn_cast<SLabel>();
                if (lbl && refs.count(lbl->Name()) == 0) {
                    seq.erase(seq.begin() + static_cast<ptrdiff_t>(i));
                    changed = true;
                } else {
                    ++i;
                }
            }
            return changed;
        }

    } // anonymous namespace

    bool RemoveUnreferencedLabels(std::vector<SNode *> &root, SNodeFactory &) {
        std::unordered_set<std::string_view> refs;
        CountAllGotoRefs(root, refs);
        return ForEachSeqPostOrder(
            root, [&](std::vector<SNode *> &seq) {
                return RemoveDeadLabelsInSeq(seq, refs);
            });
    }

    // ---------------------------------------------------------------
    // DuplicateSwitchCaseTargets — replace case bodies ending in
    // `SGoto L` with a cloned copy of L's body, making switches
    // goto-free even when the target is shared, a collapsed block,
    // or (critically) another switch.
    //
    // Safety invariant: the goto/label pairing of the whole tree must
    // stay consistent.  We enforce this by refusing to clone any
    // subtree that defines an SLabel or clang::LabelStmt — that would
    // duplicate a label definition and break Clang's one-decl-per-label
    // rule.  Outbound gotos *from* the clone are allowed: they add a
    // new reference to an already-live label, and RemoveUnreferencedLabels
    // recomputes liveness from scratch after the pass.
    // ---------------------------------------------------------------

    namespace {

        constexpr size_t kMaxCloneStmts = 8;

        // Aggregate clone budget for a single switch.  kMaxCloneStmts
        // bounds one clone; this bounds the sum across all case arms so
        // a switch with many arms all targeting the same label can't
        // clone a medium body into every arm.
        constexpr size_t kMaxSwitchCloneTotal = 24;

        size_t CountCloneStmts(const SNode *node);

        /// Count approximate clang::Stmt-equivalent size of a sequence.
        size_t CountCloneSeq(const std::vector<SNode *> &seq) {
            size_t n = 0;
            for (auto *c : seq) {
                n += CountCloneStmts(c);
                if (n > kMaxCloneStmts) return n;
            }
            return n;
        }

        /// Count approximate clang::Stmt-equivalent size of a subtree.
        size_t CountCloneStmts(const SNode *node) {
            if (!node) return 0;
            if (node->dyn_cast<SStmt>())
                return 1;
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                return 1 + CountCloneSeq(ite->ThenList())
                         + CountCloneSeq(ite->ElseList());
            }
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                size_t n = 1;
                for (auto &c : sw->Cases()) {
                    n += CountCloneSeq(c.body_list);
                    if (n > kMaxCloneStmts) return n;
                }
                n += CountCloneSeq(sw->DefaultBodyList());
                return n;
            }
            if (auto *lbl = node->dyn_cast<SLabel>())
                return 1 + CountCloneSeq(lbl->BodyList());
            return 1; // goto / break / continue / return
        }

        /// True iff \p node is safe to deep-clone: contains no label
        /// definitions (SLabel or clang::LabelStmt), no loops, and no
        /// SFor (loops would be duplicated, changing complexity).
        bool SubtreeIsSafeToClone(const SNode *node) {
            if (!node) return true;
            if (auto *st = node->dyn_cast<SStmt>()) {
                // Reject an SStmt that defines a label (cloning would
                // duplicate the definition) or contains a call (cloning
                // a call inflates the output with extra call sites that
                // do not exist in the input P-Code).
                std::function<bool(const clang::Stmt *)> has_unsafe =
                    [&](const clang::Stmt *cs) -> bool {
                        if (!cs) return false;
                        if (llvm::isa<clang::LabelStmt>(cs)) return true;
                        if (llvm::isa<clang::CallExpr>(cs)) return true;
                        for (const auto *c : cs->children())
                            if (has_unsafe(c)) return true;
                        return false;
                    };
                return !has_unsafe(st->Stmt());
            }
            // Reject SLabel (would duplicate definition) and all loops.
            switch (node->Kind()) {
                case SNodeKind::kLabel:
                case SNodeKind::kWhile:
                case SNodeKind::kDoWhile:
                case SNodeKind::kFor:
                    return false;
                default:
                    break;
            }
            // Sequence / SIfThenElse / SSwitch: safe iff every child is safe.
            // Leaves (SGoto/SBreak/SContinue/SReturn) have no children
            // and fall through to safe.
            bool safe = true;
            node->for_each_child([&](SNode *c) {
                if (safe && !SubtreeIsSafeToClone(c)) safe = false;
            });
            return safe;
        }

        SNode *CloneSNode(SNode *src, SNodeFactory &factory);

        /// Deep-clone a sequence (drops null clone results).
        std::vector<SNode *> CloneSeq(const std::vector<SNode *> &src,
                                      SNodeFactory &factory) {
            std::vector<SNode *> out;
            out.reserve(src.size());
            for (auto *c : src)
                if (auto *cl = CloneSNode(c, factory)) out.push_back(cl);
            return out;
        }

        /// Deep-clone a subtree.  Pre-condition: SubtreeIsSafeToClone(src).
        /// clang::Stmt* pointers are shared — clang::Stmt has no single
        /// parent, so aliasing the same stmt in two SStmt nodes is legal.
        SNode *CloneSNode(SNode *src, SNodeFactory &factory) {
            if (!src) return nullptr;
            if (auto *st = src->dyn_cast<SStmt>())
                return factory.Make<SStmt>(st->Stmt());
            if (auto *ite = src->dyn_cast<SIfThenElse>()) {
                return factory.Make<SIfThenElse>(
                    ite->Cond(),
                    CloneSeq(ite->ThenList(), factory),
                    CloneSeq(ite->ElseList(), factory));
            }
            if (auto *sw = src->dyn_cast<SSwitch>()) {
                auto *out = factory.Make<SSwitch>(sw->Discriminant());
                for (auto &c : sw->Cases())
                    out->AddCase(c.value, CloneSeq(c.body_list, factory));
                if (!sw->DefaultBodyList().empty())
                    out->SetDefaultBody(
                        CloneSeq(sw->DefaultBodyList(), factory));
                return out;
            }
            if (auto *g = src->dyn_cast<SGoto>())
                return factory.Make<SGoto>(factory.Intern(g->Target()));
            if (auto *br = src->dyn_cast<SBreak>())
                return factory.Make<SBreak>(br->Depth());
            if (src->dyn_cast<SContinue>())
                return factory.Make<SContinue>();
            if (auto *ret = src->dyn_cast<SReturn>())
                return factory.Make<SReturn>(ret->Value());
            return nullptr;
        }

        bool NeedsTerminatorBreak(const SNode *node);

        /// Sequence form: a sequence needs a break iff its last element
        /// does (an empty sequence falls through → needs a break).
        bool NeedsTerminatorBreakSeq(const std::vector<SNode *> &seq) {
            if (seq.empty()) return true;
            return NeedsTerminatorBreak(seq.back());
        }

        /// True iff \p node already ends in a terminator that transfers
        /// out of the enclosing switch; false if we need to append an
        /// SBreak so fallthrough does not leak into the next case.
        bool NeedsTerminatorBreak(const SNode *node) {
            if (!node) return true;
            if (node->dyn_cast<SReturn>()
                || node->dyn_cast<SBreak>()
                || node->dyn_cast<SContinue>()
                || node->dyn_cast<SGoto>())
                return false;
            if (auto *st = node->dyn_cast<SStmt>()) {
                auto *last = st->Stmt();
                if (llvm::isa<clang::ReturnStmt>(last)
                    || llvm::isa<clang::BreakStmt>(last)
                    || llvm::isa<clang::ContinueStmt>(last)
                    || llvm::isa<clang::GotoStmt>(last))
                    return false;
                return true;
            }
            if (auto *lbl = node->dyn_cast<SLabel>())
                return NeedsTerminatorBreakSeq(lbl->BodyList());
            return true;
        }

        std::string_view TrailingGotoTarget(const SNode *body);

        /// Return the trailing goto target name if the sequence ends in
        /// a goto.  Empty otherwise.
        std::string_view TrailingGotoTargetSeq(
            const std::vector<SNode *> &body
        ) {
            if (body.empty()) return {};
            return TrailingGotoTarget(body.back());
        }

        /// Return the trailing goto target name if \p body ends in a
        /// goto (bare SGoto, SStmt holding a GotoStmt, or SLabel
        /// wrapping any of those).  Empty otherwise.
        std::string_view TrailingGotoTarget(const SNode *body) {
            if (!body) return {};
            if (auto *g = body->dyn_cast<SGoto>()) return g->Target();
            if (auto *st = body->dyn_cast<SStmt>()) {
                if (auto *gs = llvm::dyn_cast_or_null<clang::GotoStmt>(
                        st->Stmt()))
                    return gs->getLabel()->getName();
                return {};
            }
            if (auto *lbl = body->dyn_cast<SLabel>()) {
                return TrailingGotoTargetSeq(lbl->BodyList());
            }
            return {};
        }

        /// Try to clone \p target_label's body.  Returns an empty vector
        /// if unsafe or too large.
        std::vector<SNode *> TryCloneLabelBody(
            std::string_view target_label,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            SNodeFactory &factory
        ) {
            auto it = labels.find(target_label);
            if (it == labels.end()) return {};
            auto &label_body = it->second.label->BodyList();
            if (label_body.empty()) return {};
            if (CountCloneSeq(label_body) > kMaxCloneStmts) return {};
            for (auto *c : label_body)
                if (!SubtreeIsSafeToClone(c)) return {};
            return CloneSeq(label_body, factory);
        }

        /// Build a replacement case body by splicing \p clone in place
        /// of the trailing goto in \p existing.  Returns the new body
        /// sequence, or an empty vector on failure.
        std::vector<SNode *> BuildSplicedSeq(
            const std::vector<SNode *> &existing,
            const std::vector<SNode *> &clone, SNodeFactory &factory
        ) {
            if (existing.empty() || clone.empty()) return {};

            bool need_break = NeedsTerminatorBreakSeq(clone);
            SNode *last = existing.back();
            std::vector<SNode *> out(existing.begin(), existing.end() - 1);

            auto append_clone = [&]() {
                for (auto *c : clone) out.push_back(c);
                if (need_break) out.push_back(factory.Make<SBreak>());
            };

            // Last element is a bare SGoto — drop it, append clone.
            if (last->dyn_cast<SGoto>()) {
                append_clone();
                return out;
            }

            // Last element is an SStmt holding a GotoStmt — drop it,
            // append clone.  (Any non-goto prefix statements are
            // already separate SStmt siblings retained in `out`.)
            if (auto *st = last->dyn_cast<SStmt>()) {
                if (!llvm::isa<clang::GotoStmt>(st->Stmt()))
                    return {};
                append_clone();
                return out;
            }

            // Last element is an SLabel wrapping a trailing goto: splice
            // inside the label body, keeping the label node itself so
            // outside gotos still resolve.
            if (auto *lbl = last->dyn_cast<SLabel>()) {
                std::vector<SNode *> inner =
                    BuildSplicedSeq(lbl->BodyList(), clone, factory);
                if (inner.empty()) return {};
                lbl->BodyList() = std::move(inner);
                out.push_back(lbl);
                return out;
            }

            return {};
        }

        bool DuplicateInSwitch(
            SSwitch *sw, SNodeFactory &factory,
            const std::unordered_map<std::string_view, LabelEntry> &labels
        ) {
            bool changed = false;
            // Aggregate clone budget across every arm of this switch.
            size_t cloned_total = 0;
            for (auto &c : sw->Cases()) {
                auto target = TrailingGotoTargetSeq(c.body_list);
                if (target.empty()) continue;
                std::vector<SNode *> clone =
                    TryCloneLabelBody(target, labels, factory);
                if (clone.empty()) continue;
                size_t clone_size = CountCloneSeq(clone);
                if (cloned_total + clone_size > kMaxSwitchCloneTotal)
                    continue;
                std::vector<SNode *> spliced =
                    BuildSplicedSeq(c.body_list, clone, factory);
                if (spliced.empty()) continue;
                c.body_list = std::move(spliced);
                cloned_total += clone_size;
                changed = true;
            }
            if (!sw->DefaultBodyList().empty()) {
                auto target = TrailingGotoTargetSeq(sw->DefaultBodyList());
                if (!target.empty()) {
                    std::vector<SNode *> clone =
                        TryCloneLabelBody(target, labels, factory);
                    if (!clone.empty()
                        && cloned_total + CountCloneSeq(clone)
                               <= kMaxSwitchCloneTotal) {
                        std::vector<SNode *> spliced = BuildSplicedSeq(
                            sw->DefaultBodyList(), clone, factory);
                        if (!spliced.empty()) {
                            sw->DefaultBodyList() = std::move(spliced);
                            changed = true;
                        }
                    }
                }
            }
            return changed;
        }

        bool WalkAndDuplicate(
            SNode *node, SNodeFactory &factory,
            const std::unordered_map<std::string_view, LabelEntry> &labels
        ) {
            if (!node) return false;
            bool changed = false;
            // Run the switch-specific duplication on SSwitch nodes
            // before descending — DuplicateInSwitch can replace case
            // bodies, and we want recursion to see the new shapes.
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                if (DuplicateInSwitch(sw, factory, labels)) changed = true;
            }
            // Uniform descent into every SNode child slot.
            node->for_each_child([&](SNode *c) {
                if (WalkAndDuplicate(c, factory, labels)) changed = true;
            });
            return changed;
        }

        /// Debug assertion: every remaining goto target resolves to a
        /// live SLabel.  Runs only in debug builds; aborts on failure
        /// so misbehaviour is caught before a broken TU ships.
        void VerifyGotoLabelPairing(std::vector<SNode *> &root) {
#ifndef NDEBUG
            std::unordered_map<std::string_view, LabelEntry> labels;
            CollectLabels(root, labels);
            std::unordered_set<std::string_view> refs;
            CountAllGotoRefs(root, refs);
            for (auto name : refs) {
                if (labels.find(name) == labels.end()) {
                    LOG(ERROR) << "DuplicateSwitchCaseTargets: dangling "
                               << "goto target '" << std::string(name)
                               << "' after duplication\n";
                    assert(false && "dangling goto target after duplication");
                }
            }
#else
            (void)root;
#endif
        }

    } // anonymous namespace

    bool DuplicateSwitchCaseTargets(std::vector<SNode *> &root,
                                    SNodeFactory &factory) {
        bool any_changed = false;
        // Re-scan labels on each iteration: a previous duplication may
        // expose new opportunities (e.g., cloning a label body that
        // itself contained a case goto to another label).  Bound the
        // loop to avoid pathological growth.
        for (int pass = 0; pass < 4; ++pass) {
            std::unordered_map<std::string_view, LabelEntry> labels;
            CollectLabels(root, labels);
            bool did = false;
            for (SNode *c : root)
                if (WalkAndDuplicate(c, factory, labels)) did = true;
            if (!did) break;
            any_changed = true;
        }
        if (any_changed) VerifyGotoLabelPairing(root);
        return any_changed;
    }

} // namespace patchestry::ast
