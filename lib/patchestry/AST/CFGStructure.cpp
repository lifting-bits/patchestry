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

    // ---------------------------------------------------------------
    // RecomputeRPO — pre-pass Phase 4
    // ---------------------------------------------------------------
    //
    // DFS from entry on active (non-collapsed) nodes to assign fresh
    // RPO positions.  Stores result in rpo_pos_[node_id] = position.
    // Used by ComputeDominatorTree's intersect function.

    void CFGStructure::RecomputeRPO() {
        const size_t n = graph_.nodes.size();
        constexpr size_t kNone = CNode::kNone;
        rpo_pos_.assign(n, kNone);

        // Iterative post-order DFS
        std::vector<size_t> post_order;
        std::vector<bool> visited(n, false);

        struct Frame {
            size_t id;
            size_t child_idx;
        };
        std::vector<Frame> stack;

        size_t entry = graph_.entry;
        if (entry >= n || graph_.Node(entry).IsCollapsed()) return;

        stack.push_back({entry, 0});
        visited[entry] = true;

        while (!stack.empty()) {
            auto &top = stack.back();
            auto &node = graph_.Node(top.id);

            if (top.child_idx < node.succs.size()) {
                size_t child = node.succs[top.child_idx];
                ++top.child_idx;
                if (child < n && !visited[child] && !graph_.Node(child).IsCollapsed()) {
                    visited[child] = true;
                    stack.push_back({child, 0});
                }
            } else {
                post_order.push_back(top.id);
                stack.pop_back();
            }
        }

        // Reverse post-order: first visited = position 0
        size_t pos = 0;
        for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
            rpo_pos_[*it] = pos++;
        }
    }

    // ---------------------------------------------------------------
    // ComputeDominatorTree — pre-pass Phase 5
    // ---------------------------------------------------------------
    //
    // Cooper-Harvey-Kennedy iterative dominator algorithm.
    // Uses rpo_pos_[] from Phase 4 for RPO comparisons.

    void CFGStructure::ComputeDominatorTree() {
        const size_t n = graph_.nodes.size();
        constexpr size_t kNone = CNode::kNone;
        idom_.assign(n, kNone);

        size_t entry = graph_.entry;
        if (entry >= n || graph_.Node(entry).IsCollapsed()) return;
        if (rpo_pos_[entry] == kNone) return;

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
                while (rpo_pos_[a] > rpo_pos_[b]) a = idom_[a];
                while (rpo_pos_[b] > rpo_pos_[a]) b = idom_[b];
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
                while (rev_rpo_pos[a] > rev_rpo_pos[b]) a = rev_idom[a];
                while (rev_rpo_pos[b] > rev_rpo_pos[a]) b = rev_idom[b];
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
    // NormalizeConditionPolarityIPdom — pre-pass Phase 7
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

        // Pre-pass Phase 1: identify back-edges so subsequent phases
        // can distinguish loop-back predecessors from forward predecessors.
        MarkBackEdges(graph_);

        // Pre-pass Phase 3: merge conditional forwarders into predecessors.
        MergeConditionalForwarders(graph_);

        // Pre-pass Phase 4: recompute RPO positions after merges/swaps.
        // Provides correct ordering for dominator computation.
        RecomputeRPO();

        // Pre-pass Phase 5: compute immediate dominators.
        ComputeDominatorTree();

        // Pre-pass Phase 6: compute immediate post-dominators.
        ComputePostDominatorTree();

        // Re-mark back-edges after topology changes, then discover loops.
        // Loop detection must use original P-Code polarity (stable).
        MarkBackEdges(graph_);
        OrderLoops();

        // Pre-pass Phase 7: refine polarity using ipdom.
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
        // SNode for the emitter to consume.
        for (auto &node : graph_.nodes) {
            if (!node.IsCollapsed() && node.structured == nullptr) {
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
                if (!node.structured) continue;
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
                // Wrap: SSeq { existing structured, SGoto(succ_label) }
                auto *seq = factory_.Make<SSeq>();
                seq->AddChild(node.structured);
                seq->AddChild(factory_.Make<SGoto>(
                    factory_.Intern(sn.original_label)));
                node.structured = seq;
            }
        }

        // Ensure labels are preserved on active representative nodes.
        // When IdentifyInternal collapses nodes into a representative,
        // the rule-supplied SNode may not include the representative's
        // own label.  Wrap it now so goto targets remain valid.
        //
        // Check recursively: the label might be inside an SSeq child
        // (e.g., after the explicit-goto pass wrapped the node).
        auto has_label = [](const SNode *sn, std::string_view name) -> bool {
            std::function<bool(const SNode *)> check = [&](const SNode *n) -> bool {
                if (!n) return false;
                if (auto *lbl = n->dyn_cast<SLabel>())
                    return lbl->Name() == name || check(lbl->Body());
                if (auto *seq = n->dyn_cast<SSeq>()) {
                    for (auto *c : seq->Children())
                        if (check(c)) return true;
                }
                return false;
            };
            return check(sn);
        };

        for (auto &node : graph_.nodes) {
            if (node.IsCollapsed()) continue;
            if (node.original_label.empty()) continue;
            if (!node.structured) continue;
            if (has_label(node.structured, node.original_label)) continue;
            node.structured = factory_.Make<SLabel>(
                factory_.Intern(node.original_label), node.structured);
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
    // Action:  Merge into SSeq, collapse via IdentifyInternal.
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
        // by collapsed nodes.  After predecessor collapse, b.preds
        // may show 1 active pred, but collapsed nodes' SNode trees
        // can still hold clang::GotoStmt to B's label.  Merging B
        // would consume the label, creating dangling goto references.
        if (!b.original_label.empty()) {
            for (const auto &n : graph_.nodes) {
                if (n.id == id || n.id == b_id) continue;
                if (!n.IsCollapsed()) {
                    // Active node with goto edge to B — label needed.
                    for (size_t i = 0; i < n.succs.size(); ++i) {
                        if (n.succs[i] == b_id && n.IsGotoOut(i))
                            return false;
                    }
                } else {
                    // Collapsed node might still hold clang::GotoStmt
                    // to B's label.
                    for (size_t s : n.succs) {
                        if (s == b_id) return false;
                    }
                }
            }
        }

        // Build the merged SNode.
        auto *seq = factory_.Make<SSeq>();

        // Add A's content (strip terminal — the A→B edge is absorbed).
        SNode *a_node = BuildLeafSNode(id, /*include_terminal=*/false);
        if (a_node) seq->AddChild(a_node);

        // Add B's content.
        SNode *b_node = BuildLeafSNode(b_id);
        if (b_node) seq->AddChild(b_node);

        // Collapse {A, B} into the representative node.
        graph_.IdentifyInternal({id, b_id}, CNode::BlockType::kSequence, seq);

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

    SNode *CFGStructure::WrapWithPriorContent(size_t id, SNode *child) {
        auto &a = graph_.Node(id);
        SNode *result = child;
        if (a.structured) {
            auto *seq = factory_.Make<SSeq>();
            seq->AddChild(a.structured);
            seq->AddChild(child);
            result = seq;
        } else if (!a.stmts.empty()) {
            auto *seq = factory_.Make<SSeq>();
            auto *a_blk = factory_.Make<SBlock>();
            for (auto *s : a.stmts) a_blk->AddStmt(s);
            seq->AddChild(a_blk);
            seq->AddChild(child);
            result = seq;
        }
        if (!a.original_label.empty()) {
            result = factory_.Make<SLabel>(
                factory_.Intern(a.original_label), result);
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
            auto *then_body = BuildLeafSNode(t_id, /*include_terminal=*/false);
            auto *if_node = factory_.Make<SIfThenElse>(
                a.branch_cond, then_body, nullptr);

            SNode *result = WrapWithPriorContent(id, if_node);

            graph_.IdentifyInternal({id, t_id}, CNode::BlockType::kIf, result);
            return true;
        }

        // Case 1b: T is a conditional forwarder (no stmts, just a branch)
        // with one arm going to F (merge).  Merge conditions:
        //   if (c1 && inner_cond) goto other_succ;
        // Guard: must not have a structured SNode — collapsed nodes have
        // stmts cleared by IdentifyInternal but carry content in structured.
        if (!skip_case1
            && HasSoleRealPredecessor(t_id, id) && t.is_conditional
            && t.stmts.empty() && !t.structured
            && t.succs.size() == 2 && t.branch_cond)
        {
            bool t_s0_is_merge = (t.succs[0] == f_id);
            bool t_s1_is_merge = (t.succs[1] == f_id);
            if (t_s0_is_merge || t_s1_is_merge) {
                size_t goto_target = t_s0_is_merge ? t.succs[1] : t.succs[0];
                auto &target_node = graph_.Node(goto_target);

                // Refuse if target has no label — would produce dangling goto.
                if (!target_node.original_label.empty()) {
                    // succs[1] = taken (cond true).  If taken goes to merge,
                    // the goto fires when cond is false — negate.
                    clang::Expr *inner_cond = t_s1_is_merge
                        ? NegateExpr(ctx_, t.branch_cond)
                        : t.branch_cond;
                    auto *merged_cond = clang::BinaryOperator::Create(
                        ctx_,
                        EnsureRValue(ctx_, a.branch_cond),
                        EnsureRValue(ctx_, inner_cond),
                        clang::BO_LAnd, ctx_.BoolTy, clang::VK_PRValue,
                        clang::OK_Ordinary, clang::SourceLocation(),
                        clang::FPOptionsOverride());

                    auto *if_goto = factory_.Make<SIfThenElse>(
                        merged_cond,
                        factory_.Make<SGoto>(factory_.Intern(target_node.original_label)),
                        nullptr);

                    SNode *result = WrapWithPriorContent(id, if_goto);

                    graph_.IdentifyInternal({id, t_id}, CNode::BlockType::kIf, result);
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
            auto *then_body = BuildLeafSNode(f_id, /*include_terminal=*/false);
            // The condition is for the taken branch (T = merge).
            // The body executes when the condition is false — negate.
            auto *if_node = factory_.Make<SIfThenElse>(
                NegateExpr(ctx_, a.branch_cond), then_body, nullptr);

            SNode *result = WrapWithPriorContent(id, if_node);

            graph_.IdentifyInternal({id, f_id}, CNode::BlockType::kIf, result);
            return true;
        }

        // Case 2b: F is a conditional forwarder (no stmts, just a branch)
        // with one arm going to T (merge).  Outer condition negated, then
        // merged with inner:  if (!c1 && inner_cond) goto other_succ;
        // Guard: must not have a structured SNode (same as Case 1b).
        if (!skip_case2
            && HasSoleRealPredecessor(f_id, id) && f.is_conditional
            && f.stmts.empty() && !f.structured
            && f.succs.size() == 2 && f.branch_cond)
        {
            bool f_s0_is_merge = (f.succs[0] == t_id);
            bool f_s1_is_merge = (f.succs[1] == t_id);
            if (f_s0_is_merge || f_s1_is_merge) {
                size_t goto_target = f_s0_is_merge ? f.succs[1] : f.succs[0];
                auto &target_node = graph_.Node(goto_target);

                if (!target_node.original_label.empty()) {
                    // Outer condition: F is the not-taken arm, so body
                    // executes when c1 is false — negate outer.
                    clang::Expr *outer_cond = NegateExpr(ctx_, a.branch_cond);
                    clang::Expr *inner_cond = f_s1_is_merge
                        ? NegateExpr(ctx_, f.branch_cond)
                        : f.branch_cond;
                    auto *merged_cond = clang::BinaryOperator::Create(
                        ctx_,
                        EnsureRValue(ctx_, outer_cond),
                        EnsureRValue(ctx_, inner_cond),
                        clang::BO_LAnd, ctx_.BoolTy, clang::VK_PRValue,
                        clang::OK_Ordinary, clang::SourceLocation(),
                        clang::FPOptionsOverride());

                    auto *if_goto = factory_.Make<SIfThenElse>(
                        merged_cond,
                        factory_.Make<SGoto>(factory_.Intern(target_node.original_label)),
                        nullptr);

                    SNode *result = WrapWithPriorContent(id, if_goto);

                    graph_.IdentifyInternal({id, f_id}, CNode::BlockType::kIf, result);
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
            auto *then_body = BuildLeafSNode(body_id, /*include_terminal=*/false);

            // Negate condition if body is the not-taken arm (orient==1).
            clang::Expr *cond = orient == 0
                ? a.branch_cond
                : NegateExpr(ctx_, a.branch_cond);

            auto *if_node = factory_.Make<SIfThenElse>(cond, then_body, nullptr);

            SNode *result = WrapWithPriorContent(id, if_node);

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
                auto *then_body = BuildLeafSNode(t_id, /*include_terminal=*/false);
                auto *else_body = BuildLeafSNode(f_id, /*include_terminal=*/false);
                auto *if_node = factory_.Make<SIfThenElse>(
                    a.branch_cond, then_body, else_body);

                SNode *result = WrapWithPriorContent(id, if_node);

                graph_.IdentifyInternal(
                    {id, t_id, f_id}, CNode::BlockType::kIf, result);
                return true;
            }

            // Relaxed: one arm shared
            size_t sole_id;
            bool sole_is_taken;
            if (t_sole && !f_sole && f.stmts.size() <= 2) {
                sole_id = t_id; sole_is_taken = true;
            } else if (f_sole && !t_sole && t.stmts.size() <= 2) {
                sole_id = f_id; sole_is_taken = false;
            } else {
                return false;
            }

            {
                auto *sole_body = BuildLeafSNode(sole_id, /*include_terminal=*/false);
                clang::Expr *cond = sole_is_taken
                    ? a.branch_cond
                    : NegateExpr(ctx_, a.branch_cond);
                auto *if_node = factory_.Make<SIfThenElse>(
                    cond, sole_body, nullptr);

                SNode *result = WrapWithPriorContent(id, if_node);

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
        auto *then_body = BuildLeafSNode(t_id, /*include_terminal=*/true);
        auto *else_body = BuildLeafSNode(f_id, /*include_terminal=*/true);
        auto *if_node = factory_.Make<SIfThenElse>(
            a.branch_cond, then_body, else_body);

        // A's prior content goes before the if: use a.structured
        SNode *result = WrapWithPriorContent(id, if_node);

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
            } else if (nd.structured) {
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
    // strips terminals from interior nodes, and wraps in SSeq.
    // ---------------------------------------------------------------

    SNode *CFGStructure::BuildLoopBodySNode(
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
            } else if (nd.structured) {
                // Collapsed but has structured content — include it.
                if (seen.insert(nid).second) interior.push_back(nid);
            }
        }
        std::sort(interior.begin(), interior.end());

        if (interior.empty()) {
            return factory_.Make<SBlock>(); // empty body
        }

        // Helper: for a conditional interior node, if one successor
        // exits the loop (not in bodyset), emit if (exit_cond) goto label;
        // to preserve the exit path that would be lost when the terminal
        // is stripped.
        // next_rpo_id: node ID of the next interior block in RPO order,
        // or SIZE_MAX if this is the last block.  Used to decide whether
        // stripping a terminal goto produces correct fallthrough.
        auto build_node = [&](size_t nid, size_t next_rpo_id) -> SNode * {
            auto &nd = graph_.Node(nid);

            // Collapsed nodes with a pre-built structured SNode: return
            // it directly — don't try to access succs/preds (stale).
            if (nd.IsCollapsed() && nd.structured) return nd.structured;

            SNode *leaf = BuildLeafSNode(nid, /*include_terminal=*/false);

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
                            auto *go = factory_.Make<SGoto>(
                                factory_.Intern(tn.original_label));
                            auto *w = factory_.Make<SSeq>();
                            if (leaf) w->AddChild(leaf);
                            w->AddChild(go);
                            return static_cast<SNode *>(w);
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

                    auto *leaf_blk = leaf ? leaf->dyn_cast<SBlock>() : nullptr;
                    bool leaf_empty = !nd.structured
                        && nd.original_label.empty()
                        && ((!leaf) || (leaf_blk && leaf_blk->Stmts().empty()));
                    if (leaf_empty) return if_goto;

                    auto *w = factory_.Make<SSeq>();
                    if (leaf) w->AddChild(leaf);
                    w->AddChild(if_goto);
                    return static_cast<SNode *>(w);
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

            // If the leaf is effectively empty (no stmts, no label),
            // emit just the if-goto without a wrapper.  Labeled nodes
            // are NEVER empty — the label must be preserved because
            // external gotos may target it.
            auto *leaf_blk = leaf ? leaf->dyn_cast<SBlock>() : nullptr;
            bool leaf_empty = !nd.structured
                && nd.original_label.empty()
                && ((!leaf) || (leaf_blk && leaf_blk->Stmts().empty()));
            if (leaf_empty) return if_goto;

            auto *wrapper = factory_.Make<SSeq>();
            if (leaf) wrapper->AddChild(leaf);
            wrapper->AddChild(if_goto);
            return static_cast<SNode *>(wrapper);
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
            SNode *child = build_node(interior[idx], next);
            if (child) children.push_back(child);
        }

        // Merge pass: look for SIfThenElse(cond, SGoto(L), null) nodes
        // targeting the same label and combine with ||.  Skip empty
        // SBlock nodes between them (remnants of conditional routing nodes).
        auto is_empty_block = [](SNode *n) -> bool {
            auto *blk = n->dyn_cast<SBlock>();
            return blk && blk->Stmts().empty();
        };

        for (size_t i = 0; i < children.size(); ++i) {
            auto *ite1 = children[i]->dyn_cast<SIfThenElse>();
            if (!ite1 || ite1->ElseBranch() || !ite1->ThenBranch())
                continue;
            auto *g1 = ite1->ThenBranch()->dyn_cast<SGoto>();
            if (!g1) continue;

            // Scan forward, skipping empty blocks, looking for
            // another if-goto to the same target.
            size_t j = i + 1;
            while (j < children.size() && is_empty_block(children[j]))
                ++j;
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
            // Remove empty blocks between i and j, plus j itself.
            children.erase(children.begin() + static_cast<ptrdiff_t>(i + 1),
                           children.begin() + static_cast<ptrdiff_t>(j + 1));
            --i;  // retry from same position (might merge 3+)
        }

        // Remove any remaining empty blocks.
        children.erase(
            std::remove_if(children.begin(), children.end(), is_empty_block),
            children.end());

        auto *seq = factory_.Make<SSeq>();
        for (auto *child : children) {
            seq->AddChild(child);
        }
        return seq;
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
            SNode *loop_body_snode = BuildLoopBodySNode(body, id, bodyset);

            auto *inner = factory_.Make<SSeq>();
            if (h.structured) {
                inner->AddChild(h.structured);
            } else if (!h.stmts.empty()) {
                auto *h_blk = factory_.Make<SBlock>();
                for (auto *s : h.stmts) h_blk->AddStmt(s);
                inner->AddChild(h_blk);
            }

            // Header conditional: if(branch_cond) goto taken_label;
            auto &s1_node = graph_.Node(s1);
            if (!s1_node.original_label.empty()) {
                auto *taken_goto = factory_.Make<SGoto>(
                    factory_.Intern(s1_node.original_label));
                inner->AddChild(factory_.Make<SIfThenElse>(
                    h.branch_cond, taken_goto, nullptr));
            }

            if (loop_body_snode) inner->AddChild(loop_body_snode);

            auto *while_node = factory_.Make<SWhile>(nullptr, inner);
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
            graph_.IdentifyInternal(body, CNode::BlockType::kWhile, result);
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

        SNode *loop_body_snode = BuildLoopBodySNode(body, id, bodyset);
        SNode *result = nullptr;

        if (h.structured || !h.stmts.empty()) {
            // Header has computation stmts (or prior structured content)
            // that must re-execute each iteration.  Emit as:
            //   while(1) { header_content; if (exit_cond) break; body; }
            auto *inner = factory_.Make<SSeq>();

            // 1. Header content (re-execute each iteration).
            if (h.structured) {
                inner->AddChild(h.structured);
            } else {
                auto *h_blk = factory_.Make<SBlock>();
                for (auto *s : h.stmts) h_blk->AddStmt(s);
                inner->AddChild(h_blk);
            }

            // 2. Exit test: if (exit_cond) break;
            auto *break_node = factory_.Make<SIfThenElse>(
                exit_cond, factory_.Make<SBreak>(), nullptr);
            inner->AddChild(break_node);

            // 3. Loop body.
            if (loop_body_snode) inner->AddChild(loop_body_snode);

            // while(1) — nullptr condition → emitter synthesizes true.
            result = factory_.Make<SWhile>(nullptr, inner);
        } else {
            // Pure condition header (no side-effectful stmts).
            // Emit as: while(continue_cond) { body; }
            result = factory_.Make<SWhile>(continue_cond, loop_body_snode);
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
        graph_.IdentifyInternal(body, CNode::BlockType::kWhile, result);
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
        SNode *loop_body = BuildLoopBodySNode(body, id, bodyset);

        // Include header's content in the body (executes each iteration).
        SNode *full_body = loop_body;
        if (h.structured || !h.stmts.empty()) {
            auto *seq = factory_.Make<SSeq>();
            if (h.structured) {
                seq->AddChild(h.structured);
            } else {
                auto *h_blk = factory_.Make<SBlock>();
                for (auto *s : h.stmts) h_blk->AddStmt(s);
                seq->AddChild(h_blk);
            }
            if (loop_body) seq->AddChild(loop_body);
            full_body = seq;
        }

        // CGraph: succs[0] = not-taken (cond false), succs[1] = taken (cond true).
        // If the back-edge to header is on taken (s1), continue cond = branch_cond.
        // If the back-edge is on not-taken (s0), continue cond = !branch_cond.
        clang::Expr *dowhile_cond = s1_is_header
            ? tail.branch_cond
            : NegateExpr(ctx_, tail.branch_cond);
        auto *dowhile_node = factory_.Make<SDoWhile>(full_body, dowhile_cond);

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
        graph_.IdentifyInternal(body, CNode::BlockType::kDoWhile, result);
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
        if (body.size() == 1 && body[0] == id && h.structured) {
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
        SNode *loop_body = BuildLoopBodySNode(body, id, bodyset);

        // Include header content in body.
        SNode *full_body = loop_body;
        if (h.structured || !h.stmts.empty()) {
            auto *seq = factory_.Make<SSeq>();
            if (h.structured) {
                seq->AddChild(h.structured);
            } else {
                auto *h_blk = factory_.Make<SBlock>();
                for (auto *s : h.stmts) h_blk->AddStmt(s);
                seq->AddChild(h_blk);
            }
            if (loop_body) seq->AddChild(loop_body);
            full_body = seq;
        }

        auto *inf_while = factory_.Make<SWhile>(nullptr, full_body);

        // Set header label for continue resolution (no exit label for inf loops).
        if (!h.original_label.empty())
            inf_while->SetHeaderLabel(factory_.Intern(h.original_label));

        SNode *result = inf_while;
        if (!h.original_label.empty()) {
            result = factory_.Make<SLabel>(
                factory_.Intern(h.original_label), result);
        }

        ClearMarks(graph_, body);
        graph_.IdentifyInternal(body, CNode::BlockType::kWhile, result);
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
                        case_type, clang::SourceLocation());
                    sw->AddCase(val, case_body);
                }
                continue;
            }

            // Build case body from the target block.
            SNode *case_body = nullptr;

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
                    && !tn.structured && tn.terminal
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
                    auto build_branch = [&](size_t sid) -> SNode * {
                        if (can_absorb_succ(sid)) {
                            auto &sn = graph_.Node(sid);
                            bool nested_disp = sn.stmts.empty()
                                && !sn.structured && sn.terminal;
                            SNode *body = BuildLeafSNode(sid,
                                /*include_terminal=*/nested_disp);
                            if (std::find(collapse_ids.begin(),
                                    collapse_ids.end(), sid)
                                == collapse_ids.end())
                                collapse_ids.push_back(sid);
                            return body;
                        }
                        auto &sn = graph_.Node(sid);
                        if (!sn.original_label.empty())
                            return static_cast<SNode *>(
                                factory_.Make<SGoto>(
                                    factory_.Intern(sn.original_label)));
                        return static_cast<SNode *>(factory_.Make<SBlock>());
                    };

                    SNode *then_body = build_branch(t_id);
                    SNode *else_body = build_branch(f_id);
                    case_body = factory_.Make<SIfThenElse>(
                        tn.branch_cond, then_body, else_body);

                    // Wrap with dispatcher's label if present.
                    if (!tn.original_label.empty()) {
                        case_body = factory_.Make<SLabel>(
                            factory_.Intern(tn.original_label),
                            case_body);
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
                    case_body = factory_.Make<SBlock>();
                } else {
                    const std::string &lbl = tn.original_label;
                    case_body = factory_.Make<SGoto>(factory_.Intern(lbl));
                }
            }

            if (sc.is_default) {
                sw->SetDefaultBody(case_body);
            } else {
                auto *val = clang::IntegerLiteral::Create(
                    ctx_,
                    llvm::APInt(case_width, static_cast<uint64_t>(sc.value), true),
                    case_type, clang::SourceLocation());
                sw->AddCase(val, case_body);
            }
        }

        // A's pre-switch stmts go before the switch.
        SNode *result = sw;
        if (!a.stmts.empty()) {
            auto *seq = factory_.Make<SSeq>();
            auto *a_blk = factory_.Make<SBlock>();
            for (auto *s : a.stmts) a_blk->AddStmt(s);
            seq->AddChild(a_blk);
            seq->AddChild(sw);
            result = seq;
        }

        // Preserve A's label.
        if (!a.original_label.empty()) {
            result = factory_.Make<SLabel>(
                factory_.Intern(a.original_label), result);
        }

        graph_.IdentifyInternal(
            collapse_ids, CNode::BlockType::kSwitch, result);
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
            // picking the first non-goto, non-back outgoing edge from
            // any multi-successor active node.
            for (size_t nid : active) {
                auto &n = graph_.Node(nid);
                for (size_t i = 0; i < n.succs.size(); ++i) {
                    if (!n.IsGotoOut(i) && !n.IsBackEdge(i)
                        && n.succs.size() > 1)
                    {
                        n.SetGoto(i);
                        return true;
                    }
                }
            }
            return false;
        }

        // Mark the first likely-goto edge.
        auto &fe                = likely_goto_.front();
        auto [src_id, edge_idx] = fe.GetCurrentEdge(graph_);
        if (src_id == CNode::kNone) {
            // Edge no longer exists (collapsed away).  Try next.
            for (auto &edge : likely_goto_) {
                auto [s, ei] = edge.GetCurrentEdge(graph_);
                if (s != CNode::kNone) {
                    graph_.Node(s).SetGoto(ei);
                    return true;
                }
            }
            return false;
        }

        graph_.Node(src_id).SetGoto(edge_idx);
        return true;
    }

    // ---------------------------------------------------------------
    // BuildLeafSNode — wrap a CNode's stmts in an SBlock, optionally
    //                   wrapped in an SLabel if the node has a label.
    // ---------------------------------------------------------------

    SNode *CFGStructure::BuildLeafSNode(size_t id, bool include_terminal) {
        auto &node = graph_.Node(id);

        // If this node was already structured (e.g., by a prior rule),
        // return its existing SNode.
        if (node.structured) return node.structured;

        auto *blk = factory_.Make<SBlock>();
        for (auto *s : node.stmts) {
            blk->AddStmt(s);
        }

        // Append terminal (goto/if-goto) so the existing emitter path
        // can reconstruct control flow for unstructured remainders.
        // Skipped for non-tail nodes in a sequential merge where the
        // edge is absorbed — the terminal would be a dead goto.
        if (include_terminal && node.terminal) {
            blk->AddStmt(node.terminal);
        }

        SNode *result = blk;

        if (!node.original_label.empty()) {
            result = factory_.Make<SLabel>(
                factory_.Intern(node.original_label), blk);
        }

        return result;
    }

    // ---------------------------------------------------------------
    // BuildBodySNode — sequence of leaf SNodes from a list of node ids.
    // ---------------------------------------------------------------

    SNode *CFGStructure::BuildBodySNode(const std::vector<size_t> &ids) {
        if (ids.empty()) return nullptr;
        if (ids.size() == 1) return BuildLeafSNode(ids[0]);

        auto *seq = factory_.Make<SSeq>();
        for (size_t nid : ids) {
            SNode *child = BuildLeafSNode(nid);
            if (child) seq->AddChild(child);
        }
        return seq;
    }

    // ---------------------------------------------------------------
    // InlineResidualGotos — post-structuring cleanup
    //
    // Walks SSeq nodes looking for SGoto children whose target SLabel
    // is a sibling in the same SSeq and is only referenced by that
    // one goto.  Replaces the goto with the label's body and removes
    // the label node.
    // ---------------------------------------------------------------

    namespace {

        // Count how many SGoto nodes reference a given label name
        // anywhere in the SNode tree.
        void CountGotoRefs(SNode *node,
                           std::unordered_map<std::string_view, int> &refs) {
            if (!node) return;

            if (auto *g = node->dyn_cast<SGoto>()) {
                refs[g->Target()]++;
                return;
            }
            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (auto *c : seq->Children()) CountGotoRefs(c, refs);
                return;
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                CountGotoRefs(ite->ThenBranch(), refs);
                CountGotoRefs(ite->ElseBranch(), refs);
                return;
            }
            if (auto *w = node->dyn_cast<SWhile>()) {
                CountGotoRefs(w->Body(), refs);
                return;
            }
            if (auto *dw = node->dyn_cast<SDoWhile>()) {
                CountGotoRefs(dw->Body(), refs);
                return;
            }
            if (auto *f = node->dyn_cast<SFor>()) {
                CountGotoRefs(f->Body(), refs);
                return;
            }
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases()) CountGotoRefs(c.body, refs);
                CountGotoRefs(sw->DefaultBody(), refs);
                return;
            }
            if (auto *lbl = node->dyn_cast<SLabel>()) {
                CountGotoRefs(lbl->Body(), refs);
                return;
            }
        }

        // Try to inline gotos in a single SSeq.  Returns true if changed.
        bool InlineGotosInSeq(
            SSeq *seq, SNodeFactory &factory,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            bool changed = false;

            // Build index: label name → position in this SSeq.
            std::unordered_map<std::string_view, size_t> label_pos;
            for (size_t i = 0; i < seq->Size(); ++i) {
                if (auto *lbl = (*seq)[i]->dyn_cast<SLabel>()) {
                    label_pos[lbl->Name()] = i;
                }
            }

            // Scan children for SGoto nodes that can be inlined.
            // Only inline when:
            //   (a) label is the immediate next sibling (forward goto, no skipped code), OR
            //   (b) all siblings between goto and label are empty SBlocks (nothing skipped)
            // This prevents changing control-flow semantics by executing code
            // that the goto would have jumped over.
            for (size_t i = 0; i < seq->Size(); ++i) {
                auto *g = (*seq)[i]->dyn_cast<SGoto>();
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

                // Check that all siblings between goto (i) and label (label_idx)
                // are empty SBlocks — i.e., the goto doesn't skip any real code.
                bool can_inline = true;
                for (size_t k = i + 1; k < label_idx; ++k) {
                    auto *between = (*seq)[k];
                    auto *blk = between->dyn_cast<SBlock>();
                    if (!blk || !blk->Empty()) {
                        can_inline = false;
                        break;
                    }
                }
                if (!can_inline) continue;

                auto *lbl = (*seq)[label_idx]->as<SLabel>();

                // Replace the goto with the label's body.
                SNode *body = lbl->Body();
                if (body) {
                    seq->ReplaceChild(i, body);
                } else {
                    seq->ReplaceChild(i, factory.Make<SBlock>());
                }

                // Remove the label node (and any empty blocks between).
                // Remove from the end to avoid index shifting.
                for (size_t k = label_idx; k > i; --k) {
                    seq->RemoveChild(k);
                }

                changed = true;
                // Rebuild label_pos since indices shifted.
                label_pos.clear();
                for (size_t j = 0; j < seq->Size(); ++j) {
                    if (auto *l = (*seq)[j]->dyn_cast<SLabel>()) {
                        label_pos[l->Name()] = j;
                    }
                }
            }

            return changed;
        }

        // Recursively process all SSeq nodes in the tree.
        bool InlineGotosRecursive(
            SNode *node, SNodeFactory &factory,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            if (!node) return false;
            bool changed = false;

            if (auto *seq = node->dyn_cast<SSeq>()) {
                // Process children first (bottom-up).
                for (size_t i = 0; i < seq->Size(); ++i) {
                    if (InlineGotosRecursive((*seq)[i], factory, refs))
                        changed = true;
                }
                if (InlineGotosInSeq(seq, factory, refs))
                    changed = true;
                return changed;
            }

            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                if (InlineGotosRecursive(ite->ThenBranch(), factory, refs))
                    changed = true;
                if (InlineGotosRecursive(ite->ElseBranch(), factory, refs))
                    changed = true;
                return changed;
            }
            if (auto *w = node->dyn_cast<SWhile>()) {
                return InlineGotosRecursive(w->Body(), factory, refs);
            }
            if (auto *dw = node->dyn_cast<SDoWhile>()) {
                return InlineGotosRecursive(dw->Body(), factory, refs);
            }
            if (auto *f = node->dyn_cast<SFor>()) {
                return InlineGotosRecursive(f->Body(), factory, refs);
            }
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases()) {
                    if (InlineGotosRecursive(c.body, factory, refs))
                        changed = true;
                }
                if (InlineGotosRecursive(sw->DefaultBody(), factory, refs))
                    changed = true;
                return changed;
            }
            if (auto *lbl = node->dyn_cast<SLabel>()) {
                return InlineGotosRecursive(lbl->Body(), factory, refs);
            }

            return false;
        }

    } // anonymous namespace

    // ---------------------------------------------------------------
    // EliminateGotoToNextLabel — SNode post-pass
    //
    // Walk SSeq children.  For each child followed by an SLabel,
    // chase through nesting (SLabel body → SSeq last child → SBlock
    // trailing stmt) to find the deepest trailing stmt.  If it's a
    // goto (SGoto or clang::GotoStmt) targeting the next label, or
    // an IfStmt with one arm being such a goto, eliminate it.
    // ---------------------------------------------------------------

    namespace {

        /// Chase through SLabel/SSeq/SBlock to find the deepest trailing
        /// SNode or clang::Stmt.  Returns {leaf_snode, clang_stmt_or_null}.
        /// The leaf_snode is the SNode containing the trailing stmt.
        struct TrailingInfo {
            SNode *container = nullptr;  // innermost SNode (SBlock, SGoto, etc.)
            clang::Stmt *stmt = nullptr; // if container is SBlock, its last stmt
        };

        TrailingInfo DeepTrailingSNode(SNode *node) {
            if (!node) return {};
            if (auto *lbl = node->dyn_cast<SLabel>()) {
                return DeepTrailingSNode(lbl->Body());
            }
            if (auto *seq = node->dyn_cast<SSeq>()) {
                auto &ch = seq->Children();
                if (ch.empty()) return {};
                return DeepTrailingSNode(ch.back());
            }
            if (auto *blk = node->dyn_cast<SBlock>()) {
                if (blk->Empty()) return {};
                return {blk, blk->Stmts().back()};
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

        bool EliminateInSSeq(SSeq *seq, SNodeFactory &factory,
                             clang::ASTContext &ctx);

        bool EliminateRecursive(SNode *node, SNodeFactory &factory,
                                clang::ASTContext &ctx) {
            if (!node) return false;
            bool changed = false;
            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (auto *child : seq->Children())
                    if (EliminateRecursive(child, factory, ctx)) changed = true;
                if (EliminateInSSeq(seq, factory, ctx)) changed = true;
            } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                if (EliminateRecursive(ite->ThenBranch(), factory, ctx)) changed = true;
                if (EliminateRecursive(ite->ElseBranch(), factory, ctx)) changed = true;
            } else if (auto *w = node->dyn_cast<SWhile>()) {
                if (EliminateRecursive(w->Body(), factory, ctx)) changed = true;
            } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
                if (EliminateRecursive(dw->Body(), factory, ctx)) changed = true;
            } else if (auto *lbl = node->dyn_cast<SLabel>()) {
                if (EliminateRecursive(lbl->Body(), factory, ctx)) changed = true;
            } else if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases())
                    if (EliminateRecursive(c.body, factory, ctx)) changed = true;
                if (EliminateRecursive(sw->DefaultBody(), factory, ctx)) changed = true;
            }
            return changed;
        }

        // Forward declaration — defined in InlineCrossScopeSingleRef namespace.
        bool SNodeAlwaysTerminates(SNode *node);

        bool EliminateInSSeq(SSeq *seq, SNodeFactory &factory,
                             clang::ASTContext &ctx) {
            auto &children = seq->Children();
            bool changed = false;
            bool local_changed = true;
            // Cap restarts to avoid O(N²) on large SSeq (e.g., 100+
            // uncollapsed leaf nodes from a partially-structured graph).
            int restart_budget = 50;

            while (local_changed && restart_budget-- > 0) {
                local_changed = false;
                for (size_t i = 0; i + 1 < children.size(); ++i) {
                    // Find the next SLabel sibling, possibly skipping
                    // dead code after a terminating child.  Also unwrap
                    // one level of SSeq nesting to find wrapped labels.
                    SLabel *nxt_label = nullptr;
                    for (size_t k = i + 1; k < children.size() && !nxt_label; ++k) {
                        nxt_label = children[k]->dyn_cast<SLabel>();
                        if (!nxt_label) {
                            if (auto *ns = children[k]->dyn_cast<SSeq>()) {
                                if (!ns->Empty())
                                    nxt_label = (*ns)[0]->dyn_cast<SLabel>();
                            }
                        }
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
                        // Find the parent SSeq containing this SGoto and remove it
                        // If it's a direct child, remove from this SSeq
                        if (info.container == children[i]) {
                            seq->RemoveChild(i);
                        } else {
                            // It's nested — find the parent SSeq's last child
                            // and remove the SGoto from there
                            // Walk to find the innermost SSeq containing it
                            std::function<bool(SNode *)> remove_trailing;
                            remove_trailing = [&](SNode *n) -> bool {
                                if (auto *s = n->dyn_cast<SSeq>()) {
                                    auto &ch = s->Children();
                                    if (!ch.empty() && ch.back() == info.container) {
                                        s->RemoveChild(ch.size() - 1);
                                        return true;
                                    }
                                    if (!ch.empty()) return remove_trailing(ch.back());
                                }
                                if (auto *l = n->dyn_cast<SLabel>())
                                    return remove_trailing(l->Body());
                                return false;
                            };
                            remove_trailing(children[i]);
                        }
                        local_changed = true; changed = true;
                        break;
                    }

                    // --- Check clang::GotoStmt in SBlock ---
                    if (info.stmt) {
                        auto clang_tgt = ClangGotoTarget(info.stmt);
                        if (!clang_tgt.empty() && clang_tgt == next_name) {
                            // Remove the trailing GotoStmt from the SBlock
                            auto *blk = info.container->dyn_cast<SBlock>();
                            if (blk && !blk->Empty()) {
                                blk->Stmts().pop_back();
                                local_changed = true; changed = true;
                                break;
                            }
                        }

                        // --- Check clang::IfStmt with goto arm ---
                        if (auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(info.stmt)) {
                            auto else_tgt = ClangGotoTarget(ifs->getElse());
                            auto then_tgt = ClangGotoTarget(ifs->getThen());

                            if (!else_tgt.empty() && else_tgt == next_name) {
                                // else goto L; L: → drop else arm
                                auto loc = ifs->getIfLoc();
                                auto *new_if = clang::IfStmt::Create(
                                    ctx, loc, clang::IfStatementKind::Ordinary,
                                    nullptr, nullptr,
                                    ifs->getCond(), loc, loc,
                                    ifs->getThen(), loc, nullptr);
                                auto *blk = info.container->dyn_cast<SBlock>();
                                if (blk && !blk->Empty()) {
                                    blk->Stmts().back() = new_if;
                                    local_changed = true; changed = true;
                                    break;
                                }
                            }

                            if (!then_tgt.empty() && then_tgt == next_name
                                && !ifs->getElse()) {
                                // if(c) goto L; L: → nop (remove the if-stmt)
                                auto *blk = info.container->dyn_cast<SBlock>();
                                if (blk && !blk->Empty()) {
                                    blk->Stmts().pop_back();
                                    local_changed = true; changed = true;
                                    break;
                                }
                            }

                            if (!then_tgt.empty() && then_tgt == next_name
                                && ifs->getElse()) {
                                // if(c) goto L; else S; L: → if(!c) S
                                auto *neg = NegateExpr(ctx, ifs->getCond());
                                auto loc = ifs->getIfLoc();
                                auto *new_if = clang::IfStmt::Create(
                                    ctx, loc, clang::IfStatementKind::Ordinary,
                                    nullptr, nullptr,
                                    neg, loc, loc,
                                    ifs->getElse(), loc, nullptr);
                                auto *blk = info.container->dyn_cast<SBlock>();
                                if (blk && !blk->Empty()) {
                                    blk->Stmts().back() = new_if;
                                    local_changed = true; changed = true;
                                    break;
                                }
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
                                seq->RemoveChild(i);
                            } else {
                                // Nested — chase to the parent SSeq and
                                // remove the trailing SIfThenElse.
                                std::function<bool(SNode *)> remove_trailing;
                                remove_trailing = [&](SNode *n) -> bool {
                                    if (auto *s = n->dyn_cast<SSeq>()) {
                                        auto &ch = s->Children();
                                        if (!ch.empty() && ch.back() == info.container) {
                                            s->RemoveChild(ch.size() - 1);
                                            return true;
                                        }
                                        if (!ch.empty()) return remove_trailing(ch.back());
                                    }
                                    if (auto *l = n->dyn_cast<SLabel>())
                                        return remove_trailing(l->Body());
                                    return false;
                                };
                                remove_trailing(children[i]);
                            }
                            local_changed = true; changed = true;
                            break;
                        }
                        if (!then_tgt.empty() && then_tgt == next_name
                            && ite->ElseBranch() && ite->Cond()) {
                            auto *neg = NegateExpr(ctx, ite->Cond());
                            auto *new_ite = factory.Make<SIfThenElse>(
                                neg, ite->ElseBranch(), nullptr);
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

    bool EliminateGotoToNextLabel(SNode *root, SNodeFactory &factory,
                                  clang::ASTContext &ctx) {
        if (!root) return false;
        return EliminateRecursive(root, factory, ctx);
    }

    bool InlineResidualGotos(SNode *root, SNodeFactory &factory) {
        if (!root) return false;

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
            if (!InlineGotosRecursive(root, factory, refs))
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
    // target label live in *different* SSeq nodes.  When a label has
    // exactly one goto reference, its body always terminates, and no
    // fallthrough can reach the label, the body is moved (not cloned)
    // into the goto's slot and the label node is deleted.
    // ---------------------------------------------------------------

    namespace {

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

            if (auto *blk = node->dyn_cast<SBlock>()) {
                if (blk->Empty()) return false;
                auto *last = blk->Stmts().back();
                if (llvm::isa<clang::ReturnStmt>(last)
                    || llvm::isa<clang::BreakStmt>(last)
                    || llvm::isa<clang::ContinueStmt>(last)
                    || llvm::isa<clang::GotoStmt>(last))
                    return true;
                // clang::IfStmt where both arms terminate.
                if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(last)) {
                    auto *th = ifs->getThen();
                    auto *el = ifs->getElse();
                    if (!th || !el) return false;
                    auto terminates = [](clang::Stmt *s) {
                        if (llvm::isa<clang::ReturnStmt>(s)
                            || llvm::isa<clang::BreakStmt>(s)
                            || llvm::isa<clang::GotoStmt>(s))
                            return true;
                        if (auto *cs = llvm::dyn_cast<clang::CompoundStmt>(s)) {
                            if (cs->body_empty()) return false;
                            auto *b = cs->body_back();
                            return llvm::isa<clang::ReturnStmt>(b)
                                || llvm::isa<clang::BreakStmt>(b)
                                || llvm::isa<clang::GotoStmt>(b);
                        }
                        return false;
                    };
                    return terminates(th) && terminates(el);
                }
                return false;
            }
            if (auto *seq = node->dyn_cast<SSeq>()) {
                if (seq->Empty()) return false;
                return SNodeAlwaysTerminates(
                    seq->Children().back());
            }
            if (auto *lbl = node->dyn_cast<SLabel>()) {
                return SNodeAlwaysTerminates(lbl->Body());
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                // Both branches must terminate (else fallthrough
                // when one arm is missing).
                if (!ite->ThenBranch() || !ite->ElseBranch())
                    return false;
                return SNodeAlwaysTerminates(ite->ThenBranch())
                    && SNodeAlwaysTerminates(ite->ElseBranch());
            }
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                // Every case + default must terminate, and default
                // must be present (otherwise unmatched values fall
                // out of the switch).
                if (!sw->DefaultBody()) return false;
                if (!SNodeAlwaysTerminates(sw->DefaultBody()))
                    return false;
                for (auto &c : sw->Cases()) {
                    if (!SNodeAlwaysTerminates(c.body))
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

            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (auto *c : seq->Children())
                    if (SubtreeHasLabel(c)) return true;
                return false;
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                return SubtreeHasLabel(ite->ThenBranch())
                    || SubtreeHasLabel(ite->ElseBranch());
            }
            if (auto *w = node->dyn_cast<SWhile>())
                return SubtreeHasLabel(w->Body());
            if (auto *dw = node->dyn_cast<SDoWhile>())
                return SubtreeHasLabel(dw->Body());
            if (auto *f = node->dyn_cast<SFor>())
                return SubtreeHasLabel(f->Body());
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases())
                    if (SubtreeHasLabel(c.body)) return true;
                return SubtreeHasLabel(sw->DefaultBody());
            }
            // SBlock may hold a clang::LabelStmt — check too.
            if (auto *blk = node->dyn_cast<SBlock>()) {
                for (auto *s : blk->Stmts())
                    if (llvm::isa<clang::LabelStmt>(s)) return true;
                return false;
            }
            return false;
        }

        /// Locate an SLabel by name anywhere in the tree, returning
        /// the enclosing SSeq and the label's index within it.  Only
        /// labels that are *direct children of an SSeq* are returned —
        /// labels embedded as bodies of if/while/switch or as the sole
        /// child of an SLabel do not qualify (removing them requires
        /// parent-slot mutation which the caller cannot perform with a
        /// generic SSeq interface).
        struct LabelLoc {
            SSeq *parent = nullptr;
            size_t idx = 0;
        };

        bool FindLabelInSSeq(SNode *node, std::string_view name,
                             LabelLoc &out) {
            if (!node) return false;
            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < seq->Size(); ++i) {
                    if (auto *lbl = (*seq)[i]->dyn_cast<SLabel>()) {
                        if (lbl->Name() == name) {
                            out.parent = seq;
                            out.idx = i;
                            return true;
                        }
                    }
                }
                for (size_t i = 0; i < seq->Size(); ++i) {
                    if (FindLabelInSSeq((*seq)[i], name, out))
                        return true;
                }
                return false;
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                return FindLabelInSSeq(ite->ThenBranch(), name, out)
                    || FindLabelInSSeq(ite->ElseBranch(), name, out);
            }
            if (auto *w = node->dyn_cast<SWhile>())
                return FindLabelInSSeq(w->Body(), name, out);
            if (auto *dw = node->dyn_cast<SDoWhile>())
                return FindLabelInSSeq(dw->Body(), name, out);
            if (auto *f = node->dyn_cast<SFor>())
                return FindLabelInSSeq(f->Body(), name, out);
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases())
                    if (FindLabelInSSeq(c.body, name, out)) return true;
                return FindLabelInSSeq(sw->DefaultBody(), name, out);
            }
            if (auto *lbl = node->dyn_cast<SLabel>())
                return FindLabelInSSeq(lbl->Body(), name, out);
            return false;
        }

        /// Attempt to inline a single goto slot.  `get` returns the
        /// current node occupying the slot; `set` installs a
        /// replacement.  Returns true if the goto was inlined.
        bool TryInlineGotoSlot(
            std::function<SNode *()> get,
            std::function<void(SNode *)> set,
            SNode *root,
            std::unordered_map<std::string_view, int> &refs
        ) {
            auto *n = get();
            auto *g = n ? n->dyn_cast<SGoto>() : nullptr;
            if (!g) return false;

            auto target = g->Target();
            auto it = refs.find(target);
            if (it == refs.end() || it->second != 1) return false;

            LabelLoc loc;
            if (!FindLabelInSSeq(root, target, loc)) return false;

            auto *lbl = (*loc.parent)[loc.idx]->as<SLabel>();
            SNode *body = lbl->Body();
            if (!body) return false;

            if (!SNodeAlwaysTerminates(body)) return false;
            if (SubtreeHasLabel(body)) return false;

            // No fallthrough may reach the label in its original
            // position.  Require a preceding terminating sibling.
            if (loc.idx == 0) return false;
            if (!SNodeAlwaysTerminates((*loc.parent)[loc.idx - 1]))
                return false;

            // Splice: install body in goto's slot, remove original
            // label from its SSeq.
            set(body);
            loc.parent->RemoveChild(loc.idx);
            refs[target] = 0;
            return true;
        }

        /// Walk every SNode slot in the tree and try inlining any
        /// SGoto found there.  Returns true on first successful
        /// inline (caller reruns with fresh ref counts).
        bool CrossScopeInlineRecursive(
            SNode *node, SNode *root,
            std::unordered_map<std::string_view, int> &refs
        ) {
            if (!node) return false;

            if (auto *seq = node->dyn_cast<SSeq>()) {
                // Visit each child slot.  Recurse first so innermost
                // gotos are handled before parent-level scans.
                for (size_t i = 0; i < seq->Size(); ++i) {
                    if (CrossScopeInlineRecursive(
                            (*seq)[i], root, refs))
                        return true;
                }
                for (size_t i = 0; i < seq->Size(); ++i) {
                    if (TryInlineGotoSlot(
                            [seq, i]() { return (*seq)[i]; },
                            [seq, i](SNode *n) { seq->ReplaceChild(i, n); },
                            root, refs))
                        return true;
                }
                return false;
            }

            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                if (CrossScopeInlineRecursive(
                        ite->ThenBranch(), root, refs))
                    return true;
                if (CrossScopeInlineRecursive(
                        ite->ElseBranch(), root, refs))
                    return true;
                if (TryInlineGotoSlot(
                        [ite]() { return ite->ThenBranch(); },
                        [ite](SNode *n) { ite->SetThenBranch(n); },
                        root, refs))
                    return true;
                if (TryInlineGotoSlot(
                        [ite]() { return ite->ElseBranch(); },
                        [ite](SNode *n) { ite->SetElseBranch(n); },
                        root, refs))
                    return true;
                return false;
            }
            if (auto *w = node->dyn_cast<SWhile>()) {
                if (CrossScopeInlineRecursive(w->Body(), root, refs))
                    return true;
                return TryInlineGotoSlot(
                    [w]() { return w->Body(); },
                    [w](SNode *n) { w->SetBody(n); },
                    root, refs);
            }
            if (auto *dw = node->dyn_cast<SDoWhile>()) {
                if (CrossScopeInlineRecursive(dw->Body(), root, refs))
                    return true;
                return TryInlineGotoSlot(
                    [dw]() { return dw->Body(); },
                    [dw](SNode *n) { dw->SetBody(n); },
                    root, refs);
            }
            if (auto *f = node->dyn_cast<SFor>()) {
                if (CrossScopeInlineRecursive(f->Body(), root, refs))
                    return true;
                return TryInlineGotoSlot(
                    [f]() { return f->Body(); },
                    [f](SNode *n) { f->SetBody(n); },
                    root, refs);
            }
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (size_t ci = 0; ci < sw->Cases().size(); ++ci) {
                    auto &c = sw->Cases()[ci];
                    if (CrossScopeInlineRecursive(c.body, root, refs))
                        return true;
                }
                if (CrossScopeInlineRecursive(
                        sw->DefaultBody(), root, refs))
                    return true;
                for (size_t ci = 0; ci < sw->Cases().size(); ++ci) {
                    if (TryInlineGotoSlot(
                            [sw, ci]() { return sw->Cases()[ci].body; },
                            [sw, ci](SNode *n) {
                                sw->Cases()[ci].body = n;
                                if (n) n->SetParent(sw);
                            },
                            root, refs))
                        return true;
                }
                if (TryInlineGotoSlot(
                        [sw]() { return sw->DefaultBody(); },
                        [sw](SNode *n) { sw->SetDefaultBody(n); },
                        root, refs))
                    return true;
                return false;
            }
            if (auto *lbl = node->dyn_cast<SLabel>()) {
                if (CrossScopeInlineRecursive(lbl->Body(), root, refs))
                    return true;
                return TryInlineGotoSlot(
                    [lbl]() { return lbl->Body(); },
                    [lbl](SNode *n) { lbl->SetBody(n); },
                    root, refs);
            }
            return false;
        }

    } // anonymous namespace

    bool InlineCrossScopeSingleRef(SNode *root, SNodeFactory & /*factory*/) {
        if (!root) return false;

        std::unordered_map<std::string_view, int> refs;
        CountGotoRefs(root, refs);

        bool any_changed = false;
        // Fixed-point iteration — each pass eliminates at least one
        // goto, so the initial ref count bounds the loop.
        size_t max_passes = std::min(refs.size() + 1, size_t{20});
        for (size_t p = 0; p < max_passes; ++p) {
            if (!CrossScopeInlineRecursive(root, root, refs))
                break;
            any_changed = true;
            refs.clear();
            CountGotoRefs(root, refs);
        }
        return any_changed;
    }

    // ---------------------------------------------------------------
    // RemoveDeadSSeqChildren — strip unreachable SSeq children
    //
    // Walk SSeq nodes bottom-up.  When child[i] always terminates
    // (SNodeAlwaysTerminates), remove children [i+1..end) that are
    // NOT SLabel nodes.  SLabel nodes are preserved because they may
    // be goto targets from other scopes — removing them without a
    // full reference count would break goto/label pairing.
    // ---------------------------------------------------------------

    namespace {

        /// Check if an SNode contains any SLabel (directly or nested).
        bool ContainsLabel(SNode *node) {
            if (!node) return false;
            if (node->dyn_cast<SLabel>()) return true;
            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (auto *c : seq->Children())
                    if (ContainsLabel(c)) return true;
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                return ContainsLabel(ite->ThenBranch())
                    || ContainsLabel(ite->ElseBranch());
            }
            return false;
        }

        bool RemoveDeadInSSeq(SSeq *seq) {
            auto &children = seq->Children();
            bool changed = false;
            for (size_t i = 0; i + 1 < children.size(); ++i) {
                if (!SNodeAlwaysTerminates(children[i])) continue;
                // children[i] terminates — everything after it that
                // contains no labels is dead code.
                size_t j = i + 1;
                while (j < children.size()) {
                    if (ContainsLabel(children[j])) {
                        ++j; // keep — may contain goto targets
                    } else {
                        seq->RemoveChild(j);
                        changed = true;
                    }
                }
                break; // only process first terminator
            }
            return changed;
        }

        bool RemoveDeadRecursive(SNode *node) {
            if (!node) return false;
            bool changed = false;
            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < seq->Size(); ++i)
                    if (RemoveDeadRecursive((*seq)[i])) changed = true;
                if (RemoveDeadInSSeq(seq)) changed = true;
            } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                if (RemoveDeadRecursive(ite->ThenBranch())) changed = true;
                if (RemoveDeadRecursive(ite->ElseBranch())) changed = true;
            } else if (auto *w = node->dyn_cast<SWhile>()) {
                if (RemoveDeadRecursive(w->Body())) changed = true;
            } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
                if (RemoveDeadRecursive(dw->Body())) changed = true;
            } else if (auto *f = node->dyn_cast<SFor>()) {
                if (RemoveDeadRecursive(f->Body())) changed = true;
            } else if (auto *lbl = node->dyn_cast<SLabel>()) {
                if (RemoveDeadRecursive(lbl->Body())) changed = true;
            } else if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases())
                    if (RemoveDeadRecursive(c.body)) changed = true;
                if (RemoveDeadRecursive(sw->DefaultBody())) changed = true;
            }
            return changed;
        }

    } // anonymous namespace

    bool RemoveDeadSSeqChildren(SNode *root) {
        if (!root) return false;
        return RemoveDeadRecursive(root);
    }

    // ---------------------------------------------------------------
    // ConvertGotoToBreakContinue — replace gotos to loop exit/header
    //
    // Walks the SNode tree with a scope stack of enclosing loops.
    // For each trailing clang::GotoStmt in an SBlock:
    //   - target matches loop ExitLabel → replace with SBreak
    //   - target matches loop HeaderLabel → replace with SContinue
    // Also handles SGoto SNodes from SelectAndMarkGotoEdge.
    // ---------------------------------------------------------------

    namespace {

        struct LoopScope {
            std::string_view exit_label;
            std::string_view header_label;
        };

        /// Recursively convert gotos to break/continue in the SNode tree.
        /// The scope stack tracks enclosing loops.
        bool ConvertGotosInNode(
            SNode *node, SNodeFactory &factory,
            std::vector<LoopScope> &scopes
        ) {
            if (!node) return false;
            bool changed = false;

            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < seq->Size(); ++i) {
                    if (ConvertGotosInNode((*seq)[i], factory, scopes))
                        changed = true;
                }
                // Check SGoto children targeting loop labels.
                for (size_t i = 0; i < seq->Size(); ++i) {
                    auto *g = (*seq)[i]->dyn_cast<SGoto>();
                    if (!g) continue;
                    std::string_view target = g->Target();
                    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
                        if (!it->exit_label.empty() && target == it->exit_label) {
                            seq->ReplaceChild(i, factory.Make<SBreak>());
                            changed = true;
                            break;
                        }
                        if (!it->header_label.empty() && target == it->header_label) {
                            seq->ReplaceChild(i, factory.Make<SContinue>());
                            changed = true;
                            break;
                        }
                    }
                }
                // Check SBlock children whose trailing GotoStmt targets a loop label.
                // Since SBlock holds clang::Stmt* and we need to insert SNode (SBreak),
                // we split: remove the goto from the SBlock, insert SBreak after it.
                for (size_t i = 0; i < seq->Size(); ++i) {
                    auto *block = (*seq)[i]->dyn_cast<SBlock>();
                    if (!block || block->Empty()) continue;
                    auto *last = block->Stmts().back();
                    auto *goto_stmt = llvm::dyn_cast<clang::GotoStmt>(last);
                    if (!goto_stmt || !goto_stmt->getLabel()) continue;

                    std::string_view target = goto_stmt->getLabel()->getName();
                    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
                        if (!it->exit_label.empty() && target == it->exit_label) {
                            block->Stmts().pop_back();
                            seq->InsertChild(i + 1, factory.Make<SBreak>());
                            ++i; // skip the just-inserted SBreak
                            changed = true;
                            break;
                        }
                        if (!it->header_label.empty() && target == it->header_label) {
                            block->Stmts().pop_back();
                            seq->InsertChild(i + 1, factory.Make<SContinue>());
                            ++i; // skip the just-inserted SContinue
                            changed = true;
                            break;
                        }
                    }
                }
                return changed;
            }

            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                // Check if then/else branches target a loop label.
                // Handle both SGoto SNodes and SBlocks with trailing GotoStmt.
                // Track which arms were replaced to skip redundant recursion.
                bool then_replaced = false, else_replaced = false;
                for (int arm = 0; arm < 2; ++arm) {
                    SNode *branch = arm == 0
                        ? ite->ThenBranch() : ite->ElseBranch();
                    if (!branch) continue;

                    std::string_view target;

                    // Case 1: branch is SGoto SNode.
                    if (auto *g = branch->dyn_cast<SGoto>()) {
                        target = g->Target();
                    }
                    // Case 2: branch is SBlock with trailing GotoStmt.
                    else if (auto *block = branch->dyn_cast<SBlock>()) {
                        if (!block->Empty()) {
                            if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(
                                    block->Stmts().back()))
                                if (gs->getLabel())
                                    target = gs->getLabel()->getName();
                        }
                    }

                    if (target.empty()) continue;

                    for (auto sit = scopes.rbegin(); sit != scopes.rend(); ++sit) {
                        SNode *replacement = nullptr;
                        if (!sit->exit_label.empty() && target == sit->exit_label)
                            replacement = factory.Make<SBreak>();
                        else if (!sit->header_label.empty() && target == sit->header_label)
                            replacement = factory.Make<SContinue>();
                        if (!replacement) continue;

                        if (branch->isa<SGoto>()) {
                            // Direct replacement.
                            if (arm == 0) ite->SetThenBranch(replacement);
                            else ite->SetElseBranch(replacement);
                        } else {
                            // SBlock: remove trailing goto, wrap block + break/continue.
                            auto *block = branch->as<SBlock>();
                            block->Stmts().pop_back();
                            if (block->Empty()) {
                                // Block only had the goto — replace entirely.
                                if (arm == 0) ite->SetThenBranch(replacement);
                                else ite->SetElseBranch(replacement);
                            } else {
                                auto *seq = factory.Make<SSeq>();
                                seq->AddChild(block);
                                seq->AddChild(replacement);
                                if (arm == 0) ite->SetThenBranch(seq);
                                else ite->SetElseBranch(seq);
                            }
                        }
                        if (arm == 0) then_replaced = true;
                        else else_replaced = true;
                        changed = true;
                        break;
                    }
                }
                // Recurse only into branches that were NOT already replaced.
                if (!then_replaced) {
                    if (ConvertGotosInNode(ite->ThenBranch(), factory, scopes))
                        changed = true;
                }
                if (!else_replaced) {
                    if (ConvertGotosInNode(ite->ElseBranch(), factory, scopes))
                        changed = true;
                }
                return changed;
            }

            if (auto *w = node->dyn_cast<SWhile>()) {
                scopes.push_back({w->ExitLabel(), w->HeaderLabel()});
                if (ConvertGotosInNode(w->Body(), factory, scopes))
                    changed = true;
                scopes.pop_back();
                return changed;
            }
            if (auto *dw = node->dyn_cast<SDoWhile>()) {
                scopes.push_back({dw->ExitLabel(), dw->HeaderLabel()});
                if (ConvertGotosInNode(dw->Body(), factory, scopes))
                    changed = true;
                scopes.pop_back();
                return changed;
            }
            if (auto *f = node->dyn_cast<SFor>()) {
                scopes.push_back({f->ExitLabel(), f->HeaderLabel()});
                if (ConvertGotosInNode(f->Body(), factory, scopes))
                    changed = true;
                scopes.pop_back();
                return changed;
            }

            if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases()) {
                    if (ConvertGotosInNode(c.body, factory, scopes))
                        changed = true;
                }
                if (ConvertGotosInNode(sw->DefaultBody(), factory, scopes))
                    changed = true;
                return changed;
            }
            if (auto *lbl = node->dyn_cast<SLabel>()) {
                return ConvertGotosInNode(lbl->Body(), factory, scopes);
            }

            return false;
        }

    } // anonymous namespace

    bool ConvertGotoToBreakContinue(SNode *root, SNodeFactory &factory) {
        if (!root) return false;
        std::vector<LoopScope> scopes;
        return ConvertGotosInNode(root, factory, scopes);
    }

    // ---------------------------------------------------------------
    // ConvertGotoToReturn — replace goto-to-return patterns
    //
    // When an SGoto targets a label whose body is a single SReturn
    // (or an SSeq whose last child is SReturn with no other control
    // flow), replace the goto with a cloned SReturn.
    // ---------------------------------------------------------------

    namespace {

        /// Check if an SBlock is a safe, return-terminating block that
        /// can be duplicated at goto sites.  Must:
        ///   - end with clang::ReturnStmt
        ///   - have ≤ max_stmts statements
        ///   - contain no GotoStmt or LabelStmt (no new label references)
        bool IsSafeReturnBlock(SBlock *block, size_t max_stmts = 8) {
            if (!block || block->Empty()) return false;
            if (block->Size() > max_stmts) return false;
            if (!llvm::isa<clang::ReturnStmt>(block->Stmts().back()))
                return false;
            for (auto *s : block->Stmts()) {
                if (llvm::isa<clang::GotoStmt>(s)
                    || llvm::isa<clang::LabelStmt>(s))
                    return false;
            }
            return true;
        }

        /// Check if a label body is a return-terminating block safe for
        /// duplication.  Returns the SBlock if:
        ///   - body is an SBlock passing IsSafeReturnBlock, OR
        ///   - body is an SSeq whose last child is such an SBlock
        ///     AND all prior children are also safe SBlocks (no control flow).
        /// Returns nullptr otherwise.
        SBlock *ExtractReturnBlock(SNode *body) {
            if (!body) return nullptr;
            if (auto *block = body->dyn_cast<SBlock>()) {
                if (IsSafeReturnBlock(block)) return block;
                return nullptr;
            }
            if (auto *seq = body->dyn_cast<SSeq>()) {
                if (seq->Empty()) return nullptr;
                auto *last = seq->Children().back()->dyn_cast<SBlock>();
                if (last && IsSafeReturnBlock(last)) return last;
                return nullptr;
            }
            return nullptr;
        }

        /// Info about a label's position in the SNode tree.
        struct LabelEntry {
            SLabel *label;
            SSeq *parent_seq;   // parent SSeq (or nullptr if not in SSeq)
            size_t index;       // position in parent_seq
        };

        /// Build global label→LabelEntry map.
        void CollectLabels(SNode *node,
                           std::unordered_map<std::string_view, LabelEntry> &labels,
                           SSeq *parent = nullptr, size_t idx = 0) {
            if (!node) return;
            if (auto *lbl = node->dyn_cast<SLabel>()) {
                labels[lbl->Name()] = {lbl, parent, idx};
                CollectLabels(lbl->Body(), labels, parent, idx);
                return;
            }
            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < seq->Size(); ++i)
                    CollectLabels((*seq)[i], labels, seq, i);
                return;
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                CollectLabels(ite->ThenBranch(), labels);
                CollectLabels(ite->ElseBranch(), labels);
                return;
            }
            if (auto *w = node->dyn_cast<SWhile>()) {
                CollectLabels(w->Body(), labels);
                return;
            }
            if (auto *dw = node->dyn_cast<SDoWhile>()) {
                CollectLabels(dw->Body(), labels);
                return;
            }
            if (auto *f = node->dyn_cast<SFor>()) {
                CollectLabels(f->Body(), labels);
                return;
            }
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases()) CollectLabels(c.body, labels);
                CollectLabels(sw->DefaultBody(), labels);
                return;
            }
        }

        /// Collect the fallthrough tail stmts starting from a label's
        /// position in its parent SSeq.  Follows consecutive SLabel/SBlock
        /// siblings, collecting their stmts.  Returns true if the tail
        /// ends with ReturnStmt and total stmts ≤ max_stmts.  All stmts
        /// must be safe (no GotoStmt/LabelStmt except the label markers).
        bool CollectReturnTail(
            const LabelEntry &entry,
            std::vector<clang::Stmt *> &out,
            size_t max_stmts = 6
        ) {
            if (!entry.parent_seq) return false;
            auto *seq = entry.parent_seq;

            out.clear();
            for (size_t j = entry.index; j < seq->Size(); ++j) {
                auto *child = (*seq)[j];

                // Extract stmts from SLabel(body=SBlock) or bare SBlock.
                SBlock *block = nullptr;
                if (auto *lbl = child->dyn_cast<SLabel>()) {
                    if (lbl->Body())
                        block = lbl->Body()->dyn_cast<SBlock>();
                } else if (auto *blk = child->dyn_cast<SBlock>()) {
                    block = blk;
                } else {
                    break; // non-label/non-block sibling — stop
                }

                if (!block) break;

                for (auto *s : block->Stmts()) {
                    if (llvm::isa<clang::GotoStmt>(s)
                        || llvm::isa<clang::LabelStmt>(s))
                        return false; // unsafe stmt
                    out.push_back(s);
                    if (out.size() > max_stmts) return false;
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

        /// Check if a clang::Stmt always terminates (for dead-code trim).
        bool StmtTerminates(clang::Stmt *s) {
            if (!s) return false;
            if (llvm::isa<clang::ReturnStmt>(s) || llvm::isa<clang::BreakStmt>(s)
                || llvm::isa<clang::ContinueStmt>(s) || llvm::isa<clang::GotoStmt>(s))
                return true;
            if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(s))
                return ifs->getThen() && ifs->getElse()
                    && StmtTerminates(ifs->getThen())
                    && StmtTerminates(ifs->getElse());
            if (auto *cs = llvm::dyn_cast<clang::CompoundStmt>(s))
                return !cs->body_empty() && StmtTerminates(cs->body_back());
            return false;
        }

        /// Trim dead stmts after the first terminator in a stmt vector.
        void TrimDeadStmts(std::vector<clang::Stmt *> &stmts) {
            for (size_t i = 0; i < stmts.size(); ++i) {
                if (StmtTerminates(stmts[i]) && i + 1 < stmts.size()) {
                    stmts.erase(stmts.begin() + static_cast<ptrdiff_t>(i + 1),
                                stmts.end());
                    return;
                }
            }
        }

        /// Build a clang::ReturnStmt from an SReturn.
        clang::ReturnStmt *MakeReturn(clang::ASTContext &ctx, SReturn *sr) {
            auto loc = clang::SourceLocation();
            return clang::ReturnStmt::Create(ctx, loc, sr->Value(), nullptr);
        }

        /// Try to flatten a terminating SNode body into clang::Stmts.
        /// Handles patterns:
        ///   - SBlock ending in return → copy stmts
        ///   - SReturn → build ReturnStmt
        ///   - SSeq{SBlock..., SReturn} → collect stmts + return
        ///   - SSeq{SBlock..., SIfThenElse(c, term, term)} → stmts + IfStmt
        /// Returns false if the body is too complex to flatten.
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
            // SLabel: unwrap and flatten the inner body.
            if (auto *lbl = body->dyn_cast<SLabel>()) {
                return FlattenTerminatingBody(lbl->Body(), ctx, out, max_stmts);
            }
            if (auto *blk = body->dyn_cast<SBlock>()) {
                if (IsSafeReturnBlock(blk, max_stmts)) {
                    for (auto *s : blk->Stmts()) out.push_back(s);
                    return out.size() <= max_stmts;
                }
                return false;
            }
            if (auto *seq = body->dyn_cast<SSeq>()) {
                if (seq->Empty()) return false;
                // Collect SBlock prefix children.
                for (size_t i = 0; i + 1 < seq->Size(); ++i) {
                    auto *child = (*seq)[i]->dyn_cast<SBlock>();
                    if (!child) return false;
                    for (auto *s : child->Stmts()) {
                        if (llvm::isa<clang::GotoStmt>(s)
                            || llvm::isa<clang::LabelStmt>(s))
                            return false;
                        out.push_back(s);
                        if (out.size() > max_stmts) return false;
                    }
                }
                auto *last = seq->Children().back();
                // Last child is SReturn.
                if (auto *sr = last->dyn_cast<SReturn>()) {
                    out.push_back(MakeReturn(ctx, sr));
                    return out.size() <= max_stmts;
                }
                // Last child is SBlock ending in return.
                if (auto *lb = last->dyn_cast<SBlock>()) {
                    if (IsSafeReturnBlock(lb, max_stmts)) {
                        for (auto *s : lb->Stmts()) out.push_back(s);
                        return out.size() <= max_stmts;
                    }
                    return false;
                }
                // Last child is SIfThenElse where both arms terminate.
                if (auto *ite = last->dyn_cast<SIfThenElse>()) {
                    if (!ite->Cond() || !ite->ThenBranch())
                        return false;
                    // Flatten both arms recursively.
                    std::vector<clang::Stmt *> then_stmts, else_stmts;
                    if (!FlattenTerminatingBody(
                            ite->ThenBranch(), ctx, then_stmts, max_stmts))
                        return false;
                    clang::Stmt *then_body = nullptr;
                    if (then_stmts.size() == 1)
                        then_body = then_stmts[0];
                    else {
                        auto loc = clang::SourceLocation();
                        then_body = clang::CompoundStmt::Create(
                            ctx, then_stmts, clang::FPOptionsOverride(),
                            loc, loc);
                    }
                    clang::Stmt *else_body = nullptr;
                    if (ite->ElseBranch()) {
                        if (!FlattenTerminatingBody(
                                ite->ElseBranch(), ctx, else_stmts, max_stmts))
                            return false;
                        if (else_stmts.size() == 1)
                            else_body = else_stmts[0];
                        else {
                            auto loc = clang::SourceLocation();
                            else_body = clang::CompoundStmt::Create(
                                ctx, else_stmts, clang::FPOptionsOverride(),
                                loc, loc);
                        }
                    }
                    auto loc = clang::SourceLocation();
                    auto *new_if = clang::IfStmt::Create(
                        ctx, loc, clang::IfStatementKind::Ordinary,
                        nullptr, nullptr, ite->Cond(), loc, loc,
                        then_body, loc, else_body);
                    out.push_back(new_if);
                    return out.size() <= max_stmts;
                }
                return false;
            }
            return false;
        }

        /// Check if the last stmt in an SBlock is a GotoStmt targeting
        /// a return-terminating label.  If so, replace with the return stmts.
        /// First tries the label's direct body (ExtractReturnBlock), then
        /// falls back to collecting the fallthrough tail through consecutive
        /// labels in the parent SSeq (CollectReturnTail).
        bool TryReplaceBlockTrailingGoto(
            SBlock *block, SNodeFactory &/*factory*/,
            clang::ASTContext &ctx,
            const std::unordered_map<std::string_view, LabelEntry> &labels
        ) {
            if (!block || block->Empty()) return false;
            auto *last = block->Stmts().back();
            auto *goto_stmt = llvm::dyn_cast<clang::GotoStmt>(last);
            if (!goto_stmt || !goto_stmt->getLabel()) return false;

            std::string_view target = goto_stmt->getLabel()->getName();
            auto it = labels.find(target);
            if (it == labels.end()) return false;

            // Try 1: label body is directly a return-terminating block.
            auto *ret_block = ExtractReturnBlock(it->second.label->Body());
            if (ret_block && ret_block != block) {
                block->Stmts().pop_back();
                for (auto *s : ret_block->Stmts())
                    block->AddStmt(s);
                TrimDeadStmts(block->Stmts());
                return true;
            }

            // Try 2: follow the fallthrough chain through consecutive
            // labels in the parent SSeq to collect a return-terminating tail.
            std::vector<clang::Stmt *> tail;
            if (CollectReturnTail(it->second, tail)) {
                block->Stmts().pop_back();
                for (auto *s : tail)
                    block->AddStmt(s);
                TrimDeadStmts(block->Stmts());
                return true;
            }

            // Try 3: resolve goto chains (label body ends in another
            // goto → follow until a return-terminating body).
            std::vector<clang::Stmt *> chain;
            if (ResolveGotoChain(target, labels, ctx, chain)) {
                block->Stmts().pop_back();
                for (auto *s : chain)
                    block->AddStmt(s);
                TrimDeadStmts(block->Stmts());
                return true;
            }

            return false;
        }

        /// Follow a goto chain through labels collecting stmts until
        /// a return-terminating body is reached.  Each link in the
        /// chain is a label whose SBlock body ends in a GotoStmt to
        /// the next link.  Returns true if a return-terminated
        /// sequence was assembled within the budget.
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

                SNode *body = it->second.label->Body();
                if (!body) return false;

                // Try direct: body is return-terminating (no goto inside).
                auto *ret_block = ExtractReturnBlock(body);
                if (ret_block) {
                    for (auto *s : ret_block->Stmts())
                        out.push_back(s);
                    TrimDeadStmts(out);
                    return out.size() <= max_stmts;
                }

                // Try fallthrough tail from label's SSeq position.
                std::vector<clang::Stmt *> tail;
                if (CollectReturnTail(it->second, tail)) {
                    for (auto *s : tail) out.push_back(s);
                    TrimDeadStmts(out);
                    return out.size() <= max_stmts;
                }

                // Try flattening complex terminating body (SIfThenElse etc.)
                std::vector<clang::Stmt *> flat;
                if (FlattenTerminatingBody(body, ctx, flat, max_stmts)) {
                    for (auto *s : flat) out.push_back(s);
                    TrimDeadStmts(out);
                    return out.size() <= max_stmts;
                }

                // Check if body ends in a goto (passthrough).  The goto
                // may be a clang::GotoStmt trailing an SBlock, or an
                // SGoto SNode at the end of an SSeq.  Collect the
                // non-goto stmts, then follow the chain.
                std::string_view next_target;

                if (auto *b = body->dyn_cast<SBlock>()) {
                    if (b->Empty()) return false;
                    auto *last_s = b->Stmts().back();
                    auto *gs2 = llvm::dyn_cast<clang::GotoStmt>(last_s);
                    if (!gs2 || !gs2->getLabel()) return false;
                    for (size_t i = 0; i + 1 < b->Size(); ++i) {
                        if (llvm::isa<clang::LabelStmt>(b->Stmts()[i]))
                            return false;
                        out.push_back(b->Stmts()[i]);
                        if (out.size() > max_stmts) return false;
                    }
                    next_target = gs2->getLabel()->getName();
                } else if (auto *seq = body->dyn_cast<SSeq>()) {
                    if (seq->Empty()) return false;
                    // Last child may be SGoto or SBlock with trailing goto.
                    auto *last_child = seq->Children().back();
                    if (auto *sg = last_child->dyn_cast<SGoto>()) {
                        next_target = sg->Target();
                    } else if (auto *lb = last_child->dyn_cast<SBlock>()) {
                        if (lb->Empty()) return false;
                        auto *gs2 = llvm::dyn_cast<clang::GotoStmt>(
                            lb->Stmts().back());
                        if (!gs2 || !gs2->getLabel()) return false;
                        next_target = gs2->getLabel()->getName();
                        // Collect non-goto stmts from this trailing block.
                        for (size_t i = 0; i + 1 < lb->Size(); ++i) {
                            if (llvm::isa<clang::LabelStmt>(lb->Stmts()[i]))
                                return false;
                            out.push_back(lb->Stmts()[i]);
                            if (out.size() > max_stmts) return false;
                        }
                    } else {
                        return false;
                    }
                    // Collect stmts from earlier SSeq children (SBlocks).
                    for (size_t ci = 0; ci + 1 < seq->Size(); ++ci) {
                        auto *child_blk = (*seq)[ci]->dyn_cast<SBlock>();
                        if (!child_blk) return false;
                        for (auto *s : child_blk->Stmts()) {
                            if (llvm::isa<clang::LabelStmt>(s))
                                return false;
                            out.push_back(s);
                            if (out.size() > max_stmts) return false;
                        }
                    }
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
            auto loc = clang::SourceLocation();
            return clang::CompoundStmt::Create(
                ctx, stmts, clang::FPOptionsOverride(), loc, loc);
        }

        /// Scan all stmts in an SBlock for clang::IfStmt whose then/else
        /// arm is a GotoStmt targeting a return-terminating label.
        /// Replaces the goto arm with the label's body wrapped in a
        /// CompoundStmt.  Handles bare gotos and CompoundStmt-wrapped gotos.
        bool TryReplaceIfGuardedGotos(
            SBlock *block,
            clang::ASTContext &ctx,
            const std::unordered_map<std::string_view, LabelEntry> &labels
        ) {
            if (!block || block->Empty()) return false;
            bool changed = false;

            for (size_t i = 0; i < block->Stmts().size(); ++i) {
                auto *ifs = llvm::dyn_cast<clang::IfStmt>(block->Stmts()[i]);
                if (!ifs) continue;

                // Check then-arm.
                auto then_info = ExtractIfArmGoto(ifs->getThen());
                if (then_info.gs) {
                    std::vector<clang::Stmt *> body;
                    if (ResolveGotoReturnBody(then_info.gs, labels, ctx, body)) {
                        auto *replacement = BuildReplacementArm(
                            ctx, then_info.compound, body);
                        ifs->setThen(replacement);
                        changed = true;
                    }
                }

                // Check else-arm.
                auto else_info = ExtractIfArmGoto(ifs->getElse());
                if (else_info.gs) {
                    std::vector<clang::Stmt *> body;
                    if (ResolveGotoReturnBody(else_info.gs, labels, ctx, body)) {
                        auto *replacement = BuildReplacementArm(
                            ctx, else_info.compound, body);
                        ifs->setElse(replacement);
                        changed = true;
                    }
                }
            }
            return changed;
        }

        /// Replace goto-to-return in a single node, recursing into children.
        bool ReplaceGotoWithReturn(
            SNode *node, SNodeFactory &factory,
            clang::ASTContext &ctx,
            const std::unordered_map<std::string_view, LabelEntry> &labels
        ) {
            if (!node) return false;
            bool changed = false;

            // SBlock: check trailing goto AND non-trailing if-guarded gotos.
            if (auto *block = node->dyn_cast<SBlock>()) {
                if (TryReplaceBlockTrailingGoto(block, factory, ctx, labels))
                    changed = true;
                if (TryReplaceIfGuardedGotos(block, ctx, labels))
                    changed = true;
                return changed;
            }

            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < seq->Size(); ++i) {
                    if (ReplaceGotoWithReturn((*seq)[i], factory, ctx, labels))
                        changed = true;
                }
                // Also check SGoto SNode children (from SelectAndMarkGotoEdge).
                for (size_t i = 0; i < seq->Size(); ++i) {
                    auto *g = (*seq)[i]->dyn_cast<SGoto>();
                    if (!g) continue;
                    auto it = labels.find(g->Target());
                    if (it == labels.end()) continue;

                    // Try direct return block first.
                    auto *ret_block = ExtractReturnBlock(
                        it->second.label->Body());
                    if (ret_block) {
                        auto *clone = factory.Make<SBlock>();
                        clone->SetLabel(ret_block->Label());
                        for (auto *s : ret_block->Stmts())
                            clone->AddStmt(s);
                        seq->ReplaceChild(i, clone);
                        changed = true;
                        continue;
                    }
                    // Try fallthrough tail.
                    std::vector<clang::Stmt *> tail;
                    if (CollectReturnTail(it->second, tail)) {
                        auto *clone = factory.Make<SBlock>();
                        for (auto *s : tail)
                            clone->AddStmt(s);
                        seq->ReplaceChild(i, clone);
                        changed = true;
                        continue;
                    }
                    // Try goto chain resolution.
                    std::vector<clang::Stmt *> chain;
                    if (ResolveGotoChain(g->Target(), labels, ctx, chain)) {
                        auto *clone = factory.Make<SBlock>();
                        for (auto *s : chain)
                            clone->AddStmt(s);
                        seq->ReplaceChild(i, clone);
                        changed = true;
                    }
                }
                return changed;
            }

            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                if (ReplaceGotoWithReturn(ite->ThenBranch(), factory, ctx, labels))
                    changed = true;
                if (ReplaceGotoWithReturn(ite->ElseBranch(), factory, ctx, labels))
                    changed = true;
                // Check if then/else branch is a bare SGoto to a
                // chain-resolvable target.
                if (auto *tg = ite->ThenBranch()
                        ? ite->ThenBranch()->dyn_cast<SGoto>() : nullptr) {
                    std::vector<clang::Stmt *> chain;
                    if (ResolveGotoChain(tg->Target(), labels, ctx, chain)) {
                        auto *clone = factory.Make<SBlock>();
                        for (auto *s : chain) clone->AddStmt(s);
                        ite->SetThenBranch(clone);
                        changed = true;
                    }
                }
                if (auto *eg = ite->ElseBranch()
                        ? ite->ElseBranch()->dyn_cast<SGoto>() : nullptr) {
                    std::vector<clang::Stmt *> chain;
                    if (ResolveGotoChain(eg->Target(), labels, ctx, chain)) {
                        auto *clone = factory.Make<SBlock>();
                        for (auto *s : chain) clone->AddStmt(s);
                        ite->SetElseBranch(clone);
                        changed = true;
                    }
                }
                return changed;
            }
            if (auto *w = node->dyn_cast<SWhile>()) {
                return ReplaceGotoWithReturn(w->Body(), factory, ctx, labels);
            }
            if (auto *dw = node->dyn_cast<SDoWhile>()) {
                return ReplaceGotoWithReturn(dw->Body(), factory, ctx, labels);
            }
            if (auto *f = node->dyn_cast<SFor>()) {
                return ReplaceGotoWithReturn(f->Body(), factory, ctx, labels);
            }
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases()) {
                    if (ReplaceGotoWithReturn(c.body, factory, ctx, labels))
                        changed = true;
                }
                if (ReplaceGotoWithReturn(sw->DefaultBody(), factory, ctx, labels))
                    changed = true;
                return changed;
            }
            if (auto *lbl = node->dyn_cast<SLabel>()) {
                return ReplaceGotoWithReturn(lbl->Body(), factory, ctx, labels);
            }

            return false;
        }

    } // anonymous namespace

    bool ConvertGotoToReturn(SNode *root, SNodeFactory &factory,
                             clang::ASTContext &ctx) {
        if (!root) return false;

        std::unordered_map<std::string_view, LabelEntry> labels;
        CollectLabels(root, labels);

        // Single pass: chain resolution (ResolveGotoChain) already
        // follows multi-hop goto chains in one shot, so iterating
        // here would duplicate epilogue bodies exponentially.
        bool any_changed = ReplaceGotoWithReturn(root, factory, ctx, labels);

        return any_changed;
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
        // targets inside SBlock stmts.
        void CountAllGotoRefs(
            const SNode *node,
            std::unordered_set<std::string_view> &refs
        ) {
            if (!node) return;
            if (auto *g = node->dyn_cast<SGoto>()) {
                refs.insert(g->Target());
                return;
            }
            if (auto *blk = node->dyn_cast<SBlock>()) {
                // Recursively walk clang::Stmt trees for GotoStmt.
                std::function<void(clang::Stmt *)> walk =
                    [&](clang::Stmt *s) {
                    if (!s) return;
                    if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(s)) {
                        refs.insert(gs->getLabel()->getName());
                        return;
                    }
                    for (auto *child : s->children()) walk(child);
                };
                for (auto *s : blk->Stmts()) walk(s);
                return;
            }
            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (auto *c : seq->Children()) CountAllGotoRefs(c, refs);
                return;
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                CountAllGotoRefs(ite->ThenBranch(), refs);
                CountAllGotoRefs(ite->ElseBranch(), refs);
                return;
            }
            if (auto *w = node->dyn_cast<SWhile>()) {
                CountAllGotoRefs(w->Body(), refs); return;
            }
            if (auto *dw = node->dyn_cast<SDoWhile>()) {
                CountAllGotoRefs(dw->Body(), refs); return;
            }
            if (auto *f = node->dyn_cast<SFor>()) {
                CountAllGotoRefs(f->Body(), refs); return;
            }
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases()) CountAllGotoRefs(c.body, refs);
                CountAllGotoRefs(sw->DefaultBody(), refs);
                return;
            }
            if (auto *lbl = node->dyn_cast<SLabel>()) {
                CountAllGotoRefs(lbl->Body(), refs);
                return;
            }
        }

        // Remove unreferenced SLabel children from SSeq nodes.
        bool RemoveDeadLabelsInSeq(
            SSeq *seq,
            const std::unordered_set<std::string_view> &refs
        ) {
            bool changed = false;
            for (size_t i = 0; i < seq->Size(); ) {
                auto *lbl = (*seq)[i]->dyn_cast<SLabel>();
                if (lbl && refs.count(lbl->Name()) == 0) {
                    seq->RemoveChild(i);
                    changed = true;
                } else {
                    ++i;
                }
            }
            return changed;
        }

        bool RemoveDeadLabelsRecursive(
            SNode *node,
            const std::unordered_set<std::string_view> &refs
        ) {
            if (!node) return false;
            bool changed = false;
            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < seq->Size(); ++i)
                    if (RemoveDeadLabelsRecursive((*seq)[i], refs))
                        changed = true;
                if (RemoveDeadLabelsInSeq(seq, refs))
                    changed = true;
                return changed;
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                if (RemoveDeadLabelsRecursive(ite->ThenBranch(), refs)) changed = true;
                if (RemoveDeadLabelsRecursive(ite->ElseBranch(), refs)) changed = true;
                return changed;
            }
            if (auto *w = node->dyn_cast<SWhile>())
                return RemoveDeadLabelsRecursive(w->Body(), refs);
            if (auto *dw = node->dyn_cast<SDoWhile>())
                return RemoveDeadLabelsRecursive(dw->Body(), refs);
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases())
                    if (RemoveDeadLabelsRecursive(c.body, refs)) changed = true;
                if (RemoveDeadLabelsRecursive(sw->DefaultBody(), refs)) changed = true;
                return changed;
            }
            if (auto *lbl = node->dyn_cast<SLabel>())
                return RemoveDeadLabelsRecursive(lbl->Body(), refs);
            return false;
        }

    } // anonymous namespace

    bool RemoveUnreferencedLabels(SNode *root, SNodeFactory &) {
        if (!root) return false;
        std::unordered_set<std::string_view> refs;
        CountAllGotoRefs(root, refs);
        return RemoveDeadLabelsRecursive(root, refs);
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

        /// Count approximate clang::Stmt-equivalent size of a subtree.
        size_t CountCloneStmts(const SNode *node) {
            if (!node) return 0;
            if (auto *blk = node->dyn_cast<SBlock>())
                return blk->Size();
            if (auto *seq = node->dyn_cast<SSeq>()) {
                size_t n = 0;
                for (auto *c : seq->Children()) {
                    n += CountCloneStmts(c);
                    if (n > kMaxCloneStmts) return n;
                }
                return n;
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                return 1 + CountCloneStmts(ite->ThenBranch())
                         + CountCloneStmts(ite->ElseBranch());
            }
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                size_t n = 1;
                for (auto &c : sw->Cases()) {
                    n += CountCloneStmts(c.body);
                    if (n > kMaxCloneStmts) return n;
                }
                n += CountCloneStmts(sw->DefaultBody());
                return n;
            }
            if (auto *lbl = node->dyn_cast<SLabel>())
                return 1 + CountCloneStmts(lbl->Body());
            return 1; // goto / break / continue / return
        }

        /// True iff \p node is safe to deep-clone: contains no label
        /// definitions (SLabel or clang::LabelStmt), no loops, and no
        /// SFor (loops would be duplicated, changing complexity).
        bool SubtreeIsSafeToClone(const SNode *node) {
            if (!node) return true;
            if (auto *blk = node->dyn_cast<SBlock>()) {
                std::function<bool(const clang::Stmt *)> has_label =
                    [&](const clang::Stmt *st) -> bool {
                        if (!st) return false;
                        if (llvm::isa<clang::LabelStmt>(st)) return true;
                        for (const auto *c : st->children())
                            if (has_label(c)) return true;
                        return false;
                    };
                for (auto *s : blk->Stmts())
                    if (has_label(s)) return false;
                return true;
            }
            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (auto *c : seq->Children())
                    if (!SubtreeIsSafeToClone(c)) return false;
                return true;
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                return SubtreeIsSafeToClone(ite->ThenBranch())
                    && SubtreeIsSafeToClone(ite->ElseBranch());
            }
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases())
                    if (!SubtreeIsSafeToClone(c.body)) return false;
                return SubtreeIsSafeToClone(sw->DefaultBody());
            }
            // Reject SLabel (would duplicate definition) and all loops.
            switch (node->Kind()) {
                case SNodeKind::kLabel:
                case SNodeKind::kWhile:
                case SNodeKind::kDoWhile:
                case SNodeKind::kFor:
                    return false;
                default:
                    return true; // SGoto / SBreak / SContinue / SReturn
            }
        }

        /// Deep-clone a subtree.  Pre-condition: SubtreeIsSafeToClone(src).
        /// clang::Stmt* pointers are shared — clang::Stmt has no single
        /// parent, so aliasing the same stmt in two SBlocks is legal.
        SNode *CloneSNode(SNode *src, SNodeFactory &factory) {
            if (!src) return nullptr;
            if (auto *blk = src->dyn_cast<SBlock>()) {
                auto *out = factory.Make<SBlock>();
                out->SetLabel(blk->Label());
                for (auto *s : blk->Stmts()) out->AddStmt(s);
                return out;
            }
            if (auto *seq = src->dyn_cast<SSeq>()) {
                auto *out = factory.Make<SSeq>();
                for (auto *c : seq->Children()) {
                    auto *cc = CloneSNode(c, factory);
                    if (cc) out->AddChild(cc);
                }
                return out;
            }
            if (auto *ite = src->dyn_cast<SIfThenElse>()) {
                return factory.Make<SIfThenElse>(
                    ite->Cond(),
                    CloneSNode(ite->ThenBranch(), factory),
                    CloneSNode(ite->ElseBranch(), factory));
            }
            if (auto *sw = src->dyn_cast<SSwitch>()) {
                auto *out = factory.Make<SSwitch>(sw->Discriminant());
                for (auto &c : sw->Cases())
                    out->AddCase(c.value, CloneSNode(c.body, factory));
                if (sw->DefaultBody())
                    out->SetDefaultBody(CloneSNode(sw->DefaultBody(), factory));
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
            if (auto *blk = node->dyn_cast<SBlock>()) {
                if (blk->Empty()) return true;
                auto *last = blk->Stmts().back();
                if (llvm::isa<clang::ReturnStmt>(last)
                    || llvm::isa<clang::BreakStmt>(last)
                    || llvm::isa<clang::ContinueStmt>(last)
                    || llvm::isa<clang::GotoStmt>(last))
                    return false;
                return true;
            }
            if (auto *seq = node->dyn_cast<SSeq>()) {
                if (seq->Empty()) return true;
                return NeedsTerminatorBreak(seq->Children().back());
            }
            return true;
        }

        /// Return the trailing goto target name if \p body ends in a
        /// goto (bare SGoto, SBlock with trailing GotoStmt, SSeq whose
        /// last child is one of those, or SLabel wrapping any of those).
        /// Empty otherwise.
        std::string_view TrailingGotoTarget(const SNode *body) {
            if (!body) return {};
            if (auto *g = body->dyn_cast<SGoto>()) return g->Target();
            if (auto *blk = body->dyn_cast<SBlock>()) {
                if (blk->Empty()) return {};
                if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(
                        blk->Stmts().back()))
                    return gs->getLabel()->getName();
                return {};
            }
            if (auto *seq = body->dyn_cast<SSeq>()) {
                if (seq->Empty()) return {};
                return TrailingGotoTarget(seq->Children().back());
            }
            if (auto *lbl = body->dyn_cast<SLabel>()) {
                return TrailingGotoTarget(lbl->Body());
            }
            return {};
        }

        /// Try to clone \p target_label's body.  Returns nullptr if
        /// unsafe or too large.
        SNode *TryCloneLabelBody(
            std::string_view target_label,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            SNodeFactory &factory
        ) {
            auto it = labels.find(target_label);
            if (it == labels.end()) return nullptr;
            SNode *label_body = it->second.label->Body();
            if (!label_body) return nullptr;
            if (CountCloneStmts(label_body) > kMaxCloneStmts) return nullptr;
            if (!SubtreeIsSafeToClone(label_body)) return nullptr;
            return CloneSNode(label_body, factory);
        }

        /// Build a replacement case body by splicing \p clone in place
        /// of the trailing goto in \p existing.  Returns the new body
        /// node (possibly a fresh SSeq) or nullptr on failure.
        ///
        /// Handles:
        ///   - existing is SGoto → clone (+ break)
        ///   - existing is SBlock ending in GotoStmt → SSeq(trimmed_block, clone, break?)
        ///   - existing is SSeq ending in SGoto → SSeq(prefix_children..., clone, break?)
        ///   - existing is SSeq ending in SBlock-with-trailing-goto → recurse
        SNode *BuildSplicedBody(
            SNode *existing, SNode *clone, SNodeFactory &factory
        ) {
            if (!existing || !clone) return nullptr;

            bool need_break = NeedsTerminatorBreak(clone);
            auto wrap_with_break = [&](SNode *n) -> SNode * {
                if (!need_break) return n;
                auto *seq = factory.Make<SSeq>();
                seq->AddChild(n);
                seq->AddChild(factory.Make<SBreak>());
                return seq;
            };

            // Pure SGoto — replace with clone (+ break).
            if (existing->dyn_cast<SGoto>()) {
                return wrap_with_break(clone);
            }

            // SBlock whose last stmt is GotoStmt.
            if (auto *blk = existing->dyn_cast<SBlock>()) {
                if (blk->Empty()) return nullptr;
                if (!llvm::isa<clang::GotoStmt>(blk->Stmts().back()))
                    return nullptr;
                if (blk->Size() == 1) {
                    // Pure redirect wrapped in a block — replace whole.
                    return wrap_with_break(clone);
                }
                // Build a new SBlock without the trailing goto, then
                // wrap in SSeq(new_block, clone, break?).
                auto *trimmed = factory.Make<SBlock>();
                trimmed->SetLabel(blk->Label());
                for (size_t i = 0; i + 1 < blk->Size(); ++i)
                    trimmed->AddStmt(blk->Stmts()[i]);
                auto *seq = factory.Make<SSeq>();
                seq->AddChild(trimmed);
                seq->AddChild(clone);
                if (need_break) seq->AddChild(factory.Make<SBreak>());
                return seq;
            }

            // SLabel wrapping any of the above: splice inside, keep the
            // label node.  Mutate via SetBody so the label identity is
            // preserved (outside gotos may still reference this label).
            if (auto *lbl = existing->dyn_cast<SLabel>()) {
                SNode *inner = BuildSplicedBody(lbl->Body(), clone, factory);
                if (!inner) return nullptr;
                lbl->SetBody(inner);
                return lbl;
            }

            // SSeq whose last child carries the trailing goto.
            if (auto *seq = existing->dyn_cast<SSeq>()) {
                if (seq->Empty()) return nullptr;
                SNode *last = seq->Children().back();

                // Case A: last child is bare SGoto → drop it.
                if (last->dyn_cast<SGoto>()) {
                    auto *out = factory.Make<SSeq>();
                    for (size_t i = 0; i + 1 < seq->Size(); ++i)
                        out->AddChild(seq->Children()[i]);
                    out->AddChild(clone);
                    if (need_break) out->AddChild(factory.Make<SBreak>());
                    return out;
                }
                // Case B: last child is SBlock with trailing GotoStmt.
                if (auto *blk = last->dyn_cast<SBlock>()) {
                    if (!blk->Empty()
                        && llvm::isa<clang::GotoStmt>(blk->Stmts().back())) {
                        auto *out = factory.Make<SSeq>();
                        for (size_t i = 0; i + 1 < seq->Size(); ++i)
                            out->AddChild(seq->Children()[i]);
                        if (blk->Size() > 1) {
                            auto *trimmed = factory.Make<SBlock>();
                            trimmed->SetLabel(blk->Label());
                            for (size_t i = 0; i + 1 < blk->Size(); ++i)
                                trimmed->AddStmt(blk->Stmts()[i]);
                            out->AddChild(trimmed);
                        }
                        out->AddChild(clone);
                        if (need_break) out->AddChild(factory.Make<SBreak>());
                        return out;
                    }
                }
                return nullptr;
            }

            return nullptr;
        }

        bool DuplicateInSwitch(
            SSwitch *sw, SNodeFactory &factory,
            const std::unordered_map<std::string_view, LabelEntry> &labels
        ) {
            bool changed = false;
            for (auto &c : sw->Cases()) {
                auto target = TrailingGotoTarget(c.body);
                if (target.empty()) continue;
                SNode *clone = TryCloneLabelBody(target, labels, factory);
                if (!clone) continue;
                SNode *spliced = BuildSplicedBody(c.body, clone, factory);
                if (!spliced) continue;
                c.body = spliced;
                spliced->SetParent(sw);
                changed = true;
            }
            if (SNode *def = sw->DefaultBody()) {
                auto target = TrailingGotoTarget(def);
                if (!target.empty()) {
                    SNode *clone = TryCloneLabelBody(target, labels, factory);
                    if (clone) {
                        if (SNode *spliced = BuildSplicedBody(
                                def, clone, factory)) {
                            sw->SetDefaultBody(spliced);
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
            if (auto *sw = node->dyn_cast<SSwitch>()) {
                if (DuplicateInSwitch(sw, factory, labels)) changed = true;
                for (auto &c : sw->Cases())
                    if (WalkAndDuplicate(c.body, factory, labels))
                        changed = true;
                if (WalkAndDuplicate(sw->DefaultBody(), factory, labels))
                    changed = true;
                return changed;
            }
            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (auto *c : seq->Children())
                    if (WalkAndDuplicate(c, factory, labels))
                        changed = true;
                return changed;
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                if (WalkAndDuplicate(ite->ThenBranch(), factory, labels))
                    changed = true;
                if (WalkAndDuplicate(ite->ElseBranch(), factory, labels))
                    changed = true;
                return changed;
            }
            if (auto *w = node->dyn_cast<SWhile>())
                return WalkAndDuplicate(w->Body(), factory, labels);
            if (auto *dw = node->dyn_cast<SDoWhile>())
                return WalkAndDuplicate(dw->Body(), factory, labels);
            if (auto *f = node->dyn_cast<SFor>())
                return WalkAndDuplicate(f->Body(), factory, labels);
            if (auto *lbl = node->dyn_cast<SLabel>())
                return WalkAndDuplicate(lbl->Body(), factory, labels);
            return false;
        }

        /// Debug assertion: every remaining goto target resolves to a
        /// live SLabel.  Runs only in debug builds; aborts on failure
        /// so misbehaviour is caught before a broken TU ships.
        void VerifyGotoLabelPairing(SNode *root) {
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

    bool DuplicateSwitchCaseTargets(SNode *root, SNodeFactory &factory) {
        if (!root) return false;
        bool any_changed = false;
        // Re-scan labels on each iteration: a previous duplication may
        // expose new opportunities (e.g., cloning a label body that
        // itself contained a case goto to another label).  Bound the
        // loop to avoid pathological growth.
        for (int pass = 0; pass < 4; ++pass) {
            std::unordered_map<std::string_view, LabelEntry> labels;
            CollectLabels(root, labels);
            if (!WalkAndDuplicate(root, factory, labels)) break;
            any_changed = true;
        }
        if (any_changed) VerifyGotoLabelPairing(root);
        return any_changed;
    }

} // namespace patchestry::ast
