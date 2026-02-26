/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "CfgFoldInternal.hpp"

namespace patchestry::ast {
namespace detail {

    // ===============================================================
    // CFG Simplification Passes (Ghidra-style pre-structuring)
    //
    // These run BEFORE the main collapse loop to simplify the graph,
    // matching what Ghidra achieves via ActionUnreachable,
    // ActionDeterminedBranch, and ActionDeadCode before
    // ActionBlockStructure.
    // ===============================================================

    // ---------------------------------------------------------------
    // Pass 1: Fold constant branches — if(true) or if(false)
    // Remove the dead branch edge, making the block unconditional.
    // ---------------------------------------------------------------
    bool SimplifyConstantBranches(CGraph &g) {
        bool changed = false;
        for (auto &bl : g.nodes) {
            if (bl.collapsed) continue;
            if (!bl.is_conditional || bl.SizeOut() != 2) continue;
            if (!bl.branch_cond) continue;

            // Check if branch_cond is a compile-time constant.
            // Handles: IntegerLiteral(1), (1U == 1U), etc.
            clang::Expr::EvalResult eval_result;
            // Use tryEvaluateAsConstant patterns without ASTContext
            auto *cond = bl.branch_cond->IgnoreParenImpCasts();

            bool is_const = false;
            bool const_val = false;

            // Simple case: IntegerLiteral
            if (auto *il = llvm::dyn_cast<clang::IntegerLiteral>(cond)) {
                is_const = true;
                const_val = il->getValue().getBoolValue();
            }
            // Binary comparison of two identical literals
            else if (auto *bo = llvm::dyn_cast<clang::BinaryOperator>(cond)) {
                if (bo->isComparisonOp()) {
                    auto *lhs = bo->getLHS()->IgnoreParenImpCasts();
                    auto *rhs = bo->getRHS()->IgnoreParenImpCasts();
                    auto *li = llvm::dyn_cast<clang::IntegerLiteral>(lhs);
                    auto *ri = llvm::dyn_cast<clang::IntegerLiteral>(rhs);
                    if (li && ri) {
                        auto lv = li->getValue();
                        auto rv = ri->getValue();
                        // Extend to same width for comparison
                        unsigned w = std::max(lv.getBitWidth(), rv.getBitWidth());
                        lv = lv.zextOrTrunc(w);
                        rv = rv.zextOrTrunc(w);
                        switch (bo->getOpcode()) {
                        case clang::BO_EQ:  is_const = true; const_val = (lv == rv); break;
                        case clang::BO_NE:  is_const = true; const_val = (lv != rv); break;
                        case clang::BO_LT:  is_const = true; const_val = lv.slt(rv); break;
                        case clang::BO_GT:  is_const = true; const_val = lv.sgt(rv); break;
                        case clang::BO_LE:  is_const = true; const_val = lv.sle(rv); break;
                        case clang::BO_GE:  is_const = true; const_val = lv.sge(rv); break;
                        default: break;
                        }
                    }
                }
            }

            if (!is_const) continue;

            // Don't fold if either branch targets self (loop back-edges)
            size_t dead_idx = const_val ? 0 : 1;
            size_t dead_target = bl.succs[dead_idx];
            if (dead_target == bl.id) continue;
            size_t live_target = bl.succs[1 - dead_idx];
            if (live_target == bl.id) continue;

            // Don't fold if the dead block is shared (SizeIn > 1) — other
            // non-constant branches reach it too.  Only fold when the
            // constant branch is the sole path to the dead block.
            auto &dead_bl = g.Node(dead_target);
            if (dead_bl.SizeIn() > 1) continue;

            LOG(INFO) << "SimplifyConstantBranches: block " << bl.id
                      << " condition is constant " << (const_val ? "true" : "false")
                      << ", removing edge to " << dead_target << "\n";

            g.RemoveEdge(bl.id, dead_target);
            bl.is_conditional = false;
            bl.branch_cond = nullptr;
            changed = true;
        }
        return changed;
    }

    // ---------------------------------------------------------------
    // Pass 2: Remove unreachable blocks — no predecessors (except entry)
    // ---------------------------------------------------------------
    bool SimplifyRemoveUnreachable(CGraph &g) {
        bool changed = false;
        for (auto &bl : g.nodes) {
            if (bl.collapsed) continue;
            if (bl.id == g.entry) continue;
            if (bl.SizeIn() == 0) {
                // No predecessors — unreachable.  Remove all outgoing edges
                // so successors' SizeIn decreases.
                LOG(INFO) << "SimplifyRemoveUnreachable: block " << bl.id
                          << " is unreachable, collapsing\n";
                while (!bl.succs.empty()) {
                    g.RemoveEdge(bl.id, bl.succs.back());
                }
                bl.collapsed = true;
                changed = true;
            }
        }
        return changed;
    }

    // ---------------------------------------------------------------
    // Pass 3: Eliminate empty blocks — merge blocks with no stmts,
    // no branch_cond, no structured content, no label, and SizeOut≤1
    // into their single successor by redirecting predecessors.
    // ---------------------------------------------------------------
    bool SimplifyEliminateEmptyBlocks(CGraph &g) {
        bool changed = false;
        for (auto &bl : g.nodes) {
            if (bl.collapsed) continue;
            if (bl.id == g.entry) continue;
            if (!bl.stmts.empty()) continue;
            if (bl.branch_cond) continue;
            if (bl.structured) continue;
            if (bl.is_conditional) continue;
            if (!bl.label.empty()) continue;  // keep labeled blocks (goto targets)
            if (bl.SizeOut() != 1) continue;
            // Don't merge if the successor already has this block as its only pred
            // and FoldSequence would handle it
            size_t succ_id = bl.succs[0];
            if (bl.IsGotoOut(0) || bl.IsBackEdge(0)) continue;

            // Redirect all predecessors to point to the successor
            for (size_t p : bl.preds) {
                auto &pn = g.Node(p);
                for (size_t i = 0; i < pn.succs.size(); ++i) {
                    if (pn.succs[i] == bl.id) {
                        pn.succs[i] = succ_id;
                    }
                }
                // Add pred to successor's pred list
                g.Node(succ_id).preds.push_back(p);
            }
            // Remove this block from successor's pred list
            auto &sp = g.Node(succ_id).preds;
            sp.erase(std::remove(sp.begin(), sp.end(), bl.id), sp.end());

            // Update entry if needed
            if (g.entry == bl.id) g.entry = succ_id;

            bl.collapsed = true;
            changed = true;
            LOG(INFO) << "SimplifyEliminateEmptyBlocks: merged block " << bl.id
                      << " into " << succ_id << "\n";
        }
        return changed;
    }

    // ---------------------------------------------------------------
    // Pass 4: Remove dead code after unconditional gotos/returns.
    // If a block has stmts after a GotoStmt or ReturnStmt that were
    // not popped by resolveEdges, remove the trailing dead stmts.
    // ---------------------------------------------------------------
    bool SimplifyDeadStmtsAfterTransfer(CGraph &g) {
        bool changed = false;
        for (auto &bl : g.nodes) {
            if (bl.collapsed) continue;
            if (bl.stmts.size() < 2) continue;

            for (size_t i = 0; i + 1 < bl.stmts.size(); ++i) {
                bool is_transfer = false;
                if (llvm::isa<clang::ReturnStmt>(bl.stmts[i]))
                    is_transfer = true;
                else if (llvm::isa<clang::GotoStmt>(bl.stmts[i]))
                    is_transfer = true;

                if (is_transfer) {
                    size_t dead_count = bl.stmts.size() - i - 1;
                    LOG(INFO) << "SimplifyDeadStmtsAfterTransfer: block " << bl.id
                              << " removing " << dead_count << " dead stmts after transfer\n";
                    bl.stmts.resize(i + 1);
                    changed = true;
                    break;
                }
            }
        }
        return changed;
    }

    // ---------------------------------------------------------------
    // Master simplification pass — runs all sub-passes iteratively
    // until no more changes.  Mirrors Ghidra's pre-structuring
    // normalization (ActionUnreachable + ActionDeterminedBranch).
    // ---------------------------------------------------------------
    void SimplifyCGraph(CGraph &g) {
        bool progress = true;
        unsigned iterations = 0;
        while (progress && iterations < 10) {
            progress = false;
            progress |= SimplifyConstantBranches(g);
            progress |= SimplifyRemoveUnreachable(g);
            progress |= SimplifyEliminateEmptyBlocks(g);
            progress |= SimplifyDeadStmtsAfterTransfer(g);
            ++iterations;
        }
    }

    // ---------------------------------------------------------------
    // AND/OR condition collapsing (ResolveConditionChain)
    // ---------------------------------------------------------------

    // Detects chained if-gotos and collapses them into compound conditions.
    // OR pattern (i==0): bl->false leads to orblock, both reach same clauseblock => BO_LOr
    // AND pattern (i==1): bl->true leads to orblock, both reach same clauseblock => BO_LAnd
    bool ResolveConditionChain(CGraph &g, size_t id, SNodeFactory &/*factory*/,
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

            // Clear the stale terminal — its goto labels reference the
            // original edge targets (e.g. the absorbed orblock), not the
            // rewired targets.  Without this, ResolveEdgeLabel picks up
            // the terminal's LabelDecl and emits a goto to the wrong block.
            bl.terminal = nullptr;

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

    void ResolveAllConditionChains(CGraph &g, SNodeFactory &factory,
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
    void
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
    bool ResolveLoopBodyTracing(
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
    bool ResolveGotoSelection(CGraph &g, std::list<detail::LoopBody> &loopbody) {
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
    // Merge-point goto fallback — when TraceDAG PushBranches hits its
    // iteration limit and can't select gotos, use a simpler heuristic:
    // for each conditional block, mark the edge to the successor with
    // the highest SizeIn as goto.  This breaks shared-merge-point
    // deadlocks that prevent FoldIfElse/FoldIfThen from firing.
    // ---------------------------------------------------------------
    bool ResolveMergePointGotos(CGraph &g) {
        // Find the best candidate: any block with an edge to a
        // high-SizeIn successor. Breaking the highest-SizeIn edge
        // first has the biggest cascading effect.
        size_t best_src = CNode::kNone;
        size_t best_idx = 0;
        size_t best_sizein = 1;  // only mark if SizeIn > 1

        for (auto &bl : g.nodes) {
            if (bl.collapsed) continue;
            if (bl.SizeOut() < 2) continue;

            for (size_t i = 0; i < bl.succs.size(); ++i) {
                if (bl.IsGotoOut(i) || bl.IsBackEdge(i)) continue;
                size_t s = bl.succs[i];
                auto &sn = g.Node(s);
                if (sn.collapsed) continue;
                if (sn.SizeIn() > best_sizein) {
                    best_sizein = sn.SizeIn();
                    best_src = bl.id;
                    best_idx = i;
                }
            }
        }

        if (best_src == CNode::kNone) return false;

        g.Node(best_src).SetGoto(best_idx);
        LOG(INFO) << "ResolveMergePointGotos: marked edge "
                  << best_src << "[" << best_idx << "] → "
                  << g.Node(best_src).succs[best_idx]
                  << " (sizeIn=" << best_sizein << ") as goto\n";
        return true;
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
    bool ResolveSwitchGuards(CGraph &g) {
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
            size_t walk_limit = g.nodes.size();
            while (walk_limit-- > 0) {
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
    bool ResolveClauseSplit(CGraph &g) {
        // Use index-based loop because push_back below may
        // reallocate g.nodes, invalidating range-based iterators.
        size_t n = g.nodes.size();
        for (size_t ni = 0; ni < n; ++ni) {
            auto &bl = g.nodes[ni];
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
    bool ResolveJoinAbsorb(CGraph &g) {
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

    // ---------------------------------------------------------------
    // Multi-way exit goto marking (Fix 4) — for non-switch blocks
    // with SizeOut > 2, mark edges to the highest-SizeIn successor
    // as kGoto.  Repeats until all non-switch blocks have SizeOut ≤ 2.
    // This makes the graph 2-way reducible for FoldIfElse/FoldIfThen.
    // ---------------------------------------------------------------
    bool ResolveMultiWayExitGotos(CGraph &g) {
        bool changed = false;
        bool progress = true;
        while (progress) {
            progress = false;
            for (auto &bl : g.nodes) {
                if (bl.collapsed) continue;
                if (bl.SizeOut() <= 2) continue;
                // Skip switch blocks — they have their own fold rule
                if (!bl.switch_cases.empty()) continue;

                // Find the successor with the highest SizeIn — this is
                // the exit/merge point that blocks structuring.
                size_t best_succ = CNode::kNone;
                size_t best_sizein = 0;
                size_t best_idx = 0;
                for (size_t i = 0; i < bl.succs.size(); ++i) {
                    if (bl.IsGotoOut(i)) continue;  // already a goto
                    size_t s = bl.succs[i];
                    auto &sn = g.Node(s);
                    if (sn.collapsed) continue;
                    if (sn.SizeIn() > best_sizein) {
                        best_sizein = sn.SizeIn();
                        best_succ = s;
                        best_idx = i;
                    }
                }
                if (best_succ == CNode::kNone) continue;
                // Only mark if the successor is a shared merge point
                if (best_sizein <= 1) continue;

                bl.SetGoto(best_idx);
                LOG(INFO) << "ResolveMultiWayExitGotos: marked edge "
                          << bl.id << " → " << best_succ
                          << " (sizeIn=" << best_sizein << ") as goto\n";
                changed = true;
                progress = true;
            }
        }
        return changed;
    }

    // ---------------------------------------------------------------
    // Switch-skip goto conversion (Fix 2) — Ghidra's checkSwitchSkips.
    // For switch blocks where case edges go directly to the exit block,
    // mark those case-to-exit edges as kGoto.  This allows FoldSwitch
    // to proceed even when some cases share the exit merge point.
    // ---------------------------------------------------------------
    [[maybe_unused]] static bool PreprocessSwitchSkips(CGraph &g) {
        bool changed = false;
        for (auto &bl : g.nodes) {
            if (bl.collapsed) continue;
            if (bl.switch_cases.empty()) continue;
            if (!bl.IsSwitchOut()) continue;

            // Collect case target succ indices
            std::unordered_set<size_t> case_succ_indices;
            for (const auto &entry : bl.switch_cases) {
                case_succ_indices.insert(entry.succ_index);
            }

            // Find exit block: first non-case successor, OR the
            // successor with the highest SizeIn among all successors.
            size_t exit_id = CNode::kNone;
            // Strategy 1: explicit non-case successor
            for (size_t si = 0; si < bl.succs.size(); ++si) {
                if (case_succ_indices.count(si)) continue;
                size_t s = bl.succs[si];
                if (!g.Node(s).collapsed) {
                    exit_id = s;
                    break;
                }
            }
            // Strategy 2: if no explicit exit, use highest-SizeIn successor
            if (exit_id == CNode::kNone) {
                size_t best_sizein = 0;
                for (size_t s : bl.succs) {
                    auto &sn = g.Node(s);
                    if (sn.collapsed) continue;
                    if (sn.SizeIn() > best_sizein) {
                        best_sizein = sn.SizeIn();
                        exit_id = s;
                    }
                }
            }
            if (exit_id == CNode::kNone) continue;

            // Mark case edges that go directly to the exit as kGoto
            for (size_t si = 0; si < bl.succs.size(); ++si) {
                if (bl.succs[si] != exit_id) continue;
                if (bl.IsGotoOut(si)) continue;
                // Only mark case edges (non-default) that skip to exit
                if (!case_succ_indices.count(si)) continue;

                bl.SetGoto(si);
                LOG(INFO) << "PreprocessSwitchSkips: marked case edge "
                          << bl.id << "[" << si << "] → " << exit_id
                          << " as goto (switch skip)\n";
                changed = true;
            }
        }
        return changed;
    }

    /// Try control-equivalence hoisting transforms to unblock the main
    /// collapse rules before falling back to goto selection.
    bool ResolveControlEquivHoist(CGraph &g) {
        if (ResolveClauseSplit(g)) return true;
        if (ResolveJoinAbsorb(g)) return true;
        return false;
    }

    // ---------------------------------------------------------------
    // Main collapse loop
    // ---------------------------------------------------------------

    size_t FoldMainLoop(CGraph &g, SNodeFactory &factory,
                        clang::ASTContext &ctx, CGraphDotTracer &tracer) {
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

                    if (FoldGoto(g, n.id, factory, ctx)) { tracer.Dump(g, "FoldGoto"); change = true; continue; }
                    if (FoldSequence(g, n.id, factory)) { tracer.Dump(g, "FoldSequence"); change = true; continue; }
                    if (FoldIfThen(g, n.id, factory, ctx)) { tracer.Dump(g, "FoldIfThen"); change = true; continue; }
                    if (FoldIfElse(g, n.id, factory, ctx)) { tracer.Dump(g, "FoldIfElse"); change = true; continue; }
                    if (FoldWhileLoop(g, n.id, factory, ctx)) { tracer.Dump(g, "FoldWhileLoop"); change = true; continue; }
                    if (FoldDoWhileLoop(g, n.id, factory, ctx)) { tracer.Dump(g, "FoldDoWhileLoop"); change = true; continue; }
                    if (FoldInfiniteLoop(g, n.id, factory, ctx)) { tracer.Dump(g, "FoldInfiniteLoop"); change = true; continue; }
                    if (FoldSwitch(g, n.id, factory, ctx)) { tracer.Dump(g, "FoldSwitch"); change = true; continue; }
                }
            } while (change);

            // Try IfNoExit as fallback (Ghidra applies this only when stuck)
            change = false;
            for (auto &n : g.nodes) {
                if (n.collapsed) continue;
                if (FoldIfForcedGoto(g, n.id, factory, ctx)) {
                    tracer.Dump(g, "FoldIfForcedGoto");
                    change = true;
                    break;
                }
                if (FoldCaseFallthrough(g, n.id)) {
                    tracer.Dump(g, "FoldCaseFallthrough");
                    change = true;
                    break;
                }
                if (FoldIfThenGoto(g, n.id, factory, ctx)) {
                    tracer.Dump(g, "FoldIfThenGoto");
                    change = true;
                    break;
                }
                if (FoldIfElseChain(g, n.id, factory, ctx)) {
                    tracer.Dump(g, "FoldIfElseChain");
                    change = true;
                    break;
                }
            }
        } while (change);

        return isolated_count;
    }

} // namespace detail
} // namespace patchestry::ast
