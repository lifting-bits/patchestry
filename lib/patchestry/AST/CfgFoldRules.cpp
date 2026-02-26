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
    // Condition negation helper
    // ---------------------------------------------------------------

    clang::Expr *NegateCond(clang::Expr *cond, clang::ASTContext &ctx) {
        // Double negation elimination: !!x → x
        if (auto *uo = llvm::dyn_cast<clang::UnaryOperator>(cond)) {
            if (uo->getOpcode() == clang::UO_LNot) {
                return uo->getSubExpr();
            }
        }
        // Flip comparison operators: (a <= b) → (a > b), etc.
        // This avoids the precedence bug where !a <= b parses as (!a) <= b.
        if (auto *bo = llvm::dyn_cast<clang::BinaryOperator>(cond)) {
            if (!bo->isComparisonOp()) goto fallback_paren;
            auto flipped = clang::BinaryOperator::negateComparisonOp(bo->getOpcode());
            if (flipped != bo->getOpcode()) {
                return clang::BinaryOperator::Create(
                    ctx, bo->getLHS(), bo->getRHS(), flipped,
                    bo->getType(), bo->getValueKind(), bo->getObjectKind(),
                    bo->getOperatorLoc(), clang::FPOptionsOverride());
            }
        }
        // Fallback: wrap in ParenExpr to ensure correct precedence
        // when the ! is applied to a non-trivial sub-expression.
        fallback_paren:
        auto *parened = new (ctx) clang::ParenExpr(
            clang::SourceLocation(), clang::SourceLocation(), cond);
        return clang::UnaryOperator::Create(
            ctx, parened, clang::UO_LNot, ctx.IntTy,
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
        // Wrap with SLabel if this CNode carries a label.
        // The label is consumed (cleared) after wrapping so that
        // fold rules that separately wrap with SLabel won't
        // double-wrap.  The label persists in the SNode tree via
        // the SLabel node; RefineDeadLabels removes unused ones.
        if (!n.label.empty()) {
            bool already_labeled = false;
            if (auto *lbl = result->dyn_cast<SLabel>()) {
                already_labeled = (lbl->Name() == n.label);
            } else if (auto *seq = result->dyn_cast<SSeq>()) {
                if (seq->Size() > 0) {
                    if (auto *lbl = (*seq)[0]->dyn_cast<SLabel>()) {
                        already_labeled = (lbl->Name() == n.label);
                    }
                }
            }
            if (!already_labeled) {
                result = factory.Make<SLabel>(factory.Intern(n.label), result);
            }
            n.label.clear();
        }
        return result;
    }

    // ---------------------------------------------------------------
    // Ghidra-style exit-goto emitter — reconstructs control flow
    // that resolveEdges popped from content stmts.
    //
    // In Ghidra, gotos are never in operations — they're pure edge
    // metadata.  When a fold rule collapses blocks, edges are
    // remapped but never lost.  In Patchestry, resolveEdges pops
    // the goto stmts and stores them as CNode::terminal.  This
    // function appends gotos to an SNode body for edges leaving
    // the collapse set that the fold rule didn't handle.
    // ---------------------------------------------------------------

    /// Extract the goto target label name from a terminal stmt for a
    /// specific edge index.  Returns empty string if no match.
    ///
    /// For conditional terminals (IfStmt with goto arms):
    ///   edge 0 = false branch (else arm)
    ///   edge 1 = true branch (then arm)
    /// For unconditional terminals (GotoStmt):
    ///   edge 0 = the goto target
    static std::string TerminalLabelForEdge(const CNode &node, size_t edge_idx) {
        if (!node.terminal) return {};

        std::function<std::string(const clang::Stmt *)> get_label =
            [&](const clang::Stmt *s) -> std::string {
            if (!s) return {};
            if (auto *go = llvm::dyn_cast<clang::GotoStmt>(s)) {
                if (auto *ii = go->getLabel()->getIdentifier())
                    return ii->getName().str();
            }
            // Goto inside a CompoundStmt wrapper
            if (auto *cs = llvm::dyn_cast<clang::CompoundStmt>(s)) {
                if (cs->size() == 1)
                    return get_label(cs->body_front());
            }
            return {};
        };

        // Unconditional goto
        if (auto *go = llvm::dyn_cast<clang::GotoStmt>(node.terminal)) {
            if (edge_idx == 0) {
                if (auto *ii = go->getLabel()->getIdentifier())
                    return ii->getName().str();
            }
            return {};
        }

        // Conditional: if(cond) goto A; [else goto B;]
        if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(node.terminal)) {
            if (edge_idx == 1)  // true branch = then arm
                return get_label(ifs->getThen());
            if (edge_idx == 0)  // false branch = else arm
                return get_label(ifs->getElse());
        }

        return {};
    }

    /// Resolve the label for a goto target.
    /// Prefers the terminal's LabelDecl name (preserves Clang AST label
    /// identity) and falls back to CNode field resolution.
    std::string ResolveTargetLabel(const CGraph &g, size_t target_id) {
        size_t c = target_id;
        while (g.Node(c).collapsed
               && g.Node(c).collapsed_into != CNode::kNone)
            c = g.Node(c).collapsed_into;
        auto &tn = g.Node(c);
        if (!tn.label.empty()) return tn.label;
        if (!tn.original_label.empty()) return tn.original_label;
        if (c != target_id && !g.Node(target_id).original_label.empty())
            return g.Node(target_id).original_label;
        return "block_" + std::to_string(target_id);
    }

    /// Resolve the label for edge `edge_idx` of `source`, preferring
    /// the terminal's LabelDecl for exact identity.
    static std::string ResolveEdgeLabel(const CGraph &g, const CNode &source,
                                        size_t edge_idx) {
        // First: try the terminal stmt's LabelDecl (exact Clang identity)
        auto tl = TerminalLabelForEdge(source, edge_idx);
        if (!tl.empty()) return tl;
        // Fallback: resolve from target CNode fields
        if (edge_idx < source.succs.size())
            return ResolveTargetLabel(g, source.succs[edge_idx]);
        return "block_unknown";
    }


    /// Build the inline body for a target block, recursively following
    /// single-input blocks.  If the target has SizeIn==1 and is not
    /// collapsed, its content is inlined and its id is added to
    /// extra_ids for later collapse.  Otherwise emit a goto.
    ///
    /// source/edge_idx: the source node and edge index that reaches
    /// target_id.  Used for terminal-based label resolution.
    SNode *BuildInlineOrGoto(
            CGraph &g, size_t target_id,
            std::unordered_set<size_t> &id_set,
            size_t exit_id,
            std::vector<size_t> &extra_ids,
            SNodeFactory &factory, clang::ASTContext &ctx,
            size_t depth,
            const CNode *source, size_t edge_idx) {
        auto &tn = g.Node(target_id);
        // Don't inline if: collapsed, already in set, is exit, or too deep
        if (tn.collapsed || id_set.count(target_id) ||
            target_id == exit_id || depth > 32) {
            // Prefer terminal's LabelDecl name for exact label identity
            auto label = (source)
                ? ResolveEdgeLabel(g, *source, edge_idx)
                : ResolveTargetLabel(g, target_id);
            return factory.Make<SGoto>(factory.Intern(label));
        }
        // Only inline if exclusively reached from the collapse set
        if (tn.SizeIn() != 1) {
            auto label = (source)
                ? ResolveEdgeLabel(g, *source, edge_idx)
                : ResolveTargetLabel(g, target_id);
            return factory.Make<SGoto>(factory.Intern(label));
        }

        // Inline: build body from the target block
        id_set.insert(target_id);
        extra_ids.push_back(target_id);

        // Follow single-succ chain from this block
        SNode *body = LeafFromNode(tn, factory);
        size_t walk = target_id;
        while (g.Node(walk).SizeOut() == 1) {
            size_t next = g.Node(walk).succs[0];
            auto &nn = g.Node(next);
            if (nn.collapsed || id_set.count(next) ||
                nn.SizeIn() != 1 || next == exit_id)
                break;
            id_set.insert(next);
            extra_ids.push_back(next);
            auto *next_body = LeafFromNode(nn, factory);
            auto *seq = factory.Make<SSeq>();
            seq->AddChild(body);
            seq->AddChild(next_body);
            body = seq;
            walk = next;
        }

        // Recursively handle exits of the inlined tail
        body = EmitExitGotos(g, body, g.Node(walk),
                             id_set, exit_id, extra_ids,
                             factory, ctx, depth + 1);
        return body;
    }

    /// Append the terminal control-flow stmt for external edges leaving
    /// a clause tail.  Uses CNode::terminal (the original goto/if-goto
    /// preserved by resolveEdges) instead of reconstructing from edge
    /// metadata.
    ///
    /// When an exit target has SizeIn==1 (exclusively owned), its
    /// content is inlined recursively instead of emitting a goto.
    SNode *EmitExitGotos(
            CGraph &g, SNode *body, CNode &tail,
            std::unordered_set<size_t> &id_set,
            size_t exit_id,
            std::vector<size_t> &extra_ids,
            SNodeFactory &factory, clang::ASTContext &ctx,
            size_t depth) {
        // Collect external exits
        std::vector<std::pair<size_t, size_t>> ext; // (edge_idx, target_id)
        for (size_t i = 0; i < tail.succs.size(); ++i) {
            size_t s = tail.succs[i];
            if (id_set.count(s)) continue;
            if (s == exit_id) continue;
            ext.push_back({i, s});
        }
        if (ext.empty()) return body;

        auto *seq = body->dyn_cast<SSeq>();
        if (!seq) { seq = factory.Make<SSeq>(); seq->AddChild(body); }

        // Build exit gotos/inlines from edge metadata.
        // NOTE: we keep the terminal field for DOT visualization but use
        // edge metadata for SNode construction because:
        // (a) selective inlining (SizeIn==1 targets) requires per-exit control
        // (b) CIR emission can't handle raw Clang GotoStmt inside SBlock
        if (ext.size() == 1 && !tail.is_conditional) {
            auto *exit_body = BuildInlineOrGoto(
                g, ext[0].second, id_set, exit_id,
                extra_ids, factory, ctx, depth,
                &tail, ext[0].first);
            seq->AddChild(exit_body);
        } else if (tail.is_conditional && tail.succs.size() == 2) {
            clang::Expr *cond = tail.branch_cond;
            if (!cond)
                cond = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(32, 1), ctx.IntTy,
                    clang::SourceLocation());
            bool t_ext = false, f_ext = false;
            size_t t_tgt = CNode::kNone, f_tgt = CNode::kNone;
            size_t t_idx = 0, f_idx = 0;
            for (auto &[idx, tid] : ext) {
                if (idx == 1) { t_ext = true; t_tgt = tid; t_idx = idx; }
                if (idx == 0) { f_ext = true; f_tgt = tid; f_idx = idx; }
            }
            SNode *t_body = t_ext
                ? BuildInlineOrGoto(g, t_tgt, id_set, exit_id,
                                    extra_ids, factory, ctx, depth,
                                    &tail, t_idx)
                : nullptr;
            SNode *f_body = f_ext
                ? BuildInlineOrGoto(g, f_tgt, id_set, exit_id,
                                    extra_ids, factory, ctx, depth,
                                    &tail, f_idx)
                : nullptr;
            if (t_body && f_body) {
                seq->AddChild(factory.Make<SIfThenElse>(cond, t_body, f_body));
            } else if (t_body) {
                seq->AddChild(factory.Make<SIfThenElse>(cond, t_body, nullptr));
            } else if (f_body) {
                seq->AddChild(factory.Make<SIfThenElse>(
                    NegateCond(cond, ctx), f_body, nullptr));
            }
        } else {
            for (auto &[idx, tid] : ext) {
                auto *exit_body = BuildInlineOrGoto(
                    g, tid, id_set, exit_id,
                    extra_ids, factory, ctx, depth,
                    &tail, idx);
                seq->AddChild(exit_body);
            }
        }
        return seq;
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
        // Don't chain into a conditional node — FoldIfThen/FoldIfElse need
        // it as a separate head to match the if-pattern.  Absorbing it into
        // a sequence buries the branch_cond in an SSeq where it is invisible
        // to downstream if-rules (see decode_frame uVar7!=0 loss).
        if (next.is_conditional && next.SizeOut() > 1) {
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
            // Don't chain into a conditional tail (same rationale as above).
            if (nxtNode.is_conditional && nxtNode.SizeOut() > 1) {
                break;
            }
            chain.push_back(nxt);
            cur = nxt;
        }

        for (size_t i = 1; i < chain.size(); ++i) {
            seq->AddChild(LeafFromNode(g.Node(chain[i]), factory));
        }

        g.IdentifyInternal(chain, CNode::BlockType::kSequence, seq);
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
                if (head_content->Kind() == SNodeKind::kBlock) {
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

            g.IdentifyInternal({ id, clause_id }, CNode::BlockType::kIf, if_node);
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
            if (head_content->Kind() == SNodeKind::kBlock) {
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

        g.IdentifyInternal({ id, tc_id, fc_id }, CNode::BlockType::kIf, if_node);
        return true;
    }

    // Rule: If-else-if chain
    //
    //   H1(cond1) → body1 →→ exit
    //             → H2(cond2) → body2 →→ exit
    //                        → H3(cond3) → body3 →→ exit
    //                                   → default →→ exit
    //
    // All true-branch bodies (and the final false branch) must reach
    // a common exit block, possibly through single-successor chains.
    // Collapses into nested if-else-if SNodes.
    bool FoldIfElseChain(CGraph &g, size_t id, SNodeFactory &factory,
                         clang::ASTContext &ctx) {
        auto &bl = g.Node(id);
        if (bl.collapsed || bl.SizeOut() != 2) return false;
        if (bl.IsSwitchOut()) return false;
        if (!bl.IsDecisionOut(0) || !bl.IsDecisionOut(1)) return false;

        // Follow the chain of false branches.
        // Collect: [(head_id, true_clause_id), ...] and final_else_id.
        struct ChainEntry { size_t head; size_t clause; };
        std::vector<ChainEntry> chain;
        std::vector<size_t> all_ids;

        size_t cur = id;
        size_t exit_id = CNode::kNone;

        while (true) {
            auto &cn = g.Node(cur);
            if (cn.collapsed || cn.SizeOut() != 2) break;
            if (cn.IsSwitchOut()) break;
            if (!cn.IsDecisionOut(0) || !cn.IsDecisionOut(1)) break;

            size_t tc_id = cn.succs[1];  // true clause
            size_t fc_id = cn.succs[0];  // false clause (next head or final else)
            auto &tc = g.Node(tc_id);

            if (tc.collapsed || tc.SizeIn() != 1) break;

            // Find where the true clause exits (follow single-succ chain).
            size_t tc_exit = CNode::kNone;
            std::vector<size_t> tc_chain;
            {
                size_t walk = tc_id;
                tc_chain.push_back(walk);
                while (true) {
                    auto &w = g.Node(walk);
                    if (w.SizeOut() == 0) { tc_exit = CNode::kNone; break; }
                    if (w.SizeOut() != 1) break;
                    size_t next = w.succs[0];
                    if (g.Node(next).SizeIn() != 1 || next == cur) {
                        tc_exit = next;
                        break;
                    }
                    tc_chain.push_back(next);
                    walk = next;
                }
            }

            // Validate exit consistency
            if (exit_id == CNode::kNone) {
                exit_id = tc_exit;
            } else if (tc_exit != exit_id && tc_exit != CNode::kNone) {
                break; // Different exits
            }

            chain.push_back({cur, tc_id});
            all_ids.push_back(cur);
            for (size_t c : tc_chain) all_ids.push_back(c);

            // Next in chain: the false branch must be single-in
            auto &fc = g.Node(fc_id);
            if (fc.collapsed) break;
            if (fc.SizeIn() != 1) break;

            // Is the false clause the next conditional (continuing chain)?
            if (fc.SizeOut() == 2 && !fc.IsSwitchOut()
                && fc.IsDecisionOut(0) && fc.IsDecisionOut(1)) {
                cur = fc_id;
                continue;
            }

            // Final else clause — check if it exits to exit_id
            size_t fc_exit = CNode::kNone;
            std::vector<size_t> fc_chain;
            {
                size_t walk = fc_id;
                fc_chain.push_back(walk);
                while (true) {
                    auto &w = g.Node(walk);
                    if (w.SizeOut() == 0) { fc_exit = CNode::kNone; break; }
                    if (w.SizeOut() != 1) break;
                    size_t next = w.succs[0];
                    if (g.Node(next).SizeIn() != 1 || next == cur) {
                        fc_exit = next;
                        break;
                    }
                    fc_chain.push_back(next);
                    walk = next;
                }
            }

            if (fc_exit == exit_id || exit_id == CNode::kNone) {
                // Include the final else clause
                for (size_t c : fc_chain) all_ids.push_back(c);
            }
            break;
        }

        // Need at least 2 levels to be worth chaining
        if (chain.size() < 2) return false;

        // Build id_set for EmitExitGotos membership checks.
        // extra_ids collects blocks inlined by EmitExitGotos
        // (SizeIn==1 successors absorbed into the clause body).
        std::unordered_set<size_t> id_set(all_ids.begin(), all_ids.end());
        std::vector<size_t> extra_ids;

        // Build nested if-else from bottom up
        SNode *result = nullptr;

        // Check if we have a final else (last false branch)
        size_t last_head = chain.back().head;
        size_t last_fc = g.Node(last_head).succs[0];
        if (std::find(all_ids.begin(), all_ids.end(), last_fc) != all_ids.end()) {
            // Build else body from last_fc chain
            result = LeafFromNode(g.Node(last_fc), factory);
            size_t walk = last_fc;
            while (g.Node(walk).SizeOut() == 1) {
                size_t next = g.Node(walk).succs[0];
                if (std::find(all_ids.begin(), all_ids.end(), next) == all_ids.end()) break;
                auto *next_body = LeafFromNode(g.Node(next), factory);
                auto *seq = factory.Make<SSeq>();
                seq->AddChild(result);
                seq->AddChild(next_body);
                result = seq;
                walk = next;
            }
            result = EmitExitGotos(g, result, g.Node(walk),
                                   id_set, exit_id, extra_ids,
                                   factory, ctx);

            // Same exit_id goto fix as for then_body above.
            if (exit_id != CNode::kNone) {
                auto &etail = g.Node(walk);
                bool exits_to_exit = false;
                for (size_t s : etail.succs) {
                    if (s == exit_id && !id_set.count(s))
                        exits_to_exit = true;
                }
                if (exits_to_exit) {
                    std::string label = ResolveTargetLabel(g, exit_id);
                    auto *goto_node = factory.Make<SGoto>(
                        factory.Intern(label));
                    auto *seq = result->dyn_cast<SSeq>();
                    if (!seq) {
                        seq = factory.Make<SSeq>();
                        seq->AddChild(result);
                    }
                    seq->AddChild(goto_node);
                    result = seq;
                }
            }
        }

        // Build from bottom of chain upward
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            auto &head = g.Node(it->head);
            clang::Expr *cond = head.branch_cond;
            if (!cond) {
                cond = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
            }

            // Build then body (follow single-succ chain from clause)
            SNode *then_body = LeafFromNode(g.Node(it->clause), factory);
            size_t walk = it->clause;
            while (g.Node(walk).SizeOut() == 1) {
                size_t next = g.Node(walk).succs[0];
                if (std::find(all_ids.begin(), all_ids.end(), next) == all_ids.end()) break;
                auto *next_body = LeafFromNode(g.Node(next), factory);
                auto *seq = factory.Make<SSeq>();
                seq->AddChild(then_body);
                seq->AddChild(next_body);
                then_body = seq;
                walk = next;
            }
            // Emit gotos / inline single-input successors
            then_body = EmitExitGotos(g, then_body, g.Node(walk),
                                      id_set, exit_id, extra_ids,
                                      factory, ctx);

            // EmitExitGotos skips exit_id (assumes natural fallthrough),
            // but when the collapsed block has multiple external successors,
            // natural fallthrough to exit_id isn't guaranteed.  Append an
            // explicit goto for clauses whose only exit is exit_id.
            if (exit_id != CNode::kNone) {
                auto &tail = g.Node(walk);
                bool exits_to_exit = false;
                for (size_t s : tail.succs) {
                    if (s == exit_id && !id_set.count(s))
                        exits_to_exit = true;
                }
                if (exits_to_exit) {
                    std::string label = ResolveTargetLabel(g, exit_id);
                    auto *goto_node = factory.Make<SGoto>(
                        factory.Intern(label));
                    auto *seq = then_body->dyn_cast<SSeq>();
                    if (!seq) {
                        seq = factory.Make<SSeq>();
                        seq->AddChild(then_body);
                    }
                    seq->AddChild(goto_node);
                    then_body = seq;
                }
            }

            // Else body: accumulated result, or inline/goto for
            // external false branch
            SNode *else_body = result;
            if (!else_body) {
                size_t fc_id = head.succs[0];
                if (!id_set.count(fc_id) && fc_id != exit_id) {
                    else_body = BuildInlineOrGoto(
                        g, fc_id, id_set, exit_id,
                        extra_ids, factory, ctx, 0,
                        &head, 0);
                } else {
                    else_body = factory.Make<SBlock>();
                }
            }
            result = factory.Make<SIfThenElse>(cond, then_body, else_body);

            // Prepend head content
            SNode *head_content = LeafFromNode(head, factory);
            bool has_content = false;
            if (head_content) {
                if (head_content->Kind() == SNodeKind::kBlock)
                    has_content = !head_content->as<SBlock>()->Stmts().empty();
                else
                    has_content = true;
            }
            if (has_content) {
                auto *seq = factory.Make<SSeq>();
                seq->AddChild(head_content);
                seq->AddChild(result);
                result = seq;
            }
        }

        if (!bl.label.empty()) {
            result = factory.Make<SLabel>(factory.Intern(bl.label), result);
            bl.label.clear();
        }

        // Include blocks inlined by EmitExitGotos
        for (size_t eid : extra_ids) all_ids.push_back(eid);
        g.IdentifyInternal(all_ids, CNode::BlockType::kIf, result);
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
            if (clause.SizeIn() != 1) continue;

            // Simple case: single-block body that loops back directly.
            bool simple_body = (clause.SizeOut() == 1 && clause.succs[0] == id);

            // Multi-block case: walk a chain from clause until we find
            // a node that loops back to the header.  Handles:
            // (a) single-in/single-out chains
            // (b) conditional nodes where one branch loops back and the
            //     other exits (internal if-break pattern in loop body)
            std::vector<size_t> chain;
            bool multi_body = false;
            if (!simple_body) {
                size_t cur = clause_id;
                std::unordered_set<size_t> visited;
                while (true) {
                    if (visited.count(cur)) break;
                    visited.insert(cur);
                    auto &cn = g.Node(cur);
                    if (cn.collapsed) break;
                    if (cur != clause_id && cn.SizeIn() != 1) break;
                    chain.push_back(cur);
                    // Direct loop-back
                    if (cn.SizeOut() == 1 && cn.succs[0] == id) {
                        multi_body = true;
                        break;
                    }
                    // Conditional: one branch loops back via a back-edge,
                    // the other exits.  Only match genuine back-edges
                    // (kBack flag set) to avoid mismatching nested loops.
                    if (cn.SizeOut() == 2) {
                        int back_idx = -1;
                        for (size_t si2 = 0; si2 < 2; ++si2) {
                            if (cn.succs[si2] == id && cn.IsBackEdge(si2)) {
                                back_idx = static_cast<int>(si2);
                                break;
                            }
                            auto &sb = g.Node(cn.succs[si2]);
                            if (!sb.collapsed && sb.SizeIn() == 1 && sb.SizeOut() == 1
                                && sb.succs[0] == id && sb.IsBackEdge(0)) {
                                // Successor loops back to header via one hop
                                back_idx = static_cast<int>(si2);
                                chain.push_back(cn.succs[si2]);
                                break;
                            }
                        }
                        if (back_idx >= 0) {
                            multi_body = true;
                            break;
                        }
                    }
                    if (cn.SizeOut() != 1) break;
                    cur = cn.succs[0];
                }
            }

            if (!simple_body && !multi_body) continue;

            // Validate predecessor connectivity: every chain node's
            // single predecessor must be the header or another chain
            // node, otherwise we'd collapse a node owned by a
            // different subgraph.
            if (multi_body) {
                std::unordered_set<size_t> chain_set(chain.begin(), chain.end());
                chain_set.insert(id); // header
                for (size_t cid : chain) {
                    auto &cn = g.Node(cid);
                    if (cn.SizeIn() != 1 || !chain_set.count(cn.preds[0])) {
                        multi_body = false;
                        break;
                    }
                }
                if (!multi_body) continue;
            }

            clang::Expr *cond = bl.branch_cond;
            if (!cond) {
                cond = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
            }

            // Negate condition when body is on false branch
            if (i == 0) {
                cond = NegateCond(cond, ctx);
            }

            // Build the body SNode — simple or multi-block chain.
            SNode *clause_body = nullptr;
            if (simple_body) {
                clause_body = LeafFromNode(clause, factory);
            } else {
                auto *seq = factory.Make<SSeq>();
                for (size_t cid : chain) {
                    auto *node_body = LeafFromNode(g.Node(cid), factory);
                    if (node_body) seq->AddChild(node_body);
                }
                clause_body = seq;
            }

            // Determine which while-loop pattern to emit based on
            // the header's content.
            SNode *while_node;

            // Case A: all header stmts are expressions — use comma
            // operator in the while condition:
            //   while (stmt1, stmt2, cond) { body }
            bool all_expr = !bl.stmts.empty() && bl.branch_cond;
            if (all_expr) {
                for (auto *s : bl.stmts) {
                    if (!llvm::isa<clang::Expr>(s)
                        || llvm::isa<clang::DeclStmt>(s)) {
                        all_expr = false;
                        break;
                    }
                }
            }

            bool has_content = !bl.stmts.empty() || bl.structured;

            if (all_expr) {
                // Build comma chain: (stmt1, (stmt2, cond))
                auto op_loc = clang::SourceLocation();
                clang::Expr *while_cond = cond;
                for (auto it = bl.stmts.rbegin(); it != bl.stmts.rend(); ++it) {
                    auto *expr = llvm::dyn_cast<clang::Expr>(*it);
                    while_cond = clang::BinaryOperator::Create(
                        ctx, expr, while_cond, clang::BO_Comma,
                        while_cond->getType(), clang::VK_PRValue,
                        clang::OK_Ordinary, op_loc,
                        clang::FPOptionsOverride());
                }
                while_node = factory.Make<SWhile>(while_cond, clause_body);
                if (!bl.label.empty()) {
                    while_node = factory.Make<SLabel>(
                        factory.Intern(bl.label), while_node);
                    bl.label.clear();
                }
            } else if (has_content) {
                // Case B: header has DeclStmts, structured content, or
                // mix — emit while(true) { header; if(exit) break; body }
                auto *true_lit = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());

                clang::Expr *exit_cond = bl.branch_cond;
                if (!exit_cond) {
                    exit_cond = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                }
                if (i == 1) {
                    exit_cond = NegateCond(exit_cond, ctx);
                }

                auto *break_node = factory.Make<SBreak>();
                auto *if_break = factory.Make<SIfThenElse>(exit_cond, break_node, nullptr);

                // Use LeafFromNode to capture both raw stmts AND
                // structured content (e.g., inner while-loop from
                // a prior fold).
                SNode *header_block = LeafFromNode(bl, factory);

                auto *seq = factory.Make<SSeq>();
                seq->AddChild(header_block);
                seq->AddChild(if_break);
                seq->AddChild(clause_body);
                while_node = factory.Make<SWhile>(true_lit, seq);
            } else {
                // Case C: no header content — simple while(cond) { body }
                while_node = factory.Make<SWhile>(cond, clause_body);
                if (!bl.label.empty()) {
                    while_node = factory.Make<SLabel>(factory.Intern(bl.label), while_node);
                    bl.label.clear();
                }
            }

            // Set scope labels for break/continue resolution.
            // Exit = the non-body successor; header = this block.
            {
                size_t exit_succ = bl.succs[1 - i];
                auto &exit_node = g.Node(exit_succ);
                std::string_view exit_lbl = !exit_node.original_label.empty()
                    ? factory.Intern(exit_node.original_label)
                    : std::string_view{};
                std::string_view header_lbl = !bl.original_label.empty()
                    ? factory.Intern(bl.original_label)
                    : std::string_view{};
                // Find the SWhile inside possible SLabel wrapper
                SNode *wn = while_node;
                if (auto *lbl = wn->dyn_cast<SLabel>()) wn = lbl->Body();
                if (auto *w = wn->dyn_cast<SWhile>()) {
                    w->SetExitLabel(exit_lbl);
                    w->SetHeaderLabel(header_lbl);
                }
            }

            // Collapse: simple body is just {header, clause}, multi-block
            // includes the entire chain.
            std::vector<size_t> collapse_ids;
            collapse_ids.push_back(id);
            if (simple_body) {
                collapse_ids.push_back(clause_id);
            } else {
                for (size_t cid : chain) {
                    collapse_ids.push_back(cid);
                }
            }
            g.IdentifyInternal(collapse_ids, CNode::BlockType::kWhile, while_node);
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
            auto *dw = factory.Make<SDoWhile>(body, cond);
            // Set scope labels: exit = non-self successor, header = self
            {
                size_t exit_succ = bl.succs[1 - i];
                auto &exit_node = g.Node(exit_succ);
                if (!exit_node.original_label.empty())
                    dw->SetExitLabel(factory.Intern(exit_node.original_label));
                if (!bl.original_label.empty())
                    dw->SetHeaderLabel(factory.Intern(bl.original_label));
            }
            SNode *dowhile_node = dw;
            if (!saved_label.empty()) {
                dowhile_node = factory.Make<SLabel>(factory.Intern(saved_label), dowhile_node);
            }

            g.IdentifyInternal({ id }, CNode::BlockType::kDoWhile, dowhile_node);
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
        g.IdentifyInternal({ id }, CNode::BlockType::kInfLoop, loop);
        return true;
    }

    // Rule: If with no exit (clause has zero out edges, or clause's
    // single out-edge is a back-edge — i.e. loop continuation).
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
            if (clause.SizeIn() != 1) continue;
            // Accept: terminal clause (SizeOut==0) OR
            // clause whose single out-edge is a back-edge (loop continue).
            bool is_terminal = (clause.SizeOut() == 0);
            bool is_back_only = (clause.SizeOut() == 1 && clause.IsBackEdge(0));
            if (!is_terminal && !is_back_only) continue;
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

            // If the clause continues the loop via a back-edge, remove
            // the edge BEFORE CollapseNodes so it isn't collected as an
            // ext_succ (which would create a spurious self-loop on the
            // collapsed node).  Order matters: collapse first would
            // inherit the back-edge; remove first makes the clause
            // terminal so only the header's other branch survives.
            if (is_back_only) {
                g.RemoveEdge(clause_id, clause.succs[0]);
            }

            SNode *if_node = factory.Make<SIfThenElse>(cond, clause_body, nullptr);

            // Prepend the head block's accumulated content (from prior
            // collapses) so it is not lost.
            SNode *head_content = LeafFromNode(bl, factory);
            bool has_head = false;
            if (head_content) {
                if (head_content->Kind() == SNodeKind::kBlock) {
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

            g.IdentifyInternal({ id, clause_id }, CNode::BlockType::kIf, if_node);
            return true;
        }
        return false;
    }

    // ---------------------------------------------------------------
    // FoldIfThenGoto — handles a conditional where one branch is a
    // single-entry clause and the other is a shared goto target
    // (SizeIn > 1).  Folds the clause into an if-then body and
    // emits a goto for the shared branch, reducing its SizeIn to
    // unblock downstream folds (FoldIfElse, FoldIfThen, etc.).
    //
    //  BEFORE:   bl ──→ clause ──→ X        AFTER:  rep ──→ X
    //            │                                  (structured: if(cond){clause}
    //            └──→ shared (SizeIn>1)              else{goto shared_label;})
    //                                        shared.SizeIn decremented
    // ---------------------------------------------------------------
    bool FoldIfThenGoto(CGraph &g, size_t id, SNodeFactory &factory,
                        clang::ASTContext &ctx) {
        auto &bl = g.Node(id);
        if (bl.collapsed || bl.SizeOut() != 2) return false;
        if (bl.IsSwitchOut()) return false;
        if (bl.IsGotoOut(0) || bl.IsGotoOut(1)) return false;

        for (size_t i = 0; i < 2; ++i) {
            size_t clause_id = bl.succs[i];
            if (clause_id == id) continue;
            auto &clause = g.Node(clause_id);
            if (clause.collapsed) continue;
            if (clause.SizeIn() != 1) continue;
            if (clause.SizeOut() != 1) continue;
            if (!bl.IsDecisionOut(i)) continue;
            if (clause.IsGotoOut(0)) continue;

            // The other branch must be a shared goto target.
            size_t shared_id = bl.succs[1 - i];
            auto &shared = g.Node(shared_id);
            if (shared.SizeIn() <= 1) continue;
            if (shared.label.empty()) continue;

            // Build condition (negate if clause is on false branch)
            clang::Expr *cond = bl.branch_cond;
            if (!cond) {
                cond = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
            }
            if (i == 0) {
                cond = NegateCond(cond, ctx);
            }

            auto *clause_body = LeafFromNode(clause, factory);
            auto *goto_node = factory.Make<SGoto>(factory.Intern(shared.label));
            SNode *if_node = factory.Make<SIfThenElse>(cond, clause_body, goto_node);

            // Prepend head block's accumulated content so it is not lost.
            SNode *head_content = LeafFromNode(bl, factory);
            bool has_head = false;
            if (head_content) {
                if (head_content->Kind() == SNodeKind::kBlock) {
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

            // Remove the edge to the shared target BEFORE collapse so it
            // is not collected as an ext_succ.  This reduces shared.SizeIn.
            g.RemoveEdge(id, shared_id);

            g.IdentifyInternal({ id, clause_id }, CNode::BlockType::kIf, if_node);
            return true;
        }
        return false;
    }

    // Aggressive inlining helper: follow the single-successor chain
    // from a case entry block, building an SSeq of all block contents.
    // Stops at exit_id, returns (sizeOut==0), back-edges, or blocks
    // with multiple outgoing edges (inner control flow).
    // Adds all traversed blocks to |all_collapse|.
    SNode *InlineCaseChain(CGraph &g, size_t start_id, size_t exit_id,
                           SNodeFactory &factory, clang::ASTContext &ctx,
                           std::unordered_set<size_t> &chain_visited,
                           std::unordered_set<size_t> &all_collapse) {
        if (start_id == exit_id) return nullptr;
        auto &bl = g.Node(start_id);
        if (bl.collapsed) return nullptr;
        if (chain_visited.count(start_id)) return nullptr; // cycle guard

        // Don't absorb nodes that are themselves switches — leave them
        // for a separate FoldSwitch pass so inner switches get properly
        // restructured with their own InlineCaseChain.
        if (bl.IsSwitchOut() && !bl.switch_cases.empty()) {
            return nullptr;
        }

        chain_visited.insert(start_id);
        all_collapse.insert(start_id);

        // Use structured content if already folded by a prior rule,
        // otherwise build leaf from raw stmts.  Defer LeafFromNode
        // until we know we need it (avoids orphaned SLabel wrappers).
        auto get_body = [&]() -> SNode * {
            if (bl.structured) return bl.structured;
            return LeafFromNode(bl, factory);
        };

        // No successors (return/tail) — just the body
        if (bl.SizeOut() == 0) return get_body();

        // Single successor to exit block — body only (emitter adds break)
        if (bl.SizeOut() == 1 && bl.succs[0] == exit_id) return get_body();

        // Single successor — recursively inline the next block
        if (bl.SizeOut() == 1) {
            size_t next = bl.succs[0];
            auto *next_body = InlineCaseChain(
                g, next, exit_id, factory, ctx, chain_visited, all_collapse);
            if (next_body) {
                auto *this_body = get_body();
                auto *seq = factory.Make<SSeq>();
                seq->AddChild(this_body);
                seq->AddChild(next_body);
                return seq;
            }
            return get_body();
        }

        // Multiple successors — use structured content if available,
        // otherwise recursively build an if-else for 2-successor
        // conditional nodes (common from splitAtInternalControlFlow).
        if (bl.structured) {
            return bl.structured;
        }
        if (bl.SizeOut() == 2 && bl.is_conditional) {
            size_t t_id = bl.succs[0];
            size_t f_id = bl.succs[1];
            auto *t_body = InlineCaseChain(
                g, t_id, exit_id, factory, ctx, chain_visited, all_collapse);
            auto *f_body = InlineCaseChain(
                g, f_id, exit_id, factory, ctx, chain_visited, all_collapse);
            if (t_body || f_body) {
                if (!t_body) t_body = factory.Make<SBlock>();
                if (!f_body) f_body = factory.Make<SBlock>();
                clang::Expr *cond = bl.branch_cond;
                if (!cond) {
                    cond = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy,
                        clang::SourceLocation());
                }
                auto *if_node = factory.Make<SIfThenElse>(cond, t_body, f_body);
                auto *this_body = get_body();
                bool empty_body = llvm::isa<SBlock>(this_body)
                    && static_cast<SBlock*>(this_body)->Stmts().empty();
                if (empty_body) {
                    return if_node;
                }
                auto *seq = factory.Make<SSeq>();
                seq->AddChild(this_body);
                seq->AddChild(if_node);
                return seq;
            }
        }
        return get_body();
    }

    // Rule: Switch statement (with aggressive case-body inlining)
    bool FoldSwitch(CGraph &g, size_t id, SNodeFactory &factory,
                         clang::ASTContext &ctx) {
        auto &bl = g.Node(id);
        if (bl.collapsed || !bl.IsSwitchOut()) {
            return false;
        }
        // Must have switch_cases metadata to fire.
        if (bl.switch_cases.empty()) return false;

        // Collect the set of case-target successor ids for quick lookup.
        std::unordered_set<size_t> case_succ_ids;
        for (const auto &entry : bl.switch_cases) {
            if (entry.succ_index < bl.succs.size()) {
                case_succ_ids.insert(bl.succs[entry.succ_index]);
            }
        }

        // Find exit block: prefer a non-case successor (explicit exit),
        // then fall back to the highest-SizeIn case successor (switch-skip
        // pattern where some cases exit directly to the merge point).
        size_t exit_id = std::numeric_limits<size_t>::max();
        for (size_t s : bl.succs) {
            if (case_succ_ids.count(s)) continue; // skip case targets
            auto &sn = g.Node(s);
            if (sn.collapsed) continue;
            // Non-case successor = exit/fallback block
            exit_id = s;
            break;
        }
        // Fallback: if no explicit exit, look for a case target that is
        // a shared merge point (SizeIn > 1). This handles switch-skip
        // patterns where some cases go directly to the exit.
        if (exit_id == std::numeric_limits<size_t>::max()) {
            size_t best_sizein = 1;
            for (size_t s : bl.succs) {
                auto &sn = g.Node(s);
                if (sn.collapsed) continue;
                if (sn.SizeIn() > best_sizein) {
                    best_sizein = sn.SizeIn();
                    exit_id = s;
                }
            }
        }

        // If no explicit exit found, check for loop-interior switch:
        // all case targets either ARE a back-edge target or have a single
        // successor that is a back-edge to a common loop header.
        // In that case, use the loop header as the logical exit (each case
        // body "breaks" from the switch and the loop continues).
        [[maybe_unused]] bool loop_interior_switch = false;
        if (exit_id == std::numeric_limits<size_t>::max()) {
            size_t back_target = std::numeric_limits<size_t>::max();
            bool valid = true;
            for (size_t si = 0; si < bl.succs.size(); ++si) {
                size_t s = bl.succs[si];
                // Direct back-edge from switch to a target (e.g. case 111)
                if (bl.IsBackEdge(si)) {
                    if (back_target == std::numeric_limits<size_t>::max()) back_target = s;
                    else if (back_target != s) { valid = false; break; }
                    continue;
                }
                auto &sn = g.Node(s);
                if (sn.collapsed) { valid = false; break; }
                // Case target with single successor = back-edge to header
                if (sn.SizeOut() == 1 && sn.IsBackEdge(0)) {
                    size_t bt = sn.succs[0];
                    if (back_target == std::numeric_limits<size_t>::max()) back_target = bt;
                    else if (back_target != bt) { valid = false; break; }
                    continue;
                }
                // Case target terminates (return/exit) — compatible
                if (sn.SizeOut() == 0) continue;
                // Otherwise incompatible
                valid = false;
                break;
            }
            if (valid && back_target != std::numeric_limits<size_t>::max()) {
                exit_id = back_target;
                loop_interior_switch = true;
            }
        }

        // Validate: each case entry must have sizeIn==1 and not be collapsed.
        // Exceptions:
        //   - exit_id (loop header or merge point) is always skipped
        //   - case targets that ARE the exit block get empty bodies (switch-skip)
        for (size_t s : bl.succs) {
            if (s == exit_id) continue;
            auto &sn = g.Node(s);
            if (sn.collapsed) return false;
            if (sn.SizeIn() != 1) {
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

        // Track all blocks to collapse (aggressive inlining may pull in
        // blocks beyond the direct case successors).
        std::unordered_set<size_t> all_collapse;
        all_collapse.insert(id);

        // Build case arms.  When a successor has multiple case values
        // (e.g. case 1: case 2: case 3: body), only the LAST value
        // carries the body; preceding values get nullptr (fallthrough).
        unsigned iw = ctx.getIntWidth(ctx.IntTy);
        for (size_t si = 0; si < bl.succs.size(); ++si) {
            size_t s = bl.succs[si];
            auto it = succ_to_values.find(si);
            if (it == succ_to_values.end()) continue;

            const auto &vals = it->second;
            SNode *body = nullptr;

            if (s == exit_id) {
                // Case targets the exit/merge block directly — give it
                // an empty body (emitter adds break; from the switch).
                // This handles both loop-interior switches and
                // switch-skip patterns (Ghidra's checkSwitchSkips).
                body = factory.Make<SBlock>();
            } else {
                // Aggressive inlining: follow the successor chain from this
                // case entry, inlining all reachable single-successor blocks.
                std::unordered_set<size_t> chain_visited;
                body = InlineCaseChain(
                    g, s, exit_id, factory, ctx, chain_visited, all_collapse);
                if (!body) body = LeafFromNode(g.Node(s), factory);
            }

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

        // Build collapse set from all_collapse, but only include blocks
        // whose predecessors are ALL within the collapse set (safe to remove).
        // The switch block itself is always included.
        std::vector<size_t> collapse_ids;
        collapse_ids.push_back(id);
        for (size_t nid : all_collapse) {
            if (nid == id) continue;
            bool all_preds_internal = true;
            for (size_t p : g.Node(nid).preds) {
                if (all_collapse.count(p) == 0) {
                    all_preds_internal = false;
                    break;
                }
            }
            if (all_preds_internal) {
                collapse_ids.push_back(nid);
            }
        }

        if (!bl.label.empty()) {
            sw_node = factory.Make<SLabel>(factory.Intern(bl.label), sw_node);
            bl.label.clear();
        }
        size_t rep                 = g.IdentifyInternal(collapse_ids, CNode::BlockType::kSwitch, sw_node);
        // The collapsed node has no real branch condition — prevent
        // FoldGoto from emitting a spurious if(1) goto.
        g.Node(rep).is_conditional = false;
        g.Node(rep).switch_cases.clear();
        return true;
    }

    // Rule: Wrap goto edges as SGoto nodes.
    // Also handles non-conditional 2-successor blocks where one edge
    // is a back-edge: the forward exit is treated as an implicit goto
    // (Ghidra handles this via ruleBlockGoto in the main loop rather
    // than a separate pre-pass).
    bool FoldGoto(CGraph &g, size_t id, SNodeFactory &factory,
                       clang::ASTContext &ctx) {
        auto &bl = g.Node(id);
        if (bl.collapsed) return false;

        // Ghidra-style: for non-conditional 2-successor blocks where
        // one edge is a back-edge, mark the forward exit as kGoto so
        // it gets processed below.  This replaces the BackEdgePrePass.
        if (!bl.is_conditional && bl.SizeOut() == 2) {
            int back_idx = -1;
            for (size_t i = 0; i < 2; ++i) {
                if (bl.IsBackEdge(i)) back_idx = static_cast<int>(i);
            }
            if (back_idx >= 0) {
                size_t exit_idx = 1 - static_cast<size_t>(back_idx);
                if (!bl.IsGotoOut(exit_idx)) {
                    bl.SetGoto(exit_idx);
                }
            }
        }

        for (size_t i = 0; i < bl.succs.size(); ++i) {
            if (!bl.IsGotoOut(i)) {
                continue;
            }

            // Resolve goto target label.  Try current label first,
            // then original_label (survives CollapseNodes clearing),
            // then follow collapsed_into chain to the representative.
            size_t target_id = bl.succs[i];
            std::string target_label;
            {
                size_t cur = target_id;
                // Follow collapsed_into chain to representative
                while (g.Node(cur).collapsed
                       && g.Node(cur).collapsed_into != CNode::kNone) {
                    cur = g.Node(cur).collapsed_into;
                }
                auto &tn = g.Node(cur);
                if (!tn.label.empty())
                    target_label = tn.label;
                else if (!tn.original_label.empty())
                    target_label = tn.original_label;
                else if (cur != target_id
                         && !g.Node(target_id).original_label.empty())
                    target_label = g.Node(target_id).original_label;
                else
                    target_label = "block_" + std::to_string(target_id);
            }
            auto *goto_node = factory.Make<SGoto>(factory.Intern(target_label));

            // Conditional block with one goto edge: emit if(cond) goto
            // and keep the non-goto successor edge.
            if (bl.is_conditional && bl.succs.size() == 2 && bl.branch_cond) {
                clang::Expr *cond = bl.branch_cond;
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


} // namespace detail
} // namespace patchestry::ast
