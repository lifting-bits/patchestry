/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/SNodePostPasses.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Util/Log.hpp>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/PrettyPrinter.h>
#include <clang/AST/Stmt.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace patchestry::ast {

    // Count goto refs (SGoto + clang::GotoStmt inside SStmt) per label.
    static void CountGotoRefs(SNode *node,
                              std::unordered_map<std::string_view, int> &refs) {
        if (!node) return;

        // Leaf cases for_each_child can't reach.
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

        node->for_each_child([&](SNode *c) { CountGotoRefs(c, refs); });
    }

    static void CountGotoRefs(const std::vector< SNode * > &seq,
                              std::unordered_map<std::string_view, int> &refs) {
        for (auto *c : seq) CountGotoRefs(c, refs);
    }

    // Replace seq[i] with `repl` and advance `i` so the surrounding
    // loop's ++i steps to the first node after the splice. Empty `repl`
    // would underflow `i` (size_t) — asserted.
    static void SpliceAndAdvance(std::vector< SNode * > &seq,
                                 size_t &i,
                                 const std::vector< SNode * > &repl) {
        assert(!repl.empty()
               && "SpliceAndAdvance requires non-empty replacement");
        seq.erase(seq.begin() + static_cast<ptrdiff_t>(i));
        seq.insert(seq.begin() + static_cast<ptrdiff_t>(i),
                   repl.begin(), repl.end());
        i += repl.size() - 1;
    }

    // Spill clang::Stmt* into SStmt SNodes; drops nulls.
    static void AppendStmts(SNodeFactory &factory,
                            std::vector< SNode * > &out,
                            const std::vector< clang::Stmt * > &stmts) {
        for (auto *s : stmts)
            if (s) out.push_back(factory.Make< SStmt >(s));
    }

    // Invoke fn(vector<SNode*>&) on each body-vector slot.
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

    // Post-order body-vector walk. `stop_after_change` unwinds on first
    // mutation so ancestor loops never resume over a vector the worker
    // may have mutated (required for cross-scope workers).
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

    namespace {

        bool InlineGotosInSeq(
            std::vector< SNode * > &seq, SNodeFactory & /*factory*/,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            bool changed = false;

            std::unordered_map<std::string_view, size_t> label_pos;
            for (size_t i = 0; i < seq.size(); ++i) {
                if (auto *lbl = seq[i]->dyn_cast<SLabel>()) {
                    label_pos[lbl->Name()] = i;
                }
            }

            // Only inline when the label is the immediate next sibling
            // (forward, nothing skipped).
            for (size_t i = 0; i < seq.size(); ++i) {
                auto *g = seq[i]->dyn_cast<SGoto>();
                if (!g) continue;

                // Sole-child gotos would orphan a parent guarding cond
                // (e.g. `if (c) goto L` collapsed to `if (c) ;`).
                if (seq.size() == 1) continue;

                auto target = g->Target();
                auto lp = label_pos.find(target);
                if (lp == label_pos.end()) continue;

                // Single-ref label only.
                auto rc = refs.find(target);
                if (rc == refs.end() || rc->second != 1) continue;

                size_t label_idx = lp->second;
                if (label_idx <= i) continue;
                if (label_idx != i + 1) continue;

                auto *lbl = seq[label_idx]->as<SLabel>();
                std::vector< SNode * > repl = lbl->BodyList();

                seq.erase(seq.begin() + static_cast<ptrdiff_t>(i),
                          seq.begin() + static_cast<ptrdiff_t>(label_idx) + 1);
                seq.insert(seq.begin() + static_cast<ptrdiff_t>(i),
                           repl.begin(), repl.end());

                changed = true;
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

    // EliminateGotoToNextLabel — eliminate trailing gotos pointing at
    // the immediately-following sibling label, including IfStmt-arm gotos.

    namespace {

        /// Innermost SNode trailing this subtree; `stmt` is set when it's
        /// an SStmt.
        struct TrailingInfo {
            SNode *container = nullptr;
            clang::Stmt *stmt = nullptr;
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
            return {node, nullptr};
        }

        std::string_view SNodeGotoTarget(SNode *node) {
            if (auto *g = node->dyn_cast<SGoto>()) return g->Target();
            return {};
        }

        std::string ClangGotoTarget(clang::Stmt *s) {
            if (auto *gs = llvm::dyn_cast_or_null<clang::GotoStmt>(s))
                return gs->getLabel()->getName().str();
            return {};
        }

        clang::Stmt *BuildClangSeqOrNull(
            clang::ASTContext &ctx,
            const std::vector<clang::Stmt *> &stmts
        ) {
            if (stmts.empty())
                return new (ctx) clang::NullStmt(VirtualLoc(ctx));
            if (stmts.size() == 1)
                return stmts.front();
            auto loc = VirtualLoc(ctx);
            return clang::CompoundStmt::Create(
                ctx, stmts, clang::FPOptionsOverride(), loc, loc);
        }

        void AppendClangStmtList(
            clang::Stmt *stmt,
            std::vector<clang::Stmt *> &out
        ) {
            if (!stmt)
                return;
            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
                for (clang::Stmt *child : compound->body())
                    if (child)
                        out.push_back(child);
                return;
            }
            if (!llvm::isa<clang::NullStmt>(stmt))
                out.push_back(stmt);
        }

        bool ClangTailCanBeScopedBeforeNextLabel(const clang::Stmt *stmt) {
            if (!stmt)
                return true;
            if (llvm::isa<clang::LabelStmt>(stmt)
                || llvm::isa<clang::CaseStmt>(stmt)
                || llvm::isa<clang::DefaultStmt>(stmt)
                || llvm::isa<clang::DeclStmt>(stmt))
                return false;
            for (const clang::Stmt *child : stmt->children())
                if (!ClangTailCanBeScopedBeforeNextLabel(child))
                    return false;
            return true;
        }

        bool ClangTailCanBeScopedBeforeNextLabel(
            llvm::ArrayRef<clang::Stmt *> stmts
        ) {
            for (clang::Stmt *stmt : stmts)
                if (!ClangTailCanBeScopedBeforeNextLabel(stmt))
                    return false;
            return true;
        }

        bool ClangIfGotoArmToNextLabel(
            clang::IfStmt *ifs,
            std::string_view target,
            bool &goto_in_then
        ) {
            goto_in_then = false;
            if (!ifs || !ifs->getCond())
                return false;

            std::string then_target = ClangGotoTarget(ifs->getThen());
            if (!then_target.empty() && then_target == target) {
                goto_in_then = true;
                return true;
            }

            std::string else_target = ClangGotoTarget(ifs->getElse());
            if (!else_target.empty() && else_target == target) {
                goto_in_then = false;
                return true;
            }

            return false;
        }

        bool ClangLabelNameEquals(llvm::StringRef name, std::string_view target) {
            return name.size() == target.size()
                && name == llvm::StringRef(target.data(), target.size());
        }

        bool LocalRefCountEquals(
            const std::unordered_map<std::string_view, int> &refs,
            std::string_view name,
            int expected
        ) {
            int count = 0;
            if (auto it = refs.find(name); it != refs.end())
                count = it->second;
            return count == expected;
        }

        clang::IfStmt *BuildClangNextLabelScopeIf(
            clang::ASTContext &ctx,
            clang::IfStmt *ifs,
            bool goto_in_then,
            llvm::ArrayRef<clang::Stmt *> tail
        ) {
            std::vector<clang::Stmt *> scoped;
            if (goto_in_then) {
                AppendClangStmtList(ifs->getElse(), scoped);
            } else {
                AppendClangStmtList(ifs->getThen(), scoped);
            }
            scoped.insert(scoped.end(), tail.begin(), tail.end());

            clang::Expr *cond = ifs->getCond();
            if (goto_in_then)
                cond = NegateExpr(ctx, cond);

            auto loc = ifs->getIfLoc();
            return clang::IfStmt::Create(
                ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                cond, loc, loc, BuildClangSeqOrNull(ctx, scoped), loc, nullptr);
        }

        clang::Stmt *StripClangGotoToNextLabelFromTail(
            clang::ASTContext &ctx,
            clang::Stmt *stmt,
            std::string_view target,
            unsigned &stripped
        ) {
            if (!stmt)
                return stmt;

            if (auto *go = llvm::dyn_cast<clang::GotoStmt>(stmt)) {
                if (go->getLabel()
                    && ClangLabelNameEquals(
                        go->getLabel()->getName(), target)) {
                    ++stripped;
                    return new (ctx) clang::NullStmt(VirtualLoc(ctx));
                }
                return stmt;
            }

            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
                if (compound->body_empty())
                    return stmt;
                std::vector<clang::Stmt *> body(
                    compound->body_begin(), compound->body_end());
                unsigned before = stripped;
                body.back() = StripClangGotoToNextLabelFromTail(
                    ctx, body.back(), target, stripped);
                return stripped != before ? BuildClangSeqOrNull(ctx, body)
                                          : stmt;
            }

            if (auto *label = llvm::dyn_cast<clang::LabelStmt>(stmt)) {
                clang::Stmt *sub = StripClangGotoToNextLabelFromTail(
                    ctx, label->getSubStmt(), target, stripped);
                if (sub != label->getSubStmt())
                    label->setSubStmt(sub);
                return stmt;
            }

            if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(stmt)) {
                unsigned before = stripped;
                ifs->setThen(StripClangGotoToNextLabelFromTail(
                    ctx, ifs->getThen(), target, stripped));
                if (ifs->getElse())
                    ifs->setElse(StripClangGotoToNextLabelFromTail(
                        ctx, ifs->getElse(), target, stripped));
                return stripped != before ? ifs : stmt;
            }

            return stmt;
        }

        clang::Stmt *ScopeifyClangGotoToNextLabel(
            clang::ASTContext &ctx,
            clang::Stmt *stmt,
            std::string_view target,
            unsigned &scopeified
        ) {
            if (!stmt)
                return stmt;

            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
                if (compound->body_empty())
                    return stmt;
                std::vector<clang::Stmt *> body(
                    compound->body_begin(), compound->body_end());

                for (size_t i = body.size(); i-- > 0;) {
                    auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(body[i]);
                    bool goto_in_then = false;
                    if (!ClangIfGotoArmToNextLabel(
                            ifs, target, goto_in_then))
                        continue;

                    llvm::ArrayRef<clang::Stmt *> tail(body.data() + i + 1,
                                                       body.size() - i - 1);
                    if (!ClangTailCanBeScopedBeforeNextLabel(tail))
                        continue;

                    std::vector<clang::Stmt *> replacement;
                    replacement.reserve(i + 1);
                    replacement.insert(
                        replacement.end(), body.begin(),
                        body.begin() + static_cast<ptrdiff_t>(i));
                    replacement.push_back(
                        BuildClangNextLabelScopeIf(
                            ctx, ifs, goto_in_then, tail));
                    ++scopeified;
                    return BuildClangSeqOrNull(ctx, replacement);
                }

                unsigned before = scopeified;
                body.back() = ScopeifyClangGotoToNextLabel(
                    ctx, body.back(), target, scopeified);
                return scopeified != before ? BuildClangSeqOrNull(ctx, body)
                                            : stmt;
            }

            if (auto *label = llvm::dyn_cast<clang::LabelStmt>(stmt)) {
                clang::Stmt *sub = ScopeifyClangGotoToNextLabel(
                    ctx, label->getSubStmt(), target, scopeified);
                if (sub != label->getSubStmt())
                    label->setSubStmt(sub);
                return stmt;
            }

            if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(stmt)) {
                unsigned before = scopeified;
                ifs->setThen(ScopeifyClangGotoToNextLabel(
                    ctx, ifs->getThen(), target, scopeified));
                if (ifs->getElse())
                    ifs->setElse(ScopeifyClangGotoToNextLabel(
                        ctx, ifs->getElse(), target, scopeified));
                return scopeified != before ? ifs : stmt;
            }

            return stmt;
        }

        std::string_view DirectSNodeGotoToNextLabelTarget(SNode *node) {
            if (!node)
                return {};
            if (auto *go = node->dyn_cast<SGoto>())
                return go->Target();
            if (auto *stmt = node->dyn_cast<SStmt>()) {
                if (auto *go =
                        llvm::dyn_cast_or_null<clang::GotoStmt>(
                            stmt->Stmt()))
                    if (go->getLabel())
                        return go->getLabel()->getName();
            }
            return {};
        }

        std::string_view SingleSNodeGotoListTarget(
            const std::vector<SNode *> &seq
        ) {
            if (seq.size() != 1)
                return {};
            return DirectSNodeGotoToNextLabelTarget(seq.front());
        }

        bool SNodeTailCanBeScopedBeforeNextLabel(SNode *node) {
            if (!node)
                return true;
            if (node->dyn_cast<SLabel>() || node->dyn_cast<SWhile>()
                || node->dyn_cast<SDoWhile>() || node->dyn_cast<SFor>()
                || node->dyn_cast<SSwitch>())
                return false;
            if (auto *stmt = node->dyn_cast<SStmt>())
                return ClangTailCanBeScopedBeforeNextLabel(stmt->Stmt());
            bool safe = true;
            node->for_each_child([&](SNode *child) {
                if (safe && !SNodeTailCanBeScopedBeforeNextLabel(child))
                    safe = false;
            });
            return safe;
        }

        bool SNodeTailCanBeScopedBeforeNextLabel(
            llvm::ArrayRef<SNode *> seq
        ) {
            for (SNode *node : seq)
                if (!SNodeTailCanBeScopedBeforeNextLabel(node))
                    return false;
            return true;
        }

        bool ScopeifySNodeGotoToNextLabelInSeq(
            std::vector<SNode *> &seq,
            std::string_view target,
            SNodeFactory &factory,
            clang::ASTContext &ctx
        ) {
            for (size_t i = seq.size(); i-- > 0;) {
                auto *ifs = seq[i]->dyn_cast<SIfThenElse>();
                if (!ifs || !ifs->Cond())
                    continue;

                bool goto_in_then = false;
                std::string_view then_target =
                    SingleSNodeGotoListTarget(ifs->ThenList());
                if (!then_target.empty() && then_target == target) {
                    goto_in_then = true;
                } else {
                    std::string_view else_target =
                        SingleSNodeGotoListTarget(ifs->ElseList());
                    if (else_target.empty() || else_target != target)
                        continue;
                    goto_in_then = false;
                }

                llvm::ArrayRef<SNode *> tail(seq.data() + i + 1,
                                             seq.size() - i - 1);
                if (!SNodeTailCanBeScopedBeforeNextLabel(tail))
                    continue;

                std::vector<SNode *> scoped;
                if (goto_in_then) {
                    scoped.insert(
                        scoped.end(), ifs->ElseList().begin(),
                        ifs->ElseList().end());
                } else {
                    scoped.insert(
                        scoped.end(), ifs->ThenList().begin(),
                        ifs->ThenList().end());
                }
                scoped.insert(scoped.end(), tail.begin(), tail.end());

                clang::Expr *cond = ifs->Cond();
                if (goto_in_then)
                    cond = NegateExpr(ctx, cond);

                auto *replacement = factory.Make<SIfThenElse>(
                    cond, std::move(scoped), std::vector<SNode *>{});

                std::vector<SNode *> prefix(
                    seq.begin(), seq.begin() + static_cast<ptrdiff_t>(i));
                prefix.push_back(replacement);
                seq = std::move(prefix);
                return true;
            }
            return false;
        }

        bool RepairSNodeSeqTailToNextLabel(
            std::vector<SNode *> &seq,
            std::string_view target,
            SNodeFactory &factory,
            clang::ASTContext &ctx
        );

        bool RepairSNodeNodeTailToNextLabel(
            SNode *node,
            std::string_view target,
            SNodeFactory &factory,
            clang::ASTContext &ctx
        ) {
            if (!node)
                return false;

            if (auto *stmt = node->dyn_cast<SStmt>()) {
                unsigned stripped = 0;
                clang::Stmt *stripped_stmt =
                    StripClangGotoToNextLabelFromTail(
                        ctx, stmt->Stmt(), target, stripped);
                if (stripped != 0) {
                    stmt->SetStmt(stripped_stmt);
                    return true;
                }

                unsigned scopeified = 0;
                clang::Stmt *scopeified_stmt =
                    ScopeifyClangGotoToNextLabel(
                        ctx, stmt->Stmt(), target, scopeified);
                if (scopeified != 0) {
                    stmt->SetStmt(scopeified_stmt);
                    return true;
                }
                return false;
            }

            if (auto *ifs = node->dyn_cast<SIfThenElse>()) {
                if (RepairSNodeSeqTailToNextLabel(
                        ifs->ThenList(), target, factory, ctx))
                    return true;
                if (RepairSNodeSeqTailToNextLabel(
                        ifs->ElseList(), target, factory, ctx))
                    return true;
                return false;
            }

            if (auto *label = node->dyn_cast<SLabel>())
                return RepairSNodeSeqTailToNextLabel(
                    label->BodyList(), target, factory, ctx);

            return false;
        }

        bool RepairSNodeSeqTailToNextLabel(
            std::vector<SNode *> &seq,
            std::string_view target,
            SNodeFactory &factory,
            clang::ASTContext &ctx
        ) {
            if (seq.empty())
                return false;

            if (ScopeifySNodeGotoToNextLabelInSeq(seq, target, factory, ctx))
                return true;

            SNode *last = seq.back();
            std::string_view direct = DirectSNodeGotoToNextLabelTarget(last);
            if (!direct.empty() && direct == target) {
                seq.pop_back();
                return true;
            }

            return RepairSNodeNodeTailToNextLabel(
                last, target, factory, ctx);
        }

        // Forward declaration — defined in InlineCrossScopeSingleRef namespace.
        bool SNodeAlwaysTerminates(SNode *node);

        bool EliminateInSeq(std::vector< SNode * > &children,
                            SNodeFactory &factory,
                            clang::ASTContext &ctx,
                            const std::unordered_map<std::string_view, int> &refs) {
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

                        bool next_label_is_single_ref =
                            LocalRefCountEquals(refs, next_name, 1);

                        if (st && next_label_is_single_ref) {
                            unsigned stripped = 0;
                            clang::Stmt *stripped_stmt =
                                StripClangGotoToNextLabelFromTail(
                                    ctx, info.stmt, next_name, stripped);
                            if (stripped != 0) {
                                st->SetStmt(stripped_stmt);
                                local_changed = true; changed = true;
                                break;
                            }

                            unsigned scopeified = 0;
                            clang::Stmt *scopeified_stmt =
                                ScopeifyClangGotoToNextLabel(
                                    ctx, info.stmt, next_name, scopeified);
                            if (scopeified != 0) {
                                st->SetStmt(scopeified_stmt);
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

                    if (LocalRefCountEquals(refs, next_name, 1)
                        && RepairSNodeNodeTailToNextLabel(
                            children[i], next_name, factory, ctx)) {
                        local_changed = true; changed = true;
                        break;
                    }
                }
            }
            return changed;
        }

    } // anonymous namespace (EliminateGotoToNextLabel helpers)

    bool EliminateGotoToNextLabel(std::vector< SNode * > &root,
                                  SNodeFactory &factory,
                                  clang::ASTContext &ctx) {
        std::unordered_map<std::string_view, int> refs;
        CountGotoRefs(root, refs);
        return ForEachSeqPostOrder(
            root, [&](std::vector< SNode * > &seq) {
                return EliminateInSeq(seq, factory, ctx, refs);
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

    // InlineCrossScopeSingleRef — cross-scope single-ref label inliner
    //
    // Extends InlineResidualGotos to the case where the goto and its
    // target label live in *different* sibling sequences.  When a label has
    // exactly one goto reference, its body always terminates, and no
    // fallthrough can reach the label, the body is moved (not cloned)
    // into the goto's slot and the label node is deleted.

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

                // Guarding-conditional preservation: same reasoning as
                // InlineGotosInSeq — when this goto is the sole element
                // in its sequence, that sequence is the body of an
                // enclosing SIfThenElse/SLabel and removing the goto
                // would orphan the parent's guard.  Keep the goto so the
                // condition's guarding role stays visible in the
                // emitted C.
                if (seq.size() == 1) continue;

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

    // ScopeifyIfGotos — `if(c) goto L; stmts; L:` → `if(!c){stmts}`.
    // Region-aware variant tolerates intermediate labels iff every ref
    // to them lives inside the region (closed region is movable).

    namespace {

        void CollectScopeifyLabels(
            SNode *node,
            std::unordered_set<std::string_view> &labels
        ) {
            if (!node) return;
            if (auto *lbl = node->dyn_cast<SLabel>())
                labels.insert(lbl->Name());
            // Deliberately skips clang::LabelStmt inside SStmt
            // (narrower than SubtreeHasLabel).
            node->for_each_child(
                [&](SNode *c) { CollectScopeifyLabels(c, labels); });
        }

        /// True when every label appearing inside `region` has all of its
        /// goto references inside `region` as well — i.e. the region is
        /// closed with respect to its own labels and may be moved or
        /// rewrapped without breaking goto/label pairing.
        bool ScopeifyLabelsAreRegionLocal(
            const std::vector<SNode *> &region,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            std::unordered_set<std::string_view> labels;
            std::unordered_map<std::string_view, int> region_refs;

            for (SNode *node : region) {
                CollectScopeifyLabels(node, labels);
                CountGotoRefs(node, region_refs);
            }

            for (auto label : labels) {
                int global_count = 0;
                if (auto it = refs.find(label); it != refs.end())
                    global_count = it->second;

                int region_count = 0;
                if (auto it = region_refs.find(label);
                    it != region_refs.end())
                    region_count = it->second;

                if (global_count != region_count)
                    return false;
            }
            return true;
        }

        std::string ScopeifyGotoTarget(SNode *node) {
            if (!node) return {};
            if (auto *g = node->dyn_cast<SGoto>())
                return std::string(g->Target());
            if (auto *st = node->dyn_cast<SStmt>()) {
                if (auto *gs = llvm::dyn_cast_or_null<clang::GotoStmt>(
                        st->Stmt()))
                    return gs->getLabel()->getName().str();
            }
            return {};
        }

        std::string ScopeifyTrailingGotoTarget(
            const std::vector<SNode *> &body
        ) {
            if (body.empty())
                return {};
            return ScopeifyGotoTarget(body.back());
        }

        std::string ScopeifySingleGotoListTarget(
            const std::vector<SNode *> &body
        ) {
            if (body.size() != 1)
                return {};
            return ScopeifyGotoTarget(body.front());
        }

        bool RefCountIsOne(
            const std::unordered_map<std::string_view, int> &refs,
            std::string_view label
        ) {
            auto it = refs.find(label);
            return it != refs.end() && it->second == 1;
        }

        size_t FindDirectLabelIndex(
            const std::vector<SNode *> &children,
            size_t begin,
            std::string_view target
        ) {
            for (size_t i = begin; i < children.size(); ++i) {
                if (auto *lbl = children[i]->dyn_cast<SLabel>())
                    if (lbl->Name() == target)
                        return i;
            }
            return children.size();
        }

        bool SeqHasLabel(const std::vector<SNode *> &seq) {
            for (SNode *node : seq)
                if (SubtreeHasLabel(node))
                    return true;
            return false;
        }

        // Structure a nested two-way if-goto / goto dispatch:
        //
        //   if (outer) { ...; if (c) goto T; else goto E; }
        //   goto J;
        // T: then_body;
        // E: else_body;
        // J: join_body;
        //
        // into:
        //
        //   if (outer) { ...; if (c) { then_body } else_body }
        //   join_body;
        //
        // Only fires when T, E and J are all single-ref, region-local
        // labels arranged contiguously right after the dispatch.
        bool ScopeifyTwoWayLocalDispatchInSeq(
            std::vector<SNode *> &children,
            SNodeFactory &factory,
            clang::ASTContext &ctx,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            for (size_t i = 0; i + 1 < children.size(); ++i) {
                auto *outer = children[i]->dyn_cast<SIfThenElse>();
                if (!outer || !outer->Cond() || outer->ThenList().empty())
                    continue;

                std::string join_target = ScopeifyGotoTarget(children[i + 1]);
                if (join_target.empty())
                    continue;
                std::string_view join_view(join_target);
                if (!RefCountIsOne(refs, join_view))
                    continue;

                auto *dispatch =
                    outer->ThenList().back()->dyn_cast<SIfThenElse>();
                if (!dispatch || !dispatch->Cond())
                    continue;

                std::string then_target =
                    ScopeifySingleGotoListTarget(dispatch->ThenList());
                std::string else_target =
                    ScopeifySingleGotoListTarget(dispatch->ElseList());
                if (then_target.empty() || else_target.empty()
                    || then_target == else_target)
                    continue;

                std::string_view then_view(then_target);
                std::string_view else_view(else_target);
                if (!RefCountIsOne(refs, then_view)
                    || !RefCountIsOne(refs, else_view))
                    continue;

                size_t then_idx =
                    FindDirectLabelIndex(children, i + 2, then_view);
                size_t else_idx =
                    FindDirectLabelIndex(children, i + 2, else_view);
                size_t join_idx =
                    FindDirectLabelIndex(children, i + 2, join_view);
                if (then_idx >= children.size() || else_idx >= children.size()
                    || join_idx >= children.size())
                    continue;

                size_t first_idx = std::min(then_idx, else_idx);
                size_t second_idx = std::max(then_idx, else_idx);
                if (first_idx != i + 2 || second_idx != first_idx + 1
                    || join_idx != second_idx + 1)
                    continue;

                auto *first_label = children[first_idx]->as<SLabel>();
                auto *second_label = children[second_idx]->as<SLabel>();
                if (SeqHasLabel(first_label->BodyList())
                    || SeqHasLabel(second_label->BodyList()))
                    continue;

                std::vector<SNode *> then_body = outer->ThenList();
                then_body.pop_back();

                std::vector<SNode *> first_body =
                    std::move(first_label->BodyList());
                std::vector<SNode *> second_body =
                    std::move(second_label->BodyList());
                std::vector<SNode *> join_body =
                    std::move(children[join_idx]->as<SLabel>()->BodyList());

                clang::Expr *first_cond = nullptr;
                if (first_label->Name() == then_view)
                    first_cond = dispatch->Cond();
                else
                    first_cond =
                        NegateExpr(ctx, CloneExpr(ctx, dispatch->Cond()));

                then_body.push_back(factory.Make<SIfThenElse>(
                    first_cond, std::move(first_body),
                    std::vector<SNode *>{}));
                then_body.insert(then_body.end(), second_body.begin(),
                                 second_body.end());
                then_body = factory.MakeSeq(std::move(then_body));
                outer->SetThenBranch(std::move(then_body));

                children.erase(children.begin() + static_cast<ptrdiff_t>(i + 1),
                               children.begin()
                                   + static_cast<ptrdiff_t>(join_idx) + 1);
                children.insert(children.begin() + static_cast<ptrdiff_t>(i + 1),
                                join_body.begin(), join_body.end());
                return true;
            }
            return false;
        }

        bool ScopeifyInSeq(std::vector< SNode * > &children,
                           SNodeFactory &factory,
                           clang::ASTContext &ctx,
                           const std::unordered_map<std::string_view, int> &refs) {
            bool any_change = false;
            bool changed = true;

            while (changed) {
                changed = false;
                if (ScopeifyTwoWayLocalDispatchInSeq(
                        children, factory, ctx, refs)) {
                    changed = true;
                    any_change = true;
                    continue;
                }

                for (size_t i = 0; i < children.size(); ++i) {
                    auto *ite = children[i]->dyn_cast<SIfThenElse>();
                    if (!ite || !ite->Cond()) continue;

                    // `if(c){pfx; goto L;} skipped; L:body;` → with
                    // single-ref L → `if(c){pfx;} else{skipped;} body;`.
                    if (ite->ElseList().empty()
                        && ite->ThenList().size() > 1) {
                        std::string trailing_target =
                            ScopeifyTrailingGotoTarget(ite->ThenList());
                        std::string_view trailing_view(trailing_target);

                        size_t label_idx = children.size();
                        if (!trailing_view.empty()) {
                            for (size_t j = i + 1; j < children.size(); ++j) {
                                if (auto *lbl = children[j]->dyn_cast<SLabel>()) {
                                    if (lbl->Name() == trailing_view) {
                                        label_idx = j;
                                        break;
                                    }
                                }
                            }
                        }

                        if (label_idx < children.size()) {
                            auto rc = refs.find(trailing_view);
                            if (rc != refs.end() && rc->second == 1) {
                                std::vector<SNode *> region_nodes;
                                for (size_t j = i + 1; j < label_idx; ++j)
                                    region_nodes.push_back(children[j]);

                                if (ScopeifyLabelsAreRegionLocal(
                                        region_nodes, refs)) {
                                    std::vector<SNode *> then_body =
                                        ite->ThenList();
                                    then_body.pop_back();
                                    then_body =
                                        factory.MakeSeq(std::move(then_body));

                                    std::vector<SNode *> else_body;
                                    else_body.reserve(label_idx - i - 1);
                                    for (size_t j = i + 1; j < label_idx; ++j)
                                        else_body.push_back(children[j]);
                                    else_body =
                                        factory.MakeSeq(std::move(else_body));

                                    auto *new_if = factory.Make<SIfThenElse>(
                                        ite->Cond(), std::move(then_body),
                                        std::move(else_body));

                                    auto *lbl = children[label_idx]->as<SLabel>();
                                    std::vector<SNode *> replacements;
                                    replacements.push_back(new_if);
                                    for (SNode *c : lbl->BodyList())
                                        replacements.push_back(c);

                                    children.erase(
                                        children.begin()
                                            + static_cast<ptrdiff_t>(i),
                                        children.begin()
                                            + static_cast<ptrdiff_t>(label_idx)
                                            + 1);
                                    children.insert(
                                        children.begin()
                                            + static_cast<ptrdiff_t>(i),
                                        replacements.begin(),
                                        replacements.end());

                                    changed = true;
                                    any_change = true;
                                    break;
                                }
                            }
                        }
                    }

                    std::string target = ScopeifyGotoTarget(
                        ite->ThenBranch());
                    if (target.empty()) continue;
                    std::string_view target_view(target);

                    // Find target SLabel in same body vector (forward only)
                    size_t label_idx = children.size();
                    for (size_t j = i + 1; j < children.size(); ++j) {
                        if (auto *lbl = children[j]->dyn_cast<SLabel>()) {
                            if (lbl->Name() == target_view) {
                                label_idx = j;
                                break;
                            }
                        }
                    }
                    if (label_idx >= children.size()) continue;
                    const bool has_else = !ite->ElseList().empty();
                    // Adjacent if-then without else is handled by
                    // EliminateGotoToNextLabel.
                    if (!has_else && label_idx == i + 1) continue;

                    // Single reference only
                    auto rc = refs.find(target_view);
                    if (rc == refs.end() || rc->second != 1) continue;

                    std::vector<SNode *> region_nodes;
                    if (has_else) {
                        region_nodes.insert(
                            region_nodes.end(),
                            ite->ElseList().begin(), ite->ElseList().end());
                    }
                    for (size_t j = i + 1; j < label_idx; ++j)
                        region_nodes.push_back(children[j]);

                    // Intermediate labels are safe only when every goto
                    // reference to those labels also lives inside the
                    // region being scoped.  That preserves local label
                    // traffic while refusing to move labels that are
                    // externally jumped into.
                    if (!ScopeifyLabelsAreRegionLocal(region_nodes, refs))
                        continue;

                    // Build scoped body from intermediates.  MakeSeq
                    // normalizes (drops nulls).
                    std::vector<SNode *> scoped_children;
                    scoped_children.reserve(region_nodes.size());
                    if (has_else) {
                        for (SNode *c : ite->ElseList())
                            scoped_children.push_back(c);
                    }
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

    // AbsorbCrossScopeIfGoto: fold
    //   if(outer){ if(inner) goto L; goto L2; } L:body
    // into
    //   if(outer && !inner) goto L2; body
    // The nested-if-then variant ScopeifyIfGotos doesn't handle.

    namespace {

        bool AbsorbCrossScopeIfGotoInSeq(
            std::vector< SNode * > &children,
            SNodeFactory &factory,
            clang::ASTContext &ctx,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            for (size_t i = 0; i + 1 < children.size(); ++i) {
                auto *outer = children[i]->dyn_cast<SIfThenElse>();
                if (!outer || !outer->Cond()) continue;
                if (!outer->ElseList().empty()) continue;
                const auto &then = outer->ThenList();
                if (then.size() < 2) continue;

                // Penultimate stmt must be SIfThenElse(inner, [SGoto L], null)
                auto *inner =
                    then[then.size() - 2]->dyn_cast<SIfThenElse>();
                if (!inner || !inner->Cond()) continue;
                if (!inner->ElseList().empty()) continue;
                std::string l_target =
                    ScopeifySingleGotoListTarget(inner->ThenList());
                if (l_target.empty()) continue;

                // Last stmt must be SGoto L2
                std::string l2_target = ScopeifyGotoTarget(then.back());
                if (l2_target.empty()) continue;
                if (l_target == l2_target) continue;

                std::string_view l_view(l_target);
                if (!RefCountIsOne(refs, l_view)) continue;

                size_t label_idx =
                    FindDirectLabelIndex(children, i + 1, l_view);
                if (label_idx >= children.size()) continue;

                // L must be the immediate next sibling — otherwise the
                // fold re-routes the inner-true path through intermediates
                // it originally skipped.
                if (label_idx != i + 1) continue;

                // Replace [inner_if, goto_L2] with `if(!inner) goto L2;`
                auto *new_goto = factory.Make<SGoto>(
                    factory.Intern(l2_target));
                auto *neg_inner = NegateExpr(
                    ctx, CloneExpr(ctx, inner->Cond()));
                auto *new_inner = factory.Make<SIfThenElse>(
                    neg_inner, std::vector<SNode *>{new_goto},
                    std::vector<SNode *>{});

                std::vector<SNode *> new_then(
                    then.begin(),
                    then.begin() + static_cast<ptrdiff_t>(then.size() - 2));
                new_then.push_back(new_inner);
                outer->SetThenBranch(std::move(new_then));

                // SLabel L is now unreferenced; unwrap.
                auto *lbl = children[label_idx]->as<SLabel>();
                std::vector<SNode *> body = lbl->BodyList();
                children.erase(
                    children.begin() + static_cast<ptrdiff_t>(label_idx));
                children.insert(
                    children.begin() + static_cast<ptrdiff_t>(label_idx),
                    body.begin(), body.end());
                return true;
            }
            return false;
        }

    } // anonymous namespace

    bool AbsorbCrossScopeIfGoto(std::vector< SNode * > &root,
                                SNodeFactory &factory,
                                clang::ASTContext &ctx) {
        bool any_changed = false;
        for (int pass = 0; pass < 8; ++pass) {
            std::unordered_map<std::string_view, int> refs;
            CountGotoRefs(root, refs);
            bool did = ForEachSeqPostOrder(
                root, [&](std::vector< SNode * > &seq) {
                    return AbsorbCrossScopeIfGotoInSeq(seq, factory, ctx, refs);
                });
            if (!did) break;
            any_changed = true;
        }
        return any_changed;
    }

    // RemoveDeadSSeqChildren — strip unreachable siblings.  Bottom-up:
    // when child[i] always terminates, remove siblings [i+1..end) that
    // are not SLabel nodes.  SLabels are preserved — they may be goto
    // targets from other scopes.

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
                // Both arms must terminate (if-then-without-else stays
                // false — Absorb only targets that shape).
                return !ite->ThenList().empty() && !ite->ElseList().empty()
                    && IsStrongTerminator(ite->ThenList().back())
                    && IsStrongTerminator(ite->ElseList().back());
            }
            // Loops/switches are NOT terminators here — their internals
            // get mutated by Absorb passes.
            return false;
        }

        bool RemoveDeadInSeq(std::vector< SNode * > &children) {
            bool changed = false;

            for (size_t i = 0; i + 1 < children.size(); ++i) {
                if (!IsStrongTerminator(children[i])) continue;
                // Trim label-free siblings; a label is a goto re-entry point.
                while (i + 1 < children.size()
                       && !ContainsLabel(children[i + 1])) {
                    children.erase(
                        children.begin() + static_cast<ptrdiff_t>(i) + 1);
                    changed = true;
                }
                break;
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

    // ConvertGotoToBreakContinue: goto-to-loop-exit → break;
    // goto-to-loop-header → continue.

    namespace {

        struct LoopScope {
            std::string_view exit_label;
            std::string_view header_label;
        };

        bool ConvertGotosInSeq(
            std::vector< SNode * > &seq, SNodeFactory &factory,
            std::vector<LoopScope> &scopes
        ) {
            bool changed = false;

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

    // ConvertGotoToReturn — replace goto-to-return patterns
    //
    // When an SGoto targets a label whose body is a single SReturn
    // (or a sequence whose last child is SReturn with no other control
    // flow), replace the goto with a cloned SReturn.

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

                SpliceAndAdvance(seq, i, repl);
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

    // CollapsePassThroughLabels / SimplifyEmptyControlFlow /
    // MergeRedundantGotoGuards.
    //
    // Labels whose body is just `goto other_label` add noise to the
    // structured output.  CollapsePassThroughLabels retargets SGoto
    // nodes through those aliases.  SimplifyEmptyControlFlow folds
    // empty label wrappers onto the following sibling and deletes
    // empty conditional shells that became no-ops.  MergeRedundantGoto-
    // Guards merges adjacent / else-if guarded gotos that target the
    // same label, preserving short-circuit condition order.

    namespace {

        // True iff \p body is a single statement that is just `goto T`,
        // either as an SGoto node or an SStmt wrapping a clang::GotoStmt.
        // Writes the target label name into \p target.
        bool SinglePassThroughTarget(
            const std::vector<SNode *> &body,
            std::string &target
        ) {
            if (body.size() != 1) return false;

            if (auto *g = body.front()->dyn_cast<SGoto>()) {
                target = std::string(g->Target());
                return true;
            }

            if (auto *st = body.front()->dyn_cast<SStmt>()) {
                if (auto *gs = llvm::dyn_cast_or_null<clang::GotoStmt>(
                        st->Stmt())) {
                    assert(gs->getLabel()
                           && "clang::GotoStmt missing target label");
                    target = gs->getLabel()->getName().str();
                    return true;
                }
            }

            return false;
        }

        void CollectPassThroughAliases(
            SNode *node,
            std::unordered_map<std::string_view, std::string> &aliases
        ) {
            if (!node) return;

            if (auto *lbl = node->dyn_cast<SLabel>()) {
                std::string target;
                if (SinglePassThroughTarget(lbl->BodyList(), target))
                    aliases[lbl->Name()] = std::move(target);
            }

            node->for_each_child(
                [&](SNode *c) { CollectPassThroughAliases(c, aliases); });
        }

        void CollectPassThroughAliases(
            const std::vector<SNode *> &seq,
            std::unordered_map<std::string_view, std::string> &aliases
        ) {
            for (auto *c : seq) CollectPassThroughAliases(c, aliases);
        }

        // Follow an alias chain to its final target.  Returns {} if the
        // chain is cyclic (a self-referential goto loop) so the caller
        // leaves the goto untouched.
        std::string ResolveAlias(
            std::string_view start,
            const std::unordered_map<std::string_view, std::string> &aliases
        ) {
            std::string current(start);
            std::unordered_set<std::string> seen;
            while (true) {
                if (!seen.insert(current).second)
                    return {};

                auto it = aliases.find(current);
                if (it == aliases.end())
                    return current;
                current = it->second;
            }
        }

        bool RewriteSGotoAliases(
            SNode *node,
            const std::unordered_map<std::string_view, std::string> &aliases,
            SNodeFactory &factory
        ) {
            if (!node) return false;

            if (auto *g = node->dyn_cast<SGoto>()) {
                std::string resolved = ResolveAlias(g->Target(), aliases);
                if (resolved.empty() || resolved == g->Target())
                    return false;
                g->SetTarget(factory.Intern(resolved));
                return true;
            }

            bool changed = false;
            node->for_each_child([&](SNode *c) {
                if (RewriteSGotoAliases(c, aliases, factory))
                    changed = true;
            });
            return changed;
        }

        bool RewriteSGotoAliases(
            const std::vector<SNode *> &seq,
            const std::unordered_map<std::string_view, std::string> &aliases,
            SNodeFactory &factory
        ) {
            bool changed = false;
            for (auto *c : seq)
                if (RewriteSGotoAliases(c, aliases, factory)) changed = true;
            return changed;
        }

        // Fold runs of empty SLabel wrappers onto the following sibling
        // (or, at the end of a sequence, onto each other) so that the
        // labels nest instead of cluttering the sequence.
        bool AttachEmptyLabelsInSeq(std::vector<SNode *> &seq) {
            bool changed = false;
            for (size_t i = 0; i < seq.size(); ++i) {
                auto *first = seq[i]->dyn_cast<SLabel>();
                if (!first || !first->BodyList().empty())
                    continue;

                std::vector<SLabel *> labels;
                size_t j = i;
                while (j < seq.size()) {
                    auto *lbl = seq[j]->dyn_cast<SLabel>();
                    if (!lbl || !lbl->BodyList().empty())
                        break;
                    labels.push_back(lbl);
                    ++j;
                }

                if (j == seq.size()) {
                    if (labels.size() < 2)
                        continue;
                    for (size_t k = labels.size() - 1; k > 0; --k) {
                        labels[k - 1]->AppendChild(labels[k]);
                    }
                    seq.erase(seq.begin() + static_cast<ptrdiff_t>(i + 1),
                              seq.end());
                    changed = true;
                    continue;
                }

                SNode *child = seq[j];
                for (size_t k = labels.size(); k > 0; --k) {
                    auto *lbl = labels[k - 1];
                    lbl->AppendChild(child);
                    child = lbl;
                }
                seq.erase(seq.begin() + static_cast<ptrdiff_t>(i + 1),
                          seq.begin() + static_cast<ptrdiff_t>(j + 1));
                changed = true;
            }
            return changed;
        }

        bool IsSideEffectFree(clang::Expr *expr, clang::ASTContext &ctx) {
            return !expr || !expr->HasSideEffects(ctx);
        }

        // Drop `if (c) {}` shells whose condition is side-effect free and
        // flip `if (c) {} else { B }` into `if (!c) { B }`.
        bool SimplifyEmptyIfsInSeq(
            std::vector<SNode *> &seq,
            clang::ASTContext &ctx
        ) {
            bool changed = false;
            for (size_t i = 0; i < seq.size(); ) {
                auto *ite = seq[i]->dyn_cast<SIfThenElse>();
                if (!ite) {
                    ++i;
                    continue;
                }

                bool then_empty = ite->ThenList().empty();
                bool else_empty = ite->ElseList().empty();

                if (then_empty && else_empty) {
                    if (!IsSideEffectFree(ite->Cond(), ctx)) {
                        ++i;
                        continue;
                    }
                    seq.erase(seq.begin() + static_cast<ptrdiff_t>(i));
                    changed = true;
                    continue;
                }

                if (then_empty && !else_empty && ite->Cond()) {
                    std::vector<SNode *> else_body =
                        std::move(ite->ElseList());
                    ite->SetCond(NegateExpr(ctx, CloneExpr(ctx, ite->Cond())));
                    ite->SetThenBranch(std::move(else_body));
                    ite->SetElseBranch(std::vector<SNode *>{});
                    changed = true;
                }

                ++i;
            }
            return changed;
        }

        // True iff \p body is a single statement that is just `goto T`,
        // returning the target as a string_view (no copy).
        bool SingleSGotoListTarget(
            const std::vector<SNode *> &body,
            std::string_view &target
        ) {
            if (body.size() != 1)
                return false;
            if (auto *g = body.front()->dyn_cast<SGoto>()) {
                target = g->Target();
                return true;
            }
            if (auto *st = body.front()->dyn_cast<SStmt>()) {
                if (auto *gs = llvm::dyn_cast_or_null<clang::GotoStmt>(
                        st->Stmt())) {
                    assert(gs->getLabel()
                           && "clang::GotoStmt missing target label");
                    target = gs->getLabel()->getName();
                    return true;
                }
            }
            return false;
        }

        clang::Stmt *SingleClangStmt(clang::Stmt *stmt) {
            while (auto *cs = llvm::dyn_cast_or_null<clang::CompoundStmt>(
                       stmt)) {
                if (cs->size() != 1)
                    return stmt;
                stmt = *cs->body_begin();
            }
            return stmt;
        }

        bool ClangGotoTarget(clang::Stmt *stmt, llvm::StringRef &target) {
            auto *gs = llvm::dyn_cast_or_null<clang::GotoStmt>(
                SingleClangStmt(stmt));
            if (!gs)
                return false;
            target = gs->getLabel()->getName();
            return true;
        }

        clang::Expr *ComparableExpr(clang::Expr *expr) {
            while (expr) {
                expr = expr->IgnoreParens();
                if (auto *ice = llvm::dyn_cast<clang::ImplicitCastExpr>(expr)) {
                    expr = ice->getSubExpr();
                    continue;
                }
                if (auto *cse = llvm::dyn_cast<clang::CStyleCastExpr>(expr)) {
                    expr = cse->getSubExpr();
                    continue;
                }
                break;
            }
            return expr;
        }

        bool ExprStructurallyEqual(clang::Expr *lhs, clang::Expr *rhs) {
            lhs = ComparableExpr(lhs);
            rhs = ComparableExpr(rhs);
            if (lhs == rhs)
                return true;
            if (!lhs || !rhs || lhs->getStmtClass() != rhs->getStmtClass())
                return false;

            if (auto *lbo = llvm::dyn_cast<clang::BinaryOperator>(lhs)) {
                auto *rbo = llvm::cast<clang::BinaryOperator>(rhs);
                return lbo->getOpcode() == rbo->getOpcode()
                    && ExprStructurallyEqual(lbo->getLHS(), rbo->getLHS())
                    && ExprStructurallyEqual(lbo->getRHS(), rbo->getRHS());
            }

            if (auto *luo = llvm::dyn_cast<clang::UnaryOperator>(lhs)) {
                auto *ruo = llvm::cast<clang::UnaryOperator>(rhs);
                return luo->getOpcode() == ruo->getOpcode()
                    && ExprStructurallyEqual(luo->getSubExpr(),
                                             ruo->getSubExpr());
            }

            if (auto *ldr = llvm::dyn_cast<clang::DeclRefExpr>(lhs)) {
                auto *rdr = llvm::cast<clang::DeclRefExpr>(rhs);
                return ldr->getDecl() == rdr->getDecl();
            }

            if (auto *lil = llvm::dyn_cast<clang::IntegerLiteral>(lhs)) {
                auto *ril = llvm::cast<clang::IntegerLiteral>(rhs);
                return lil->getValue() == ril->getValue();
            }

            if (auto *lbl = llvm::dyn_cast<clang::CXXBoolLiteralExpr>(lhs)) {
                auto *rbl = llvm::cast<clang::CXXBoolLiteralExpr>(rhs);
                return lbl->getValue() == rbl->getValue();
            }

            if (auto *lme = llvm::dyn_cast<clang::MemberExpr>(lhs)) {
                auto *rme = llvm::cast<clang::MemberExpr>(rhs);
                return lme->getMemberDecl() == rme->getMemberDecl()
                    && ExprStructurallyEqual(lme->getBase(), rme->getBase());
            }

            if (auto *las = llvm::dyn_cast<clang::ArraySubscriptExpr>(lhs)) {
                auto *ras = llvm::cast<clang::ArraySubscriptExpr>(rhs);
                return ExprStructurallyEqual(las->getLHS(), ras->getLHS())
                    && ExprStructurallyEqual(las->getRHS(), ras->getRHS());
            }

            return false;
        }

        clang::Expr *CreateLogicalBinary(
            clang::ASTContext &ctx,
            clang::Expr *lhs,
            clang::Expr *rhs,
            clang::BinaryOperatorKind opcode
        ) {
            auto loc = lhs ? lhs->getExprLoc() : VirtualLoc(ctx);
            return clang::BinaryOperator::Create(
                ctx, EnsureRValue(ctx, CloneExpr(ctx, lhs)),
                EnsureRValue(ctx, CloneExpr(ctx, rhs)), opcode,
                ctx.BoolTy, clang::VK_PRValue, clang::OK_Ordinary, loc,
                clang::FPOptionsOverride());
        }

        // Drop any sub-term of \p expr that is structurally equal to
        // \p covered_term, recursing through ||/&& trees.  Used to keep
        // a merged condition minimal: `(A || B) || B` collapses to
        // `A || B`.
        clang::Expr *RemoveOrCoveredTerm(
            clang::ASTContext &ctx,
            clang::Expr *expr,
            clang::Expr *covered_term
        ) {
            expr = ComparableExpr(expr);
            if (!expr)
                return nullptr;
            if (ExprStructurallyEqual(expr, covered_term))
                return nullptr;

            auto *bo = llvm::dyn_cast<clang::BinaryOperator>(expr);
            if (!bo)
                return expr;

            if (bo->getOpcode() == clang::BO_LOr) {
                clang::Expr *lhs =
                    RemoveOrCoveredTerm(ctx, bo->getLHS(), covered_term);
                clang::Expr *rhs =
                    RemoveOrCoveredTerm(ctx, bo->getRHS(), covered_term);
                if (!lhs) return rhs;
                if (!rhs) return lhs;
                if (lhs == bo->getLHS() && rhs == bo->getRHS())
                    return expr;
                return CreateLogicalBinary(ctx, lhs, rhs, clang::BO_LOr);
            }

            if (bo->getOpcode() == clang::BO_LAnd) {
                clang::Expr *lhs =
                    RemoveOrCoveredTerm(ctx, bo->getLHS(), covered_term);
                clang::Expr *rhs =
                    RemoveOrCoveredTerm(ctx, bo->getRHS(), covered_term);
                if (!lhs || !rhs)
                    return nullptr;
                if (lhs == bo->getLHS() && rhs == bo->getRHS())
                    return expr;
                return CreateLogicalBinary(ctx, lhs, rhs, clang::BO_LAnd);
            }

            return expr;
        }

        clang::Expr *BuildShortCircuitOr(
            clang::ASTContext &ctx,
            clang::Expr *lhs,
            clang::Expr *rhs
        ) {
            lhs = RemoveOrCoveredTerm(ctx, lhs, rhs);
            if (!lhs)
                return CloneExpr(ctx, rhs);
            return CreateLogicalBinary(ctx, lhs, rhs, clang::BO_LOr);
        }

        clang::Expr *BuildShortCircuitAnd(
            clang::ASTContext &ctx,
            clang::Expr *lhs,
            clang::Expr *rhs
        ) {
            return CreateLogicalBinary(ctx, lhs, rhs, clang::BO_LAnd);
        }

        // `if (A) { if (B) goto L; }`  ->  `if (A && B) goto L;`
        bool FlattenNestedGotoGuardsInSeq(
            std::vector<SNode *> &seq,
            SNodeFactory &factory,
            clang::ASTContext &ctx
        ) {
            bool changed = false;
            for (size_t i = 0; i < seq.size(); ++i) {
                auto *outer = seq[i]->dyn_cast<SIfThenElse>();
                if (!outer || !outer->Cond() || !outer->ElseList().empty()
                    || outer->ThenList().size() != 1)
                    continue;

                auto *inner = outer->ThenList().front()->dyn_cast<SIfThenElse>();
                if (!inner || !inner->Cond() || !inner->ElseList().empty())
                    continue;

                std::string_view target;
                if (!SingleSGotoListTarget(inner->ThenList(), target))
                    continue;

                std::vector<SNode *> then_body = {
                    factory.Make<SGoto>(factory.Intern(target)) };
                seq[i] = factory.Make<SIfThenElse>(
                    BuildShortCircuitAnd(ctx, outer->Cond(), inner->Cond()),
                    std::move(then_body), std::vector<SNode *>{});
                changed = true;
            }
            return changed;
        }

        // `if (A) goto L; else if (B) goto L;`  ->  `if (A || B) goto L;`
        bool MergeElseIfGotoGuardsInSeq(
            std::vector<SNode *> &seq,
            SNodeFactory &factory,
            clang::ASTContext &ctx
        ) {
            bool changed = false;
            for (size_t i = 0; i < seq.size(); ++i) {
                auto *outer = seq[i]->dyn_cast<SIfThenElse>();
                if (!outer || !outer->Cond() || outer->ElseList().size() != 1)
                    continue;

                std::string_view outer_target;
                if (!SingleSGotoListTarget(outer->ThenList(), outer_target))
                    continue;

                auto *inner = outer->ElseList().front()->dyn_cast<SIfThenElse>();
                if (!inner || !inner->Cond())
                    continue;

                std::string_view inner_target;
                if (!SingleSGotoListTarget(inner->ThenList(), inner_target)
                    || inner_target != outer_target)
                    continue;

                std::vector<SNode *> else_body = std::move(inner->ElseList());
                std::vector<SNode *> then_body = {
                    factory.Make<SGoto>(factory.Intern(outer_target)) };
                auto *merged = factory.Make<SIfThenElse>(
                    BuildShortCircuitOr(ctx, outer->Cond(), inner->Cond()),
                    std::move(then_body),
                    std::move(else_body));
                seq[i] = merged;
                changed = true;
            }
            return changed;
        }

        // `if (A) goto L; if (B) goto L;`  ->  `if (A || B) goto L;`
        bool MergeAdjacentGotoGuardsInSeq(
            std::vector<SNode *> &seq,
            SNodeFactory &factory,
            clang::ASTContext &ctx
        ) {
            bool changed = false;
            for (size_t i = 0; i + 1 < seq.size(); ) {
                auto *first = seq[i]->dyn_cast<SIfThenElse>();
                auto *second = seq[i + 1]->dyn_cast<SIfThenElse>();
                if (!first || !second || !first->Cond() || !second->Cond()
                    || !first->ElseList().empty()) {
                    ++i;
                    continue;
                }

                std::string_view first_target;
                std::string_view second_target;
                if (!SingleSGotoListTarget(first->ThenList(), first_target)
                    || !SingleSGotoListTarget(second->ThenList(), second_target)
                    || first_target != second_target) {
                    ++i;
                    continue;
                }

                std::vector<SNode *> else_body = std::move(second->ElseList());
                std::vector<SNode *> then_body = {
                    factory.Make<SGoto>(factory.Intern(first_target)) };
                auto *merged = factory.Make<SIfThenElse>(
                    BuildShortCircuitOr(ctx, first->Cond(), second->Cond()),
                    std::move(then_body),
                    std::move(else_body));
                seq[i] = merged;
                seq.erase(seq.begin() + static_cast<ptrdiff_t>(i + 1));
                changed = true;
            }
            return changed;
        }

        // Same as MergeElseIfGotoGuardsInSeq but for an else-if chain
        // expressed as a raw clang::IfStmt held inside an SStmt.
        bool MergeClangElseIfGotoGuardsInSeq(
            std::vector<SNode *> &seq,
            clang::ASTContext &ctx
        ) {
            bool changed = false;
            for (SNode *node : seq) {
                auto *st = node->dyn_cast<SStmt>();
                if (!st)
                    continue;

                auto *outer = llvm::dyn_cast_or_null<clang::IfStmt>(
                    SingleClangStmt(st->Stmt()));
                if (!outer || !outer->getCond() || !outer->getThen()
                    || !outer->getElse())
                    continue;

                llvm::StringRef outer_target;
                if (!ClangGotoTarget(outer->getThen(), outer_target))
                    continue;

                auto *inner = llvm::dyn_cast<clang::IfStmt>(
                    SingleClangStmt(outer->getElse()));
                if (!inner || !inner->getCond() || !inner->getThen())
                    continue;

                llvm::StringRef inner_target;
                if (!ClangGotoTarget(inner->getThen(), inner_target)
                    || inner_target != outer_target)
                    continue;

                auto loc = outer->getIfLoc();
                st->SetStmt(clang::IfStmt::Create(
                    ctx, loc, clang::IfStatementKind::Ordinary, nullptr,
                    nullptr,
                    BuildShortCircuitOr(ctx, outer->getCond(),
                                        inner->getCond()),
                    loc, loc, outer->getThen(), loc, inner->getElse()));
                changed = true;
            }
            return changed;
        }

        bool MergeGotoGuardsInSeq(
            std::vector<SNode *> &seq,
            SNodeFactory &factory,
            clang::ASTContext &ctx
        ) {
            bool any_changed = false;
            bool changed = true;
            while (changed) {
                changed = false;
                if (FlattenNestedGotoGuardsInSeq(seq, factory, ctx))
                    changed = true;
                if (MergeElseIfGotoGuardsInSeq(seq, factory, ctx))
                    changed = true;
                if (MergeAdjacentGotoGuardsInSeq(seq, factory, ctx))
                    changed = true;
                if (MergeClangElseIfGotoGuardsInSeq(seq, ctx))
                    changed = true;
                if (changed)
                    any_changed = true;
            }
            return any_changed;
        }

    } // anonymous namespace

    bool CollapsePassThroughLabels(std::vector<SNode *> &root,
                                   SNodeFactory &factory) {
        std::unordered_map<std::string_view, std::string> aliases;
        CollectPassThroughAliases(root, aliases);
        if (aliases.empty()) return false;
        return RewriteSGotoAliases(root, aliases, factory);
    }

    bool SimplifyEmptyControlFlow(std::vector<SNode *> &root,
                                  clang::ASTContext &ctx) {
        bool changed = false;
        changed |= ForEachSeqPostOrder(root, AttachEmptyLabelsInSeq);
        changed |= ForEachSeqPostOrder(
            root, [&](std::vector<SNode *> &seq) {
                return SimplifyEmptyIfsInSeq(seq, ctx);
            });
        changed |= ForEachSeqPostOrder(root, AttachEmptyLabelsInSeq);
        return changed;
    }

    bool MergeRedundantGotoGuards(std::vector<SNode *> &root,
                                  SNodeFactory &factory,
                                  clang::ASTContext &ctx) {
        return ForEachSeqPostOrder(
            root, [&](std::vector<SNode *> &seq) {
                return MergeGotoGuardsInSeq(seq, factory, ctx);
            });
    }

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
    // new reference to an already-live label.

    namespace {

        constexpr size_t kMaxCloneStmts = 8;

        // Aggregate clone budget for a single switch.  kMaxCloneStmts
        // bounds one clone; this bounds the sum across all case arms so
        // a switch with many arms all targeting the same label can't
        // clone a medium body into every arm.
        constexpr size_t kMaxSwitchCloneTotal = 24;

        // Aggregate clone budget for the general residual-goto target
        // duplicator.  Keep this modest: it is a readability cleanup, not an
        // unbounded tail-duplication optimizer.
        constexpr size_t kMaxGeneralCloneTotal = 64;

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
            // Sequences / SIfThenElse / SSwitch: safe iff every child is safe.
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

        /// Try to clone the complete terminating tail reached by
        /// \p target_label.  The first chunk is the label body; if that
        /// body falls through, include following unlabeled siblings until
        /// the cloned sequence terminates.  Returns an empty vector if the
        /// tail is unsafe, non-terminating, or too large.
        std::vector<SNode *> TryCloneLabelBody(
            std::string_view target_label,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            SNodeFactory &factory
        ) {
            auto it = labels.find(target_label);
            if (it == labels.end()) return {};
            auto *parent_seq = it->second.parent_seq;
            if (!parent_seq) return {};

            size_t label_idx = parent_seq->size();
            for (size_t i = 0; i < parent_seq->size(); ++i) {
                if ((*parent_seq)[i] == it->second.label) {
                    label_idx = i;
                    break;
                }
            }
            if (label_idx == parent_seq->size()) return {};

            std::vector<SNode *> tail;
            auto &label_body = it->second.label->BodyList();
            if (label_body.empty()) return {};
            tail.insert(tail.end(), label_body.begin(), label_body.end());

            for (size_t i = label_idx + 1;
                 !SeqAlwaysTerminates(tail) && i < parent_seq->size(); ++i) {
                SNode *next = (*parent_seq)[i];
                if (next->dyn_cast<SLabel>()) return {};
                tail.push_back(next);
            }

            if (!SeqAlwaysTerminates(tail)) return {};
            if (CountCloneSeq(tail) > kMaxCloneStmts) return {};
            for (auto *c : tail)
                if (!SubtreeIsSafeToClone(c)) return {};
            return CloneSeq(tail, factory);
        }

        bool SeqHasBreakContinue(const std::vector<SNode *> &seq);
        bool SeqHasGoto(const std::vector<SNode *> &seq);
        bool SeqHasLocalLiveIns(const std::vector<SNode *> &seq);
        std::unordered_set<const clang::VarDecl *>
        SeqLocalLiveIns(const std::vector<SNode *> &seq);

        bool SubtreeHasBreakContinue(const SNode *node) {
            if (!node) return false;
            if (node->dyn_cast<SBreak>() || node->dyn_cast<SContinue>())
                return true;
            bool found = false;
            node->for_each_child([&](SNode *child) {
                if (!found && SubtreeHasBreakContinue(child))
                    found = true;
            });
            return found;
        }

        bool SeqHasBreakContinue(const std::vector<SNode *> &seq) {
            for (auto *child : seq)
                if (SubtreeHasBreakContinue(child))
                    return true;
            return false;
        }

        bool ClangStmtHasGoto(const clang::Stmt *stmt) {
            if (!stmt) return false;
            if (llvm::isa<clang::GotoStmt>(stmt)) return true;
            for (const clang::Stmt *child : stmt->children())
                if (ClangStmtHasGoto(child)) return true;
            return false;
        }

        bool SubtreeHasGoto(const SNode *node) {
            if (!node) return false;
            if (node->dyn_cast<SGoto>()) return true;
            if (auto *st = node->dyn_cast<SStmt>())
                return ClangStmtHasGoto(st->Stmt());

            bool found = false;
            node->for_each_child([&](SNode *child) {
                if (!found && SubtreeHasGoto(child))
                    found = true;
            });
            return found;
        }

        bool SeqHasGoto(const std::vector<SNode *> &seq) {
            for (auto *child : seq)
                if (SubtreeHasGoto(child))
                    return true;
            return false;
        }

        bool IsLocalVarDecl(const clang::VarDecl *decl) {
            return decl && !llvm::isa<clang::ParmVarDecl>(decl)
                && !decl->hasGlobalStorage();
        }

        const clang::VarDecl *WrittenLocalDecl(clang::Expr *expr) {
            if (!expr) return nullptr;
            expr = expr->IgnoreParenImpCasts();
            if (auto *decl_ref = llvm::dyn_cast<clang::DeclRefExpr>(expr))
                if (auto *var = llvm::dyn_cast<clang::VarDecl>(
                        decl_ref->getDecl()))
                    if (IsLocalVarDecl(var))
                        return var;
            return nullptr;
        }

        void ExprCollectLocalLiveIns(
            clang::Expr *expr,
            const std::unordered_set<const clang::VarDecl *> &defined,
            std::unordered_set<const clang::VarDecl *> &live_ins
        );

        void StmtCollectLocalLiveIns(
            clang::Stmt *stmt,
            std::unordered_set<const clang::VarDecl *> &defined,
            std::unordered_set<const clang::VarDecl *> &live_ins
        );

        void ExprCollectLocalLiveIns(
            clang::Expr *expr,
            const std::unordered_set<const clang::VarDecl *> &defined,
            std::unordered_set<const clang::VarDecl *> &live_ins
        ) {
            if (!expr) return;
            expr = expr->IgnoreParenImpCasts();

            if (auto *decl_ref = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
                if (auto *var = llvm::dyn_cast<clang::VarDecl>(
                        decl_ref->getDecl()))
                    if (IsLocalVarDecl(var) && !defined.contains(var))
                        live_ins.insert(var);
                return;
            }

            for (clang::Stmt *child : expr->children()) {
                auto *child_expr = llvm::dyn_cast_or_null<clang::Expr>(child);
                if (child_expr)
                    ExprCollectLocalLiveIns(child_expr, defined, live_ins);
            }
        }

        void StmtCollectLocalLiveIns(
            clang::Stmt *stmt,
            std::unordered_set<const clang::VarDecl *> &defined,
            std::unordered_set<const clang::VarDecl *> &live_ins
        ) {
            if (!stmt) return;

            if (auto *decl_stmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
                for (clang::Decl *decl : decl_stmt->decls()) {
                    auto *var = llvm::dyn_cast<clang::VarDecl>(decl);
                    if (!var) continue;
                    if (var->getInit())
                        ExprCollectLocalLiveIns(
                            var->getInit(), defined, live_ins);
                    if (IsLocalVarDecl(var))
                        defined.insert(var);
                }
                return;
            }

            if (auto *bin = llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
                if (bin->isAssignmentOp()) {
                    ExprCollectLocalLiveIns(
                        bin->getRHS(), defined, live_ins);
                    if (!WrittenLocalDecl(bin->getLHS()))
                        ExprCollectLocalLiveIns(
                            bin->getLHS(), defined, live_ins);
                    if (const clang::VarDecl *written =
                            WrittenLocalDecl(bin->getLHS()))
                        defined.insert(written);
                    return;
                }
            }

            if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(stmt)) {
                ExprCollectLocalLiveIns(ifs->getCond(), defined, live_ins);
                auto then_defined = defined;
                auto else_defined = defined;
                StmtCollectLocalLiveIns(
                    ifs->getThen(), then_defined, live_ins);
                if (ifs->getElse()) {
                    StmtCollectLocalLiveIns(
                        ifs->getElse(), else_defined, live_ins);
                    for (const clang::VarDecl *decl : then_defined)
                        if (else_defined.contains(decl))
                            defined.insert(decl);
                }
                return;
            }

            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
                for (clang::Stmt *child : compound->body())
                    StmtCollectLocalLiveIns(child, defined, live_ins);
                return;
            }

            if (auto *expr = llvm::dyn_cast<clang::Expr>(stmt))
                return ExprCollectLocalLiveIns(expr, defined, live_ins);

            for (clang::Stmt *child : stmt->children()) {
                StmtCollectLocalLiveIns(child, defined, live_ins);
            }
        }

        void SNodeCollectLocalLiveIns(
            SNode *node,
            std::unordered_set<const clang::VarDecl *> &defined,
            std::unordered_set<const clang::VarDecl *> &live_ins
        ) {
            if (!node) return;
            if (auto *stmt = node->dyn_cast<SStmt>())
                return StmtCollectLocalLiveIns(
                    stmt->Stmt(), defined, live_ins);
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                ExprCollectLocalLiveIns(ite->Cond(), defined, live_ins);
                auto then_defined = defined;
                auto else_defined = defined;
                for (SNode *child : ite->ThenList())
                    SNodeCollectLocalLiveIns(
                        child, then_defined, live_ins);
                for (SNode *child : ite->ElseList())
                    SNodeCollectLocalLiveIns(
                        child, else_defined, live_ins);
                if (!ite->ElseList().empty())
                    for (const clang::VarDecl *decl : then_defined)
                        if (else_defined.contains(decl))
                            defined.insert(decl);
                return;
            }
            node->for_each_child([&](SNode *child) {
                SNodeCollectLocalLiveIns(child, defined, live_ins);
            });
        }

        std::unordered_set<const clang::VarDecl *>
        SeqLocalLiveIns(const std::vector<SNode *> &seq) {
            std::unordered_set<const clang::VarDecl *> defined;
            std::unordered_set<const clang::VarDecl *> live_ins;
            for (SNode *child : seq)
                SNodeCollectLocalLiveIns(child, defined, live_ins);
            return live_ins;
        }

        bool SeqHasLocalLiveIns(const std::vector<SNode *> &seq) {
            return !SeqLocalLiveIns(seq).empty();
        }

        void StmtCollectGuaranteedLocalDefs(
            clang::Stmt *stmt,
            std::unordered_set<const clang::VarDecl *> &defined
        );

        void SNodeCollectGuaranteedLocalDefs(
            SNode *node,
            std::unordered_set<const clang::VarDecl *> &defined
        );

        void StmtCollectGuaranteedLocalDefs(
            clang::Stmt *stmt,
            std::unordered_set<const clang::VarDecl *> &defined
        ) {
            if (!stmt) return;

            if (auto *decl_stmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
                for (clang::Decl *decl : decl_stmt->decls()) {
                    if (auto *var = llvm::dyn_cast<clang::VarDecl>(decl))
                        if (IsLocalVarDecl(var))
                            defined.insert(var);
                }
                return;
            }

            if (auto *bin = llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
                if (bin->isAssignmentOp()) {
                    if (const clang::VarDecl *written =
                            WrittenLocalDecl(bin->getLHS()))
                        defined.insert(written);
                    return;
                }
            }

            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
                for (clang::Stmt *child : compound->body())
                    StmtCollectGuaranteedLocalDefs(child, defined);
                return;
            }

            if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(stmt)) {
                auto then_defined = defined;
                auto else_defined = defined;
                StmtCollectGuaranteedLocalDefs(ifs->getThen(), then_defined);
                if (!ifs->getElse()) return;
                StmtCollectGuaranteedLocalDefs(ifs->getElse(), else_defined);
                for (const clang::VarDecl *decl : then_defined)
                    if (else_defined.contains(decl))
                        defined.insert(decl);
            }
        }

        void SNodeCollectGuaranteedLocalDefs(
            SNode *node,
            std::unordered_set<const clang::VarDecl *> &defined
        ) {
            if (!node) return;
            if (auto *stmt = node->dyn_cast<SStmt>()) {
                StmtCollectGuaranteedLocalDefs(stmt->Stmt(), defined);
                return;
            }
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                auto then_defined = defined;
                auto else_defined = defined;
                for (SNode *child : ite->ThenList())
                    SNodeCollectGuaranteedLocalDefs(child, then_defined);
                if (ite->ElseList().empty()) return;
                for (SNode *child : ite->ElseList())
                    SNodeCollectGuaranteedLocalDefs(child, else_defined);
                for (const clang::VarDecl *decl : then_defined)
                    if (else_defined.contains(decl))
                        defined.insert(decl);
                return;
            }
            node->for_each_child([&](SNode *child) {
                SNodeCollectGuaranteedLocalDefs(child, defined);
            });
        }

        std::unordered_set<const clang::VarDecl *> GuaranteedLocalDefsBefore(
            const std::vector<SNode *> &seq, size_t end
        ) {
            std::unordered_set<const clang::VarDecl *> defined;
            for (size_t i = 0; i < end; ++i)
                SNodeCollectGuaranteedLocalDefs(seq[i], defined);
            return defined;
        }

        bool LiveInsSatisfiedBy(
            const std::vector<SNode *> &seq,
            const std::unordered_set<const clang::VarDecl *> &defined
        ) {
            for (const clang::VarDecl *decl : SeqLocalLiveIns(seq))
                if (!defined.contains(decl))
                    return false;
            return true;
        }

        void CollectCompoundPrefixDefs(
            clang::CompoundStmt *compound,
            std::unordered_set<const clang::VarDecl *> &defined
        ) {
            if (!compound || compound->body_empty()) return;
            for (clang::Stmt *stmt : compound->body()) {
                if (stmt == compound->body_back())
                    break;
                StmtCollectGuaranteedLocalDefs(stmt, defined);
            }
        }

        /// Try to clone a target body for cross-arm inlining.  Unlike the
        /// direct switch-case clone path, this is used inside nested
        /// conditional arms, so avoid break/continue whose target would
        /// depend on the original lexical scope.
        std::vector<SNode *> TryCloneTerminatingLabelBody(
            std::string_view target_label,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            SNodeFactory &factory
        ) {
            auto clone = TryCloneLabelBody(target_label, labels, factory);
            if (SeqHasBreakContinue(clone)) return {};
            return clone;
        }

        std::string_view DirectGotoTarget(const SNode *node);
        std::vector<SNode *> BuildSplicedSeq(
            const std::vector<SNode *> &existing,
            const std::vector<SNode *> &clone, SNodeFactory &factory);

        std::vector<SNode *> TryCloneSmallEpilogueTarget(
            std::string_view target_label,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            SNodeFactory &factory
        ) {
            std::vector<SNode *> tail =
                TryCloneLabelBody(target_label, labels, factory);
            if (tail.empty()) return {};

            // Producer label: clone one hop through a terminal goto to the
            // shared epilogue, dropping the goto itself.
            if (auto next_target = TrailingGotoTargetSeq(tail);
                !next_target.empty()) {
                std::vector<SNode *> epilogue =
                    TryCloneLabelBody(next_target, labels, factory);
                if (epilogue.empty()) return {};
                std::vector<SNode *> spliced =
                    BuildSplicedSeq(tail, epilogue, factory);
                if (spliced.empty()) return {};
                tail = std::move(spliced);
            }

            if (!SeqAlwaysTerminates(tail)) return {};
            if (SeqHasBreakContinue(tail)) return {};
            if (SeqHasGoto(tail)) return {};
            if (CountCloneSeq(tail) > kMaxCloneStmts) return {};
            for (auto *child : tail)
                if (!SubtreeIsSafeToClone(child)) return {};
            return CloneSeq(tail, factory);
        }

        clang::Stmt *BuildSingleOrCompound(
            clang::ASTContext &ctx,
            const std::vector<clang::Stmt *> &stmts
        ) {
            if (stmts.empty()) return nullptr;
            if (stmts.size() == 1) return stmts.front();
            auto loc = VirtualLoc(ctx);
            return clang::CompoundStmt::Create(
                ctx, stmts, clang::FPOptionsOverride(), loc, loc);
        }

        bool FlattenClonedSNodeToStmt(
            SNode *node,
            clang::ASTContext &ctx,
            std::vector<clang::Stmt *> &out
        );

        bool FlattenClonedSeqToStmts(
            const std::vector<SNode *> &seq,
            clang::ASTContext &ctx,
            std::vector<clang::Stmt *> &out
        ) {
            for (SNode *node : seq)
                if (!FlattenClonedSNodeToStmt(node, ctx, out))
                    return false;
            return true;
        }

        bool FlattenClonedSNodeToStmt(
            SNode *node,
            clang::ASTContext &ctx,
            std::vector<clang::Stmt *> &out
        ) {
            if (!node) return true;
            if (auto *stmt = node->dyn_cast<SStmt>()) {
                out.push_back(stmt->Stmt());
                return true;
            }
            if (auto *ret = node->dyn_cast<SReturn>()) {
                out.push_back(MakeReturn(ctx, ret));
                return true;
            }
            if (node->dyn_cast<SBreak>()) {
                out.push_back(new (ctx) clang::BreakStmt(VirtualLoc(ctx)));
                return true;
            }
            if (node->dyn_cast<SContinue>()) {
                out.push_back(new (ctx) clang::ContinueStmt(VirtualLoc(ctx)));
                return true;
            }
            if (auto *label = node->dyn_cast<SLabel>())
                return FlattenClonedSeqToStmts(label->BodyList(), ctx, out);
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                std::vector<clang::Stmt *> then_stmts;
                std::vector<clang::Stmt *> else_stmts;
                if (!FlattenClonedSeqToStmts(
                        ite->ThenList(), ctx, then_stmts))
                    return false;
                if (!FlattenClonedSeqToStmts(
                        ite->ElseList(), ctx, else_stmts))
                    return false;

                auto loc = VirtualLoc(ctx);
                out.push_back(clang::IfStmt::Create(
                    ctx, loc, clang::IfStatementKind::Ordinary,
                    nullptr, nullptr, CloneExpr(ctx, ite->Cond()), loc, loc,
                    BuildSingleOrCompound(ctx, then_stmts), loc,
                    BuildSingleOrCompound(ctx, else_stmts)));
                return true;
            }
            return false;
        }

        bool IsSingleGotoTo(const std::vector<SNode *> &seq,
                            std::string_view target) {
            if (seq.size() != 1) return false;
            auto got = TrailingGotoTargetSeq(seq);
            return !got.empty() && got == target;
        }

        std::string_view DirectGotoTarget(const SNode *node) {
            if (!node) return {};
            if (auto *g = node->dyn_cast<SGoto>()) return g->Target();
            if (auto *st = node->dyn_cast<SStmt>()) {
                if (auto *gs = llvm::dyn_cast_or_null<clang::GotoStmt>(
                        st->Stmt()))
                    if (gs->getLabel())
                        return gs->getLabel()->getName();
            }
            return {};
        }

        size_t FindNextDirectLabel(
            const std::vector<SNode *> &seq, size_t begin
        ) {
            for (size_t i = begin; i < seq.size(); ++i)
                if (seq[i]->dyn_cast<SLabel>())
                    return i;
            return seq.size();
        }

        bool RefCountEquals(
            const std::unordered_map<std::string_view, int> &refs,
            std::string_view label, int expected
        ) {
            auto it = refs.find(label);
            int count = it == refs.end() ? 0 : it->second;
            return count == expected;
        }

        bool ClangStmtIsSiblingArmCloneSafe(const clang::Stmt *stmt) {
            if (!stmt) return true;
            if (llvm::isa<clang::LabelStmt>(stmt)
                || llvm::isa<clang::GotoStmt>(stmt)
                || llvm::isa<clang::BreakStmt>(stmt)
                || llvm::isa<clang::ContinueStmt>(stmt)
                || llvm::isa<clang::SwitchStmt>(stmt)
                || llvm::isa<clang::WhileStmt>(stmt)
                || llvm::isa<clang::DoStmt>(stmt)
                || llvm::isa<clang::ForStmt>(stmt))
                return false;

            for (const clang::Stmt *child : stmt->children())
                if (!ClangStmtIsSiblingArmCloneSafe(child))
                    return false;
            return true;
        }

        bool SNodeIsSiblingArmCloneSafe(const SNode *node) {
            if (!node) return true;
            if (auto *stmt = node->dyn_cast<SStmt>())
                return ClangStmtIsSiblingArmCloneSafe(stmt->Stmt());

            switch (node->Kind()) {
                case SNodeKind::kLabel:
                case SNodeKind::kGoto:
                case SNodeKind::kBreak:
                case SNodeKind::kContinue:
                case SNodeKind::kSwitch:
                case SNodeKind::kWhile:
                case SNodeKind::kDoWhile:
                case SNodeKind::kFor:
                    return false;
                default:
                    break;
            }

            bool safe = true;
            node->for_each_child([&](SNode *child) {
                if (safe && !SNodeIsSiblingArmCloneSafe(child))
                    safe = false;
            });
            return safe;
        }

        bool SiblingArmTailIsCloneSafe(const std::vector<SNode *> &tail) {
            if (tail.empty()) return false;
            if (CountCloneSeq(tail) > kMaxCloneStmts) return false;
            if (SeqHasLabel(tail) || SeqHasGoto(tail)
                || SeqHasBreakContinue(tail))
                return false;
            for (SNode *child : tail)
                if (!SNodeIsSiblingArmCloneSafe(child))
                    return false;
            return true;
        }

        bool SiblingArmTailIsMoveSafe(const std::vector<SNode *> &tail) {
            if (tail.empty()) return false;
            for (size_t i = 0; i < tail.size(); ++i) {
                if (i + 1 == tail.size()
                    && !DirectGotoTarget(tail[i]).empty())
                    continue;
                if (!SNodeIsSiblingArmCloneSafe(tail[i]))
                    return false;
            }
            return true;
        }

        std::string SiblingArmStmtText(
            clang::ASTContext &ctx, clang::Stmt *stmt
        ) {
            std::string out;
            llvm::raw_string_ostream os(out);
            clang::PrintingPolicy policy(ctx.getLangOpts());
            policy.SuppressTagKeyword = true;
            if (stmt)
                stmt->printPretty(os, nullptr, policy);
            os.flush();
            return out;
        }

        clang::Stmt *SingleSuffixStmtThroughUnreferencedLabel(
            SNode *node,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            if (!node)
                return nullptr;
            if (auto *stmt = node->dyn_cast<SStmt>())
                return stmt->Stmt();
            auto *label = node->dyn_cast<SLabel>();
            if (!label || !RefCountEquals(refs, label->Name(), 0)
                || label->BodyList().size() != 1)
                return nullptr;
            auto *stmt = label->BodyList().front()->dyn_cast<SStmt>();
            return stmt ? stmt->Stmt() : nullptr;
        }

        bool SameSiblingArmSuffixNode(
            clang::ASTContext &ctx, SNode *lhs, SNode *rhs,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            clang::Stmt *lhs_stmt =
                SingleSuffixStmtThroughUnreferencedLabel(lhs, refs);
            clang::Stmt *rhs_stmt =
                SingleSuffixStmtThroughUnreferencedLabel(rhs, refs);
            if (!lhs_stmt || !rhs_stmt)
                return false;
            if (!ClangStmtIsSiblingArmCloneSafe(lhs_stmt)
                || !ClangStmtIsSiblingArmCloneSafe(rhs_stmt))
                return false;
            return SiblingArmStmtText(ctx, lhs_stmt)
                == SiblingArmStmtText(ctx, rhs_stmt);
        }

        bool ClangStmtCanMoveAcrossSiblingArm(clang::Stmt *stmt) {
            if (!stmt)
                return true;
            if (llvm::isa<clang::DeclStmt>(stmt))
                return false;
            return ClangStmtIsSiblingArmCloneSafe(stmt);
        }

        size_t CommonSiblingArmSuffixSize(
            clang::ASTContext &ctx,
            const std::vector<SNode *> &lhs,
            const std::vector<SNode *> &rhs,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            size_t count = 0;
            while (count < lhs.size() && count < rhs.size()) {
                SNode *left = lhs[lhs.size() - count - 1];
                SNode *right = rhs[rhs.size() - count - 1];
                if (!SameSiblingArmSuffixNode(ctx, left, right, refs))
                    break;
                ++count;
            }
            return count;
        }

        bool TryBuildCompoundCommonSuffixParts(
            clang::ASTContext &ctx,
            const std::vector<SNode *> &label_tail,
            const std::vector<SNode *> &source_arm,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory,
            std::vector<SNode *> &label_prefix,
            std::vector<SNode *> &source_prefix,
            std::vector<SNode *> &common_suffix
        ) {
            label_prefix.clear();
            source_prefix.clear();
            common_suffix.clear();

            if (label_tail.size() != 1 || source_arm.empty())
                return false;
            auto *label_stmt = label_tail.front()->dyn_cast<SStmt>();
            if (!label_stmt)
                return false;
            auto *compound =
                llvm::dyn_cast_or_null<clang::CompoundStmt>(
                    label_stmt->Stmt());
            if (!compound || compound->size() < 2)
                return false;

            clang::Stmt *source_suffix =
                SingleSuffixStmtThroughUnreferencedLabel(
                    source_arm.back(), refs);
            if (!source_suffix)
                return false;

            std::vector<clang::Stmt *> label_stmts(
                compound->body_begin(), compound->body_end());
            clang::Stmt *label_suffix = label_stmts.back();
            if (!ClangStmtCanMoveAcrossSiblingArm(label_suffix)
                || !ClangStmtCanMoveAcrossSiblingArm(source_suffix))
                return false;
            if (SiblingArmStmtText(ctx, label_suffix)
                != SiblingArmStmtText(ctx, source_suffix))
                return false;

            for (size_t i = 0; i + 1 < label_stmts.size(); ++i) {
                if (!ClangStmtCanMoveAcrossSiblingArm(label_stmts[i]))
                    return false;
                label_prefix.push_back(factory.Make<SStmt>(label_stmts[i]));
            }
            if (!SiblingArmTailIsCloneSafe(label_prefix))
                return false;

            source_prefix.insert(
                source_prefix.end(), source_arm.begin(), source_arm.end() - 1);
            common_suffix.push_back(factory.Make<SStmt>(label_suffix));
            return !label_prefix.empty() && !source_prefix.empty();
        }

        bool BuildSiblingArmLabelTail(
            std::vector<SNode *> &arm, size_t label_idx,
            std::vector<SNode *> &tail,
            bool require_small_tail = true
        ) {
            tail.clear();
            if (label_idx >= arm.size())
                return false;

            auto *label = arm[label_idx]->dyn_cast<SLabel>();
            if (!label)
                return false;

            tail.insert(
                tail.end(), label->BodyList().begin(),
                label->BodyList().end());
            for (size_t i = label_idx + 1; i < arm.size(); ++i)
                tail.push_back(arm[i]);

            if (require_small_tail)
                return SiblingArmTailIsCloneSafe(tail);
            return SiblingArmTailIsMoveSafe(tail);
        }

        bool BuildSiblingArmConditionMergeTail(
            std::vector<SNode *> &arm,
            std::vector<SNode *> &tail
        ) {
            tail.clear();
            if (arm.empty())
                return false;

            auto *label = arm.front()->dyn_cast<SLabel>();
            if (!label)
                return false;

            tail.insert(
                tail.end(), label->BodyList().begin(),
                label->BodyList().end());
            for (size_t i = 1; i < arm.size(); ++i)
                tail.push_back(arm[i]);

            return !tail.empty();
        }

        bool AppendCloneableSiblingArmTailNode(
            SNode *node,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory,
            std::vector<SNode *> &out
        );

        bool AppendCloneableSiblingArmTailSeq(
            const std::vector<SNode *> &seq,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory,
            std::vector<SNode *> &out
        ) {
            for (SNode *node : seq)
                if (!AppendCloneableSiblingArmTailNode(
                        node, refs, factory, out))
                    return false;
            return true;
        }

        bool AppendCloneableSiblingArmTailNode(
            SNode *node,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory,
            std::vector<SNode *> &out
        ) {
            if (!node)
                return true;

            if (auto *label = node->dyn_cast<SLabel>()) {
                if (!RefCountEquals(refs, label->Name(), 0))
                    return false;
                return AppendCloneableSiblingArmTailSeq(
                    label->BodyList(), refs, factory, out);
            }

            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                std::vector<SNode *> then_body;
                std::vector<SNode *> else_body;
                if (!AppendCloneableSiblingArmTailSeq(
                        ite->ThenList(), refs, factory, then_body))
                    return false;
                if (!AppendCloneableSiblingArmTailSeq(
                        ite->ElseList(), refs, factory, else_body))
                    return false;
                out.push_back(factory.Make<SIfThenElse>(
                    ite->Cond(), std::move(then_body), std::move(else_body)));
                return true;
            }

            out.push_back(node);
            return true;
        }

        bool BuildCloneableSiblingArmLabelTail(
            std::vector<SNode *> &arm,
            size_t label_idx,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory,
            std::vector<SNode *> &tail
        ) {
            tail.clear();
            if (label_idx >= arm.size())
                return false;

            auto *label = arm[label_idx]->dyn_cast<SLabel>();
            if (!label)
                return false;

            std::vector<SNode *> raw_tail;
            raw_tail.insert(
                raw_tail.end(), label->BodyList().begin(),
                label->BodyList().end());
            for (size_t i = label_idx + 1; i < arm.size(); ++i)
                raw_tail.push_back(arm[i]);

            if (!AppendCloneableSiblingArmTailSeq(
                    raw_tail, refs, factory, tail))
                return false;
            return SiblingArmTailIsCloneSafe(tail);
        }

        void UnwrapSiblingArmLabel(
            std::vector<SNode *> &arm, size_t label_idx
        ) {
            auto *label = arm[label_idx]->as<SLabel>();
            std::vector<SNode *> body = label->BodyList();
            arm.erase(arm.begin() + static_cast<ptrdiff_t>(label_idx));
            arm.insert(
                arm.begin() + static_cast<ptrdiff_t>(label_idx),
                body.begin(), body.end());
        }

        bool IsSiblingArmDescentBarrier(const SNode *node) {
            if (!node) return true;
            return node->dyn_cast<SWhile>() || node->dyn_cast<SDoWhile>()
                || node->dyn_cast<SFor>() || node->dyn_cast<SSwitch>();
        }

        bool ReplaceTerminalGotoToSiblingLabelInSeq(
            std::vector<SNode *> &seq,
            std::string_view target,
            const std::vector<SNode *> &tail,
            SNodeFactory &factory,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            if (seq.empty())
                return false;

            size_t last_idx = seq.size() - 1;
            std::string_view direct = DirectGotoTarget(seq[last_idx]);
            if (!direct.empty()) {
                if (direct != target)
                    return false;
                std::vector<SNode *> clone = CloneSeq(tail, factory);
                if (clone.empty())
                    return false;
                SpliceAndAdvance(seq, last_idx, clone);
                return true;
            }

            SNode *last = seq[last_idx];
            if (IsSiblingArmDescentBarrier(last))
                return false;

            if (auto *ite = last->dyn_cast<SIfThenElse>()) {
                if (ReplaceTerminalGotoToSiblingLabelInSeq(
                        ite->ThenList(), target, tail, factory, refs))
                    return true;
                if (ReplaceTerminalGotoToSiblingLabelInSeq(
                        ite->ElseList(), target, tail, factory, refs))
                    return true;
            }

            if (auto *label = last->dyn_cast<SLabel>()) {
                if (RefCountEquals(refs, label->Name(), 0))
                    return ReplaceTerminalGotoToSiblingLabelInSeq(
                        label->BodyList(), target, tail, factory, refs);
            }

            return false;
        }

        bool TryFoldSiblingArmDirection(
            std::vector<SNode *> &label_arm,
            std::vector<SNode *> &goto_arm,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory
        ) {
            for (size_t i = 0; i < label_arm.size(); ++i) {
                auto *label = label_arm[i]->dyn_cast<SLabel>();
                if (!label)
                    continue;
                if (!RefCountEquals(refs, label->Name(), 1))
                    continue;

                std::vector<SNode *> tail;
                if (!BuildCloneableSiblingArmLabelTail(
                        label_arm, i, refs, factory, tail))
                    continue;

                if (!ReplaceTerminalGotoToSiblingLabelInSeq(
                        goto_arm, label->Name(), tail, factory, refs))
                    continue;

                UnwrapSiblingArmLabel(label_arm, i);
                return true;
            }
            return false;
        }

        bool SiblingArmFallthroughRemainderIsSafe(
            const std::vector<SNode *> &seq
        ) {
            return !SeqHasLabel(seq) && !SeqHasGoto(seq)
                && !SeqHasBreakContinue(seq);
        }

        bool ReplaceGuardedGotoToSiblingLabelInSeq(
            std::vector<SNode *> &seq,
            std::string_view target,
            const std::vector<SNode *> &tail,
            SNodeFactory &factory
        ) {
            for (size_t i = 0; i < seq.size(); ++i) {
                auto *guard = seq[i]->dyn_cast<SIfThenElse>();
                if (!guard || !guard->Cond())
                    continue;

                bool goto_in_then = IsSingleGotoTo(
                    guard->ThenList(), target);
                bool goto_in_else = IsSingleGotoTo(
                    guard->ElseList(), target);
                if (goto_in_then == goto_in_else)
                    continue;

                std::vector<SNode *> fallthrough =
                    goto_in_then ? guard->ElseList() : guard->ThenList();
                fallthrough.insert(
                    fallthrough.end(), seq.begin() + static_cast<ptrdiff_t>(i)
                                            + 1,
                    seq.end());
                if (!SiblingArmFallthroughRemainderIsSafe(fallthrough))
                    continue;

                std::vector<SNode *> cloned_tail = CloneSeq(tail, factory);
                if (cloned_tail.empty())
                    continue;

                std::vector<SNode *> prefix(
                    seq.begin(), seq.begin() + static_cast<ptrdiff_t>(i));
                std::vector<SNode *> fallthrough_body =
                    factory.MakeSeq(std::move(fallthrough));

                if (goto_in_then) {
                    prefix.push_back(factory.Make<SIfThenElse>(
                        guard->Cond(), std::move(cloned_tail),
                        std::move(fallthrough_body)));
                } else {
                    prefix.push_back(factory.Make<SIfThenElse>(
                        guard->Cond(), std::move(fallthrough_body),
                        std::move(cloned_tail)));
                }

                seq = std::move(prefix);
                return true;
            }
            return false;
        }

        bool TryFoldSiblingArmGuardedDispatch(
            std::vector<SNode *> &label_arm,
            std::vector<SNode *> &goto_arm,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory
        ) {
            for (size_t i = 0; i < label_arm.size(); ++i) {
                auto *label = label_arm[i]->dyn_cast<SLabel>();
                if (!label)
                    continue;
                if (!RefCountEquals(refs, label->Name(), 1))
                    continue;

                std::vector<SNode *> tail;
                if (!BuildSiblingArmLabelTail(label_arm, i, tail))
                    continue;

                if (!ReplaceGuardedGotoToSiblingLabelInSeq(
                        goto_arm, label->Name(), tail, factory))
                    continue;

                UnwrapSiblingArmLabel(label_arm, i);
                return true;
            }
            return false;
        }

        bool ExtractLeadingSiblingArmGotoDispatch(
            const std::vector<SNode *> &source_arm,
            std::string_view target,
            const std::unordered_map<std::string_view, int> &refs,
            clang::ASTContext &ctx,
            clang::Expr *&guard_cond,
            std::vector<SNode *> &source_remainder
        ) {
            guard_cond = nullptr;
            source_remainder.clear();
            if (source_arm.empty())
                return false;

            if (auto *label = source_arm.front()->dyn_cast<SLabel>()) {
                if (!RefCountEquals(refs, label->Name(), 0))
                    return false;

                std::vector<SNode *> unwrapped(
                    label->BodyList().begin(), label->BodyList().end());
                unwrapped.insert(
                    unwrapped.end(), source_arm.begin() + 1,
                    source_arm.end());
                return ExtractLeadingSiblingArmGotoDispatch(
                    unwrapped, target, refs, ctx, guard_cond,
                    source_remainder);
            }

            auto *guard = source_arm.front()->dyn_cast<SIfThenElse>();
            if (!guard || !guard->Cond())
                return false;

            auto append_rest = [&]() {
                source_remainder.insert(
                    source_remainder.end(), source_arm.begin() + 1,
                    source_arm.end());
            };

            if (guard->ThenList().size() == 1
                && DirectGotoTarget(guard->ThenList().front()) == target) {
                guard_cond = CloneExpr(ctx, guard->Cond());
                source_remainder.insert(
                    source_remainder.end(), guard->ElseList().begin(),
                    guard->ElseList().end());
                append_rest();
                return true;
            }

            if (guard->ElseList().size() == 1
                && DirectGotoTarget(guard->ElseList().front()) == target) {
                guard_cond = NegateExpr(ctx, CloneExpr(ctx, guard->Cond()));
                source_remainder.insert(
                    source_remainder.end(), guard->ThenList().begin(),
                    guard->ThenList().end());
                append_rest();
                return true;
            }

            return false;
        }

        bool TryFoldSiblingArmConditionMerge(
            SIfThenElse *ite,
            bool label_in_then,
            const std::unordered_map<std::string_view, int> &refs,
            clang::ASTContext &ctx
        ) {
            std::vector<SNode *> &label_arm =
                label_in_then ? ite->ThenList() : ite->ElseList();
            std::vector<SNode *> &source_arm =
                label_in_then ? ite->ElseList() : ite->ThenList();
            if (label_arm.empty() || source_arm.empty())
                return false;

            auto *label = label_arm.front()->dyn_cast<SLabel>();
            if (!label)
                return false;
            if (!RefCountEquals(refs, label->Name(), 1))
                return false;

            std::vector<SNode *> label_tail;
            if (!BuildSiblingArmConditionMergeTail(label_arm, label_tail))
                return false;

            clang::Expr *guard_cond = nullptr;
            std::vector<SNode *> source_remainder;
            if (!ExtractLeadingSiblingArmGotoDispatch(
                    source_arm, label->Name(), refs, ctx, guard_cond,
                    source_remainder))
                return false;
            if (!guard_cond)
                return false;

            if (label_in_then) {
                ite->SetCond(
                    BuildShortCircuitOr(ctx, ite->Cond(), guard_cond));
                ite->SetThenBranch(std::move(label_tail));
                ite->SetElseBranch(std::move(source_remainder));
            } else {
                clang::Expr *source_cond = BuildShortCircuitAnd(
                    ctx, ite->Cond(), NegateExpr(ctx, guard_cond));
                ite->SetCond(source_cond);
                ite->SetThenBranch(std::move(source_remainder));
                ite->SetElseBranch(std::move(label_tail));
            }
            return true;
        }

        bool TryFoldSiblingArmCommonSuffix(
            std::vector<SNode *> &seq,
            size_t if_idx,
            bool label_in_then,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory,
            clang::ASTContext &ctx
        ) {
            auto *ite = seq[if_idx]->dyn_cast<SIfThenElse>();
            if (!ite || ite->ThenList().empty() || ite->ElseList().empty())
                return false;

            std::vector<SNode *> &label_arm =
                label_in_then ? ite->ThenList() : ite->ElseList();
            std::vector<SNode *> &source_arm =
                label_in_then ? ite->ElseList() : ite->ThenList();
            if (label_arm.empty() || source_arm.empty())
                return false;

            auto *label = label_arm.front()->dyn_cast<SLabel>();
            if (!label || !RefCountEquals(refs, label->Name(), 1))
                return false;

            std::vector<SNode *> label_tail;
            if (!BuildSiblingArmLabelTail(label_arm, 0, label_tail))
                return false;

            std::vector<SNode *> label_prefix;
            std::vector<SNode *> source_prefix;
            std::vector<SNode *> common_suffix;

            size_t suffix =
                CommonSiblingArmSuffixSize(ctx, label_tail, source_arm, refs);
            if (suffix != 0 && suffix < label_tail.size()
                && suffix < source_arm.size()) {
                label_prefix.assign(
                    label_tail.begin(),
                    label_tail.end() - static_cast<ptrdiff_t>(suffix));
                source_prefix.assign(
                    source_arm.begin(),
                    source_arm.end() - static_cast<ptrdiff_t>(suffix));
                common_suffix.assign(
                    label_tail.end() - static_cast<ptrdiff_t>(suffix),
                    label_tail.end());

                if (!SiblingArmTailIsCloneSafe(label_prefix)
                    || source_prefix.empty() || common_suffix.empty())
                    return false;
            } else if (!TryBuildCompoundCommonSuffixParts(
                           ctx, label_tail, source_arm, refs, factory,
                           label_prefix, source_prefix, common_suffix))
            {
                return false;
            }

            if (!ReplaceTerminalGotoToSiblingLabelInSeq(
                    source_prefix, label->Name(), label_prefix, factory, refs))
                return false;

            if (label_in_then) {
                ite->SetThenBranch(std::move(label_prefix));
                ite->SetElseBranch(std::move(source_prefix));
            } else {
                ite->SetThenBranch(std::move(source_prefix));
                ite->SetElseBranch(std::move(label_prefix));
            }
            seq.insert(
                seq.begin() + static_cast<ptrdiff_t>(if_idx + 1),
                common_suffix.begin(), common_suffix.end());
            return true;
        }

        bool FoldSiblingArmEntriesInIf(
            SIfThenElse *ite,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory,
            clang::ASTContext &ctx
        ) {
            if (!ite || ite->ThenList().empty() || ite->ElseList().empty())
                return false;

            if (TryFoldSiblingArmConditionMerge(
                    ite, /*label_in_then=*/true, refs, ctx))
                return true;
            if (TryFoldSiblingArmConditionMerge(
                    ite, /*label_in_then=*/false, refs, ctx))
                return true;

            if (TryFoldSiblingArmDirection(
                    ite->ThenList(), ite->ElseList(), refs, factory))
                return true;
            if (TryFoldSiblingArmDirection(
                    ite->ElseList(), ite->ThenList(), refs, factory))
                return true;
            if (TryFoldSiblingArmGuardedDispatch(
                    ite->ThenList(), ite->ElseList(), refs, factory))
                return true;
            if (TryFoldSiblingArmGuardedDispatch(
                    ite->ElseList(), ite->ThenList(), refs, factory))
                return true;
            return false;
        }

        bool FoldSiblingArmEntriesInSeq(
            std::vector<SNode *> &seq,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory,
            clang::ASTContext &ctx
        ) {
            for (size_t i = 0; i < seq.size(); ++i) {
                SNode *node = seq[i];
                auto *ite = node->dyn_cast<SIfThenElse>();
                if (!ite)
                    continue;
                if (FoldSiblingArmEntriesInIf(ite, refs, factory, ctx))
                    return true;
                if (TryFoldSiblingArmCommonSuffix(
                        seq, i, /*label_in_then=*/true, refs, factory, ctx))
                    return true;
                if (TryFoldSiblingArmCommonSuffix(
                        seq, i, /*label_in_then=*/false, refs, factory, ctx))
                    return true;
            }
            return false;
        }

        clang::Expr *BuildGuardPathCondition(
            clang::ASTContext &ctx,
            const std::vector<clang::Expr *> &terms
        ) {
            if (terms.empty()) return nullptr;
            clang::Expr *cond = CloneExpr(ctx, terms.front());
            for (size_t i = 1; i < terms.size(); ++i)
                cond = BuildShortCircuitAnd(ctx, cond, terms[i]);
            return cond;
        }

        bool ExtractMovableFallthroughTail(
            std::vector<SNode *> &seq,
            size_t target_idx,
            size_t join_idx,
            std::vector<SNode *> &out
        ) {
            out.clear();
            if (target_idx >= seq.size() || join_idx <= target_idx
                || join_idx > seq.size())
                return false;

            auto *target_label = seq[target_idx]->dyn_cast<SLabel>();
            if (!target_label) return false;

            out.insert(out.end(), target_label->BodyList().begin(),
                       target_label->BodyList().end());
            for (size_t i = target_idx + 1; i < join_idx; ++i) {
                if (seq[i]->dyn_cast<SLabel>())
                    return false;
                out.push_back(seq[i]);
            }

            if (out.empty()) return false;
            if (SeqHasLabel(out) || SeqHasGoto(out)) {
                out.clear();
                return false;
            }
            return true;
        }

        enum class GuardLeafKind { Invalid, JoinOnly, TargetPath };

        GuardLeafKind MergeGuardLeafKinds(
            GuardLeafKind lhs, GuardLeafKind rhs
        ) {
            if (lhs == GuardLeafKind::Invalid || rhs == GuardLeafKind::Invalid)
                return GuardLeafKind::Invalid;
            if (lhs == GuardLeafKind::TargetPath
                && rhs == GuardLeafKind::TargetPath)
                return GuardLeafKind::Invalid;
            if (lhs == GuardLeafKind::TargetPath
                || rhs == GuardLeafKind::TargetPath)
                return GuardLeafKind::TargetPath;
            return GuardLeafKind::JoinOnly;
        }

        GuardLeafKind ClassifySNodeGuardSeq(
            const std::vector<SNode *> &body,
            std::string_view target_label,
            std::string_view join_label,
            clang::ASTContext &ctx,
            std::vector<clang::Expr *> &target_terms
        );

        GuardLeafKind ClassifySNodeGuardIf(
            const SIfThenElse *ifs,
            std::string_view target_label,
            std::string_view join_label,
            clang::ASTContext &ctx,
            std::vector<clang::Expr *> &target_terms
        ) {
            if (!ifs || !ifs->Cond() || ifs->ElseList().empty())
                return GuardLeafKind::Invalid;

            std::vector<clang::Expr *> then_terms;
            std::vector<clang::Expr *> else_terms;
            GuardLeafKind then_kind = ClassifySNodeGuardSeq(
                ifs->ThenList(), target_label, join_label, ctx, then_terms);
            GuardLeafKind else_kind = ClassifySNodeGuardSeq(
                ifs->ElseList(), target_label, join_label, ctx, else_terms);

            GuardLeafKind merged =
                MergeGuardLeafKinds(then_kind, else_kind);
            if (merged != GuardLeafKind::TargetPath)
                return merged;

            if (then_kind == GuardLeafKind::TargetPath) {
                target_terms.push_back(CloneExpr(ctx, ifs->Cond()));
                target_terms.insert(target_terms.end(), then_terms.begin(),
                                    then_terms.end());
            } else {
                target_terms.push_back(
                    NegateExpr(ctx, CloneExpr(ctx, ifs->Cond())));
                target_terms.insert(target_terms.end(), else_terms.begin(),
                                    else_terms.end());
            }
            return GuardLeafKind::TargetPath;
        }

        GuardLeafKind ClassifySNodeGuardSeq(
            const std::vector<SNode *> &body,
            std::string_view target_label,
            std::string_view join_label,
            clang::ASTContext &ctx,
            std::vector<clang::Expr *> &target_terms
        ) {
            if (body.size() != 1)
                return GuardLeafKind::Invalid;

            std::string_view direct = DirectGotoTarget(body.front());
            if (!direct.empty()) {
                if (direct == target_label)
                    return GuardLeafKind::TargetPath;
                if (direct == join_label)
                    return GuardLeafKind::JoinOnly;
                return GuardLeafKind::Invalid;
            }

            if (auto *nested = body.front()->dyn_cast<SIfThenElse>())
                return ClassifySNodeGuardIf(
                    nested, target_label, join_label, ctx, target_terms);

            return GuardLeafKind::Invalid;
        }

        GuardLeafKind ClassifyClangGuardStmt(
            clang::Stmt *stmt,
            std::string_view target_label,
            std::string_view join_label,
            clang::ASTContext &ctx,
            std::vector<clang::Expr *> &target_terms
        );

        GuardLeafKind ClassifyClangGuardIf(
            clang::IfStmt *ifs,
            std::string_view target_label,
            std::string_view join_label,
            clang::ASTContext &ctx,
            std::vector<clang::Expr *> &target_terms
        ) {
            if (!ifs || !ifs->getCond() || !ifs->getThen()
                || !ifs->getElse())
                return GuardLeafKind::Invalid;

            std::vector<clang::Expr *> then_terms;
            std::vector<clang::Expr *> else_terms;
            GuardLeafKind then_kind = ClassifyClangGuardStmt(
                ifs->getThen(), target_label, join_label, ctx, then_terms);
            GuardLeafKind else_kind = ClassifyClangGuardStmt(
                ifs->getElse(), target_label, join_label, ctx, else_terms);

            GuardLeafKind merged =
                MergeGuardLeafKinds(then_kind, else_kind);
            if (merged != GuardLeafKind::TargetPath)
                return merged;

            if (then_kind == GuardLeafKind::TargetPath) {
                target_terms.push_back(CloneExpr(ctx, ifs->getCond()));
                target_terms.insert(target_terms.end(), then_terms.begin(),
                                    then_terms.end());
            } else {
                target_terms.push_back(
                    NegateExpr(ctx, CloneExpr(ctx, ifs->getCond())));
                target_terms.insert(target_terms.end(), else_terms.begin(),
                                    else_terms.end());
            }
            return GuardLeafKind::TargetPath;
        }

        GuardLeafKind ClassifyClangGuardStmt(
            clang::Stmt *stmt,
            std::string_view target_label,
            std::string_view join_label,
            clang::ASTContext &ctx,
            std::vector<clang::Expr *> &target_terms
        ) {
            if (!stmt) return GuardLeafKind::Invalid;

            if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(stmt)) {
                if (!gs->getLabel()) return GuardLeafKind::Invalid;
                std::string_view name = gs->getLabel()->getName();
                if (name == target_label)
                    return GuardLeafKind::TargetPath;
                if (name == join_label)
                    return GuardLeafKind::JoinOnly;
                return GuardLeafKind::Invalid;
            }

            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
                if (compound->body_empty())
                    return GuardLeafKind::Invalid;
                auto it = compound->body_begin();
                ++it;
                if (it != compound->body_end())
                    return GuardLeafKind::Invalid;
                return ClassifyClangGuardStmt(
                    compound->body_front(), target_label, join_label, ctx,
                    target_terms);
            }

            if (auto *nested = llvm::dyn_cast<clang::IfStmt>(stmt))
                return ClassifyClangGuardIf(
                    nested, target_label, join_label, ctx, target_terms);

            return GuardLeafKind::Invalid;
        }

        bool FoldGuardedFallthroughInSeq(
            std::vector<SNode *> &seq,
            const std::unordered_map<std::string_view, int> &refs,
            clang::ASTContext &ctx
        ) {
            for (size_t i = 0; i + 2 < seq.size(); ++i) {
                size_t target_idx = FindNextDirectLabel(seq, i + 1);
                if (target_idx != i + 1)
                    continue;
                size_t join_idx = FindNextDirectLabel(seq, target_idx + 1);
                if (join_idx >= seq.size())
                    continue;

                auto *target_label = seq[target_idx]->as<SLabel>();
                auto *join_label = seq[join_idx]->as<SLabel>();
                if (!RefCountEquals(refs, target_label->Name(), 1))
                    continue;

                std::vector<SNode *> target_body;
                if (!ExtractMovableFallthroughTail(
                        seq, target_idx, join_idx, target_body))
                    continue;
                if (auto *guard_if = seq[i]->dyn_cast<SIfThenElse>()) {
                    std::vector<clang::Expr *> target_terms;
                    GuardLeafKind kind = ClassifySNodeGuardIf(
                        guard_if, target_label->Name(), join_label->Name(),
                        ctx, target_terms);
                    if (kind != GuardLeafKind::TargetPath)
                        continue;

                    clang::Expr *cond =
                        BuildGuardPathCondition(ctx, target_terms);
                    if (!cond) continue;

                    guard_if->SetCond(cond);
                    guard_if->SetThenBranch(std::move(target_body));
                    guard_if->SetElseBranch(std::vector<SNode *>{});

                    seq.erase(seq.begin() + static_cast<ptrdiff_t>(target_idx),
                              seq.begin() + static_cast<ptrdiff_t>(join_idx));
                    return true;
                }

                auto *stmt_node = seq[i]->dyn_cast<SStmt>();
                if (!stmt_node)
                    continue;
                auto *guard_if = llvm::dyn_cast_or_null<clang::IfStmt>(
                    stmt_node->Stmt());
                if (!guard_if)
                    continue;

                std::vector<clang::Expr *> target_terms;
                GuardLeafKind kind = ClassifyClangGuardIf(
                    guard_if, target_label->Name(), join_label->Name(), ctx,
                    target_terms);
                if (kind != GuardLeafKind::TargetPath)
                    continue;

                clang::Expr *cond =
                    BuildGuardPathCondition(ctx, target_terms);
                if (!cond) continue;

                std::vector<clang::Stmt *> body_stmts;
                if (!FlattenClonedSeqToStmts(target_body, ctx, body_stmts)
                    || body_stmts.empty())
                    continue;

                guard_if->setCond(cond);
                guard_if->setThen(BuildSingleOrCompound(ctx, body_stmts));
                guard_if->setElse(nullptr);

                seq.erase(seq.begin() + static_cast<ptrdiff_t>(target_idx),
                          seq.begin() + static_cast<ptrdiff_t>(join_idx));
                return true;
            }
            return false;
        }

        struct NestedEntryLabelLoc {
            SLabel *label = nullptr;
            std::vector<SNode *> *parent_seq = nullptr;
            size_t idx = 0;
        };

        struct ClangNestedEntryLabelLoc {
            clang::LabelStmt *label = nullptr;
            clang::IfStmt *owner_if = nullptr;
            bool is_then = false;
        };

        bool FindNestedEntryLabelInSeq(
            std::vector<SNode *> &seq,
            std::string_view target,
            NestedEntryLabelLoc &out
        );

        bool FindNestedEntryLabelInNode(
            SNode *node,
            std::string_view target,
            NestedEntryLabelLoc &out
        ) {
            if (!node) return false;

            // Entering loops or switches below their header/case dispatch
            // changes break/continue/case scope.  Leave those to later,
            // loop-aware duplication phases.
            if (node->dyn_cast<SWhile>() || node->dyn_cast<SDoWhile>()
                || node->dyn_cast<SFor>() || node->dyn_cast<SSwitch>())
                return false;

            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                if (FindNestedEntryLabelInSeq(ite->ThenList(), target, out))
                    return true;
                if (FindNestedEntryLabelInSeq(ite->ElseList(), target, out))
                    return true;
            }
            if (auto *lbl = node->dyn_cast<SLabel>())
                return FindNestedEntryLabelInSeq(lbl->BodyList(), target, out);
            return false;
        }

        bool FindNestedEntryLabelInSeq(
            std::vector<SNode *> &seq,
            std::string_view target,
            NestedEntryLabelLoc &out
        ) {
            for (size_t i = 0; i < seq.size(); ++i) {
                auto *lbl = seq[i]->dyn_cast<SLabel>();
                if (lbl && lbl->Name() == target) {
                    if (i + 1 != seq.size())
                        return false;
                    out = {lbl, &seq, i};
                    return true;
                }

                NestedEntryLabelLoc nested;
                if (!FindNestedEntryLabelInNode(seq[i], target, nested))
                    continue;
                if (i + 1 != seq.size())
                    return false;
                out = nested;
                return true;
            }
            return false;
        }

        bool FindNestedEntryLabelInLaterSibling(
            std::vector<SNode *> &seq,
            size_t begin,
            std::string_view target,
            size_t &carrier_idx,
            NestedEntryLabelLoc &out
        ) {
            for (size_t i = begin; i < seq.size(); ++i) {
                NestedEntryLabelLoc loc;
                if (FindNestedEntryLabelInNode(seq[i], target, loc)) {
                    carrier_idx = i;
                    out = loc;
                    return true;
                }
            }
            return false;
        }

        bool FindClangNestedEntryLabelInStmt(
            clang::Stmt *stmt,
            std::string_view target,
            ClangNestedEntryLabelLoc &out
        ) {
            auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(stmt);
            if (!ifs)
                return false;

            auto arm_label = [&](clang::Stmt *arm,
                                 bool is_then) -> clang::LabelStmt * {
                auto *label = llvm::dyn_cast_or_null<clang::LabelStmt>(arm);
                if (!label || !label->getDecl())
                    return nullptr;
                if (label->getDecl()->getName() != llvm::StringRef(target))
                    return nullptr;
                out = {label, ifs, is_then};
                return label;
            };

            if (arm_label(ifs->getThen(), true))
                return true;
            if (arm_label(ifs->getElse(), false))
                return true;
            return false;
        }

        bool FindClangNestedEntryLabelInLaterSibling(
            std::vector<SNode *> &seq,
            size_t begin,
            std::string_view target,
            size_t &carrier_idx,
            ClangNestedEntryLabelLoc &out
        ) {
            for (size_t i = begin; i < seq.size(); ++i) {
                auto *stmt_node = seq[i]->dyn_cast<SStmt>();
                if (!stmt_node)
                    continue;
                ClangNestedEntryLabelLoc loc;
                if (FindClangNestedEntryLabelInStmt(
                        stmt_node->Stmt(), target, loc)) {
                    carrier_idx = i;
                    out = loc;
                    return true;
                }
            }
            return false;
        }

        size_t CountClangEntryStmts(const clang::Stmt *stmt) {
            if (!stmt) return 0;
            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
                size_t count = 0;
                for (const clang::Stmt *child : compound->body()) {
                    count += CountClangEntryStmts(child);
                    if (count > kMaxCloneStmts)
                        return count;
                }
                return count;
            }
            return 1;
        }

        bool ClangEntryStmtIsCloneSafe(const clang::Stmt *stmt) {
            if (!stmt) return false;
            if (CountClangEntryStmts(stmt) > kMaxCloneStmts)
                return false;

            std::function<bool(const clang::Stmt *)> safe =
                [&](const clang::Stmt *cur) -> bool {
                    if (!cur) return true;
                    if (llvm::isa<clang::LabelStmt>(cur)
                        || llvm::isa<clang::GotoStmt>(cur)
                        || llvm::isa<clang::BreakStmt>(cur)
                        || llvm::isa<clang::ContinueStmt>(cur)
                        || llvm::isa<clang::SwitchStmt>(cur)
                        || llvm::isa<clang::WhileStmt>(cur)
                        || llvm::isa<clang::DoStmt>(cur)
                        || llvm::isa<clang::ForStmt>(cur))
                        return false;
                    for (const clang::Stmt *child : cur->children())
                        if (!safe(child))
                            return false;
                    return true;
                };
            return safe(stmt);
        }

        bool CrossScopeEntryBodyIsCloneSafe(
            const std::vector<SNode *> &body
        ) {
            if (body.empty()) return false;
            if (CountCloneSeq(body) > kMaxCloneStmts) return false;
            if (SeqHasBreakContinue(body) || SeqHasGoto(body)
                || SeqHasLabel(body))
                return false;

            std::function<bool(const clang::Stmt *)> clang_safe =
                [&](const clang::Stmt *stmt) -> bool {
                    if (!stmt) return true;
                    if (llvm::isa<clang::LabelStmt>(stmt)
                        || llvm::isa<clang::GotoStmt>(stmt)
                        || llvm::isa<clang::BreakStmt>(stmt)
                        || llvm::isa<clang::ContinueStmt>(stmt)
                        || llvm::isa<clang::SwitchStmt>(stmt)
                        || llvm::isa<clang::WhileStmt>(stmt)
                        || llvm::isa<clang::DoStmt>(stmt)
                        || llvm::isa<clang::ForStmt>(stmt))
                        return false;
                    for (const clang::Stmt *child : stmt->children())
                        if (!clang_safe(child))
                            return false;
                    return true;
                };

            std::function<bool(const SNode *)> snode_safe =
                [&](const SNode *node) -> bool {
                    if (!node) return true;
                    if (auto *stmt = node->dyn_cast<SStmt>())
                        return clang_safe(stmt->Stmt());
                    switch (node->Kind()) {
                        case SNodeKind::kLabel:
                        case SNodeKind::kGoto:
                        case SNodeKind::kBreak:
                        case SNodeKind::kContinue:
                        case SNodeKind::kWhile:
                        case SNodeKind::kDoWhile:
                        case SNodeKind::kFor:
                        case SNodeKind::kSwitch:
                            return false;
                        default:
                            break;
                    }
                    bool safe = true;
                    node->for_each_child([&](SNode *child) {
                        if (safe && !snode_safe(child))
                            safe = false;
                    });
                    return safe;
                };

            for (SNode *child : body)
                if (!snode_safe(child))
                    return false;
            return true;
        }

        std::string_view SingleGotoTargetSeq(const std::vector<SNode *> &body) {
            if (body.size() != 1) return {};
            return DirectGotoTarget(body.front());
        }

        std::string_view SingleGotoTargetClangStmt(clang::Stmt *stmt) {
            if (!stmt) return {};
            if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(stmt))
                if (gs->getLabel())
                    return gs->getLabel()->getName();
            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
                if (compound->body_empty())
                    return {};
                auto it = compound->body_begin();
                ++it;
                if (it != compound->body_end())
                    return {};
                return SingleGotoTargetClangStmt(compound->body_front());
            }
            return {};
        }

        clang::Expr *CrossScopeEntryGuardCond(
            SNode *node,
            std::string_view &target,
            clang::ASTContext &ctx
        ) {
            target = {};
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                if (!ite->Cond() || !ite->ElseList().empty())
                    return nullptr;
                target = SingleGotoTargetSeq(ite->ThenList());
                if (target.empty())
                    return nullptr;
                return CloneExpr(ctx, ite->Cond());
            }

            auto *stmt_node = node->dyn_cast<SStmt>();
            if (!stmt_node)
                return nullptr;
            auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(
                stmt_node->Stmt());
            if (!ifs || !ifs->getCond() || ifs->getElse())
                return nullptr;
            target = SingleGotoTargetClangStmt(ifs->getThen());
            if (target.empty())
                return nullptr;
            return CloneExpr(ctx, ifs->getCond());
        }

        bool RepairCrossScopeEntriesInSeq(
            std::vector<SNode *> &seq,
            SNodeFactory &factory,
            clang::ASTContext &ctx,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            for (size_t i = 0; i + 1 < seq.size(); ++i) {
                std::string_view target;
                clang::Expr *cond =
                    CrossScopeEntryGuardCond(seq[i], target, ctx);
                if (!cond)
                    continue;
                if (!RefCountEquals(refs, target, 1))
                    continue;

                size_t carrier_idx = seq.size();
                NestedEntryLabelLoc loc;
                if (!FindNestedEntryLabelInLaterSibling(
                        seq, i + 1, target, carrier_idx, loc))
                    continue;
                if (!loc.label || !loc.parent_seq)
                    continue;

                std::vector<SNode *> &label_body = loc.label->BodyList();
                if (!CrossScopeEntryBodyIsCloneSafe(label_body))
                    continue;

                std::vector<SNode *> cloned_body =
                    CloneSeq(label_body, factory);
                std::vector<SNode *> unwrapped_body = label_body;

                loc.parent_seq->erase(
                    loc.parent_seq->begin()
                        + static_cast<ptrdiff_t>(loc.idx));
                loc.parent_seq->insert(
                    loc.parent_seq->begin()
                        + static_cast<ptrdiff_t>(loc.idx),
                    unwrapped_body.begin(), unwrapped_body.end());

                std::vector<SNode *> else_body;
                else_body.reserve(carrier_idx - i);
                for (size_t j = i + 1; j <= carrier_idx; ++j)
                    else_body.push_back(seq[j]);

                auto *new_if = factory.Make<SIfThenElse>(
                    cond, std::move(cloned_body), std::move(else_body));

                seq.erase(seq.begin() + static_cast<ptrdiff_t>(i),
                          seq.begin() + static_cast<ptrdiff_t>(carrier_idx)
                              + 1);
                seq.insert(seq.begin() + static_cast<ptrdiff_t>(i), new_if);
                return true;
            }

            for (size_t i = 0; i + 1 < seq.size(); ++i) {
                std::string_view target;
                clang::Expr *cond =
                    CrossScopeEntryGuardCond(seq[i], target, ctx);
                if (!cond)
                    continue;
                if (!RefCountEquals(refs, target, 1))
                    continue;

                size_t carrier_idx = seq.size();
                ClangNestedEntryLabelLoc loc;
                if (!FindClangNestedEntryLabelInLaterSibling(
                        seq, i + 1, target, carrier_idx, loc))
                    continue;
                if (!loc.label || !loc.owner_if)
                    continue;

                clang::Stmt *entry_stmt = loc.label->getSubStmt();
                if (!ClangEntryStmtIsCloneSafe(entry_stmt))
                    continue;

                if (loc.is_then)
                    loc.owner_if->setThen(entry_stmt);
                else
                    loc.owner_if->setElse(entry_stmt);

                std::vector<SNode *> cloned_body;
                cloned_body.push_back(factory.Make<SStmt>(entry_stmt));

                std::vector<SNode *> else_body;
                else_body.reserve(carrier_idx - i);
                for (size_t j = i + 1; j <= carrier_idx; ++j)
                    else_body.push_back(seq[j]);

                auto *new_if = factory.Make<SIfThenElse>(
                    cond, std::move(cloned_body), std::move(else_body));

                seq.erase(seq.begin() + static_cast<ptrdiff_t>(i),
                          seq.begin() + static_cast<ptrdiff_t>(carrier_idx)
                              + 1);
                seq.insert(seq.begin() + static_cast<ptrdiff_t>(i), new_if);
                return true;
            }
            return false;
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

        bool SpliceTerminatingGotoTargetsInSeq(
            std::vector<SNode *> &seq,
            SNodeFactory &factory,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            size_t &cloned_total,
            size_t max_clone_total,
            bool descend_into_labels,
            bool require_goto_free_clone
        ) {
            bool changed = false;

            for (auto *child : seq) {
                if (!descend_into_labels && child->dyn_cast<SLabel>())
                    continue;
                ForEachBodyList(child, [&](std::vector<SNode *> &body) {
                    if (SpliceTerminatingGotoTargetsInSeq(
                            body, factory, labels, cloned_total,
                            max_clone_total, descend_into_labels,
                            require_goto_free_clone))
                        changed = true;
                });
            }

            for (size_t i = 0; i < seq.size(); ++i) {
                auto target = DirectGotoTarget(seq[i]);
                if (target.empty()) continue;

                std::vector<SNode *> clone =
                    TryCloneTerminatingLabelBody(target, labels, factory);
                if (clone.empty()) continue;
                if (IsSingleGotoTo(clone, target)) continue;
                if (require_goto_free_clone && SeqHasGoto(clone)) continue;
                if (require_goto_free_clone && SeqHasLocalLiveIns(clone))
                    continue;

                size_t clone_size = CountCloneSeq(clone);
                if (cloned_total + clone_size > max_clone_total)
                    continue;

                cloned_total += clone_size;
                SpliceAndAdvance(seq, i, clone);
                changed = true;
            }

            auto target = TrailingGotoTargetSeq(seq);
            if (target.empty()) return changed;

            std::vector<SNode *> clone =
                TryCloneTerminatingLabelBody(target, labels, factory);
            if (clone.empty()) return changed;
            if (IsSingleGotoTo(clone, target)) return changed;
            if (require_goto_free_clone && SeqHasGoto(clone)) return changed;
            if (require_goto_free_clone && SeqHasLocalLiveIns(clone))
                return changed;

            size_t clone_size = CountCloneSeq(clone);
            if (cloned_total + clone_size > max_clone_total)
                return changed;

            std::vector<SNode *> spliced =
                BuildSplicedSeq(seq, clone, factory);
            if (spliced.empty()) return changed;

            seq = std::move(spliced);
            cloned_total += clone_size;
            return true;
        }

        bool SpliceSmallEpilogueTargetsInSeq(
            std::vector<SNode *> &seq,
            SNodeFactory &factory,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            size_t &cloned_total,
            clang::ASTContext &ctx
        ) {
            bool changed = false;

            for (auto *child : seq) {
                ForEachBodyList(child, [&](std::vector<SNode *> &body) {
                    if (SpliceSmallEpilogueTargetsInSeq(
                            body, factory, labels, cloned_total, ctx))
                        changed = true;
                });
            }

            for (size_t i = 0; i < seq.size(); ++i) {
                if (auto *stmt_node = seq[i]->dyn_cast<SStmt>()) {
                    if (auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(
                            stmt_node->Stmt())) {
                        auto defs_before = GuaranteedLocalDefsBefore(seq, i);
                        auto try_replace_arm = [&](clang::Stmt *arm,
                                                   bool is_then) -> bool {
                            auto info = ExtractIfArmGoto(arm);
                            if (!info.gs || !info.gs->getLabel()) return false;

                            std::vector<SNode *> clone =
                                TryCloneSmallEpilogueTarget(
                                    info.gs->getLabel()->getName(), labels,
                                    factory);
                            if (clone.empty()) return false;

                            auto arm_defs = defs_before;
                            CollectCompoundPrefixDefs(
                                info.compound, arm_defs);
                            if (!LiveInsSatisfiedBy(clone, arm_defs))
                                return false;

                            size_t clone_size = CountCloneSeq(clone);
                            if (cloned_total + clone_size
                                > kMaxGeneralCloneTotal)
                                return false;

                            std::vector<clang::Stmt *> replacement_stmts;
                            if (!FlattenClonedSeqToStmts(
                                    clone, ctx, replacement_stmts))
                                return false;

                            clang::CompoundStmt *replacement =
                                BuildReplacementArm(
                                    ctx, info.compound, replacement_stmts);
                            if (is_then)
                                ifs->setThen(replacement);
                            else
                                ifs->setElse(replacement);
                            cloned_total += clone_size;
                            return true;
                        };

                        if (try_replace_arm(ifs->getThen(), true))
                            changed = true;
                        if (try_replace_arm(ifs->getElse(), false))
                            changed = true;
                    }
                }

                auto target = DirectGotoTarget(seq[i]);
                if (target.empty()) continue;

                std::vector<SNode *> clone =
                    TryCloneSmallEpilogueTarget(target, labels, factory);
                if (clone.empty()) continue;

                auto defined = GuaranteedLocalDefsBefore(seq, i);
                if (!LiveInsSatisfiedBy(clone, defined))
                    continue;

                size_t clone_size = CountCloneSeq(clone);
                if (cloned_total + clone_size > kMaxGeneralCloneTotal)
                    continue;

                cloned_total += clone_size;
                SpliceAndAdvance(seq, i, clone);
                changed = true;
            }

            return changed;
        }

        constexpr size_t kMaxStackGuardCloneTotal = 256;

        bool IsStackFailCall(const clang::CallExpr *call) {
            if (!call) return false;
            const clang::FunctionDecl *callee = call->getDirectCallee();
            if (!callee) return false;
            return callee->getName().contains("stack_chk_fail");
        }

        bool ClangStmtIsStackGuardCloneSafe(
            const clang::Stmt *stmt, bool &has_stack_fail
        ) {
            if (!stmt) return true;

            if (auto *label = llvm::dyn_cast<clang::LabelStmt>(stmt))
                return ClangStmtIsStackGuardCloneSafe(
                    label->getSubStmt(), has_stack_fail);

            if (auto *call = llvm::dyn_cast<clang::CallExpr>(stmt)) {
                if (!IsStackFailCall(call)) return false;
                has_stack_fail = true;
            }

            for (const clang::Stmt *child : stmt->children())
                if (!ClangStmtIsStackGuardCloneSafe(child, has_stack_fail))
                    return false;
            return true;
        }

        bool SNodeIsStackGuardCloneSafe(const SNode *node, bool &has_stack_fail);

        bool SeqIsStackGuardCloneSafe(
            const std::vector<SNode *> &seq, bool &has_stack_fail
        ) {
            for (const SNode *child : seq)
                if (!SNodeIsStackGuardCloneSafe(child, has_stack_fail))
                    return false;
            return true;
        }

        bool SNodeIsStackGuardCloneSafe(const SNode *node, bool &has_stack_fail) {
            if (!node) return true;

            if (auto *stmt = node->dyn_cast<SStmt>())
                return ClangStmtIsStackGuardCloneSafe(
                    stmt->Stmt(), has_stack_fail);
            if (auto *ret = node->dyn_cast<SReturn>())
                return ClangStmtIsStackGuardCloneSafe(
                    ret->Value(), has_stack_fail);
            if (auto *label = node->dyn_cast<SLabel>())
                return SeqIsStackGuardCloneSafe(
                    label->BodyList(), has_stack_fail);
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                if (!ClangStmtIsStackGuardCloneSafe(
                        ite->Cond(), has_stack_fail))
                    return false;
                return SeqIsStackGuardCloneSafe(
                           ite->ThenList(), has_stack_fail)
                    && SeqIsStackGuardCloneSafe(
                           ite->ElseList(), has_stack_fail);
            }

            switch (node->Kind()) {
                case SNodeKind::kGoto:
                case SNodeKind::kBreak:
                case SNodeKind::kContinue:
                case SNodeKind::kSwitch:
                case SNodeKind::kWhile:
                case SNodeKind::kDoWhile:
                case SNodeKind::kFor:
                    return false;
                default:
                    return true;
            }
        }

        bool StackGuardTailIsSafe(const std::vector<SNode *> &tail) {
            if (tail.empty()) return false;
            if (!SeqAlwaysTerminates(tail)) return false;
            if (SeqHasGoto(tail)) return false;
            if (SeqHasBreakContinue(tail)) return false;
            if (CountCloneSeq(tail) > kMaxCloneStmts) return false;

            bool has_stack_fail = false;
            return SeqIsStackGuardCloneSafe(tail, has_stack_fail)
                && has_stack_fail;
        }

        bool CollectLabelTailForStackGuardClone(
            std::string_view target_label,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            std::vector<SNode *> &tail
        ) {
            auto it = labels.find(target_label);
            if (it == labels.end()) return false;
            auto *parent_seq = it->second.parent_seq;
            if (!parent_seq) return false;

            size_t label_idx = parent_seq->size();
            for (size_t i = 0; i < parent_seq->size(); ++i) {
                if ((*parent_seq)[i] == it->second.label) {
                    label_idx = i;
                    break;
                }
            }
            if (label_idx == parent_seq->size()) return false;

            tail.clear();
            auto &label_body = it->second.label->BodyList();
            if (label_body.empty()) return false;
            tail.insert(tail.end(), label_body.begin(), label_body.end());

            for (size_t i = label_idx + 1;
                 !SeqAlwaysTerminates(tail) && i < parent_seq->size(); ++i) {
                SNode *next = (*parent_seq)[i];
                if (next->dyn_cast<SLabel>()) return false;
                tail.push_back(next);
            }

            return StackGuardTailIsSafe(tail);
        }

        SNode *CloneStackGuardNodeDroppingLabels(
            SNode *src, SNodeFactory &factory);

        bool AppendStackGuardCloneNode(
            SNode *src, SNodeFactory &factory, std::vector<SNode *> &out
        ) {
            if (!src) return true;
            if (auto *label = src->dyn_cast<SLabel>()) {
                for (SNode *child : label->BodyList())
                    if (!AppendStackGuardCloneNode(child, factory, out))
                        return false;
                return true;
            }

            SNode *clone = CloneStackGuardNodeDroppingLabels(src, factory);
            if (!clone) return false;
            out.push_back(clone);
            return true;
        }

        std::vector<SNode *> CloneStackGuardSeqDroppingLabels(
            const std::vector<SNode *> &src, SNodeFactory &factory
        ) {
            std::vector<SNode *> out;
            out.reserve(src.size());
            for (SNode *child : src)
                if (!AppendStackGuardCloneNode(child, factory, out))
                    return {};
            return out;
        }

        SNode *CloneStackGuardNodeDroppingLabels(
            SNode *src, SNodeFactory &factory
        ) {
            if (!src) return nullptr;
            if (auto *stmt = src->dyn_cast<SStmt>()) {
                clang::Stmt *body = stmt->Stmt();
                if (auto *label = llvm::dyn_cast_or_null<clang::LabelStmt>(
                        body))
                    body = label->getSubStmt();
                return body ? factory.Make<SStmt>(body) : nullptr;
            }
            if (auto *ite = src->dyn_cast<SIfThenElse>()) {
                std::vector<SNode *> then_body =
                    CloneStackGuardSeqDroppingLabels(
                        ite->ThenList(), factory);
                std::vector<SNode *> else_body =
                    CloneStackGuardSeqDroppingLabels(
                        ite->ElseList(), factory);
                return factory.Make<SIfThenElse>(
                    ite->Cond(), std::move(then_body), std::move(else_body));
            }
            if (auto *ret = src->dyn_cast<SReturn>())
                return factory.Make<SReturn>(ret->Value());
            return nullptr;
        }

        std::vector<SNode *> TryCloneStackGuardReturnTarget(
            std::string_view target_label,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            SNodeFactory &factory
        ) {
            std::vector<SNode *> tail;
            if (!CollectLabelTailForStackGuardClone(
                    target_label, labels, tail))
                return {};

            std::vector<SNode *> clone =
                CloneStackGuardSeqDroppingLabels(tail, factory);
            if (clone.empty()) return {};
            if (!SeqAlwaysTerminates(clone)) return {};
            if (SeqHasGoto(clone)) return {};
            return clone;
        }

        bool SpliceStackGuardReturnTargetsInSeq(
            std::vector<SNode *> &seq,
            SNodeFactory &factory,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            size_t &cloned_total,
            clang::ASTContext &ctx,
            const std::unordered_set<const clang::VarDecl *> &ambient_defs
        ) {
            bool changed = false;

            for (auto *child : seq) {
                ForEachBodyList(child, [&](std::vector<SNode *> &body) {
                    if (SpliceStackGuardReturnTargetsInSeq(
                            body, factory, labels, cloned_total, ctx,
                            ambient_defs))
                        changed = true;
                });
            }

            for (size_t i = 0; i < seq.size(); ++i) {
                if (auto *stmt_node = seq[i]->dyn_cast<SStmt>()) {
                    if (auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(
                            stmt_node->Stmt())) {
                        auto defs_before = GuaranteedLocalDefsBefore(seq, i);
                        defs_before.insert(
                            ambient_defs.begin(), ambient_defs.end());
                        auto try_replace_arm = [&](clang::Stmt *arm,
                                                   bool is_then) -> bool {
                            auto info = ExtractIfArmGoto(arm);
                            if (!info.gs || !info.gs->getLabel()) return false;

                            std::vector<SNode *> clone =
                                TryCloneStackGuardReturnTarget(
                                    info.gs->getLabel()->getName(), labels,
                                    factory);
                            if (clone.empty()) return false;

                            auto arm_defs = defs_before;
                            CollectCompoundPrefixDefs(
                                info.compound, arm_defs);
                            if (!LiveInsSatisfiedBy(clone, arm_defs))
                                return false;

                            size_t clone_size = CountCloneSeq(clone);
                            if (cloned_total + clone_size
                                > kMaxStackGuardCloneTotal)
                                return false;

                            std::vector<clang::Stmt *> replacement_stmts;
                            if (!FlattenClonedSeqToStmts(
                                    clone, ctx, replacement_stmts))
                                return false;

                            clang::CompoundStmt *replacement =
                                BuildReplacementArm(
                                    ctx, info.compound, replacement_stmts);
                            if (is_then)
                                ifs->setThen(replacement);
                            else
                                ifs->setElse(replacement);
                            cloned_total += clone_size;
                            return true;
                        };

                        if (try_replace_arm(ifs->getThen(), true))
                            changed = true;
                        if (try_replace_arm(ifs->getElse(), false))
                            changed = true;
                    }
                }

                auto target = DirectGotoTarget(seq[i]);
                if (target.empty()) continue;

                std::vector<SNode *> clone =
                    TryCloneStackGuardReturnTarget(target, labels, factory);
                if (clone.empty()) continue;

                auto defined = GuaranteedLocalDefsBefore(seq, i);
                defined.insert(ambient_defs.begin(), ambient_defs.end());
                if (!LiveInsSatisfiedBy(clone, defined))
                    continue;

                size_t clone_size = CountCloneSeq(clone);
                if (cloned_total + clone_size > kMaxStackGuardCloneTotal)
                    continue;

                cloned_total += clone_size;
                SpliceAndAdvance(seq, i, clone);
                changed = true;
            }

            return changed;
        }

        std::unordered_set<const clang::VarDecl *> CollectEntryPrefixDefs(
            std::vector<SNode *> &root
        ) {
            std::unordered_set<const clang::VarDecl *> defined;
            for (SNode *node : root) {
                // Once a residual label appears, later code can be reached
                // by gotos from multiple places.  Only the unlabeled entry
                // prefix is safe as an ambient fact for every label body.
                if (node->dyn_cast<SLabel>())
                    break;
                SNodeCollectGuaranteedLocalDefs(node, defined);
            }
            return defined;
        }

        bool ClangStmtIsEntrySuffixCloneSafe(const clang::Stmt *stmt) {
            if (!stmt) return true;
            if (llvm::isa<clang::LabelStmt>(stmt)
                || llvm::isa<clang::GotoStmt>(stmt)
                || llvm::isa<clang::BreakStmt>(stmt)
                || llvm::isa<clang::ContinueStmt>(stmt)
                || llvm::isa<clang::SwitchStmt>(stmt)
                || llvm::isa<clang::WhileStmt>(stmt)
                || llvm::isa<clang::DoStmt>(stmt)
                || llvm::isa<clang::ForStmt>(stmt)
                || llvm::isa<clang::DeclStmt>(stmt))
                return false;
            for (const clang::Stmt *child : stmt->children())
                if (!ClangStmtIsEntrySuffixCloneSafe(child))
                    return false;
            return true;
        }

        bool SNodeIsEntrySuffixCloneSafe(
            const SNode *node,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            if (!node) return true;
            if (auto *stmt = node->dyn_cast<SStmt>())
                return ClangStmtIsEntrySuffixCloneSafe(stmt->Stmt());
            if (auto *label = node->dyn_cast<SLabel>()) {
                if (!RefCountEquals(refs, label->Name(), 0))
                    return false;
                for (SNode *child : label->BodyList())
                    if (!SNodeIsEntrySuffixCloneSafe(child, refs))
                        return false;
                return true;
            }

            switch (node->Kind()) {
                case SNodeKind::kGoto:
                case SNodeKind::kBreak:
                case SNodeKind::kContinue:
                case SNodeKind::kSwitch:
                case SNodeKind::kWhile:
                case SNodeKind::kDoWhile:
                case SNodeKind::kFor:
                    return false;
                default:
                    break;
            }

            bool safe = true;
            node->for_each_child([&](SNode *child) {
                if (safe && !SNodeIsEntrySuffixCloneSafe(child, refs))
                    safe = false;
            });
            return safe;
        }

        bool CollectSplitEntrySuffixTail(
            std::string_view target_label,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            const std::unordered_map<std::string_view, int> &refs,
            std::vector<SNode *> &tail
        ) {
            auto it = labels.find(target_label);
            if (it == labels.end() || !it->second.parent_seq)
                return false;

            auto *parent_seq = it->second.parent_seq;
            size_t label_idx = parent_seq->size();
            for (size_t i = 0; i < parent_seq->size(); ++i) {
                if ((*parent_seq)[i] == it->second.label) {
                    label_idx = i;
                    break;
                }
            }
            if (label_idx == parent_seq->size())
                return false;

            tail.clear();
            tail.insert(tail.end(), it->second.label->BodyList().begin(),
                        it->second.label->BodyList().end());

            for (size_t i = label_idx + 1;
                 !SeqAlwaysTerminates(tail) && i < parent_seq->size(); ++i) {
                SNode *next = (*parent_seq)[i];
                if (auto *label = next->dyn_cast<SLabel>()) {
                    if (!RefCountEquals(refs, label->Name(), 0))
                        return false;
                    tail.insert(tail.end(), label->BodyList().begin(),
                                label->BodyList().end());
                    continue;
                }
                tail.push_back(next);
            }

            return !tail.empty();
        }

        bool BuildEntrySuffixSplicedSeq(
            const std::vector<SNode *> &existing,
            const std::vector<SNode *> &clone,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory,
            std::vector<SNode *> &out
        ) {
            if (existing.empty() || clone.empty())
                return false;

            SNode *last = existing.back();
            out.assign(existing.begin(), existing.end() - 1);

            auto append_clone = [&]() {
                out.insert(out.end(), clone.begin(), clone.end());
                if (NeedsTerminatorBreakSeq(clone))
                    out.push_back(factory.Make<SBreak>());
            };

            if (last->dyn_cast<SGoto>()) {
                append_clone();
                return true;
            }

            if (auto *stmt = last->dyn_cast<SStmt>()) {
                if (!llvm::isa<clang::GotoStmt>(stmt->Stmt()))
                    return false;
                append_clone();
                return true;
            }

            if (auto *label = last->dyn_cast<SLabel>()) {
                if (!RefCountEquals(refs, label->Name(), 0))
                    return false;
                std::vector<SNode *> inner;
                if (!BuildEntrySuffixSplicedSeq(
                        label->BodyList(), clone, refs, factory, inner))
                    return false;
                out.insert(out.end(), inner.begin(), inner.end());
                return true;
            }

            return false;
        }

        std::vector<SNode *> CloneEntrySuffixSeqDroppingDeadLabels(
            const std::vector<SNode *> &src,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory);

        SNode *CloneEntrySuffixNodeDroppingDeadLabels(
            SNode *src,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory
        ) {
            if (!src) return nullptr;
            if (auto *stmt = src->dyn_cast<SStmt>())
                return factory.Make<SStmt>(stmt->Stmt());
            if (auto *ite = src->dyn_cast<SIfThenElse>()) {
                std::vector<SNode *> then_body =
                    CloneEntrySuffixSeqDroppingDeadLabels(
                        ite->ThenList(), refs, factory);
                if (then_body.empty() && !ite->ThenList().empty())
                    return nullptr;
                std::vector<SNode *> else_body =
                    CloneEntrySuffixSeqDroppingDeadLabels(
                        ite->ElseList(), refs, factory);
                if (else_body.empty() && !ite->ElseList().empty())
                    return nullptr;
                return factory.Make<SIfThenElse>(
                    ite->Cond(), std::move(then_body), std::move(else_body));
            }
            if (auto *ret = src->dyn_cast<SReturn>())
                return factory.Make<SReturn>(ret->Value());
            return nullptr;
        }

        std::vector<SNode *> CloneEntrySuffixSeqDroppingDeadLabels(
            const std::vector<SNode *> &src,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory
        ) {
            std::vector<SNode *> out;
            out.reserve(src.size());
            for (SNode *child : src) {
                if (auto *label = child->dyn_cast<SLabel>()) {
                    if (!RefCountEquals(refs, label->Name(), 0))
                        return {};
                    std::vector<SNode *> body =
                        CloneEntrySuffixSeqDroppingDeadLabels(
                            label->BodyList(), refs, factory);
                    if (body.empty() && !label->BodyList().empty())
                        return {};
                    out.insert(out.end(), body.begin(), body.end());
                    continue;
                }

                SNode *clone =
                    CloneEntrySuffixNodeDroppingDeadLabels(
                        child, refs, factory);
                if (!clone) return {};
                out.push_back(clone);
            }
            return out;
        }

        std::vector<SNode *> TryCloneSplitEntrySuffix(
            std::string_view target_label,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory,
            std::unordered_set<std::string_view> &visiting
        ) {
            if (!visiting.insert(target_label).second)
                return {};
            auto fail = [&]() -> std::vector<SNode *> {
                visiting.erase(target_label);
                return {};
            };

            std::vector<SNode *> tail;
            if (!CollectSplitEntrySuffixTail(
                    target_label, labels, refs, tail))
                return fail();

            if (std::string_view next_target = TrailingGotoTargetSeq(tail);
                !next_target.empty()) {
                std::vector<SNode *> next_clone =
                    TryCloneSplitEntrySuffix(
                        next_target, labels, refs, factory, visiting);
                if (next_clone.empty())
                    return fail();

                std::vector<SNode *> spliced;
                if (!BuildEntrySuffixSplicedSeq(
                        tail, next_clone, refs, factory, spliced))
                    return fail();
                tail = std::move(spliced);
            }

            if (!SeqAlwaysTerminates(tail))
                return fail();
            if (SeqHasGoto(tail) || SeqHasBreakContinue(tail))
                return fail();
            if (CountCloneSeq(tail) > kMaxCloneStmts)
                return fail();
            for (SNode *child : tail)
                if (!SNodeIsEntrySuffixCloneSafe(child, refs))
                    return fail();

            std::vector<SNode *> clone =
                CloneEntrySuffixSeqDroppingDeadLabels(tail, refs, factory);
            visiting.erase(target_label);
            if (clone.empty()) return {};
            if (!SeqAlwaysTerminates(clone) || SeqHasLabel(clone)
                || SeqHasGoto(clone))
                return {};
            return clone;
        }

        std::vector<SNode *> TryCloneSplitEntrySuffix(
            std::string_view target_label,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory
        ) {
            std::unordered_set<std::string_view> visiting;
            return TryCloneSplitEntrySuffix(
                target_label, labels, refs, factory, visiting);
        }

        bool SpliceSplitEntrySuffixGotosInSeq(
            std::vector<SNode *> &seq,
            SNodeFactory &factory,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            const std::unordered_map<std::string_view, int> &refs,
            size_t &cloned_total,
            clang::ASTContext &ctx,
            const std::unordered_set<const clang::VarDecl *> &ambient_defs
        ) {
            bool changed = false;

            for (SNode *child : seq) {
                ForEachBodyList(child, [&](std::vector<SNode *> &body) {
                    if (SpliceSplitEntrySuffixGotosInSeq(
                            body, factory, labels, refs, cloned_total, ctx,
                            ambient_defs))
                        changed = true;
                });
            }

            for (size_t i = 0; i < seq.size(); ++i) {
                if (auto *stmt_node = seq[i]->dyn_cast<SStmt>()) {
                    if (auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(
                            stmt_node->Stmt())) {
                        auto defs_before = ambient_defs;
                        auto local_defs = GuaranteedLocalDefsBefore(seq, i);
                        defs_before.insert(local_defs.begin(), local_defs.end());

                        auto try_replace_arm = [&](clang::Stmt *arm,
                                                   bool is_then) -> bool {
                            auto info = ExtractIfArmGoto(arm);
                            if (!info.gs || !info.gs->getLabel()) return false;

                            std::vector<SNode *> clone =
                                TryCloneSplitEntrySuffix(
                                    info.gs->getLabel()->getName(), labels,
                                    refs, factory);
                            if (clone.empty()) return false;

                            auto arm_defs = defs_before;
                            CollectCompoundPrefixDefs(
                                info.compound, arm_defs);
                            if (!LiveInsSatisfiedBy(clone, arm_defs))
                                return false;

                            size_t clone_size = CountCloneSeq(clone);
                            if (cloned_total + clone_size
                                > kMaxGeneralCloneTotal)
                                return false;

                            std::vector<clang::Stmt *> replacement_stmts;
                            if (!FlattenClonedSeqToStmts(
                                    clone, ctx, replacement_stmts))
                                return false;

                            clang::CompoundStmt *replacement =
                                BuildReplacementArm(
                                    ctx, info.compound, replacement_stmts);
                            if (is_then)
                                ifs->setThen(replacement);
                            else
                                ifs->setElse(replacement);
                            cloned_total += clone_size;
                            return true;
                        };

                        if (try_replace_arm(ifs->getThen(), true))
                            changed = true;
                        if (try_replace_arm(ifs->getElse(), false))
                            changed = true;
                    }
                }

                auto target = DirectGotoTarget(seq[i]);
                if (target.empty()) continue;

                std::vector<SNode *> clone =
                    TryCloneSplitEntrySuffix(target, labels, refs, factory);
                if (clone.empty()) continue;

                auto defined = ambient_defs;
                auto local_defs = GuaranteedLocalDefsBefore(seq, i);
                defined.insert(local_defs.begin(), local_defs.end());
                if (!LiveInsSatisfiedBy(clone, defined))
                    continue;

                size_t clone_size = CountCloneSeq(clone);
                if (cloned_total + clone_size > kMaxGeneralCloneTotal)
                    continue;

                cloned_total += clone_size;
                SpliceAndAdvance(seq, i, clone);
                changed = true;
            }

            return changed;
        }

        constexpr size_t kMaxCleanupCloneTotal = 128;
        constexpr size_t kMaxCleanupCloneHops  = 6;

        bool IsCleanupCall(const clang::CallExpr *call) {
            if (!call) return false;
            const clang::FunctionDecl *callee = call->getDirectCallee();
            if (!callee) return false;

            llvm::StringRef name = callee->getName();
            return name.contains("unref")
                || name.contains("free")
                || name.contains("delete")
                || name.contains("close")
                || name.contains("dealloc")
                || name == "fwrite"
                || name == "puts"
                || name == "printf";
        }

        bool ClangStmtIsCleanupCloneSafe(
            const clang::Stmt *stmt, bool &has_cleanup_call
        ) {
            if (!stmt) return true;
            if (llvm::isa<clang::LabelStmt>(stmt)
                || llvm::isa<clang::GotoStmt>(stmt))
                return false;

            if (auto *call = llvm::dyn_cast<clang::CallExpr>(stmt)) {
                if (!IsCleanupCall(call)) return false;
                has_cleanup_call = true;
            }

            for (const clang::Stmt *child : stmt->children())
                if (!ClangStmtIsCleanupCloneSafe(child, has_cleanup_call))
                    return false;
            return true;
        }

        bool SNodeIsCleanupCloneSafe(
            const SNode *node, bool &has_cleanup_call
        ) {
            if (!node) return true;

            if (auto *stmt = node->dyn_cast<SStmt>())
                return ClangStmtIsCleanupCloneSafe(
                    stmt->Stmt(), has_cleanup_call);
            if (auto *ret = node->dyn_cast<SReturn>())
                return ClangStmtIsCleanupCloneSafe(
                    ret->Value(), has_cleanup_call);

            // Keep this pass focused on straight-line cleanup ladders.  More
            // complex control is handled by the existing non-call cloners.
            switch (node->Kind()) {
                case SNodeKind::kStmt:
                case SNodeKind::kReturn:
                    return true;
                default:
                    return false;
            }
        }

        SNode *CloneCleanupNode(SNode *src, SNodeFactory &factory) {
            if (!src) return nullptr;
            if (auto *stmt = src->dyn_cast<SStmt>())
                return factory.Make<SStmt>(stmt->Stmt());
            if (auto *ret = src->dyn_cast<SReturn>())
                return factory.Make<SReturn>(ret->Value());
            return nullptr;
        }

        bool FindLabelIndex(const LabelEntry &entry, size_t &index) {
            if (!entry.parent_seq) return false;
            auto &seq = *entry.parent_seq;
            for (size_t i = 0; i < seq.size(); ++i) {
                if (seq[i] == entry.label) {
                    index = i;
                    return true;
                }
            }
            return false;
        }

        bool AppendCleanupLabelBody(
            const LabelEntry &entry,
            SNodeFactory &factory,
            std::vector<SNode *> &clone,
            std::string_view &next_target,
            bool &has_cleanup_call
        ) {
            next_target = {};
            for (SNode *child : entry.label->BodyList()) {
                if (auto target = DirectGotoTarget(child); !target.empty()) {
                    next_target = target;
                    return true;
                }

                if (!SNodeIsCleanupCloneSafe(child, has_cleanup_call))
                    return false;
                SNode *cloned = CloneCleanupNode(child, factory);
                if (!cloned) return false;
                clone.push_back(cloned);
                if (CountCloneSeq(clone) > kMaxCloneStmts)
                    return false;
            }

            if (SeqAlwaysTerminates(clone))
                return true;

            size_t label_idx = 0;
            if (!FindLabelIndex(entry, label_idx)) return false;
            auto &parent_seq = *entry.parent_seq;
            if (label_idx + 1 >= parent_seq.size()) return false;
            auto *next_label = parent_seq[label_idx + 1]->dyn_cast<SLabel>();
            if (!next_label) return false;
            next_target = next_label->Name();
            return true;
        }

        std::vector<SNode *> TryCloneCleanupReturnTarget(
            std::string_view target_label,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            SNodeFactory &factory
        ) {
            std::vector<SNode *> clone;
            std::unordered_set<std::string_view> visited;
            std::string_view current = target_label;
            bool has_cleanup_call = false;

            for (size_t hop = 0; hop < kMaxCleanupCloneHops; ++hop) {
                if (!visited.insert(current).second) return {};
                auto it = labels.find(current);
                if (it == labels.end()) return {};

                std::string_view next_target;
                if (!AppendCleanupLabelBody(
                        it->second, factory, clone, next_target,
                        has_cleanup_call))
                    return {};

                if (SeqAlwaysTerminates(clone))
                    break;
                if (next_target.empty())
                    return {};
                current = next_target;
            }

            if (!SeqAlwaysTerminates(clone)) return {};
            if (!has_cleanup_call) return {};
            if (SeqHasGoto(clone)) return {};
            if (SeqHasBreakContinue(clone)) return {};
            if (CountCloneSeq(clone) > kMaxCloneStmts) return {};
            return clone;
        }

        bool SpliceCleanupReturnTargetsInSeq(
            std::vector<SNode *> &seq,
            SNodeFactory &factory,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            size_t &cloned_total,
            clang::ASTContext &ctx
        ) {
            bool changed = false;

            for (auto *child : seq) {
                ForEachBodyList(child, [&](std::vector<SNode *> &body) {
                    if (SpliceCleanupReturnTargetsInSeq(
                            body, factory, labels, cloned_total, ctx))
                        changed = true;
                });
            }

            for (size_t i = 0; i < seq.size(); ++i) {
                if (auto *stmt_node = seq[i]->dyn_cast<SStmt>()) {
                    if (auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(
                            stmt_node->Stmt())) {
                        auto defs_before = GuaranteedLocalDefsBefore(seq, i);
                        auto try_replace_arm = [&](clang::Stmt *arm,
                                                   bool is_then) -> bool {
                            auto info = ExtractIfArmGoto(arm);
                            if (!info.gs || !info.gs->getLabel()) return false;

                            std::vector<SNode *> clone =
                                TryCloneCleanupReturnTarget(
                                    info.gs->getLabel()->getName(), labels,
                                    factory);
                            if (clone.empty()) return false;

                            auto arm_defs = defs_before;
                            CollectCompoundPrefixDefs(
                                info.compound, arm_defs);
                            if (!LiveInsSatisfiedBy(clone, arm_defs))
                                return false;

                            size_t clone_size = CountCloneSeq(clone);
                            if (cloned_total + clone_size
                                > kMaxCleanupCloneTotal)
                                return false;

                            std::vector<clang::Stmt *> replacement_stmts;
                            if (!FlattenClonedSeqToStmts(
                                    clone, ctx, replacement_stmts))
                                return false;

                            clang::CompoundStmt *replacement =
                                BuildReplacementArm(
                                    ctx, info.compound, replacement_stmts);
                            if (is_then)
                                ifs->setThen(replacement);
                            else
                                ifs->setElse(replacement);
                            cloned_total += clone_size;
                            return true;
                        };

                        if (try_replace_arm(ifs->getThen(), true))
                            changed = true;
                        if (try_replace_arm(ifs->getElse(), false))
                            changed = true;
                    }
                }

                auto target = DirectGotoTarget(seq[i]);
                if (target.empty()) continue;

                std::vector<SNode *> clone =
                    TryCloneCleanupReturnTarget(target, labels, factory);
                if (clone.empty()) continue;

                auto defined = GuaranteedLocalDefsBefore(seq, i);
                if (!LiveInsSatisfiedBy(clone, defined))
                    continue;

                size_t clone_size = CountCloneSeq(clone);
                if (cloned_total + clone_size > kMaxCleanupCloneTotal)
                    continue;

                cloned_total += clone_size;
                SpliceAndAdvance(seq, i, clone);
                changed = true;
            }

            return changed;
        }

        struct SwitchFallthroughTarget {
            std::vector<SNode *> body;
        };

        bool OpensBreakScope(const SNode *node) {
            if (!node) return false;
            switch (node->Kind()) {
                case SNodeKind::kSwitch:
                case SNodeKind::kWhile:
                case SNodeKind::kDoWhile:
                case SNodeKind::kFor:
                    return true;
                default:
                    return false;
            }
        }

        bool IsSmallSwitchFallthroughBody(const std::vector<SNode *> &body) {
            if (body.empty()) return false;
            if (SeqAlwaysTerminates(body)) return false;
            if (SeqHasGoto(body)) return false;
            if (SeqHasBreakContinue(body)) return false;
            if (SeqHasLocalLiveIns(body)) return false;
            if (CountCloneSeq(body) > kMaxCloneStmts) return false;
            for (SNode *child : body)
                if (!SubtreeIsSafeToClone(child))
                    return false;
            return true;
        }

        void CollectSwitchFallthroughTargetsInSeq(
            std::vector<SNode *> &seq,
            std::unordered_map<std::string_view, SwitchFallthroughTarget> &targets
        ) {
            for (size_t i = 0; i < seq.size(); ++i) {
                SNode *child = seq[i];
                bool reaches_switch_break = i + 1 == seq.size();

                if (auto *label = child->dyn_cast<SLabel>()) {
                    if (reaches_switch_break
                        && IsSmallSwitchFallthroughBody(label->BodyList()))
                        targets[label->Name()] = {label->BodyList()};
                    if (reaches_switch_break)
                        CollectSwitchFallthroughTargetsInSeq(
                            label->BodyList(), targets);
                    continue;
                }

                if (!reaches_switch_break || OpensBreakScope(child))
                    continue;

                if (auto *ite = child->dyn_cast<SIfThenElse>()) {
                    CollectSwitchFallthroughTargetsInSeq(
                        ite->ThenList(), targets);
                    CollectSwitchFallthroughTargetsInSeq(
                        ite->ElseList(), targets);
                }
            }
        }

        std::vector<SNode *> CloneSwitchFallthroughTarget(
            std::string_view target,
            const std::unordered_map<std::string_view, SwitchFallthroughTarget> &targets,
            SNodeFactory &factory
        ) {
            auto it = targets.find(target);
            if (it == targets.end()) return {};
            std::vector<SNode *> clone = CloneSeq(it->second.body, factory);
            if (clone.empty()) return {};
            clone.push_back(factory.Make<SBreak>());
            return clone;
        }

        bool SpliceSwitchFallthroughTargetsInSeq(
            std::vector<SNode *> &seq,
            SNodeFactory &factory,
            const std::unordered_map<std::string_view, SwitchFallthroughTarget> &targets,
            size_t &cloned_total,
            clang::ASTContext &ctx
        ) {
            bool changed = false;

            for (SNode *child : seq) {
                if (OpensBreakScope(child))
                    continue;
                ForEachBodyList(child, [&](std::vector<SNode *> &body) {
                    if (SpliceSwitchFallthroughTargetsInSeq(
                            body, factory, targets, cloned_total, ctx))
                        changed = true;
                });
            }

            for (size_t i = 0; i < seq.size(); ++i) {
                if (auto *stmt_node = seq[i]->dyn_cast<SStmt>()) {
                    if (auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(
                            stmt_node->Stmt())) {
                        auto try_replace_arm = [&](clang::Stmt *arm,
                                                   bool is_then) -> bool {
                            auto info = ExtractIfArmGoto(arm);
                            if (!info.gs || !info.gs->getLabel()) return false;

                            std::vector<SNode *> clone =
                                CloneSwitchFallthroughTarget(
                                    info.gs->getLabel()->getName(), targets,
                                    factory);
                            if (clone.empty()) return false;

                            size_t clone_size = CountCloneSeq(clone);
                            if (cloned_total + clone_size
                                > kMaxGeneralCloneTotal)
                                return false;

                            std::vector<clang::Stmt *> replacement_stmts;
                            if (!FlattenClonedSeqToStmts(
                                    clone, ctx, replacement_stmts))
                                return false;

                            clang::CompoundStmt *replacement =
                                BuildReplacementArm(
                                    ctx, info.compound, replacement_stmts);
                            if (is_then)
                                ifs->setThen(replacement);
                            else
                                ifs->setElse(replacement);
                            cloned_total += clone_size;
                            return true;
                        };

                        if (try_replace_arm(ifs->getThen(), true))
                            changed = true;
                        if (try_replace_arm(ifs->getElse(), false))
                            changed = true;
                    }
                }

                auto target = DirectGotoTarget(seq[i]);
                if (target.empty()) continue;

                std::vector<SNode *> clone =
                    CloneSwitchFallthroughTarget(target, targets, factory);
                if (clone.empty()) continue;

                size_t clone_size = CountCloneSeq(clone);
                if (cloned_total + clone_size > kMaxGeneralCloneTotal)
                    continue;

                cloned_total += clone_size;
                SpliceAndAdvance(seq, i, clone);
                changed = true;
            }

            return changed;
        }

        bool FindLabelEntryIndex(const LabelEntry &entry, size_t &index) {
            if (!entry.parent_seq || !entry.label)
                return false;
            auto &seq = *entry.parent_seq;
            for (size_t i = 0; i < seq.size(); ++i) {
                if (seq[i] == entry.label) {
                    index = i;
                    return true;
                }
            }
            return false;
        }

        bool MoveSwitchLocalBodyIsSafe(const std::vector<SNode *> &body) {
            if (body.empty()) return false;
            if (!SeqAlwaysTerminates(body)) return false;
            if (SeqHasLabel(body)) return false;
            if (SeqHasGoto(body)) return false;
            if (SeqHasBreakContinue(body)) return false;
            if (CountCloneSeq(body) > kMaxCloneStmts) return false;

            std::function<bool(const SNode *)> safe_node =
                [&](const SNode *node) -> bool {
                    if (!node) return true;
                    if (auto *stmt = node->dyn_cast<SStmt>()) {
                        std::function<bool(const clang::Stmt *)> safe_stmt =
                            [&](const clang::Stmt *stmt) -> bool {
                                if (!stmt) return true;
                                if (llvm::isa<clang::LabelStmt>(stmt)
                                    || llvm::isa<clang::BreakStmt>(stmt)
                                    || llvm::isa<clang::ContinueStmt>(stmt)
                                    || llvm::isa<clang::SwitchStmt>(stmt)
                                    || llvm::isa<clang::WhileStmt>(stmt)
                                    || llvm::isa<clang::DoStmt>(stmt)
                                    || llvm::isa<clang::ForStmt>(stmt))
                                    return false;
                                for (const clang::Stmt *child : stmt->children())
                                    if (!safe_stmt(child))
                                        return false;
                                return true;
                            };
                        return safe_stmt(stmt->Stmt());
                    }
                    switch (node->Kind()) {
                        case SNodeKind::kLabel:
                        case SNodeKind::kBreak:
                        case SNodeKind::kContinue:
                        case SNodeKind::kWhile:
                        case SNodeKind::kDoWhile:
                        case SNodeKind::kFor:
                        case SNodeKind::kSwitch:
                            return false;
                        default:
                            break;
                    }
                    bool safe = true;
                    node->for_each_child([&](SNode *child) {
                        if (safe && !safe_node(child))
                            safe = false;
                    });
                    return safe;
                };

            for (SNode *child : body)
                if (!safe_node(child))
                    return false;
            return true;
        }

        bool PrepareSwitchLocalMoveTarget(
            std::string_view target,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            const std::unordered_map<std::string_view, int> &refs,
            LabelEntry &entry,
            std::vector<SNode *> &body
        ) {
            auto ref_it = refs.find(target);
            if (ref_it == refs.end() || ref_it->second != 1)
                return false;

            auto label_it = labels.find(target);
            if (label_it == labels.end())
                return false;

            size_t label_idx = 0;
            if (!FindLabelEntryIndex(label_it->second, label_idx))
                return false;
            if (label_idx == 0)
                return false;
            if (!SNodeAlwaysTerminates(
                    (*label_it->second.parent_seq)[label_idx - 1]))
                return false;

            body = label_it->second.label->BodyList();
            if (!MoveSwitchLocalBodyIsSafe(body))
                return false;

            entry = label_it->second;
            return true;
        }

        // Returns the index from which the label was erased, or CNode::kNone
        // if not found.  Callers iterating the same parent_seq with a stale
        // outer index must adjust when the removed index precedes theirs.
        size_t RemoveMovedSwitchLocalLabel(const LabelEntry &entry) {
            size_t label_idx = 0;
            if (!FindLabelEntryIndex(entry, label_idx))
                return CNode::kNone;
            entry.parent_seq->erase(
                entry.parent_seq->begin()
                    + static_cast<ptrdiff_t>(label_idx));
            return label_idx;
        }

        bool MoveSwitchLocalTargetsInSeq(
            std::vector<SNode *> &seq,
            clang::ASTContext &ctx,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            bool changed = false;

            for (SNode *child : seq) {
                if (OpensBreakScope(child))
                    continue;
                ForEachBodyList(child, [&](std::vector<SNode *> &body) {
                    if (MoveSwitchLocalTargetsInSeq(
                            body, ctx, labels, refs))
                        changed = true;
                });
            }

            for (size_t i = 0; i < seq.size(); ++i) {
                if (auto *stmt_node = seq[i]->dyn_cast<SStmt>()) {
                    if (auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(
                            stmt_node->Stmt())) {
                        auto try_replace_arm = [&](clang::Stmt *arm,
                                                   bool is_then) -> bool {
                            auto info = ExtractIfArmGoto(arm);
                            if (!info.gs || !info.gs->getLabel()) return false;

                            LabelEntry entry;
                            std::vector<SNode *> body;
                            if (!PrepareSwitchLocalMoveTarget(
                                    info.gs->getLabel()->getName(), labels,
                                    refs, entry, body))
                                return false;

                            std::vector<clang::Stmt *> replacement_stmts;
                            if (!FlattenClonedSeqToStmts(
                                    body, ctx, replacement_stmts))
                                return false;

                            size_t removed_idx =
                                RemoveMovedSwitchLocalLabel(entry);
                            // If the label lived in our own seq at an
                            // earlier index, decrement i so the outer
                            // loop doesn't skip past the now-shifted
                            // sibling at i+1.
                            if (entry.parent_seq == &seq
                                && removed_idx != CNode::kNone
                                && removed_idx <= i) {
                                --i;
                            }
                            clang::CompoundStmt *replacement =
                                BuildReplacementArm(
                                    ctx, info.compound, replacement_stmts);
                            if (is_then)
                                ifs->setThen(replacement);
                            else
                                ifs->setElse(replacement);
                            return true;
                        };

                        if (try_replace_arm(ifs->getThen(), true))
                            changed = true;
                        if (try_replace_arm(ifs->getElse(), false))
                            changed = true;
                    }
                }

                auto target = DirectGotoTarget(seq[i]);
                if (target.empty()) continue;

                LabelEntry entry;
                std::vector<SNode *> body;
                if (!PrepareSwitchLocalMoveTarget(
                        target, labels, refs, entry, body))
                    continue;

                size_t removed_idx = RemoveMovedSwitchLocalLabel(entry);
                // If the label was in our seq before our goto, the erase
                // shifted us — adjust i so we operate on the correct goto.
                if (entry.parent_seq == &seq
                    && removed_idx != CNode::kNone
                    && removed_idx < i) {
                    --i;
                }
                SpliceAndAdvance(seq, i, body);
                changed = true;
            }

            return changed;
        }

        bool MoveSwitchLocalTargetsInSwitch(
            SSwitch *sw,
            clang::ASTContext &ctx,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            bool changed = false;
            for (auto &c : sw->Cases()) {
                if (MoveSwitchLocalTargetsInSeq(
                        c.body_list, ctx, labels, refs))
                    changed = true;
            }
            if (MoveSwitchLocalTargetsInSeq(
                    sw->DefaultBodyList(), ctx, labels, refs))
                changed = true;
            return changed;
        }

        bool WalkAndMoveSwitchLocalTargets(
            std::vector<SNode *> &seq,
            SNodeFactory &factory,
            clang::ASTContext &ctx,
            const std::unordered_map<std::string_view, LabelEntry> &labels,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            bool changed = false;
            for (SNode *node : seq) {
                if (auto *sw = node->dyn_cast<SSwitch>())
                    if (MoveSwitchLocalTargetsInSwitch(
                            sw, ctx, labels, refs))
                        changed = true;
                ForEachBodyList(node, [&](std::vector<SNode *> &body) {
                    if (WalkAndMoveSwitchLocalTargets(
                            body, factory, ctx, labels, refs))
                        changed = true;
                });
            }
            return changed;
        }

        bool DuplicateInSwitchFallthroughTargets(
            SSwitch *sw, SNodeFactory &factory, clang::ASTContext &ctx
        ) {
            std::unordered_map<std::string_view, SwitchFallthroughTarget> targets;
            for (auto &c : sw->Cases())
                CollectSwitchFallthroughTargetsInSeq(c.body_list, targets);
            CollectSwitchFallthroughTargetsInSeq(sw->DefaultBodyList(), targets);
            if (targets.empty()) return false;

            bool changed = false;
            size_t cloned_total = 0;
            for (auto &c : sw->Cases())
                if (SpliceSwitchFallthroughTargetsInSeq(
                        c.body_list, factory, targets, cloned_total, ctx))
                    changed = true;
            if (SpliceSwitchFallthroughTargetsInSeq(
                    sw->DefaultBodyList(), factory, targets, cloned_total, ctx))
                changed = true;
            return changed;
        }

        bool WalkAndDuplicateSwitchFallthroughTargets(
            std::vector<SNode *> &seq, SNodeFactory &factory,
            clang::ASTContext &ctx
        ) {
            bool changed = false;
            for (SNode *node : seq) {
                if (auto *sw = node->dyn_cast<SSwitch>())
                    if (DuplicateInSwitchFallthroughTargets(sw, factory, ctx))
                        changed = true;
                ForEachBodyList(node, [&](std::vector<SNode *> &body) {
                    if (WalkAndDuplicateSwitchFallthroughTargets(
                            body, factory, ctx))
                        changed = true;
                });
            }
            return changed;
        }

        struct LoopContinueTarget {
            std::vector<SNode *> body;
            bool require_live_ins = false;
        };

        bool OpensContinueScope(const SNode *node) {
            if (!node) return false;
            switch (node->Kind()) {
                case SNodeKind::kWhile:
                case SNodeKind::kDoWhile:
                case SNodeKind::kFor:
                    return true;
                default:
                    return false;
            }
        }

        bool ClangStmtEndsInContinue(const clang::Stmt *stmt) {
            if (!stmt) return false;
            if (llvm::isa<clang::ContinueStmt>(stmt)) return true;
            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
                if (compound->body_empty()) return false;
                return ClangStmtEndsInContinue(compound->body_back());
            }
            if (auto *label = llvm::dyn_cast<clang::LabelStmt>(stmt))
                return ClangStmtEndsInContinue(label->getSubStmt());
            if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(stmt)) {
                if (!ifs->getThen() || !ifs->getElse()) return false;
                return ClangStmtEndsInContinue(ifs->getThen())
                    && ClangStmtEndsInContinue(ifs->getElse());
            }
            return false;
        }

        bool SNodeEndsInContinue(const SNode *node);

        bool SeqEndsInContinue(const std::vector<SNode *> &seq) {
            if (seq.empty()) return false;
            return SNodeEndsInContinue(seq.back());
        }

        bool SNodeEndsInContinue(const SNode *node) {
            if (!node) return false;
            if (node->dyn_cast<SContinue>()) return true;
            if (auto *st = node->dyn_cast<SStmt>())
                return ClangStmtEndsInContinue(st->Stmt());
            if (auto *label = node->dyn_cast<SLabel>())
                return SeqEndsInContinue(label->BodyList());
            if (auto *ite = node->dyn_cast<SIfThenElse>())
                return SeqEndsInContinue(ite->ThenList())
                    && SeqEndsInContinue(ite->ElseList());
            return false;
        }

        bool SubtreeHasBreak(const SNode *node) {
            if (!node) return false;
            if (node->dyn_cast<SBreak>()) return true;
            bool found = false;
            node->for_each_child([&](SNode *child) {
                if (!found && SubtreeHasBreak(child))
                    found = true;
            });
            return found;
        }

        bool SeqHasBreak(const std::vector<SNode *> &seq) {
            for (SNode *child : seq)
                if (SubtreeHasBreak(child))
                    return true;
            return false;
        }

        bool IsSmallLoopContinueBodyAllowCalls(
            const std::vector<SNode *> &body,
            const std::unordered_map<std::string_view, int> &refs
        );

        std::vector<SNode *> BuildLoopContinueTail(
            std::vector<SNode *> &seq, size_t label_index,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            auto *label = seq[label_index]->dyn_cast<SLabel>();
            if (!label) return {};

            std::vector<SNode *> tail;
            tail.insert(
                tail.end(), label->BodyList().begin(), label->BodyList().end());

            for (size_t i = label_index + 1;
                 !SeqAlwaysTerminates(tail) && i < seq.size(); ++i) {
                SNode *next = seq[i];
                if (next->dyn_cast<SLabel>() || OpensContinueScope(next))
                    return {};
                tail.push_back(next);
            }

            if (!IsSmallLoopContinueBodyAllowCalls(tail, refs))
                return {};
            return tail;
        }

        bool IsSmallLoopContinueBodyAllowCalls(
            const std::vector<SNode *> &body,
            const std::unordered_map<std::string_view, int> &refs
        ) {
            if (body.empty()) return false;
            if (!SeqEndsInContinue(body)) return false;
            if (!SeqAlwaysTerminates(body)) return false;
            if (SeqHasGoto(body)) return false;
            if (SeqHasBreak(body)) return false;
            if (CountCloneSeq(body) > kMaxCloneStmts) return false;

            for (size_t i = 0; i < body.size(); ++i) {
                if (i + 1 == body.size() && body[i]->dyn_cast<SContinue>())
                    continue;
                if (!SNodeIsEntrySuffixCloneSafe(body[i], refs))
                    return false;
            }
            return true;
        }

        std::vector<SNode *> BuildLoopEndContinueTail(
            std::vector<SNode *> &seq, size_t label_index,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory
        ) {
            auto *label = seq[label_index]->dyn_cast<SLabel>();
            if (!label) return {};

            std::vector<SNode *> raw_tail;
            raw_tail.insert(
                raw_tail.end(), label->BodyList().begin(),
                label->BodyList().end());
            for (size_t i = label_index + 1; i < seq.size(); ++i)
                raw_tail.push_back(seq[i]);
            if (raw_tail.empty()) return {};

            std::vector<SNode *> tail;
            tail = CloneEntrySuffixSeqDroppingDeadLabels(
                raw_tail, refs, factory);
            if (tail.empty())
                return {};

            tail.push_back(factory.Make<SContinue>());
            if (!IsSmallLoopContinueBodyAllowCalls(tail, refs))
                return {};
            return tail;
        }

        void CollectLoopContinueTargetsInSeq(
            std::vector<SNode *> &seq,
            std::unordered_map<std::string_view, LoopContinueTarget> &targets,
            const std::unordered_map<std::string_view, int> &refs,
            SNodeFactory &factory,
            bool is_loop_body_root = false
        ) {
            for (size_t i = 0; i < seq.size(); ++i) {
                SNode *child = seq[i];
                if (auto *label = child->dyn_cast<SLabel>()) {
                    std::vector<SNode *> tail =
                        BuildLoopContinueTail(seq, i, refs);
                    if (!tail.empty()) {
                        targets[label->Name()] = {std::move(tail), false};
                    } else if (is_loop_body_root) {
                        tail = BuildLoopEndContinueTail(
                            seq, i, refs, factory);
                        if (!tail.empty())
                            targets[label->Name()] = {std::move(tail), true};
                    }
                }

                if (OpensContinueScope(child))
                    continue;
                ForEachBodyList(child, [&](std::vector<SNode *> &body) {
                    CollectLoopContinueTargetsInSeq(
                        body, targets, refs, factory);
                });
            }
        }

        std::vector<SNode *> CloneLoopContinueTarget(
            std::string_view target,
            const std::unordered_map<std::string_view, LoopContinueTarget> &targets,
            SNodeFactory &factory
        ) {
            auto it = targets.find(target);
            if (it == targets.end()) return {};
            return CloneSeq(it->second.body, factory);
        }

        bool SpliceLoopContinueTargetsInSeq(
            std::vector<SNode *> &seq,
            SNodeFactory &factory,
            const std::unordered_map<std::string_view, LoopContinueTarget> &targets,
            size_t &cloned_total,
            clang::ASTContext &ctx
        ) {
            bool changed = false;

            for (SNode *child : seq) {
                if (OpensContinueScope(child))
                    continue;
                ForEachBodyList(child, [&](std::vector<SNode *> &body) {
                    if (SpliceLoopContinueTargetsInSeq(
                            body, factory, targets, cloned_total, ctx))
                        changed = true;
                });
            }

            for (size_t i = 0; i < seq.size(); ++i) {
                if (auto *stmt_node = seq[i]->dyn_cast<SStmt>()) {
                        if (auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(
                                stmt_node->Stmt())) {
                        auto defs_before = GuaranteedLocalDefsBefore(seq, i);
                        auto try_replace_arm = [&](clang::Stmt *arm,
                                                   bool is_then) -> bool {
                            auto info = ExtractIfArmGoto(arm);
                            if (!info.gs || !info.gs->getLabel()) return false;

                            auto target_it =
                                targets.find(info.gs->getLabel()->getName());
                            if (target_it == targets.end()) return false;

                            std::vector<SNode *> clone =
                                CloneLoopContinueTarget(
                                    info.gs->getLabel()->getName(), targets,
                                    factory);
                            if (clone.empty()) return false;

                            if (target_it->second.require_live_ins) {
                                auto arm_defs = defs_before;
                                CollectCompoundPrefixDefs(
                                    info.compound, arm_defs);
                                if (!LiveInsSatisfiedBy(clone, arm_defs))
                                    return false;
                            }

                            size_t clone_size = CountCloneSeq(clone);
                            if (cloned_total + clone_size
                                > kMaxGeneralCloneTotal)
                                return false;

                            std::vector<clang::Stmt *> replacement_stmts;
                            if (!FlattenClonedSeqToStmts(
                                    clone, ctx, replacement_stmts))
                                return false;

                            clang::CompoundStmt *replacement =
                                BuildReplacementArm(
                                    ctx, info.compound, replacement_stmts);
                            if (is_then)
                                ifs->setThen(replacement);
                            else
                                ifs->setElse(replacement);
                            cloned_total += clone_size;
                            return true;
                        };

                        if (try_replace_arm(ifs->getThen(), true))
                            changed = true;
                        if (try_replace_arm(ifs->getElse(), false))
                            changed = true;
                    }
                }

                auto target = DirectGotoTarget(seq[i]);
                if (target.empty()) continue;

                auto target_it = targets.find(target);
                if (target_it == targets.end()) continue;

                std::vector<SNode *> clone =
                    CloneLoopContinueTarget(target, targets, factory);
                if (clone.empty()) continue;

                if (target_it->second.require_live_ins) {
                    auto defined = GuaranteedLocalDefsBefore(seq, i);
                    if (!LiveInsSatisfiedBy(clone, defined))
                        continue;
                }

                size_t clone_size = CountCloneSeq(clone);
                if (cloned_total + clone_size > kMaxGeneralCloneTotal)
                    continue;

                cloned_total += clone_size;
                SpliceAndAdvance(seq, i, clone);
                changed = true;
            }

            return changed;
        }

        bool DuplicateInLoopContinueTargets(
            std::vector<SNode *> &body, SNodeFactory &factory,
            clang::ASTContext &ctx
        ) {
            std::unordered_map<std::string_view, int> refs;
            CountGotoRefs(body, refs);
            std::unordered_map<std::string_view, LoopContinueTarget> targets;
            CollectLoopContinueTargetsInSeq(
                body, targets, refs, factory, /*is_loop_body_root=*/true);
            if (targets.empty()) return false;

            size_t cloned_total = 0;
            return SpliceLoopContinueTargetsInSeq(
                body, factory, targets, cloned_total, ctx);
        }

        bool DuplicateInLoopContinueTargets(
            SNode *node, SNodeFactory &factory, clang::ASTContext &ctx
        ) {
            if (auto *w = node->dyn_cast<SWhile>())
                return DuplicateInLoopContinueTargets(
                    w->BodyList(), factory, ctx);
            if (auto *dw = node->dyn_cast<SDoWhile>())
                return DuplicateInLoopContinueTargets(
                    dw->BodyList(), factory, ctx);
            if (auto *f = node->dyn_cast<SFor>())
                return DuplicateInLoopContinueTargets(
                    f->BodyList(), factory, ctx);
            return false;
        }

        bool WalkAndDuplicateLoopContinueTargets(
            std::vector<SNode *> &seq, SNodeFactory &factory,
            clang::ASTContext &ctx
        ) {
            bool changed = false;
            for (SNode *node : seq) {
                if (OpensContinueScope(node))
                    if (DuplicateInLoopContinueTargets(node, factory, ctx))
                        changed = true;
                ForEachBodyList(node, [&](std::vector<SNode *> &body) {
                    if (WalkAndDuplicateLoopContinueTargets(
                            body, factory, ctx))
                        changed = true;
                });
            }
            return changed;
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
                if (SpliceTerminatingGotoTargetsInSeq(
                        c.body_list, factory, labels, cloned_total,
                        kMaxSwitchCloneTotal,
                        /*descend_into_labels=*/false,
                        /*require_goto_free_clone=*/false))
                    changed = true;
                continue;
            }
            for (auto &c : sw->Cases()) {
                if (SpliceTerminatingGotoTargetsInSeq(
                        c.body_list, factory, labels, cloned_total,
                        kMaxSwitchCloneTotal,
                        /*descend_into_labels=*/false,
                        /*require_goto_free_clone=*/false))
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
                if (SpliceTerminatingGotoTargetsInSeq(
                        sw->DefaultBodyList(), factory, labels, cloned_total,
                        kMaxSwitchCloneTotal,
                        /*descend_into_labels=*/false,
                        /*require_goto_free_clone=*/false))
                    changed = true;
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
        void CollectClangLabelNames(
            clang::Stmt *stmt,
            std::unordered_set<std::string_view> &labels
        ) {
            if (!stmt) return;
            if (auto *label = llvm::dyn_cast<clang::LabelStmt>(stmt))
                if (label->getDecl())
                    labels.insert(label->getDecl()->getName());
            for (clang::Stmt *child : stmt->children())
                CollectClangLabelNames(child, labels);
        }

        void CollectAllLabelNames(
            std::vector<SNode *> &seq,
            std::unordered_set<std::string_view> &labels
        ) {
            for (SNode *node : seq) {
                if (auto *label = node->dyn_cast<SLabel>())
                    labels.insert(label->Name());
                if (auto *stmt = node->dyn_cast<SStmt>())
                    CollectClangLabelNames(stmt->Stmt(), labels);
                ForEachBodyList(node, [&](std::vector<SNode *> &body) {
                    CollectAllLabelNames(body, labels);
                });
            }
        }

        void VerifyGotoLabelPairing(std::vector<SNode *> &root,
                                    const char *caller_tag) {
#ifndef NDEBUG
            std::unordered_set<std::string_view> labels;
            CollectAllLabelNames(root, labels);
            std::unordered_map<std::string_view, int> refs;
            CountGotoRefs(root, refs);
            for (auto &[name, _] : refs) {
                if (!labels.contains(name)) {
                    LOG(ERROR) << caller_tag << ": dangling "
                               << "goto target '" << std::string(name)
                               << "' after duplication\n";
                    assert(false && "dangling goto target after duplication");
                }
            }
#else
            (void)root;
            (void)caller_tag;
#endif
        }

    } // anonymous namespace

    bool FoldGuardedFallthroughTargets(std::vector<SNode *> &root,
                                       SNodeFactory & /*factory*/,
                                       clang::ASTContext &ctx) {
        bool any_changed = false;
        for (int pass = 0; pass < 4; ++pass) {
            std::unordered_map<std::string_view, int> refs;
            CountGotoRefs(root, refs);
            bool did = ForEachSeqPostOrder(
                root, [&](std::vector<SNode *> &seq) {
                    return FoldGuardedFallthroughInSeq(seq, refs, ctx);
                });
            if (!did)
                break;
            any_changed = true;
        }
        if (any_changed) VerifyGotoLabelPairing(root, "FoldGuardedFallthroughTargets");
        return any_changed;
    }

    bool RepairCrossScopeLabelEntries(std::vector<SNode *> &root,
                                      SNodeFactory &factory,
                                      clang::ASTContext &ctx) {
        bool any_changed = false;
        for (int pass = 0; pass < 4; ++pass) {
            std::unordered_map<std::string_view, int> refs;
            CountGotoRefs(root, refs);
            bool did = ForEachSeqPostOrder(
                root, [&](std::vector<SNode *> &seq) {
                    return RepairCrossScopeEntriesInSeq(
                        seq, factory, ctx, refs);
                });
            if (!did)
                break;
            any_changed = true;
        }
        if (any_changed) VerifyGotoLabelPairing(root, "RepairCrossScopeLabelEntries");
        return any_changed;
    }

    bool FoldSiblingArmLabelEntries(std::vector<SNode *> &root,
                                    SNodeFactory &factory,
                                    clang::ASTContext &ctx) {
        bool any_changed = false;
        for (int pass = 0; pass < 4; ++pass) {
            std::unordered_map<std::string_view, int> refs;
            CountGotoRefs(root, refs);
            bool did = ForEachSeqPostOrder(
                root,
                [&](std::vector<SNode *> &seq) {
                    return FoldSiblingArmEntriesInSeq(seq, refs, factory, ctx);
                },
                /*stop_after_change=*/true);
            if (!did)
                break;
            any_changed = true;
        }
        if (any_changed)
            VerifyGotoLabelPairing(root, "FoldSiblingArmLabelEntries");
        return any_changed;
    }

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
        if (any_changed) VerifyGotoLabelPairing(root, "DuplicateSwitchCaseTargets");
        return any_changed;
    }

    bool FoldSwitchLocalCaseTargets(std::vector<SNode *> &root,
                                    SNodeFactory &factory,
                                    clang::ASTContext &ctx) {
        bool any_changed = false;
        for (int pass = 0; pass < 4; ++pass) {
            std::unordered_map<std::string_view, LabelEntry> labels;
            CollectLabels(root, labels);
            std::unordered_map<std::string_view, int> refs;
            CountGotoRefs(root, refs);
            bool did = WalkAndMoveSwitchLocalTargets(
                root, factory, ctx, labels, refs);
            if (!did)
                break;
            any_changed = true;
        }
        if (any_changed) VerifyGotoLabelPairing(root, "FoldSwitchLocalCaseTargets");
        return any_changed;
    }

    bool DuplicateSmallTerminatingTargets(std::vector<SNode *> &root,
                                          SNodeFactory &factory) {
        bool any_changed = false;

        // Re-scan labels after each pass: cloning one target can expose a
        // trailing goto to another small terminating target.
        for (int pass = 0; pass < 4; ++pass) {
            std::unordered_map<std::string_view, LabelEntry> labels;
            CollectLabels(root, labels);
            size_t cloned_total = 0;
            bool did = SpliceTerminatingGotoTargetsInSeq(
                root, factory, labels, cloned_total, kMaxGeneralCloneTotal,
                /*descend_into_labels=*/true,
                /*require_goto_free_clone=*/true);
            if (!did)
                break;
            any_changed = true;
        }

        if (any_changed) VerifyGotoLabelPairing(root, "DuplicateSmallTerminatingTargets");
        return any_changed;
    }

    bool SplitAndCloneCrossScopeEntries(std::vector<SNode *> &root,
                                        SNodeFactory &factory,
                                        clang::ASTContext &ctx) {
        bool any_changed = false;

        for (int pass = 0; pass < 4; ++pass) {
            std::unordered_map<std::string_view, LabelEntry> labels;
            CollectLabels(root, labels);
            std::unordered_map<std::string_view, int> refs;
            CountGotoRefs(root, refs);
            std::unordered_set<const clang::VarDecl *> ambient_defs =
                CollectEntryPrefixDefs(root);
            size_t cloned_total = 0;
            bool did = SpliceSplitEntrySuffixGotosInSeq(
                root, factory, labels, refs, cloned_total, ctx,
                ambient_defs);
            if (!did)
                break;
            any_changed = true;
        }

        if (any_changed)
            VerifyGotoLabelPairing(root, "SplitAndCloneCrossScopeEntries");
        return any_changed;
    }

    bool DuplicateSmallEpilogueTargets(std::vector<SNode *> &root,
                                       SNodeFactory &factory,
                                       clang::ASTContext &ctx) {
        bool any_changed = false;

        for (int pass = 0; pass < 4; ++pass) {
            std::unordered_map<std::string_view, LabelEntry> labels;
            CollectLabels(root, labels);
            size_t cloned_total = 0;
            bool did = SpliceSmallEpilogueTargetsInSeq(
                root, factory, labels, cloned_total, ctx);
            if (!did)
                break;
            any_changed = true;
        }

        if (any_changed) VerifyGotoLabelPairing(root, "DuplicateSmallEpilogueTargets");
        return any_changed;
    }

    bool DuplicateSwitchFallthroughTargets(std::vector<SNode *> &root,
                                           SNodeFactory &factory,
                                           clang::ASTContext &ctx) {
        bool any_changed = false;

        for (int pass = 0; pass < 4; ++pass) {
            bool did =
                WalkAndDuplicateSwitchFallthroughTargets(root, factory, ctx);
            if (!did)
                break;
            any_changed = true;
        }

        if (any_changed) VerifyGotoLabelPairing(root, "DuplicateSwitchFallthroughTargets");
        return any_changed;
    }

    bool DuplicateLoopContinueTargets(std::vector<SNode *> &root,
                                      SNodeFactory &factory,
                                      clang::ASTContext &ctx) {
        bool any_changed = false;

        for (int pass = 0; pass < 4; ++pass) {
            bool did =
                WalkAndDuplicateLoopContinueTargets(root, factory, ctx);
            if (!did)
                break;
            any_changed = true;
        }

        if (any_changed) VerifyGotoLabelPairing(root, "DuplicateLoopContinueTargets");
        return any_changed;
    }

    bool DuplicateStackGuardReturnTargets(std::vector<SNode *> &root,
                                          SNodeFactory &factory,
                                          clang::ASTContext &ctx) {
        bool any_changed = false;

        for (int pass = 0; pass < 4; ++pass) {
            std::unordered_map<std::string_view, LabelEntry> labels;
            CollectLabels(root, labels);
            std::unordered_set<const clang::VarDecl *> ambient_defs =
                CollectEntryPrefixDefs(root);
            size_t cloned_total = 0;
            bool did = SpliceStackGuardReturnTargetsInSeq(
                root, factory, labels, cloned_total, ctx, ambient_defs);
            if (!did)
                break;
            any_changed = true;
        }

        if (any_changed) VerifyGotoLabelPairing(root, "DuplicateStackGuardReturnTargets");
        return any_changed;
    }

    bool DuplicateCleanupReturnTargets(std::vector<SNode *> &root,
                                       SNodeFactory &factory,
                                       clang::ASTContext &ctx) {
        bool any_changed = false;

        for (int pass = 0; pass < 4; ++pass) {
            std::unordered_map<std::string_view, LabelEntry> labels;
            CollectLabels(root, labels);
            size_t cloned_total = 0;
            bool did = SpliceCleanupReturnTargetsInSeq(
                root, factory, labels, cloned_total, ctx);
            if (!did)
                break;
            any_changed = true;
        }

        if (any_changed) VerifyGotoLabelPairing(root, "DuplicateCleanupReturnTargets");
        return any_changed;
    }

    // Lift raw clang goto/label-bearing control flow out of opaque SStmt
    // leaves so later SNode passes can repair it. Unsupported statements
    // remain opaque, and goto/label names are preserved verbatim.

    namespace {

        // Used to decide whether a CompoundStmt should be decomposed.
        bool ClangStmtContainsControlFlow(clang::Stmt *stmt) {
            if (!stmt) { return false; }
            if (llvm::isa< clang::GotoStmt >(stmt)
                || llvm::isa< clang::LabelStmt >(stmt))
            {
                return true;
            }
            for (clang::Stmt *child : stmt->children()) {
                if (ClangStmtContainsControlFlow(child)) { return true; }
            }
            return false;
        }

        // Switch case bodies recurse through the generic normalizer.
        std::vector< SNode * > NormalizeClangStmt(clang::Stmt *stmt,
                                                  SNodeFactory &factory);

        // Lift switches whose body is a clean CaseStmt/DefaultStmt list.
        // Case bodies are normalized recursively; unsupported shapes stay
        // opaque by returning nullptr.
        SSwitch *TryLiftSwitch(clang::SwitchStmt *sws, SNodeFactory &factory) {
            clang::Stmt *body = sws->getBody();
            auto *compound = llvm::dyn_cast_or_null< clang::CompoundStmt >(body);
            if (!compound) {
                return nullptr;
            }
            // Reuse the discriminant; clang AST nodes are arena-owned.
            auto *sw = factory.Make< SSwitch >(sws->getCond());
            bool saw_default = false;
            for (clang::Stmt *child : compound->body()) {
                if (!child || llvm::isa< clang::NullStmt >(child)) {
                    continue;
                }
                auto *swc = llvm::dyn_cast< clang::SwitchCase >(child);
                if (!swc) {
                    // Non-case child: leave the switch opaque.
                    return nullptr;
                }
                // Walk a fallthrough chain: collect leading case values
                // (CaseStmt nesting CaseStmt) until the terminal body.
                std::vector< clang::Expr * > chain_values;
                bool chain_has_default = false;
                clang::SwitchCase *cur = swc;
                clang::Stmt *terminal  = nullptr;
                for (;;) {
                    if (auto *cs = llvm::dyn_cast< clang::CaseStmt >(cur)) {
                        // Reuse the case value expression as-is.
                        chain_values.push_back(cs->getLHS());
                        clang::Stmt *next = cs->getSubStmt();
                        if (auto *nested =
                                llvm::dyn_cast_or_null< clang::SwitchCase >(next))
                        {
                            cur = nested;
                            continue;
                        }
                        terminal = next;
                    } else {
                        // DefaultStmt — terminates the chain.
                        chain_has_default = true;
                        terminal = llvm::cast< clang::DefaultStmt >(cur)
                                       ->getSubStmt();
                        if (auto *nested =
                                llvm::dyn_cast_or_null< clang::SwitchCase >(
                                    terminal))
                        {
                            cur = nested;
                            continue;
                        }
                    }
                    break;
                }
                // Give each case value its own normalized body.
                for (clang::Expr *val : chain_values) {
                    sw->AddCase(val, NormalizeClangStmt(terminal, factory));
                }
                if (chain_has_default) {
                    if (saw_default) {
                        // Two default arms — malformed; bail out.
                        return nullptr;
                    }
                    saw_default = true;
                    sw->SetDefaultBody(NormalizeClangStmt(terminal, factory));
                }
            }
            return sw;
        }

        // Normalize one clang statement into zero or more SNodes. Compounds
        // are split only when they contain goto/label control flow; other
        // unsupported shapes stay opaque.
        std::vector< SNode * > NormalizeClangStmt(clang::Stmt *stmt,
                                                  SNodeFactory &factory) {
            if (!stmt || llvm::isa< clang::NullStmt >(stmt)) {
                return {};
            }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                assert(gs->getLabel()
                       && "clang::GotoStmt missing target label");
                return { factory.Make< SGoto >(
                    factory.Intern(gs->getLabel()->getName())) };
            }
            if (llvm::isa< clang::BreakStmt >(stmt)) {
                return { factory.Make< SBreak >() };
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                return { factory.Make< SLabel >(
                    factory.Intern(ls->getDecl()->getName()),
                    NormalizeClangStmt(ls->getSubStmt(), factory)) };
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                // Preserve arm order and normalize each arm recursively.
                return { factory.Make< SIfThenElse >(
                    ifs->getCond(),
                    NormalizeClangStmt(ifs->getThen(), factory),
                    NormalizeClangStmt(ifs->getElse(), factory)) };
            }
            if (auto *sws = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                if (auto *sw = TryLiftSwitch(sws, factory)) {
                    return { sw };
                }
                // Not the clean switch shape; leave it opaque.
                return { factory.Make< SStmt >(stmt) };
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                // Keep plain compounds opaque to preserve their brace level.
                if (!ClangStmtContainsControlFlow(cs)) {
                    return { factory.Make< SStmt >(cs) };
                }
                std::vector< SNode * > out;
                for (clang::Stmt *child : cs->body()) {
                    std::vector< SNode * > child_nodes =
                        NormalizeClangStmt(child, factory);
                    out.insert(out.end(), child_nodes.begin(),
                               child_nodes.end());
                }
                return out;
            }
            // Any other clang stmt shape stays opaque.
            return { factory.Make< SStmt >(stmt) };
        }

    } // anonymous namespace (NormalizeRawControlFlow helpers)

    void NormalizeRawControlFlow(std::vector< SNode * > &root,
                                 SNodeFactory &factory,
                                 clang::ASTContext & /*ctx*/) {
        // Rebuild each body vector so one SStmt can expand into many SNodes.
        auto worker = [&](std::vector< SNode * > &seq) -> bool {
            bool changed = false;
            std::vector< SNode * > rebuilt;
            rebuilt.reserve(seq.size());
            for (auto *slot : seq) {
                auto *st = slot ? slot->dyn_cast< SStmt >() : nullptr;
                clang::Stmt *s = st ? st->Stmt() : nullptr;
                // Plain compounds and unsupported shapes stay opaque.
                bool is_candidate = s != nullptr
                    && (llvm::isa< clang::GotoStmt >(s)
                        || llvm::isa< clang::LabelStmt >(s)
                        || llvm::isa< clang::IfStmt >(s)
                        || llvm::isa< clang::SwitchStmt >(s)
                        || (llvm::isa< clang::CompoundStmt >(s)
                            && ClangStmtContainsControlFlow(s)));
                if (!is_candidate) {
                    rebuilt.push_back(slot);
                    continue;
                }
                std::vector< SNode * > lifted =
                    NormalizeClangStmt(s, factory);
                // Keep no-op lifts from marking the pass changed.
                if (lifted.size() == 1) {
                    if (auto *only = lifted.front()->dyn_cast< SStmt >()) {
                        if (only->Stmt() == s) {
                            rebuilt.push_back(slot);
                            continue;
                        }
                    }
                }
                rebuilt.insert(rebuilt.end(), lifted.begin(), lifted.end());
                changed = true;
            }
            if (changed) {
                seq.swap(rebuilt);
            }
            return changed;
        };

        // Repeat until every newly exposed body vector has been visited.
        while (ForEachSeqPostOrder(root, worker, /*stop_after_change=*/true)) {
        }
    }

    bool FinalizeRegionRepairs(std::vector< SNode * > &root,
                               SNodeFactory &factory,
                               clang::ASTContext &ctx) {
        bool any_changed = false;
        for (int pass = 0; pass < kMaxGotoEliminationPasses; ++pass) {
            bool changed = false;

            changed |= ConvertGotoToBreakContinue(root, factory);
            changed |= ConvertGotoToReturn(root, factory, ctx);
            changed |= DuplicateSwitchCaseTargets(root, factory);
            changed |= FoldSwitchLocalCaseTargets(root, factory, ctx);
            changed |= DuplicateSmallTerminatingTargets(root, factory);
            changed |= SplitAndCloneCrossScopeEntries(root, factory, ctx);
            changed |= DuplicateSmallEpilogueTargets(root, factory, ctx);
            changed |= DuplicateSwitchFallthroughTargets(root, factory, ctx);
            changed |= DuplicateLoopContinueTargets(root, factory, ctx);
            changed |= FoldGuardedFallthroughTargets(root, factory, ctx);
            changed |= RepairCrossScopeLabelEntries(root, factory, ctx);
            changed |= FoldSiblingArmLabelEntries(root, factory, ctx);
            changed |= DuplicateStackGuardReturnTargets(root, factory, ctx);
            changed |= DuplicateCleanupReturnTargets(root, factory, ctx);
            changed |= SimplifyEmptyControlFlow(root, ctx);
            changed |= CollapsePassThroughLabels(root, factory);
            changed |= MergeRedundantGotoGuards(root, factory, ctx);
            changed |= InlineResidualGotos(root, factory);
            changed |= InlineCrossScopeSingleRef(root, factory);
            changed |= AbsorbFallthroughIntoElse(root, factory);
            changed |= ScopeifyIfGotos(root, factory, ctx);
            changed |= AbsorbCrossScopeIfGoto(root, factory, ctx);
            changed |= EliminateGotoToNextLabel(root, factory, ctx);
            changed |= RemoveDeadSSeqChildren(root);

            if (!changed) {
                break;
            }
            any_changed = true;
        }
        return any_changed;
    }

    namespace {

        using RegionScopePath = std::vector< unsigned >;

        struct ScopedLabelDef {
            RegionScopePath scope;
        };

        struct ScopedGotoRef {
            std::string target;
            RegionScopePath scope;
        };

        bool IsScopePrefix(const RegionScopePath &prefix,
                           const RegionScopePath &scope) {
            if (prefix.size() > scope.size()) {
                return false;
            }
            return std::equal(prefix.begin(), prefix.end(), scope.begin());
        }

        std::string FormatScopePath(const RegionScopePath &scope) {
            if (scope.empty()) {
                return "<root>";
            }
            std::string out;
            for (unsigned id : scope) {
                if (!out.empty()) {
                    out += ".";
                }
                out += std::to_string(id);
            }
            return out;
        }

        void CollectClangRegionRefs(
            clang::Stmt *stmt, const RegionScopePath &scope,
            std::unordered_map< std::string, ScopedLabelDef > &labels,
            std::vector< ScopedGotoRef > &gotos
        ) {
            if (!stmt) {
                return;
            }
            if (auto *go = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                if (go->getLabel()) {
                    gotos.push_back({ go->getLabel()->getName().str(), scope });
                }
                return;
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                if (label->getDecl()) {
                    labels.emplace(
                        label->getDecl()->getName().str(),
                        ScopedLabelDef{ scope });
                }
                CollectClangRegionRefs(label->getSubStmt(), scope, labels, gotos);
                return;
            }
            for (clang::Stmt *child : stmt->children()) {
                CollectClangRegionRefs(child, scope, labels, gotos);
            }
        }

        void CollectRegionRefsSeq(
            const std::vector< SNode * > &seq, RegionScopePath &scope,
            unsigned &next_scope_id,
            std::unordered_map< std::string, ScopedLabelDef > &labels,
            std::vector< ScopedGotoRef > &gotos
        );

        void CollectRegionRefsNode(
            const SNode *node, RegionScopePath &scope,
            unsigned &next_scope_id,
            std::unordered_map< std::string, ScopedLabelDef > &labels,
            std::vector< ScopedGotoRef > &gotos
        ) {
            if (!node) {
                return;
            }
            if (auto *go = node->dyn_cast< SGoto >()) {
                gotos.push_back({ std::string(go->Target()), scope });
                return;
            }
            if (auto *stmt = node->dyn_cast< SStmt >()) {
                CollectClangRegionRefs(stmt->Stmt(), scope, labels, gotos);
                return;
            }
            if (auto *label = node->dyn_cast< SLabel >()) {
                labels.emplace(std::string(label->Name()), ScopedLabelDef{ scope });
                CollectRegionRefsSeq(
                    label->BodyList(), scope, next_scope_id, labels, gotos);
                return;
            }

            auto enter_child_region = [&](const std::vector< SNode * > &body) {
                const unsigned id = next_scope_id++;
                scope.push_back(id);
                CollectRegionRefsSeq(body, scope, next_scope_id, labels, gotos);
                scope.pop_back();
            };

            if (auto *ite = node->dyn_cast< SIfThenElse >()) {
                enter_child_region(ite->ThenList());
                enter_child_region(ite->ElseList());
                return;
            }
            if (auto *w = node->dyn_cast< SWhile >()) {
                enter_child_region(w->BodyList());
                return;
            }
            if (auto *dw = node->dyn_cast< SDoWhile >()) {
                enter_child_region(dw->BodyList());
                return;
            }
            if (auto *f = node->dyn_cast< SFor >()) {
                enter_child_region(f->BodyList());
                return;
            }
            if (auto *sw = node->dyn_cast< SSwitch >()) {
                const unsigned switch_scope = next_scope_id++;
                scope.push_back(switch_scope);
                for (const auto &case_body : sw->Cases()) {
                    CollectRegionRefsSeq(
                        case_body.body_list, scope, next_scope_id, labels, gotos);
                }
                CollectRegionRefsSeq(
                    sw->DefaultBodyList(), scope, next_scope_id, labels, gotos);
                scope.pop_back();
            }
        }

        void CollectRegionRefsSeq(
            const std::vector< SNode * > &seq, RegionScopePath &scope,
            unsigned &next_scope_id,
            std::unordered_map< std::string, ScopedLabelDef > &labels,
            std::vector< ScopedGotoRef > &gotos
        ) {
            for (const SNode *node : seq) {
                CollectRegionRefsNode(node, scope, next_scope_id, labels, gotos);
            }
        }

    } // namespace

    RegionRepairVerifierResult
    VerifyRegionRepairedBeforeLowering(const std::vector< SNode * > &root) {
        RegionRepairVerifierResult result;
        std::unordered_map< std::string, ScopedLabelDef > labels;
        std::vector< ScopedGotoRef > gotos;
        RegionScopePath root_scope;
        unsigned next_scope_id = 1;

        CollectRegionRefsSeq(root, root_scope, next_scope_id, labels, gotos);

        result.goto_refs = gotos.size();
        for (const auto &go : gotos) {
            auto label_it = labels.find(go.target);
            if (label_it == labels.end()) {
                ++result.unresolved_gotos;
                result.diagnostics.push_back(
                    "goto target `" + go.target + "` has no live label");
                continue;
            }

            const auto &label_scope = label_it->second.scope;
            if (label_scope == go.scope) {
                ++result.same_scope_gotos;
                continue;
            }
            if (IsScopePrefix(label_scope, go.scope)) {
                ++result.outward_gotos;
                continue;
            }

            ++result.cross_scope_entry_gotos;
            result.diagnostics.push_back(
                "goto target `" + go.target
                + "` enters a sibling or nested structured region"
                + " (goto scope " + FormatScopePath(go.scope)
                + ", label scope " + FormatScopePath(label_scope) + ")");
        }

        return result;
    }

} // namespace patchestry::ast
