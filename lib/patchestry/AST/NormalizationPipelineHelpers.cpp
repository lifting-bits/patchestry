/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// Implementations of shared helper functions declared in
// NormalizationPipelineInternal.hpp.  All symbols live in the
// patchestry::ast::detail namespace.

#include <algorithm>
#include <utility>

#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <llvm/Support/Casting.h>

#include "NormalizationPipelineInternal.hpp"

namespace patchestry::ast::detail {

    // =========================================================================
    // Label / goto query helpers
    // =========================================================================

    std::unordered_map< const clang::LabelDecl *, std::size_t >
    topLevelLabelIndex(const clang::CompoundStmt *body) {
        std::unordered_map< const clang::LabelDecl *, std::size_t > label_index;
        std::size_t idx = 0;
        for (const auto *stmt : body->body()) {
            if (const auto *label = llvm::dyn_cast_or_null< clang::LabelStmt >(stmt)) {
                label_index[label->getDecl()] = idx;
            }
            ++idx;
        }
        return label_index;
    }

    bool containsLabelInRange(
        const std::vector< clang::Stmt * > &stmts, std::size_t begin, std::size_t end
    ) {
        if (begin > end) {
            return false; // empty range — no labels
        }
        if (end >= stmts.size()) {
            return true; // out-of-bounds guard
        }
        for (std::size_t i = begin; i <= end; ++i) {
            if (llvm::isa< clang::LabelStmt >(stmts[i])) {
                return true;
            }
        }
        return false;
    }

    void collectGotoTargets(
        const clang::Stmt *stmt, std::vector< const clang::LabelDecl * > &targets
    ) {
        if (stmt == nullptr) {
            return;
        }
        if (const auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
            targets.push_back(gs->getLabel());
            return;
        }
        for (const auto *child : stmt->children()) {
            collectGotoTargets(child, targets);
        }
    }

    bool containsGotoTo(const clang::Stmt *stmt, const clang::LabelDecl *target) {
        if (stmt == nullptr) {
            return false;
        }
        if (const auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
            return gs->getLabel() == target;
        }
        for (const auto *child : stmt->children()) {
            if (containsGotoTo(child, target)) {
                return true;
            }
        }
        return false;
    }

    // Recursively strip LabelStmt wrappers whose LabelDecl is not referenced by
    // any GotoStmt.  When a label is stripped the sub-statement takes its place;
    // if the sub-statement is a CompoundStmt the parent CompoundStmt handler
    // flattens it.
    clang::Stmt *stripUnreferencedLabels(
        clang::ASTContext &ctx, clang::Stmt *stmt,
        const std::unordered_set< const clang::LabelDecl * > &used_labels, unsigned &removed
    ) {
        if (stmt == nullptr) {
            return nullptr;
        }

        if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
            auto *new_sub = stripUnreferencedLabels(ctx, ls->getSubStmt(), used_labels, removed);
            if (new_sub == nullptr) {
                new_sub = new (ctx) clang::NullStmt(ls->getIdentLoc(), false);
            }
            if (used_labels.count(ls->getDecl()) == 0U) {
                ++removed;
                return new_sub;
            }
            ls->setSubStmt(new_sub);
            return ls;
        }

        if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
            std::vector< clang::Stmt * > new_children;
            bool changed = false;
            for (auto *child : cs->body()) {
                auto *new_child = stripUnreferencedLabels(ctx, child, used_labels, removed);
                if (new_child == child) {
                    new_children.push_back(child);
                    continue;
                }
                changed = true;
                if (new_child == nullptr) {
                    continue;
                }
                // If a stripped LabelStmt's sub-stmt is a CompoundStmt, flatten it.
                if (auto *inner = llvm::dyn_cast< clang::CompoundStmt >(new_child)) {
                    for (auto *inner_child : inner->body()) {
                        new_children.push_back(inner_child);
                    }
                } else {
                    new_children.push_back(new_child);
                }
            }
            if (!changed) {
                return cs;
            }
            return makeCompound(ctx, new_children, cs->getLBracLoc(), cs->getRBracLoc());
        }

        if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
            auto *new_body = stripUnreferencedLabels(ctx, ws->getBody(), used_labels, removed);
            if (new_body == ws->getBody() || new_body == nullptr) {
                return ws;
            }
            return clang::WhileStmt::Create(
                ctx, nullptr, ws->getCond(), new_body, ws->getWhileLoc(), ws->getLParenLoc(),
                ws->getRParenLoc()
            );
        }

        if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
            auto *new_body = stripUnreferencedLabels(ctx, ds->getBody(), used_labels, removed);
            if (new_body == ds->getBody() || new_body == nullptr) {
                return ds;
            }
            return new (ctx) clang::DoStmt(
                new_body, ds->getCond(), ds->getDoLoc(), ds->getWhileLoc(), ds->getRParenLoc()
            );
        }

        if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
            auto *new_body = stripUnreferencedLabels(ctx, fs->getBody(), used_labels, removed);
            if (new_body == fs->getBody() || new_body == nullptr) {
                return fs;
            }
            return new (ctx) clang::ForStmt(
                ctx, fs->getInit(), fs->getCond(), nullptr, fs->getInc(), new_body,
                fs->getForLoc(), fs->getLParenLoc(), fs->getRParenLoc()
            );
        }

        if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
            auto *new_then = stripUnreferencedLabels(ctx, is->getThen(), used_labels, removed);
            auto *new_else = is->getElse() != nullptr
                ? stripUnreferencedLabels(ctx, is->getElse(), used_labels, removed)
                : nullptr;
            bool then_changed = new_then != nullptr && new_then != is->getThen();
            bool else_changed = new_else != nullptr && new_else != is->getElse();
            if (!then_changed && !else_changed) {
                return is;
            }
            auto *final_then = then_changed ? new_then : is->getThen();
            auto *final_else = else_changed ? new_else : is->getElse();
            return clang::IfStmt::Create(
                ctx, is->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr, nullptr,
                is->getCond(), is->getLParenLoc(), final_then->getBeginLoc(), final_then,
                final_else != nullptr ? final_else->getBeginLoc() : clang::SourceLocation(),
                final_else
            );
        }

        return stmt;
    }

    std::tuple< unsigned, unsigned, std::optional< clang::SourceLocation > >
    countRemainingGotos(clang::TranslationUnitDecl *tu) {
        RemainingGotoCollector collector;
        collector.TraverseDecl(tu);
        return { collector.gotos, collector.indirect_gotos, collector.first_location };
    }

    // =========================================================================
    // Constant-condition helpers
    // =========================================================================

    bool isAlwaysTrue(const clang::Expr *cond) {
        const auto *e = cond->IgnoreParenImpCasts();
        if (const auto *il = llvm::dyn_cast< clang::IntegerLiteral >(e)) {
            return il->getValue().getBoolValue();
        }
        // Look through a DeclRefExpr to its VarDecl initializer (handles materialized
        // VARNODE_TEMPORARY values such as `always_true_cond = (1 == 1)`).
        if (const auto *dr = llvm::dyn_cast< clang::DeclRefExpr >(e)) {
            if (const auto *vd = llvm::dyn_cast< clang::VarDecl >(dr->getDecl())) {
                if (const auto *init = vd->getInit()) {
                    return isAlwaysTrue(init);
                }
            }
        }
        return false;
    }

    bool isAlwaysFalse(const clang::Expr *cond) {
        const auto *e = cond->IgnoreParenCasts();
        if (const auto *il = llvm::dyn_cast< clang::IntegerLiteral >(e)) {
            return !il->getValue().getBoolValue();
        }
        // Handle constant comparisons like (0 == 1) or (0 != 0).
        // Use IgnoreParenCasts (not IgnoreParenImpCasts) to also strip
        // explicit C-style casts such as (unsigned int)0.
        if (const auto *bo = llvm::dyn_cast< clang::BinaryOperator >(e)) {
            if (bo->getOpcode() == clang::BO_EQ || bo->getOpcode() == clang::BO_NE) {
                const auto *lhs_il =
                    llvm::dyn_cast< clang::IntegerLiteral >(bo->getLHS()->IgnoreParenCasts());
                const auto *rhs_il =
                    llvm::dyn_cast< clang::IntegerLiteral >(bo->getRHS()->IgnoreParenCasts());
                if (lhs_il != nullptr && rhs_il != nullptr) {
                    bool equal = (lhs_il->getValue() == rhs_il->getValue());
                    return bo->getOpcode() == clang::BO_EQ ? !equal : equal;
                }
            }
        }
        // Look through a DeclRefExpr to its VarDecl initializer (handles materialized
        // VARNODE_TEMPORARY values such as `always_false_cond = (0 == 1)`).
        if (const auto *dr = llvm::dyn_cast< clang::DeclRefExpr >(e)) {
            if (const auto *vd = llvm::dyn_cast< clang::VarDecl >(dr->getDecl())) {
                if (const auto *init = vd->getInit()) {
                    return isAlwaysFalse(init);
                }
            }
        }
        return false;
    }

    // =========================================================================
    // CFG infrastructure
    // =========================================================================

    std::vector< std::vector< std::size_t > > buildSuccOf(
        const std::vector< clang::Stmt * > &stmts,
        const std::unordered_map< const clang::LabelDecl *, std::size_t > &label_index,
        std::size_t N
    ) {
        std::vector< std::vector< std::size_t > > succ_of(N + 1);
        for (std::size_t i = 0; i < N; ++i) {
            auto *stmt = stmts[i];
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                auto it = label_index.find(gs->getLabel());
                succ_of[i].push_back(it != label_index.end() ? it->second : N);
            } else if (llvm::isa< clang::ReturnStmt >(stmt)) {
                succ_of[i].push_back(N);
            } else if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                auto resolve = [&](const clang::Stmt *arm) -> std::size_t {
                    if (auto *arm_gs = llvm::dyn_cast_or_null< clang::GotoStmt >(arm)) {
                        auto it = label_index.find(arm_gs->getLabel());
                        return it != label_index.end() ? it->second : N;
                    }
                    if (llvm::isa_and_nonnull< clang::ReturnStmt >(arm)) {
                        return N;
                    }
                    return i + 1 < N ? i + 1 : N;
                };
                succ_of[i].push_back(resolve(is->getThen()));
                succ_of[i].push_back(
                    is->getElse() ? resolve(is->getElse()) : (i + 1 < N ? i + 1 : N)
                );
            } else {
                succ_of[i].push_back(i + 1 < N ? i + 1 : N);
            }
        }
        return succ_of;
    }

    std::vector< std::pair< std::size_t, std::size_t > > detectBackedges(
        std::size_t N, const std::vector< std::vector< std::size_t > > &succ_of
    ) {
        std::vector< std::pair< std::size_t, std::size_t > > backedges;
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j : succ_of[i]) {
                if (j < N && j <= i) {
                    backedges.emplace_back(i, j);
                }
            }
        }
        return backedges;
    }

    SimpleBlockDomTree buildDomTree(
        const std::vector< std::vector< std::size_t > > &succ_of, std::size_t entry_idx,
        std::size_t N
    ) {
        constexpr std::size_t UNDEF = SimpleBlockDomTree::UNDEF;

        // Build pred_of
        std::vector< std::vector< std::size_t > > pred_of(N + 1);
        for (std::size_t i = 0; i <= N; ++i) {
            for (std::size_t j : succ_of[i]) {
                if (j <= N) {
                    pred_of[j].push_back(i);
                }
            }
        }

        // Iterative post-order DFS to get RPO
        std::vector< std::size_t > rpo;
        std::vector< std::size_t > rpo_num(N + 1, UNDEF);
        {
            std::vector< bool > visited(N + 1, false);
            std::vector< std::pair< std::size_t, std::size_t > > stack;
            std::vector< std::size_t > post_order;
            stack.push_back({ entry_idx, 0 });
            visited[entry_idx] = true;
            while (!stack.empty()) {
                auto &[node, idx] = stack.back();
                if (idx < succ_of[node].size()) {
                    std::size_t nxt = succ_of[node][idx++];
                    if (nxt <= N && !visited[nxt]) {
                        visited[nxt] = true;
                        stack.push_back({ nxt, 0 });
                    }
                } else {
                    post_order.push_back(node);
                    stack.pop_back();
                }
            }
            std::reverse(post_order.begin(), post_order.end());
            rpo = std::move(post_order);
            for (std::size_t i = 0; i < rpo.size(); ++i) {
                rpo_num[rpo[i]] = i;
            }
        }

        // Cooper's algorithm: iterative dataflow until stable
        std::vector< std::size_t > idom(N + 1, UNDEF);
        idom[entry_idx] = entry_idx;

        auto intersect = [&](std::size_t b1, std::size_t b2) -> std::size_t {
            unsigned guard = 0;
            while (b1 != b2 && guard < 2048U) {
                while (b1 != UNDEF && b2 != UNDEF && rpo_num[b1] != UNDEF
                       && rpo_num[b2] != UNDEF && rpo_num[b1] > rpo_num[b2])
                {
                    if (idom[b1] == UNDEF) {
                        return UNDEF;
                    }
                    b1 = idom[b1];
                }
                while (b1 != UNDEF && b2 != UNDEF && rpo_num[b1] != UNDEF
                       && rpo_num[b2] != UNDEF && rpo_num[b2] > rpo_num[b1])
                {
                    if (idom[b2] == UNDEF) {
                        return UNDEF;
                    }
                    b2 = idom[b2];
                }
                if (b1 == UNDEF || b2 == UNDEF || rpo_num[b1] == UNDEF
                    || rpo_num[b2] == UNDEF)
                {
                    return UNDEF;
                }
                ++guard;
            }
            return b1;
        };

        bool changed = true;
        while (changed) {
            changed = false;
            for (std::size_t b : rpo) {
                if (b == entry_idx) {
                    continue;
                }
                std::size_t new_idom = UNDEF;
                for (std::size_t p : pred_of[b]) {
                    if (idom[p] == UNDEF) {
                        continue;
                    }
                    if (new_idom == UNDEF) {
                        new_idom = p;
                    } else {
                        new_idom = intersect(p, new_idom);
                    }
                }
                if (new_idom != idom[b]) {
                    idom[b] = new_idom;
                    changed  = true;
                }
            }
        }

        return SimpleBlockDomTree{ std::move(idom), std::move(rpo_num), entry_idx, N };
    }

    // =========================================================================
    // Unconditional-terminator predicate
    // =========================================================================

    bool isUnconditionalTerminator(const clang::Stmt *stmt) {
        if (stmt == nullptr) {
            return false;
        }
        if (llvm::isa< clang::BreakStmt >(stmt) || llvm::isa< clang::ContinueStmt >(stmt)
            || llvm::isa< clang::ReturnStmt >(stmt) || llvm::isa< clang::GotoStmt >(stmt))
        {
            return true;
        }
        if (const auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
            return is->getElse() != nullptr && isUnconditionalTerminator(is->getThen())
                   && isUnconditionalTerminator(is->getElse());
        }
        if (const auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
            if (cs->body_empty()) {
                return false;
            }
            return isUnconditionalTerminator(*(cs->body_end() - 1));
        }
        return false;
    }

} // namespace patchestry::ast::detail
