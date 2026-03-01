/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// Conditional structuring passes:
//   ConditionalStructurizePass  – convert goto-diamond patterns to if/if-else
//   IfElseRegionFormationPass   – recover nested if-else regions from flat goto sequences

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <patchestry/AST/ASTPassManager.hpp>
#include <patchestry/Util/Log.hpp>

#include "NormalizationPipelineInternal.hpp"

namespace patchestry::ast {
    namespace {

        using namespace detail;

        // =========================================================================
        // ConditionalStructurizePass
        // =========================================================================

        class ConditionalStructurizePass final : public ASTPass
        {
          public:
            explicit ConditionalStructurizePass(PipelineState &state)
                : state(state) {}

            const char *name(void) const override { return "ConditionalStructurizePass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                state.conditional_structurized = 0;

                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }

                    auto *body = llvm::dyn_cast< clang::CompoundStmt >(func->getBody());
                    if (body == nullptr) {
                        continue;
                    }

                    auto *rewritten = processStmt(ctx, body);
                    if (rewritten != body) {
                        func->setBody(rewritten);
                        state.cfg_stale = true;
                    }
                }

                runCfgExtractPass(state, ctx, options);
                return true;
            }

          private:
            PipelineState &state;

            clang::CompoundStmt *
            processCompound(clang::ASTContext &ctx, clang::CompoundStmt *compound) {
                if (compound->size() < 3U) {
                    return compound;
                }

                auto *body = compound;
                std::vector< clang::Stmt * > stmts(body->body_begin(), body->body_end());
                auto label_index = topLevelLabelIndex(body);

                for (std::size_t i = 0; i < stmts.size(); ++i) {
                    auto *if_stmt = llvm::dyn_cast< clang::IfStmt >(stmts[i]);
                    auto *if_label_wrapper = static_cast< clang::LabelStmt * >(nullptr);
                    if (if_stmt == nullptr) {
                        if (auto *lbl = llvm::dyn_cast< clang::LabelStmt >(stmts[i])) {
                            if_stmt = llvm::dyn_cast< clang::IfStmt >(lbl->getSubStmt());
                            if (if_stmt != nullptr) {
                                if_label_wrapper = lbl;
                            }
                        }
                    }
                    if (if_stmt == nullptr) {
                        continue;
                    }

                    auto *then_goto = llvm::dyn_cast_or_null< clang::GotoStmt >(
                        if_stmt->getThen()
                    );
                    auto *else_goto = llvm::dyn_cast_or_null< clang::GotoStmt >(
                        if_stmt->getElse()
                    );
                    if (then_goto == nullptr || else_goto == nullptr) {
                        // Single-sided if-goto: if (c) goto L_skip; [fallthrough_body] L_skip:
                        // Transform to: if (!c) { <fallthrough_body> }
                        if (then_goto != nullptr && else_goto == nullptr
                            && label_index.contains(then_goto->getLabel()))
                        {
                            std::size_t skip_label_idx =
                                label_index.at(then_goto->getLabel());
                            // Skip if the next statement is also a plain "if (c) goto L;"
                            // to the same label — IfElseRegionFormationPass will merge
                            // them into a single combined condition first.
                            auto *next_if_csp = (i + 1U < stmts.size())
                                ? llvm::dyn_cast< clang::IfStmt >(stmts[i + 1U])
                                : nullptr;
                            bool next_is_same_target = false;
                            if (next_if_csp != nullptr) {
                                auto *nt = llvm::dyn_cast_or_null< clang::GotoStmt >(
                                    next_if_csp->getThen()
                                );
                                next_is_same_target =
                                    nt != nullptr && nt->getLabel() == then_goto->getLabel();
                            }
                            if (skip_label_idx > i + 1 && skip_label_idx < stmts.size()
                                && !containsLabelInRange(stmts, i + 1, skip_label_idx - 1)
                                && !next_is_same_target)
                            {
                                std::vector< clang::Stmt * > fallthrough_body(
                                    stmts.begin() + static_cast< std::ptrdiff_t >(i + 1),
                                    stmts.begin()
                                        + static_cast< std::ptrdiff_t >(skip_label_idx)
                                );
                                auto loc = if_stmt->getIfLoc();
                                auto *negated_cond = clang::UnaryOperator::Create(
                                    ctx,
                                    ensureRValue(
                                        ctx,
                                        if_stmt->getCond()
                                    ),
                                    clang::UO_LNot, ctx.IntTy, clang::VK_PRValue,
                                    clang::OK_Ordinary, loc, false, clang::FPOptionsOverride()
                                );
                                auto *new_body = makeCompound(ctx, fallthrough_body, loc, loc);
                                auto *new_if   = clang::IfStmt::Create(
                                    ctx, loc, clang::IfStatementKind::Ordinary, nullptr,
                                    nullptr, negated_cond, if_stmt->getLParenLoc(),
                                    new_body->getBeginLoc(), new_body,
                                    clang::SourceLocation(), nullptr
                                );
                                std::vector< clang::Stmt * > rewritten;
                                rewritten.reserve(stmts.size());
                                rewritten.insert(
                                    rewritten.end(), stmts.begin(),
                                    stmts.begin() + static_cast< std::ptrdiff_t >(i)
                                );
                                rewritten.push_back(new_if);
                                rewritten.insert(
                                    rewritten.end(),
                                    stmts.begin()
                                        + static_cast< std::ptrdiff_t >(skip_label_idx),
                                    stmts.end()
                                );
                                stmts = std::move(rewritten);
                                body  = makeCompound(
                                    ctx, stmts, body->getLBracLoc(), body->getRBracLoc()
                                );
                                label_index = topLevelLabelIndex(body);
                                ++state.conditional_structurized;
                                i = 0;
                            }
                        }
                        continue;
                    }

                    if (!label_index.contains(then_goto->getLabel())
                        || !label_index.contains(else_goto->getLabel()))
                    {
                        continue;
                    }

                    std::size_t then_label_idx = label_index.at(then_goto->getLabel());
                    std::size_t else_label_idx = label_index.at(else_goto->getLabel());

                    bool negated_condition = false;
                    if (then_label_idx > i && else_label_idx > i
                        && then_label_idx > else_label_idx)
                    {
                        std::swap(then_label_idx, else_label_idx);
                        negated_condition = true;
                    }
                    if (then_label_idx <= i || else_label_idx <= i
                        || then_label_idx >= else_label_idx)
                    {
                        continue;
                    }

                    auto *then_label_stmt =
                        llvm::dyn_cast< clang::LabelStmt >(stmts[then_label_idx]);
                    auto *else_label_stmt =
                        llvm::dyn_cast< clang::LabelStmt >(stmts[else_label_idx]);
                    if (then_label_stmt == nullptr || else_label_stmt == nullptr) {
                        continue;
                    }

                    auto *then_term_goto =
                        llvm::dyn_cast< clang::GotoStmt >(stmts[else_label_idx - 1]);
                    if (then_term_goto == nullptr
                        && else_label_idx - 1 == then_label_idx)
                    {
                        then_term_goto = llvm::dyn_cast< clang::GotoStmt >(
                            then_label_stmt->getSubStmt()
                        );
                    }

                    // Then-body falls through to else-label (no explicit goto to join)
                    if (then_term_goto == nullptr) {
                        if (!containsLabelInRange(
                                stmts, then_label_idx + 1, else_label_idx - 1
                            ))
                        {
                            std::vector< clang::Stmt * > then_body;
                            then_body.push_back(then_label_stmt->getSubStmt());
                            for (std::size_t j = then_label_idx + 1; j < else_label_idx;
                                 ++j)
                            {
                                then_body.push_back(stmts[j]);
                            }
                            auto loc          = if_stmt->getIfLoc();
                            clang::Expr *cond = if_stmt->getCond();
                            if (negated_condition) {
                                cond = clang::UnaryOperator::Create(
                                    ctx, ensureRValue(ctx, cond), clang::UO_LNot, ctx.IntTy,
                                    clang::VK_PRValue, clang::OK_Ordinary, loc, false,
                                    clang::FPOptionsOverride()
                                );
                            }
                            auto *new_then = makeCompound(
                                ctx, then_body, then_label_stmt->getBeginLoc(),
                                then_label_stmt->getEndLoc()
                            );
                            auto *new_if = clang::IfStmt::Create(
                                ctx, loc, clang::IfStatementKind::Ordinary, nullptr,
                                nullptr, cond, if_stmt->getLParenLoc(),
                                new_then->getBeginLoc(), new_then, clang::SourceLocation(),
                                nullptr
                            );
                            std::vector< clang::Stmt * > rewritten;
                            rewritten.reserve(stmts.size());
                            if (if_label_wrapper != nullptr) {
                                if_label_wrapper->setSubStmt(new_if);
                                rewritten.insert(
                                    rewritten.end(), stmts.begin(),
                                    stmts.begin() + static_cast< std::ptrdiff_t >(i + 1)
                                );
                            } else {
                                rewritten.insert(
                                    rewritten.end(), stmts.begin(),
                                    stmts.begin() + static_cast< std::ptrdiff_t >(i)
                                );
                                rewritten.push_back(new_if);
                            }
                            rewritten.insert(
                                rewritten.end(),
                                stmts.begin()
                                    + static_cast< std::ptrdiff_t >(else_label_idx),
                                stmts.end()
                            );
                            stmts = std::move(rewritten);
                            body  = makeCompound(
                                ctx, stmts, body->getLBracLoc(), body->getRBracLoc()
                            );
                            label_index = topLevelLabelIndex(body);
                            ++state.conditional_structurized;
                            i = 0;
                        }
                        continue;
                    }

                    if (!label_index.contains(then_term_goto->getLabel())) {
                        continue;
                    }

                    std::size_t join_label_idx =
                        label_index.at(then_term_goto->getLabel());
                    if (join_label_idx < else_label_idx
                        || join_label_idx >= stmts.size())
                    {
                        continue;
                    }

                    // Special case: join_label == else_label → single-sided if
                    if (join_label_idx == else_label_idx) {
                        if (containsLabelInRange(
                                stmts, then_label_idx + 1, else_label_idx - 2
                            ))
                        {
                            continue;
                        }
                        std::vector< clang::Stmt * > then_only_body;
                        if (else_label_idx - 1 != then_label_idx) {
                            then_only_body.push_back(then_label_stmt->getSubStmt());
                        }
                        {
                            auto tb_begin =
                                static_cast< std::ptrdiff_t >(then_label_idx + 1);
                            auto tb_end =
                                static_cast< std::ptrdiff_t >(else_label_idx - 1);
                            if (tb_begin < tb_end) {
                                then_only_body.insert(
                                    then_only_body.end(), stmts.begin() + tb_begin,
                                    stmts.begin() + tb_end
                                );
                            }
                        }
                        if (then_only_body.empty()) {
                            then_only_body.push_back(
                                new (ctx) clang::NullStmt(if_stmt->getBeginLoc(), false)
                            );
                        }
                        clang::Expr *single_cond = if_stmt->getCond();
                        if (negated_condition) {
                            single_cond = clang::UnaryOperator::Create(
                                ctx, ensureRValue(ctx, single_cond), clang::UO_LNot,
                                ctx.IntTy, clang::VK_PRValue, clang::OK_Ordinary,
                                if_stmt->getIfLoc(), false, clang::FPOptionsOverride()
                            );
                        }
                        auto *single_then = makeCompound(
                            ctx, then_only_body, then_label_stmt->getBeginLoc(),
                            then_label_stmt->getEndLoc()
                        );
                        auto *single_if = clang::IfStmt::Create(
                            ctx, if_stmt->getIfLoc(), clang::IfStatementKind::Ordinary,
                            nullptr, nullptr, single_cond, if_stmt->getLParenLoc(),
                            single_then->getBeginLoc(), single_then,
                            clang::SourceLocation(), nullptr
                        );
                        std::vector< clang::Stmt * > rewritten;
                        rewritten.reserve(stmts.size());
                        if (if_label_wrapper != nullptr) {
                            if_label_wrapper->setSubStmt(single_if);
                            rewritten.insert(
                                rewritten.end(), stmts.begin(),
                                stmts.begin() + static_cast< std::ptrdiff_t >(i + 1)
                            );
                        } else {
                            rewritten.insert(
                                rewritten.end(), stmts.begin(),
                                stmts.begin() + static_cast< std::ptrdiff_t >(i)
                            );
                            rewritten.push_back(single_if);
                        }
                        rewritten.insert(
                            rewritten.end(),
                            stmts.begin()
                                + static_cast< std::ptrdiff_t >(join_label_idx),
                            stmts.end()
                        );
                        stmts = std::move(rewritten);
                        body  = makeCompound(
                            ctx, stmts, body->getLBracLoc(), body->getRBracLoc()
                        );
                        label_index = topLevelLabelIndex(body);
                        ++state.conditional_structurized;
                        i = 0;
                        continue;
                    }

                    auto *join_label_stmt =
                        llvm::dyn_cast< clang::LabelStmt >(stmts[join_label_idx]);
                    if (join_label_stmt == nullptr) {
                        continue;
                    }

                    if (containsLabelInRange(stmts, then_label_idx + 1, else_label_idx - 2)
                        || containsLabelInRange(
                            stmts, else_label_idx + 1, join_label_idx - 1
                        ))
                    {
                        continue;
                    }

                    std::vector< clang::Stmt * > then_body;
                    then_body.push_back(then_label_stmt->getSubStmt());
                    {
                        auto tb_begin = static_cast< std::ptrdiff_t >(then_label_idx + 1);
                        auto tb_end   = static_cast< std::ptrdiff_t >(else_label_idx - 1);
                        if (tb_begin < tb_end) {
                            then_body.insert(
                                then_body.end(), stmts.begin() + tb_begin,
                                stmts.begin() + tb_end
                            );
                        }
                    }
                    std::vector< clang::Stmt * > else_body;
                    else_body.push_back(else_label_stmt->getSubStmt());
                    {
                        auto eb_begin = static_cast< std::ptrdiff_t >(else_label_idx + 1);
                        auto eb_end   = static_cast< std::ptrdiff_t >(join_label_idx);
                        if (eb_begin < eb_end) {
                            else_body.insert(
                                else_body.end(), stmts.begin() + eb_begin,
                                stmts.begin() + eb_end
                            );
                        }
                    }
                    if (then_body.empty()) {
                        then_body.push_back(
                            new (ctx) clang::NullStmt(if_stmt->getBeginLoc(), false)
                        );
                    }
                    if (else_body.empty()) {
                        else_body.push_back(
                            new (ctx) clang::NullStmt(if_stmt->getBeginLoc(), false)
                        );
                    }

                    auto *new_then = makeCompound(
                        ctx, then_body, then_label_stmt->getBeginLoc(),
                        then_label_stmt->getEndLoc()
                    );
                    auto *new_else = makeCompound(
                        ctx, else_body, else_label_stmt->getBeginLoc(),
                        else_label_stmt->getEndLoc()
                    );
                    clang::Expr *two_sided_cond = if_stmt->getCond();
                    if (negated_condition) {
                        two_sided_cond = clang::UnaryOperator::Create(
                            ctx, ensureRValue(ctx, two_sided_cond), clang::UO_LNot,
                            ctx.IntTy, clang::VK_PRValue, clang::OK_Ordinary,
                            if_stmt->getIfLoc(), false, clang::FPOptionsOverride()
                        );
                    }
                    auto *new_if = clang::IfStmt::Create(
                        ctx, if_stmt->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr,
                        nullptr, two_sided_cond, if_stmt->getLParenLoc(),
                        new_then->getBeginLoc(), new_then, new_else->getBeginLoc(), new_else
                    );

                    std::vector< clang::Stmt * > rewritten;
                    rewritten.reserve(stmts.size());
                    if (if_label_wrapper != nullptr) {
                        if_label_wrapper->setSubStmt(new_if);
                        rewritten.insert(
                            rewritten.end(), stmts.begin(),
                            stmts.begin() + static_cast< std::ptrdiff_t >(i + 1)
                        );
                    } else {
                        rewritten.insert(
                            rewritten.end(), stmts.begin(),
                            stmts.begin() + static_cast< std::ptrdiff_t >(i)
                        );
                        rewritten.push_back(new_if);
                    }
                    rewritten.insert(
                        rewritten.end(),
                        stmts.begin() + static_cast< std::ptrdiff_t >(join_label_idx),
                        stmts.end()
                    );

                    stmts = std::move(rewritten);
                    body  = makeCompound(ctx, stmts, body->getLBracLoc(), body->getRBracLoc());
                    label_index = topLevelLabelIndex(body);
                    ++state.conditional_structurized;
                    i = 0;
                }

                return body;
            }

            clang::Stmt *processStmt(clang::ASTContext &ctx, clang::Stmt *stmt) {
                if (stmt == nullptr) {
                    return nullptr;
                }

                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                    bool changed    = false;
                    auto *rewritten = processCompound(ctx, cs);
                    changed         = (rewritten != cs);
                    cs              = rewritten;
                    std::vector< clang::Stmt * > children;
                    children.reserve(cs->size());
                    for (auto *child : cs->body()) {
                        auto *new_child = processStmt(ctx, child);
                        children.push_back(new_child != nullptr ? new_child : child);
                        changed = changed || (new_child != nullptr && new_child != child);
                    }
                    if (!changed) {
                        return cs;
                    }
                    return makeCompound(ctx, children, cs->getLBracLoc(), cs->getRBracLoc());
                }

                if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                    auto *new_body = processStmt(ctx, ws->getBody());
                    if (new_body == ws->getBody() || new_body == nullptr) {
                        return ws;
                    }
                    return clang::WhileStmt::Create(
                        ctx, nullptr, ws->getCond(), new_body, ws->getWhileLoc(),
                        ws->getLParenLoc(), ws->getRParenLoc()
                    );
                }

                if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                    auto *new_body = processStmt(ctx, ds->getBody());
                    if (new_body == ds->getBody() || new_body == nullptr) {
                        return ds;
                    }
                    return new (ctx) clang::DoStmt(
                        new_body, ds->getCond(), ds->getDoLoc(), ds->getWhileLoc(),
                        ds->getRParenLoc()
                    );
                }

                if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                    auto *new_body = processStmt(ctx, fs->getBody());
                    if (new_body == fs->getBody() || new_body == nullptr) {
                        return fs;
                    }
                    return new (ctx) clang::ForStmt(
                        ctx, fs->getInit(), fs->getCond(), nullptr, fs->getInc(), new_body,
                        fs->getForLoc(), fs->getLParenLoc(), fs->getRParenLoc()
                    );
                }

                if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                    auto *new_then = processStmt(ctx, is->getThen());
                    auto *new_else = is->getElse() != nullptr
                        ? processStmt(ctx, is->getElse())
                        : nullptr;
                    if ((new_then == nullptr || new_then == is->getThen())
                        && (new_else == nullptr || new_else == is->getElse()))
                    {
                        return is;
                    }
                    if (new_then == nullptr) {
                        new_then = is->getThen();
                    }
                    return clang::IfStmt::Create(
                        ctx, is->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr, nullptr,
                        is->getCond(), is->getLParenLoc(), new_then->getBeginLoc(), new_then,
                        new_else != nullptr ? new_else->getBeginLoc() : clang::SourceLocation(),
                        new_else
                    );
                }

                if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                    auto *new_sub = processStmt(ctx, ls->getSubStmt());
                    if (new_sub == nullptr || new_sub == ls->getSubStmt()) {
                        return ls;
                    }
                    return new (ctx) clang::LabelStmt(ls->getIdentLoc(), ls->getDecl(), new_sub);
                }

                return stmt;
            }
        };

        // =========================================================================
        // IfElseRegionFormationPass
        // =========================================================================

        class IfElseRegionFormationPass final : public ASTPass
        {
          public:
            explicit IfElseRegionFormationPass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "IfElseRegionFormationPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                unsigned rewrites = 0;

                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }

                    auto *rewritten = processStmt(ctx, func->getBody(), rewrites);
                    if (rewritten != func->getBody()) {
                        func->setBody(rewritten);
                        state.cfg_stale = true;
                        runCfgExtractPass(state, ctx, options);
                    }
                }

                if (options.verbose && rewrites > 0U) {
                    LOG(DEBUG) << "IfElseRegionFormationPass: rewrote " << rewrites
                               << " region(s)\n";
                }

                return true;
            }

          private:
            PipelineState &state;

            struct IfHead
            {
                clang::IfStmt *if_stmt                = nullptr;
                clang::LabelStmt *if_label_wrapper    = nullptr;
                clang::GotoStmt *then_goto            = nullptr;
                clang::GotoStmt *else_goto            = nullptr;
                bool then_is_goto                     = false;
                bool else_is_goto                     = false;
                bool has_single_goto_arm              = false;
                bool goto_on_then                     = false;
                const clang::LabelDecl *single_target = nullptr;
                std::size_t single_target_idx         = 0;
                std::size_t then_target_idx           = 0;
                std::size_t else_target_idx           = 0;
            };

            struct ArmDiscovery
            {
                std::size_t entry_idx         = 0;
                std::size_t end_exclusive_idx = 0;
                std::size_t join_idx          = 0;
                bool explicit_join_goto       = false;
            };

            static bool isUnexpectedLabelInRange(
                const std::vector< clang::Stmt * > &stmts, std::size_t begin, std::size_t end,
                const std::unordered_set< std::size_t > &allowed_label_indices
            ) {
                if (begin > end || end >= stmts.size()) {
                    return false;
                }
                for (std::size_t i = begin; i <= end; ++i) {
                    if (llvm::isa< clang::LabelStmt >(stmts[i])
                        && !allowed_label_indices.contains(i))
                    {
                        return true;
                    }
                }
                return false;
            }

            static std::unordered_map< const clang::LabelDecl *, std::vector< std::size_t > >
            buildGotoSourcesByTarget(const std::vector< clang::Stmt * > &stmts) {
                std::unordered_map< const clang::LabelDecl *, std::vector< std::size_t > >
                    result;
                for (std::size_t i = 0; i < stmts.size(); ++i) {
                    std::vector< const clang::LabelDecl * > targets;
                    collectGotoTargets(stmts[i], targets);
                    std::unordered_set< const clang::LabelDecl * > unique_targets(
                        targets.begin(), targets.end()
                    );
                    for (const auto *target : unique_targets) {
                        result[target].push_back(i);
                    }
                }
                return result;
            }

            static bool hasExternalJumpIntoRegionMiddle(
                const std::vector< clang::Stmt * > &stmts,
                const std::unordered_map< const clang::LabelDecl *, std::vector< std::size_t > >
                    &goto_sources_by_target,
                std::size_t region_begin, std::size_t region_end
            ) {
                if (region_end <= region_begin + 1U || region_end >= stmts.size()) {
                    return false;
                }
                for (std::size_t i = region_begin + 1; i < region_end; ++i) {
                    auto *label_stmt = llvm::dyn_cast< clang::LabelStmt >(stmts[i]);
                    if (label_stmt == nullptr) {
                        continue;
                    }
                    auto it = goto_sources_by_target.find(label_stmt->getDecl());
                    if (it == goto_sources_by_target.end()) {
                        continue;
                    }
                    for (std::size_t src_idx : it->second) {
                        if (src_idx < region_begin || src_idx > region_end) {
                            return true;
                        }
                    }
                }
                return false;
            }

            static std::optional< IfHead > detectIfHead(
                const std::vector< clang::Stmt * > &stmts,
                const std::unordered_map< const clang::LabelDecl *, std::size_t > &label_index,
                std::size_t i
            ) {
                IfHead head;
                head.if_stmt = llvm::dyn_cast< clang::IfStmt >(stmts[i]);
                if (head.if_stmt == nullptr) {
                    auto *wrapper = llvm::dyn_cast< clang::LabelStmt >(stmts[i]);
                    if (wrapper == nullptr) {
                        return std::nullopt;
                    }
                    head.if_stmt = llvm::dyn_cast< clang::IfStmt >(wrapper->getSubStmt());
                    if (head.if_stmt == nullptr) {
                        return std::nullopt;
                    }
                    head.if_label_wrapper = wrapper;
                }

                head.then_goto =
                    llvm::dyn_cast_or_null< clang::GotoStmt >(head.if_stmt->getThen());
                head.else_goto =
                    llvm::dyn_cast_or_null< clang::GotoStmt >(head.if_stmt->getElse());
                head.then_is_goto = head.then_goto != nullptr;
                head.else_is_goto = head.else_goto != nullptr;
                if (!head.then_is_goto && !head.else_is_goto) {
                    return std::nullopt;
                }

                if (head.then_is_goto) {
                    const auto *label = head.then_goto->getLabel();
                    if (!label_index.contains(label)) {
                        return std::nullopt;
                    }
                    head.then_target_idx = label_index.at(label);
                    if (head.then_target_idx <= i) {
                        return std::nullopt;
                    }
                }
                if (head.else_is_goto) {
                    const auto *label = head.else_goto->getLabel();
                    if (!label_index.contains(label)) {
                        return std::nullopt;
                    }
                    head.else_target_idx = label_index.at(label);
                    if (head.else_target_idx <= i) {
                        return std::nullopt;
                    }
                }

                head.has_single_goto_arm = head.then_is_goto != head.else_is_goto;
                if (head.has_single_goto_arm) {
                    head.goto_on_then  = head.then_is_goto;
                    head.single_target = head.goto_on_then ? head.then_goto->getLabel()
                                                           : head.else_goto->getLabel();
                    head.single_target_idx =
                        head.goto_on_then ? head.then_target_idx : head.else_target_idx;
                }
                return head;
            }

            static clang::Expr *maybeNegateCond(
                clang::ASTContext &ctx, clang::Expr *cond, bool negate,
                clang::SourceLocation loc
            ) {
                if (!negate) {
                    return cond;
                }
                return clang::UnaryOperator::Create(
                    ctx, ensureRValue(ctx, cond), clang::UO_LNot, ctx.IntTy, clang::VK_PRValue,
                    clang::OK_Ordinary, loc, false, clang::FPOptionsOverride()
                );
            }

            static std::vector< clang::Stmt * > buildSlice(
                const std::vector< clang::Stmt * > &stmts, std::size_t begin, std::size_t end
            ) {
                if (begin > end || end >= stmts.size()) {
                    return {};
                }
                return std::vector< clang::Stmt * >(
                    stmts.begin() + static_cast< std::ptrdiff_t >(begin),
                    stmts.begin() + static_cast< std::ptrdiff_t >(end + 1U)
                );
            }

            static void appendRange(
                std::vector< clang::Stmt * > &dst, const std::vector< clang::Stmt * > &src,
                std::size_t begin, std::size_t end
            ) {
                if (begin > end || end >= src.size()) {
                    return;
                }
                dst.insert(
                    dst.end(), src.begin() + static_cast< std::ptrdiff_t >(begin),
                    src.begin() + static_cast< std::ptrdiff_t >(end + 1U)
                );
            }

            static clang::Stmt *ensureNonEmptyCompound(
                clang::ASTContext &ctx, std::vector< clang::Stmt * > body,
                clang::SourceLocation begin_loc, clang::SourceLocation end_loc
            ) {
                if (body.empty()) {
                    body.push_back(new (ctx) clang::NullStmt(begin_loc, false));
                }
                return makeCompound(ctx, body, begin_loc, end_loc);
            }

            static bool rewriteSingleSidedRegion(
                clang::ASTContext &ctx, std::vector< clang::Stmt * > &stmts,
                std::size_t region_begin, std::size_t region_end, const IfHead &head
            ) {
                if (region_end <= region_begin + 1U || region_end >= stmts.size()) {
                    return false;
                }
                if (containsLabelInRange(stmts, region_begin + 1U, region_end - 1U)) {
                    return false;
                }

                auto body_stmts   = buildSlice(stmts, region_begin + 1U, region_end - 1U);
                clang::Expr *cond = maybeNegateCond(
                    ctx, head.if_stmt->getCond(), head.goto_on_then, head.if_stmt->getIfLoc()
                );
                auto *new_then = ensureNonEmptyCompound(
                    ctx, std::move(body_stmts), head.if_stmt->getBeginLoc(),
                    head.if_stmt->getEndLoc()
                );
                auto *new_if = clang::IfStmt::Create(
                    ctx, head.if_stmt->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr,
                    nullptr, cond, head.if_stmt->getLParenLoc(), new_then->getBeginLoc(),
                    new_then, clang::SourceLocation(), nullptr
                );

                std::vector< clang::Stmt * > rewritten;
                rewritten.reserve(stmts.size());
                if (head.if_label_wrapper != nullptr) {
                    head.if_label_wrapper->setSubStmt(new_if);
                    appendRange(rewritten, stmts, 0U, region_begin);
                } else {
                    appendRange(rewritten, stmts, 0U, region_begin - 1U);
                    rewritten.push_back(new_if);
                }
                appendRange(rewritten, stmts, region_end, stmts.size() - 1U);
                stmts = std::move(rewritten);
                return true;
            }

            static bool rewriteTwoSidedRegion(
                clang::ASTContext &ctx, std::vector< clang::Stmt * > &stmts,
                std::size_t region_begin, std::size_t target_idx, std::size_t split_goto_idx,
                std::size_t join_idx, const IfHead &head,
                const std::unordered_map< const clang::LabelDecl *, std::vector< std::size_t > >
                    &goto_sources
            ) {
                if (!(region_begin < split_goto_idx && split_goto_idx < target_idx
                      && target_idx < join_idx && join_idx < stmts.size()))
                {
                    return false;
                }

                auto *target_label_stmt = llvm::dyn_cast< clang::LabelStmt >(stmts[target_idx]);
                auto *join_label_stmt   = llvm::dyn_cast< clang::LabelStmt >(stmts[join_idx]);
                if (target_label_stmt == nullptr || join_label_stmt == nullptr) {
                    return false;
                }

                auto has_cross_arm_jump = [&](std::size_t src_begin, std::size_t src_end,
                                              std::size_t tgt_begin, std::size_t tgt_end) {
                    for (std::size_t k = tgt_begin; k < tgt_end && k < stmts.size(); ++k) {
                        const auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmts[k]);
                        if (ls == nullptr) {
                            continue;
                        }
                        auto it = goto_sources.find(ls->getDecl());
                        if (it == goto_sources.end()) {
                            continue;
                        }
                        for (std::size_t src : it->second) {
                            if (src >= src_begin && src < src_end) {
                                return true;
                            }
                        }
                    }
                    return false;
                };
                if (has_cross_arm_jump(target_idx, join_idx, region_begin + 1U, split_goto_idx)
                    || has_cross_arm_jump(
                        region_begin + 1U, split_goto_idx, target_idx + 1U, join_idx
                    ))
                {
                    return false;
                }

                std::vector< clang::Stmt * > fallthrough_body;
                if (split_goto_idx > region_begin + 1U) {
                    appendRange(
                        fallthrough_body, stmts, region_begin + 1U, split_goto_idx - 1U
                    );
                }

                std::vector< clang::Stmt * > goto_body;
                if (!isNullStmt(target_label_stmt->getSubStmt())) {
                    goto_body.push_back(target_label_stmt->getSubStmt());
                }
                if (join_idx > target_idx + 1U) {
                    appendRange(goto_body, stmts, target_idx + 1U, join_idx - 1U);
                }

                std::vector< clang::Stmt * > then_body;
                std::vector< clang::Stmt * > else_body;
                if (head.goto_on_then) {
                    then_body = goto_body;
                    else_body = fallthrough_body;
                } else {
                    then_body = fallthrough_body;
                    else_body = goto_body;
                }

                auto *new_then = ensureNonEmptyCompound(
                    ctx, std::move(then_body), head.if_stmt->getBeginLoc(),
                    head.if_stmt->getEndLoc()
                );
                auto *new_else = ensureNonEmptyCompound(
                    ctx, std::move(else_body), head.if_stmt->getBeginLoc(),
                    head.if_stmt->getEndLoc()
                );
                auto *new_if = clang::IfStmt::Create(
                    ctx, head.if_stmt->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr,
                    nullptr, head.if_stmt->getCond(), head.if_stmt->getLParenLoc(),
                    new_then->getBeginLoc(), new_then, new_else->getBeginLoc(), new_else
                );

                std::vector< clang::Stmt * > rewritten;
                rewritten.reserve(stmts.size());
                if (head.if_label_wrapper != nullptr) {
                    head.if_label_wrapper->setSubStmt(new_if);
                    appendRange(rewritten, stmts, 0U, region_begin);
                } else {
                    appendRange(rewritten, stmts, 0U, region_begin - 1U);
                    rewritten.push_back(new_if);
                }
                appendRange(rewritten, stmts, join_idx, stmts.size() - 1U);
                stmts = std::move(rewritten);
                return true;
            }

            static std::optional< ArmDiscovery > discoverArmWithExplicitJoin(
                const std::vector< clang::Stmt * > &stmts,
                const std::unordered_map< const clang::LabelDecl *, std::size_t > &label_index,
                std::size_t entry_idx, std::size_t stop_before_idx
            ) {
                if (entry_idx >= stmts.size() || stop_before_idx <= entry_idx
                    || stop_before_idx > stmts.size())
                {
                    return std::nullopt;
                }
                auto *entry_label_stmt = llvm::dyn_cast< clang::LabelStmt >(stmts[entry_idx]);
                if (entry_label_stmt == nullptr) {
                    return std::nullopt;
                }

                std::optional< std::size_t > first_term_source_idx;
                std::optional< std::size_t > first_join_idx;
                std::unordered_set< std::size_t > distinct_join_indices;

                auto consider_goto = [&](const clang::GotoStmt *goto_stmt,
                                         std::size_t source_idx) {
                    if (goto_stmt == nullptr || !label_index.contains(goto_stmt->getLabel())) {
                        return;
                    }
                    const std::size_t candidate_join_idx =
                        label_index.at(goto_stmt->getLabel());
                    if (candidate_join_idx <= source_idx) {
                        return;
                    }
                    distinct_join_indices.insert(candidate_join_idx);
                    if (!first_term_source_idx.has_value()) {
                        first_term_source_idx = source_idx;
                        first_join_idx        = candidate_join_idx;
                    }
                };

                consider_goto(
                    llvm::dyn_cast< clang::GotoStmt >(entry_label_stmt->getSubStmt()),
                    entry_idx
                );
                for (std::size_t j = entry_idx + 1U; j < stop_before_idx; ++j) {
                    if (llvm::isa< clang::LabelStmt >(stmts[j])) {
                        break;
                    }
                    consider_goto(llvm::dyn_cast< clang::GotoStmt >(stmts[j]), j);
                    if (first_term_source_idx.has_value() && *first_term_source_idx == j) {
                        break;
                    }
                }

                if (!first_term_source_idx.has_value() || !first_join_idx.has_value()
                    || distinct_join_indices.size() != 1U)
                {
                    return std::nullopt;
                }
                if (*first_join_idx <= stop_before_idx - 1U) {
                    return std::nullopt;
                }
                ArmDiscovery arm;
                arm.entry_idx          = entry_idx;
                arm.end_exclusive_idx  = *first_term_source_idx;
                arm.join_idx           = *first_join_idx;
                arm.explicit_join_goto = true;
                return arm;
            }

            static std::optional< ArmDiscovery > discoverArmToKnownJoin(
                const std::vector< clang::Stmt * > &stmts,
                const std::unordered_map< const clang::LabelDecl *, std::size_t > &label_index,
                std::size_t entry_idx, std::size_t join_idx
            ) {
                if (entry_idx >= stmts.size() || join_idx >= stmts.size()
                    || entry_idx + 1U > join_idx)
                {
                    return std::nullopt;
                }
                auto *entry_label_stmt = llvm::dyn_cast< clang::LabelStmt >(stmts[entry_idx]);
                auto *join_label_stmt  = llvm::dyn_cast< clang::LabelStmt >(stmts[join_idx]);
                if (entry_label_stmt == nullptr || join_label_stmt == nullptr) {
                    return std::nullopt;
                }

                auto is_forward_goto = [&](const clang::GotoStmt *goto_stmt,
                                           std::size_t source_idx) {
                    if (goto_stmt == nullptr || !label_index.contains(goto_stmt->getLabel())) {
                        return std::optional< std::size_t >{};
                    }
                    const std::size_t target_idx = label_index.at(goto_stmt->getLabel());
                    if (target_idx <= source_idx) {
                        return std::optional< std::size_t >{};
                    }
                    return std::optional< std::size_t >{ target_idx };
                };

                std::optional< std::size_t > explicit_term_source_idx;
                bool saw_non_join_forward_target = false;

                if (auto target = is_forward_goto(
                        llvm::dyn_cast< clang::GotoStmt >(entry_label_stmt->getSubStmt()),
                        entry_idx
                    );
                    target.has_value())
                {
                    if (*target == join_idx) {
                        explicit_term_source_idx = entry_idx;
                    } else {
                        saw_non_join_forward_target = true;
                    }
                }

                for (std::size_t j = entry_idx + 1U; j < join_idx; ++j) {
                    if (llvm::isa< clang::LabelStmt >(stmts[j])) {
                        return std::nullopt;
                    }
                    auto target =
                        is_forward_goto(llvm::dyn_cast< clang::GotoStmt >(stmts[j]), j);
                    if (!target.has_value()) {
                        continue;
                    }
                    if (*target == join_idx) {
                        if (!explicit_term_source_idx.has_value()) {
                            explicit_term_source_idx = j;
                        }
                        break;
                    }
                    saw_non_join_forward_target = true;
                }

                if (saw_non_join_forward_target) {
                    return std::nullopt;
                }

                if (explicit_term_source_idx.has_value()) {
                    ArmDiscovery arm;
                    arm.entry_idx          = entry_idx;
                    arm.end_exclusive_idx  = *explicit_term_source_idx;
                    arm.join_idx           = join_idx;
                    arm.explicit_join_goto = true;
                    return arm;
                }

                ArmDiscovery arm;
                arm.entry_idx          = entry_idx;
                arm.end_exclusive_idx  = join_idx;
                arm.join_idx           = join_idx;
                arm.explicit_join_goto = false;
                return arm;
            }

            static std::vector< clang::Stmt * >
            buildArmBody(const std::vector< clang::Stmt * > &stmts, const ArmDiscovery &arm) {
                std::vector< clang::Stmt * > body;
                if (arm.entry_idx >= stmts.size()
                    || arm.end_exclusive_idx > stmts.size())
                {
                    return body;
                }
                auto *entry_label_stmt =
                    llvm::dyn_cast< clang::LabelStmt >(stmts[arm.entry_idx]);
                if (entry_label_stmt == nullptr) {
                    return body;
                }

                if (!(arm.explicit_join_goto && arm.end_exclusive_idx == arm.entry_idx)
                    && !isNullStmt(entry_label_stmt->getSubStmt()))
                {
                    body.push_back(entry_label_stmt->getSubStmt());
                }
                if (arm.end_exclusive_idx > arm.entry_idx + 1U) {
                    appendRange(body, stmts, arm.entry_idx + 1U, arm.end_exclusive_idx - 1U);
                }
                return body;
            }

            static bool rewriteTwoArmSharedJoinRegion(
                clang::ASTContext &ctx, std::vector< clang::Stmt * > &stmts,
                std::size_t region_begin, std::size_t join_idx, const IfHead &head,
                const ArmDiscovery &then_arm, const ArmDiscovery &else_arm
            ) {
                if (join_idx >= stmts.size() || region_begin >= join_idx) {
                    return false;
                }
                auto *join_label_stmt = llvm::dyn_cast< clang::LabelStmt >(stmts[join_idx]);
                if (join_label_stmt == nullptr) {
                    return false;
                }

                auto then_body = buildArmBody(stmts, then_arm);
                auto else_body = buildArmBody(stmts, else_arm);
                auto *new_then = ensureNonEmptyCompound(
                    ctx, std::move(then_body), head.if_stmt->getBeginLoc(),
                    head.if_stmt->getEndLoc()
                );
                auto *new_else = ensureNonEmptyCompound(
                    ctx, std::move(else_body), head.if_stmt->getBeginLoc(),
                    head.if_stmt->getEndLoc()
                );
                auto *new_if = clang::IfStmt::Create(
                    ctx, head.if_stmt->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr,
                    nullptr, head.if_stmt->getCond(), head.if_stmt->getLParenLoc(),
                    new_then->getBeginLoc(), new_then, new_else->getBeginLoc(), new_else
                );

                std::vector< clang::Stmt * > rewritten;
                rewritten.reserve(stmts.size());
                if (head.if_label_wrapper != nullptr) {
                    head.if_label_wrapper->setSubStmt(new_if);
                    appendRange(rewritten, stmts, 0U, region_begin);
                } else {
                    appendRange(rewritten, stmts, 0U, region_begin - 1U);
                    rewritten.push_back(new_if);
                }
                appendRange(rewritten, stmts, join_idx, stmts.size() - 1U);
                stmts = std::move(rewritten);
                return true;
            }

            // Merge a run of consecutive "if (cN) goto L;" statements (all pointing to
            // the same label, no else arm, no label wrapper) into a single
            // "if (c0 || c1 || ... || cN) goto L;". The last if in the run may also
            // carry an "else goto M;" arm, which is preserved on the merged if.
            //
            // Unlike detectIfHead, this does NOT require the target label to exist in
            // the local label_index — the goto target may be outside the current
            // compound (e.g. jumping out of a loop), which is the common case for
            // multi-guard patterns in decompiled P-Code.
            //
            // Returns true and mutates stmts on success.
            static bool mergeConsecutiveSameTargetIfs(
                clang::ASTContext &ctx, std::vector< clang::Stmt * > &stmts, std::size_t i
            ) {
                // stmts[i] must be a plain "if (c) goto L;" with no else arm
                auto *if0 = llvm::dyn_cast< clang::IfStmt >(stmts[i]);
                if (if0 == nullptr) {
                    return false;
                }
                auto *then_goto0 =
                    llvm::dyn_cast_or_null< clang::GotoStmt >(if0->getThen());
                if (then_goto0 == nullptr || if0->getElse() != nullptr) {
                    return false;
                }
                const clang::LabelDecl *target = then_goto0->getLabel();

                // Collect subsequent ifs with the same then-target.
                // Interior ifs must be plain (no else); the last one may have an
                // "else goto M;" (two-sided), which terminates the run.
                std::vector< clang::IfStmt * > ifs;
                ifs.push_back(if0);
                const clang::GotoStmt *terminator_else_goto = nullptr;
                for (std::size_t j = i + 1U; j < stmts.size(); ++j) {
                    if (llvm::isa< clang::LabelStmt >(stmts[j])) {
                        break; // label is an external entry point — cannot merge across it
                    }
                    auto *next_if = llvm::dyn_cast< clang::IfStmt >(stmts[j]);
                    if (next_if == nullptr) {
                        break; // non-if, non-label: stop
                    }
                    auto *next_then =
                        llvm::dyn_cast_or_null< clang::GotoStmt >(next_if->getThen());
                    if (next_then == nullptr || next_then->getLabel() != target) {
                        break; // different then-target
                    }
                    auto *next_else =
                        llvm::dyn_cast_or_null< clang::GotoStmt >(next_if->getElse());
                    if (next_if->getElse() != nullptr && next_else == nullptr) {
                        break; // non-goto else arm — cannot absorb safely
                    }
                    ifs.push_back(next_if);
                    if (next_else != nullptr) {
                        terminator_else_goto = next_else;
                        break; // two-sided if terminates the run
                    }
                }
                if (ifs.size() < 2U) {
                    return false;
                }

                // Build combined condition: c0 || c1 || ... || c_{N-1}
                const auto loc        = if0->getIfLoc();
                clang::Expr *combined = ensureRValue(ctx, if0->getCond());
                for (std::size_t k = 1U; k < ifs.size(); ++k) {
                    auto *rhs = ensureRValue(ctx, ifs[k]->getCond());
                    combined  = clang::BinaryOperator::Create(
                        ctx, combined, rhs, clang::BO_LOr, ctx.IntTy, clang::VK_PRValue,
                        clang::OK_Ordinary, loc, clang::FPOptionsOverride()
                    );
                }

                // Build merged then-goto and optional else-goto
                auto *merged_goto = new (ctx) clang::GotoStmt(
                    const_cast< clang::LabelDecl * >(target), loc, loc
                );
                clang::Stmt *else_stmt = nullptr;
                if (terminator_else_goto != nullptr) {
                    else_stmt = new (ctx) clang::GotoStmt(
                        terminator_else_goto->getLabel(),
                        loc, loc
                    );
                }
                auto *merged_if = clang::IfStmt::Create(
                    ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr, combined,
                    if0->getLParenLoc(), merged_goto->getBeginLoc(), merged_goto,
                    else_stmt != nullptr ? else_stmt->getBeginLoc() : clang::SourceLocation(),
                    else_stmt
                );

                // Replace stmts[i .. i+ifs.size()-1] with the single merged if
                std::vector< clang::Stmt * > rewritten;
                rewritten.reserve(stmts.size() - ifs.size() + 1U);
                appendRange(rewritten, stmts, 0U, i - 1U); // safe at i==0: appendRange guard
                rewritten.push_back(merged_if);
                appendRange(rewritten, stmts, i + ifs.size(), stmts.size() - 1U);
                stmts = std::move(rewritten);
                return true;
            }

            // Remove a redundant "else goto L_b;" arm when L_b: is the very next statement.
            // Pattern:  if (c) goto L_a; else goto L_b;
            //           L_b: sub_stmt...
            // Becomes:  if (c) goto L_a;
            //           sub_stmt...            ← L_b inlined if it has no other incoming gotos
            //   — or if L_b has other gotos —
            //           L_b: sub_stmt...       ← label kept
            //
            // After stripping the else arm, L_b may lose its only incoming goto, becoming
            // an orphan.  Inlining it here lets mergeConsecutiveSameTargetIfs see the
            // sub-statement (typically another "if (c) goto L_a;") on the very next sweep.
            static bool eliminateRedundantElseGoto(
                clang::ASTContext &ctx, std::vector< clang::Stmt * > &stmts, std::size_t i
            ) {
                auto *if_stmt = llvm::dyn_cast< clang::IfStmt >(stmts[i]);
                if (if_stmt == nullptr || if_stmt->getElse() == nullptr) {
                    return false;
                }
                auto *else_goto =
                    llvm::dyn_cast< clang::GotoStmt >(if_stmt->getElse());
                if (else_goto == nullptr) {
                    return false;
                }
                if (i + 1U >= stmts.size()) {
                    return false;
                }
                auto *next_label = llvm::dyn_cast< clang::LabelStmt >(stmts[i + 1U]);
                if (next_label == nullptr) {
                    return false;
                }
                if (next_label->getDecl() != else_goto->getLabel()) {
                    return false;
                }
                // The else arm falls through directly to its target — it is a no-op.
                auto *new_if = clang::IfStmt::Create(
                    ctx, if_stmt->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr, nullptr,
                    if_stmt->getCond(), if_stmt->getLParenLoc(),
                    if_stmt->getThen()->getBeginLoc(), if_stmt->getThen(),
                    clang::SourceLocation(), nullptr
                );
                stmts[i] = new_if;

                // After stripping the else-goto, check whether L_b is now an orphan
                // (no remaining goto in the compound targets it).  If so, inline it:
                // replace LabelStmt(L_b, sub) at stmts[i+1] with sub directly.
                const clang::LabelDecl *lb = next_label->getDecl();
                bool lb_still_targeted     = false;
                std::vector< const clang::LabelDecl * > tmp;
                for (std::size_t k = 0; k < stmts.size(); ++k) {
                    tmp.clear();
                    detail::collectGotoTargets(stmts[k], tmp);
                    for (const auto *t : tmp) {
                        if (t == lb) {
                            lb_still_targeted = true;
                            break;
                        }
                    }
                    if (lb_still_targeted) {
                        break;
                    }
                }
                if (!lb_still_targeted) {
                    stmts[i + 1U] = next_label->getSubStmt();
                }
                return true;
            }

            static clang::CompoundStmt *rewriteCompoundRegions(
                clang::ASTContext &ctx, clang::CompoundStmt *compound, unsigned &rewrites,
                bool &changed
            ) {
                if (compound == nullptr || compound->size() < 3U) {
                    changed = false;
                    return compound;
                }

                std::vector< clang::Stmt * > stmts(
                    compound->body_begin(), compound->body_end()
                );
                changed = false;

                for (std::size_t i = 0; i < stmts.size(); ++i) {
                    // Merge consecutive same-target ifs BEFORE detectIfHead.
                    // This handles cases where the goto target is outside the current
                    // compound (e.g. breaking out of a loop), so detectIfHead would
                    // return nullopt and never reach this check otherwise.
                    if (mergeConsecutiveSameTargetIfs(ctx, stmts, i)) {
                        changed = true;
                        ++rewrites;
                        i = 0;
                        continue;
                    }

                    // After merging, the combined if may have an else-goto arm that
                    // targets the very next statement — remove it as a no-op fallthrough.
                    if (eliminateRedundantElseGoto(ctx, stmts, i)) {
                        changed = true;
                        ++rewrites;
                        i = 0;
                        continue;
                    }

                    auto label_index = topLevelLabelIndex(makeCompound(ctx, stmts));
                    auto maybe_head  = detectIfHead(stmts, label_index, i);
                    if (!maybe_head.has_value()) {
                        continue;
                    }
                    const auto head = *maybe_head;

                    const std::size_t region_begin = i;
                    auto goto_sources_by_target    = buildGotoSourcesByTarget(stmts);

                    if (head.then_is_goto && head.else_is_goto) {
                        const std::size_t then_entry_idx = head.then_target_idx;
                        const std::size_t else_entry_idx = head.else_target_idx;
                        if (then_entry_idx == else_entry_idx
                            || then_entry_idx <= region_begin
                            || else_entry_idx <= region_begin)
                        {
                            continue;
                        }

                        const std::size_t early_entry_idx =
                            std::min(then_entry_idx, else_entry_idx);
                        const std::size_t late_entry_idx =
                            std::max(then_entry_idx, else_entry_idx);
                        auto early_arm = discoverArmWithExplicitJoin(
                            stmts, label_index, early_entry_idx, late_entry_idx
                        );
                        if (!early_arm.has_value()) {
                            continue;
                        }
                        if (early_arm->join_idx <= late_entry_idx
                            || early_arm->join_idx >= stmts.size())
                        {
                            continue;
                        }
                        auto late_arm = discoverArmToKnownJoin(
                            stmts, label_index, late_entry_idx, early_arm->join_idx
                        );
                        if (!late_arm.has_value()) {
                            continue;
                        }

                        const std::size_t join_idx = early_arm->join_idx;
                        if (early_arm->end_exclusive_idx > late_entry_idx
                            || late_arm->end_exclusive_idx > join_idx)
                        {
                            continue;
                        }
                        if (hasExternalJumpIntoRegionMiddle(
                                stmts, goto_sources_by_target, region_begin, join_idx
                            ))
                        {
                            continue;
                        }

                        const ArmDiscovery &then_arm =
                            (then_entry_idx == early_entry_idx) ? *early_arm : *late_arm;
                        const ArmDiscovery &else_arm =
                            (else_entry_idx == early_entry_idx) ? *early_arm : *late_arm;
                        if (rewriteTwoArmSharedJoinRegion(
                                ctx, stmts, region_begin, join_idx, head, then_arm, else_arm
                            ))
                        {
                            changed = true;
                            ++rewrites;
                            i = 0;
                            continue;
                        }
                    }

                    if (!head.has_single_goto_arm) {
                        continue;
                    }

                    const std::size_t target_idx = head.single_target_idx;
                    if (target_idx >= stmts.size()) {
                        continue;
                    }

                    // Two-sided region with split goto
                    std::optional< std::size_t > split_goto_idx;
                    std::optional< std::size_t > join_idx;
                    for (std::size_t j = region_begin + 1U; j < target_idx; ++j) {
                        auto *split_goto = llvm::dyn_cast< clang::GotoStmt >(stmts[j]);
                        if (split_goto == nullptr
                            || !label_index.contains(split_goto->getLabel()))
                        {
                            continue;
                        }
                        std::size_t candidate_join_idx =
                            label_index.at(split_goto->getLabel());
                        if (candidate_join_idx > target_idx) {
                            split_goto_idx = j;
                            join_idx       = candidate_join_idx;
                            break;
                        }
                    }

                    if (split_goto_idx.has_value() && join_idx.has_value()) {
                        const std::size_t region_end = *join_idx;
                        if (!hasExternalJumpIntoRegionMiddle(
                                stmts, goto_sources_by_target, region_begin, region_end
                            )
                            && rewriteTwoSidedRegion(
                                ctx, stmts, region_begin, target_idx, *split_goto_idx,
                                *join_idx, head, goto_sources_by_target
                            ))
                        {
                            changed = true;
                            ++rewrites;
                            i = 0;
                            continue;
                        }
                    }

                    // Single-sided skip region
                    const std::size_t region_end = target_idx;
                    if (hasExternalJumpIntoRegionMiddle(
                            stmts, goto_sources_by_target, region_begin, region_end
                        ))
                    {
                        continue;
                    }
                    if (rewriteSingleSidedRegion(ctx, stmts, region_begin, region_end, head)) {
                        changed = true;
                        ++rewrites;
                        i = 0;
                    }
                }

                if (!changed) {
                    return compound;
                }

                return makeCompound(
                    ctx, stmts, compound->getLBracLoc(), compound->getRBracLoc()
                );
            }

            clang::Stmt *
            processStmt(clang::ASTContext &ctx, clang::Stmt *stmt, unsigned &rewrites) {
                if (stmt == nullptr) {
                    return nullptr;
                }
                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                    bool changed         = false;
                    auto *flat_rewritten = rewriteCompoundRegions(ctx, cs, rewrites, changed);
                    cs                   = flat_rewritten;
                    std::vector< clang::Stmt * > children;
                    children.reserve(cs->size());
                    for (auto *child : cs->body()) {
                        auto *new_child = processStmt(ctx, child, rewrites);
                        children.push_back(new_child != nullptr ? new_child : child);
                        changed = changed || (new_child != nullptr && new_child != child);
                    }
                    if (!changed) {
                        return cs;
                    }
                    return makeCompound(ctx, children, cs->getLBracLoc(), cs->getRBracLoc());
                }

                if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                    auto *new_body = processStmt(ctx, ws->getBody(), rewrites);
                    if (new_body == ws->getBody() || new_body == nullptr) {
                        return ws;
                    }
                    return clang::WhileStmt::Create(
                        ctx, nullptr, ws->getCond(), new_body, ws->getWhileLoc(),
                        ws->getLParenLoc(), ws->getRParenLoc()
                    );
                }

                if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                    auto *new_body = processStmt(ctx, ds->getBody(), rewrites);
                    if (new_body == ds->getBody() || new_body == nullptr) {
                        return ds;
                    }
                    return new (ctx) clang::DoStmt(
                        new_body, ds->getCond(), ds->getDoLoc(), ds->getWhileLoc(),
                        ds->getRParenLoc()
                    );
                }

                if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                    auto *new_body = processStmt(ctx, fs->getBody(), rewrites);
                    if (new_body == fs->getBody() || new_body == nullptr) {
                        return fs;
                    }
                    return new (ctx) clang::ForStmt(
                        ctx, fs->getInit(), fs->getCond(), nullptr, fs->getInc(), new_body,
                        fs->getForLoc(), fs->getLParenLoc(), fs->getRParenLoc()
                    );
                }

                if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                    auto *new_then = processStmt(ctx, is->getThen(), rewrites);
                    auto *new_else = is->getElse() != nullptr
                        ? processStmt(ctx, is->getElse(), rewrites)
                        : nullptr;
                    if ((new_then == nullptr || new_then == is->getThen())
                        && (new_else == nullptr || new_else == is->getElse()))
                    {
                        return is;
                    }
                    if (new_then == nullptr) {
                        new_then = is->getThen();
                    }
                    return clang::IfStmt::Create(
                        ctx, is->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr, nullptr,
                        is->getCond(), is->getLParenLoc(), new_then->getBeginLoc(), new_then,
                        new_else != nullptr ? new_else->getBeginLoc() : clang::SourceLocation(),
                        new_else
                    );
                }

                // Recurse into LabelStmt sub-statements so that WhileStmts and other
                // compound-bearing nodes wrapped in a CFG-block label are still processed.
                if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                    auto *new_sub = processStmt(ctx, ls->getSubStmt(), rewrites);
                    if (new_sub == nullptr || new_sub == ls->getSubStmt()) {
                        return ls;
                    }
                    return new (ctx) clang::LabelStmt(ls->getIdentLoc(), ls->getDecl(), new_sub);
                }

                return stmt;
            }
        };

    } // anonymous namespace

    namespace detail {

        void addConditionalStructurizePass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< ConditionalStructurizePass >(state));
        }

        void addIfElseRegionFormationPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< IfElseRegionFormationPass >(state));
        }

        void addConditionalPasses(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< ConditionalStructurizePass >(state));
            pm.add_pass(std::make_unique< IfElseRegionFormationPass >(state));
        }

    } // namespace detail

} // namespace patchestry::ast
