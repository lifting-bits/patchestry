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
                    if (body == nullptr || body->size() < 3U) {
                        continue;
                    }

                    std::vector< clang::Stmt * > stmts(body->body_begin(), body->body_end());
                    auto label_index = topLevelLabelIndex(body);
                    bool changed     = false;

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
                                if (skip_label_idx > i + 1 && skip_label_idx < stmts.size()
                                    && !containsLabelInRange(stmts, i + 1, skip_label_idx - 1))
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
                                            const_cast< clang::Expr * >(if_stmt->getCond())
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
                                    func->setBody(body);
                                    label_index = topLevelLabelIndex(body);
                                    changed     = true;
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
                                func->setBody(body);
                                label_index = topLevelLabelIndex(body);
                                changed     = true;
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
                            func->setBody(body);
                            label_index = topLevelLabelIndex(body);
                            changed     = true;
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
                        func->setBody(body);
                        label_index = topLevelLabelIndex(body);
                        changed     = true;
                        ++state.conditional_structurized;
                        i = 0;
                    }

                    if (changed) {
                        (void) changed;
                    }
                }

                runCfgExtractPass(state, ctx, options);
                return true;
            }

          private:
            PipelineState &state;
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

                return stmt;
            }
        };

        // =========================================================================
        // IfGotoChainMergePass
        // =========================================================================

        // Merges consecutive chains of the form:
        //   if (C0) goto COMMON; else goto L1;
        //   L1: if (C1) goto COMMON; else goto L2;
        //   L2: if (C2) goto COMMON; else goto FINAL;
        // into:
        //   if (C0 || C1 || C2) goto COMMON; else goto FINAL;
        //
        // L1, L2 are absorbed when each has exactly one goto reference (the chain
        // link's else-goto).  This pattern commonly arises from Ghidra's serialization
        // of sentinel checks (e.g. getopt returning '?' or ':') that all divert to the
        // same exit label before reaching the main switch.

        class IfGotoChainMergePass final : public ASTPass
        {
          public:
            explicit IfGotoChainMergePass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "IfGotoChainMergePass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                state.if_goto_chains_merged = 0;

                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }
                    auto *body = llvm::dyn_cast< clang::CompoundStmt >(func->getBody());
                    if (body == nullptr) {
                        continue;
                    }

                    // Count all goto targets once per function before any merging.
                    std::vector< const clang::LabelDecl * > all_targets;
                    collectGotoTargets(body, all_targets);
                    RefCountMap ref_count;
                    for (auto *lbl : all_targets) {
                        ++ref_count[lbl];
                    }

                    auto *new_body = processStmt(ctx, body, ref_count);
                    if (new_body != body) {
                        func->setBody(new_body);
                    }
                }

                return true;
            }

          private:
            PipelineState &state;
            using RefCountMap = std::unordered_map< const clang::LabelDecl *, unsigned >;

            // Returns true when `stmt` has the shape: if (cond) goto A; else goto B;
            static bool isIfGoto(const clang::Stmt *stmt) {
                const auto *is = llvm::dyn_cast_or_null< clang::IfStmt >(stmt);
                return is != nullptr
                    && llvm::isa_and_nonnull< clang::GotoStmt >(is->getThen())
                    && llvm::isa_and_nonnull< clang::GotoStmt >(is->getElse());
            }

            // Builds a left-associative logical-OR of all expressions in `conds`.
            static clang::Expr *buildOrChain(
                clang::ASTContext &ctx,
                const std::vector< clang::Expr * > &conds,
                clang::SourceLocation loc
            ) {
                clang::Expr *result = ensureRValue(ctx, conds[0]);
                for (std::size_t k = 1; k < conds.size(); ++k) {
                    result = clang::BinaryOperator::Create(
                        ctx, result, ensureRValue(ctx, conds[k]), clang::BO_LOr, ctx.IntTy,
                        clang::VK_PRValue, clang::OK_Ordinary, loc, clang::FPOptionsOverride()
                    );
                }
                return result;
            }

            clang::CompoundStmt *processCompound(
                clang::ASTContext &ctx,
                clang::CompoundStmt *compound,
                const RefCountMap &ref_count
            ) {
                std::vector< clang::Stmt * > stmts(
                    compound->body_begin(), compound->body_end()
                );
                bool changed = false;

                for (std::size_t i = 0; i < stmts.size(); ++i) {
                    // The chain head: a bare IfStmt or a LabelStmt wrapping one.
                    auto *head_if         = llvm::dyn_cast< clang::IfStmt >(stmts[i]);
                    auto *head_label_stmt = static_cast< clang::LabelStmt * >(nullptr);
                    if (head_if == nullptr) {
                        head_label_stmt = llvm::dyn_cast< clang::LabelStmt >(stmts[i]);
                        if (head_label_stmt != nullptr) {
                            head_if =
                                llvm::dyn_cast< clang::IfStmt >(head_label_stmt->getSubStmt());
                        }
                    }
                    if (head_if == nullptr || !isIfGoto(head_if)) {
                        continue;
                    }

                    auto *head_then = llvm::cast< clang::GotoStmt >(head_if->getThen());
                    auto *head_else = llvm::cast< clang::GotoStmt >(head_if->getElse());
                    const clang::LabelDecl *common_target = head_then->getLabel();

                    // Extend the chain forward as long as the pattern holds.
                    std::vector< clang::Expr * > conds;
                    conds.push_back(head_if->getCond());
                    auto *final_else   = head_else;
                    std::size_t chain_end = i; // stmts[i+1..chain_end] are absorbed

                    while (chain_end + 1 < stmts.size()) {
                        auto *next_lbl =
                            llvm::dyn_cast< clang::LabelStmt >(stmts[chain_end + 1]);
                        // The current tail's else-goto must point to this label.
                        if (next_lbl == nullptr
                            || next_lbl->getDecl() != final_else->getLabel())
                        {
                            break;
                        }
                        // Only absorb when no other goto references this label.
                        auto ref_it = ref_count.find(next_lbl->getDecl());
                        if (ref_it == ref_count.end() || ref_it->second != 1) {
                            break;
                        }
                        // The label's sub-stmt must be another if-goto with same then-target.
                        auto *link_if =
                            llvm::dyn_cast_or_null< clang::IfStmt >(next_lbl->getSubStmt());
                        if (link_if == nullptr || !isIfGoto(link_if)) {
                            break;
                        }
                        auto *link_then = llvm::cast< clang::GotoStmt >(link_if->getThen());
                        if (link_then->getLabel() != common_target) {
                            break;
                        }

                        conds.push_back(link_if->getCond());
                        final_else = llvm::cast< clang::GotoStmt >(link_if->getElse());
                        ++chain_end;
                    }

                    if (conds.size() < 2) {
                        continue; // nothing to merge
                    }

                    auto loc = head_if->getIfLoc();
                    auto *merged_cond = buildOrChain(ctx, conds, loc);
                    auto *merged_then = new (ctx) clang::GotoStmt(
                        const_cast< clang::LabelDecl * >(common_target), loc,
                        head_then->getLabelLoc()
                    );
                    auto *merged_else_stmt = new (ctx) clang::GotoStmt(
                        final_else->getLabel(), loc, final_else->getLabelLoc()
                    );
                    auto *merged_if = clang::IfStmt::Create(
                        ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                        merged_cond, loc, loc, merged_then, loc, merged_else_stmt
                    );

                    // Preserve any label wrapper on the chain head.
                    clang::Stmt *replacement = merged_if;
                    if (head_label_stmt != nullptr) {
                        replacement = new (ctx) clang::LabelStmt(
                            head_label_stmt->getIdentLoc(), head_label_stmt->getDecl(), merged_if
                        );
                    }

                    // Rebuild stmts: [i] → replacement, [i+1..chain_end] absorbed.
                    std::vector< clang::Stmt * > new_stmts;
                    new_stmts.reserve(stmts.size() - (chain_end - i));
                    for (std::size_t k = 0; k < stmts.size(); ++k) {
                        if (k == i) {
                            new_stmts.push_back(replacement);
                        } else if (k > i && k <= chain_end) {
                            // absorbed — skip
                        } else {
                            new_stmts.push_back(stmts[k]);
                        }
                    }

                    state.if_goto_chains_merged += static_cast< unsigned >(chain_end - i);
                    stmts   = std::move(new_stmts);
                    changed = true;
                }

                // Recurse into each statement (using the possibly-updated stmts list).
                std::vector< clang::Stmt * > final_stmts;
                final_stmts.reserve(stmts.size());
                bool nested_changed = false;
                for (auto *stmt : stmts) {
                    auto *processed = processStmt(ctx, stmt, ref_count);
                    final_stmts.push_back(processed);
                    nested_changed |= (processed != stmt);
                }

                if (!changed && !nested_changed) {
                    return compound;
                }
                return makeCompound(
                    ctx, final_stmts, compound->getLBracLoc(), compound->getRBracLoc()
                );
            }

            clang::Stmt *processStmt(
                clang::ASTContext &ctx, clang::Stmt *stmt, const RefCountMap &ref_count
            ) {
                if (stmt == nullptr) {
                    return nullptr;
                }
                if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                    return processCompound(ctx, compound, ref_count);
                }
                if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                    auto *new_body = processStmt(ctx, ws->getBody(), ref_count);
                    if (new_body == ws->getBody()) {
                        return stmt;
                    }
                    return clang::WhileStmt::Create(
                        ctx, nullptr, ws->getCond(), new_body, ws->getWhileLoc(),
                        ws->getLParenLoc(), ws->getRParenLoc()
                    );
                }
                if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                    auto *new_body = processStmt(ctx, fs->getBody(), ref_count);
                    if (new_body == fs->getBody()) {
                        return stmt;
                    }
                    return new (ctx) clang::ForStmt(
                        ctx, fs->getInit(), fs->getCond(), nullptr, fs->getInc(), new_body,
                        fs->getForLoc(), fs->getLParenLoc(), fs->getRParenLoc()
                    );
                }
                if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                    auto *new_then = processStmt(ctx, is->getThen(), ref_count);
                    auto *new_else = processStmt(ctx, is->getElse(), ref_count);
                    if (new_then == is->getThen() && new_else == is->getElse()) {
                        return stmt;
                    }
                    if (new_then == nullptr) {
                        new_then = new (ctx) clang::NullStmt(is->getIfLoc(), false);
                    }
                    return clang::IfStmt::Create(
                        ctx, is->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr, nullptr,
                        is->getCond(), is->getLParenLoc(), new_then->getBeginLoc(), new_then,
                        new_else != nullptr ? new_else->getBeginLoc() : clang::SourceLocation(),
                        new_else
                    );
                }
                if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                    auto *new_sub = processStmt(ctx, ls->getSubStmt(), ref_count);
                    if (new_sub == ls->getSubStmt()) {
                        return stmt;
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

        void addIfGotoChainMergePass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< IfGotoChainMergePass >(state));
        }

        void addConditionalPasses(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< ConditionalStructurizePass >(state));
            pm.add_pass(std::make_unique< IfElseRegionFormationPass >(state));
        }

    } // namespace detail

} // namespace patchestry::ast
