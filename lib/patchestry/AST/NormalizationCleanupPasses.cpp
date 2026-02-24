/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// Cleanup / verification passes:
//   DeadCfgPruningPass          – prune dead code after unconditional terminators
//   AstCleanupPass              – strip unreferenced labels and null statements
//   TrailingJumpElimPass        – remove implicit trailing continue from loop bodies
//   StackCanaryRecognitionPass  – stub; no-op canary reorganisation
//   NoGotoVerificationPass      – verify no goto remains after elimination pipeline
//   CleanupTailExtractionPass   – label and extract function cleanup/return tail sections

#include <string>
#include <unordered_set>
#include <vector>

#include <patchestry/AST/ASTPassManager.hpp>
#include <patchestry/Util/Log.hpp>

#include "NormalizationPipelineInternal.hpp"

namespace patchestry::ast {
    namespace {

        using namespace detail;

        // =========================================================================
        // File-local helpers: dead-code pruning utilities
        // =========================================================================


        static bool areEquivalentTerminators(const clang::Stmt *a, const clang::Stmt *b) {
            if (a == nullptr || b == nullptr) {
                return false;
            }
            if (llvm::isa< clang::BreakStmt >(a) && llvm::isa< clang::BreakStmt >(b)) {
                return true;
            }
            if (llvm::isa< clang::ContinueStmt >(a) && llvm::isa< clang::ContinueStmt >(b)) {
                return true;
            }
            const auto *ga = llvm::dyn_cast< clang::GotoStmt >(a);
            const auto *gb = llvm::dyn_cast< clang::GotoStmt >(b);
            if (ga != nullptr && gb != nullptr) {
                return ga->getLabel() == gb->getLabel();
            }
            const auto *ra = llvm::dyn_cast< clang::ReturnStmt >(a);
            const auto *rb = llvm::dyn_cast< clang::ReturnStmt >(b);
            if (ra != nullptr && rb != nullptr) {
                return ra->getRetValue() == nullptr && rb->getRetValue() == nullptr;
            }
            return false;
        }

        // Forward declaration — pruneCompound and pruneStmt are mutually recursive.
        static clang::Stmt *pruneStmt(
            clang::ASTContext &ctx, clang::Stmt *stmt, unsigned &pruned
        );

        static clang::CompoundStmt *pruneCompound(
            clang::ASTContext &ctx, clang::CompoundStmt *cs, unsigned &pruned
        ) {
            std::vector< clang::Stmt * > result;
            result.reserve(cs->size());
            bool after_terminator = false;
            bool changed          = false;

            for (auto *stmt : cs->body()) {
                if (stmt == nullptr) {
                    continue;
                }

                if (after_terminator) {
                    if (llvm::isa< clang::LabelStmt >(stmt)) {
                        after_terminator = false;
                    } else {
                        ++pruned;
                        changed = true;
                        continue;
                    }
                }

                auto *simplified = pruneStmt(ctx, stmt, pruned);
                if (simplified != stmt) {
                    changed = true;
                }
                if (simplified == nullptr) {
                    continue;
                }

                result.push_back(simplified);

                if (isUnconditionalTerminator(simplified)) {
                    after_terminator = true;
                }
            }

            if (!changed && result.size() == cs->size()) {
                return cs;
            }
            return makeCompound(ctx, result, cs->getLBracLoc(), cs->getRBracLoc());
        }

        static clang::Stmt *pruneStmt(
            clang::ASTContext &ctx, clang::Stmt *stmt, unsigned &pruned
        ) {
            if (stmt == nullptr) {
                return nullptr;
            }

            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                return pruneCompound(ctx, cs, pruned);
            }

            if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                const clang::Expr *cond  = is->getCond();
                clang::Stmt *then_branch = is->getThen();
                clang::Stmt *else_branch = is->getElse();

                if (isAlwaysTrue(cond)) {
                    ++pruned;
                    return pruneStmt(ctx, then_branch, pruned);
                }
                if (isAlwaysFalse(cond)) {
                    ++pruned;
                    return else_branch != nullptr
                        ? pruneStmt(ctx, else_branch, pruned)
                        : new (ctx) clang::NullStmt(is->getIfLoc(), false);
                }

                if (else_branch != nullptr
                    && areEquivalentTerminators(then_branch, else_branch))
                {
                    ++pruned;
                    return pruneStmt(ctx, then_branch, pruned);
                }

                if (else_branch != nullptr) {
                    if (llvm::isa< clang::BreakStmt >(then_branch)
                        && llvm::isa< clang::ContinueStmt >(else_branch))
                    {
                        ++pruned;
                        return clang::IfStmt::Create(
                            ctx, is->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr,
                            nullptr, const_cast< clang::Expr * >(cond), is->getLParenLoc(),
                            then_branch->getBeginLoc(), then_branch, clang::SourceLocation(),
                            nullptr
                        );
                    }
                    if (llvm::isa< clang::ContinueStmt >(then_branch)
                        && llvm::isa< clang::BreakStmt >(else_branch))
                    {
                        ++pruned;
                        auto *neg_cond = clang::UnaryOperator::Create(
                            ctx,
                            ensureRValue(ctx, const_cast< clang::Expr * >(cond)),
                            clang::UO_LNot, ctx.IntTy, clang::VK_PRValue, clang::OK_Ordinary,
                            is->getIfLoc(), false, clang::FPOptionsOverride()
                        );
                        return clang::IfStmt::Create(
                            ctx, is->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr,
                            nullptr, neg_cond, is->getLParenLoc(),
                            else_branch->getBeginLoc(), else_branch, clang::SourceLocation(),
                            nullptr
                        );
                    }
                }

                auto *new_then = pruneStmt(ctx, then_branch, pruned);
                auto *new_else =
                    else_branch != nullptr ? pruneStmt(ctx, else_branch, pruned) : nullptr;
                if (new_then == nullptr) {
                    new_then = new (ctx) clang::NullStmt(is->getIfLoc(), false);
                }
                if (new_then == then_branch && new_else == else_branch) {
                    return is;
                }
                return clang::IfStmt::Create(
                    ctx, is->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr, nullptr,
                    const_cast< clang::Expr * >(cond), is->getLParenLoc(),
                    new_then->getBeginLoc(), new_then,
                    new_else != nullptr ? new_else->getBeginLoc() : clang::SourceLocation(),
                    new_else
                );
            }

            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                unsigned local_pruned = 0;
                auto *new_body        = pruneStmt(ctx, ws->getBody(), local_pruned);
                if (local_pruned == 0) {
                    return ws;
                }
                pruned += local_pruned;
                return clang::WhileStmt::Create(
                    ctx, nullptr, ws->getCond(), new_body, ws->getWhileLoc(),
                    ws->getLParenLoc(), ws->getRParenLoc()
                );
            }

            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                unsigned local_pruned = 0;
                auto *new_body        = pruneStmt(ctx, ds->getBody(), local_pruned);
                if (local_pruned == 0) {
                    return ds;
                }
                pruned += local_pruned;
                return new (ctx) clang::DoStmt(
                    new_body, ds->getCond(), ds->getDoLoc(), ds->getWhileLoc(),
                    ds->getRParenLoc()
                );
            }

            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                unsigned local_pruned = 0;
                auto *new_body        = pruneStmt(ctx, fs->getBody(), local_pruned);
                if (local_pruned == 0) {
                    return fs;
                }
                pruned += local_pruned;
                return new (ctx) clang::ForStmt(
                    ctx, fs->getInit(), fs->getCond(), nullptr, fs->getInc(), new_body,
                    fs->getForLoc(), fs->getLParenLoc(), fs->getRParenLoc()
                );
            }

            return stmt;
        }

        // =========================================================================
        // File-local helpers: AST cleanup utilities
        // =========================================================================

        static clang::Stmt *cleanupStmt(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_set< const clang::LabelDecl * > &live_labels
        ) {
            if (stmt == nullptr) {
                return nullptr;
            }

            if (isNullStmt(stmt)) {
                return nullptr;
            }

            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                auto *sub = cleanupStmt(ctx, label->getSubStmt(), live_labels);
                if (!live_labels.contains(label->getDecl())) {
                    return sub;
                }
                if (sub == nullptr) {
                    sub = new (ctx) clang::NullStmt(label->getBeginLoc(), false);
                }
                return new (ctx)
                    clang::LabelStmt(label->getIdentLoc(), label->getDecl(), sub);
            }

            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > flattened;
                for (auto *child : compound->body()) {
                    auto *cleaned = cleanupStmt(ctx, child, live_labels);
                    if (cleaned == nullptr) {
                        continue;
                    }
                    if (auto *nested = llvm::dyn_cast< clang::CompoundStmt >(cleaned)) {
                        for (auto *nested_child : nested->body()) {
                            if (!isNullStmt(nested_child)) {
                                flattened.push_back(nested_child);
                            }
                        }
                    } else {
                        flattened.push_back(cleaned);
                    }
                }
                return makeCompound(
                    ctx, flattened, compound->getLBracLoc(), compound->getRBracLoc()
                );
            }

            if (auto *if_stmt = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                auto *then_stmt = cleanupStmt(ctx, if_stmt->getThen(), live_labels);
                auto *else_stmt = cleanupStmt(ctx, if_stmt->getElse(), live_labels);
                if (then_stmt == nullptr) {
                    then_stmt = new (ctx) clang::NullStmt(if_stmt->getIfLoc(), false);
                }
                return clang::IfStmt::Create(
                    ctx, if_stmt->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr, nullptr,
                    if_stmt->getCond(), if_stmt->getLParenLoc(), then_stmt->getBeginLoc(),
                    then_stmt,
                    else_stmt != nullptr ? else_stmt->getBeginLoc() : clang::SourceLocation(),
                    else_stmt
                );
            }

            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                auto *new_body = cleanupStmt(ctx, ws->getBody(), live_labels);
                if (new_body == ws->getBody()) {
                    return ws;
                }
                if (new_body == nullptr) {
                    new_body = new (ctx) clang::NullStmt(ws->getWhileLoc(), false);
                }
                return clang::WhileStmt::Create(
                    ctx, nullptr, ws->getCond(), new_body, ws->getWhileLoc(),
                    ws->getLParenLoc(), ws->getRParenLoc()
                );
            }

            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                auto *new_body = cleanupStmt(ctx, ds->getBody(), live_labels);
                if (new_body == ds->getBody()) {
                    return ds;
                }
                if (new_body == nullptr) {
                    new_body = new (ctx) clang::NullStmt(ds->getDoLoc(), false);
                }
                return new (ctx) clang::DoStmt(
                    new_body, ds->getCond(), ds->getDoLoc(), ds->getWhileLoc(),
                    ds->getRParenLoc()
                );
            }

            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                auto *new_body = cleanupStmt(ctx, fs->getBody(), live_labels);
                if (new_body == fs->getBody()) {
                    return fs;
                }
                if (new_body == nullptr) {
                    new_body = new (ctx) clang::NullStmt(fs->getForLoc(), false);
                }
                return new (ctx) clang::ForStmt(
                    ctx, fs->getInit(), fs->getCond(), nullptr, fs->getInc(), new_body,
                    fs->getForLoc(), fs->getLParenLoc(), fs->getRParenLoc()
                );
            }

            return stmt;
        }

        // =========================================================================
        // DeadCfgPruningPass
        // =========================================================================

        class DeadCfgPruningPass final : public ASTPass
        {
          public:
            explicit DeadCfgPruningPass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "DeadCfgPruningPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                unsigned local_pruned = 0;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }
                    auto *body = llvm::dyn_cast< clang::CompoundStmt >(func->getBody());
                    if (body == nullptr) {
                        continue;
                    }
                    auto *pruned_body = pruneCompound(ctx, body, local_pruned);
                    if (pruned_body != body) {
                        func->setBody(pruned_body);
                    }
                }

                state.dead_stmts_pruned += local_pruned;

                if (local_pruned > 0) {
                    if (options.verbose) {
                        LOG(DEBUG) << "DeadCfgPruningPass: pruned " << local_pruned
                                   << " dead statement(s)\n";
                    }
                    state.cfg_stale = true;
                    runGotoCanonicalizePass(state, ctx, options);
                    runCfgExtractPass(state, ctx, options);
                }

                return true;
            }

          private:
            PipelineState &state;
        };

        // =========================================================================
        // AstCleanupPass
        // =========================================================================

        class AstCleanupPass final : public ASTPass
        {
          public:
            explicit AstCleanupPass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "AstCleanupPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                LabelUseCollector collector;
                collector.TraverseDecl(ctx.getTranslationUnitDecl());

                bool changed = false;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }

                    auto *cleaned = cleanupStmt(ctx, func->getBody(), collector.live_labels);
                    if (cleaned != nullptr && cleaned != func->getBody()) {
                        func->setBody(cleaned);
                        changed = true;
                    }
                }

                if (changed) {
                    state.cfg_stale = true;
                    runCfgExtractPass(state, ctx, options);
                }
                return true;
            }

          private:
            PipelineState &state;
        };

        // =========================================================================
        // TrailingJumpElimPass
        // =========================================================================

        class TrailingJumpElimPass final : public ASTPass
        {
          public:
            explicit TrailingJumpElimPass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "TrailingJumpElimPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                unsigned eliminated = 0;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }
                    auto *new_body = processStmt(ctx, func->getBody(), eliminated);
                    if (new_body != func->getBody()) {
                        func->setBody(new_body);
                    }
                }

                if (eliminated > 0 && options.verbose) {
                    LOG(DEBUG) << "TrailingJumpElimPass: eliminated " << eliminated
                               << " trailing jump(s)\n";
                }
                return true;
            }

          private:
            [[maybe_unused]] PipelineState &state;

            static clang::Stmt *replaceTrailingContinueWithBreak(
                clang::ASTContext &ctx, clang::Stmt *stmt, unsigned &eliminated
            ) {
                if (stmt == nullptr) {
                    return nullptr;
                }
                if (llvm::isa< clang::ContinueStmt >(stmt)) {
                    ++eliminated;
                    return new (ctx) clang::BreakStmt(stmt->getBeginLoc());
                }
                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                    if (cs->body_empty()) {
                        return cs;
                    }
                    auto *last     = *(cs->body_end() - 1);
                    auto *new_last = replaceTrailingContinueWithBreak(ctx, last, eliminated);
                    if (new_last == last) {
                        return cs;
                    }
                    std::vector< clang::Stmt * > stmts(cs->body_begin(), cs->body_end() - 1);
                    stmts.push_back(new_last);
                    return makeCompound(ctx, stmts, cs->getLBracLoc(), cs->getRBracLoc());
                }
                if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                    auto *new_then =
                        replaceTrailingContinueWithBreak(ctx, is->getThen(), eliminated);
                    auto *new_else = is->getElse() != nullptr
                        ? replaceTrailingContinueWithBreak(ctx, is->getElse(), eliminated)
                        : nullptr;
                    if (new_then == is->getThen() && new_else == is->getElse()) {
                        return is;
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
                return stmt;
            }

            static clang::Stmt *convertSwitchContinueToBreak(
                clang::ASTContext &ctx, clang::SwitchStmt *sw, unsigned &eliminated
            ) {
                auto *sw_body = llvm::dyn_cast_or_null< clang::CompoundStmt >(sw->getBody());
                if (sw_body == nullptr) {
                    return sw;
                }

                bool changed = false;
                std::vector< clang::Stmt * > new_cases;
                new_cases.reserve(sw_body->size());

                for (auto *child : sw_body->body()) {
                    if (auto *cs = llvm::dyn_cast< clang::CaseStmt >(child)) {
                        auto *new_sub =
                            replaceTrailingContinueWithBreak(ctx, cs->getSubStmt(), eliminated);
                        if (new_sub != cs->getSubStmt()) {
                            auto *new_cs = clang::CaseStmt::Create(
                                ctx, cs->getLHS(), cs->getRHS(), cs->getCaseLoc(),
                                cs->getEllipsisLoc(), cs->getColonLoc()
                            );
                            new_cs->setSubStmt(new_sub);
                            new_cases.push_back(new_cs);
                            changed = true;
                        } else {
                            new_cases.push_back(child);
                        }
                    } else if (auto *ds = llvm::dyn_cast< clang::DefaultStmt >(child)) {
                        auto *new_sub =
                            replaceTrailingContinueWithBreak(ctx, ds->getSubStmt(), eliminated);
                        if (new_sub != ds->getSubStmt()) {
                            new_cases.push_back(new (ctx) clang::DefaultStmt(
                                ds->getDefaultLoc(), ds->getColonLoc(), new_sub
                            ));
                            changed = true;
                        } else {
                            new_cases.push_back(child);
                        }
                    } else {
                        new_cases.push_back(child);
                    }
                }

                if (!changed) {
                    return sw;
                }
                auto *new_sw = clang::SwitchStmt::Create(
                    ctx, nullptr, nullptr, sw->getCond(), sw->getLParenLoc(), sw->getRParenLoc()
                );
                new_sw->setBody(
                    makeCompound(ctx, new_cases, sw_body->getLBracLoc(), sw_body->getRBracLoc())
                );
                return new_sw;
            }

            static clang::Stmt *convertLoopBodySwitchContinues(
                clang::ASTContext &ctx, clang::Stmt *body_stmt, unsigned &eliminated
            ) {
                auto *body_cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(body_stmt);
                if (body_cs == nullptr || body_cs->body_empty()) {
                    return body_stmt;
                }
                auto *last = *(body_cs->body_end() - 1);
                auto *sw   = llvm::dyn_cast< clang::SwitchStmt >(last);
                if (sw == nullptr) {
                    return body_stmt;
                }
                auto *new_sw = convertSwitchContinueToBreak(ctx, sw, eliminated);
                if (new_sw == sw) {
                    return body_stmt;
                }
                std::vector< clang::Stmt * > stmts(
                    body_cs->body_begin(), body_cs->body_end() - 1
                );
                stmts.push_back(new_sw);
                return makeCompound(ctx, stmts, body_cs->getLBracLoc(), body_cs->getRBracLoc());
            }

            static clang::CompoundStmt *stripTrailingContinue(
                clang::ASTContext &ctx, clang::Stmt *body_stmt, unsigned &eliminated
            ) {
                auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(body_stmt);
                if (cs == nullptr || cs->body_empty()) {
                    return cs;
                }
                auto *last = *(cs->body_end() - 1);
                if (!llvm::isa< clang::ContinueStmt >(last)) {
                    return cs;
                }
                ++eliminated;
                std::vector< clang::Stmt * > stmts(cs->body_begin(), cs->body_end() - 1);
                return makeCompound(ctx, stmts, cs->getLBracLoc(), cs->getRBracLoc());
            }

            clang::Stmt *processStmt(
                clang::ASTContext &ctx, clang::Stmt *stmt, unsigned &eliminated
            ) {
                if (stmt == nullptr) {
                    return nullptr;
                }

                if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                    auto *trimmed = stripTrailingContinue(ctx, ws->getBody(), eliminated);
                    auto *effective = trimmed != nullptr ? static_cast< clang::Stmt * >(trimmed)
                                                         : ws->getBody();
                    effective      = convertLoopBodySwitchContinues(ctx, effective, eliminated);
                    auto *new_body = processStmt(ctx, effective, eliminated);
                    if (new_body == ws->getBody()) {
                        return ws;
                    }
                    return clang::WhileStmt::Create(
                        ctx, nullptr, ws->getCond(), new_body, ws->getWhileLoc(),
                        ws->getLParenLoc(), ws->getRParenLoc()
                    );
                }

                if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                    auto *trimmed   = stripTrailingContinue(ctx, fs->getBody(), eliminated);
                    auto *effective = trimmed != nullptr ? static_cast< clang::Stmt * >(trimmed)
                                                         : fs->getBody();
                    effective      = convertLoopBodySwitchContinues(ctx, effective, eliminated);
                    auto *new_body = processStmt(ctx, effective, eliminated);
                    if (new_body == fs->getBody()) {
                        return fs;
                    }
                    return new (ctx) clang::ForStmt(
                        ctx, fs->getInit(), fs->getCond(), nullptr, fs->getInc(), new_body,
                        fs->getForLoc(), fs->getLParenLoc(), fs->getRParenLoc()
                    );
                }

                if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                    auto *trimmed = stripTrailingContinue(ctx, ds->getBody(), eliminated);
                    auto *effective = trimmed != nullptr ? static_cast< clang::Stmt * >(trimmed)
                                                         : ds->getBody();
                    effective      = convertLoopBodySwitchContinues(ctx, effective, eliminated);
                    auto *new_body = processStmt(ctx, effective, eliminated);
                    if (new_body == ds->getBody()) {
                        return ds;
                    }
                    return new (ctx) clang::DoStmt(
                        new_body, ds->getCond(), ds->getDoLoc(), ds->getWhileLoc(),
                        ds->getRParenLoc()
                    );
                }

                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                    bool changed = false;
                    std::vector< clang::Stmt * > result;
                    result.reserve(cs->size());
                    for (auto *s : cs->body()) {
                        auto *ns = processStmt(ctx, s, eliminated);
                        if (ns != s) {
                            changed = true;
                        }
                        if (ns != nullptr) {
                            result.push_back(ns);
                        }
                    }
                    if (!changed) {
                        return cs;
                    }
                    return makeCompound(ctx, result, cs->getLBracLoc(), cs->getRBracLoc());
                }

                if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                    auto *new_then = processStmt(ctx, is->getThen(), eliminated);
                    auto *new_else = is->getElse() != nullptr
                        ? processStmt(ctx, is->getElse(), eliminated)
                        : nullptr;
                    if (new_then == is->getThen() && new_else == is->getElse()) {
                        return is;
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

                return stmt;
            }
        };

        // =========================================================================
        // StackCanaryRecognitionPass (stub)
        // =========================================================================

        class StackCanaryRecognitionPass final : public ASTPass
        {
          public:
            explicit StackCanaryRecognitionPass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "StackCanaryRecognitionPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }
                (void) ctx;
                return true;
            }

          private:
            [[maybe_unused]] PipelineState &state;
        };

        // =========================================================================
        // NoGotoVerificationPass
        // =========================================================================

        class NoGotoVerificationPass final : public ASTPass
        {
          public:
            explicit NoGotoVerificationPass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "NoGotoVerificationPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                auto [remaining_gotos, remaining_indirect_gotos, first_violation] =
                    countRemainingGotos(ctx.getTranslationUnitDecl());

                if (remaining_gotos == 0 && remaining_indirect_gotos == 0) {
                    return true;
                }

                std::string message = "verification found "
                    + std::to_string(remaining_gotos) + " goto(s) and "
                    + std::to_string(remaining_indirect_gotos)
                    + " indirect goto(s) after goto-elimination pipeline";
                if (state.used_irreducible_fallback) {
                    message += "; irreducible fallback was used";
                }

                if (first_violation.has_value()) {
                    emitDiagnostic(
                        ctx, *first_violation,
                        options.goto_elimination_strict ? clang::DiagnosticsEngine::Error
                                                        : clang::DiagnosticsEngine::Warning,
                        message
                    );
                } else {
                    emitDiagnostic(
                        ctx, clang::SourceLocation(),
                        options.goto_elimination_strict ? clang::DiagnosticsEngine::Error
                                                        : clang::DiagnosticsEngine::Warning,
                        message
                    );
                }

                if (options.goto_elimination_strict) {
                    LOG(ERROR) << "NoGotoVerificationPass failed in strict mode: " << message
                               << "\n";
                    return false;
                }

                LOG(WARNING) << "NoGotoVerificationPass warning: " << message << "\n";
                return true;
            }

          private:
            PipelineState &state;
        };

    } // anonymous namespace

    namespace detail {

        void runAstCleanupPass(
            PipelineState &state, clang::ASTContext &ctx, const patchestry::Options &options
        ) {
            AstCleanupPass(state).run(ctx, options);
        }

        void addDeadCfgPruningPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< DeadCfgPruningPass >(state));
        }

        void addAstCleanupPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< AstCleanupPass >(state));
        }

        void addTrailingJumpElimPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< TrailingJumpElimPass >(state));
        }

        void addNoGotoVerificationPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< NoGotoVerificationPass >(state));
        }

        void addCleanupPasses(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< DeadCfgPruningPass >(state));
            pm.add_pass(std::make_unique< AstCleanupPass >(state));
            pm.add_pass(std::make_unique< TrailingJumpElimPass >(state));
            pm.add_pass(std::make_unique< NoGotoVerificationPass >(state));
        }

    } // namespace detail

} // namespace patchestry::ast
