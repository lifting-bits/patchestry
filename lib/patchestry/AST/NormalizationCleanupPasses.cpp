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
//   SingleUseTempInliningPass   – inline single-use temporary variables into if conditions
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

                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }

                    auto *cleaned = cleanupStmt(ctx, func->getBody(), collector.live_labels);
                    if (cleaned != nullptr) {
                        func->setBody(cleaned);
                    }
                }

                runCfgExtractPass(state, ctx, options);
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
                    auto *new_body = processStmt(
                        ctx, trimmed != nullptr ? trimmed : ws->getBody(), eliminated
                    );
                    if (new_body == ws->getBody()) {
                        return ws;
                    }
                    return clang::WhileStmt::Create(
                        ctx, nullptr, ws->getCond(), new_body, ws->getWhileLoc(),
                        ws->getLParenLoc(), ws->getRParenLoc()
                    );
                }

                if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                    auto *trimmed = stripTrailingContinue(ctx, ds->getBody(), eliminated);
                    auto *new_body = processStmt(
                        ctx, trimmed != nullptr ? trimmed : ds->getBody(), eliminated
                    );
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

        // =========================================================================
        // SingleUseTempInliningPass
        // =========================================================================

        class SingleUseTempInliningPass final : public ASTPass
        {
          public:
            explicit SingleUseTempInliningPass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "SingleUseTempInliningPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                unsigned local_inlined = 0;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }
                    processFunction(ctx, func, local_inlined);
                }

                state.single_use_temps_inlined += local_inlined;
                if (local_inlined > 0 && options.verbose) {
                    LOG(DEBUG) << "SingleUseTempInliningPass: inlined " << local_inlined
                               << " temporary variable(s)\n";
                }
                return true;
            }

          private:
            PipelineState &state;

            static bool isPureExpr(const clang::Expr *e) {
                if (e == nullptr) {
                    return false;
                }

                struct ImpurityChecker
                    : public clang::RecursiveASTVisitor< ImpurityChecker >
                {
                    bool impure = false;

                    bool VisitCallExpr(clang::CallExpr *) {
                        impure = true;
                        return false;
                    }

                    bool VisitUnaryOperator(clang::UnaryOperator *uo) {
                        if (uo->isIncrementDecrementOp()) {
                            impure = true;
                            return false;
                        }
                        return true;
                    }

                    bool VisitCompoundAssignOperator(clang::CompoundAssignOperator *) {
                        impure = true;
                        return false;
                    }
                } checker;

                checker.TraverseStmt(const_cast< clang::Expr * >(e));
                return !checker.impure;
            }

            static unsigned countUses(clang::Stmt *body, const clang::VarDecl *V) {
                struct UseCounter : public clang::RecursiveASTVisitor< UseCounter >
                {
                    const clang::VarDecl *target = nullptr;
                    unsigned count               = 0;

                    bool VisitDeclRefExpr(clang::DeclRefExpr *dre) {
                        if (dre->getDecl() == target) {
                            ++count;
                        }
                        return true;
                    }
                } counter;

                counter.target = V;
                counter.TraverseStmt(body);
                return counter.count;
            }

            static clang::Expr *substituteInCond(
                clang::ASTContext &ctx, clang::Expr *cond, const clang::VarDecl *V,
                clang::Expr *I
            ) {
                auto *stripped = cond->IgnoreParenImpCasts();

                if (auto *dre = llvm::dyn_cast< clang::DeclRefExpr >(stripped)) {
                    if (dre->getDecl() == V) {
                        return ensureRValue(ctx, I);
                    }
                }

                if (auto *uo = llvm::dyn_cast< clang::UnaryOperator >(stripped)) {
                    if (uo->getOpcode() == clang::UO_LNot) {
                        auto *sub = uo->getSubExpr()->IgnoreParenImpCasts();
                        if (auto *dre = llvm::dyn_cast< clang::DeclRefExpr >(sub)) {
                            if (dre->getDecl() == V) {
                                return clang::UnaryOperator::Create(
                                    ctx, ensureRValue(ctx, I), clang::UO_LNot, ctx.IntTy,
                                    clang::VK_PRValue, clang::OK_Ordinary,
                                    uo->getOperatorLoc(), false, clang::FPOptionsOverride()
                                );
                            }
                        }
                    }
                }

                return nullptr;
            }

            static clang::IfStmt *getNextIfStmt(
                const std::vector< clang::Stmt * > &stmts, std::size_t i
            ) {
                if (i + 1 >= stmts.size()) {
                    return nullptr;
                }
                auto *next = stmts[i + 1];
                if (auto *is = llvm::dyn_cast< clang::IfStmt >(next)) {
                    return is;
                }
                if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(next)) {
                    return llvm::dyn_cast_or_null< clang::IfStmt >(ls->getSubStmt());
                }
                return nullptr;
            }

            clang::Stmt *rewriteChildStmt(
                clang::ASTContext &ctx, clang::Stmt *s, clang::FunctionDecl *func,
                unsigned &inlined
            ) {
                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                    return processCompound(ctx, cs, func, inlined);
                }
                if (auto *is = llvm::dyn_cast< clang::IfStmt >(s)) {
                    auto *then_s   = is->getThen();
                    auto *new_then = rewriteChildStmt(ctx, then_s, func, inlined);
                    auto *else_s   = is->getElse();
                    auto *new_else = else_s != nullptr
                        ? rewriteChildStmt(ctx, else_s, func, inlined)
                        : nullptr;
                    if (new_then == then_s && new_else == else_s) {
                        return s;
                    }
                    return clang::IfStmt::Create(
                        ctx, is->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr, nullptr,
                        is->getCond(), is->getLParenLoc(), new_then->getBeginLoc(), new_then,
                        new_else != nullptr ? new_else->getBeginLoc() : clang::SourceLocation(),
                        new_else
                    );
                }
                if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
                    auto *body     = ws->getBody();
                    auto *new_body = rewriteChildStmt(ctx, body, func, inlined);
                    if (new_body == body) {
                        return s;
                    }
                    return clang::WhileStmt::Create(
                        ctx, nullptr, ws->getCond(), new_body, ws->getWhileLoc(),
                        ws->getLParenLoc(), ws->getRParenLoc()
                    );
                }
                if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
                    auto *body     = fs->getBody();
                    auto *new_body = rewriteChildStmt(ctx, body, func, inlined);
                    if (new_body == body) {
                        return s;
                    }
                    return new (ctx) clang::ForStmt(
                        ctx, fs->getInit(), fs->getCond(), nullptr, fs->getInc(), new_body,
                        fs->getForLoc(), fs->getLParenLoc(), fs->getRParenLoc()
                    );
                }
                if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
                    auto *sub     = ls->getSubStmt();
                    auto *new_sub = rewriteChildStmt(ctx, sub, func, inlined);
                    if (new_sub == sub) {
                        return s;
                    }
                    return new (ctx)
                        clang::LabelStmt(ls->getIdentLoc(), ls->getDecl(), new_sub);
                }
                return s;
            }

            clang::CompoundStmt *processCompound(
                clang::ASTContext &ctx, clang::CompoundStmt *cs, clang::FunctionDecl *func,
                unsigned &inlined
            ) {
                bool child_changed = false;
                std::vector< clang::Stmt * > stmts(cs->body_begin(), cs->body_end());
                for (auto &s : stmts) {
                    auto *new_s = rewriteChildStmt(ctx, s, func, inlined);
                    if (new_s != s) {
                        s             = new_s;
                        child_changed = true;
                    }
                }
                if (child_changed) {
                    cs = makeCompound(ctx, stmts, cs->getLBracLoc(), cs->getRBracLoc());
                }

                stmts.assign(cs->body_begin(), cs->body_end());
                for (std::size_t i = 0; i + 1 < stmts.size(); ++i) {
                    clang::LabelStmt *wrapper_label = nullptr;
                    auto *ds = llvm::dyn_cast< clang::DeclStmt >(stmts[i]);
                    if (ds == nullptr) {
                        if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmts[i])) {
                            if (auto *inner_ds = llvm::dyn_cast_or_null< clang::DeclStmt >(
                                    ls->getSubStmt()
                                ))
                            {
                                ds            = inner_ds;
                                wrapper_label = ls;
                            }
                        }
                    }
                    if (ds == nullptr || !ds->isSingleDecl()) {
                        continue;
                    }
                    auto *V =
                        llvm::dyn_cast_or_null< clang::VarDecl >(ds->getSingleDecl());
                    if (V == nullptr) {
                        continue;
                    }
                    auto *I = V->getInit();
                    if (I == nullptr || !isPureExpr(I)) {
                        continue;
                    }
                    if (countUses(func->getBody(), V) != 1) {
                        continue;
                    }
                    auto *if_stmt = getNextIfStmt(stmts, i);
                    if (if_stmt == nullptr) {
                        continue;
                    }
                    auto *new_cond = substituteInCond(ctx, if_stmt->getCond(), V, I);
                    if (new_cond == nullptr) {
                        continue;
                    }

                    auto *new_if = clang::IfStmt::Create(
                        ctx, if_stmt->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr,
                        nullptr, new_cond, if_stmt->getLParenLoc(),
                        if_stmt->getThen()->getBeginLoc(), if_stmt->getThen(),
                        if_stmt->getElse() != nullptr ? if_stmt->getElse()->getBeginLoc()
                                                      : clang::SourceLocation(),
                        if_stmt->getElse()
                    );

                    auto *next = stmts[i + 1];
                    if (llvm::isa< clang::IfStmt >(next)) {
                        stmts[i + 1] = new_if;
                    } else if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(next)) {
                        stmts[i + 1] = new (ctx)
                            clang::LabelStmt(ls->getIdentLoc(), ls->getDecl(), new_if);
                    }

                    if (wrapper_label != nullptr) {
                        stmts[i] = new (ctx) clang::LabelStmt(
                            wrapper_label->getIdentLoc(), wrapper_label->getDecl(),
                            new (ctx) clang::NullStmt(wrapper_label->getIdentLoc(), false)
                        );
                    } else {
                        stmts.erase(stmts.begin() + static_cast< std::ptrdiff_t >(i));
                    }
                    ++inlined;
                    return makeCompound(ctx, stmts, cs->getLBracLoc(), cs->getRBracLoc());
                }

                return cs;
            }

            void processFunction(
                clang::ASTContext &ctx, clang::FunctionDecl *func, unsigned &inlined
            ) {
                auto *body =
                    llvm::dyn_cast_or_null< clang::CompoundStmt >(func->getBody());
                if (body == nullptr) {
                    return;
                }
                bool any_changed = true;
                while (any_changed) {
                    unsigned before = inlined;
                    auto *new_body  = processCompound(ctx, body, func, inlined);
                    if (new_body != body) {
                        func->setBody(new_body);
                        body = new_body;
                    }
                    any_changed = (inlined > before);
                }
            }
        };

        // =========================================================================
        // CleanupTailExtractionPass
        // =========================================================================

        class CleanupTailExtractionPass final : public ASTPass
        {
          public:
            explicit CleanupTailExtractionPass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "CleanupTailExtractionPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                unsigned extracted = 0;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }

                    auto *body = llvm::dyn_cast< clang::CompoundStmt >(func->getBody());
                    if (body == nullptr || body->size() < 2U) {
                        continue;
                    }

                    std::vector< clang::Stmt * > stmts(body->body_begin(), body->body_end());

                    std::size_t last_ret = stmts.size();
                    for (std::size_t i = stmts.size(); i > 0; --i) {
                        if (llvm::isa< clang::ReturnStmt >(stmts[i - 1])) {
                            last_ret = i - 1;
                            break;
                        }
                    }
                    if (last_ret == stmts.size()) {
                        continue;
                    }

                    std::size_t tail_start = last_ret;
                    while (tail_start > 0 && isCleanupTailStmt(stmts[tail_start - 1])) {
                        --tail_start;
                    }

                    if (tail_start == 0) {
                        continue;
                    }
                    if (tail_start == last_ret) {
                        continue;
                    }
                    if (llvm::isa< clang::LabelStmt >(stmts[tail_start])) {
                        continue;
                    }

                    std::string label_name = "__cleanup_tail_"
                        + std::to_string(state.cleanup_tails_extracted + extracted);
                    auto *label_decl = clang::LabelDecl::Create(
                        ctx, func, stmts[tail_start]->getBeginLoc(),
                        &ctx.Idents.get(label_name)
                    );
                    label_decl->setDeclContext(func);

                    std::vector< clang::Stmt * > tail_stmts(
                        stmts.begin() + static_cast< std::ptrdiff_t >(tail_start),
                        stmts.end()
                    );
                    auto *tail_compound = makeCompound(
                        ctx, tail_stmts, stmts[tail_start]->getBeginLoc(),
                        stmts.back()->getEndLoc()
                    );
                    auto *cleanup_label = new (ctx) clang::LabelStmt(
                        stmts[tail_start]->getBeginLoc(), label_decl, tail_compound
                    );

                    std::vector< clang::Stmt * > new_stmts(
                        stmts.begin(),
                        stmts.begin() + static_cast< std::ptrdiff_t >(tail_start)
                    );
                    new_stmts.push_back(cleanup_label);

                    func->setBody(
                        makeCompound(ctx, new_stmts, body->getLBracLoc(), body->getRBracLoc())
                    );
                    ++extracted;
                }

                state.cleanup_tails_extracted += extracted;

                if (extracted > 0) {
                    if (options.verbose) {
                        LOG(DEBUG) << "CleanupTailExtractionPass: extracted " << extracted
                                   << " cleanup tail(s)\n";
                    }
                    runCfgExtractPass(state, ctx, options);
                }

                return true;
            }

          private:
            PipelineState &state;

            static bool isCleanupTailStmt(const clang::Stmt *stmt) {
                if (stmt == nullptr) {
                    return true;
                }
                if (llvm::isa< clang::ReturnStmt >(stmt)) {
                    return true;
                }
                if (llvm::isa< clang::NullStmt >(stmt)) {
                    return true;
                }
                if (llvm::isa< clang::DeclStmt >(stmt)) {
                    return true;
                }
                if (auto *bin = llvm::dyn_cast< clang::BinaryOperator >(stmt)) {
                    return bin->isAssignmentOp();
                }
                if (llvm::isa< clang::CallExpr >(stmt)) {
                    return true;
                }
                if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                    return isCleanupTailStmt(ls->getSubStmt());
                }
                if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                    if (!isCleanupTailStmt(is->getThen())) {
                        return false;
                    }
                    if (is->getElse() != nullptr && !isCleanupTailStmt(is->getElse())) {
                        return false;
                    }
                    return true;
                }
                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                    for (const auto *child : cs->body()) {
                        if (!isCleanupTailStmt(child)) {
                            return false;
                        }
                    }
                    return true;
                }
                if (const auto *expr = llvm::dyn_cast< clang::Expr >(stmt)) {
                    return llvm::isa< clang::CallExpr >(expr->IgnoreParenImpCasts());
                }
                return false;
            }
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

        void addSingleUseTempInliningPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< SingleUseTempInliningPass >(state));
        }

        void addCleanupTailExtractionPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< CleanupTailExtractionPass >(state));
        }

        void addNoGotoVerificationPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< NoGotoVerificationPass >(state));
        }

        void addCleanupPasses(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< DeadCfgPruningPass >(state));
            pm.add_pass(std::make_unique< AstCleanupPass >(state));
            pm.add_pass(std::make_unique< TrailingJumpElimPass >(state));
            pm.add_pass(std::make_unique< SingleUseTempInliningPass >(state));
            pm.add_pass(std::make_unique< CleanupTailExtractionPass >(state));
            pm.add_pass(std::make_unique< NoGotoVerificationPass >(state));
        }

    } // namespace detail

} // namespace patchestry::ast
