/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// Loop structuring passes:
//   LoopStructurizePass          – do-while from back-edge goto patterns
//   WhileLoopStructurizePass     – while-loop from RPO head/body/back-edge pattern
//   DegenerateLoopUnwrapPass     – unwrap degenerate while(1) with no state changes
//   LoopConditionRecoveryPass    – lift while(1){if(c)break;body} → while(!c){body}
//   DegenerateWhileElimPass      – eliminate while(false) / do{body}while(false)
//   NaturalLoopRecoveryPass      – dominator-analysis loop recovery (while + for)
//   BackedgeLoopStructurizePass  – embedded-exit back-edge goto → while loop

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <patchestry/AST/ASTPassManager.hpp>
#include <patchestry/Util/Log.hpp>

#include "NormalizationPipelineInternal.hpp"

namespace patchestry::ast {
    namespace {

        using namespace detail;

        // =========================================================================
        // File-local helpers used by DegenerateLoopUnwrapPass
        // =========================================================================

        static void collectContinues(
            const clang::Stmt *stmt, std::vector< clang::ContinueStmt * > &result
        ) {
            if (stmt == nullptr) {
                return;
            }
            if (llvm::isa< clang::WhileStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::SwitchStmt >(stmt))
            {
                return;
            }
            if (auto *cont = llvm::dyn_cast< clang::ContinueStmt >(
                    const_cast< clang::Stmt * >(stmt)
                ))
            {
                result.push_back(cont);
                return;
            }
            for (const auto *child : stmt->children()) {
                collectContinues(child, result);
            }
        }

        static void collectLocalDecls(
            const clang::Stmt *stmt, std::unordered_set< const clang::VarDecl * > &locals
        ) {
            if (stmt == nullptr) {
                return;
            }
            if (const auto *ds = llvm::dyn_cast< clang::DeclStmt >(stmt)) {
                for (const auto *decl : ds->decls()) {
                    if (const auto *vd = llvm::dyn_cast< clang::VarDecl >(decl)) {
                        locals.insert(vd);
                    }
                }
            }
            for (const auto *child : stmt->children()) {
                collectLocalDecls(child, locals);
            }
        }

        static bool writesNonLocalVar(
            const clang::Stmt *stmt,
            const std::unordered_set< const clang::VarDecl * > &locals
        ) {
            if (stmt == nullptr) {
                return false;
            }
            if (const auto *binop = llvm::dyn_cast< clang::BinaryOperator >(stmt)) {
                if (binop->isAssignmentOp()) {
                    const clang::Expr *lhs = binop->getLHS()->IgnoreParenImpCasts();
                    if (llvm::isa< clang::UnaryOperator >(lhs)
                        || llvm::isa< clang::MemberExpr >(lhs)
                        || llvm::isa< clang::ArraySubscriptExpr >(lhs))
                    {
                        return true;
                    }
                    if (const auto *dre = llvm::dyn_cast< clang::DeclRefExpr >(lhs)) {
                        if (const auto *vd =
                                llvm::dyn_cast< clang::VarDecl >(dre->getDecl()))
                        {
                            if (!locals.count(vd)) {
                                return true;
                            }
                        }
                    }
                }
            }
            for (const auto *child : stmt->children()) {
                if (writesNonLocalVar(child, locals)) {
                    return true;
                }
            }
            return false;
        }

        static clang::Stmt *replaceContinueWithGoto(
            clang::ASTContext &ctx, clang::Stmt *stmt, clang::LabelDecl *restart_label
        ) {
            if (stmt == nullptr) {
                return nullptr;
            }
            if (llvm::isa< clang::WhileStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::SwitchStmt >(stmt))
            {
                return stmt;
            }
            if (llvm::isa< clang::ContinueStmt >(stmt)) {
                return new (ctx) clang::GotoStmt(
                    restart_label, stmt->getBeginLoc(), stmt->getBeginLoc()
                );
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > new_stmts;
                for (auto *child : cs->body()) {
                    new_stmts.push_back(replaceContinueWithGoto(ctx, child, restart_label));
                }
                return makeCompound(ctx, new_stmts, cs->getLBracLoc(), cs->getRBracLoc());
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(replaceContinueWithGoto(ctx, ls->getSubStmt(), restart_label));
                return ls;
            }
            if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                auto *new_then = replaceContinueWithGoto(ctx, is->getThen(), restart_label);
                clang::Stmt *new_else = is->getElse()
                    ? replaceContinueWithGoto(ctx, is->getElse(), restart_label)
                    : nullptr;
                return clang::IfStmt::Create(
                    ctx, is->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr, nullptr,
                    is->getCond(), is->getLParenLoc(),
                    new_then ? new_then->getBeginLoc() : is->getIfLoc(), new_then,
                    new_else ? new_else->getBeginLoc() : clang::SourceLocation(), new_else
                );
            }
            return stmt;
        }

        // =========================================================================
        // LoopStructurizePass
        // =========================================================================

        class LoopStructurizePass final : public ASTPass
        {
          public:
            explicit LoopStructurizePass(PipelineState &state)
                : state(state) {}

            const char *name(void) const override { return "LoopStructurizePass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                state.loops_structurized = 0;
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
                    auto label_index = topLevelLabelIndex(body);

                    for (std::size_t i = 0; i < stmts.size(); ++i) {
                        auto *head_label = llvm::dyn_cast< clang::LabelStmt >(stmts[i]);
                        if (head_label == nullptr || i + 1 >= stmts.size()) {
                            continue;
                        }

                        auto *head_sub_if = llvm::dyn_cast_or_null< clang::IfStmt >(
                            head_label->getSubStmt()
                        );

                        std::size_t tail_pos = i + 1;
                        while (tail_pos < stmts.size()
                               && !llvm::isa< clang::IfStmt >(stmts[tail_pos])
                               && !llvm::isa< clang::LabelStmt >(stmts[tail_pos]))
                        {
                            ++tail_pos;
                        }

                        auto *tail_if = llvm::dyn_cast_or_null< clang::IfStmt >(
                            tail_pos < stmts.size() ? stmts[tail_pos] : nullptr
                        );
                        bool head_if_in_label = false;
                        if (tail_if == nullptr && head_sub_if != nullptr
                            && tail_pos == i + 1
                            && llvm::isa_and_nonnull< clang::LabelStmt >(stmts[tail_pos]))
                        {
                            tail_if          = head_sub_if;
                            head_if_in_label = true;
                        }

                        if (tail_if == nullptr) {
                            continue;
                        }

                        if (!head_if_in_label && tail_pos + 1 >= stmts.size()) {
                            continue;
                        }

                        auto *then_goto = llvm::dyn_cast_or_null< clang::GotoStmt >(
                            tail_if->getThen()
                        );
                        auto *else_goto = llvm::dyn_cast_or_null< clang::GotoStmt >(
                            tail_if->getElse()
                        );
                        if (then_goto == nullptr || else_goto == nullptr) {
                            continue;
                        }

                        if (then_goto->getLabel() != head_label->getDecl()) {
                            continue;
                        }

                        if (!label_index.contains(else_goto->getLabel())) {
                            continue;
                        }

                        std::size_t exit_idx     = label_index.at(else_goto->getLabel());
                        std::size_t expected_exit = head_if_in_label ? i + 1 : tail_pos + 1;
                        if (exit_idx != expected_exit) {
                            continue;
                        }

                        std::size_t body_end = head_if_in_label ? i + 1 : tail_pos;
                        std::vector< clang::Stmt * > loop_body;
                        if (!head_if_in_label) {
                            auto *head_sub = head_label->getSubStmt();
                            if (head_sub != nullptr && !isNullStmt(head_sub)) {
                                loop_body.push_back(head_sub);
                            }
                        }
                        loop_body.insert(
                            loop_body.end(),
                            stmts.begin() + static_cast< std::ptrdiff_t >(i + 1),
                            stmts.begin() + static_cast< std::ptrdiff_t >(body_end)
                        );
                        if (loop_body.empty()) {
                            loop_body.push_back(
                                new (ctx) clang::NullStmt(tail_if->getIfLoc(), false)
                            );
                        }
                        auto *loop_compound = makeCompound(
                            ctx, loop_body, head_label->getBeginLoc(), head_label->getEndLoc()
                        );
                        auto *do_stmt = new (ctx) clang::DoStmt(
                            loop_compound, tail_if->getCond(), tail_if->getIfLoc(),
                            tail_if->getIfLoc(), tail_if->getEndLoc()
                        );

                        std::vector< clang::Stmt * > rewritten;
                        rewritten.reserve(stmts.size());
                        for (std::size_t j = 0; j < i; ++j) {
                            if (const auto *gs =
                                    llvm::dyn_cast< clang::GotoStmt >(stmts[j]))
                            {
                                if (gs->getLabel() == head_label->getDecl()) {
                                    continue;
                                }
                            }
                            rewritten.push_back(stmts[j]);
                        }
                        rewritten.push_back(do_stmt);
                        rewritten.insert(
                            rewritten.end(),
                            stmts.begin() + static_cast< std::ptrdiff_t >(exit_idx),
                            stmts.end()
                        );

                        stmts = std::move(rewritten);
                        body  = makeCompound(ctx, stmts, body->getLBracLoc(), body->getRBracLoc());
                        func->setBody(body);
                        ++state.loops_structurized;
                        break;
                    }
                }

                runCfgExtractPass(state, ctx, options);
                return true;
            }

          private:
            PipelineState &state;
        };

        // =========================================================================
        // WhileLoopStructurizePass
        // =========================================================================

        class WhileLoopStructurizePass final : public ASTPass
        {
          public:
            explicit WhileLoopStructurizePass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "WhileLoopStructurizePass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                unsigned local_structurized = 0;
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
                    if (options.verbose) {
                        LOG(DEBUG)
                            << "WhileLoopStructurizePass: scanning func with " << stmts.size()
                            << " stmts, " << label_index.size() << " labels\n";
                        for (std::size_t di = 0; di < stmts.size(); ++di) {
                            std::string kind = "other";
                            if (llvm::isa< clang::LabelStmt >(stmts[di])) {
                                kind = "LabelStmt";
                            } else if (llvm::isa< clang::IfStmt >(stmts[di])) {
                                kind = "IfStmt";
                            } else if (llvm::isa< clang::GotoStmt >(stmts[di])) {
                                kind = "GotoStmt";
                            } else if (llvm::isa< clang::ReturnStmt >(stmts[di])) {
                                kind = "ReturnStmt";
                            }
                            LOG(DEBUG) << "  [" << di << "] " << kind << "\n";
                        }
                    }

                    for (std::size_t i = 0; i < stmts.size(); ++i) {
                        auto *head_label = llvm::dyn_cast< clang::LabelStmt >(stmts[i]);
                        if (head_label == nullptr || i + 1 >= stmts.size()) {
                            continue;
                        }

                        auto *head_if =
                            llvm::dyn_cast_or_null< clang::IfStmt >(head_label->getSubStmt());
                        std::size_t head_if_pos = i;

                        if (head_if == nullptr) {
                            for (std::size_t scan = i + 1; scan < stmts.size(); ++scan) {
                                if (auto *flat_if =
                                        llvm::dyn_cast< clang::IfStmt >(stmts[scan]))
                                {
                                    head_if     = flat_if;
                                    head_if_pos = scan;
                                    break;
                                }
                                if (llvm::isa< clang::LabelStmt >(stmts[scan])) {
                                    break;
                                }
                            }
                        }

                        if (head_if == nullptr) {
                            continue;
                        }

                        auto *then_goto =
                            llvm::dyn_cast_or_null< clang::GotoStmt >(head_if->getThen());
                        auto *else_goto =
                            llvm::dyn_cast_or_null< clang::GotoStmt >(head_if->getElse());
                        if (then_goto == nullptr || else_goto == nullptr) {
                            continue;
                        }

                        if (then_goto->getLabel() == head_label->getDecl()
                            || else_goto->getLabel() == head_label->getDecl())
                        {
                            continue;
                        }

                        if (!label_index.contains(else_goto->getLabel())) {
                            continue;
                        }
                        std::size_t lexit_idx = label_index.at(else_goto->getLabel());
                        if (lexit_idx != head_if_pos + 1) {
                            continue;
                        }

                        if (!label_index.contains(then_goto->getLabel())) {
                            continue;
                        }
                        std::size_t lbody_idx = label_index.at(then_goto->getLabel());
                        if (lbody_idx <= lexit_idx) {
                            continue;
                        }

                        std::size_t back_edge_pos = stmts.size();
                        for (std::size_t p = lbody_idx; p < stmts.size(); ++p) {
                            auto *gs = llvm::dyn_cast_or_null< clang::GotoStmt >(stmts[p]);
                            if (gs != nullptr && gs->getLabel() == head_label->getDecl()) {
                                back_edge_pos = p;
                            }
                        }

                        // --- Inverted pattern ---
                        // The standard pattern requires then→body (further forward) and
                        // else→exit (immediately after head). But decompilers sometimes emit
                        // the exit condition in the then-arm and the body in the else-arm:
                        //   if (exit_cond) goto Lexit; else goto Lbody;
                        // In RPO Lbody appears immediately after the IfStmt (lexit_idx) and
                        // Lexit is further forward (lbody_idx), so the back-edge search above
                        // starts at the wrong position.  When no back-edge is found with the
                        // standard interpretation, try the inverted one.
                        if (back_edge_pos == stmts.size()) {
                            // In the inverted pattern:
                            //   inv_lbody_idx = lexit_idx   (else_goto is actually the body)
                            //   inv_lexit_idx = lbody_idx   (then_goto is actually the exit)
                            std::size_t inv_lbody_idx = lexit_idx;
                            std::size_t inv_lexit_idx = lbody_idx;

                            // Pre-IfStmt stmts [i+1 .. inv_lbody_idx-2] must not contain
                            // back-edges (spurious check for the inverted case).
                            bool inv_spurious = false;
                            for (std::size_t p = i + 1; p < inv_lbody_idx; ++p) {
                                auto *gs =
                                    llvm::dyn_cast_or_null< clang::GotoStmt >(stmts[p]);
                                if (gs != nullptr
                                    && gs->getLabel() == head_label->getDecl())
                                {
                                    inv_spurious = true;
                                    break;
                                }
                            }
                            if (inv_spurious) {
                                continue;
                            }

                            // Back-edges must lie in [inv_lbody_idx .. inv_lexit_idx - 1].
                            std::size_t inv_back_edge_pos = stmts.size();
                            for (std::size_t p = inv_lbody_idx; p < inv_lexit_idx; ++p) {
                                auto *gs =
                                    llvm::dyn_cast_or_null< clang::GotoStmt >(stmts[p]);
                                if (gs != nullptr
                                    && gs->getLabel() == head_label->getDecl())
                                {
                                    inv_back_edge_pos = p;
                                }
                            }
                            if (inv_back_edge_pos == stmts.size()) {
                                continue;  // no back-edge in either pattern
                            }

                            // Build while(true) { pre_stmts; if(exit_cond) break; body }
                            // The pre-IfStmt assignments [i+1 .. head_if_pos-1] run each
                            // iteration before the exit test, so they go into the loop body.
                            std::vector< clang::Stmt * > inv_body_stmts;
                            for (std::size_t p = i + 1; p < head_if_pos; ++p) {
                                inv_body_stmts.push_back(stmts[p]);
                            }

                            // Convert the inverted IfStmt to "if (exit_cond) break;".
                            auto *break_stmt =
                                new (ctx) clang::BreakStmt(head_if->getIfLoc());
                            auto *inv_if_break = clang::IfStmt::Create(
                                ctx, head_if->getIfLoc(),
                                clang::IfStatementKind::Ordinary, nullptr, nullptr,
                                head_if->getCond(), head_if->getLParenLoc(),
                                break_stmt->getBeginLoc(), break_stmt,
                                clang::SourceLocation(), nullptr
                            );
                            inv_body_stmts.push_back(inv_if_break);

                            for (std::size_t p = inv_lbody_idx; p < inv_back_edge_pos; ++p) {
                                inv_body_stmts.push_back(stmts[p]);
                            }
                            if (inv_body_stmts.empty()) {
                                inv_body_stmts.push_back(
                                    new (ctx) clang::NullStmt(head_if->getIfLoc(), false)
                                );
                            }

                            auto *inv_body_compound = makeCompound(
                                ctx, inv_body_stmts, head_label->getBeginLoc(),
                                head_label->getEndLoc()
                            );
                            auto *inv_while = clang::WhileStmt::Create(
                                ctx, nullptr,
                                makeBoolTrue(ctx, head_if->getIfLoc()),
                                inv_body_compound, head_if->getIfLoc(),
                                head_if->getIfLoc(), head_if->getEndLoc()
                            );
                            auto *inv_new_head = new (ctx) clang::LabelStmt(
                                head_label->getIdentLoc(), head_label->getDecl(), inv_while
                            );

                            std::vector< clang::Stmt * > inv_rewritten;
                            inv_rewritten.reserve(stmts.size());
                            inv_rewritten.insert(
                                inv_rewritten.end(), stmts.begin(),
                                stmts.begin() + static_cast< std::ptrdiff_t >(i)
                            );
                            inv_rewritten.push_back(inv_new_head);
                            inv_rewritten.insert(
                                inv_rewritten.end(),
                                stmts.begin()
                                    + static_cast< std::ptrdiff_t >(inv_back_edge_pos + 1),
                                stmts.end()
                            );

                            stmts = std::move(inv_rewritten);
                            body  = makeCompound(
                                ctx, stmts, body->getLBracLoc(), body->getRBracLoc()
                            );
                            func->setBody(body);
                            ++local_structurized;
                            break;
                        }

                        if (back_edge_pos == stmts.size()) {
                            continue;
                        }

                        bool spurious = false;
                        for (std::size_t p = i + 1; p < lbody_idx; ++p) {
                            auto *gs = llvm::dyn_cast_or_null< clang::GotoStmt >(stmts[p]);
                            if (gs != nullptr && gs->getLabel() == head_label->getDecl()) {
                                spurious = true;
                                break;
                            }
                        }
                        if (spurious) {
                            continue;
                        }

                        std::vector< clang::Stmt * > body_stmts(
                            stmts.begin() + static_cast< std::ptrdiff_t >(lbody_idx),
                            stmts.begin() + static_cast< std::ptrdiff_t >(back_edge_pos)
                        );
                        if (body_stmts.empty()) {
                            body_stmts.push_back(
                                new (ctx) clang::NullStmt(head_if->getIfLoc(), false)
                            );
                        }
                        auto *body_compound = makeCompound(
                            ctx, body_stmts, head_label->getBeginLoc(), head_label->getEndLoc()
                        );

                        auto *while_stmt = clang::WhileStmt::Create(
                            ctx, nullptr, head_if->getCond(), body_compound,
                            head_if->getIfLoc(), head_if->getIfLoc(), head_if->getEndLoc()
                        );

                        auto *new_head = new (ctx) clang::LabelStmt(
                            head_label->getIdentLoc(), head_label->getDecl(), while_stmt
                        );

                        std::vector< clang::Stmt * > rewritten;
                        rewritten.reserve(stmts.size());
                        rewritten.insert(
                            rewritten.end(), stmts.begin(),
                            stmts.begin() + static_cast< std::ptrdiff_t >(i)
                        );
                        rewritten.push_back(new_head);
                        rewritten.insert(
                            rewritten.end(),
                            stmts.begin() + static_cast< std::ptrdiff_t >(i + 1),
                            stmts.begin() + static_cast< std::ptrdiff_t >(lbody_idx)
                        );
                        rewritten.insert(
                            rewritten.end(),
                            stmts.begin() + static_cast< std::ptrdiff_t >(back_edge_pos + 1),
                            stmts.end()
                        );

                        stmts = std::move(rewritten);
                        body  = makeCompound(
                            ctx, stmts, body->getLBracLoc(), body->getRBracLoc()
                        );
                        func->setBody(body);
                        ++local_structurized;
                        break;
                    }
                }

                if (local_structurized > 0) {
                    state.loops_structurized += local_structurized;
                    if (options.verbose) {
                        LOG(DEBUG) << "WhileLoopStructurizePass: structurized "
                                   << local_structurized << " while loop(s)\n";
                    }
                    runCfgExtractPass(state, ctx, options);
                }

                return true;
            }

          private:
            PipelineState &state;
        };

        // =========================================================================
        // DegenerateLoopUnwrapPass
        // =========================================================================

        class DegenerateLoopUnwrapPass final : public ASTPass
        {
          public:
            explicit DegenerateLoopUnwrapPass(PipelineState &state)
                : state(state) {}

            const char *name(void) const override { return "DegenerateLoopUnwrapPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                unsigned unwrap_count = 0;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }

                    auto *body = llvm::dyn_cast< clang::CompoundStmt >(func->getBody());
                    if (body == nullptr || body->size() != 1) {
                        continue;
                    }

                    auto *ws = llvm::dyn_cast< clang::WhileStmt >(*body->body_begin());
                    if (ws == nullptr || !isAlwaysTrue(ws->getCond())) {
                        continue;
                    }

                    auto *loop_body = llvm::dyn_cast< clang::CompoundStmt >(ws->getBody());
                    if (loop_body == nullptr) {
                        continue;
                    }

                    std::vector< clang::ContinueStmt * > continues;
                    collectContinues(loop_body, continues);

                    bool is_degenerate = false;
                    if (continues.empty()) {
                        is_degenerate = true;
                    } else {
                        std::unordered_set< const clang::VarDecl * > locals;
                        collectLocalDecls(loop_body, locals);
                        is_degenerate = !writesNonLocalVar(loop_body, locals);
                    }

                    if (!is_degenerate) {
                        continue;
                    }

                    std::vector< clang::Stmt * > flat_stmts(
                        loop_body->body_begin(), loop_body->body_end()
                    );
                    while (!flat_stmts.empty()
                           && llvm::isa< clang::BreakStmt >(flat_stmts.back()))
                    {
                        flat_stmts.pop_back();
                    }

                    if (!continues.empty()) {
                        std::string label_name =
                            "__loop_restart_" + std::to_string(unwrap_count);
                        auto *restart_label = clang::LabelDecl::Create(
                            ctx, func, ws->getBeginLoc(), &ctx.Idents.get(label_name)
                        );
                        restart_label->setDeclContext(func);

                        std::vector< clang::Stmt * > rewritten;
                        rewritten.reserve(flat_stmts.size() + 1U);
                        auto *null_stmt = new (ctx) clang::NullStmt(ws->getBeginLoc(), false);
                        rewritten.push_back(
                            new (ctx) clang::LabelStmt(
                                ws->getBeginLoc(), restart_label, null_stmt
                            )
                        );
                        for (auto *s : flat_stmts) {
                            rewritten.push_back(replaceContinueWithGoto(ctx, s, restart_label));
                        }
                        flat_stmts = std::move(rewritten);
                    }

                    func->setBody(makeCompound(
                        ctx, flat_stmts, body->getLBracLoc(), body->getRBracLoc()
                    ));
                    ++unwrap_count;
                }

                if (unwrap_count > 0) {
                    if (options.verbose) {
                        LOG(DEBUG) << "DegenerateLoopUnwrapPass: unwrapped " << unwrap_count
                                   << " degenerate while(1) loop(s)\n";
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
        // LoopConditionRecoveryPass
        // =========================================================================

        class LoopConditionRecoveryPass final : public ASTPass
        {
          public:
            explicit LoopConditionRecoveryPass(PipelineState &state)
                : state(state) {}

            const char *name(void) const override { return "LoopConditionRecoveryPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                unsigned recovered = 0;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }
                    auto *new_body = processStmt(ctx, func->getBody(), recovered);
                    if (new_body != func->getBody()) {
                        func->setBody(new_body);
                    }
                }

                if (recovered > 0 && options.verbose) {
                    LOG(DEBUG) << "LoopConditionRecoveryPass: recovered " << recovered
                               << " loop condition(s)\n";
                }
                return true;
            }

          private:
            [[maybe_unused]] PipelineState &state;

            static clang::Expr *negateCondition(clang::ASTContext &ctx, clang::Expr *cond) {
                auto *stripped = cond->IgnoreParenImpCasts();
                if (auto *uo = llvm::dyn_cast< clang::UnaryOperator >(stripped)) {
                    if (uo->getOpcode() == clang::UO_LNot) {
                        return uo->getSubExpr();
                    }
                }
                return clang::UnaryOperator::Create(
                    ctx, ensureRValue(ctx, cond), clang::UO_LNot, ctx.IntTy, clang::VK_PRValue,
                    clang::OK_Ordinary, cond->getBeginLoc(), false, clang::FPOptionsOverride()
                );
            }

            clang::WhileStmt *tryRecoverWhileCondition(
                clang::ASTContext &ctx, clang::WhileStmt *ws, unsigned &recovered
            ) {
                if (!isAlwaysTrue(ws->getCond())) {
                    return ws;
                }
                auto *body_cs = llvm::dyn_cast< clang::CompoundStmt >(ws->getBody());
                if (body_cs == nullptr || body_cs->body_empty()) {
                    return ws;
                }

                auto *first      = *body_cs->body_begin();
                auto *leading_if = llvm::dyn_cast< clang::IfStmt >(first);
                if (leading_if == nullptr) {
                    return ws;
                }

                if (!llvm::isa< clang::BreakStmt >(leading_if->getThen())
                    || leading_if->getElse() != nullptr)
                {
                    return ws;
                }

                auto *new_cond =
                    negateCondition(ctx, const_cast< clang::Expr * >(leading_if->getCond()));
                std::vector< clang::Stmt * > rest(
                    body_cs->body_begin() + 1, body_cs->body_end()
                );
                auto *new_body =
                    makeCompound(ctx, rest, body_cs->getLBracLoc(), body_cs->getRBracLoc());
                ++recovered;
                return clang::WhileStmt::Create(
                    ctx, nullptr, new_cond, new_body, ws->getWhileLoc(), ws->getLParenLoc(),
                    ws->getRParenLoc()
                );
            }

            clang::Stmt *processStmt(
                clang::ASTContext &ctx, clang::Stmt *stmt, unsigned &recovered
            ) {
                if (stmt == nullptr) {
                    return nullptr;
                }

                if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                    auto *recovered_ws = tryRecoverWhileCondition(ctx, ws, recovered);
                    auto *new_body     = processStmt(ctx, recovered_ws->getBody(), recovered);
                    if (new_body == recovered_ws->getBody()) {
                        return recovered_ws;
                    }
                    return clang::WhileStmt::Create(
                        ctx, nullptr, recovered_ws->getCond(), new_body,
                        recovered_ws->getWhileLoc(), recovered_ws->getLParenLoc(),
                        recovered_ws->getRParenLoc()
                    );
                }

                if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                    auto *new_body = processStmt(ctx, ds->getBody(), recovered);
                    if (new_body == ds->getBody()) {
                        return ds;
                    }
                    return new (ctx) clang::DoStmt(
                        new_body, ds->getCond(), ds->getDoLoc(), ds->getWhileLoc(),
                        ds->getRParenLoc()
                    );
                }

                if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                    auto *new_then = processStmt(ctx, is->getThen(), recovered);
                    auto *new_else = is->getElse() != nullptr
                        ? processStmt(ctx, is->getElse(), recovered)
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

                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                    return processCompound(ctx, cs, recovered);
                }

                return stmt;
            }

            clang::CompoundStmt *processCompound(
                clang::ASTContext &ctx, clang::CompoundStmt *cs, unsigned &recovered
            ) {
                bool changed = false;
                std::vector< clang::Stmt * > result;
                result.reserve(cs->size());

                for (auto *stmt : cs->body()) {
                    auto *processed = processStmt(ctx, stmt, recovered);
                    if (processed != stmt) {
                        changed = true;
                    }

                    auto *if_stmt = llvm::dyn_cast_or_null< clang::IfStmt >(processed);
                    if (if_stmt != nullptr) {
                        auto *then_br = if_stmt->getThen();
                        auto *else_br = if_stmt->getElse();

                        if (llvm::isa_and_nonnull< clang::BreakStmt >(then_br)
                            && llvm::isa_and_nonnull< clang::ContinueStmt >(else_br))
                        {
                            processed = clang::IfStmt::Create(
                                ctx, if_stmt->getIfLoc(), clang::IfStatementKind::Ordinary,
                                nullptr, nullptr, if_stmt->getCond(), if_stmt->getLParenLoc(),
                                then_br->getBeginLoc(), then_br, clang::SourceLocation(), nullptr
                            );
                            ++recovered;
                            changed = true;
                        } else if (llvm::isa_and_nonnull< clang::ContinueStmt >(then_br)
                                   && llvm::isa_and_nonnull< clang::BreakStmt >(else_br))
                        {
                            auto *neg = negateCondition(
                                ctx, const_cast< clang::Expr * >(if_stmt->getCond())
                            );
                            processed = clang::IfStmt::Create(
                                ctx, if_stmt->getIfLoc(), clang::IfStatementKind::Ordinary,
                                nullptr, nullptr, neg, if_stmt->getLParenLoc(),
                                else_br->getBeginLoc(), else_br, clang::SourceLocation(), nullptr
                            );
                            ++recovered;
                            changed = true;
                        }
                    }

                    if (processed != nullptr) {
                        result.push_back(processed);
                    }
                }

                if (!changed) {
                    return cs;
                }
                return makeCompound(ctx, result, cs->getLBracLoc(), cs->getRBracLoc());
            }
        };

        // =========================================================================
        // DegenerateWhileElimPass
        // =========================================================================

        class DegenerateWhileElimPass final : public ASTPass
        {
          public:
            explicit DegenerateWhileElimPass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "DegenerateWhileElimPass"; }

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

                    bool local_changed = true;
                    while (local_changed) {
                        local_changed  = false;
                        auto *new_body = processStmt(ctx, func->getBody(), eliminated, local_changed);
                        if (new_body != nullptr && new_body != func->getBody()) {
                            func->setBody(new_body);
                        } else if (new_body == nullptr) {
                            auto *empty = clang::CompoundStmt::Create(
                                ctx, {}, clang::FPOptionsOverride(), clang::SourceLocation(),
                                clang::SourceLocation()
                            );
                            func->setBody(empty);
                        }
                    }
                }

                state.degenerate_while_eliminated += eliminated;

                if (eliminated > 0) {
                    if (options.verbose) {
                        LOG(DEBUG) << "DegenerateWhileElimPass: eliminated " << eliminated
                                   << " degenerate loop(s)\n";
                    }
                    runGotoCanonicalizePass(state, ctx, options);
                    runCfgExtractPass(state, ctx, options);
                }

                return true;
            }

          private:
            PipelineState &state;

            // Walk `stmt` replacing direct (non-nested) BreakStmt nodes with
            // NullStmt. Stops recursing into nested loops/switches so that
            // their break statements are left untouched.
            static clang::Stmt *replaceBreaksWithNull(
                clang::ASTContext &ctx, clang::Stmt *stmt
            ) {
                if (stmt == nullptr) {
                    return nullptr;
                }
                if (llvm::isa< clang::BreakStmt >(stmt)) {
                    return new (ctx) clang::NullStmt(stmt->getBeginLoc(), false);
                }
                if (llvm::isa< clang::WhileStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt)
                    || llvm::isa< clang::DoStmt >(stmt)
                    || llvm::isa< clang::SwitchStmt >(stmt))
                {
                    return stmt;
                }
                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                    bool changed = false;
                    std::vector< clang::Stmt * > result;
                    result.reserve(cs->size());
                    for (auto *s : cs->body()) {
                        auto *ns = replaceBreaksWithNull(ctx, s);
                        if (ns != s) {
                            changed = true;
                        }
                        result.push_back(ns != nullptr ? ns : s);
                    }
                    if (!changed) {
                        return cs;
                    }
                    return makeCompound(ctx, result, cs->getLBracLoc(), cs->getRBracLoc());
                }
                if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                    auto *new_then = replaceBreaksWithNull(ctx, is->getThen());
                    auto *new_else = replaceBreaksWithNull(ctx, is->getElse());
                    if (new_then == is->getThen() && new_else == is->getElse()) {
                        return is;
                    }
                    if (new_then == nullptr) {
                        new_then = new (ctx) clang::NullStmt(is->getIfLoc(), false);
                    }
                    return clang::IfStmt::Create(
                        ctx, is->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr, nullptr,
                        const_cast< clang::Expr * >(is->getCond()), is->getLParenLoc(),
                        new_then->getBeginLoc(), new_then,
                        new_else != nullptr ? new_else->getBeginLoc() : clang::SourceLocation(),
                        new_else
                    );
                }
                if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                    auto *new_sub = replaceBreaksWithNull(ctx, ls->getSubStmt());
                    if (new_sub == ls->getSubStmt()) {
                        return ls;
                    }
                    if (new_sub == nullptr) {
                        new_sub = new (ctx) clang::NullStmt(ls->getBeginLoc(), false);
                    }
                    return new (ctx) clang::LabelStmt(ls->getIdentLoc(), ls->getDecl(), new_sub);
                }
                return stmt;
            }

            static bool hasContinue(const clang::Stmt *stmt) {
                if (stmt == nullptr) {
                    return false;
                }
                if (llvm::isa< clang::WhileStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt)
                    || llvm::isa< clang::DoStmt >(stmt)
                    || llvm::isa< clang::SwitchStmt >(stmt))
                {
                    return false;
                }
                if (llvm::isa< clang::ContinueStmt >(stmt)) {
                    return true;
                }
                for (const auto *child : stmt->children()) {
                    if (hasContinue(child)) {
                        return true;
                    }
                }
                return false;
            }

            // Returns true only when every execution path through `stmt` ends
            // with a BreakStmt or ReturnStmt.  GotoStmt is intentionally excluded
            // because a goto pointing back to the loop header is a loop-back, not
            // an exit, and treating it as a terminator would cause the degenerate
            // while-elimination to fire incorrectly on structured loops.
            static bool isBreakOrReturnTerminator(const clang::Stmt *stmt) {
                if (stmt == nullptr) {
                    return false;
                }
                if (llvm::isa< clang::BreakStmt >(stmt) || llvm::isa< clang::ReturnStmt >(stmt)) {
                    return true;
                }
                if (const auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                    return is->getElse() != nullptr && isBreakOrReturnTerminator(is->getThen())
                           && isBreakOrReturnTerminator(is->getElse());
                }
                if (const auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                    if (cs->body_empty()) {
                        return false;
                    }
                    return isBreakOrReturnTerminator(*(cs->body_end() - 1));
                }
                return false;
            }

            static clang::Stmt *processStmt(
                clang::ASTContext &ctx, clang::Stmt *stmt, unsigned &eliminated, bool &changed
            ) {
                if (stmt == nullptr) {
                    return nullptr;
                }

                if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                    bool inner     = false;
                    auto *new_body = processStmt(ctx, ws->getBody(), eliminated, inner);
                    if (inner) {
                        changed = true;
                    }

                    auto *effective_body = new_body != nullptr ? new_body : ws->getBody();
                    if (isAlwaysTrue(ws->getCond()) && !hasContinue(ws->getBody())
                        && isBreakOrReturnTerminator(effective_body))
                    {
                        ++eliminated;
                        changed = true;
                        return replaceBreaksWithNull(ctx, effective_body);
                    }

                    if (isAlwaysFalse(ws->getCond()) && !hasContinue(ws->getBody())) {
                        ++eliminated;
                        changed = true;
                        return nullptr;
                    }

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
                    bool inner     = false;
                    auto *new_body = processStmt(ctx, ds->getBody(), eliminated, inner);
                    if (inner) {
                        changed = true;
                    }

                    if (isAlwaysFalse(ds->getCond()) && !hasContinue(ds->getBody())) {
                        ++eliminated;
                        changed = true;
                        if (new_body == nullptr) {
                            return new (ctx) clang::NullStmt(ds->getDoLoc(), false);
                        }
                        return new_body;
                    }

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

                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                    bool any_changed = false;
                    std::vector< clang::Stmt * > result;
                    result.reserve(cs->size());
                    for (auto *s : cs->body()) {
                        bool inner = false;
                        auto *ns   = processStmt(ctx, s, eliminated, inner);
                        if (inner) {
                            any_changed = true;
                            changed     = true;
                        }
                        if (ns == nullptr) {
                            any_changed = true;
                            changed     = true;
                            continue;
                        }
                        if (auto *inner_cs = llvm::dyn_cast< clang::CompoundStmt >(ns)) {
                            for (auto *child : inner_cs->body()) {
                                result.push_back(child);
                            }
                        } else {
                            result.push_back(ns);
                        }
                    }
                    if (!any_changed) {
                        return cs;
                    }
                    return makeCompound(ctx, result, cs->getLBracLoc(), cs->getRBracLoc());
                }

                if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                    bool inner = false;
                    auto *sub  = processStmt(ctx, ls->getSubStmt(), eliminated, inner);
                    if (!inner) {
                        return ls;
                    }
                    changed = true;
                    if (sub == nullptr) {
                        sub = new (ctx) clang::NullStmt(ls->getBeginLoc(), false);
                    }
                    return new (ctx) clang::LabelStmt(ls->getIdentLoc(), ls->getDecl(), sub);
                }

                if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                    bool inner     = false;
                    auto *new_then = processStmt(ctx, is->getThen(), eliminated, inner);
                    auto *new_else = is->getElse()
                        ? processStmt(ctx, is->getElse(), eliminated, inner)
                        : nullptr;
                    if (!inner) {
                        return is;
                    }
                    changed = true;
                    if (new_then == nullptr) {
                        new_then = new (ctx) clang::NullStmt(is->getIfLoc(), false);
                    }
                    return clang::IfStmt::Create(
                        ctx, is->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr, nullptr,
                        is->getCond(), is->getLParenLoc(), new_then->getBeginLoc(), new_then,
                        new_else ? new_else->getBeginLoc() : clang::SourceLocation(), new_else
                    );
                }

                return stmt;
            }
        };

        // =========================================================================
        // NaturalLoopRecoveryPass
        // =========================================================================

        class NaturalLoopRecoveryPass final : public ASTPass
        {
          public:
            explicit NaturalLoopRecoveryPass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "NaturalLoopRecoveryPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                unsigned recovered    = 0;
                unsigned for_upgraded = 0;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }
                    processFunction(ctx, func, recovered, for_upgraded, options);
                }

                state.natural_loops_recovered += recovered;
                state.for_loops_upgraded      += for_upgraded;

                if ((recovered + for_upgraded) > 0) {
                    if (options.verbose) {
                        LOG(DEBUG)
                            << "NaturalLoopRecoveryPass: recovered " << recovered
                            << " while-loop(s), upgraded " << for_upgraded << " for-loop(s)\n";
                    }
                    runGotoCanonicalizePass(state, ctx, options);
                    runCfgExtractPass(state, ctx, options);
                }

                return true;
            }

          private:
            PipelineState &state;

            static bool isHeaderAlreadyStructured(const clang::LabelStmt *ls) {
                if (ls == nullptr) {
                    return false;
                }
                const auto *sub = ls->getSubStmt();
                return llvm::isa_and_nonnull< clang::WhileStmt >(sub)
                    || llvm::isa_and_nonnull< clang::DoStmt >(sub)
                    || llvm::isa_and_nonnull< clang::ForStmt >(sub);
            }

            static clang::GotoStmt *extractFirstGoto(clang::Stmt *s) {
                if (s == nullptr) {
                    return nullptr;
                }
                if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
                    return gs;
                }
                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                    for (auto *child : cs->body()) {
                        if (auto *found = extractFirstGoto(child)) {
                            return found;
                        }
                    }
                }
                return nullptr;
            }

            clang::Stmt *rewriteChildStmt(
                clang::ASTContext &ctx, clang::Stmt *s, unsigned &recovered,
                unsigned &for_upgraded, const patchestry::Options &options
            ) {
                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                    return processCompound(ctx, cs, recovered, for_upgraded, options);
                }
                if (auto *is = llvm::dyn_cast< clang::IfStmt >(s)) {
                    auto *then_s   = is->getThen();
                    auto *new_then =
                        rewriteChildStmt(ctx, then_s, recovered, for_upgraded, options);
                    auto *else_s   = is->getElse();
                    auto *new_else = else_s != nullptr
                        ? rewriteChildStmt(ctx, else_s, recovered, for_upgraded, options)
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
                    auto *body = ws->getBody();
                    auto *new_body =
                        rewriteChildStmt(ctx, body, recovered, for_upgraded, options);
                    if (new_body == body) {
                        return s;
                    }
                    return clang::WhileStmt::Create(
                        ctx, nullptr, ws->getCond(), new_body, ws->getWhileLoc(),
                        ws->getLParenLoc(), ws->getRParenLoc()
                    );
                }
                if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
                    auto *body = fs->getBody();
                    auto *new_body =
                        rewriteChildStmt(ctx, body, recovered, for_upgraded, options);
                    if (new_body == body) {
                        return s;
                    }
                    return new (ctx) clang::ForStmt(
                        ctx, fs->getInit(), fs->getCond(), nullptr, fs->getInc(), new_body,
                        fs->getForLoc(), fs->getLParenLoc(), fs->getRParenLoc()
                    );
                }
                if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
                    auto *sub = ls->getSubStmt();
                    auto *new_sub =
                        rewriteChildStmt(ctx, sub, recovered, for_upgraded, options);
                    if (new_sub == sub) {
                        return s;
                    }
                    return new (ctx)
                        clang::LabelStmt(ls->getIdentLoc(), ls->getDecl(), new_sub);
                }
                return s;
            }

            clang::CompoundStmt *processCompound(
                clang::ASTContext &ctx, clang::CompoundStmt *compound, unsigned &recovered,
                unsigned &for_upgraded, const patchestry::Options &options
            ) {
                bool child_changed = false;
                {
                    std::vector< clang::Stmt * > stmts(
                        compound->body_begin(), compound->body_end()
                    );
                    for (auto &s : stmts) {
                        auto *new_s =
                            rewriteChildStmt(ctx, s, recovered, for_upgraded, options);
                        if (new_s != s) {
                            s             = new_s;
                            child_changed = true;
                        }
                    }
                    if (child_changed) {
                        compound = makeCompound(
                            ctx, stmts, compound->getLBracLoc(), compound->getRBracLoc()
                        );
                    }
                }

                bool any_changed = true;
                while (any_changed) {
                    any_changed = false;
                    if (compound->size() < 3U) {
                        break;
                    }

                    std::vector< clang::Stmt * > stmts(
                        compound->body_begin(), compound->body_end()
                    );
                    const std::size_t N = stmts.size();
                    auto label_index    = topLevelLabelIndex(compound);
                    if (label_index.empty()) {
                        break;
                    }

                    auto succ_of   = buildSuccOf(stmts, label_index, N);
                    auto backedges = detectBackedges(N, succ_of);
                    if (backedges.empty()) {
                        break;
                    }

                    auto domtree = buildDomTree(succ_of, /*entry=*/0, N);

                    std::sort(backedges.begin(), backedges.end(), [](auto &a, auto &b) {
                        return a.second > b.second;
                    });

                    for (auto &[latch, header] : backedges) {
                        if (!domtree.dominates(header, latch)) {
                            continue;
                        }

                        auto *head_label =
                            llvm::dyn_cast_or_null< clang::LabelStmt >(stmts[header]);
                        if (head_label == nullptr) {
                            continue;
                        }
                        if (isHeaderAlreadyStructured(head_label)) {
                            continue;
                        }

                        auto *head_if = llvm::dyn_cast_or_null< clang::IfStmt >(
                            head_label->getSubStmt()
                        );
                        std::size_t head_if_pos = header;
                        if (head_if == nullptr && header + 1 < N) {
                            if (auto *flat_if =
                                    llvm::dyn_cast< clang::IfStmt >(stmts[header + 1]))
                            {
                                head_if     = flat_if;
                                head_if_pos = header + 1;
                            }
                        }
                        if (head_if == nullptr) {
                            continue;
                        }

                        auto *then_goto =
                            llvm::dyn_cast_or_null< clang::GotoStmt >(head_if->getThen());
                        auto *else_goto =
                            llvm::dyn_cast_or_null< clang::GotoStmt >(head_if->getElse());
                        std::size_t exit_idx = stmts.size();

                        if (then_goto != nullptr && else_goto != nullptr) {
                            auto exit_it = label_index.find(else_goto->getLabel());
                            if (exit_it == label_index.end()) {
                                continue;
                            }
                            exit_idx = exit_it->second;
                        } else if (head_if->getElse() == nullptr) {
                            auto *inner_goto = extractFirstGoto(head_if->getThen());
                            if (inner_goto == nullptr) {
                                continue;
                            }
                            auto exit_it = label_index.find(inner_goto->getLabel());
                            if (exit_it == label_index.end() || exit_it->second <= latch) {
                                continue;
                            }
                            exit_idx = exit_it->second;
                        } else {
                            continue;
                        }

                        if (tryEmitForLoop(
                                ctx, stmts, label_index, header, latch, head_if_pos, head_if,
                                exit_idx, for_upgraded
                            ))
                        {
                            compound = makeCompound(
                                ctx, stmts, compound->getLBracLoc(), compound->getRBracLoc()
                            );
                            any_changed = true;
                            break;
                        }

                        if (tryEmitWhileLoop(
                                ctx, stmts, label_index, header, latch, head_if_pos, head_if,
                                exit_idx, recovered, options
                            ))
                        {
                            compound = makeCompound(
                                ctx, stmts, compound->getLBracLoc(), compound->getRBracLoc()
                            );
                            any_changed = true;
                            break;
                        }
                    }
                }

                return compound;
            }

            void processFunction(
                clang::ASTContext &ctx, clang::FunctionDecl *func, unsigned &recovered,
                unsigned &for_upgraded, const patchestry::Options &options
            ) {
                auto *body =
                    llvm::dyn_cast_or_null< clang::CompoundStmt >(func->getBody());
                if (body == nullptr) {
                    return;
                }
                bool any_changed = true;
                while (any_changed) {
                    unsigned r0 = recovered, f0 = for_upgraded;
                    auto *new_body =
                        processCompound(ctx, body, recovered, for_upgraded, options);
                    if (new_body != body) {
                        func->setBody(new_body);
                        body = new_body;
                    }
                    any_changed = (recovered > r0 || for_upgraded > f0);
                }
            }

            static bool tryEmitForLoop(
                clang::ASTContext &ctx, std::vector< clang::Stmt * > &stmts,
                const std::unordered_map< const clang::LabelDecl *, std::size_t > &label_index,
                std::size_t header, std::size_t latch, std::size_t head_if_pos,
                clang::IfStmt *head_if, std::size_t exit_idx, unsigned &for_upgraded
            ) {
                if (header == 0 || latch < head_if_pos + 2U) {
                    return false;
                }

                auto *init_stmt = stmts[header - 1];
                auto *init_bin  = llvm::dyn_cast_or_null< clang::BinaryOperator >(init_stmt);
                if (init_bin == nullptr || !init_bin->isAssignmentOp()) {
                    return false;
                }
                auto *init_lhs = llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                    init_bin->getLHS()->IgnoreParenImpCasts()
                );
                if (init_lhs == nullptr) {
                    return false;
                }
                auto *loop_var =
                    llvm::dyn_cast_or_null< clang::VarDecl >(init_lhs->getDecl());
                if (loop_var == nullptr) {
                    return false;
                }

                clang::Expr *for_cond = nullptr;
                auto *then_goto = llvm::dyn_cast_or_null< clang::GotoStmt >(head_if->getThen());
                auto *else_goto = llvm::dyn_cast_or_null< clang::GotoStmt >(head_if->getElse());

                if (then_goto != nullptr && else_goto != nullptr) {
                    auto then_it = label_index.find(then_goto->getLabel());
                    if (then_it == label_index.end()) {
                        return false;
                    }
                    const std::size_t then_idx = then_it->second;
                    if (then_idx <= head_if_pos || then_idx >= latch) {
                        return false;
                    }
                    for_cond = head_if->getCond();
                } else if (head_if->getElse() == nullptr) {
                    auto *exit_cond = head_if->getCond()->IgnoreParenImpCasts();
                    if (auto *uo = llvm::dyn_cast< clang::UnaryOperator >(exit_cond)) {
                        if (uo->getOpcode() == clang::UO_LNot) {
                            for_cond = uo->getSubExpr();
                        }
                    }
                    if (for_cond == nullptr) {
                        return false;
                    }
                } else {
                    return false;
                }

                auto *cond_bin = llvm::dyn_cast_or_null< clang::BinaryOperator >(
                    for_cond->IgnoreParenImpCasts()
                );
                if (cond_bin == nullptr
                    || (!cond_bin->isComparisonOp()
                        && cond_bin->getOpcode() != clang::BO_NE))
                {
                    return false;
                }
                auto *cond_lhs_dre = llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                    cond_bin->getLHS()->IgnoreParenImpCasts()
                );
                auto *cond_rhs_dre = llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                    cond_bin->getRHS()->IgnoreParenImpCasts()
                );
                const bool cond_uses_var =
                    (cond_lhs_dre != nullptr && cond_lhs_dre->getDecl() == loop_var)
                    || (cond_rhs_dre != nullptr && cond_rhs_dre->getDecl() == loop_var);
                if (!cond_uses_var) {
                    return false;
                }

                auto *incr_stmt        = stmts[latch - 1];
                clang::Expr *incr_expr = nullptr;
                if (auto *uo = llvm::dyn_cast< clang::UnaryOperator >(incr_stmt)) {
                    if (uo->isIncrementOp()) {
                        auto *sub = llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                            uo->getSubExpr()->IgnoreParenImpCasts()
                        );
                        if (sub != nullptr && sub->getDecl() == loop_var) {
                            incr_expr = uo;
                        }
                    }
                } else if (auto *bo = llvm::dyn_cast< clang::BinaryOperator >(incr_stmt)) {
                    if (bo->getOpcode() == clang::BO_AddAssign) {
                        auto *lhs_dre = llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                            bo->getLHS()->IgnoreParenImpCasts()
                        );
                        if (lhs_dre != nullptr && lhs_dre->getDecl() == loop_var) {
                            incr_expr = bo;
                        }
                    } else if (bo->isAssignmentOp()) {
                        auto *lhs_dre = llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                            bo->getLHS()->IgnoreParenImpCasts()
                        );
                        if (lhs_dre != nullptr && lhs_dre->getDecl() == loop_var) {
                            auto *rhs = bo->getRHS()->IgnoreParenImpCasts();
                            if (auto *rhs_bin =
                                    llvm::dyn_cast< clang::BinaryOperator >(rhs))
                            {
                                if (rhs_bin->getOpcode() == clang::BO_Add
                                    || rhs_bin->getOpcode() == clang::BO_Sub)
                                {
                                    auto *rhs_lhs = llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                                        rhs_bin->getLHS()->IgnoreParenImpCasts()
                                    );
                                    if (rhs_lhs != nullptr
                                        && rhs_lhs->getDecl() == loop_var)
                                    {
                                        incr_expr = bo;
                                    }
                                }
                            }
                        }
                    }
                }
                if (incr_expr == nullptr) {
                    return false;
                }

                std::vector< clang::Stmt * > body_stmts;
                for (std::size_t k = head_if_pos + 1; k + 1 < latch; ++k) {
                    body_stmts.push_back(stmts[k]);
                }
                auto loc            = stmts[header]->getBeginLoc();
                auto *body_compound = makeCompound(ctx, body_stmts, loc, loc);

                auto *for_stmt = new (ctx) clang::ForStmt(
                    ctx, init_stmt, for_cond, nullptr, incr_expr, body_compound, loc, loc, loc
                );

                std::vector< clang::Stmt * > new_stmts;
                for (std::size_t k = 0; k + 1 < header; ++k) {
                    new_stmts.push_back(stmts[k]);
                }
                new_stmts.push_back(for_stmt);
                for (std::size_t k = exit_idx; k < stmts.size(); ++k) {
                    new_stmts.push_back(stmts[k]);
                }
                stmts = std::move(new_stmts);
                ++for_upgraded;
                return true;
            }

            static bool tryEmitWhileLoop(
                clang::ASTContext &ctx, std::vector< clang::Stmt * > &stmts,
                const std::unordered_map< const clang::LabelDecl *, std::size_t > &label_index,
                std::size_t header, std::size_t latch, std::size_t head_if_pos,
                clang::IfStmt *head_if, std::size_t exit_idx, unsigned &recovered,
                const patchestry::Options &options
            ) {
                if (exit_idx == 0 || exit_idx >= stmts.size()) {
                    return false;
                }

                auto *then_goto = llvm::dyn_cast< clang::GotoStmt >(head_if->getThen());
                auto *else_goto = llvm::dyn_cast< clang::GotoStmt >(head_if->getElse());
                if (then_goto == nullptr || else_goto == nullptr) {
                    return false;
                }

                auto then_it = label_index.find(then_goto->getLabel());
                if (then_it == label_index.end()) {
                    return false;
                }
                std::size_t then_idx = then_it->second;
                if (then_idx <= header || then_idx > latch) {
                    return false;
                }

                std::vector< clang::Stmt * > body_stmts;
                for (std::size_t k = head_if_pos + 1; k < latch; ++k) {
                    body_stmts.push_back(stmts[k]);
                }
                if (!body_stmts.empty()) {
                    auto *last = body_stmts.back();
                    if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(last)) {
                        auto *head_label = llvm::dyn_cast< clang::LabelStmt >(stmts[header]);
                        if (head_label != nullptr
                            && gs->getLabel() == head_label->getDecl())
                        {
                            body_stmts.pop_back();
                        }
                    }
                }
                if (body_stmts.empty()) {
                    body_stmts.push_back(
                        new (ctx) clang::NullStmt(head_if->getIfLoc(), false)
                    );
                }

                auto loc            = stmts[header]->getBeginLoc();
                auto *body_compound = makeCompound(ctx, body_stmts, loc, loc);
                auto *while_stmt    = clang::WhileStmt::Create(
                    ctx, nullptr, head_if->getCond(), body_compound, loc, loc, loc
                );

                auto *head_label = llvm::dyn_cast< clang::LabelStmt >(stmts[header]);
                auto *new_head   = new (ctx) clang::LabelStmt(
                    head_label->getIdentLoc(), head_label->getDecl(), while_stmt
                );

                std::vector< clang::Stmt * > new_stmts;
                for (std::size_t k = 0; k < header; ++k) {
                    new_stmts.push_back(stmts[k]);
                }
                new_stmts.push_back(new_head);
                for (std::size_t k = header + 1; k < head_if_pos; ++k) {
                    new_stmts.push_back(stmts[k]);
                }
                for (std::size_t k = exit_idx; k < stmts.size(); ++k) {
                    new_stmts.push_back(stmts[k]);
                }
                stmts = std::move(new_stmts);
                ++recovered;

                if (options.verbose) {
                    LOG(DEBUG) << "NaturalLoopRecoveryPass: emitted WhileStmt for natural "
                               << "loop (header=" << header << ", latch=" << latch << ")\n";
                }
                return true;
            }
        };

        // =========================================================================
        // BackedgeLoopStructurizePass
        // =========================================================================

        class BackedgeLoopStructurizePass final : public ASTPass
        {
          public:
            explicit BackedgeLoopStructurizePass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "BackedgeLoopStructurizePass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                unsigned local_converted = 0;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }
                    processFunction(ctx, func, local_converted, options);
                }

                state.backedge_loops_structured += local_converted;
                if (local_converted > 0) {
                    if (options.verbose) {
                        LOG(DEBUG)
                            << "BackedgeLoopStructurizePass: converted " << local_converted
                            << " back-edge goto(s) to while loop(s)\n";
                    }
                    runAstCleanupPass(state, ctx, options);
                }
                return true;
            }

          private:
            PipelineState &state;

            void processFunction(
                clang::ASTContext &ctx, clang::FunctionDecl *func, unsigned &converted,
                const patchestry::Options & /*options*/
            ) {
                bool any_changed = true;
                while (any_changed) {
                    auto *body =
                        llvm::dyn_cast_or_null< clang::CompoundStmt >(func->getBody());
                    if (body == nullptr) {
                        break;
                    }
                    std::vector< clang::Stmt * > stmts(body->body_begin(), body->body_end());
                    const auto label_index = topLevelLabelIndex(body);
                    any_changed = tryConvert(ctx, func, stmts, label_index, converted);
                }
            }

            static bool tryConvert(
                clang::ASTContext &ctx, clang::FunctionDecl *func,
                std::vector< clang::Stmt * > &stmts,
                const std::unordered_map< const clang::LabelDecl *, std::size_t >
                    & /*label_index*/,
                unsigned &converted
            ) {
                const std::size_t N = stmts.size();

                for (std::size_t h = 0; h < N; ++h) {
                    auto *head_label = llvm::dyn_cast< clang::LabelStmt >(stmts[h]);
                    if (head_label == nullptr) {
                        continue;
                    }
                    const auto *lhead_decl = head_label->getDecl();

                    for (std::size_t latch = h + 2; latch < N; ++latch) {
                        auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmts[latch]);
                        if (gs == nullptr || gs->getLabel() != lhead_decl) {
                            continue;
                        }

                        const std::size_t p = latch - 1U;
                        auto *exit_if       = llvm::dyn_cast< clang::IfStmt >(stmts[p]);
                        if (exit_if == nullptr || exit_if->getElse() != nullptr) {
                            continue;
                        }

                        bool has_intermediate_backedge = false;
                        for (std::size_t k = h + 1U; k < p; ++k) {
                            if (containsGotoTo(stmts[k], lhead_decl)) {
                                has_intermediate_backedge = true;
                                break;
                            }
                        }
                        if (has_intermediate_backedge) {
                            continue;
                        }

                        auto *exit_block = exit_if->getThen();
                        if (containsGotoTo(exit_block, lhead_decl)) {
                            continue;
                        }

                        const auto loc = exit_if->getIfLoc();

                        auto *neg_cond = clang::UnaryOperator::Create(
                            ctx,
                            ensureRValue(
                                ctx, const_cast< clang::Expr * >(exit_if->getCond())
                            ),
                            clang::UO_LNot, ctx.IntTy, clang::VK_PRValue,
                            clang::OK_Ordinary, loc, false, clang::FPOptionsOverride()
                        );

                        std::vector< clang::Stmt * > body_stmts;
                        auto *sub = head_label->getSubStmt();
                        if (sub != nullptr && !llvm::isa< clang::NullStmt >(sub)) {
                            body_stmts.push_back(sub);
                        }
                        for (std::size_t k = h + 1U; k < p; ++k) {
                            body_stmts.push_back(stmts[k]);
                        }
                        auto *body_compound = makeCompound(ctx, body_stmts, loc, loc);

                        auto *while_stmt = clang::WhileStmt::Create(
                            ctx, nullptr, neg_cond, body_compound, loc, loc,
                            exit_if->getEndLoc()
                        );

                        std::vector< clang::Stmt * > after_loop;
                        if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(exit_block)) {
                            for (auto *child : cs->body()) {
                                after_loop.push_back(child);
                            }
                        } else {
                            after_loop.push_back(exit_block);
                        }

                        std::vector< clang::Stmt * > new_stmts(
                            stmts.begin(),
                            stmts.begin() + static_cast< std::ptrdiff_t >(h)
                        );
                        new_stmts.push_back(while_stmt);
                        for (auto *s : after_loop) {
                            new_stmts.push_back(s);
                        }
                        for (std::size_t k = latch + 1U; k < N; ++k) {
                            new_stmts.push_back(stmts[k]);
                        }

                        auto *body = llvm::dyn_cast< clang::CompoundStmt >(func->getBody());
                        func->setBody(makeCompound(
                            ctx, new_stmts, body->getLBracLoc(), body->getRBracLoc()
                        ));
                        ++converted;
                        return true;
                    }
                }
                return false;
            }
        };

        class WhileToForUpgradePass final : public ASTPass
        {
          public:
            explicit WhileToForUpgradePass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "WhileToForUpgradePass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                unsigned upgraded = 0;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }
                    auto *body =
                        llvm::dyn_cast_or_null< clang::CompoundStmt >(func->getBody());
                    if (body == nullptr) {
                        continue;
                    }
                    auto *new_body = processCompound(ctx, body, upgraded);
                    if (new_body != body) {
                        func->setBody(new_body);
                    }
                }

                state.for_loops_upgraded += upgraded;
                if (upgraded > 0 && options.verbose) {
                    LOG(DEBUG) << "WhileToForUpgradePass: upgraded " << upgraded
                               << " while loop(s) to for loops\n";
                }
                return true;
            }

          private:
            PipelineState &state;

            // Returns the VarDecl being assigned if stmt is a plain assignment (=)
            // with a DeclRefExpr on the LHS; nullptr otherwise.
            static const clang::VarDecl *extractInitVar(const clang::Stmt *stmt) {
                const auto *bo = llvm::dyn_cast_or_null< clang::BinaryOperator >(stmt);
                if (bo == nullptr || bo->getOpcode() != clang::BO_Assign) {
                    return nullptr;
                }
                const auto *lhs_dre = llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                    bo->getLHS()->IgnoreParenImpCasts()
                );
                if (lhs_dre == nullptr) {
                    return nullptr;
                }
                return llvm::dyn_cast_or_null< clang::VarDecl >(lhs_dre->getDecl());
            }

            // Returns true if cond is a comparison operator that references loop_var.
            static bool condUsesVar(
                const clang::Expr *cond, const clang::VarDecl *loop_var
            ) {
                const auto *bin = llvm::dyn_cast_or_null< clang::BinaryOperator >(
                    cond->IgnoreParenImpCasts()
                );
                if (bin == nullptr || !bin->isComparisonOp()) {
                    return false;
                }
                const auto *lhs_dre = llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                    bin->getLHS()->IgnoreParenImpCasts()
                );
                const auto *rhs_dre = llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                    bin->getRHS()->IgnoreParenImpCasts()
                );
                return (lhs_dre != nullptr && lhs_dre->getDecl() == loop_var)
                       || (rhs_dre != nullptr && rhs_dre->getDecl() == loop_var);
            }

            // Returns the increment Expr from stmt if it modifies loop_var in one
            // of the recognized patterns (++var, var+=expr, var=var±expr); nullptr
            // otherwise.
            static clang::Expr *extractIncrExpr(
                clang::Stmt *stmt, const clang::VarDecl *loop_var
            ) {
                if (auto *uo = llvm::dyn_cast_or_null< clang::UnaryOperator >(stmt)) {
                    if (uo->isIncrementOp()) {
                        const auto *sub = llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                            uo->getSubExpr()->IgnoreParenImpCasts()
                        );
                        if (sub != nullptr && sub->getDecl() == loop_var) {
                            return uo;
                        }
                    }
                } else if (auto *bo = llvm::dyn_cast_or_null< clang::BinaryOperator >(stmt)) {
                    if (bo->getOpcode() == clang::BO_AddAssign) {
                        const auto *lhs_dre = llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                            bo->getLHS()->IgnoreParenImpCasts()
                        );
                        if (lhs_dre != nullptr && lhs_dre->getDecl() == loop_var) {
                            return bo;
                        }
                    } else if (bo->isAssignmentOp()) {
                        const auto *lhs_dre = llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                            bo->getLHS()->IgnoreParenImpCasts()
                        );
                        if (lhs_dre != nullptr && lhs_dre->getDecl() == loop_var) {
                            const auto *rhs = bo->getRHS()->IgnoreParenImpCasts();
                            if (const auto *rhs_bin =
                                    llvm::dyn_cast< clang::BinaryOperator >(rhs))
                            {
                                if (rhs_bin->getOpcode() == clang::BO_Add
                                    || rhs_bin->getOpcode() == clang::BO_Sub)
                                {
                                    const auto *rhs_lhs =
                                        llvm::dyn_cast_or_null< clang::DeclRefExpr >(
                                            rhs_bin->getLHS()->IgnoreParenImpCasts()
                                        );
                                    if (rhs_lhs != nullptr
                                        && rhs_lhs->getDecl() == loop_var)
                                    {
                                        return bo;
                                    }
                                }
                            }
                        }
                    }
                }
                return nullptr;
            }

            // Returns true if stmt contains a ContinueStmt that is not nested inside
            // another loop (while/for/do), which would make a while→for rewrite unsafe.
            static bool bodyHasDirectContinue(const clang::Stmt *stmt) {
                if (stmt == nullptr) {
                    return false;
                }
                if (llvm::isa< clang::WhileStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt)
                    || llvm::isa< clang::DoStmt >(stmt))
                {
                    return false;
                }
                if (llvm::isa< clang::ContinueStmt >(stmt)) {
                    return true;
                }
                for (const auto *child : stmt->children()) {
                    if (bodyHasDirectContinue(child)) {
                        return true;
                    }
                }
                return false;
            }

            // Attempt to upgrade stmts[i] (init) + stmts[i+1] (while) to a ForStmt.
            // On success, stmts[i] is replaced with the ForStmt, stmts[i+1] is erased,
            // and upgraded is incremented.
            static bool tryUpgradeInCompound(
                clang::ASTContext &ctx, std::vector< clang::Stmt * > &stmts,
                std::size_t i, unsigned &upgraded
            ) {
                if (i + 1 >= stmts.size()) {
                    return false;
                }
                const clang::VarDecl *loop_var = extractInitVar(stmts[i]);
                if (loop_var == nullptr) {
                    return false;
                }
                auto *ws = llvm::dyn_cast_or_null< clang::WhileStmt >(stmts[i + 1]);
                if (ws == nullptr) {
                    return false;
                }
                if (!condUsesVar(ws->getCond(), loop_var)) {
                    return false;
                }
                auto *body = llvm::dyn_cast_or_null< clang::CompoundStmt >(ws->getBody());
                if (body == nullptr || body->body_empty()) {
                    return false;
                }
                clang::Expr *incr_expr = extractIncrExpr(body->body_back(), loop_var);
                if (incr_expr == nullptr) {
                    return false;
                }
                if (bodyHasDirectContinue(body)) {
                    return false;
                }

                std::vector< clang::Stmt * > body_stmts(
                    body->body_begin(), std::prev(body->body_end())
                );
                const auto loc = stmts[i]->getBeginLoc();
                auto *new_body =
                    makeCompound(ctx, body_stmts, body->getLBracLoc(), body->getRBracLoc());
                auto *for_stmt = new (ctx) clang::ForStmt(
                    ctx, stmts[i], ws->getCond(), nullptr, incr_expr, new_body, loc, loc, loc
                );
                stmts[i] = for_stmt;
                stmts.erase(stmts.begin() + static_cast< std::ptrdiff_t >(i) + 1);
                ++upgraded;
                return true;
            }

            // Recursively rewrite s, descending into loops and conditionals.
            static clang::Stmt *rewriteStmt(
                clang::ASTContext &ctx, clang::Stmt *s, unsigned &upgraded
            ) {
                if (s == nullptr) {
                    return s;
                }
                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                    return processCompound(ctx, cs, upgraded);
                }
                if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
                    auto *body     = ws->getBody();
                    auto *new_body = rewriteStmt(ctx, body, upgraded);
                    if (new_body == body) {
                        return ws;
                    }
                    return clang::WhileStmt::Create(
                        ctx, nullptr, ws->getCond(), new_body, ws->getWhileLoc(),
                        ws->getLParenLoc(), ws->getRParenLoc()
                    );
                }
                if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
                    auto *body     = fs->getBody();
                    auto *new_body = rewriteStmt(ctx, body, upgraded);
                    if (new_body == body) {
                        return fs;
                    }
                    return new (ctx) clang::ForStmt(
                        ctx, fs->getInit(), fs->getCond(), nullptr, fs->getInc(),
                        new_body, fs->getForLoc(), fs->getLParenLoc(), fs->getRParenLoc()
                    );
                }
                if (auto *is = llvm::dyn_cast< clang::IfStmt >(s)) {
                    auto *then_s   = is->getThen();
                    auto *else_s   = is->getElse();
                    auto *new_then = rewriteStmt(ctx, then_s, upgraded);
                    auto *new_else =
                        else_s != nullptr ? rewriteStmt(ctx, else_s, upgraded) : nullptr;
                    if (new_then == then_s && new_else == else_s) {
                        return is;
                    }
                    return clang::IfStmt::Create(
                        ctx, is->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr,
                        nullptr, is->getCond(), is->getLParenLoc(),
                        new_then->getBeginLoc(), new_then,
                        new_else != nullptr ? new_else->getBeginLoc()
                                            : clang::SourceLocation(),
                        new_else
                    );
                }
                return s;
            }

            // Process a CompoundStmt: recurse into nested statements first, then
            // scan for adjacent (init, while) pairs and upgrade them to for loops.
            static clang::CompoundStmt *processCompound(
                clang::ASTContext &ctx, clang::CompoundStmt *cs, unsigned &upgraded
            ) {
                std::vector< clang::Stmt * > stmts(cs->body_begin(), cs->body_end());

                for (auto &s : stmts) {
                    s = rewriteStmt(ctx, s, upgraded);
                }

                bool any_upgraded = true;
                while (any_upgraded) {
                    any_upgraded = false;
                    for (std::size_t i = 0; i + 1 < stmts.size(); ++i) {
                        if (tryUpgradeInCompound(ctx, stmts, i, upgraded)) {
                            any_upgraded = true;
                            break;
                        }
                    }
                }

                return makeCompound(ctx, stmts, cs->getLBracLoc(), cs->getRBracLoc());
            }
        };

    } // anonymous namespace

    namespace detail {

        void addWhileLoopStructurizePass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< WhileLoopStructurizePass >(state));
        }

        void addLoopStructurizePass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< LoopStructurizePass >(state));
        }

        void addDegenerateLoopUnwrapPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< DegenerateLoopUnwrapPass >(state));
        }

        void addLoopConditionRecoveryPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< LoopConditionRecoveryPass >(state));
        }

        void addDegenerateWhileElimPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< DegenerateWhileElimPass >(state));
        }

        void addNaturalLoopRecoveryPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< NaturalLoopRecoveryPass >(state));
        }

        void addBackedgeLoopStructurizePass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< BackedgeLoopStructurizePass >(state));
        }

        void addWhileToForUpgradePass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< WhileToForUpgradePass >(state));
        }

        void addLoopPasses(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< WhileLoopStructurizePass >(state));
            pm.add_pass(std::make_unique< LoopStructurizePass >(state));
            pm.add_pass(std::make_unique< DegenerateLoopUnwrapPass >(state));
            pm.add_pass(std::make_unique< LoopConditionRecoveryPass >(state));
            pm.add_pass(std::make_unique< DegenerateWhileElimPass >(state));
            pm.add_pass(std::make_unique< NaturalLoopRecoveryPass >(state));
            pm.add_pass(std::make_unique< BackedgeLoopStructurizePass >(state));
        }

    } // namespace detail

} // namespace patchestry::ast
