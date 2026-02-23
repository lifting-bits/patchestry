/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// Switch / irreducible-flow passes:
//   SwitchRecoveryPass      – replace IndirectGotoStmt with a SwitchStmt skeleton
//   IrreducibleFallbackPass – wrap irreducible goto bodies in while(1) and convert
//                             goto statements to break/continue

#include <unordered_map>
#include <vector>

#include <patchestry/AST/ASTPassManager.hpp>
#include <patchestry/Util/Log.hpp>

#include "NormalizationPipelineInternal.hpp"

namespace patchestry::ast {
    namespace {

        using namespace detail;

        // =========================================================================
        // File-local helpers
        // =========================================================================

        // Rewrites goto/indirect-goto statements to break/continue so the body can be
        // placed inside a while(1) loop.  Backward gotos (target index ≤ current_index)
        // become continue; forward gotos become break.
        static clang::Stmt *rewriteStmtFallback(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< const clang::LabelDecl *, std::size_t > &label_index,
            std::size_t current_index, unsigned &rewrites
        ) {
            if (stmt == nullptr) {
                return nullptr;
            }

            if (auto *goto_stmt = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                ++rewrites;
                if (label_index.contains(goto_stmt->getLabel())
                    && label_index.at(goto_stmt->getLabel()) <= current_index)
                {
                    return new (ctx) clang::ContinueStmt(goto_stmt->getGotoLoc());
                }
                return new (ctx) clang::BreakStmt(goto_stmt->getGotoLoc());
            }

            if (auto *indirect = llvm::dyn_cast< clang::IndirectGotoStmt >(stmt)) {
                ++rewrites;
                return new (ctx) clang::BreakStmt(indirect->getGotoLoc());
            }

            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                return rewriteStmtFallback(
                    ctx, label->getSubStmt(), label_index, current_index, rewrites
                );
            }

            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > rewritten;
                for (auto *child : compound->body()) {
                    auto *new_child = rewriteStmtFallback(
                        ctx, child, label_index, current_index, rewrites
                    );
                    if (new_child != nullptr) {
                        rewritten.push_back(new_child);
                    }
                }
                return makeCompound(
                    ctx, rewritten, compound->getLBracLoc(), compound->getRBracLoc()
                );
            }

            if (auto *if_stmt = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                auto *new_then = rewriteStmtFallback(
                    ctx, if_stmt->getThen(), label_index, current_index, rewrites
                );
                auto *new_else = rewriteStmtFallback(
                    ctx, if_stmt->getElse(), label_index, current_index, rewrites
                );
                if (new_then == nullptr) {
                    new_then = new (ctx) clang::NullStmt(if_stmt->getIfLoc(), false);
                }
                return clang::IfStmt::Create(
                    ctx, if_stmt->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr, nullptr,
                    if_stmt->getCond(), if_stmt->getLParenLoc(), new_then->getBeginLoc(), new_then,
                    new_else != nullptr ? new_else->getBeginLoc() : clang::SourceLocation(),
                    new_else
                );
            }

            return stmt;
        }

        // Returns true if `stmt` (at position `current_idx` in the flat body) contains
        // any GotoStmt whose target label has an index ≤ current_idx — i.e., a back-edge.
        static bool stmtHasBackwardGoto(
            const clang::Stmt *stmt, std::size_t current_idx,
            const std::unordered_map< const clang::LabelDecl *, std::size_t > &label_index
        ) {
            if (stmt == nullptr) {
                return false;
            }
            if (const auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                auto it = label_index.find(gs->getLabel());
                return it != label_index.end() && it->second <= current_idx;
            }
            for (const auto *child : stmt->children()) {
                if (stmtHasBackwardGoto(child, current_idx, label_index)) {
                    return true;
                }
            }
            return false;
        }

        // Returns true if the function body contains at least one backward goto (loop back-edge).
        // Functions whose remaining gotos are all forward do not need a while(1) wrapper.
        static bool bodyHasBackwardGoto(
            const clang::CompoundStmt *body,
            const std::unordered_map< const clang::LabelDecl *, std::size_t > &label_index
        ) {
            std::size_t idx = 0;
            for (const auto *stmt : body->body()) {
                if (stmtHasBackwardGoto(stmt, idx, label_index)) {
                    return true;
                }
                ++idx;
            }
            return false;
        }

        // =========================================================================
        // SwitchRecoveryPass
        // =========================================================================

        class SwitchRecoveryPass final : public ASTPass
        {
          public:
            explicit SwitchRecoveryPass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "SwitchRecoveryPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                state.indirect_switches_built = 0;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }

                    auto *body = llvm::dyn_cast< clang::CompoundStmt >(func->getBody());
                    if (body == nullptr) {
                        continue;
                    }

                    std::vector< clang::Stmt * > stmts;
                    stmts.reserve(body->size());
                    bool changed = false;
                    for (auto *stmt : body->body()) {
                        auto rewrite_stmt = [&](auto &&self, clang::Stmt *node) -> clang::Stmt * {
                            if (node == nullptr) {
                                return nullptr;
                            }
                            if (auto *indirect =
                                    llvm::dyn_cast< clang::IndirectGotoStmt >(node))
                            {
                                auto *switch_stmt = clang::SwitchStmt::Create(
                                    ctx, nullptr, nullptr,
                                    makeIntLiteral(ctx, 0, ctx.IntTy, indirect->getBeginLoc()),
                                    indirect->getBeginLoc(), indirect->getEndLoc()
                                );
                                std::vector< clang::Stmt * > sw_body;
                                const auto case_count = std::max< std::size_t >(
                                    state.cfgs.empty()
                                        ? 0
                                        : state.cfgs.front().addr_label_exprs.size(),
                                    1U
                                );
                                for (std::size_t i = 0; i < case_count; ++i) {
                                    auto *case_stmt = clang::CaseStmt::Create(
                                        ctx,
                                        makeIntLiteral(
                                            ctx, i, ctx.IntTy, indirect->getBeginLoc()
                                        ),
                                        nullptr, indirect->getBeginLoc(),
                                        indirect->getBeginLoc(), indirect->getBeginLoc()
                                    );
                                    case_stmt->setSubStmt(
                                        new (ctx) clang::BreakStmt(indirect->getBeginLoc())
                                    );
                                    sw_body.push_back(case_stmt);
                                }
                                sw_body.push_back(new (ctx) clang::DefaultStmt(
                                    indirect->getBeginLoc(), indirect->getEndLoc(),
                                    new (ctx) clang::BreakStmt(indirect->getBeginLoc())
                                ));
                                switch_stmt->setBody(makeCompound(
                                    ctx, sw_body, indirect->getBeginLoc(), indirect->getEndLoc()
                                ));
                                ++state.indirect_switches_built;
                                changed = true;
                                return switch_stmt;
                            }
                            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(node)) {
                                std::vector< clang::Stmt * > children;
                                for (auto *child : compound->body()) {
                                    auto *rewritten = self(self, child);
                                    if (rewritten != nullptr) {
                                        children.push_back(rewritten);
                                    }
                                }
                                return makeCompound(
                                    ctx, children, compound->getLBracLoc(),
                                    compound->getRBracLoc()
                                );
                            }
                            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(node)) {
                                auto *sub = self(self, label->getSubStmt());
                                if (sub == nullptr) {
                                    sub = new (ctx) clang::NullStmt(label->getBeginLoc(), false);
                                }
                                return new (ctx)
                                    clang::LabelStmt(label->getIdentLoc(), label->getDecl(), sub);
                            }
                            if (auto *if_stmt = llvm::dyn_cast< clang::IfStmt >(node)) {
                                auto *new_then = self(self, if_stmt->getThen());
                                auto *new_else = self(self, if_stmt->getElse());
                                if (new_then == nullptr) {
                                    new_then =
                                        new (ctx) clang::NullStmt(if_stmt->getIfLoc(), false);
                                }
                                return clang::IfStmt::Create(
                                    ctx, if_stmt->getIfLoc(),
                                    clang::IfStatementKind::Ordinary, nullptr, nullptr,
                                    if_stmt->getCond(), if_stmt->getLParenLoc(),
                                    new_then->getBeginLoc(), new_then,
                                    new_else != nullptr ? new_else->getBeginLoc()
                                                        : clang::SourceLocation(),
                                    new_else
                                );
                            }
                            return node;
                        };

                        stmts.push_back(rewrite_stmt(rewrite_stmt, stmt));
                    }

                    if (changed) {
                        func->setBody(makeCompound(
                            ctx, stmts, body->getLBracLoc(), body->getRBracLoc()
                        ));
                    }
                }

                runCfgExtractPass(state, ctx, options);
                return true;
            }

          private:
            PipelineState &state;
        };

        // =========================================================================
        // IrreducibleFallbackPass
        // =========================================================================

        class IrreducibleFallbackPass final : public ASTPass
        {
          public:
            explicit IrreducibleFallbackPass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "IrreducibleFallbackPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                state.fallback_rewrites         = 0;
                state.used_irreducible_fallback = false;

                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }

                    auto *body = llvm::dyn_cast< clang::CompoundStmt >(func->getBody());
                    if (body == nullptr) {
                        continue;
                    }

                    auto label_index = topLevelLabelIndex(body);

                    // Only wrap in while(1) if there is at least one backward goto.
                    // Functions with only forward gotos have an acyclic CFG — wrapping them
                    // would produce dead continue statements and obscure the structure.
                    if (!bodyHasBackwardGoto(body, label_index)) {
                        continue;
                    }

                    unsigned before_rewrites = state.fallback_rewrites;
                    std::vector< clang::Stmt * > rewritten_body;
                    rewritten_body.reserve(body->size() + 1U);
                    std::size_t idx = 0;
                    for (auto *stmt : body->body()) {
                        auto *rewritten = rewriteStmtFallback(
                            ctx, stmt, label_index, idx, state.fallback_rewrites
                        );
                        if (rewritten != nullptr) {
                            rewritten_body.push_back(rewritten);
                        }
                        ++idx;
                    }

                    if (state.fallback_rewrites == before_rewrites) {
                        continue;
                    }

                    state.used_irreducible_fallback = true;
                    rewritten_body.push_back(new (ctx) clang::BreakStmt(body->getRBracLoc()));
                    auto *loop_body = makeCompound(
                        ctx, rewritten_body, body->getLBracLoc(), body->getRBracLoc()
                    );
                    auto *loop_stmt = clang::WhileStmt::Create(
                        ctx, nullptr, makeBoolTrue(ctx, body->getLBracLoc()), loop_body,
                        body->getLBracLoc(), body->getRBracLoc(), body->getRBracLoc()
                    );
                    std::vector< clang::Stmt * > wrapped{ loop_stmt };
                    func->setBody(makeCompound(
                        ctx, wrapped, body->getLBracLoc(), body->getRBracLoc()
                    ));
                }

                runCfgExtractPass(state, ctx, options);
                return true;
            }

          private:
            PipelineState &state;
        };

    } // anonymous namespace

    namespace detail {

        void addSwitchRecoveryPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< SwitchRecoveryPass >(state));
        }

        void addIrreducibleFallbackPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< IrreducibleFallbackPass >(state));
        }

        void addSwitchPasses(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< SwitchRecoveryPass >(state));
            pm.add_pass(std::make_unique< IrreducibleFallbackPass >(state));
        }

    } // namespace detail

} // namespace patchestry::ast
