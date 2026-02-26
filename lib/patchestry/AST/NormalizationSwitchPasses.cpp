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
#include <unordered_set>
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
                auto *new_sub = rewriteStmtFallback(
                    ctx, label->getSubStmt(), label_index, current_index, rewrites
                );
                if (new_sub == nullptr) {
                    new_sub = new (ctx) clang::NullStmt(label->getBeginLoc(), false);
                }
                return new (ctx)
                    clang::LabelStmt(label->getIdentLoc(), label->getDecl(), new_sub);
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

        // =========================================================================
        // SwitchGotoInliningPass
        // =========================================================================

        // After IrreducibleFallbackPass has converted all backward gotos inside
        // while(1)-wrapped compound bodies to continue/break, any switch whose
        // case arms are pure forward-goto-to-sibling-label sequences can be cleaned
        // up by inlining the label body directly into the case arm.  The label is
        // then removed from the enclosing compound (when all its goto references
        // originated inside this switch).

        class SwitchGotoInliningPass final : public ASTPass
        {
          public:
            explicit SwitchGotoInliningPass(PipelineState &state) : state(state) {}

            const char *name(void) const override { return "SwitchGotoInliningPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                state.switch_cases_inlined = 0;

                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }
                    auto *body = llvm::dyn_cast< clang::CompoundStmt >(func->getBody());
                    if (body == nullptr) {
                        continue;
                    }

                    // Count all goto targets once per function before any inlining.
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

            // Returns the single GotoStmt that is the entire body of a CaseStmt or
            // DefaultStmt, or nullptr if the body is anything other than a lone goto.
            static const clang::GotoStmt *getSingleGoto(const clang::Stmt *case_or_default) {
                const clang::Stmt *sub = nullptr;
                if (const auto *cs = llvm::dyn_cast_or_null< clang::CaseStmt >(case_or_default))
                {
                    sub = cs->getSubStmt();
                } else if (const auto *ds =
                               llvm::dyn_cast_or_null< clang::DefaultStmt >(case_or_default))
                {
                    sub = ds->getSubStmt();
                }
                return llvm::dyn_cast_or_null< clang::GotoStmt >(sub);
            }

            // Maps a case/default child Stmt* to the full ordered body to inline
            // (the label's sub-stmt + all trailing siblings up to the next label).
            using CaseBodyMap =
                std::unordered_map< clang::Stmt *, std::vector< clang::Stmt * > >;

            // For a label at flat index `label_pos` inside `all_stmts`, returns the
            // full sequence of statements to inline: the label's sub-stmt (if non-null)
            // followed by all trailing siblings up to — but not including — the next
            // LabelStmt or any BreakStmt.  BreakStmts are excluded because they exit
            // the enclosing while loop at the compound level but would only exit the
            // switch if inlined into a case arm, changing semantics.
            static std::vector< clang::Stmt * > buildLabelBody(
                const std::vector< clang::Stmt * > &all_stmts, std::size_t label_pos
            ) {
                auto *ls = llvm::cast< clang::LabelStmt >(all_stmts[label_pos]);
                std::vector< clang::Stmt * > body;

                clang::Stmt *sub = ls->getSubStmt();
                if (sub != nullptr && !llvm::isa< clang::NullStmt >(sub)) {
                    body.push_back(sub);
                }

                for (std::size_t j = label_pos + 1; j < all_stmts.size(); ++j) {
                    if (llvm::isa< clang::LabelStmt >(all_stmts[j])) {
                        break;
                    }
                    if (llvm::isa< clang::BreakStmt >(all_stmts[j])) {
                        break; // leave while-loop exit break in the compound
                    }
                    body.push_back(all_stmts[j]);
                }
                return body;
            }

            // Rebuilds `sw` using `case_bodies` for per-case inlining decisions.
            // Qualifying arms get the pre-built body vector inlined as a CompoundStmt;
            // non-qualifying arms are kept unchanged.
            clang::SwitchStmt *buildInlinedSwitch(
                clang::ASTContext &ctx, clang::SwitchStmt *sw, const CaseBodyMap &case_bodies
            ) {
                const auto *sw_body = llvm::dyn_cast< clang::CompoundStmt >(sw->getBody());
                std::vector< clang::Stmt * > new_cases;
                new_cases.reserve(sw_body->size());

                for (auto *child : sw_body->body()) {
                    auto it = case_bodies.find(child);
                    if (it == case_bodies.end()) {
                        new_cases.push_back(child);
                        continue;
                    }

                    clang::Stmt *inlined_body = makeCompound(ctx, it->second);
                    ++state.switch_cases_inlined;

                    if (auto *cs = llvm::dyn_cast< clang::CaseStmt >(child)) {
                        auto *new_cs = clang::CaseStmt::Create(
                            ctx, cs->getLHS(), cs->getRHS(), cs->getCaseLoc(),
                            cs->getEllipsisLoc(), cs->getColonLoc()
                        );
                        new_cs->setSubStmt(inlined_body);
                        new_cases.push_back(new_cs);
                    } else if (auto *ds = llvm::dyn_cast< clang::DefaultStmt >(child)) {
                        new_cases.push_back(new (ctx) clang::DefaultStmt(
                            ds->getDefaultLoc(), ds->getColonLoc(), inlined_body
                        ));
                    } else {
                        new_cases.push_back(child);
                    }
                }

                auto *new_sw = clang::SwitchStmt::Create(
                    ctx, nullptr, nullptr, sw->getCond(), sw->getLParenLoc(), sw->getRParenLoc()
                );
                new_sw->setBody(
                    makeCompound(ctx, new_cases, sw_body->getLBracLoc(), sw_body->getRBracLoc())
                );
                return new_sw;
            }

            clang::CompoundStmt *processCompound(
                clang::ASTContext &ctx, clang::CompoundStmt *compound,
                const RefCountMap &ref_count
            ) {
                // Snapshot the flat statement list for index-based operations.
                std::vector< clang::Stmt * > all_stmts(
                    compound->body_begin(), compound->body_end()
                );
                const std::size_t N = all_stmts.size();

                auto label_idx = topLevelLabelIndex(compound);

                // Build the full body range for every sibling label.
                std::unordered_map< const clang::LabelDecl *, std::vector< clang::Stmt * > >
                    label_bodies;
                for (std::size_t i = 0; i < N; ++i) {
                    if (const auto *ls = llvm::dyn_cast< clang::LabelStmt >(all_stmts[i])) {
                        label_bodies[ls->getDecl()] = buildLabelBody(all_stmts, i);
                    }
                }

                // For each switch in this compound collect per-case inline decisions.
                std::unordered_map< clang::SwitchStmt *, CaseBodyMap > switch_inlines;
                std::unordered_set< const clang::LabelDecl * > absorbed_labels;

                for (std::size_t i = 0; i < N; ++i) {
                    auto *sw = llvm::dyn_cast< clang::SwitchStmt >(all_stmts[i]);
                    if (sw == nullptr) {
                        continue;
                    }
                    const auto *sw_body =
                        llvm::dyn_cast_or_null< clang::CompoundStmt >(sw->getBody());
                    if (sw_body == nullptr || sw_body->body_empty()) {
                        continue;
                    }

                    CaseBodyMap case_map;
                    std::unordered_map< const clang::LabelDecl *, unsigned > inlined_count;

                    for (auto *child : sw_body->body()) {
                        const auto *gs = getSingleGoto(child);
                        if (gs == nullptr) {
                            continue;
                        }
                        const auto *lbl = gs->getLabel();
                        auto idx_it     = label_idx.find(lbl);
                        if (idx_it == label_idx.end() || idx_it->second <= i) {
                            continue; // not a forward sibling
                        }
                        auto body_it = label_bodies.find(lbl);
                        if (body_it == label_bodies.end()) {
                            continue;
                        }
                        case_map[child] = body_it->second;
                        ++inlined_count[lbl];
                    }

                    if (case_map.empty()) {
                        continue;
                    }

                    // Mark labels fully absorbed by this switch (all refs covered).
                    for (const auto &[lbl, count] : inlined_count) {
                        auto rc_it     = ref_count.find(lbl);
                        unsigned total = (rc_it != ref_count.end()) ? rc_it->second : 0u;
                        if (total == count) {
                            absorbed_labels.insert(lbl);
                        }
                    }

                    switch_inlines.emplace(sw, std::move(case_map));
                }

                if (switch_inlines.empty()) {
                    // Nothing to inline at this level; recurse into nested stmts.
                    std::vector< clang::Stmt * > new_stmts;
                    bool changed = false;
                    for (auto *stmt : all_stmts) {
                        auto *processed = processStmt(ctx, stmt, ref_count);
                        new_stmts.push_back(processed);
                        changed |= (processed != stmt);
                    }
                    if (!changed) {
                        return compound;
                    }
                    return makeCompound(
                        ctx, new_stmts, compound->getLBracLoc(), compound->getRBracLoc()
                    );
                }

                // Compute which flat positions to skip: absorbed LabelStmts and their
                // owned trailing siblings (same range as buildLabelBody, excluding any
                // leading BreakStmt which must remain for the while-loop exit path).
                std::unordered_set< std::size_t > positions_to_skip;
                for (std::size_t i = 0; i < N; ++i) {
                    const auto *ls = llvm::dyn_cast< clang::LabelStmt >(all_stmts[i]);
                    if (ls == nullptr || !absorbed_labels.count(ls->getDecl())) {
                        continue;
                    }
                    positions_to_skip.insert(i); // the LabelStmt itself
                    for (std::size_t j = i + 1; j < N; ++j) {
                        if (llvm::isa< clang::LabelStmt >(all_stmts[j])) {
                            break;
                        }
                        if (llvm::isa< clang::BreakStmt >(all_stmts[j])) {
                            break; // keep the while-loop exit break in the compound
                        }
                        positions_to_skip.insert(j);
                    }
                }

                // Rebuild: inline candidate switches, drop skipped positions,
                // recurse into everything else.
                std::vector< clang::Stmt * > new_stmts;
                new_stmts.reserve(N);

                for (std::size_t i = 0; i < N; ++i) {
                    if (positions_to_skip.count(i)) {
                        continue;
                    }

                    if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(all_stmts[i])) {
                        auto it = switch_inlines.find(sw);
                        if (it != switch_inlines.end()) {
                            new_stmts.push_back(buildInlinedSwitch(ctx, sw, it->second));
                            continue;
                        }
                    }

                    new_stmts.push_back(processStmt(ctx, all_stmts[i], ref_count));
                }

                return makeCompound(
                    ctx, new_stmts, compound->getLBracLoc(), compound->getRBracLoc()
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
                if (auto *while_stmt = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                    auto *new_body = processStmt(ctx, while_stmt->getBody(), ref_count);
                    if (new_body == while_stmt->getBody()) {
                        return stmt;
                    }
                    return clang::WhileStmt::Create(
                        ctx, nullptr, while_stmt->getCond(), new_body,
                        while_stmt->getWhileLoc(), while_stmt->getLParenLoc(),
                        while_stmt->getRParenLoc()
                    );
                }
                if (auto *if_stmt = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                    auto *new_then = processStmt(ctx, if_stmt->getThen(), ref_count);
                    auto *new_else = processStmt(ctx, if_stmt->getElse(), ref_count);
                    if (new_then == if_stmt->getThen() && new_else == if_stmt->getElse()) {
                        return stmt;
                    }
                    if (new_then == nullptr) {
                        new_then = new (ctx) clang::NullStmt(if_stmt->getIfLoc(), false);
                    }
                    return clang::IfStmt::Create(
                        ctx, if_stmt->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr,
                        nullptr, if_stmt->getCond(), if_stmt->getLParenLoc(),
                        new_then->getBeginLoc(), new_then,
                        new_else != nullptr ? new_else->getBeginLoc() : clang::SourceLocation(),
                        new_else
                    );
                }
                if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                    auto *new_sub = processStmt(ctx, label->getSubStmt(), ref_count);
                    if (new_sub == label->getSubStmt()) {
                        return stmt;
                    }
                    return new (ctx)
                        clang::LabelStmt(label->getIdentLoc(), label->getDecl(), new_sub);
                }
                return stmt;
            }
        };

    } // anonymous namespace

    namespace detail {

        void addSwitchRecoveryPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< SwitchRecoveryPass >(state));
        }

        void addIrreducibleFallbackPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< IrreducibleFallbackPass >(state));
        }

        void addSwitchGotoInliningPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< SwitchGotoInliningPass >(state));
        }

        void addSwitchPasses(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< SwitchRecoveryPass >(state));
            pm.add_pass(std::make_unique< IrreducibleFallbackPass >(state));
        }

    } // namespace detail

} // namespace patchestry::ast
