/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/ClangEmitter.hpp>
#include <patchestry/Util/Log.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>

namespace patchestry::ast {

    namespace detail {
        clang::Expr *ensureRValue(clang::ASTContext &ctx, clang::Expr *expr) {
            if (expr == nullptr || !expr->isGLValue()) return expr;
            return clang::ImplicitCastExpr::Create(
                ctx, expr->getType(), clang::CK_LValueToRValue, expr, nullptr,
                clang::VK_PRValue, clang::FPOptionsOverride());
        }

        clang::CompoundStmt *makeCompound(
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &stmts,
            clang::SourceLocation l = clang::SourceLocation(),
            clang::SourceLocation r = clang::SourceLocation()) {
            return clang::CompoundStmt::Create(ctx, stmts, clang::FPOptionsOverride(), l, r);
        }
    } // namespace detail

    namespace {

        // Deep-clone a Clang Expr tree to prevent shared Expr* nodes.
        // CfgFoldStructure reuses branch_cond pointers across SNode conditions
        // (e.g., original and negated forms). CIR lowering requires tree-unique
        // Expr* nodes, so every condition must be cloned before emission.
        clang::Expr *cloneExpr(clang::ASTContext &ctx, clang::Expr *expr) {
            if (!expr) return nullptr;

            auto loc = expr->getExprLoc();

            if (auto *il = llvm::dyn_cast< clang::IntegerLiteral >(expr)) {
                return clang::IntegerLiteral::Create(
                    ctx, il->getValue(), il->getType(), loc
                );
            }
            if (auto *dre = llvm::dyn_cast< clang::DeclRefExpr >(expr)) {
                return clang::DeclRefExpr::Create(
                    ctx, dre->getQualifierLoc(), dre->getTemplateKeywordLoc(),
                    dre->getDecl(), dre->refersToEnclosingVariableOrCapture(),
                    loc, dre->getType(), dre->getValueKind()
                );
            }
            if (auto *bo = llvm::dyn_cast< clang::BinaryOperator >(expr)) {
                return clang::BinaryOperator::Create(
                    ctx, cloneExpr(ctx, bo->getLHS()), cloneExpr(ctx, bo->getRHS()),
                    bo->getOpcode(), bo->getType(), bo->getValueKind(),
                    bo->getObjectKind(), loc, clang::FPOptionsOverride()
                );
            }
            if (auto *uo = llvm::dyn_cast< clang::UnaryOperator >(expr)) {
                return clang::UnaryOperator::Create(
                    ctx, cloneExpr(ctx, uo->getSubExpr()), uo->getOpcode(),
                    uo->getType(), uo->getValueKind(), uo->getObjectKind(),
                    loc, false, clang::FPOptionsOverride()
                );
            }
            if (auto *ice = llvm::dyn_cast< clang::ImplicitCastExpr >(expr)) {
                return clang::ImplicitCastExpr::Create(
                    ctx, ice->getType(), ice->getCastKind(),
                    cloneExpr(ctx, ice->getSubExpr()), nullptr,
                    ice->getValueKind(), clang::FPOptionsOverride()
                );
            }
            if (auto *pe = llvm::dyn_cast< clang::ParenExpr >(expr)) {
                return new (ctx) clang::ParenExpr(
                    loc, loc, cloneExpr(ctx, pe->getSubExpr())
                );
            }
            if (auto *cse = llvm::dyn_cast< clang::CStyleCastExpr >(expr)) {
                auto *cloned_sub = cloneExpr(ctx, cse->getSubExpr());
                return clang::CStyleCastExpr::Create(
                    ctx, cse->getType(), cse->getValueKind(), cse->getCastKind(),
                    cloned_sub, nullptr, clang::FPOptionsOverride(),
                    ctx.getTrivialTypeSourceInfo(cse->getType()),
                    cse->getLParenLoc(), cse->getRParenLoc()
                );
            }
            // Fallback: return original (safe for leaf expressions that won't be shared)
            return expr;
        }

        // Check if a stmt ends with a control flow terminator (goto/break/continue/return).
        bool endsWithTerminator(clang::Stmt *s) {
            if (!s) return false;
            if (llvm::isa< clang::GotoStmt >(s) || llvm::isa< clang::BreakStmt >(s) ||
                llvm::isa< clang::ContinueStmt >(s) || llvm::isa< clang::ReturnStmt >(s))
                return true;
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s))
                return !cs->body_empty() && endsWithTerminator(cs->body_back());
            return false;
        }

        // Recursively convert SNode tree to Clang Stmt*
        class Emitter {
          public:
            Emitter(clang::ASTContext &ctx, clang::FunctionDecl *fn)
                : ctx_(ctx), fn_(fn) {}

            clang::Stmt *emit(const SNode *node) {
                if (!node) return nullptr;

                switch (node->Kind()) {
                case SNodeKind::SEQ:
                    return emitSeq(node->as< SSeq >());
                case SNodeKind::BLOCK:
                    return emitBlock(node->as< SBlock >());
                case SNodeKind::IF_THEN_ELSE:
                    return emitIfThenElse(node->as< SIfThenElse >());
                case SNodeKind::WHILE:
                    return emitWhile(node->as< SWhile >());
                case SNodeKind::DO_WHILE:
                    return emitDoWhile(node->as< SDoWhile >());
                case SNodeKind::FOR:
                    return emitFor(node->as< SFor >());
                case SNodeKind::SWITCH:
                    return emitSwitch(node->as< SSwitch >());
                case SNodeKind::GOTO:
                    return emitGoto(node->as< SGoto >());
                case SNodeKind::LABEL:
                    return emitLabel(node->as< SLabel >());
                case SNodeKind::BREAK:
                    return emitBreak(node->as< SBreak >());
                case SNodeKind::CONTINUE:
                    return emitContinue();
                case SNodeKind::RETURN:
                    return emitReturn(node->as< SReturn >());
                }
                return nullptr;
            }

            // After emitting, add LabelStmt for any goto targets that don't have
            // corresponding label definitions. This prevents CIR goto/label mismatch.
            clang::Stmt *fixupMissingLabels(clang::Stmt *body) {
                std::vector< clang::Stmt * > stmts;
                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(body)) {
                    for (auto *s : cs->body()) stmts.push_back(s);
                } else {
                    stmts.push_back(body);
                }

                bool added = false;
                for (auto &[name, decl] : labels_) {
                    if (emitted_labels_.find(name) == emitted_labels_.end()) {
                        auto *null_stmt = new (ctx_) clang::NullStmt(loc());
                        auto *label_stmt = new (ctx_) clang::LabelStmt(
                            loc(), decl, null_stmt
                        );
                        stmts.push_back(label_stmt);
                        added = true;
                        LOG(WARNING) << "Added missing label definition for '"
                                     << name << "'\n";
                    }
                }

                if (added) {
                    return detail::makeCompound(ctx_, stmts);
                }
                return body;
            }

          private:
            clang::SourceLocation loc() const { return clang::SourceLocation(); }

            clang::Stmt *emitSeq(const SSeq *seq) {
                std::vector< clang::Stmt * > stmts;
                for (const auto *child : seq->Children()) {
                    auto *s = emit(child);
                    if (s) stmts.push_back(s);
                }
                return detail::makeCompound(ctx_, stmts);
            }

            clang::Stmt *emitBlock(const SBlock *block) {
                if (block->Size() == 1) return block->Stmts()[0];
                return detail::makeCompound(ctx_, block->Stmts());
            }

            clang::Stmt *emitIfThenElse(const SIfThenElse *ite) {
                auto *cond = detail::ensureRValue(ctx_, cloneExpr(ctx_, ite->Cond()));
                auto *then_stmt = emit(ite->ThenBranch());
                auto *else_stmt = ite->ElseBranch() ? emit(ite->ElseBranch()) : nullptr;

                if (!then_stmt) then_stmt = new (ctx_) clang::NullStmt(loc());

                return clang::IfStmt::Create(
                    ctx_, loc(), clang::IfStatementKind::Ordinary,
                    nullptr, nullptr, cond, loc(), loc(),
                    then_stmt, loc(), else_stmt
                );
            }

            clang::Stmt *emitWhile(const SWhile *w) {
                auto *cond = detail::ensureRValue(ctx_, cloneExpr(ctx_, w->Cond()));
                auto *body = emit(w->Body());
                if (!body) body = new (ctx_) clang::NullStmt(loc());

                return clang::WhileStmt::Create(
                    ctx_, nullptr, cond, body, loc(), loc(), loc()
                );
            }

            clang::Stmt *emitDoWhile(const SDoWhile *dw) {
                auto *body = emit(dw->Body());
                if (!body) body = new (ctx_) clang::NullStmt(loc());
                auto *cond = detail::ensureRValue(ctx_, cloneExpr(ctx_, dw->Cond()));

                return new (ctx_) clang::DoStmt(body, cond, loc(), loc(), loc());
            }

            clang::Stmt *emitFor(const SFor *f) {
                auto *body = emit(f->Body());
                if (!body) body = new (ctx_) clang::NullStmt(loc());

                return new (ctx_) clang::ForStmt(
                    ctx_, f->Init(),
                    f->Cond() ? detail::ensureRValue(ctx_, cloneExpr(ctx_, f->Cond())) : nullptr,
                    nullptr, f->Inc(), body, loc(), loc(), loc()
                );
            }

            clang::Stmt *emitSwitch(const SSwitch *sw) {
                auto *disc = detail::ensureRValue(ctx_, sw->Discriminant());
                auto *switch_stmt = clang::SwitchStmt::Create(
                    ctx_, nullptr, nullptr, disc, loc(), loc()
                );

                // Build the switch body as a compound stmt with cases
                std::vector< clang::Stmt * > body_stmts;

                for (const auto &c : sw->Cases()) {
                    auto *case_stmt = clang::CaseStmt::Create(
                        ctx_, c.value, nullptr, loc(), loc(), loc()
                    );

                    clang::Stmt *case_body = emit(c.body);
                    if (!case_body) case_body = new (ctx_) clang::NullStmt(loc());

                    // Add break after case body unless it already terminates
                    std::vector< clang::Stmt * > case_stmts = {case_body};
                    if (!endsWithTerminator(case_body)) {
                        case_stmts.push_back(new (ctx_) clang::BreakStmt(loc()));
                    }
                    case_stmt->setSubStmt(detail::makeCompound(ctx_, case_stmts));

                    body_stmts.push_back(case_stmt);
                    switch_stmt->addSwitchCase(case_stmt);
                }

                if (sw->DefaultBody()) {
                    auto *def_stmt = new (ctx_) clang::DefaultStmt(
                        loc(), loc(), emit(sw->DefaultBody())
                    );
                    body_stmts.push_back(def_stmt);
                    switch_stmt->addSwitchCase(def_stmt);
                }

                switch_stmt->setBody(detail::makeCompound(ctx_, body_stmts));
                return switch_stmt;
            }

            clang::Stmt *emitGoto(const SGoto *g) {
                // Look up or create the label
                auto *label_decl = getOrCreateLabel(g->Target());
                return new (ctx_) clang::GotoStmt(label_decl, loc(), loc());
            }

            clang::Stmt *emitLabel(const SLabel *l) {
                auto *label_decl = getOrCreateLabel(l->Name());
                emitted_labels_.insert(std::string(l->Name()));
                auto *sub = l->Body() ? emit(l->Body()) : new (ctx_) clang::NullStmt(loc());
                return new (ctx_) clang::LabelStmt(loc(), label_decl, sub);
            }

            clang::Stmt *emitBreak(const SBreak *) {
                return new (ctx_) clang::BreakStmt(loc());
            }

            clang::Stmt *emitContinue() {
                return new (ctx_) clang::ContinueStmt(loc());
            }

            clang::Stmt *emitReturn(const SReturn *r) {
                return clang::ReturnStmt::Create(ctx_, loc(), r->Value(), nullptr);
            }

            clang::LabelDecl *getOrCreateLabel(std::string_view name) {
                std::string key(name);
                auto it = labels_.find(key);
                if (it != labels_.end()) return it->second;

                auto &idents = ctx_.Idents;
                auto &ident = idents.get(llvm::StringRef(name.data(), name.size()));
                auto *decl = clang::LabelDecl::Create(ctx_, fn_, loc(), &ident);
                labels_[key] = decl;
                return decl;
            }

            clang::ASTContext &ctx_;
            clang::FunctionDecl *fn_;
            std::unordered_map< std::string, clang::LabelDecl * > labels_;
            std::unordered_set< std::string > emitted_labels_;
        };

    } // namespace

    // Collect all VarDecls referenced by DeclRefExprs in a Stmt tree.
    static void collectReferencedVars(clang::Stmt *s,
                                      std::unordered_set< clang::VarDecl * > &vars,
                                      std::unordered_set< clang::Stmt * > &seen) {
        if (!s || !seen.insert(s).second) return;
        if (auto *dre = llvm::dyn_cast< clang::DeclRefExpr >(s)) {
            if (auto *vd = llvm::dyn_cast< clang::VarDecl >(dre->getDecl())) {
                vars.insert(vd);
            }
        }
        for (auto *child : s->children()) {
            collectReferencedVars(child, vars, seen);
        }
    }

    // Collect all VarDecls that already have a DeclStmt in the Stmt tree.
    static void collectDeclaredVars(clang::Stmt *s,
                                    std::unordered_set< clang::VarDecl * > &vars,
                                    std::unordered_set< clang::Stmt * > &seen) {
        if (!s || !seen.insert(s).second) return;
        if (auto *ds = llvm::dyn_cast< clang::DeclStmt >(s)) {
            for (auto *d : ds->decls()) {
                if (auto *vd = llvm::dyn_cast< clang::VarDecl >(d)) {
                    vars.insert(vd);
                }
            }
        }
        for (auto *child : s->children()) {
            collectDeclaredVars(child, vars, seen);
        }
    }

    // Collect all DeclStmts from a statement tree for hoisting.
    static void collectDeclStmts(clang::Stmt *s,
                                 std::vector< clang::Stmt * > &decls,
                                 std::unordered_set< clang::Stmt * > &seen) {
        if (!s || !seen.insert(s).second) return;
        if (llvm::isa< clang::DeclStmt >(s)) {
            decls.push_back(s);
            return;
        }
        for (auto *child : s->children()) {
            collectDeclStmts(child, decls, seen);
        }
    }

    // Remove DeclStmts from their original positions in the tree.
    static clang::Stmt *stripDeclStmts(
        clang::ASTContext &ctx, clang::Stmt *s,
        const std::unordered_set< clang::Stmt * > &decl_set
    ) {
        if (!s) return nullptr;
        if (decl_set.count(s)) return nullptr;

        auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s);
        if (!cs) return s;

        std::vector< clang::Stmt * > filtered;
        for (auto *child : cs->body()) {
            if (decl_set.count(child)) continue;
            auto *stripped = stripDeclStmts(ctx, child, decl_set);
            if (stripped) filtered.push_back(stripped);
        }
        return detail::makeCompound(ctx, filtered);
    }

    // Collect all LabelDecls referenced by GotoStmts in a Stmt tree.
    static void collectGotoTargets(clang::Stmt *s,
                                   std::unordered_set< clang::LabelDecl * > &targets,
                                   std::unordered_set< clang::Stmt * > &seen) {
        if (!s || !seen.insert(s).second) return;
        if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
            targets.insert(gs->getLabel());
            return;
        }
        for (auto *child : s->children()) {
            collectGotoTargets(child, targets, seen);
        }
    }

    // Collect all LabelDecls that have a LabelStmt definition in a Stmt tree.
    static void collectLabelDefs(clang::Stmt *s,
                                 std::unordered_set< clang::LabelDecl * > &defs,
                                 std::unordered_set< clang::Stmt * > &seen) {
        if (!s || !seen.insert(s).second) return;
        if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
            defs.insert(ls->getDecl());
        }
        for (auto *child : s->children()) {
            collectLabelDefs(child, defs, seen);
        }
    }

    // Add LabelStmt definitions for all GotoStmt targets missing from the body.
    // This handles gotos inside raw Clang AST (SBlock stmts) that bypass the
    // emitter's label tracking.
    static clang::Stmt *fixupAllMissingLabels(clang::Stmt *body,
                                               clang::ASTContext &ctx) {
        std::unordered_set< clang::LabelDecl * > targets, defs;
        std::unordered_set< clang::Stmt * > seen1, seen2;
        collectGotoTargets(body, targets, seen1);
        collectLabelDefs(body, defs, seen2);

        // Build a set of defined label names (not just pointer equality)
        // so we can skip stubs for labels that have definitions under
        // a different LabelDecl object but the same name.
        std::unordered_set< std::string > def_names;
        for (auto *ld : defs) {
            def_names.insert(ld->getName().str());
        }

        std::vector< clang::Stmt * > stmts;
        if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(body)) {
            for (auto *s : cs->body()) stmts.push_back(s);
        } else {
            stmts.push_back(body);
        }

        bool added = false;
        for (auto *ld : targets) {
            if (defs.count(ld)) continue;
            // Skip if a label with the same name already has a definition
            // (may be a different LabelDecl object from SLabel emission)
            if (def_names.count(ld->getName().str())) continue;
            auto *null_stmt = new (ctx) clang::NullStmt(clang::SourceLocation());
            auto *label_stmt = new (ctx) clang::LabelStmt(
                clang::SourceLocation(), ld, null_stmt
            );
            stmts.push_back(label_stmt);
            added = true;
        }

        if (added) {
            return detail::makeCompound(ctx, stmts);
        }
        return body;
    }

    void emitClangAST(SNode *root, clang::FunctionDecl *fn,
                      clang::ASTContext &ctx) {
        Emitter emitter(ctx, fn);
        auto *body = emitter.emit(root);
        if (!body) {
            body = detail::makeCompound(ctx, {});
        }

        // Fixup: add label definitions for any goto targets missing from the tree
        body = emitter.fixupMissingLabels(body);

        // Fixup: scan full AST for GotoStmt targets without LabelStmt definitions
        // (handles gotos in raw Clang AST inside SBlock nodes)
        body = fixupAllMissingLabels(body, ctx);

        // Phase 1: Hoist all existing DeclStmts to the top of the function.
        std::vector< clang::Stmt * > decl_stmts;
        {
            std::unordered_set< clang::Stmt * > seen;
            collectDeclStmts(body, decl_stmts, seen);
        }

        if (!decl_stmts.empty()) {
            std::unordered_set< clang::Stmt * > decl_set(
                decl_stmts.begin(), decl_stmts.end()
            );
            body = stripDeclStmts(ctx, body, decl_set);
        }

        // Phase 2: Synthesize DeclStmts for any VarDecls referenced but not
        // declared in the body. CfgFoldStructure may drop unreachable blocks
        // containing DeclStmts while retaining blocks that reference those vars.
        // CIR crashes with "DeclRefExpr for decl not entered in LocalDeclMap"
        // when it encounters a reference to an undeclared variable.
        {
            std::unordered_set< clang::VarDecl * > referenced, declared;
            std::unordered_set< clang::Stmt * > seen1, seen2;
            collectReferencedVars(body, referenced, seen1);
            collectDeclaredVars(body, declared, seen2);
            // Also count vars from hoisted DeclStmts
            for (auto *ds : decl_stmts) {
                std::unordered_set< clang::Stmt * > seen3;
                collectDeclaredVars(ds, declared, seen3);
            }

            for (auto *vd : referenced) {
                if (declared.count(vd)) continue;
                // Skip function parameters — they're already in CIR's map
                if (llvm::isa< clang::ParmVarDecl >(vd)) continue;
                // Skip file-scope / extern globals — CIR asserts isLocalVarDecl()
                if (!vd->isLocalVarDecl()) continue;
                // Synthesize a DeclStmt for this missing variable
                auto *ds = new (ctx) clang::DeclStmt(
                    clang::DeclGroupRef(vd),
                    clang::SourceLocation(), clang::SourceLocation()
                );
                decl_stmts.push_back(ds);
            }
        }

        // Build final body: hoisted decls + control flow
        std::vector< clang::Stmt * > all_stmts;
        all_stmts.insert(all_stmts.end(), decl_stmts.begin(), decl_stmts.end());
        if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(body)) {
            for (auto *s : cs->body()) all_stmts.push_back(s);
        } else if (body) {
            all_stmts.push_back(body);
        }
        body = detail::makeCompound(ctx, all_stmts);

        fn->setBody(body);
    }

    // ---- Pretty-print cleanup (patchir-decomp only) ----

    namespace {
        // If stmt is a LabelStmt wrapping a CompoundStmt, push the label inside:
        //   LabelStmt(CompoundStmt{s1, s2, ...}) → CompoundStmt{LabelStmt(s1), s2, ...}
        // Otherwise return the stmt unchanged.
        clang::Stmt *pushLabelInside(clang::ASTContext &ctx, clang::Stmt *s) {
            auto *ls = llvm::dyn_cast_or_null< clang::LabelStmt >(s);
            if (!ls) return s;
            auto *inner = llvm::dyn_cast< clang::CompoundStmt >(ls->getSubStmt());
            if (!inner || inner->body_empty()) return s;

            auto it = inner->body_begin();
            ls->setSubStmt(*it);
            std::vector< clang::Stmt * > stmts;
            stmts.push_back(ls);
            for (++it; it != inner->body_end(); ++it)
                stmts.push_back(*it);
            return detail::makeCompound(ctx, stmts);
        }

        // Replace a trailing GotoStmt in a case body with break or continue.
        // Returns the modified stmt, or the original if no replacement was made.
        clang::Stmt *replaceTrailingGoto(clang::ASTContext &ctx, clang::Stmt *s,
                                          const std::string &break_label,
                                          const std::string &continue_label) {
            if (!s) return s;

            // Direct GotoStmt
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
                std::string name = gs->getLabel()->getName().str();
                if (!break_label.empty() && name == break_label)
                    return new (ctx) clang::BreakStmt(clang::SourceLocation());
                if (!continue_label.empty() && name == continue_label)
                    return new (ctx) clang::ContinueStmt(clang::SourceLocation());
                return s;
            }

            // CompoundStmt — check/replace last stmt
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                if (cs->body_empty()) return s;
                auto *last = *(cs->body_end() - 1);
                auto *replaced = replaceTrailingGoto(ctx, last, break_label, continue_label);
                if (replaced == last) return s;

                std::vector< clang::Stmt * > stmts;
                for (auto it = cs->body_begin(); std::next(it) != cs->body_end(); ++it)
                    stmts.push_back(*it);
                stmts.push_back(replaced);
                return detail::makeCompound(ctx, stmts);
            }

            return s;
        }

        // Walk case/default bodies in a SwitchStmt and convert trailing gotos
        // to break (if targeting break_label) or continue (if targeting continue_label).
        void convertSwitchCaseGotos(clang::ASTContext &ctx, clang::SwitchStmt *sw,
                                     const std::string &break_label,
                                     const std::string &continue_label) {
            auto *body = sw->getBody();
            auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(body);
            if (!cs) return;

            for (auto *child : cs->body()) {
                if (auto *case_s = llvm::dyn_cast< clang::CaseStmt >(child)) {
                    auto *sub = case_s->getSubStmt();
                    auto *r = replaceTrailingGoto(ctx, sub, break_label, continue_label);
                    if (r != sub) case_s->setSubStmt(r);
                } else if (auto *def_s = llvm::dyn_cast< clang::DefaultStmt >(child)) {
                    auto *sub = def_s->getSubStmt();
                    auto *r = replaceTrailingGoto(ctx, sub, break_label, continue_label);
                    if (r != sub) def_s->setSubStmt(r);
                }
            }
        }

        // Check if ALL case/default bodies in a switch end with goto to the same
        // label (and that label is NOT the break/continue label). Returns the
        // common label name, or empty string if not uniform.
        std::string findCommonTrailingGoto(clang::SwitchStmt *sw) {
            auto *body = sw->getBody();
            auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(body);
            if (!cs) return {};

            std::string common;
            auto getTrailingGotoLabel = [](clang::Stmt *s) -> std::string {
                if (!s) return {};
                if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s))
                    return gs->getLabel()->getName().str();
                if (auto *c = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                    if (!c->body_empty()) {
                        if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(c->body_back()))
                            return gs->getLabel()->getName().str();
                    }
                }
                return {};
            };

            for (auto *child : cs->body()) {
                clang::Stmt *sub = nullptr;
                if (auto *case_s = llvm::dyn_cast< clang::CaseStmt >(child))
                    sub = case_s->getSubStmt();
                else if (auto *def_s = llvm::dyn_cast< clang::DefaultStmt >(child))
                    sub = def_s->getSubStmt();
                else continue;

                auto label = getTrailingGotoLabel(sub);
                if (label.empty()) return {};
                if (common.empty()) common = label;
                else if (common != label) return {};
            }
            return common;
        }

        // Recursively clean up a Stmt tree:
        //  - Flatten nested CompoundStmts
        //  - Push LabelStmt(CompoundStmt) patterns into CompoundStmt{LabelStmt, ...}
        //  - Convert gotos inside switch cases to break/continue
        //  - Hoist common trailing gotos out of switch
        //
        // continue_label: label of enclosing loop header (for goto → continue)
        clang::Stmt *cleanupStmtTree(clang::ASTContext &ctx, clang::Stmt *s,
                                      const std::string &continue_label = "") {
            if (!s) return nullptr;

            // Handle IfStmt: recurse into then/else, push labels inside
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
                ifs->setThen(pushLabelInside(ctx,
                    cleanupStmtTree(ctx, ifs->getThen(), continue_label)));
                if (ifs->getElse())
                    ifs->setElse(pushLabelInside(ctx,
                        cleanupStmtTree(ctx, ifs->getElse(), continue_label)));
                return s;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
                ws->setBody(pushLabelInside(ctx,
                    cleanupStmtTree(ctx, ws->getBody(), continue_label)));
                return s;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(s)) {
                ds->setBody(pushLabelInside(ctx,
                    cleanupStmtTree(ctx, ds->getBody(), continue_label)));
                return s;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
                fs->setBody(pushLabelInside(ctx,
                    cleanupStmtTree(ctx, fs->getBody(), continue_label)));
                return s;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
                // If this label wraps a loop, set it as the continue target
                auto *sub = ls->getSubStmt();
                std::string new_cont;
                if (sub && (llvm::isa< clang::WhileStmt >(sub) ||
                            llvm::isa< clang::DoStmt >(sub) ||
                            llvm::isa< clang::ForStmt >(sub))) {
                    new_cont = ls->getDecl()->getName().str();
                }
                ls->setSubStmt(cleanupStmtTree(ctx, sub,
                    new_cont.empty() ? continue_label : new_cont));
                return s;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(s)) {
                sw->setBody(cleanupStmtTree(ctx, sw->getBody(), continue_label));
                return s;
            }
            if (auto *cs_node = llvm::dyn_cast< clang::CaseStmt >(s)) {
                cs_node->setSubStmt(cleanupStmtTree(ctx, cs_node->getSubStmt(), continue_label));
                return s;
            }
            if (auto *def = llvm::dyn_cast< clang::DefaultStmt >(s)) {
                def->setSubStmt(cleanupStmtTree(ctx, def->getSubStmt(), continue_label));
                return s;
            }

            // CompoundStmt: recurse, flatten nested compounds, push labels inside,
            // then convert switch case gotos to break/continue.
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                // --- First pass: recurse, flatten, push labels ---
                std::vector< clang::Stmt * > children;
                for (auto *child : cs->body()) {
                    auto *cleaned = cleanupStmtTree(ctx, child, continue_label);
                    if (!cleaned) continue;

                    // Flatten nested CompoundStmts
                    if (auto *inner_cs = llvm::dyn_cast< clang::CompoundStmt >(cleaned)) {
                        for (auto *gc : inner_cs->body())
                            children.push_back(gc);
                    }
                    // Push label inside compound
                    else if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(cleaned)) {
                        if (auto *lcs = llvm::dyn_cast< clang::CompoundStmt >(ls->getSubStmt())) {
                            auto it = lcs->body_begin();
                            if (it != lcs->body_end()) {
                                ls->setSubStmt(*it);
                                children.push_back(ls);
                                for (++it; it != lcs->body_end(); ++it)
                                    children.push_back(*it);
                            } else {
                                children.push_back(ls);
                            }
                        } else {
                            children.push_back(cleaned);
                        }
                    }
                    else {
                        children.push_back(cleaned);
                    }
                }

                // --- Second pass: convert gotos in switch case bodies ---
                for (size_t i = 0; i < children.size(); ++i) {
                    auto *sw = llvm::dyn_cast< clang::SwitchStmt >(children[i]);
                    if (!sw) continue;

                    // Find label immediately after switch → break target
                    std::string break_label;
                    if (i + 1 < children.size()) {
                        if (auto *next_ls = llvm::dyn_cast< clang::LabelStmt >(children[i + 1])) {
                            break_label = next_ls->getDecl()->getName().str();
                        }
                    }

                    // Convert case gotos to break/continue
                    if (!break_label.empty() || !continue_label.empty()) {
                        convertSwitchCaseGotos(ctx, sw, break_label, continue_label);
                    }

                    // Hoist: if ALL cases goto the same label (not break/continue
                    // target), replace with breaks and add goto after switch.
                    if (break_label.empty()) {
                        std::string common = findCommonTrailingGoto(sw);
                        if (!common.empty() && common != continue_label) {
                            // Replace all trailing gotos with break
                            convertSwitchCaseGotos(ctx, sw, common, "");
                            // Find the LabelDecl for the common target by scanning
                            // the function body for a matching goto.
                            clang::LabelDecl *target_decl = nullptr;
                            std::function< void(clang::Stmt *) > findLabel =
                                [&](clang::Stmt *st) {
                                    if (!st || target_decl) return;
                                    if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(st)) {
                                        if (gs->getLabel()->getName().str() == common)
                                            target_decl = gs->getLabel();
                                        return;
                                    }
                                    if (auto *ls2 = llvm::dyn_cast< clang::LabelStmt >(st)) {
                                        if (ls2->getDecl()->getName().str() == common)
                                            target_decl = ls2->getDecl();
                                    }
                                    for (auto *c : st->children()) findLabel(c);
                                };
                            // Scan the entire children vector for the label
                            for (auto *c : children) findLabel(c);
                            if (target_decl) {
                                auto loc = clang::SourceLocation();
                                auto *hoisted_goto = new (ctx) clang::GotoStmt(
                                    target_decl, loc, loc);
                                children.insert(children.begin() + static_cast< long >(i) + 1,
                                                hoisted_goto);
                                ++i; // skip the inserted goto
                            }
                        }
                    }
                }

                return detail::makeCompound(ctx, children);
            }

            return s;
        }
    } // namespace

    // Remove LabelStmts that are not the target of any GotoStmt.
    // Replaces dead LabelStmt with its sub-statement.
    static clang::Stmt *removeDeadLabels(clang::ASTContext &ctx, clang::Stmt *s,
                                          const std::unordered_set< clang::LabelDecl * > &live) {
        if (!s) return nullptr;

        if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
            auto *sub = removeDeadLabels(ctx, ls->getSubStmt(), live);
            if (!live.count(ls->getDecl())) {
                return sub;
            }
            ls->setSubStmt(sub ? sub : new (ctx) clang::NullStmt(clang::SourceLocation()));
            return ls;
        }

        // Recurse into compound and flatten away any nulls from removed labels
        if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
            std::vector< clang::Stmt * > children;
            for (auto *child : cs->body()) {
                auto *cleaned = removeDeadLabels(ctx, child, live);
                if (cleaned) children.push_back(cleaned);
            }
            return detail::makeCompound(ctx, children);
        }

        // Recurse into structured statements
        if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
            ifs->setThen(removeDeadLabels(ctx, ifs->getThen(), live));
            if (ifs->getElse())
                ifs->setElse(removeDeadLabels(ctx, ifs->getElse(), live));
            return s;
        }
        if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
            ws->setBody(removeDeadLabels(ctx, ws->getBody(), live));
            return s;
        }
        if (auto *ds = llvm::dyn_cast< clang::DoStmt >(s)) {
            ds->setBody(removeDeadLabels(ctx, ds->getBody(), live));
            return s;
        }
        if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
            fs->setBody(removeDeadLabels(ctx, fs->getBody(), live));
            return s;
        }
        if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(s)) {
            sw->setBody(removeDeadLabels(ctx, sw->getBody(), live));
            return s;
        }
        if (auto *cs_node = llvm::dyn_cast< clang::CaseStmt >(s)) {
            cs_node->setSubStmt(removeDeadLabels(ctx, cs_node->getSubStmt(), live));
            return s;
        }
        if (auto *def = llvm::dyn_cast< clang::DefaultStmt >(s)) {
            def->setSubStmt(removeDeadLabels(ctx, def->getSubStmt(), live));
            return s;
        }

        return s;
    }

    void cleanupPrettyPrint(clang::FunctionDecl *fn, clang::ASTContext &ctx) {
        if (!fn || !fn->hasBody()) return;
        auto *body = cleanupStmtTree(ctx, fn->getBody());
        if (body) fn->setBody(body);

        // Remove labels that are not the target of any goto.
        // Run after cleanupStmtTree which may convert gotos to break/continue.
        std::unordered_set< clang::LabelDecl * > goto_targets;
        std::unordered_set< clang::Stmt * > seen;
        collectGotoTargets(fn->getBody(), goto_targets, seen);
        body = removeDeadLabels(ctx, fn->getBody(), goto_targets);
        if (body) fn->setBody(body);
    }

} // namespace patchestry::ast
