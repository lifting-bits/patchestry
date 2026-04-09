/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/ClangEmitter.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Util/Log.hpp>

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>

namespace patchestry::ast {

    namespace detail {
        static clang::CompoundStmt *MakeCompound(
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &stmts,
            clang::SourceLocation l = clang::SourceLocation(),
            clang::SourceLocation r = clang::SourceLocation()) {
            return clang::CompoundStmt::Create(ctx, stmts, clang::FPOptionsOverride(), l, r);
        }
    } // namespace detail

    namespace {

        // Deep-clone a Clang Expr tree to prevent shared Expr* nodes.
        // The CGraph pipeline may reuse branch_cond pointers across SNode
        // conditions (e.g., original and negated forms). CIR lowering requires
        // tree-unique Expr* nodes, so every condition must be cloned before emission.
        clang::Expr *CloneExpr(clang::ASTContext &ctx, clang::Expr *expr) {
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
                    ctx, CloneExpr(ctx, bo->getLHS()), CloneExpr(ctx, bo->getRHS()),
                    bo->getOpcode(), bo->getType(), bo->getValueKind(),
                    bo->getObjectKind(), loc, clang::FPOptionsOverride()
                );
            }
            if (auto *uo = llvm::dyn_cast< clang::UnaryOperator >(expr)) {
                return clang::UnaryOperator::Create(
                    ctx, CloneExpr(ctx, uo->getSubExpr()), uo->getOpcode(),
                    uo->getType(), uo->getValueKind(), uo->getObjectKind(),
                    loc, false, clang::FPOptionsOverride()
                );
            }
            if (auto *ice = llvm::dyn_cast< clang::ImplicitCastExpr >(expr)) {
                return clang::ImplicitCastExpr::Create(
                    ctx, ice->getType(), ice->getCastKind(),
                    CloneExpr(ctx, ice->getSubExpr()), nullptr,
                    ice->getValueKind(), clang::FPOptionsOverride()
                );
            }
            if (auto *pe = llvm::dyn_cast< clang::ParenExpr >(expr)) {
                return new (ctx) clang::ParenExpr(
                    loc, loc, CloneExpr(ctx, pe->getSubExpr())
                );
            }
            if (auto *cse = llvm::dyn_cast< clang::CStyleCastExpr >(expr)) {
                auto *cloned_sub = CloneExpr(ctx, cse->getSubExpr());
                return clang::CStyleCastExpr::Create(
                    ctx, cse->getType(), cse->getValueKind(), cse->getCastKind(),
                    cloned_sub, nullptr, clang::FPOptionsOverride(),
                    ctx.getTrivialTypeSourceInfo(cse->getType()),
                    cse->getLParenLoc(), cse->getRParenLoc()
                );
            }
            if (auto *ase = llvm::dyn_cast< clang::ArraySubscriptExpr >(expr)) {
                return new (ctx) clang::ArraySubscriptExpr(
                    CloneExpr(ctx, ase->getLHS()),
                    CloneExpr(ctx, ase->getRHS()),
                    ase->getType(), ase->getValueKind(),
                    ase->getObjectKind(), loc
                );
            }
            if (auto *me = llvm::dyn_cast< clang::MemberExpr >(expr)) {
                return clang::MemberExpr::CreateImplicit(
                    ctx, CloneExpr(ctx, me->getBase()),
                    me->isArrow(), me->getMemberDecl(),
                    me->getType(), me->getValueKind(),
                    me->getObjectKind()
                );
            }
            if (auto *ce = llvm::dyn_cast< clang::CallExpr >(expr)) {
                llvm::SmallVector< clang::Expr *, 4 > args;
                for (auto *a : ce->arguments())
                    args.push_back(CloneExpr(ctx, a));
                auto *cloned = clang::CallExpr::Create(
                    ctx, CloneExpr(ctx, ce->getCallee()), args,
                    ce->getType(), ce->getValueKind(), loc,
                    clang::FPOptionsOverride()
                );
                return cloned;
            }
            if (auto *co = llvm::dyn_cast< clang::ConditionalOperator >(expr)) {
                return new (ctx) clang::ConditionalOperator(
                    CloneExpr(ctx, co->getCond()),
                    loc, CloneExpr(ctx, co->getTrueExpr()),
                    loc, CloneExpr(ctx, co->getFalseExpr()),
                    co->getType(), co->getValueKind(),
                    co->getObjectKind()
                );
            }
            // Fallback: wrap in a ParenExpr to force a unique AST node.
            // This prevents shared Expr* pointers from causing CIR lowering
            // assertions when the same condition is used in multiple SNodes.
            LOG(WARNING) << "CloneExpr: unhandled expression type "
                         << expr->getStmtClassName() << ", wrapping in ParenExpr\n";
            return new (ctx) clang::ParenExpr(loc, loc, expr);
        }

    } // close anonymous namespace

    namespace detail {
        bool EndsWithTerminator(clang::Stmt *s) {
            llvm::SmallVector< clang::Stmt *, 4 > worklist;
            if (s) worklist.push_back(s);

            while (!worklist.empty()) {
                auto *cur = worklist.pop_back_val();
                if (!cur) return false;

                if (llvm::isa< clang::GotoStmt >(cur)
                    || llvm::isa< clang::BreakStmt >(cur)
                    || llvm::isa< clang::ContinueStmt >(cur)
                    || llvm::isa< clang::ReturnStmt >(cur))
                    continue;
                if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(cur)) {
                    if (cs->body_empty()) return false;
                    worklist.push_back(cs->body_back());
                    continue;
                }
                if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(cur)) {
                    worklist.push_back(ls->getSubStmt());
                    continue;
                }
                if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(cur)) {
                    if (!ifs->getThen() || !ifs->getElse()) return false;
                    worklist.push_back(ifs->getThen());
                    worklist.push_back(ifs->getElse());
                    continue;
                }
                return false;
            }
            return true;
        }
    } // namespace detail

    namespace {

        // Recursively convert SNode tree to Clang Stmt*
        class Emitter {
          public:
            Emitter(clang::ASTContext &ctx, clang::FunctionDecl *fn)
                : ctx_(ctx), fn_(fn) {}

            clang::Stmt *Emit(const SNode *node) {
                if (!node) return nullptr;

                switch (node->Kind()) {
                case SNodeKind::kSeq:
                    return EmitSeq(node->as< SSeq >());
                case SNodeKind::kBlock:
                    return EmitBlock(node->as< SBlock >());
                case SNodeKind::kIfThenElse:
                    return EmitIfThenElse(node->as< SIfThenElse >());
                case SNodeKind::kWhile:
                    return EmitWhile(node->as< SWhile >());
                case SNodeKind::kDoWhile:
                    return EmitDoWhile(node->as< SDoWhile >());
                case SNodeKind::kFor:
                    return EmitFor(node->as< SFor >());
                case SNodeKind::kSwitch:
                    return EmitSwitch(node->as< SSwitch >());
                case SNodeKind::kGoto:
                    return EmitGoto(node->as< SGoto >());
                case SNodeKind::kLabel:
                    return EmitLabel(node->as< SLabel >());
                case SNodeKind::kBreak:
                    return EmitBreak(node->as< SBreak >());
                case SNodeKind::kContinue:
                    return EmitContinue();
                case SNodeKind::kReturn:
                    return EmitReturn(node->as< SReturn >());
                }
                llvm_unreachable("unhandled SNodeKind in Emitter::Emit");
            }

            // After emitting, add LabelStmt for any goto targets that don't have
            // corresponding label definitions. This prevents CIR goto/label mismatch.
          private:
            clang::SourceLocation Loc() const { return clang::SourceLocation(); }

            clang::Stmt *EmitSeq(const SSeq *seq) {
                std::vector< clang::Stmt * > stmts;
                for (const auto *child : seq->Children()) {
                    auto *s = Emit(child);
                    if (s) stmts.push_back(s);
                }
                return detail::MakeCompound(ctx_, stmts);
            }

            clang::Stmt *EmitBlock(const SBlock *block) {
                if (block->Size() == 1) return block->Stmts()[0];
                return detail::MakeCompound(ctx_, block->Stmts());
            }

            clang::Stmt *EmitIfThenElse(const SIfThenElse *ite) {
                LOG_FATAL_IF(!ite->Cond(),
                    "SIfThenElse has null condition - structuring "
                    "rule produced a branch without a guard expression");
                auto *cond = EnsureRValue(ctx_, CloneExpr(ctx_, ite->Cond()));
                auto *then_stmt = Emit(ite->ThenBranch());
                auto *else_stmt = ite->ElseBranch() ? Emit(ite->ElseBranch()) : nullptr;

                if (!then_stmt) then_stmt = new (ctx_) clang::NullStmt(Loc());

                // Unwrap CompoundStmt around a single IfStmt in the else
                // branch so Clang's printer emits "else if" not "else { if".
                if (auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(else_stmt)) {
                    if (cs->size() == 1 && llvm::isa< clang::IfStmt >(cs->body_front()))
                        else_stmt = cs->body_front();
                }

                return clang::IfStmt::Create(
                    ctx_, Loc(), clang::IfStatementKind::Ordinary,
                    nullptr, nullptr, cond, Loc(), Loc(),
                    then_stmt, Loc(), else_stmt
                );
            }

            clang::Stmt *EmitWhile(const SWhile *w) {
                clang::Expr *cond = nullptr;
                if (w->Cond()) {
                    cond = EnsureRValue(ctx_, CloneExpr(ctx_, w->Cond()));
                } else {
                    // while(1) — SWhile with null condition.
                    cond = clang::IntegerLiteral::Create(
                        ctx_, llvm::APInt(32, 1), ctx_.IntTy, Loc());
                }
                auto *body = Emit(w->Body());
                if (!body) body = new (ctx_) clang::NullStmt(Loc());

                return clang::WhileStmt::Create(
                    ctx_, nullptr, cond, body, Loc(), Loc(), Loc()
                );
            }

            clang::Stmt *EmitDoWhile(const SDoWhile *dw) {
                auto *body = Emit(dw->Body());
                if (!body) body = new (ctx_) clang::NullStmt(Loc());
                auto *cond = EnsureRValue(ctx_, CloneExpr(ctx_, dw->Cond()));

                return new (ctx_) clang::DoStmt(body, cond, Loc(), Loc(), Loc());
            }

            clang::Stmt *EmitFor(const SFor *f) {
                auto *body = Emit(f->Body());
                if (!body) body = new (ctx_) clang::NullStmt(Loc());

                return new (ctx_) clang::ForStmt(
                    ctx_, f->Init(),
                    f->Cond() ? EnsureRValue(ctx_, CloneExpr(ctx_, f->Cond())) : nullptr,
                    nullptr, f->Inc(), body, Loc(), Loc(), Loc()
                );
            }

            clang::Stmt *EmitSwitch(const SSwitch *sw) {
                auto *disc = EnsureRValue(ctx_, CloneExpr(ctx_, sw->Discriminant()));
                auto *switch_stmt = clang::SwitchStmt::Create(
                    ctx_, nullptr, nullptr, disc, Loc(), Loc()
                );

                // Build the switch body as a compound stmt with cases
                std::vector< clang::Stmt * > body_stmts;

                for (const auto &c : sw->Cases()) {
                    auto *case_stmt = clang::CaseStmt::Create(
                        ctx_, c.value, nullptr, Loc(), Loc(), Loc()
                    );

                    if (c.body == nullptr) {
                        // Fallthrough stub: case N: (no body, falls into next case)
                        case_stmt->setSubStmt(new (ctx_) clang::NullStmt(Loc()));
                    } else {
                        clang::Stmt *case_body = Emit(c.body);
                        if (!case_body) {
                            case_body = new (ctx_) clang::NullStmt(Loc());
                        }

                        std::vector< clang::Stmt * > case_stmts = { case_body };
                        if (!detail::EndsWithTerminator(case_body)) {
                            case_stmts.push_back(new (ctx_) clang::BreakStmt(Loc()));
                        }
                        case_stmt->setSubStmt(detail::MakeCompound(ctx_, case_stmts));
                    }

                    body_stmts.push_back(case_stmt);
                    switch_stmt->addSwitchCase(case_stmt);
                }

                if (sw->DefaultBody()) {
                    clang::Stmt *def_body = Emit(sw->DefaultBody());
                    if (!def_body) {
                        def_body = new (ctx_) clang::NullStmt(Loc());
                    }

                    std::vector< clang::Stmt * > def_stmts = { def_body };
                    if (!detail::EndsWithTerminator(def_body)) {
                        def_stmts.push_back(new (ctx_) clang::BreakStmt(Loc()));
                    }

                    auto *def_stmt = new (ctx_)
                        clang::DefaultStmt(Loc(), Loc(), detail::MakeCompound(ctx_, def_stmts));
                    body_stmts.push_back(def_stmt);
                    switch_stmt->addSwitchCase(def_stmt);
                }

                switch_stmt->setBody(detail::MakeCompound(ctx_, body_stmts));
                return switch_stmt;
            }

            clang::Stmt *EmitGoto(const SGoto *g) {
                // Look up or create the label
                auto *label_decl = GetOrCreateLabel(g->Target());
                return new (ctx_) clang::GotoStmt(label_decl, Loc(), Loc());
            }

            clang::Stmt *EmitLabel(const SLabel *l) {
                auto *label_decl = GetOrCreateLabel(l->Name());
                emitted_labels_.insert(std::string(l->Name()));
                auto *sub = l->Body() ? Emit(l->Body()) : new (ctx_) clang::NullStmt(Loc());
                return new (ctx_) clang::LabelStmt(Loc(), label_decl, sub);
            }

            clang::Stmt *EmitBreak(const SBreak *) {
                return new (ctx_) clang::BreakStmt(Loc());
            }

            clang::Stmt *EmitContinue() {
                return new (ctx_) clang::ContinueStmt(Loc());
            }

            clang::Stmt *EmitReturn(const SReturn *r) {
                return clang::ReturnStmt::Create(ctx_, Loc(), r->Value(), nullptr);
            }

            clang::LabelDecl *GetOrCreateLabel(std::string_view name) {
                std::string key(name);
                auto it = labels_.find(key);
                if (it != labels_.end()) return it->second;

                // Check goto_labels_ cache (populated from raw Clang AST
                // GotoStmts before emission).  Reusing the same LabelDecl
                // objects that GotoStmts reference prevents CIR "goto/label
                // mismatch" from pointer identity mismatches.
                auto gl = goto_labels_.find(key);
                if (gl != goto_labels_.end()) {
                    labels_[key] = gl->second;
                    return gl->second;
                }

                auto &idents = ctx_.Idents;
                auto &ident = idents.get(llvm::StringRef(name.data(), name.size()));
                auto *decl = clang::LabelDecl::Create(ctx_, fn_, Loc(), &ident);
                labels_[key] = decl;
                return decl;
            }

          public:
            // Pre-scan: collect LabelDecl objects referenced by GotoStmts
            // in raw Clang AST (SBlock stmts).  Must be called before Emit().
            void CollectGotoLabelDecls(SNode *node) {
                if (!node) return;
                if (auto *blk = node->dyn_cast< SBlock >()) {
                    std::function< void(clang::Stmt *) > scan =
                        [&](clang::Stmt *s) {
                            if (!s) return;
                            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
                                auto *ld = gs->getLabel();
                                goto_labels_[ld->getName().str()] = ld;
                                return;
                            }
                            for (auto *child : s->children()) scan(child);
                        };
                    for (auto *s : blk->Stmts()) scan(s);
                    return; // SBlock has no SNode children
                }
                if (auto *seq = node->dyn_cast< SSeq >()) {
                    for (size_t i = 0; i < seq->Size(); ++i)
                        CollectGotoLabelDecls((*seq)[i]);
                } else if (auto *ite = node->dyn_cast< SIfThenElse >()) {
                    CollectGotoLabelDecls(ite->ThenBranch());
                    CollectGotoLabelDecls(ite->ElseBranch());
                } else if (auto *sw = node->dyn_cast< SSwitch >()) {
                    for (auto &c : sw->Cases())
                        CollectGotoLabelDecls(c.body);
                    CollectGotoLabelDecls(sw->DefaultBody());
                } else if (auto *lbl = node->dyn_cast< SLabel >()) {
                    CollectGotoLabelDecls(lbl->Body());
                } else if (auto *w = node->dyn_cast< SWhile >()) {
                    CollectGotoLabelDecls(w->Body());
                } else if (auto *dw = node->dyn_cast< SDoWhile >()) {
                    CollectGotoLabelDecls(dw->Body());
                } else if (auto *f = node->dyn_cast< SFor >()) {
                    CollectGotoLabelDecls(f->Body());
                }
            }

            clang::ASTContext &ctx_;
            clang::FunctionDecl *fn_;
            std::unordered_map< std::string, clang::LabelDecl * > labels_;
            std::unordered_map< std::string, clang::LabelDecl * > goto_labels_;
            std::unordered_set< std::string > emitted_labels_;
        };

    } // namespace

    // Collect all VarDecls referenced by DeclRefExprs in a Stmt tree.
    static void CollectReferencedVars(clang::Stmt *s,
                                      std::unordered_set< clang::VarDecl * > &vars,
                                      std::unordered_set< clang::Stmt * > &seen) {
        if (!s || !seen.insert(s).second) return;
        if (auto *dre = llvm::dyn_cast< clang::DeclRefExpr >(s)) {
            if (auto *vd = llvm::dyn_cast< clang::VarDecl >(dre->getDecl())) {
                vars.insert(vd);
            }
        }
        for (auto *child : s->children()) {
            CollectReferencedVars(child, vars, seen);
        }
    }

    // Collect all VarDecls that already have a DeclStmt in the Stmt tree.
    static void CollectDeclaredVars(clang::Stmt *s,
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
            CollectDeclaredVars(child, vars, seen);
        }
    }

    // Collect all DeclStmts from a statement tree for hoisting.
    // Skips DeclStmts inside for-loop init (they belong there).
    static void CollectDeclStmts(clang::Stmt *s,
                                 std::vector< clang::Stmt * > &decls,
                                 std::unordered_set< clang::Stmt * > &seen) {
        if (!s || !seen.insert(s).second) return;
        if (llvm::isa< clang::DeclStmt >(s)) {
            decls.push_back(s);
            return;
        }
        // For ForStmt, skip the init — its DeclStmt belongs in the for-init
        // and must not be hoisted (would cause duplicate VarDecl in CIR).
        if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
            // Only recurse into body — not init (its DeclStmt belongs
            // in the for-loop).  Cond and inc are expressions, not
            // statement lists, so they can't contain DeclStmts.
            CollectDeclStmts(fs->getBody(), decls, seen);
            return;
        }
        for (auto *child : s->children()) {
            CollectDeclStmts(child, decls, seen);
        }
    }

    // Remove DeclStmts from their original positions in the tree.
    // Recurses into CompoundStmt children and also into structured
    // statement bodies (IfStmt, WhileStmt, ForStmt, LabelStmt, etc.)
    // so that DeclStmts nested directly under them are stripped too.
    static clang::Stmt *StripDeclStmts(
        clang::ASTContext &ctx, clang::Stmt *s,
        const std::unordered_set< clang::Stmt * > &decl_set
    ) {
        if (!s) return nullptr;
        if (decl_set.count(s)) return nullptr;

        // Guarantee a non-null Stmt* for set* methods that require one.
        auto safe = [&](clang::Stmt *r) -> clang::Stmt * {
            return r ? r : new (ctx) clang::NullStmt(clang::SourceLocation());
        };

        // Recurse into structured statement bodies
        if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
            ifs->setThen(safe(StripDeclStmts(ctx, ifs->getThen(), decl_set)));
            if (ifs->getElse())
                ifs->setElse(safe(StripDeclStmts(ctx, ifs->getElse(), decl_set)));
            return s;
        }
        if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
            ws->setBody(safe(StripDeclStmts(ctx, ws->getBody(), decl_set)));
            return s;
        }
        if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
            fs->setBody(safe(StripDeclStmts(ctx, fs->getBody(), decl_set)));
            return s;
        }
        if (auto *ds = llvm::dyn_cast< clang::DoStmt >(s)) {
            ds->setBody(safe(StripDeclStmts(ctx, ds->getBody(), decl_set)));
            return s;
        }
        if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
            ls->setSubStmt(safe(StripDeclStmts(ctx, ls->getSubStmt(), decl_set)));
            return s;
        }
        if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(s)) {
            sw->setBody(safe(StripDeclStmts(ctx, sw->getBody(), decl_set)));
            return s;
        }
        if (auto *cs_node = llvm::dyn_cast< clang::CaseStmt >(s)) {
            cs_node->setSubStmt(safe(StripDeclStmts(ctx, cs_node->getSubStmt(), decl_set)));
            return s;
        }
        if (auto *def = llvm::dyn_cast< clang::DefaultStmt >(s)) {
            def->setSubStmt(safe(StripDeclStmts(ctx, def->getSubStmt(), decl_set)));
            return s;
        }

        auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s);
        if (!cs) return s;

        std::vector< clang::Stmt * > filtered;
        for (auto *child : cs->body()) {
            if (decl_set.count(child)) continue;
            auto *stripped = StripDeclStmts(ctx, child, decl_set);
            if (stripped) filtered.push_back(stripped);
        }
        return detail::MakeCompound(ctx, filtered);
    }



    void EmitClangAST(SNode *root, clang::FunctionDecl *fn,
                      clang::ASTContext &ctx) {
        Emitter emitter(ctx, fn);
        // Pre-scan: collect LabelDecl objects from raw Clang GotoStmts
        // so EmitLabel can reuse the same objects (pointer identity match).
        emitter.CollectGotoLabelDecls(root);
        auto *body = emitter.Emit(root);
        if (!body) {
            body = detail::MakeCompound(ctx, {});
        }

        // Phase 1: Hoist all existing DeclStmts to the top of the function.
        std::vector< clang::Stmt * > decl_stmts;
        {
            std::unordered_set< clang::Stmt * > seen;
            CollectDeclStmts(body, decl_stmts, seen);
        }

        if (!decl_stmts.empty()) {
            std::unordered_set< clang::Stmt * > decl_set(
                decl_stmts.begin(), decl_stmts.end()
            );
            body = StripDeclStmts(ctx, body, decl_set);
        }

        // Phase 2: Synthesize DeclStmts for any VarDecls referenced but not
        // declared in the body. Unreachable blocks may be dropped during graph
        // construction while retaining blocks that reference those vars.
        // CIR crashes with "DeclRefExpr for decl not entered in LocalDeclMap"
        // when it encounters a reference to an undeclared variable.
        {
            std::unordered_set< clang::VarDecl * > referenced, declared;
            std::unordered_set< clang::Stmt * > seen1, seen2;
            CollectReferencedVars(body, referenced, seen1);
            CollectDeclaredVars(body, declared, seen2);
            // Also count vars from hoisted DeclStmts
            for (auto *ds : decl_stmts) {
                std::unordered_set< clang::Stmt * > seen3;
                CollectDeclaredVars(ds, declared, seen3);
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
        if (auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(body)) {
            for (auto *s : cs->body()) all_stmts.push_back(s);
        } else if (body) {
            all_stmts.push_back(body);
        }
        body = detail::MakeCompound(ctx, all_stmts);

        fn->setBody(body);
    }


} // namespace patchestry::ast
