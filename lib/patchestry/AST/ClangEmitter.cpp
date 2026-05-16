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
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &stmts) {
            auto loc = VirtualLoc(ctx);
            return clang::CompoundStmt::Create(ctx, stmts, clang::FPOptionsOverride(), loc, loc);
        }
    } // namespace detail

    namespace {

        // Deep-clone a Clang Expr tree to prevent shared Expr* nodes.
        // The CGraph pipeline may reuse branch_cond pointers across SNode
        // conditions (e.g., original and negated forms). CIR lowering requires
        // tree-unique Expr* nodes, so every condition must be cloned before emission.
        // CloneExpr moved to Utils.cpp as a public helper so the
        // structuring pipeline (CFGStructure.cpp) can proactively
        // clone shared branch_cond expressions when building
        // conjunctive/disjunctive merged conditions.  Use
        // patchestry::ast::CloneExpr from Utils.hpp.
        using patchestry::ast::CloneExpr;

    } // anonymous namespace

    // Shared utility — used by both ClangEmitter and ClangEmitterCleanup.
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
                : ctx_(ctx), fn_(fn), loc_(VirtualLoc(ctx)) {}

            clang::Stmt *Emit(const SNode *node) {
                if (!node) return nullptr;

                switch (node->Kind()) {
                case SNodeKind::kStmt:
                    return node->as< SStmt >()->Stmt();
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
            clang::SourceLocation Loc() const { return loc_; }

            clang::Stmt *EmitIfThenElse(const SIfThenElse *ite) {
                LOG_FATAL_IF(!ite->Cond(),
                    "SIfThenElse has null condition - structuring "
                    "rule produced a branch without a guard expression");
                auto *cond = EnsureRValue(ctx_, CloneExpr(ctx_, ite->Cond()));
                // Layer C Stage 3c: arms are std::vector<SNode*>.
                // Then-arm: required (NullStmt for empty list).
                // Else-arm: optional (nullptr for empty list = no else
                // clause; NullStmt would render as `else ;`).
                auto *then_stmt = EmitBodyList(ite->ThenList());
                // No else clause when the slot is empty.  Use
                // ElseBranch() to also handle the defensive
                // "[nullptr]" case (matches the prior gating which
                // skipped Emit when ElseBranch() returned null).
                clang::Stmt *else_stmt = nullptr;
                if (ite->ElseBranch()) {
                    else_stmt = EmitBodyList(ite->ElseList());
                }

                // Defensive else-if unwrap: when the else slot already
                // produces a CompoundStmt around a single IfStmt, the
                // pretty-printer would emit `else { if(...) }` — pull
                // the IfStmt out so it renders as `else if (...)`.
                // With vector storage this branch is hard to hit (size 1
                // already emits a bare IfStmt), but a multi-element list
                // that flattens to a single IfStmt via downstream
                // optimisation could still trigger it.  Cheap to keep.
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

            // Layer C Stage 3b helper: render a vector<SNode*> body
            // into a single clang::Stmt — empty → NullStmt, size 1 →
            // emit the lone child directly, size > 1 → CompoundStmt.
            // Loop body slots in WhileStmt/DoStmt/ForStmt require a
            // single Stmt, so the multi-element case must wrap.
            clang::Stmt *EmitBodyList(const std::vector< SNode * > &body_list) {
                if (body_list.empty())
                    return new (ctx_) clang::NullStmt(Loc());
                if (body_list.size() == 1) {
                    auto *s = Emit(body_list[0]);
                    return s ? s : new (ctx_) clang::NullStmt(Loc());
                }
                std::vector< clang::Stmt * > stmts;
                stmts.reserve(body_list.size());
                for (const auto *c : body_list)
                    if (auto *s = Emit(c)) stmts.push_back(s);
                return detail::MakeCompound(ctx_, stmts);
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
                auto *body = EmitBodyList(w->BodyList());

                return clang::WhileStmt::Create(
                    ctx_, nullptr, cond, body, Loc(), Loc(), Loc()
                );
            }

            clang::Stmt *EmitDoWhile(const SDoWhile *dw) {
                auto *body = EmitBodyList(dw->BodyList());
                clang::Expr *cond = nullptr;
                if (dw->Cond()) {
                    cond = EnsureRValue(ctx_, CloneExpr(ctx_, dw->Cond()));
                } else {
                    // do { ... } while(1) — SDoWhile with null condition.
                    cond = clang::IntegerLiteral::Create(
                        ctx_, llvm::APInt(32, 1), ctx_.IntTy, Loc());
                }

                return new (ctx_) clang::DoStmt(body, cond, Loc(), Loc(), Loc());
            }

            clang::Stmt *EmitFor(const SFor *f) {
                auto *body = EmitBodyList(f->BodyList());

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

                // Layer C Stage 3d: SCase.body_list and SSwitch.default_
                // are std::vector<SNode*>.  Empty list = fallthrough stub
                // (preserves the prior c.body == nullptr behaviour).
                // Build all child stmts, then optionally append a break
                // and wrap in CompoundStmt (CaseStmt requires a single
                // sub-stmt slot).
                auto build_case_substmt = [&](const std::vector< SNode * > &body_list)
                    -> clang::Stmt * {
                    if (body_list.empty()) {
                        // Fallthrough stub.
                        return new (ctx_) clang::NullStmt(Loc());
                    }
                    std::vector< clang::Stmt * > stmts;
                    stmts.reserve(body_list.size() + 1);
                    for (const auto *child : body_list)
                        if (auto *s = Emit(child)) stmts.push_back(s);
                    if (stmts.empty())
                        stmts.push_back(new (ctx_) clang::NullStmt(Loc()));
                    if (!detail::EndsWithTerminator(stmts.back()))
                        stmts.push_back(new (ctx_) clang::BreakStmt(Loc()));
                    return detail::MakeCompound(ctx_, stmts);
                };

                for (const auto &c : sw->Cases()) {
                    auto *case_stmt = clang::CaseStmt::Create(
                        ctx_, c.value, nullptr, Loc(), Loc(), Loc()
                    );
                    case_stmt->setSubStmt(build_case_substmt(c.body_list));
                    body_stmts.push_back(case_stmt);
                    switch_stmt->addSwitchCase(case_stmt);
                }

                if (!sw->DefaultBodyList().empty()) {
                    auto *def_stmt = new (ctx_) clang::DefaultStmt(
                        Loc(), Loc(),
                        build_case_substmt(sw->DefaultBodyList()));
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
                // Layer C Stage 3a: SLabel.body_ is std::vector<SNode*>.
                // Emit each child stmt and wrap in CompoundStmt only when
                // there's more than one (LabelStmt requires a single
                // sub-stmt slot).  Empty body → NullStmt.
                const auto &body_list = l->BodyList();
                clang::Stmt *sub = nullptr;
                if (body_list.empty()) {
                    sub = new (ctx_) clang::NullStmt(Loc());
                } else if (body_list.size() == 1) {
                    sub = Emit(body_list[0]);
                    if (!sub) sub = new (ctx_) clang::NullStmt(Loc());
                } else {
                    std::vector< clang::Stmt * > stmts;
                    stmts.reserve(body_list.size());
                    for (const auto *c : body_list)
                        if (auto *s = Emit(c)) stmts.push_back(s);
                    sub = detail::MakeCompound(ctx_, stmts);
                }
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
            // in raw Clang AST (SStmt stmts).  Must be called before Emit().
            void CollectGotoLabelDecls(SNode *node) {
                if (!node) return;
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
                if (auto *st = node->dyn_cast< SStmt >()) {
                    scan(st->Stmt());
                    return; // SStmt has no SNode children
                }
                node->for_each_child([&](SNode *c) {
                    CollectGotoLabelDecls(c);
                });
            }

            clang::ASTContext &ctx_;
            clang::FunctionDecl *fn_;
            clang::SourceLocation loc_;
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
            return r ? r : new (ctx) clang::NullStmt(VirtualLoc(ctx));
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



    // Common DeclStmt-hoisting + setBody finalization.  Used by both
    // the SNode* and std::vector<SNode*> entry points (Layer C Stage 4
    // extracted this so the vector overload doesn't need to duplicate it).
    static void FinalizeFunctionBody(clang::Stmt *body,
                                     clang::FunctionDecl *fn,
                                     clang::ASTContext &ctx) {
        if (!body) body = detail::MakeCompound(ctx, {});

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
        // declared in the body.  CIR crashes on undeclared local var refs.
        {
            std::unordered_set< clang::VarDecl * > referenced, declared;
            std::unordered_set< clang::Stmt * > seen1, seen2;
            CollectReferencedVars(body, referenced, seen1);
            CollectDeclaredVars(body, declared, seen2);
            for (auto *ds : decl_stmts) {
                std::unordered_set< clang::Stmt * > seen3;
                CollectDeclaredVars(ds, declared, seen3);
            }
            for (auto *vd : referenced) {
                if (declared.count(vd)) continue;
                if (llvm::isa< clang::ParmVarDecl >(vd)) continue;
                if (!vd->isLocalVarDecl()) continue;
                auto loc = VirtualLoc(ctx);
                auto *ds = new (ctx) clang::DeclStmt(
                    clang::DeclGroupRef(vd), loc, loc);
                decl_stmts.push_back(ds);
            }
        }

        // Build final body: hoisted decls + control flow.
        std::vector< clang::Stmt * > all_stmts;
        all_stmts.insert(all_stmts.end(), decl_stmts.begin(), decl_stmts.end());
        if (auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(body)) {
            for (auto *s : cs->body()) all_stmts.push_back(s);
        } else if (body) {
            all_stmts.push_back(body);
        }
        fn->setBody(detail::MakeCompound(ctx, all_stmts));
    }

    void EmitClangAST(const std::vector< SNode * > &root_children,
                      clang::FunctionDecl *fn, clang::ASTContext &ctx) {
        // Layer C Stage 4: take the function-body slot as a vector
        // directly so callers no longer need to wrap in an SSeq.
        Emitter emitter(ctx, fn);
        for (auto *c : root_children) emitter.CollectGotoLabelDecls(c);

        std::vector< clang::Stmt * > body_stmts;
        body_stmts.reserve(root_children.size());
        for (auto *c : root_children)
            if (auto *s = emitter.Emit(c)) body_stmts.push_back(s);
        FinalizeFunctionBody(detail::MakeCompound(ctx, body_stmts), fn, ctx);
    }

    void EmitClangAST(SNode *root, clang::FunctionDecl *fn,
                      clang::ASTContext &ctx) {
        Emitter emitter(ctx, fn);
        emitter.CollectGotoLabelDecls(root);
        FinalizeFunctionBody(emitter.Emit(root), fn, ctx);
    }

    namespace {

        std::string LabelName(const clang::LabelDecl *decl) {
            if (!decl) return "<null-label>";
            return decl->getName().str();
        }

        std::string StmtName(const clang::Stmt *stmt) {
            if (!stmt) return "<null-stmt>";
            return stmt->getStmtClassName();
        }

        void AddClangEmissionDiagnostic(
            ClangEmissionValidationReport &report,
            const std::string &message
        ) {
            report.diagnostics.push_back(message);
        }

        bool IsEmptyLabelSubStmt(clang::Stmt *stmt) {
            if (!stmt) return true;
            if (llvm::isa<clang::NullStmt>(stmt)) return true;
            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt))
                return compound->body_empty();
            return false;
        }

        struct ClangEmissionValidationState {
            ClangEmissionValidationReport report;
            std::unordered_map<clang::LabelDecl *, unsigned> label_defs;
            std::unordered_map<clang::LabelDecl *, unsigned> goto_refs;
            std::unordered_map<std::string, unsigned> label_name_defs;
            std::unordered_set<clang::LabelDecl *> empty_label_defs;
        };

        void ValidateClangStmtRecursive(
            clang::Stmt *stmt,
            ClangEmissionValidationState &state,
            unsigned loop_depth,
            unsigned switch_depth,
            const char *context
        );

        void ValidateClangCompound(
            clang::CompoundStmt *compound,
            ClangEmissionValidationState &state,
            unsigned loop_depth,
            unsigned switch_depth,
            const char *context
        ) {
            bool after_terminator = false;
            for (auto *child : compound->body()) {
                if (!child) continue;
                if (after_terminator) {
                    if (llvm::isa<clang::LabelStmt>(child)
                        || llvm::isa<clang::CaseStmt>(child)
                        || llvm::isa<clang::DefaultStmt>(child)) {
                        after_terminator = false;
                    } else {
                        ++state.report.unreachable_statements;
                        AddClangEmissionDiagnostic(
                            state.report,
                            std::string("unreachable ")
                                + StmtName(child)
                                + " after terminator in " + context);
                    }
                }
                ValidateClangStmtRecursive(child, state, loop_depth,
                                           switch_depth, context);
                if (detail::EndsWithTerminator(child))
                    after_terminator = true;
            }
        }

        void ValidateClangStmtRecursive(
            clang::Stmt *stmt,
            ClangEmissionValidationState &state,
            unsigned loop_depth,
            unsigned switch_depth,
            const char *context
        ) {
            if (!stmt) return;

            if (auto *label = llvm::dyn_cast<clang::LabelStmt>(stmt)) {
                auto *decl = label->getDecl();
                ++state.report.emitted_labels;
                ++state.label_defs[decl];
                ++state.label_name_defs[LabelName(decl)];
                if (IsEmptyLabelSubStmt(label->getSubStmt())) {
                    ++state.report.empty_labels;
                    state.empty_label_defs.insert(decl);
                }
                ValidateClangStmtRecursive(label->getSubStmt(), state,
                                           loop_depth, switch_depth, "label");
                return;
            }

            if (auto *go = llvm::dyn_cast<clang::GotoStmt>(stmt)) {
                ++state.report.emitted_gotos;
                ++state.goto_refs[go->getLabel()];
                return;
            }

            if (llvm::isa<clang::BreakStmt>(stmt)) {
                ++state.report.break_stmts;
                if (loop_depth == 0 && switch_depth == 0) {
                    AddClangEmissionDiagnostic(
                        state.report,
                        std::string("break outside loop/switch in ")
                            + context);
                }
                return;
            }

            if (llvm::isa<clang::ContinueStmt>(stmt)) {
                ++state.report.continue_stmts;
                if (loop_depth == 0) {
                    AddClangEmissionDiagnostic(
                        state.report,
                        std::string("continue outside loop in ") + context);
                }
                return;
            }

            if (llvm::isa<clang::ReturnStmt>(stmt))
                return;

            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
                ValidateClangCompound(compound, state, loop_depth,
                                      switch_depth, "compound");
                return;
            }

            if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(stmt)) {
                if (!ifs->getThen()) {
                    AddClangEmissionDiagnostic(
                        state.report,
                        std::string("if without then body in ") + context);
                }
                ValidateClangStmtRecursive(ifs->getThen(), state, loop_depth,
                                           switch_depth, "if-then");
                ValidateClangStmtRecursive(ifs->getElse(), state, loop_depth,
                                           switch_depth, "if-else");
                return;
            }

            if (auto *while_stmt = llvm::dyn_cast<clang::WhileStmt>(stmt)) {
                ValidateClangStmtRecursive(while_stmt->getBody(), state,
                                           loop_depth + 1, switch_depth,
                                           "while");
                return;
            }

            if (auto *do_stmt = llvm::dyn_cast<clang::DoStmt>(stmt)) {
                ValidateClangStmtRecursive(do_stmt->getBody(), state,
                                           loop_depth + 1, switch_depth,
                                           "do-while");
                return;
            }

            if (auto *for_stmt = llvm::dyn_cast<clang::ForStmt>(stmt)) {
                ValidateClangStmtRecursive(for_stmt->getBody(), state,
                                           loop_depth + 1, switch_depth,
                                           "for");
                return;
            }

            if (auto *switch_stmt = llvm::dyn_cast<clang::SwitchStmt>(stmt)) {
                ++state.report.switch_stmts;
                if (!switch_stmt->getBody()) {
                    AddClangEmissionDiagnostic(
                        state.report,
                        std::string("switch without body in ") + context);
                }
                ValidateClangStmtRecursive(switch_stmt->getBody(), state,
                                           loop_depth, switch_depth + 1,
                                           "switch");
                return;
            }

            if (auto *case_stmt = llvm::dyn_cast<clang::CaseStmt>(stmt)) {
                ++state.report.case_stmts;
                if (IsEmptyLabelSubStmt(case_stmt->getSubStmt())) {
                    AddClangEmissionDiagnostic(
                        state.report,
                        std::string("empty switch case body in ") + context);
                }
                ValidateClangStmtRecursive(case_stmt->getSubStmt(), state,
                                           loop_depth, switch_depth,
                                           "switch-case");
                return;
            }

            if (auto *default_stmt = llvm::dyn_cast<clang::DefaultStmt>(stmt)) {
                ++state.report.default_stmts;
                if (IsEmptyLabelSubStmt(default_stmt->getSubStmt())) {
                    AddClangEmissionDiagnostic(
                        state.report,
                        std::string("empty switch default body in ")
                            + context);
                }
                ValidateClangStmtRecursive(default_stmt->getSubStmt(), state,
                                           loop_depth, switch_depth,
                                           "switch-default");
                return;
            }

            for (auto *child : stmt->children()) {
                ValidateClangStmtRecursive(child, state, loop_depth,
                                           switch_depth, context);
            }
        }

    } // namespace

    ClangEmissionValidationReport
    ValidateEmittedClangAST(clang::FunctionDecl *fn) {
        ClangEmissionValidationState state;
        if (!fn) {
            AddClangEmissionDiagnostic(state.report,
                                       "missing function declaration");
            return state.report;
        }
        if (!fn->getBody()) {
            AddClangEmissionDiagnostic(
                state.report,
                "function " + fn->getNameAsString() + " has no body");
            return state.report;
        }

        ValidateClangStmtRecursive(fn->getBody(), state, /*loop_depth=*/0,
                                   /*switch_depth=*/0, "function");

        for (const auto &[target, count] : state.goto_refs) {
            if (!state.label_defs.contains(target)) {
                state.report.dangling_gotos += count;
                AddClangEmissionDiagnostic(
                    state.report,
                    "dangling goto target " + LabelName(target));
            }
        }

        for (const auto &[label, count] : state.label_defs) {
            if (count > 1) {
                state.report.duplicate_labels += count - 1;
                AddClangEmissionDiagnostic(
                    state.report,
                    "duplicate label definition " + LabelName(label));
            }
            if (state.empty_label_defs.contains(label)
                && !state.goto_refs.contains(label)) {
                AddClangEmissionDiagnostic(
                    state.report,
                    "empty unreferenced label " + LabelName(label));
            }
        }

        for (const auto &[name, count] : state.label_name_defs) {
            if (count > 1) {
                AddClangEmissionDiagnostic(
                    state.report,
                    "duplicate label name " + name);
            }
        }

        return state.report;
    }

} // namespace patchestry::ast
