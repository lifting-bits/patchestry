/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "CfgFoldInternal.hpp"

namespace patchestry::ast {
namespace detail {

    // ---------------------------------------------------------------
    // Post-collapse transforms: RefineBreakContinue, RefineWhileToFor, RefineDeadLabels
    // ---------------------------------------------------------------

    // Legacy RefineBreakContinue — replaced by ScopeBreak (Phase 4).
    // Kept for reference; will be removed in cleanup.
    [[maybe_unused]]
    void RefineBreakContinue(SNode *node, std::string_view loop_exit_label,
                    std::string_view loop_header_label, SNodeFactory &factory) {
        if (!node) return;

        if (auto *seq = node->dyn_cast<SSeq>()) {
            // Recurse into children. When entering a loop child, we need to
            // determine the new loop exit/header labels from sibling context.
            for (size_t i = 0; i < seq->Size(); ++i) {
                auto *child = (*seq)[i];

                // For SWhile/SDoWhile, compute new loop context labels
                if (auto *w = child->dyn_cast<SWhile>()) {
                    // New loop_exit_label: label of the next sibling (if SLabel)
                    std::string_view new_exit;
                    if (i + 1 < seq->Size()) {
                        if (auto *lbl = (*seq)[i + 1]->dyn_cast<SLabel>()) {
                            new_exit = lbl->Name();
                        }
                    }
                    if (new_exit.empty()) new_exit = loop_exit_label;

                    // New loop_header_label: if prev sibling is SLabel wrapping
                    // this while, use that label. Also check if the while body's
                    // first child is an SLabel (header-stmts-inside-loop pattern).
                    std::string_view new_header;
                    if (i > 0) {
                        if (auto *lbl = (*seq)[i - 1]->dyn_cast<SLabel>()) {
                            new_header = lbl->Name();
                        }
                    }
                    if (new_header.empty()) {
                        if (auto *body_seq = w->Body() ? w->Body()->dyn_cast<SSeq>() : nullptr) {
                            if (body_seq->Size() > 0) {
                                if (auto *lbl = (*body_seq)[0]->dyn_cast<SLabel>()) {
                                    new_header = lbl->Name();
                                }
                            }
                        }
                    }

                    RefineBreakContinue(w->Body(), new_exit, new_header, factory);
                    continue;
                }

                if (auto *dw = child->dyn_cast<SDoWhile>()) {
                    std::string_view new_exit;
                    if (i + 1 < seq->Size()) {
                        if (auto *lbl = (*seq)[i + 1]->dyn_cast<SLabel>()) {
                            new_exit = lbl->Name();
                        }
                    }
                    if (new_exit.empty()) new_exit = loop_exit_label;

                    std::string_view new_header;
                    if (i > 0) {
                        if (auto *lbl = (*seq)[i - 1]->dyn_cast<SLabel>()) {
                            new_header = lbl->Name();
                        }
                    }

                    RefineBreakContinue(dw->Body(), new_exit, new_header, factory);
                    continue;
                }

                // For non-loop children, recurse with current loop context
                RefineBreakContinue(child, loop_exit_label, loop_header_label, factory);
            }

            // After recursing, check if last child is SGoto targeting loop exit/header
            if (seq->Size() > 0) {
                auto *last = (*seq)[seq->Size() - 1];
                if (auto *g = last->dyn_cast<SGoto>()) {
                    if (!loop_exit_label.empty() && g->Target() == loop_exit_label) {
                        seq->ReplaceChild(seq->Size() - 1, factory.Make<SBreak>());
                    } else if (!loop_header_label.empty() && g->Target() == loop_header_label) {
                        seq->ReplaceChild(seq->Size() - 1, factory.Make<SContinue>());
                    }
                }
            }
        }
        else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            RefineBreakContinue(ite->ThenBranch(), loop_exit_label, loop_header_label, factory);
            RefineBreakContinue(ite->ElseBranch(), loop_exit_label, loop_header_label, factory);
        }
        else if (auto *sw = node->dyn_cast<SSwitch>()) {
            // Switch does not change loop context
            for (auto &c : sw->Cases()) {
                RefineBreakContinue(c.body, loop_exit_label, loop_header_label, factory);
            }
            RefineBreakContinue(sw->DefaultBody(), loop_exit_label, loop_header_label, factory);
        }
        else if (auto *lbl = node->dyn_cast<SLabel>()) {
            RefineBreakContinue(lbl->Body(), loop_exit_label, loop_header_label, factory);
        }
        else if (auto *f = node->dyn_cast<SFor>()) {
            // SFor has its own loop context
            RefineBreakContinue(f->Body(), loop_exit_label, loop_header_label, factory);
        }
        // SBlock, SGoto, SBreak, SContinue, SReturn: leaf nodes, nothing to do
    }

    // ScopeBreak: scope-aware break/continue conversion using
    // exit/header labels stored on loop SNode types (Phase 4).
    // Replaces the sibling-matching heuristics of RefineBreakContinue.
    void ScopeBreak(SNode *node, std::string_view exit_label,
                    std::string_view header_label, SNodeFactory &factory) {
        if (!node) return;

        // Replace SGoto targeting loop exit/header with break/continue.
        auto tryReplace = [&](SNode *parent, size_t child_idx) {
            auto *seq = parent->dyn_cast<SSeq>();
            if (!seq || child_idx >= seq->Size()) return;
            auto *g = (*seq)[child_idx]->dyn_cast<SGoto>();
            if (!g) return;
            if (!exit_label.empty() && g->Target() == exit_label) {
                seq->ReplaceChild(child_idx, factory.Make<SBreak>());
            } else if (!header_label.empty() && g->Target() == header_label) {
                seq->ReplaceChild(child_idx, factory.Make<SContinue>());
            }
        };

        if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ++i) {
                auto *child = (*seq)[i];

                // Enter loop scope: use loop's own exit/header labels.
                if (auto *w = child->dyn_cast<SWhile>()) {
                    ScopeBreak(w->Body(), w->ExitLabel(), w->HeaderLabel(), factory);
                    continue;
                }
                if (auto *dw = child->dyn_cast<SDoWhile>()) {
                    ScopeBreak(dw->Body(), dw->ExitLabel(), dw->HeaderLabel(), factory);
                    continue;
                }
                if (auto *f = child->dyn_cast<SFor>()) {
                    ScopeBreak(f->Body(), f->ExitLabel(), f->HeaderLabel(), factory);
                    continue;
                }

                // Non-loop children: inherit current loop scope.
                ScopeBreak(child, exit_label, header_label, factory);
            }
            // Check last child for goto → break/continue.
            if (seq->Size() > 0) {
                tryReplace(seq, seq->Size() - 1);
            }
        }
        else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            ScopeBreak(ite->ThenBranch(), exit_label, header_label, factory);
            ScopeBreak(ite->ElseBranch(), exit_label, header_label, factory);
        }
        else if (auto *sw = node->dyn_cast<SSwitch>()) {
            // Switch inherits enclosing loop scope (break in switch != loop break).
            // Don't convert gotos to break inside switch — C break exits switch, not loop.
            for (auto &c : sw->Cases())
                ScopeBreak(c.body, exit_label, header_label, factory);
            ScopeBreak(sw->DefaultBody(), exit_label, header_label, factory);
        }
        else if (auto *lbl = node->dyn_cast<SLabel>()) {
            ScopeBreak(lbl->Body(), exit_label, header_label, factory);
        }
        else if (auto *w = node->dyn_cast<SWhile>()) {
            ScopeBreak(w->Body(), w->ExitLabel(), w->HeaderLabel(), factory);
        }
        else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            ScopeBreak(dw->Body(), dw->ExitLabel(), dw->HeaderLabel(), factory);
        }
        else if (auto *f = node->dyn_cast<SFor>()) {
            ScopeBreak(f->Body(), f->ExitLabel(), f->HeaderLabel(), factory);
        }
    }

    // --- RefineWhileToFor helpers ---

    bool IsAssignOrDecl(clang::Stmt *s) {
        if (!s) return false;
        if (auto *bo = llvm::dyn_cast<clang::BinaryOperator>(s)) {
            return bo->getOpcode() == clang::BO_Assign;
        }
        return llvm::isa<clang::DeclStmt>(s);
    }

    bool IsIncrement(clang::Stmt *s) {
        if (!s) return false;
        if (auto *uo = llvm::dyn_cast<clang::UnaryOperator>(s)) {
            auto op = uo->getOpcode();
            return op == clang::UO_PreInc || op == clang::UO_PostInc
                || op == clang::UO_PreDec || op == clang::UO_PostDec;
        }
        if (auto *bo = llvm::dyn_cast<clang::BinaryOperator>(s)) {
            auto op = bo->getOpcode();
            return op == clang::BO_Assign || op == clang::BO_AddAssign
                || op == clang::BO_SubAssign;
        }
        return false;
    }

    // Extract the DeclRefExpr from an assignment or decl statement
    clang::DeclRefExpr *GetAssignTarget(clang::Stmt *s) {
        if (!s) return nullptr;
        if (auto *bo = llvm::dyn_cast<clang::BinaryOperator>(s)) {
            return llvm::dyn_cast<clang::DeclRefExpr>(bo->getLHS()->IgnoreParenCasts());
        }
        if (auto *uo = llvm::dyn_cast<clang::UnaryOperator>(s)) {
            return llvm::dyn_cast<clang::DeclRefExpr>(uo->getSubExpr()->IgnoreParenCasts());
        }
        if (auto *ds = llvm::dyn_cast<clang::DeclStmt>(s)) {
            if (ds->isSingleDecl()) {
                if (auto *vd = llvm::dyn_cast<clang::VarDecl>(ds->getSingleDecl())) {
                    (void)vd;
                    // DeclStmt doesn't directly produce a DeclRefExpr;
                    // we can't easily manufacture one without an ASTContext.
                    // Return nullptr — SameVariable will handle this via VarDecl matching.
                    return nullptr;
                }
            }
        }
        return nullptr;
    }

    // Extract the VarDecl referenced by a statement (works for assign, unary, decl)
    clang::VarDecl *GetReferencedVar(clang::Stmt *s) {
        if (auto *dre = GetAssignTarget(s)) {
            return llvm::dyn_cast<clang::VarDecl>(dre->getDecl());
        }
        if (auto *ds = llvm::dyn_cast<clang::DeclStmt>(s)) {
            if (ds->isSingleDecl()) {
                return llvm::dyn_cast<clang::VarDecl>(ds->getSingleDecl());
            }
        }
        return nullptr;
    }

    // Check if an expression tree contains a reference to the given VarDecl
    bool ContainsVarRef(clang::Stmt *s, clang::VarDecl *vd) {
        if (!s || !vd) return false;
        if (auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(s)) {
            return dre->getDecl() == vd;
        }
        for (auto *child : s->children()) {
            if (ContainsVarRef(child, vd)) return true;
        }
        return false;
    }

    // Check that init, cond, and inc all reference the same variable
    bool SameVariable(clang::Stmt *init, clang::Expr *cond, clang::Expr *inc) {
        auto *init_var = GetReferencedVar(init);
        if (!init_var) return false;
        auto *inc_var = GetReferencedVar(inc);
        if (!inc_var) return false;
        if (init_var != inc_var) return false;
        return ContainsVarRef(cond, init_var);
    }

    // RefineWhileToFor: convert init/while(cond)/inc patterns to SFor
    void RefineWhileToFor(SNode *node, SNodeFactory &factory, clang::ASTContext &ctx) {
        if (!node) return;

        if (auto *seq = node->dyn_cast<SSeq>()) {
            // Scan for pattern: SBlock(init), SWhile(cond, SSeq(..., SBlock(inc)))
            for (size_t i = 0; i < seq->Size(); ++i) {
                auto *w = (*seq)[i]->dyn_cast<SWhile>();
                if (!w) continue;

                // Check for init before the while
                clang::Stmt *init_stmt = nullptr;
                bool has_init = false;
                if (i > 0) {
                    auto *prev = (*seq)[i - 1]->dyn_cast<SBlock>();
                    if (prev && prev->Size() == 1 && IsAssignOrDecl(prev->Stmts()[0])) {
                        init_stmt = prev->Stmts()[0];
                        has_init = true;
                    }
                }

                // Check for inc at end of while body
                clang::Expr *inc_expr = nullptr;
                auto *body_seq = w->Body() ? w->Body()->dyn_cast<SSeq>() : nullptr;
                if (body_seq && body_seq->Size() > 0) {
                    auto *last = (*body_seq)[body_seq->Size() - 1]->dyn_cast<SBlock>();
                    if (last && last->Size() == 1 && IsIncrement(last->Stmts()[0])) {
                        inc_expr = llvm::dyn_cast<clang::Expr>(last->Stmts()[0]);
                    }
                }

                if (has_init && inc_expr && SameVariable(init_stmt, w->Cond(), inc_expr)) {
                    // Remove inc from body
                    body_seq->RemoveChild(body_seq->Size() - 1);
                    // Build SFor
                    auto *for_node = factory.Make<SFor>(init_stmt, w->Cond(), inc_expr, w->Body());
                    // Replace [i-1, i+1) with the for node
                    seq->ReplaceRange(i - 1, i + 1, for_node);
                    --i; // adjust for removed element
                }
            }

            // Recurse into remaining children
            for (size_t i = 0; i < seq->Size(); ++i) {
                RefineWhileToFor((*seq)[i], factory, ctx);
            }
        }
        else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            RefineWhileToFor(ite->ThenBranch(), factory, ctx);
            RefineWhileToFor(ite->ElseBranch(), factory, ctx);
        }
        else if (auto *w = node->dyn_cast<SWhile>()) {
            RefineWhileToFor(w->Body(), factory, ctx);
        }
        else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            RefineWhileToFor(dw->Body(), factory, ctx);
        }
        else if (auto *f = node->dyn_cast<SFor>()) {
            RefineWhileToFor(f->Body(), factory, ctx);
        }
        else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases()) {
                RefineWhileToFor(c.body, factory, ctx);
            }
            RefineWhileToFor(sw->DefaultBody(), factory, ctx);
        }
        else if (auto *lbl = node->dyn_cast<SLabel>()) {
            RefineWhileToFor(lbl->Body(), factory, ctx);
        }
    }

    // --- RefineDeadLabels: dead label removal ---

    // Collect all SGoto target names in the tree
    void CollectGotoTargets(SNode *node, std::unordered_set<std::string> &targets) {
        if (!node) return;

        if (auto *g = node->dyn_cast<SGoto>()) {
            targets.emplace(g->Target());
        }
        else if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ++i) {
                CollectGotoTargets((*seq)[i], targets);
            }
        }
        else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            CollectGotoTargets(ite->ThenBranch(), targets);
            CollectGotoTargets(ite->ElseBranch(), targets);
        }
        else if (auto *w = node->dyn_cast<SWhile>()) {
            CollectGotoTargets(w->Body(), targets);
        }
        else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            CollectGotoTargets(dw->Body(), targets);
        }
        else if (auto *f = node->dyn_cast<SFor>()) {
            CollectGotoTargets(f->Body(), targets);
        }
        else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases()) {
                CollectGotoTargets(c.body, targets);
            }
            CollectGotoTargets(sw->DefaultBody(), targets);
        }
        else if (auto *lbl = node->dyn_cast<SLabel>()) {
            CollectGotoTargets(lbl->Body(), targets);
        }
        else if (auto *blk = node->dyn_cast<SBlock>()) {
            // Scan raw Clang AST stmts for GotoStmt targets
            // (e.g. from OperationStmt switch case inlining)
            std::function<void(clang::Stmt *)> scanStmt = [&](clang::Stmt *s) {
                if (!s) return;
                if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(s)) {
                    targets.emplace(gs->getLabel()->getName().str());
                }
                for (auto *child : s->children()) scanStmt(child);
            };
            for (auto *s : blk->Stmts()) scanStmt(s);
        }
    }

    // --- RefineGotoElseNesting ---
    //
    // Detects:
    //   SSeq([..., IfThenElse(C, body, SGoto(L)), sibling0, sibling1, ...])
    //
    // and transforms to:
    //   SSeq([..., IfThenElse(C, SSeq([body, sibling0, sibling1, ...]), SGoto(L))])
    //
    // This nests the code that follows an if-with-goto-else into the
    // then-body, producing e.g. "if (length != 0) { init; while; ... }"
    // instead of "if (length) { init } else goto L; while; ...".
    //
    // Safety: only absorb if no label in the absorbed range is a goto
    // target from OUTSIDE the absorbed range (except the SGoto(L) itself).
    // For irreducible cases the transform is skipped.

    // Collect all label names defined in a subtree.
    void CollectLabels(SNode *node, std::unordered_set<std::string> &labels) {
        if (!node) return;
        if (auto *lbl = node->dyn_cast<SLabel>()) {
            labels.emplace(lbl->Name());
            CollectLabels(lbl->Body(), labels);
        } else if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ++i)
                CollectLabels((*seq)[i], labels);
        } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            CollectLabels(ite->ThenBranch(), labels);
            CollectLabels(ite->ElseBranch(), labels);
        } else if (auto *w = node->dyn_cast<SWhile>()) {
            CollectLabels(w->Body(), labels);
        } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            CollectLabels(dw->Body(), labels);
        } else if (auto *f = node->dyn_cast<SFor>()) {
            CollectLabels(f->Body(), labels);
        } else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases()) CollectLabels(c.body, labels);
            CollectLabels(sw->DefaultBody(), labels);
        }
    }

    // Phase 4: ancestor_gotos accumulates goto targets from parent
    // scopes so we can reject absorption when an ancestor goto would
    // cross into the absorbed range.
    void RefineGotoElseNesting(SNode *node, SNodeFactory &factory,
                               const std::unordered_set<std::string> &ancestor_gotos) {
        if (!node) return;

        // Collect gotos from this node to pass to children (Phase 4).
        auto makeChildAncestorGotos = [&](SNode *exclude) {
            std::unordered_set<std::string> child_ag = ancestor_gotos;
            // Add gotos from sibling branches at this level.
            // For an IfThenElse: gotos from the other branch are
            // "ancestor gotos" for the branch being recursed into.
            if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                if (exclude != ite->ThenBranch())
                    CollectGotoTargets(ite->ThenBranch(), child_ag);
                if (exclude != ite->ElseBranch())
                    CollectGotoTargets(ite->ElseBranch(), child_ag);
            }
            return child_ag;
        };

        // Recurse first (bottom-up) so inner SSeqs are resolved before outer.
        if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ++i)
                RefineGotoElseNesting((*seq)[i], factory, ancestor_gotos);
        } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            RefineGotoElseNesting(ite->ThenBranch(), factory,
                                  makeChildAncestorGotos(ite->ThenBranch()));
            RefineGotoElseNesting(ite->ElseBranch(), factory,
                                  makeChildAncestorGotos(ite->ElseBranch()));
            return;
        } else if (auto *w = node->dyn_cast<SWhile>()) {
            RefineGotoElseNesting(w->Body(), factory, ancestor_gotos);
            return;
        } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            RefineGotoElseNesting(dw->Body(), factory, ancestor_gotos);
            return;
        } else if (auto *f = node->dyn_cast<SFor>()) {
            RefineGotoElseNesting(f->Body(), factory, ancestor_gotos);
            return;
        } else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases())
                RefineGotoElseNesting(c.body, factory, ancestor_gotos);
            RefineGotoElseNesting(sw->DefaultBody(), factory, ancestor_gotos);
            return;
        } else if (auto *lbl = node->dyn_cast<SLabel>()) {
            RefineGotoElseNesting(lbl->Body(), factory, ancestor_gotos);
            return;
        } else {
            return;
        }

        // node is an SSeq — scan for IfThenElse with SGoto branch.
        // Phase 3: iterate until no more changes (fixed point).
        auto *seq = node->as<SSeq>();
        bool changed = true;
        size_t max_rounds = seq->Size() + 1;  // safety bound
        while (changed && max_rounds-- > 0) {
            changed = false;
            for (size_t i = 0; i < seq->Size(); ++i) {
                auto *child = (*seq)[i];

                // Peel through wrapping: SSeq→last child, SLabel→body
                SNode *candidate = child;
                while (candidate) {
                    if (auto *inner_seq = candidate->dyn_cast<SSeq>()) {
                        if (inner_seq->Size() == 0) break;
                        candidate = (*inner_seq)[inner_seq->Size() - 1];
                    } else if (auto *inner_lbl = candidate->dyn_cast<SLabel>()) {
                        candidate = inner_lbl->Body();
                    } else {
                        break;
                    }
                }
                auto *ite = candidate ? candidate->dyn_cast<SIfThenElse>() : nullptr;
                if (!ite) continue;

                // Match SGoto in either branch (Phase 2).
                SGoto *branch_goto = nullptr;
                bool is_then_goto = false;
                if (auto *g = ite->ElseBranch()
                        ? ite->ElseBranch()->dyn_cast<SGoto>() : nullptr) {
                    branch_goto = g;
                    is_then_goto = false;
                } else if (auto *g = ite->ThenBranch()
                        ? ite->ThenBranch()->dyn_cast<SGoto>() : nullptr) {
                    branch_goto = g;
                    is_then_goto = true;
                }
                if (!branch_goto) continue;
                if (i + 1 >= seq->Size()) continue;

                SNode *body_branch = is_then_goto
                    ? ite->ElseBranch() : ite->ThenBranch();

                std::string target(branch_goto->Target());

                // --- Phase 1: find the label boundary ---
                auto startsWithLabel = [&](SNode *n) -> bool {
                    if (!n) return false;
                    if (auto *lbl = n->dyn_cast<SLabel>())
                        return lbl->Name() == target;
                    if (auto *s = n->dyn_cast<SSeq>()) {
                        for (size_t k = 0; k < s->Size(); ++k) {
                            if (auto *lbl = (*s)[k]->dyn_cast<SLabel>())
                                return lbl->Name() == target;
                            if (auto *blk = (*s)[k]->dyn_cast<SBlock>()) {
                                if (blk->Stmts().empty()) continue;
                            }
                            break;
                        }
                    }
                    return false;
                };

                size_t label_pos = seq->Size();
                bool label_is_toplevel = false;
                for (size_t j = i + 1; j < seq->Size(); ++j) {
                    if (startsWithLabel((*seq)[j])) {
                        label_pos = j;
                        label_is_toplevel = true;
                        break;
                    }
                }

                if (!label_is_toplevel) {
                    std::unordered_set<std::string> nested_labels;
                    for (size_t j = i + 1; j < seq->Size(); ++j)
                        CollectLabels((*seq)[j], nested_labels);
                    if (nested_labels.find(target) == nested_labels.end())
                        continue;
                    label_pos = seq->Size();
                }

                if (label_pos == i + 1) continue;

                size_t absorb_end = label_is_toplevel ? label_pos : seq->Size();

                // Safety: collect labels in the absorption range.
                std::unordered_set<std::string> range_labels;
                for (size_t j = i + 1; j < absorb_end; ++j)
                    CollectLabels((*seq)[j], range_labels);

                // Collect goto targets from OUTSIDE the absorption range:
                // siblings [0..i] + ancestor gotos (Phase 4).
                std::unordered_set<std::string> external_gotos = ancestor_gotos;
                for (size_t j = 0; j <= i; ++j)
                    CollectGotoTargets((*seq)[j], external_gotos);
                external_gotos.erase(target);
                // Gotos from the body branch move with absorbed content.
                std::unordered_set<std::string> body_gotos;
                CollectGotoTargets(body_branch, body_gotos);
                for (auto &tg : body_gotos)
                    external_gotos.erase(tg);

                bool unsafe = false;
                for (auto &eg : external_gotos) {
                    if (range_labels.count(eg)) { unsafe = true; break; }
                }
                if (unsafe) continue;

                // Absorb siblings [i+1..absorb_end) into the body branch.
                auto *new_body = factory.Make<SSeq>();
                if (body_branch)
                    new_body->AddChild(body_branch);
                for (size_t j = i + 1; j < absorb_end; ++j)
                    new_body->AddChild((*seq)[j]);

                for (size_t j = absorb_end - 1; j > i; --j)
                    seq->RemoveChild(j);

                if (is_then_goto) {
                    ite->SetElseBranch(new_body);
                } else {
                    ite->SetThenBranch(new_body);
                }

                if (label_is_toplevel) {
                    if (is_then_goto) {
                        ite->SetThenBranch(nullptr);
                    } else {
                        ite->SetElseBranch(nullptr);
                    }
                }

                changed = true;
                break;  // Restart scan from the beginning of this SSeq.
            }
        }
    }

    // --- RefineHoistLabel ---
    //
    // Hoists labels out of if-else branches to the enclosing SSeq
    // when they are cross-scope goto targets.  This eliminates gotos
    // that jump from one branch of an if into the other.
    //
    // Pattern:
    //   SSeq [
    //     ...,
    //     SIfThenElse(C, then_body, SLabel(L, content)),  // label in else
    //     continuation,
    //     ...
    //   ]
    //   where SGoto(L) exists OUTSIDE this SIfThenElse
    //
    // Transform:
    //   SSeq [
    //     ...,
    //     SIfThenElse(C,
    //       SSeq[then_body, SGoto(after_L)],  // skip-goto appended
    //       SGoto(L)),                          // replaced label with goto
    //     SLabel(L, content),                   // hoisted
    //     SLabel(after_L, continuation),         // skip target
    //     ...
    //   ]

    // Find an SLabel in a branch whose name matches one of the target
    // goto names.  Searches the top-level SSeq children (not deeply
    // nested in control flow — only in SSeq/SLabel wrappers).
    SLabel *FindHoistableLabel(
            SNode *node,
            const std::unordered_set<std::string> &goto_targets) {
        if (!node) return nullptr;
        if (auto *lbl = node->dyn_cast<SLabel>()) {
            if (goto_targets.count(std::string(lbl->Name())))
                return lbl;
            return FindHoistableLabel(lbl->Body(), goto_targets);
        }
        if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t k = 0; k < seq->Size(); ++k) {
                if (auto *found = FindHoistableLabel((*seq)[k], goto_targets))
                    return found;
            }
        }
        // Descend into IfThenElse branches — labels there are
        // the primary hoist targets.
        if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            if (auto *f = FindHoistableLabel(ite->ThenBranch(), goto_targets))
                return f;
            if (auto *f = FindHoistableLabel(ite->ElseBranch(), goto_targets))
                return f;
        }
        // Don't descend into loops or switches — labels inside
        // those have different scope semantics (break/continue).
        return nullptr;
    }

    // Replace a specific SLabel anywhere in SSeq/SLabel wrappers
    // with an SGoto.  Returns the position in the nearest SSeq
    // (for insertion of subsequent nodes), or false if not found.
    bool ReplaceLabelWithGoto(SNode *node, SLabel *target_lbl,
                                     SNodeFactory &factory) {
        if (!node) return false;
        if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t k = 0; k < seq->Size(); ++k) {
                if ((*seq)[k] == target_lbl) {
                    auto *g = factory.Make<SGoto>(
                        factory.Intern(target_lbl->Name()));
                    seq->ReplaceChild(k, g);
                    return true;
                }
                if (ReplaceLabelWithGoto((*seq)[k], target_lbl, factory))
                    return true;
            }
        }
        if (auto *lbl = node->dyn_cast<SLabel>()) {
            if (lbl->Body() == target_lbl) {
                auto *g = factory.Make<SGoto>(
                    factory.Intern(target_lbl->Name()));
                lbl->SetBody(g);
                return true;
            }
            return ReplaceLabelWithGoto(lbl->Body(), target_lbl, factory);
        }
        if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            if (ite->ThenBranch() == target_lbl) {
                ite->SetThenBranch(factory.Make<SGoto>(
                    factory.Intern(target_lbl->Name())));
                return true;
            }
            if (ite->ElseBranch() == target_lbl) {
                ite->SetElseBranch(factory.Make<SGoto>(
                    factory.Intern(target_lbl->Name())));
                return true;
            }
            if (ReplaceLabelWithGoto(ite->ThenBranch(), target_lbl, factory))
                return true;
            if (ReplaceLabelWithGoto(ite->ElseBranch(), target_lbl, factory))
                return true;
        }
        return false;
    }

    // Append a goto to the end of a branch.  If the branch is an SSeq,
    // append to it; otherwise wrap in a new SSeq.
    SNode *AppendGoto(SNode *branch, std::string_view label,
                             SNodeFactory &factory) {
        auto *g = factory.Make<SGoto>(factory.Intern(label));
        if (!branch) return g;
        if (auto *seq = branch->dyn_cast<SSeq>()) {
            seq->AddChild(g);
            return seq;
        }
        auto *seq = factory.Make<SSeq>();
        seq->AddChild(branch);
        seq->AddChild(g);
        return seq;
    }

    // Check if a node (or the last node in an SSeq chain) ends with
    // a control-flow transfer that prevents fallthrough.
    bool EndsWithTransfer(SNode *n) {
        if (!n) return false;
        // Peel through SSeq tails and SLabel wrappers.
        for (;;) {
            if (auto *s = n->dyn_cast<SSeq>()) {
                if (s->Size() == 0) return false;
                n = (*s)[s->Size() - 1];
            } else if (auto *l = n->dyn_cast<SLabel>()) {
                if (!l->Body()) return false;
                n = l->Body();
            } else {
                break;
            }
        }
        if (n->dyn_cast<SGoto>() || n->dyn_cast<SBreak>()
            || n->dyn_cast<SContinue>() || n->dyn_cast<SReturn>())
            return true;
        if (auto *ite = n->dyn_cast<SIfThenElse>()) {
            return EndsWithTransfer(ite->ThenBranch())
                && EndsWithTransfer(ite->ElseBranch());
        }
        return false;
    }

    void RefineHoistLabel(SNode *root, SNodeFactory &factory,
                          std::unordered_set<std::string> &hoisted_labels) {
        if (!root) return;
        hoisted_labels.clear();

        std::unordered_set<std::string> all_gotos;
        CollectGotoTargets(root, all_gotos);
        if (all_gotos.empty()) return;

        // Collect all SLabels in the tree that are cross-scope goto
        // targets, along with the SSeq+position where they should
        // be hoisted to.  We process the outermost occurrences first.
        //
        // Strategy: walk every SSeq.  For each child that is (or
        // contains) an SIfThenElse, check if either branch has a
        // label that is in all_gotos.  If so, hoist it to THIS SSeq
        // right after the IfThenElse, add skip-gotos, and restart.

        std::function<bool(SSeq *)> processSeq = [&](SSeq *seq) -> bool {
            for (size_t i = 0; i < seq->Size(); ++i) {
                auto *child = (*seq)[i];

                // Find IfThenElse (direct, or at tail of SSeq,
                // or inside SLabel wrapping).
                SIfThenElse *ite = nullptr;
                SNode *peeled = child;
                while (peeled) {
                    if (auto *d = peeled->dyn_cast<SIfThenElse>()) {
                        ite = d; break;
                    } else if (auto *is = peeled->dyn_cast<SSeq>()) {
                        if (is->Size() == 0) break;
                        peeled = (*is)[is->Size()-1];
                    } else if (auto *lb = peeled->dyn_cast<SLabel>()) {
                        peeled = lb->Body();
                    } else {
                        break;
                    }
                }
                if (!ite) continue;

                for (int bi = 0; bi < 2; ++bi) {
                    SNode *lbl_branch = bi == 0
                        ? ite->ElseBranch() : ite->ThenBranch();

                    SLabel *lbl = FindHoistableLabel(lbl_branch, all_gotos);
                    if (!lbl) continue;

                    std::string lname(lbl->Name());

                    // Only hoist if there's a goto to L from OUTSIDE
                    // the IfThenElse (from siblings in this SSeq).
                    std::unordered_set<std::string> external_gotos;
                    for (size_t j = 0; j < seq->Size(); ++j) {
                        if (j == i) continue;
                        CollectGotoTargets((*seq)[j], external_gotos);
                    }
                    if (!external_gotos.count(lname)) continue;
                    SNode *lbl_body = lbl->Body();

                    // 1. Replace label with SGoto(L) in the branch.
                    if (lbl_branch == lbl) {
                        auto *g = factory.Make<SGoto>(factory.Intern(lname));
                        if (bi == 0) ite->SetElseBranch(g);
                        else         ite->SetThenBranch(g);
                    } else {
                        ReplaceLabelWithGoto(lbl_branch, lbl, factory);
                    }

                    // 2. Insert hoisted label right after position i.
                    seq->InsertChild(i + 1,
                        factory.Make<SLabel>(factory.Intern(lname), lbl_body));
                    hoisted_labels.insert(lname);

                    // Refresh gotos and signal restart.
                    all_gotos.clear();
                    CollectGotoTargets(root, all_gotos);
                    return true;  // restart
                }
            }
            // Pattern 2: direct SLabel children of this SSeq that are
            // goto targets from outside (e.g. after an inner hoist put
            // a label here but a goto from an ancestor's other branch
            // still targets it).  Only fire if this SSeq is a direct
            // then/else-branch of a parent IfThenElse (the typical
            // case after Pattern 1 hoisting).
            auto *parent_ite = seq->Parent()
                ? seq->Parent()->dyn_cast<SIfThenElse>() : nullptr;
            if (!parent_ite) return false;
            for (size_t i = 0; i < seq->Size(); ++i) {
                auto *lbl = (*seq)[i]->dyn_cast<SLabel>();
                if (!lbl) continue;
                std::string lname(lbl->Name());
                if (!all_gotos.count(lname)) continue;

                // Only fire if the goto to L comes from the
                // OTHER branch of the immediate parent IfThenElse.
                std::unordered_set<std::string> other_gotos;
                if (parent_ite->ThenBranch() == seq)
                    CollectGotoTargets(parent_ite->ElseBranch(), other_gotos);
                else
                    CollectGotoTargets(parent_ite->ThenBranch(), other_gotos);
                if (!other_gotos.count(lname)) continue;

                // Find the nearest ancestor SSeq to hoist into.
                SSeq *target_seq = nullptr;
                size_t target_pos = 0;
                for (SNode *cur = seq, *p = seq->Parent();
                     p; cur = p, p = p->Parent()) {
                    if (auto *ps = p->dyn_cast<SSeq>()) {
                        for (size_t j = 0; j < ps->Size(); ++j) {
                            if ((*ps)[j] == cur) {
                                target_seq = ps;
                                target_pos = j;
                                break;
                            }
                        }
                        if (target_seq) break;
                    }
                }
                if (!target_seq) continue;

                SNode *lbl_body = lbl->Body();
                seq->RemoveChild(i);

                // Clean up stale after-labels left by a prior
                // inner hoist.  If SLabel("after_"+lname, empty_body)
                // exists in this SSeq, remove it — the outer hoist
                // will create a fresh one at the correct scope.
                std::string after = "after_" + lname;
                for (size_t j = 0; j < seq->Size(); ) {
                    auto *al = (*seq)[j]->dyn_cast<SLabel>();
                    if (al && al->Name() == after) {
                        // Check if its body is empty or just an empty block.
                        bool empty_body = !al->Body();
                        if (!empty_body) {
                            if (auto *bb = al->Body()->dyn_cast<SBlock>())
                                empty_body = bb->Stmts().empty();
                        }
                        if (empty_body) {
                            seq->RemoveChild(j);
                            continue;
                        }
                    }
                    ++j;
                }

                // Insert the hoisted label after the child that
                // contains our SSeq.
                target_seq->InsertChild(target_pos + 1,
                    factory.Make<SLabel>(factory.Intern(lname), lbl_body));
                hoisted_labels.insert(lname);

                all_gotos.clear();
                CollectGotoTargets(root, all_gotos);
                return true;
            }

            return false;
        };

        // Walk the tree top-down, processing SSeqs.  When a hoist
        // happens, restart from the root (the tree structure changed).
        std::function<bool(SNode *)> walk = [&](SNode *n) -> bool {
            if (!n) return false;
            if (auto *seq = n->dyn_cast<SSeq>()) {
                if (processSeq(seq)) return true;
                for (size_t k = 0; k < seq->Size(); ++k)
                    if (walk((*seq)[k])) return true;
            } else if (auto *ite = n->dyn_cast<SIfThenElse>()) {
                if (walk(ite->ThenBranch())) return true;
                if (walk(ite->ElseBranch())) return true;
            } else if (auto *w = n->dyn_cast<SWhile>()) {
                if (walk(w->Body())) return true;
            } else if (auto *dw = n->dyn_cast<SDoWhile>()) {
                if (walk(dw->Body())) return true;
            } else if (auto *f = n->dyn_cast<SFor>()) {
                if (walk(f->Body())) return true;
            } else if (auto *sw = n->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases())
                    if (walk(c.body)) return true;
                if (walk(sw->DefaultBody())) return true;
            } else if (auto *l = n->dyn_cast<SLabel>()) {
                if (walk(l->Body())) return true;
            }
            return false;
        };

        size_t max_rounds = 20;  // safety bound
        while (max_rounds-- > 0) {
            if (!walk(root)) break;
        }
    }

    // --- RefineFallthroughGoto ---
    //
    // Removes SGoto(L) when the next sibling in the same SSeq is
    // SLabel(L, ...) — the goto is redundant (fallthrough).
    // Also removes else=SGoto(L) from IfThenElse when the next
    // sibling is SLabel(L, ...).
    // --- RefineAddSkipGotos ---
    //
    // After RefineHoistLabel moves labels to outer scopes, some
    // IfThenElse nodes have branches that fall through to a hoisted
    // label.  This pass adds goto-skip + after-label pairs.
    //
    // Pattern: SSeq [..., IfThenElse(C, then, else), SLabel(L, body), ...]
    //   where then or else doesn't end with a transfer → would fall
    //   through to L (wrong if that branch shouldn't reach L).
    void RefineAddSkipGotos(SNode *node, SNodeFactory &factory,
                            const std::unordered_set<std::string> &hoisted_labels) {
        if (!node) return;
        // Recurse.
        if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ++i)
                RefineAddSkipGotos((*seq)[i], factory, hoisted_labels);
        } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            RefineAddSkipGotos(ite->ThenBranch(), factory, hoisted_labels);
            RefineAddSkipGotos(ite->ElseBranch(), factory, hoisted_labels);
            return;
        } else if (auto *w = node->dyn_cast<SWhile>()) {
            RefineAddSkipGotos(w->Body(), factory, hoisted_labels); return;
        } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            RefineAddSkipGotos(dw->Body(), factory, hoisted_labels); return;
        } else if (auto *f = node->dyn_cast<SFor>()) {
            RefineAddSkipGotos(f->Body(), factory, hoisted_labels); return;
        } else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases()) RefineAddSkipGotos(c.body, factory, hoisted_labels);
            RefineAddSkipGotos(sw->DefaultBody(), factory, hoisted_labels); return;
        } else if (auto *lbl = node->dyn_cast<SLabel>()) {
            RefineAddSkipGotos(lbl->Body(), factory, hoisted_labels); return;
        } else { return; }

        auto *seq = node->as<SSeq>();
        SNode *tree_root = seq;
        while (tree_root->Parent()) tree_root = tree_root->Parent();
        std::unordered_set<std::string> all_gotos;
        CollectGotoTargets(tree_root, all_gotos);

        for (size_t i = 0; i + 1 < seq->Size(); ++i) {
            auto *next_lbl = (*seq)[i + 1]->dyn_cast<SLabel>();
            if (!next_lbl) continue;
            std::string lname(next_lbl->Name());
            // Only add skip-gotos for labels that were hoisted by
            // RefineHoistLabel (not for normal program labels).
            if (!hoisted_labels.count(lname)) continue;
            if (!all_gotos.count(lname)) continue;
            if (EndsWithTransfer((*seq)[i])) continue;

            // child[i] falls through to goto-target label L.
            std::string after = "after_" + lname;

            // Append skip-goto into the deepest branch that falls
            // through to the label.  Peel through SSeq/SLabel/IfThenElse
            // to find the innermost point where code falls through.
            auto *cur = (*seq)[i];
            SNode *target = cur;
            while (target) {
                if (auto *ts = target->dyn_cast<SSeq>()) {
                    if (ts->Size() == 0) break;
                    target = (*ts)[ts->Size() - 1];
                } else if (auto *tl = target->dyn_cast<SLabel>()) {
                    target = tl->Body();
                } else if (auto *ite = target->dyn_cast<SIfThenElse>()) {
                    // Descend into the non-transferring branch
                    // ONLY if exactly one branch transfers.
                    bool then_t = EndsWithTransfer(ite->ThenBranch());
                    bool else_t = EndsWithTransfer(ite->ElseBranch());
                    if (else_t && !then_t) {
                        target = ite->ThenBranch();
                    } else if (then_t && !else_t) {
                        target = ite->ElseBranch();
                    } else {
                        // Neither or both transfer — append the goto
                        // AFTER this IfThenElse in its parent SSeq.
                        auto *parent_seq = ite->Parent()
                            ? ite->Parent()->dyn_cast<SSeq>() : nullptr;
                        if (parent_seq) {
                            // Find ite's position and append after it.
                            for (size_t k = 0; k < parent_seq->Size(); ++k) {
                                if ((*parent_seq)[k] == ite) {
                                    parent_seq->InsertChild(k + 1,
                                        factory.Make<SGoto>(
                                            factory.Intern(after)));
                                    break;
                                }
                            }
                        }
                        target = nullptr;
                        break;
                    }
                } else {
                    break;
                }
            }
            if (auto *ite = target ? target->dyn_cast<SIfThenElse>()
                                   : nullptr) {
                if (!EndsWithTransfer(ite->ThenBranch()))
                    ite->SetThenBranch(
                        AppendGoto(ite->ThenBranch(), after, factory));
                if (!EndsWithTransfer(ite->ElseBranch()))
                    ite->SetElseBranch(
                        AppendGoto(ite->ElseBranch(), after, factory));
            } else if (auto *is = target
                    ? target->dyn_cast<SSeq>() : nullptr) {
                is->AddChild(factory.Make<SGoto>(factory.Intern(after)));
            } else if (auto *is = cur->dyn_cast<SSeq>()) {
                is->AddChild(factory.Make<SGoto>(factory.Intern(after)));
            } else {
                auto *wrapper = factory.Make<SSeq>();
                wrapper->AddChild(cur);
                wrapper->AddChild(factory.Make<SGoto>(factory.Intern(after)));
                seq->ReplaceChild(i, wrapper);
            }

            // Place after-label: find the first ancestor SSeq that
            // has a continuation sibling after the label L.
            SSeq *after_seq = seq;
            size_t lbl_pos = i + 1;  // position of L in after_seq
            while (lbl_pos + 1 >= after_seq->Size()) {
                // L is the last child — walk up.
                SNode *ancestor = after_seq;
                SSeq *parent_seq = nullptr;
                size_t ancestor_pos = 0;
                for (SNode *p = ancestor->Parent(); p; p = p->Parent()) {
                    parent_seq = p->dyn_cast<SSeq>();
                    if (parent_seq) {
                        for (size_t j = 0; j < parent_seq->Size(); ++j) {
                            SNode *w = ancestor;
                            while (w && w->Parent() != parent_seq)
                                w = w->Parent();
                            if (w == (*parent_seq)[j]) {
                                ancestor_pos = j;
                                goto found_parent;
                            }
                        }
                        parent_seq = nullptr;
                    }
                }
                found_parent:
                if (!parent_seq) break;
                after_seq = parent_seq;
                lbl_pos = ancestor_pos;
            }

            if (lbl_pos + 1 < after_seq->Size()) {
                auto *c = (*after_seq)[lbl_pos + 1];
                auto *ex = c->dyn_cast<SLabel>();
                if (!ex || ex->Name() != after)
                    after_seq->ReplaceChild(lbl_pos + 1,
                        factory.Make<SLabel>(factory.Intern(after), c));
            } else {
                after_seq->AddChild(factory.Make<SLabel>(
                    factory.Intern(after), factory.Make<SBlock>()));
            }
            i += 2;
        }
    }

    // --- RefineGotoToDoWhile ---
    //
    // Detects backward gotos in an SSeq that form a loop:
    //   SSeq [..., SLabel(A, body_A), ...mid..., SGoto(A), ...]
    // or with the goto inside an IfThenElse tail:
    //   SSeq [..., SLabel(A, body_A), ...mid...,
    //          IfThenElse(cond, exit, SGoto(A)), ...]
    //
    // Transforms to do-while:
    //   SSeq [..., SDoWhile(SSeq[SLabel(A, body_A), ...mid...], !cond),
    //          exit, ...]
    void RefineGotoToDoWhile(SNode *node, SNodeFactory &factory,
                             clang::ASTContext &ctx) {
        if (!node) return;
        // Recurse bottom-up.
        if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ++i)
                RefineGotoToDoWhile((*seq)[i], factory, ctx);
        } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            RefineGotoToDoWhile(ite->ThenBranch(), factory, ctx);
            RefineGotoToDoWhile(ite->ElseBranch(), factory, ctx);
            return;
        } else if (auto *w = node->dyn_cast<SWhile>()) {
            RefineGotoToDoWhile(w->Body(), factory, ctx); return;
        } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            RefineGotoToDoWhile(dw->Body(), factory, ctx); return;
        } else if (auto *f = node->dyn_cast<SFor>()) {
            RefineGotoToDoWhile(f->Body(), factory, ctx); return;
        } else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases()) RefineGotoToDoWhile(c.body, factory, ctx);
            RefineGotoToDoWhile(sw->DefaultBody(), factory, ctx); return;
        } else if (auto *lbl = node->dyn_cast<SLabel>()) {
            RefineGotoToDoWhile(lbl->Body(), factory, ctx); return;
        } else { return; }

        auto *seq = node->as<SSeq>();

        // Scan from the end for backward gotos.
        for (size_t j = 1; j < seq->Size(); ++j) {
            // Find a backward goto: either a direct SGoto at position j,
            // or an IfThenElse at position j with SGoto in one branch.
            std::string_view back_target;
            clang::Expr *exit_cond = nullptr;
            SNode *exit_body = nullptr;
            bool goto_in_else = false;  // which branch has the goto

            auto *child_j = (*seq)[j];

            // Scan child_j's tree for an IfThenElse with a backward
            // goto.  Peel through SLabel/SSeq layers.  When found
            // mid-SSeq, record the position for splitting.
            SSeq *back_seq = nullptr;    // the SSeq containing the back-edge node
            size_t back_pos = 0;         // position in back_seq

            // Recursive search for IfThenElse with SGoto in a branch.
            std::function<bool(SNode *)> findBackEdge = [&](SNode *n) -> bool {
                if (!n) return false;
                if (auto *ite = n->dyn_cast<SIfThenElse>()) {
                    auto *eg = ite->ElseBranch()
                        ? ite->ElseBranch()->dyn_cast<SGoto>() : nullptr;
                    auto *tg = ite->ThenBranch()
                        ? ite->ThenBranch()->dyn_cast<SGoto>() : nullptr;
                    if (eg) {
                        back_target = eg->Target();
                        exit_cond = ite->Cond();
                        exit_body = ite->ThenBranch();
                        goto_in_else = true;
                        return true;
                    }
                    if (tg) {
                        back_target = tg->Target();
                        exit_cond = ite->Cond();
                        exit_body = ite->ElseBranch();
                        goto_in_else = false;
                        return true;
                    }
                }
                if (auto *g = n->dyn_cast<SGoto>()) {
                    back_target = g->Target();
                    exit_cond = nullptr;
                    exit_body = nullptr;
                    return true;
                }
                if (auto *s = n->dyn_cast<SSeq>()) {
                    for (size_t k = 0; k < s->Size(); ++k) {
                        if (findBackEdge((*s)[k])) {
                            back_seq = s;
                            back_pos = k;
                            return true;
                        }
                    }
                }
                if (auto *lbl = n->dyn_cast<SLabel>()) {
                    return findBackEdge(lbl->Body());
                }
                return false;
            };

            if (!findBackEdge(child_j)) continue;

            if (back_target.empty()) continue;

            // Find the target label at position i < j in this SSeq.
            // Search direct children AND children nested in SSeq/SLabel.
            auto findLabelInChild = [](SNode *child,
                                       std::string_view target) -> bool {
                SNode *n = child;
                while (n) {
                    if (auto *lbl = n->dyn_cast<SLabel>())
                        if (lbl->Name() == target) return true;
                    if (auto *s = n->dyn_cast<SSeq>()) {
                        // Check all children of the SSeq.
                        for (size_t k = 0; k < s->Size(); ++k) {
                            if (auto *lbl = (*s)[k]->dyn_cast<SLabel>())
                                if (lbl->Name() == target) return true;
                        }
                        if (s->Size() > 0) { n = (*s)[0]; continue; }
                    }
                    break;
                }
                return false;
            };
            size_t label_pos = seq->Size();  // sentinel
            for (size_t i = 0; i < j; ++i) {
                if (findLabelInChild((*seq)[i], back_target)) {
                    label_pos = i;
                    break;
                }
            }
            if (label_pos >= j) continue;  // not a backward goto

            // Found a backward goto from position j to label at i.

            // If the back-edge is mid-SSeq, split: elements after
            // back_pos are the exit path, not the loop body.
            SNode *extra_exit = nullptr;
            if (back_seq && back_pos + 1 < back_seq->Size()) {
                auto *exit_seq = factory.Make<SSeq>();
                for (size_t k = back_pos + 1; k < back_seq->Size(); ++k)
                    exit_seq->AddChild((*back_seq)[k]);
                // Remove the exit elements from back_seq.
                for (size_t k = back_seq->Size() - 1;
                     k > back_pos; --k)
                    back_seq->RemoveChild(k);
                extra_exit = exit_seq;
            }

            // Remove the back-edge node (goto/IfThenElse) from
            // wherever it sits.
            if (back_seq) {
                back_seq->RemoveChild(back_pos);
            }

            // Build the loop body: children [label_pos..j].
            auto *loop_body = factory.Make<SSeq>();
            for (size_t k = label_pos; k <= j; ++k)
                loop_body->AddChild((*seq)[k]);

            // Build the loop condition.
            SNode *loop_node;
            if (exit_cond) {
                // Negate condition if the goto was in the else-branch:
                //   if(cond) exit; else goto A;  → do{} while(!cond)
                //   if(cond) goto A; else exit;  → do{} while(cond)
                clang::Expr *while_cond;
                if (goto_in_else) {
                    // goto in else: loop continues when !cond
                    while_cond = NegateCond(exit_cond, ctx);
                } else {
                    // goto in then: loop continues when cond
                    while_cond = exit_cond;
                }
                loop_node = factory.Make<SDoWhile>(loop_body, while_cond);
            } else {
                // Unconditional back-edge: while(1) { body }
                auto *true_lit = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(32, 1), ctx.IntTy,
                    clang::SourceLocation());
                loop_node = factory.Make<SWhile>(true_lit, loop_body);
            }

            // Replace children [label_pos..j] with the loop node +
            // exit body (if any) + extra exit (if mid-SSeq split).
            std::vector<SNode *> replacements;
            replacements.push_back(loop_node);
            if (exit_body) replacements.push_back(exit_body);
            if (extra_exit) replacements.push_back(extra_exit);

            seq->ReplaceRange(label_pos, j + 1, replacements);
            break;  // restart after modification
        }
    }

    // --- RefineGotoEndToBreak ---
    //
    // Detect goto L where L is an empty label at the end of the
    // outermost SSeq (function end).  When the goto is inside a
    // loop, convert it to break.  When it's not in a loop, the
    // goto is equivalent to "skip to function end" — wrap the
    // skipped code in if(!cond) { ... } (negate-and-wrap).
    //
    // This handles the encode_basic_field pattern:
    //   do { ...; if (ok) goto END; error_handling; } while(c);
    //   post_loop; END: ;
    // → do { ...; if (ok) break; error_handling; } while(c);
    //   post_loop;
    void RefineGotoEndToBreak(SNode *root, SNodeFactory &factory) {
        if (!root) return;
        auto *root_seq = root->dyn_cast<SSeq>();
        if (!root_seq || root_seq->Size() == 0) return;

        // Collect all goto targets and all label definitions.
        // Gotos whose targets have NO SLabel in the tree are
        // "orphan" targets — the label exists only in the Clang AST
        // as a bare label at the function exit.
        std::unordered_set<std::string> all_gotos_set;
        CollectGotoTargets(root, all_gotos_set);
        std::unordered_set<std::string> all_labels_set;
        CollectLabels(root, all_labels_set);

        // Also check for an empty SLabel at the root SSeq's end.
        if (root_seq->Size() > 0) {
            auto *last = (*root_seq)[root_seq->Size() - 1];
            if (auto *lbl = last->dyn_cast<SLabel>()) {
                bool is_empty = !lbl->Body();
                if (!is_empty) {
                    if (auto *blk = lbl->Body()->dyn_cast<SBlock>())
                        is_empty = blk->Stmts().empty();
                }
                if (is_empty) all_labels_set.erase(std::string(lbl->Name()));
            }
        }

        // Find orphan goto targets: in all_gotos but not in all_labels
        // (their SLabel doesn't exist in the SNode tree), or targets
        // of empty end-of-function labels.
        std::unordered_set<std::string> end_targets;
        for (auto &g : all_gotos_set) {
            if (!all_labels_set.count(g))
                end_targets.insert(g);
        }
        if (end_targets.empty()) return;

        // Walk the tree and replace SGoto(END) with SBreak when
        // inside a loop.  Track loop depth.
        auto isEndTarget = [&](std::string_view name) -> bool {
            return end_targets.count(std::string(name)) > 0;
        };

        std::function<bool(SNode *, bool)> walk =
            [&](SNode *node, bool in_loop) -> bool {
            if (!node) return false;
            bool changed = false;

            if (auto *seq = node->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < seq->Size(); ++i) {
                    auto *child = (*seq)[i];
                    if (auto *g = child->dyn_cast<SGoto>()) {
                        if (isEndTarget(g->Target()) && in_loop) {
                            seq->ReplaceChild(i, factory.Make<SBreak>());
                            changed = true;
                            continue;
                        }
                    }
                    if (auto *ite = child->dyn_cast<SIfThenElse>()) {
                        if (auto *eg = ite->ElseBranch()
                                ? ite->ElseBranch()->dyn_cast<SGoto>()
                                : nullptr) {
                            if (isEndTarget(eg->Target()) && in_loop) {
                                ite->SetElseBranch(factory.Make<SBreak>());
                                changed = true;
                            }
                        }
                        if (auto *tg = ite->ThenBranch()
                                ? ite->ThenBranch()->dyn_cast<SGoto>()
                                : nullptr) {
                            if (isEndTarget(tg->Target()) && in_loop) {
                                ite->SetThenBranch(factory.Make<SBreak>());
                                changed = true;
                            }
                        }
                    }
                    if (walk(child, in_loop)) changed = true;
                }
            } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                if (walk(ite->ThenBranch(), in_loop)) changed = true;
                if (walk(ite->ElseBranch(), in_loop)) changed = true;
            } else if (auto *w = node->dyn_cast<SWhile>()) {
                if (walk(w->Body(), true)) changed = true;
            } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
                if (walk(dw->Body(), true)) changed = true;
            } else if (auto *f = node->dyn_cast<SFor>()) {
                if (walk(f->Body(), true)) changed = true;
            } else if (auto *sw = node->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases())
                    if (walk(c.body, in_loop)) changed = true;
                if (walk(sw->DefaultBody(), in_loop)) changed = true;
            } else if (auto *lbl = node->dyn_cast<SLabel>()) {
                if (walk(lbl->Body(), in_loop)) changed = true;
            }
            return changed;
        };

        if (walk(root, false)) {
            // If any goto was converted to break, the END label
            // may now be dead.  RefineDeadLabels will clean it up.
        }
    }

    // Check if natural fallthrough from |node|'s position in its
    // parent chain reaches SLabel(target).  Used to detect redundant
    // gotos that jump to where execution would end up anyway.
    //
    // Walk up from node.  At each SSeq ancestor, check if node's
    // subtree is the LAST child.  If yes, execution falls out of
    // this SSeq — continue up.  If no, check if the NEXT sibling
    // starts with SLabel(target).  Also peel through IfThenElse
    // when node is in the else-branch (falls out of the if).
    bool FallsThroughToLabel(SNode *node, std::string_view target) {
        SNode *cur = node;
        for (SNode *p = cur->Parent(); p; cur = p, p = p->Parent()) {
            if (auto *seq = p->dyn_cast<SSeq>()) {
                // Find cur's position in the SSeq.
                for (size_t j = 0; j < seq->Size(); ++j) {
                    if ((*seq)[j] == cur) {
                        if (j + 1 < seq->Size()) {
                            // Check if next sibling starts with the label.
                            auto *next = (*seq)[j + 1];
                            if (auto *lbl = next->dyn_cast<SLabel>())
                                if (lbl->Name() == target) return true;
                            // Not a match — execution would reach
                            // the next sibling, not the target label.
                            return false;
                        }
                        // cur is the last child — fall out of SSeq,
                        // continue walking up.
                        break;
                    }
                }
            } else if (p->dyn_cast<SIfThenElse>()) {
                // Only fall through if cur is a branch and the
                // IfThenElse itself is at a tail position.
                // Continue walking up from the IfThenElse.
            } else if (p->dyn_cast<SLabel>()) {
                // Continue — SLabel wrapper is transparent.
            }
            // For other parents (While, For, etc.), fallthrough
            // doesn't reach outside the loop — stop.
            else {
                return false;
            }
        }
        return false;
    }

    // Remove SGoto(L) when natural fallthrough reaches L.
    // Walks the entire tree, finds SGoto nodes at branch tails,
    // and checks if they're redundant via FallsThroughToLabel.
    // --- RefineGotoSkipTrailing ---
    //
    // In an SSeq, when a child IfThenElse (possibly nested) has an
    // SGoto(L) in one branch that skips trailing code in the same
    // SSeq, move the trailing code into the non-goto branches.
    // This turns the goto into a trivial fallthrough.
    //
    // Pattern:
    //   SSeq [..., IfThenElse(C, body, SGoto(L)), trailing, ...]
    //   where L is at a higher scope
    //
    // Transform:
    //   SSeq [..., IfThenElse(C, SSeq[body, trailing], SGoto(L)), ...]
    //
    // Also handles deeply nested gotos by recursing into branches.
    void RefineGotoSkipTrailing(SNode *node, SNodeFactory &factory) {
        if (!node) return;
        // Recurse bottom-up.
        if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ++i)
                RefineGotoSkipTrailing((*seq)[i], factory);
        } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            RefineGotoSkipTrailing(ite->ThenBranch(), factory);
            RefineGotoSkipTrailing(ite->ElseBranch(), factory);
            return;
        } else if (auto *w = node->dyn_cast<SWhile>()) {
            RefineGotoSkipTrailing(w->Body(), factory); return;
        } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            RefineGotoSkipTrailing(dw->Body(), factory); return;
        } else if (auto *f = node->dyn_cast<SFor>()) {
            RefineGotoSkipTrailing(f->Body(), factory); return;
        } else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases()) RefineGotoSkipTrailing(c.body, factory);
            RefineGotoSkipTrailing(sw->DefaultBody(), factory); return;
        } else if (auto *lbl = node->dyn_cast<SLabel>()) {
            RefineGotoSkipTrailing(lbl->Body(), factory); return;
        } else { return; }

        auto *seq = node->as<SSeq>();
        // Collect all goto targets in the entire tree.
        SNode *root = seq;
        while (root->Parent()) root = root->Parent();
        std::unordered_set<std::string> all_gotos;
        CollectGotoTargets(root, all_gotos);

        for (size_t i = 0; i + 1 < seq->Size(); ++i) {
            auto *child = (*seq)[i];

            // Find an IfThenElse (possibly nested via SSeq/SLabel
            // peeling) where one branch is SGoto(L) and L is NOT
            // in this SSeq (it's at a higher scope).
            // Use the deep peel from EndsWithTransfer logic.
            SNode *tail = child;
            while (tail) {
                if (tail->dyn_cast<SIfThenElse>()) break;
                if (auto *s = tail->dyn_cast<SSeq>()) {
                    if (s->Size() == 0) { tail = nullptr; break; }
                    tail = (*s)[s->Size() - 1];
                } else if (auto *l = tail->dyn_cast<SLabel>()) {
                    tail = l->Body();
                } else { tail = nullptr; break; }
            }
            auto *ite = tail ? tail->dyn_cast<SIfThenElse>() : nullptr;
            if (!ite) continue;

            // Check if either branch (or a deeply nested branch)
            // ends with SGoto(L) targeting a higher scope.
            // Peel through SLabel/SSeq/IfThenElse to find SGoto at tail.
            std::function<SGoto *(SNode *)> findTailGoto = [&](SNode *n) -> SGoto * {
                while (n) {
                    if (auto *g = n->dyn_cast<SGoto>()) return g;
                    if (auto *s = n->dyn_cast<SSeq>()) {
                        if (s->Size() == 0) return nullptr;
                        n = (*s)[s->Size() - 1];
                    } else if (auto *l = n->dyn_cast<SLabel>()) {
                        n = l->Body();
                    } else if (auto *inner = n->dyn_cast<SIfThenElse>()) {
                        // Check if ONE branch is a goto (try else first).
                        if (auto *g = inner->ElseBranch()
                                ? inner->ElseBranch()->dyn_cast<SGoto>()
                                : nullptr) return g;
                        if (auto *g = inner->ThenBranch()
                                ? inner->ThenBranch()->dyn_cast<SGoto>()
                                : nullptr) return g;
                        // Recurse into the branch that might have a deeper goto.
                        auto *eg = findTailGoto(inner->ElseBranch());
                        if (eg) return eg;
                        return findTailGoto(inner->ThenBranch());
                    } else {
                        return nullptr;
                    }
                }
                return nullptr;
            };

            // Search both branches of the top-level IfThenElse.
            for (int bi = 0; bi < 2; ++bi) {
                SNode *goto_branch = (bi == 0)
                    ? ite->ElseBranch() : ite->ThenBranch();

                auto *sg = findTailGoto(goto_branch);
                if (!sg) continue;

                // Check L is NOT a label in this SSeq (higher scope).
                std::string target(sg->Target());
                // Check L is a real goto target (not dead).
                if (!all_gotos.count(target)) continue;

                // Find where the trailing code ends.  If L is in
                // this SSeq, trailing code is [i+1..label_pos).
                // If L is at a higher scope, trailing is [i+1..end).
                size_t label_pos = seq->Size();  // sentinel: not in seq
                for (size_t j = i + 1; j < seq->Size(); ++j) {
                    if (auto *lbl = (*seq)[j]->dyn_cast<SLabel>())
                        if (lbl->Name() == target) { label_pos = j; break; }
                }

                // If label is the immediate next sibling, it's a
                // trivial fallthrough (handled by RefineRedundantGoto).
                if (label_pos == i + 1) continue;

                // Collect trailing code: [i+1..label_pos) if label is
                // in seq, or [i+1..end) if label is at higher scope.
                size_t trail_end = (label_pos < seq->Size())
                    ? label_pos : seq->Size();
                // Must have at least one trailing element.
                if (trail_end <= i + 1) continue;

                std::vector<SNode *> trailing;
                for (size_t j = i + 1; j < trail_end; ++j)
                    trailing.push_back((*seq)[j]);

                // Recursive function: append trailing code (as a copy
                // reference — shallow, OK since SNode trees share)
                // to every branch that doesn't end with SGoto(target).
                // Returns true if trailing was placed in at least one branch.
                bool distributed = false;
                std::function<void(SIfThenElse *)> distributeTrailing =
                    [&](SIfThenElse *ife) {
                    for (int b = 0; b < 2; ++b) {
                        SNode *br = (b == 0)
                            ? ife->ThenBranch() : ife->ElseBranch();
                        if (!br) continue;
                        // If this branch IS the SGoto, skip it.
                        if (br->dyn_cast<SGoto>()) continue;
                        // If this branch ends with SGoto via peeling,
                        // check for nested IfThenElse and recurse.
                        if (EndsWithTransfer(br)) {
                            // Find the innermost IfThenElse that contains
                            // the goto and recurse into it.
                            SNode *inner = br;
                            while (inner) {
                                if (auto *s = inner->dyn_cast<SSeq>()) {
                                    if (s->Size() == 0) break;
                                    inner = (*s)[s->Size() - 1];
                                } else if (auto *l = inner->dyn_cast<SLabel>()) {
                                    inner = l->Body();
                                } else break;
                            }
                            if (auto *inner_ite = inner
                                    ? inner->dyn_cast<SIfThenElse>()
                                    : nullptr) {
                                distributeTrailing(inner_ite);
                            }
                            continue;
                        }
                        // This branch doesn't end with a transfer.
                        // Append trailing code to it.
                        auto *new_br = factory.Make<SSeq>();
                        new_br->AddChild(br);
                        for (auto *t : trailing)
                            new_br->AddChild(t);
                        if (b == 0)
                            ife->SetThenBranch(new_br);
                        else
                            ife->SetElseBranch(new_br);
                        distributed = true;
                    }
                };

                distributeTrailing(ite);

                // Only remove trailing code if it was placed somewhere.
                // Otherwise keep it in the seq to avoid losing stmts.
                if (!distributed) continue;

                // Remove the trailing code we distributed [i+1..trail_end).
                for (size_t j = trail_end - 1; j > i; --j)
                    seq->RemoveChild(j);

                break;  // Modified — stop scanning this SSeq.
            }
        }
    }

    void RefineRedundantGoto(SNode *node) {
        if (!node) return;

        if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            // Check else-branch: if it's SGoto(L) and falling through
            // from the IfThenElse reaches L, remove it.
            if (auto *eg = ite->ElseBranch()
                    ? ite->ElseBranch()->dyn_cast<SGoto>() : nullptr) {
                if (FallsThroughToLabel(ite, eg->Target()))
                    ite->SetElseBranch(nullptr);
            }
            // Check then-branch similarly.
            if (auto *tg = ite->ThenBranch()
                    ? ite->ThenBranch()->dyn_cast<SGoto>() : nullptr) {
                if (FallsThroughToLabel(ite, tg->Target()))
                    ite->SetThenBranch(nullptr);
            }
            // Recurse into remaining branches.
            RefineRedundantGoto(ite->ThenBranch());
            RefineRedundantGoto(ite->ElseBranch());
        }
        // For branches that end with SGoto at the tail of an SSeq,
        // check the SSeq's last child.
        else if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ++i)
                RefineRedundantGoto((*seq)[i]);
            // Remove SGoto(L) immediately followed by SLabel(L, ...)
            // in this SSeq (goto to next statement is redundant).
            for (size_t i = 0; i + 1 < seq->Size(); ) {
                auto *g = (*seq)[i]->dyn_cast<SGoto>();
                if (g) {
                    auto *next_lbl = (*seq)[i + 1]->dyn_cast<SLabel>();
                    if (next_lbl && next_lbl->Name() == g->Target()) {
                        seq->RemoveChild(i);
                        continue;  // re-check same index
                    }
                }
                // Pattern: if(...) {...} else goto L; L: ...
                // Also handles nested: if() { if() {} else goto L } L:
                // Peel through trailing else branches to find the deepest
                // goto that falls through to the next SLabel sibling.
                if (auto *next_lbl = (*seq)[i + 1]->dyn_cast<SLabel>()) {
                    // Walk into trailing else/then branches of IfThenElse
                    SNode *tail = (*seq)[i];
                    while (auto *ite = tail->dyn_cast<SIfThenElse>()) {
                        // Check else branch
                        if (auto *eg = ite->ElseBranch()
                                ? ite->ElseBranch()->dyn_cast<SGoto>() : nullptr) {
                            if (next_lbl->Name() == eg->Target()) {
                                ite->SetElseBranch(nullptr);
                                break;
                            }
                        }
                        // Check then branch
                        if (auto *tg = ite->ThenBranch()
                                ? ite->ThenBranch()->dyn_cast<SGoto>() : nullptr) {
                            if (next_lbl->Name() == tg->Target()) {
                                ite->SetThenBranch(nullptr);
                                break;
                            }
                        }
                        // Peel: the else branch might contain another
                        // IfThenElse whose trailing goto falls through.
                        if (ite->ElseBranch())
                            tail = ite->ElseBranch();
                        else if (ite->ThenBranch())
                            tail = ite->ThenBranch();
                        else
                            break;
                    }
                    // Also check SSeq tail containing IfThenElse
                    if (auto *inner_seq = tail->dyn_cast<SSeq>()) {
                        if (inner_seq->Size() > 0) {
                            auto *last_child = (*inner_seq)[inner_seq->Size() - 1];
                            if (auto *ite = last_child->dyn_cast<SIfThenElse>()) {
                                if (auto *eg = ite->ElseBranch()
                                        ? ite->ElseBranch()->dyn_cast<SGoto>() : nullptr) {
                                    if (next_lbl->Name() == eg->Target())
                                        ite->SetElseBranch(nullptr);
                                }
                            }
                        }
                    }
                }
                // Also strip Clang GotoStmt at the end of an SBlock
                // when the next sibling is SLabel with the same target.
                if (i + 1 < seq->Size()) {
                    if (auto *blk = (*seq)[i]->dyn_cast<SBlock>()) {
                        if (!blk->Stmts().empty()) {
                            if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(
                                    blk->Stmts().back())) {
                                std::string tgt;
                                if (auto *ii = gs->getLabel()->getIdentifier())
                                    tgt = ii->getName().str();
                                auto *next_lbl = (*seq)[i + 1]->dyn_cast<SLabel>();
                                if (next_lbl && next_lbl->Name() == tgt) {
                                    blk->Stmts().pop_back();
                                }
                            }
                        }
                    }
                }
                ++i;
            }
            // Check if the last child is an SGoto that falls through
            // to a label in a parent SSeq.
            if (seq->Size() > 0) {
                size_t last = seq->Size() - 1;
                if (auto *g = (*seq)[last]->dyn_cast<SGoto>()) {
                    if (FallsThroughToLabel(seq, g->Target()))
                        seq->RemoveChild(last);
                }
            }
        }
        else if (auto *w = node->dyn_cast<SWhile>())
            RefineRedundantGoto(w->Body());
        else if (auto *dw = node->dyn_cast<SDoWhile>())
            RefineRedundantGoto(dw->Body());
        else if (auto *f = node->dyn_cast<SFor>())
            RefineRedundantGoto(f->Body());
        else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases()) RefineRedundantGoto(c.body);
            RefineRedundantGoto(sw->DefaultBody());
        }
        else if (auto *lbl = node->dyn_cast<SLabel>())
            RefineRedundantGoto(lbl->Body());
    }

    // --- RefineSwitchCaseInline ---
    //
    // Finds SBlock nodes containing a clang::SwitchStmt where case
    // bodies are clang::GotoStmt (goto label).  For each such case,
    // finds the SLabel sibling in the parent SSeq, collects the
    // label's body stmts (following fall-through chains), and builds
    // a new case body with inlined content + break/return.
    void RefineSwitchCaseInline(SNode *node, SNodeFactory &factory,
                                clang::ASTContext &ctx) {
        if (!node) return;

        // Recurse.
        if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ++i)
                RefineSwitchCaseInline((*seq)[i], factory, ctx);
        } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            RefineSwitchCaseInline(ite->ThenBranch(), factory, ctx);
            RefineSwitchCaseInline(ite->ElseBranch(), factory, ctx);
            return;
        } else if (auto *w = node->dyn_cast<SWhile>()) {
            RefineSwitchCaseInline(w->Body(), factory, ctx); return;
        } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            RefineSwitchCaseInline(dw->Body(), factory, ctx); return;
        } else if (auto *f = node->dyn_cast<SFor>()) {
            RefineSwitchCaseInline(f->Body(), factory, ctx); return;
        } else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases()) RefineSwitchCaseInline(c.body, factory, ctx);
            RefineSwitchCaseInline(sw->DefaultBody(), factory, ctx); return;
        } else if (auto *lbl = node->dyn_cast<SLabel>()) {
            RefineSwitchCaseInline(lbl->Body(), factory, ctx); return;
        } else { return; }

        auto *seq = node->as<SSeq>();

        // Build a map: label_name → position in this SSeq.
        std::unordered_map<std::string, size_t> label_map;
        for (size_t i = 0; i < seq->Size(); ++i) {
            if (auto *lbl = (*seq)[i]->dyn_cast<SLabel>())
                label_map[std::string(lbl->Name())] = i;
        }
        if (label_map.empty()) return;

        // Scan children for clang::SwitchStmt with goto cases.
        // Peel through SLabel wrappers to find the SBlock.
        for (size_t i = 0; i < seq->Size(); ++i) {
            SNode *child = (*seq)[i];
            // Peel through SLabel to find SBlock.
            while (auto *lbl = child->dyn_cast<SLabel>()) {
                if (lbl->Body()) child = lbl->Body();
                else break;
            }
            // Also peel SSeq → first Block child.
            if (auto *inner = child->dyn_cast<SSeq>()) {
                for (size_t k = 0; k < inner->Size(); ++k) {
                    if (auto *b = (*inner)[k]->dyn_cast<SBlock>()) {
                        child = b; break;
                    }
                }
            }
            auto *blk = child->dyn_cast<SBlock>();
            if (!blk) {
                continue;
            }

            for (auto *stmt : blk->Stmts()) {
                auto *sw = llvm::dyn_cast<clang::SwitchStmt>(stmt);
                if (!sw) continue;

                // Found a SwitchStmt. Iterate through body stmts
                // to find CaseStmt nodes (getSwitchCaseList may be
                // null if addSwitchCase wasn't called by CfgBuilder).
                auto *sw_body = sw->getBody();
                if (!sw_body) continue;

                // Collect all CaseStmt nodes from the body.
                std::vector<clang::CaseStmt *> cases;
                for (auto *child_stmt : sw_body->children()) {
                    if (auto *cs = llvm::dyn_cast<clang::CaseStmt>(child_stmt))
                        cases.push_back(cs);
                }

                bool has_goto_case = false;
                for (auto *cs : cases) {
                    auto *sub = cs->getSubStmt();
                    if (auto *comp = llvm::dyn_cast<clang::CompoundStmt>(sub)) {
                        if (comp->size() > 0)
                            sub = *comp->body_begin();
                    }
                    if (llvm::isa<clang::GotoStmt>(sub))
                        has_goto_case = true;
                }
                if (!has_goto_case) continue;

                // Collect the stmts for each label by following
                // fall-through chains.  Given label at position p,
                // its body is: SLabel(p).body stmts + SLabel(p+1).body
                // stmts + ... until a return/break is found.
                // Collect stmts from an SNode (SBlock or SLabel body).
                auto collectStmts = [](SNode *n,
                        std::vector<clang::Stmt *> &out) {
                    if (!n) return;
                    if (auto *blk = n->dyn_cast<SBlock>()) {
                        for (auto *s : blk->Stmts()) out.push_back(s);
                    } else if (auto *lbl = n->dyn_cast<SLabel>()) {
                        if (auto *blk = lbl->Body()
                                ? lbl->Body()->dyn_cast<SBlock>()
                                : nullptr) {
                            for (auto *s : blk->Stmts()) out.push_back(s);
                        }
                    }
                };

                // Follow the fall-through chain from a label position.
                // Collects stmts from consecutive SLabel and SBlock
                // siblings until a return/break is found.
                auto collectChain = [&](const std::string &target)
                    -> std::vector<clang::Stmt *> {
                    std::vector<clang::Stmt *> result;
                    auto it = label_map.find(target);
                    if (it == label_map.end()) return result;

                    for (size_t pos = it->second;
                         pos < seq->Size(); ++pos) {
                        auto *child = (*seq)[pos];

                        // Collect stmts from SLabel body or SBlock.
                        if (auto *lbl = child->dyn_cast<SLabel>()) {
                            collectStmts(lbl->Body(), result);
                        } else if (auto *blk = child->dyn_cast<SBlock>()) {
                            collectStmts(blk, result);
                        } else if (pos > it->second) {
                            break;  // non-label non-block: stop chain
                        } else {
                            continue;
                        }

                        // Check if chain ends with return/break.
                        if (!result.empty()) {
                            auto *last = result.back();
                            if (llvm::isa<clang::ReturnStmt>(last) ||
                                llvm::isa<clang::BreakStmt>(last))
                                break;
                        }
                    }
                    return result;
                };

                // Rebuild each case body by inlining the goto target.
                for (auto *cs : cases) {
                    auto *sub = cs->getSubStmt();
                    if (auto *comp = llvm::dyn_cast<clang::CompoundStmt>(sub)) {
                        if (comp->size() > 0)
                            sub = *comp->body_begin();
                    }
                    auto *gs = llvm::dyn_cast<clang::GotoStmt>(sub);
                    if (!gs) continue;

                    std::string target = gs->getLabel()->getName().str();
                    auto chain = collectChain(target);
                    if (chain.empty()) continue;

                    // Build new case body: inlined stmts + break.
                    // If chain ends with return, no break needed.
                    bool has_return = false;
                    if (!chain.empty()) {
                        if (llvm::isa<clang::ReturnStmt>(chain.back()))
                            has_return = true;
                    }
                    if (!has_return) {
                        chain.push_back(new (ctx)
                            clang::BreakStmt(clang::SourceLocation()));
                    }

                    auto *new_body = clang::CompoundStmt::Create(
                        ctx, chain, clang::FPOptionsOverride(),
                        clang::SourceLocation(), clang::SourceLocation());
                    cs->setSubStmt(new_body);
                }
            }
        }
    }

    void RefineFallthroughGoto(SNode *node, SNodeFactory &factory,
                               clang::ASTContext &ctx) {
        if (!node) return;

        if (auto *seq = node->dyn_cast<SSeq>()) {
            // Recurse first.
            for (size_t i = 0; i < seq->Size(); ++i)
                RefineFallthroughGoto((*seq)[i], factory, ctx);

            // Case 0 (Ghidra ruleBlockGoto — nested): For each child
            // followed by a label, recursively strip trailing gotos to
            // that label from all branches of the child's if-else tree.
            // This handles: if(A) { ...goto NEXT; } else { ...goto NEXT; } NEXT:
            // by removing the redundant gotos and letting branches fall through.
            {
                auto stripNestedGotos = [](SNode *node, std::string_view target,
                                           auto &self) -> void {
                    if (!node) return;
                    if (node->dyn_cast<SGoto>()) {
                        // Can't modify a leaf — caller handles removal
                        return;
                    }
                    if (auto *s = node->dyn_cast<SSeq>()) {
                        // Remove trailing SGoto(target) if present
                        while (s->Size() > 0) {
                            size_t last = s->Size() - 1;
                            if (auto *g = (*s)[last]->dyn_cast<SGoto>()) {
                                if (g->Target() == target) {
                                    s->RemoveChild(last);
                                    continue;  // check next trailing child
                                }
                            }
                            break;
                        }
                        // Recurse into the last child (skip trailing gotos
                        // to OTHER targets — they're real control flow)
                        for (size_t k = s->Size(); k > 0; --k) {
                            if (!(*s)[k - 1]->dyn_cast<SGoto>()) {
                                self((*s)[k - 1], target, self);
                                break;
                            }
                        }
                        return;
                    }
                    if (auto *ite = node->dyn_cast<SIfThenElse>()) {
                        // Strip from both branches
                        if (auto *tg = ite->ThenBranch()
                                ? ite->ThenBranch()->dyn_cast<SGoto>()
                                : nullptr) {
                            if (tg->Target() == target)
                                ite->SetThenBranch(nullptr);
                        } else {
                            self(ite->ThenBranch(), target, self);
                        }
                        if (auto *eg = ite->ElseBranch()
                                ? ite->ElseBranch()->dyn_cast<SGoto>()
                                : nullptr) {
                            if (eg->Target() == target)
                                ite->SetElseBranch(nullptr);
                        } else {
                            self(ite->ElseBranch(), target, self);
                        }
                        return;
                    }
                    if (auto *lbl = node->dyn_cast<SLabel>()) {
                        self(lbl->Body(), target, self);
                        return;
                    }
                };

                for (size_t i = 0; i + 1 < seq->Size(); ++i) {
                    // Determine the label of the next sibling
                    std::string_view lbl;
                    if (auto *l = (*seq)[i + 1]->dyn_cast<SLabel>())
                        lbl = l->Name();
                    else if (auto *ns = (*seq)[i + 1]->dyn_cast<SSeq>()) {
                        if (ns->Size() > 0)
                            if (auto *l = (*ns)[0]->dyn_cast<SLabel>())
                                lbl = l->Name();
                    }
                    if (lbl.empty()) continue;
                    stripNestedGotos((*seq)[i], lbl, stripNestedGotos);
                }
            }

            // Scan for goto-then-label patterns.
            for (size_t i = 0; i + 1 < seq->Size(); ++i) {
                auto *next = (*seq)[i + 1];
                // Next must start with a label.
                std::string_view next_label;
                if (auto *lbl = next->dyn_cast<SLabel>()) {
                    next_label = lbl->Name();
                } else if (auto *ns = next->dyn_cast<SSeq>()) {
                    if (ns->Size() > 0) {
                        if (auto *lbl = (*ns)[0]->dyn_cast<SLabel>())
                            next_label = lbl->Name();
                    }
                }
                if (next_label.empty()) continue;

                auto *cur = (*seq)[i];

                // Case 1: child[i] is SGoto(L) directly.
                if (auto *g = cur->dyn_cast<SGoto>()) {
                    if (g->Target() == next_label) {
                        seq->RemoveChild(i);
                        --i;  // Re-check this position.
                        continue;
                    }
                }

                // Case 2: IfThenElse (possibly wrapped in SSeq/SLabel)
                //         with goto-branch matching next_label — either
                //         the branch itself is SGoto(L), or it ends with
                //         SGoto(L) at its tail.
                {
                    SNode *tail = cur;
                    while (tail) {
                        if (tail->dyn_cast<SIfThenElse>()) break;
                        if (auto *ts = tail->dyn_cast<SSeq>()) {
                            if (ts->Size() == 0) { tail = nullptr; break; }
                            tail = (*ts)[ts->Size() - 1];
                        } else if (auto *tl = tail->dyn_cast<SLabel>()) {
                            tail = tl->Body();
                        } else {
                            tail = nullptr;
                        }
                    }
                    if (auto *ite = tail ? tail->dyn_cast<SIfThenElse>() : nullptr) {
                        // Check direct branch gotos.
                        if (auto *eg = ite->ElseBranch()
                                ? ite->ElseBranch()->dyn_cast<SGoto>() : nullptr) {
                            if (eg->Target() == next_label)
                                ite->SetElseBranch(nullptr);
                        }
                        if (auto *tg = ite->ThenBranch()
                                ? ite->ThenBranch()->dyn_cast<SGoto>() : nullptr) {
                            if (tg->Target() == next_label)
                                ite->SetThenBranch(nullptr);
                        }

                        // Condition flip (Ghidra ruleBlockGoto):
                        // Pattern: if(cond) goto NEXT; goto OTHER; NEXT:
                        // After then-branch removal: if(cond) {} followed
                        // by goto OTHER.  Negate condition, absorb OTHER
                        // goto into the then-branch, remove the standalone
                        // goto.  This converts:
                        //   if (x != 0) goto next; goto error;
                        // into:
                        //   if (x == 0) goto error;
                        //   // fallthrough to next
                        // Condition flip (Ghidra ruleBlockGoto):
                        // Pattern: if(cond) goto A; goto B;
                        //
                        // When the only then-branch is SGoto(A) and no
                        // else, and the immediately following sibling in
                        // the SAME SSeq is SGoto(B), negate the condition:
                        //   if(!cond) goto B;
                        // This eliminates one goto and enables fallthrough
                        // to A (whether A is a label in this seq or a
                        // parent seq).
                        {
                            auto *tg = ite->ThenBranch()
                                ? ite->ThenBranch()->dyn_cast<SGoto>()
                                : nullptr;
                            if (tg && !ite->ElseBranch() && ite->Cond()) {
                                // Find the SSeq containing ite and its
                                // following sibling.  It may be `seq` (if
                                // ite == cur) or an inner SSeq (if cur is
                                // SLabel/SSeq wrapping ite).
                                SSeq *container = nullptr;
                                size_t ite_idx = 0;

                                // Case A: ite is directly in seq at position i
                                if (cur == ite) {
                                    container = seq;
                                    ite_idx = i;
                                }
                                // Case B: ite is at the tail of cur (SSeq)
                                if (!container) {
                                    if (auto *cs = cur->dyn_cast<SSeq>()) {
                                        for (size_t k = 0; k < cs->Size(); ++k) {
                                            if ((*cs)[k] == ite) {
                                                container = cs;
                                                ite_idx = k;
                                                break;
                                            }
                                        }
                                    }
                                }
                                // Case C: cur is SLabel → body is SSeq
                                if (!container) {
                                    if (auto *lbl = cur->dyn_cast<SLabel>()) {
                                        if (auto *cs = lbl->Body()
                                                ? lbl->Body()->dyn_cast<SSeq>()
                                                : nullptr) {
                                            for (size_t k = 0; k < cs->Size(); ++k) {
                                                if ((*cs)[k] == ite) {
                                                    container = cs;
                                                    ite_idx = k;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }

                                if (container
                                    && ite_idx + 1 < container->Size()) {
                                    if (auto *fg = (*container)[ite_idx + 1]
                                            ->dyn_cast<SGoto>()) {
                                        if (fg->Target() != tg->Target()) {
                                            ite->SetCond(NegateCond(
                                                ite->Cond(), ctx));
                                            ite->SetThenBranch(fg);
                                            ite->SetElseBranch(nullptr);
                                            container->RemoveChild(
                                                ite_idx + 1);
                                        }
                                    }
                                }
                            }
                        }
                        // Note: do NOT remove trailing gotos from IfThenElse
                        // branches — they may be needed to skip over hoisted
                        // labels (the goto exits the branch scope).
                    }
                }

                // Case 3: child[i] (or its tail) is SGoto(L).
                {
                    // Peel through SSeq/SLabel to find trailing SGoto.
                    auto removeTailGoto = [&](SNode *n) -> bool {
                        if (auto *g = n->dyn_cast<SGoto>())
                            return g->Target() == next_label;
                        if (auto *inner = n->dyn_cast<SSeq>()) {
                            if (inner->Size() > 0) {
                                size_t last = inner->Size() - 1;
                                if (auto *g = (*inner)[last]->dyn_cast<SGoto>()) {
                                    if (g->Target() == next_label) {
                                        inner->RemoveChild(last);
                                        return false;  // already handled
                                    }
                                }
                            }
                        }
                        return false;
                    };
                    removeTailGoto(cur);
                }
            }

            // Case 4: Condition flip (Ghidra ruleBlockGoto).
            // Pattern: ...if(cond) goto A; goto B;
            // Convert: ...if(!cond) goto B;
            // The IfThenElse may be nested inside SLabel/SSeq wrappers.
            // The following goto may be at a sibling in the same SSeq
            // or at the next element of the parent SSeq.
            for (size_t i = 0; i + 1 < seq->Size(); ++i) {
                // Peel through SLabel/SSeq to find a trailing IfThenElse.
                SNode *walk = (*seq)[i];
                SIfThenElse *ite = nullptr;
                SSeq *ite_container = nullptr;
                size_t ite_container_idx = 0;
                while (walk) {
                    if (auto *found = walk->dyn_cast<SIfThenElse>()) {
                        ite = found;
                        break;
                    }
                    if (auto *ws = walk->dyn_cast<SSeq>()) {
                        if (ws->Size() == 0) break;
                        // Check if last child is IfThenElse
                        size_t last = ws->Size() - 1;
                        if (auto *found = (*ws)[last]->dyn_cast<SIfThenElse>()) {
                            ite = found;
                            ite_container = ws;
                            ite_container_idx = last;
                            break;
                        }
                        walk = (*ws)[last];
                    } else if (auto *wl = walk->dyn_cast<SLabel>()) {
                        walk = wl->Body();
                    } else {
                        break;
                    }
                }
                if (!ite || ite->ElseBranch()) continue;
                auto *tg = ite->ThenBranch()
                    ? ite->ThenBranch()->dyn_cast<SGoto>() : nullptr;
                if (!tg || !ite->Cond()) continue;

                // Find the following SGoto: either as a sibling in the
                // ite_container SSeq, or at seq[i+1].
                SGoto *fg = nullptr;
                SSeq *fg_owner = nullptr;
                size_t fg_idx = 0;

                // Check ite_container's next sibling
                if (ite_container
                    && ite_container_idx + 1 < ite_container->Size()) {
                    fg = (*ite_container)[ite_container_idx + 1]
                        ->dyn_cast<SGoto>();
                    if (fg) {
                        fg_owner = ite_container;
                        fg_idx = ite_container_idx + 1;
                    }
                }
                // Check parent seq's next sibling
                if (!fg) {
                    fg = (*seq)[i + 1]->dyn_cast<SGoto>();
                    if (fg) {
                        fg_owner = seq;
                        fg_idx = i + 1;
                    }
                }

                if (!fg || fg->Target() == tg->Target()) continue;

                // Negate condition, swap: if(!cond) goto B;
                ite->SetCond(NegateCond(ite->Cond(), ctx));
                ite->SetThenBranch(fg);
                fg_owner->RemoveChild(fg_idx);
            }

            // Cleanup: remove SGoto(L) that immediately follows SLabel(L,...)
            // in this SSeq (dead goto to preceding label).
            for (size_t i = 1; i < seq->Size(); ) {
                auto *g = (*seq)[i]->dyn_cast<SGoto>();
                if (!g) { ++i; continue; }
                auto *prev = (*seq)[i - 1]->dyn_cast<SLabel>();
                if (prev && prev->Name() == g->Target()) {
                    seq->RemoveChild(i);
                } else {
                    ++i;
                }
            }
            // Cleanup: remove dead trailing gotos after nodes that
            // already end with a transfer (unreachable code).
            for (size_t i = 1; i < seq->Size(); ) {
                if (!(*seq)[i]->dyn_cast<SGoto>()) { ++i; continue; }
                if (EndsWithTransfer((*seq)[i - 1])) {
                    seq->RemoveChild(i);
                } else {
                    ++i;
                }
            }
        } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            RefineFallthroughGoto(ite->ThenBranch(), factory, ctx);
            RefineFallthroughGoto(ite->ElseBranch(), factory, ctx);

            // Normalize: if(cond) {} else { body } → if(!cond) { body }
            // An empty/null then-branch with a non-empty else is confusing;
            // negate the condition and swap the branches.
            auto isBranchEmpty = [](SNode *n) -> bool {
                if (!n) return true;
                if (auto *blk = n->dyn_cast<SBlock>())
                    return blk->Stmts().empty();
                if (auto *s = n->dyn_cast<SSeq>())
                    return s->Size() == 0;
                return false;
            };
            if (isBranchEmpty(ite->ThenBranch())
                && !isBranchEmpty(ite->ElseBranch())
                && ite->Cond()) {
                ite->SetCond(NegateCond(ite->Cond(), ctx));
                SNode *old_else = ite->ElseBranch();
                ite->SetElseBranch(nullptr);
                ite->SetThenBranch(old_else);
            }
        } else if (auto *w = node->dyn_cast<SWhile>()) {
            RefineFallthroughGoto(w->Body(), factory, ctx);
        } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            RefineFallthroughGoto(dw->Body(), factory, ctx);
        } else if (auto *f = node->dyn_cast<SFor>()) {
            RefineFallthroughGoto(f->Body(), factory, ctx);
        } else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases()) RefineFallthroughGoto(c.body, factory, ctx);
            RefineFallthroughGoto(sw->DefaultBody(), factory, ctx);
        } else if (auto *lbl = node->dyn_cast<SLabel>()) {
            RefineFallthroughGoto(lbl->Body(), factory, ctx);
        }
    }

    // Remove SLabel nodes whose name is not in the goto target set.
    // If the SLabel has a body, replace the label with its body.
    // If no body, remove entirely.
    void RemoveDeadLabels(SNode *node,
                          const std::unordered_set<std::string> &targets) {
        if (!node) return;

        if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ) {
                auto *child = (*seq)[i];
                if (auto *lbl = child->dyn_cast<SLabel>()) {
                    if (targets.find(std::string(lbl->Name())) == targets.end()) {
                        // Dead label -- replace with body or remove
                        if (lbl->Body()) {
                            seq->ReplaceChild(i, lbl->Body());
                            // Don't increment — re-check the replacement
                        } else {
                            seq->RemoveChild(i);
                        }
                        continue;
                    }
                }
                // Recurse into child
                RemoveDeadLabels(child, targets);
                ++i;
            }
        }
        else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            RemoveDeadLabels(ite->ThenBranch(), targets);
            RemoveDeadLabels(ite->ElseBranch(), targets);
        }
        else if (auto *w = node->dyn_cast<SWhile>()) {
            RemoveDeadLabels(w->Body(), targets);
        }
        else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            RemoveDeadLabels(dw->Body(), targets);
        }
        else if (auto *f = node->dyn_cast<SFor>()) {
            RemoveDeadLabels(f->Body(), targets);
        }
        else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases()) {
                RemoveDeadLabels(c.body, targets);
            }
            RemoveDeadLabels(sw->DefaultBody(), targets);
        }
        else if (auto *lbl = node->dyn_cast<SLabel>()) {
            RemoveDeadLabels(lbl->Body(), targets);
        }
    }

    // --- RefineEpilogueReturn: inline epilogue returns ---
    //
    // Recognizes epilogue blocks (labeled blocks ending with a return)
    // and replaces all gotos to them with the return statement.  This
    // eliminates the high fan-in epilogue pattern common in decompiled
    // code where the compiler merges all return paths into a single
    // block with a stack canary check.
    //
    // Pattern: goto EPILOGUE; ... EPILOGUE: ...; return EXPR;
    // Convert: return EXPR;

    /// Extract the return expression from a labeled block, if the block
    /// ends with a return (possibly behind a canary check).
    static clang::Expr *ExtractReturnExpr(SNode *node) {
        if (!node) return nullptr;
        // Walk through SSeq to find the last meaningful node
        if (auto *s = node->dyn_cast<SSeq>()) {
            for (size_t i = s->Size(); i > 0; --i) {
                if (auto *e = ExtractReturnExpr((*s)[i - 1]))
                    return e;
            }
            return nullptr;
        }
        // Check SBlock for ReturnStmt (prefer conditional return over
        // fallback return after stack_chk_fail).
        if (auto *blk = node->dyn_cast<SBlock>()) {
            // First pass: look for return inside if-then (canary pattern)
            for (auto *s : blk->Stmts()) {
                if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(s)) {
                    if (auto *ret = llvm::dyn_cast_or_null<clang::ReturnStmt>(
                            ifs->getThen()))
                        return ret->getRetValue();
                    // Also check CompoundStmt{ReturnStmt}
                    if (auto *cs = llvm::dyn_cast_or_null<clang::CompoundStmt>(
                            ifs->getThen())) {
                        for (auto *sub : cs->body())
                            if (auto *ret = llvm::dyn_cast<clang::ReturnStmt>(sub))
                                return ret->getRetValue();
                    }
                }
            }
            // Second pass: trailing ReturnStmt directly in block
            for (auto it = blk->Stmts().rbegin(); it != blk->Stmts().rend(); ++it) {
                if (auto *ret = llvm::dyn_cast<clang::ReturnStmt>(*it))
                    return ret->getRetValue();
            }
            return nullptr;
        }
        // Check SReturn directly
        if (auto *ret = node->dyn_cast<SReturn>())
            return ret->Value();
        // Check if-then: if (canary_ok) return EXPR; → extract EXPR
        if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            if (auto *e = ExtractReturnExpr(ite->ThenBranch()))
                return e;
            if (auto *e = ExtractReturnExpr(ite->ElseBranch()))
                return e;
        }
        // SLabel: peel
        if (auto *lbl = node->dyn_cast<SLabel>())
            return ExtractReturnExpr(lbl->Body());
        return nullptr;
    }

    /// Replace all SGoto(target) in the tree with SReturn(expr).
    static void ReplaceGotoWithReturn(SNode *node, std::string_view target,
                                       clang::Expr *return_expr,
                                       SNodeFactory &factory) {
        if (!node) return;
        if (auto *s = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < s->Size(); ++i) {
                if (auto *g = (*s)[i]->dyn_cast<SGoto>()) {
                    if (g->Target() == target) {
                        s->ReplaceChild(i, factory.Make<SReturn>(return_expr));
                    }
                } else {
                    ReplaceGotoWithReturn((*s)[i], target, return_expr, factory);
                }
            }
        } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            // Direct goto as branch → replace
            if (auto *g = ite->ThenBranch()
                    ? ite->ThenBranch()->dyn_cast<SGoto>() : nullptr) {
                if (g->Target() == target)
                    ite->SetThenBranch(factory.Make<SReturn>(return_expr));
            } else {
                ReplaceGotoWithReturn(ite->ThenBranch(), target, return_expr, factory);
            }
            if (auto *g = ite->ElseBranch()
                    ? ite->ElseBranch()->dyn_cast<SGoto>() : nullptr) {
                if (g->Target() == target)
                    ite->SetElseBranch(factory.Make<SReturn>(return_expr));
            } else {
                ReplaceGotoWithReturn(ite->ElseBranch(), target, return_expr, factory);
            }
        } else if (auto *w = node->dyn_cast<SWhile>()) {
            ReplaceGotoWithReturn(w->Body(), target, return_expr, factory);
        } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            ReplaceGotoWithReturn(dw->Body(), target, return_expr, factory);
        } else if (auto *f = node->dyn_cast<SFor>()) {
            ReplaceGotoWithReturn(f->Body(), target, return_expr, factory);
        } else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases())
                ReplaceGotoWithReturn(c.body, target, return_expr, factory);
            ReplaceGotoWithReturn(sw->DefaultBody(), target, return_expr, factory);
        } else if (auto *lbl = node->dyn_cast<SLabel>()) {
            ReplaceGotoWithReturn(lbl->Body(), target, return_expr, factory);
        }
    }

    void RefineEpilogueReturn(SNode *root, SNodeFactory &factory) {
        if (!root) return;

        // Step 1: Collect all labeled blocks that contain a return.
        // Also handle the pattern where a labeled block falls through
        // to the next labeled block which has the return (common with
        // stack canary epilogues: label1: setup; label2: return X).
        std::unordered_map<std::string, clang::Expr *> epilogues;
        std::function<void(SNode *)> findEpilogues = [&](SNode *n) {
            if (!n) return;
            if (auto *lbl = n->dyn_cast<SLabel>()) {
                auto *ret = ExtractReturnExpr(lbl->Body());
                if (ret)
                    epilogues[std::string(lbl->Name())] = ret;
                findEpilogues(lbl->Body());
                return;
            }
            if (auto *s = n->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < s->Size(); ++i)
                    findEpilogues((*s)[i]);
                // After dead-label removal, a label's content may be
                // split across the label body + following bare blocks.
                // Check: if SLabel at [i] has no return but [i+1..] bare
                // blocks do, associate the return with the label.
                for (size_t i = 0; i < s->Size(); ++i) {
                    auto *lbl = (*s)[i]->dyn_cast<SLabel>();
                    if (!lbl || epilogues.count(std::string(lbl->Name())))
                        continue;
                    // Scan following non-label siblings for a return
                    for (size_t j = i + 1; j < s->Size(); ++j) {
                        if ((*s)[j]->dyn_cast<SLabel>()) break;
                        if (auto *ret = ExtractReturnExpr((*s)[j])) {
                            epilogues[std::string(lbl->Name())] = ret;
                            break;
                        }
                    }
                }
            } else if (auto *ite = n->dyn_cast<SIfThenElse>()) {
                findEpilogues(ite->ThenBranch());
                findEpilogues(ite->ElseBranch());
            }
        };
        findEpilogues(root);
        if (epilogues.empty()) return;

        // Step 1b: Chain epilogues through gotos.
        // If a labeled block has no return but ends with goto to a known
        // epilogue, it becomes an epilogue too (with the same return expr).
        // This handles: 114_basic (setup stmts + falls through to 115_basic)
        // where 115_basic has the return.
        {
            // Build label→trailing-goto map
            std::unordered_map<std::string, std::string> label_goto;
            std::function<void(SNode *)> collectGotos = [&](SNode *n) {
                if (!n) return;
                if (auto *lbl = n->dyn_cast<SLabel>()) {
                    // Extract trailing goto from body
                    auto extractGoto = [](SNode *body) -> std::string_view {
                        if (!body) return {};
                        if (auto *g = body->dyn_cast<SGoto>()) return g->Target();
                        if (auto *s = body->dyn_cast<SSeq>()) {
                            if (s->Size() > 0)
                                if (auto *g = (*s)[s->Size()-1]->dyn_cast<SGoto>())
                                    return g->Target();
                        }
                        // SBlock without SGoto → fallthrough, don't register
                        return {};
                    };
                    auto tgt = extractGoto(lbl->Body());
                    if (!tgt.empty())
                        label_goto[std::string(lbl->Name())] = std::string(tgt);
                    collectGotos(lbl->Body());
                    return;
                }
                if (auto *s = n->dyn_cast<SSeq>())
                    for (size_t i = 0; i < s->Size(); ++i) collectGotos((*s)[i]);
                else if (auto *ite = n->dyn_cast<SIfThenElse>()) {
                    collectGotos(ite->ThenBranch());
                    collectGotos(ite->ElseBranch());
                }
            };
            collectGotos(root);

            // Propagate: follow goto chains to known epilogues
            bool changed = true;
            while (changed) {
                changed = false;
                for (auto &[label, target] : label_goto) {
                    if (epilogues.count(label)) continue;
                    auto it = epilogues.find(target);
                    if (it != epilogues.end()) {
                        epilogues[label] = it->second;
                        changed = true;
                    }
                }
            }
        }

        // Step 2: Count gotos per epilogue label
        std::unordered_map<std::string, size_t> goto_counts;
        std::function<void(SNode *)> countGotos = [&](SNode *n) {
            if (!n) return;
            if (auto *g = n->dyn_cast<SGoto>()) {
                auto it = epilogues.find(std::string(g->Target()));
                if (it != epilogues.end()) ++goto_counts[it->first];
                return;
            }
            if (auto *s = n->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < s->Size(); ++i) countGotos((*s)[i]);
            } else if (auto *ite = n->dyn_cast<SIfThenElse>()) {
                countGotos(ite->ThenBranch());
                countGotos(ite->ElseBranch());
            } else if (auto *w = n->dyn_cast<SWhile>()) {
                countGotos(w->Body());
            } else if (auto *dw = n->dyn_cast<SDoWhile>()) {
                countGotos(dw->Body());
            } else if (auto *f = n->dyn_cast<SFor>()) {
                countGotos(f->Body());
            } else if (auto *sw = n->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases()) countGotos(c.body);
                countGotos(sw->DefaultBody());
            } else if (auto *lbl = n->dyn_cast<SLabel>()) {
                countGotos(lbl->Body());
            }
        };
        countGotos(root);

        // Step 3: Replace gotos for high-fanin epilogues (>= 3 gotos)
        for (auto &[label, count] : goto_counts) {
            if (count < 3) continue;
            auto *ret_expr = epilogues[label];
            ReplaceGotoWithReturn(root, label, ret_expr, factory);
        }
    }

    // RefineDeadLabels: remove dead labels left after RefineBreakContinue converts
    // gotos to break/continue. For v1, this is dead label removal only
    // (full bump-up optimization deferred).
    void RefineDeadLabels(SNode *root) {
        std::unordered_set<std::string> targets;
        CollectGotoTargets(root, targets);
        RemoveDeadLabels(root, targets);
    }

    // --- RefineDeadCode ---
    // Remove unreachable statements after transfers (goto/return/break/
    // continue), empty SSeq/SBlock nodes, and if-then-else with both
    // branches empty.  Recurse bottom-up so inner dead code is cleaned
    // before outer checks.

    /// Returns true if the node is empty (no meaningful content).
    [[maybe_unused]]
    static bool IsEmpty(SNode *n) {
        if (!n) return true;
        if (auto *blk = n->dyn_cast<SBlock>())
            return blk->Stmts().empty();
        if (auto *s = n->dyn_cast<SSeq>())
            return s->Size() == 0;
        return false;
    }

    void RefineDeadCode(SNode *node) {
        if (!node) return;

        if (auto *seq = node->dyn_cast<SSeq>()) {
            // Recurse first
            for (size_t i = 0; i < seq->Size(); ++i)
                RefineDeadCode((*seq)[i]);

            // Remove only bare transfer stmts (goto/return/break/continue)
            // that follow another transfer — these are provably dead.
            // We do NOT remove SBlock/SIfThenElse/SLabel after transfers
            // because they may contain goto targets or side effects.
            for (size_t i = 1; i < seq->Size(); ) {
                auto *cur = (*seq)[i];
                // Only remove bare transfer nodes after a transfer
                bool cur_is_bare_transfer =
                    cur->dyn_cast<SGoto>() || cur->dyn_cast<SReturn>()
                    || cur->dyn_cast<SBreak>() || cur->dyn_cast<SContinue>();
                if (cur_is_bare_transfer && EndsWithTransfer((*seq)[i - 1])) {
                    seq->RemoveChild(i);
                } else {
                    ++i;
                }
            }
        } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            RefineDeadCode(ite->ThenBranch());
            RefineDeadCode(ite->ElseBranch());
        } else if (auto *w = node->dyn_cast<SWhile>()) {
            RefineDeadCode(w->Body());
        } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            RefineDeadCode(dw->Body());
        } else if (auto *f = node->dyn_cast<SFor>()) {
            RefineDeadCode(f->Body());
        } else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases()) RefineDeadCode(c.body);
            RefineDeadCode(sw->DefaultBody());
        } else if (auto *lbl = node->dyn_cast<SLabel>()) {
            RefineDeadCode(lbl->Body());
        }
    }

    // --- RefineCommonGotoHoist ---
    // If both branches of an if-else end with goto SAME_TARGET,
    // hoist the goto after the if-else and remove from both branches.
    // Recurse into nested structures.

    /// Extract the trailing SGoto target from an SNode, or empty string.
    static std::string_view TrailingGotoTarget(SNode *n) {
        if (!n) return {};
        if (auto *g = n->dyn_cast<SGoto>()) return g->Target();
        if (auto *s = n->dyn_cast<SSeq>()) {
            if (s->Size() > 0)
                return TrailingGotoTarget((*s)[s->Size() - 1]);
        }
        if (auto *ite = n->dyn_cast<SIfThenElse>()) {
            // Both branches must end with same goto
            auto t = TrailingGotoTarget(ite->ThenBranch());
            auto e = TrailingGotoTarget(ite->ElseBranch());
            if (!t.empty() && t == e) return t;
        }
        return {};
    }

    /// Remove the trailing SGoto from an SNode (in-place).
    static void RemoveTrailingGoto(SNode *n, std::string_view target) {
        if (!n) return;
        if (auto *s = n->dyn_cast<SSeq>()) {
            if (s->Size() > 0) {
                auto *last = (*s)[s->Size() - 1];
                if (auto *g = last->dyn_cast<SGoto>()) {
                    if (g->Target() == target) {
                        s->RemoveChild(s->Size() - 1);
                        return;
                    }
                }
                RemoveTrailingGoto(last, target);
            }
        } else if (auto *ite = n->dyn_cast<SIfThenElse>()) {
            RemoveTrailingGoto(ite->ThenBranch(), target);
            RemoveTrailingGoto(ite->ElseBranch(), target);
        }
    }

    void RefineCommonGotoHoist(SNode *node, SNodeFactory &factory) {
        if (!node) return;
        // Recurse first
        if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ++i)
                RefineCommonGotoHoist((*seq)[i], factory);

            // Scan: if child[i] is IfThenElse where both branches
            // end with goto SAME, hoist.
            for (size_t i = 0; i < seq->Size(); ++i) {
                // Peel through SLabel/SSeq to find the IfThenElse
                SNode *walk = (*seq)[i];
                while (walk) {
                    if (walk->dyn_cast<SIfThenElse>()) break;
                    if (auto *ws = walk->dyn_cast<SSeq>()) {
                        if (ws->Size() == 0) { walk = nullptr; break; }
                        walk = (*ws)[ws->Size() - 1];
                    } else if (auto *wl = walk->dyn_cast<SLabel>()) {
                        walk = wl->Body();
                    } else {
                        walk = nullptr;
                    }
                }
                auto *ite = walk ? walk->dyn_cast<SIfThenElse>() : nullptr;
                if (!ite || !ite->ThenBranch() || !ite->ElseBranch()) continue;

                auto t = TrailingGotoTarget(ite->ThenBranch());
                auto e = TrailingGotoTarget(ite->ElseBranch());
                if (t.empty() || t != e) continue;

                // Hoist: remove goto from both branches, insert after
                RemoveTrailingGoto(ite->ThenBranch(), t);
                RemoveTrailingGoto(ite->ElseBranch(), t);
                auto *goto_node = factory.Make<SGoto>(factory.Intern(t));
                seq->InsertChild(i + 1, goto_node);
                ++i;  // skip the inserted goto
            }
        } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            RefineCommonGotoHoist(ite->ThenBranch(), factory);
            RefineCommonGotoHoist(ite->ElseBranch(), factory);
        } else if (auto *w = node->dyn_cast<SWhile>()) {
            RefineCommonGotoHoist(w->Body(), factory);
        } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            RefineCommonGotoHoist(dw->Body(), factory);
        } else if (auto *f = node->dyn_cast<SFor>()) {
            RefineCommonGotoHoist(f->Body(), factory);
        } else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases()) RefineCommonGotoHoist(c.body, factory);
            RefineCommonGotoHoist(sw->DefaultBody(), factory);
        } else if (auto *lbl = node->dyn_cast<SLabel>()) {
            RefineCommonGotoHoist(lbl->Body(), factory);
        }
    }

    // --- RefineSmallBlockInline ---
    // Replace goto LABEL with the block's content when the labeled
    // block is small (≤ max_stmts stmts). This eliminates gotos to
    // trivial error handlers, pointer-advance blocks, etc.

    /// Count raw stmts in an SNode (SBlock stmts only, not recursive).
    static size_t CountBlockStmts(SNode *n) {
        if (!n) return 0;
        if (auto *blk = n->dyn_cast<SBlock>()) return blk->Stmts().size();
        if (auto *s = n->dyn_cast<SSeq>()) {
            size_t total = 0;
            for (size_t i = 0; i < s->Size(); ++i)
                total += CountBlockStmts((*s)[i]);
            return total;
        }
        return 1;  // SGoto, SIfThenElse, etc. count as 1
    }

    /// Clone an SNode tree (shallow: shares Clang Stmt* pointers).
    static SNode *CloneSNode(SNode *n, SNodeFactory &factory) {
        if (!n) return nullptr;
        if (auto *g = n->dyn_cast<SGoto>())
            return factory.Make<SGoto>(g->Target());
        if (auto *blk = n->dyn_cast<SBlock>()) {
            auto *copy = factory.Make<SBlock>();
            for (auto *s : blk->Stmts()) copy->AddStmt(s);
            return copy;
        }
        if (auto *s = n->dyn_cast<SSeq>()) {
            auto *copy = factory.Make<SSeq>();
            for (size_t i = 0; i < s->Size(); ++i)
                copy->AddChild(CloneSNode((*s)[i], factory));
            return copy;
        }
        if (auto *ite = n->dyn_cast<SIfThenElse>()) {
            return factory.Make<SIfThenElse>(
                ite->Cond(),
                CloneSNode(ite->ThenBranch(), factory),
                CloneSNode(ite->ElseBranch(), factory));
        }
        if (auto *lbl = n->dyn_cast<SLabel>()) {
            return factory.Make<SLabel>(
                factory.Intern(lbl->Name()),
                CloneSNode(lbl->Body(), factory));
        }
        if (auto *ret = n->dyn_cast<SReturn>())
            return factory.Make<SReturn>(ret->Value());
        if (n->dyn_cast<SBreak>())
            return factory.Make<SBreak>();
        if (n->dyn_cast<SContinue>())
            return factory.Make<SContinue>();
        return nullptr;  // unhandled type — don't inline
    }

    /// Replace goto LABEL with cloned block content, recursively.
    static void InlineGotoWithBlock(SNode *node, std::string_view target,
                                     SNode *block_body, SNodeFactory &factory) {
        if (!node) return;
        if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ++i) {
                if (auto *g = (*seq)[i]->dyn_cast<SGoto>()) {
                    if (g->Target() == target) {
                        auto *clone = CloneSNode(block_body, factory);
                        if (clone)
                            seq->ReplaceChild(i, clone);
                    }
                } else {
                    InlineGotoWithBlock((*seq)[i], target, block_body, factory);
                }
            }
        } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            if (auto *g = ite->ThenBranch()
                    ? ite->ThenBranch()->dyn_cast<SGoto>() : nullptr) {
                if (g->Target() == target) {
                    auto *clone = CloneSNode(block_body, factory);
                    if (clone) ite->SetThenBranch(clone);
                }
            } else {
                InlineGotoWithBlock(ite->ThenBranch(), target, block_body, factory);
            }
            if (auto *g = ite->ElseBranch()
                    ? ite->ElseBranch()->dyn_cast<SGoto>() : nullptr) {
                if (g->Target() == target) {
                    auto *clone = CloneSNode(block_body, factory);
                    if (clone) ite->SetElseBranch(clone);
                }
            } else {
                InlineGotoWithBlock(ite->ElseBranch(), target, block_body, factory);
            }
        } else if (auto *w = node->dyn_cast<SWhile>()) {
            InlineGotoWithBlock(w->Body(), target, block_body, factory);
        } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            InlineGotoWithBlock(dw->Body(), target, block_body, factory);
        } else if (auto *f = node->dyn_cast<SFor>()) {
            InlineGotoWithBlock(f->Body(), target, block_body, factory);
        } else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases())
                InlineGotoWithBlock(c.body, target, block_body, factory);
            InlineGotoWithBlock(sw->DefaultBody(), target, block_body, factory);
        } else if (auto *lbl = node->dyn_cast<SLabel>()) {
            InlineGotoWithBlock(lbl->Body(), target, block_body, factory);
        }
    }

    void RefineSmallBlockInline(SNode *root, SNodeFactory &factory) {
        if (!root) return;
        constexpr size_t kMaxInlineStmts = 4;

        // Step 1: Find small labeled blocks and their content.
        // A "small block" is SLabel(name, body) where body has
        // ≤ kMaxInlineStmts stmts (including its trailing goto).
        std::unordered_map<std::string, SNode *> small_blocks;
        std::function<void(SNode *)> findSmall = [&](SNode *n) {
            if (!n) return;
            if (auto *lbl = n->dyn_cast<SLabel>()) {
                // Only inline blocks that end with a goto (not returns
                // or fallthrough blocks like the epilogue).
                auto endsWithGoto = [](SNode *body) -> bool {
                    if (!body) return false;
                    if (body->dyn_cast<SGoto>()) return true;
                    if (auto *s = body->dyn_cast<SSeq>()) {
                        if (s->Size() > 0)
                            return (*s)[s->Size()-1]->dyn_cast<SGoto>() != nullptr;
                    }
                    return false;
                };
                size_t count = CountBlockStmts(lbl->Body());
                if (count > 0 && count <= kMaxInlineStmts
                    && endsWithGoto(lbl->Body()))
                    small_blocks[std::string(lbl->Name())] = lbl->Body();
                findSmall(lbl->Body());
                return;
            }
            if (auto *s = n->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < s->Size(); ++i)
                    findSmall((*s)[i]);
                // Also check label + following SGoto sibling.
                // After dead-label removal, a block's goto may be a bare
                // sibling: SLabel(body), SGoto(target).  Combine them.
                for (size_t i = 0; i + 1 < s->Size(); ++i) {
                    auto *lbl = (*s)[i]->dyn_cast<SLabel>();
                    if (!lbl || small_blocks.count(std::string(lbl->Name())))
                        continue;
                    auto *next_goto = (*s)[i + 1]->dyn_cast<SGoto>();
                    if (!next_goto) continue;
                    auto *combined = factory.Make<SSeq>();
                    if (lbl->Body()) combined->AddChild(lbl->Body());
                    combined->AddChild(next_goto);
                    size_t count = CountBlockStmts(combined);
                    if (count > 0 && count <= kMaxInlineStmts)
                        small_blocks[std::string(lbl->Name())] = combined;
                }
            } else if (auto *ite = n->dyn_cast<SIfThenElse>()) {
                findSmall(ite->ThenBranch());
                findSmall(ite->ElseBranch());
            }
        };
        findSmall(root);
        if (small_blocks.empty()) return;

        // Step 2: Count gotos per small block
        std::unordered_map<std::string, size_t> goto_counts;
        std::function<void(SNode *)> countGotos = [&](SNode *n) {
            if (!n) return;
            if (auto *g = n->dyn_cast<SGoto>()) {
                if (small_blocks.count(std::string(g->Target())))
                    ++goto_counts[std::string(g->Target())];
                return;
            }
            if (auto *s = n->dyn_cast<SSeq>()) {
                for (size_t i = 0; i < s->Size(); ++i) countGotos((*s)[i]);
            } else if (auto *ite = n->dyn_cast<SIfThenElse>()) {
                countGotos(ite->ThenBranch());
                countGotos(ite->ElseBranch());
            } else if (auto *w = n->dyn_cast<SWhile>()) {
                countGotos(w->Body());
            } else if (auto *dw = n->dyn_cast<SDoWhile>()) {
                countGotos(dw->Body());
            } else if (auto *f = n->dyn_cast<SFor>()) {
                countGotos(f->Body());
            } else if (auto *sw = n->dyn_cast<SSwitch>()) {
                for (auto &c : sw->Cases()) countGotos(c.body);
                countGotos(sw->DefaultBody());
            } else if (auto *lbl = n->dyn_cast<SLabel>()) {
                countGotos(lbl->Body());
            }
        };
        countGotos(root);

        // Step 3: Inline blocks with ≥ 2 gotos targeting them
        for (auto &[label, count] : goto_counts) {
            if (count < 2) continue;
            auto *body = small_blocks[label];
            if (!body) continue;
            InlineGotoWithBlock(root, label, body, factory);
        }
    }

    // --- RefineWhileTrueToDoWhile helpers ---

    // Collect all VarDecl* read by an expression (recursive)
    void CollectReadVars(clang::Stmt *s,
                         std::unordered_set<clang::VarDecl *> &vars) {
        if (!s) return;
        if (auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(s)) {
            if (auto *vd = llvm::dyn_cast<clang::VarDecl>(dre->getDecl()))
                vars.insert(vd);
        }
        for (auto *child : s->children())
            CollectReadVars(child, vars);
    }

    // Collect all VarDecl* written by SNode stmts (assignments)
    void CollectSNodeWrites(const SNode *node,
                            std::unordered_set<clang::VarDecl *> &vars) {
        if (!node) return;
        if (auto *blk = node->dyn_cast<SBlock>()) {
            for (auto *s : blk->Stmts()) {
                auto *vd = GetReferencedVar(s);
                if (vd) vars.insert(vd);
            }
        } else if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ++i)
                CollectSNodeWrites((*seq)[i], vars);
        } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            CollectSNodeWrites(ite->ThenBranch(), vars);
            CollectSNodeWrites(ite->ElseBranch(), vars);
        } else if (auto *lbl = node->dyn_cast<SLabel>()) {
            CollectSNodeWrites(lbl->Body(), vars);
        }
    }

    bool IsLiteralOne(clang::Expr *e) {
        if (!e) return false;
        if (auto *il = llvm::dyn_cast<clang::IntegerLiteral>(e))
            return il->getValue() == 1;
        return false;
    }

    // RefineWhileTrueToDoWhile: convert while(true) with if-break
    // to do-while when safe.
    [[maybe_unused]]
    void RefineWhileTrueToDoWhile(SNode *node, SNodeFactory &factory,
                                   clang::ASTContext &ctx) {
        if (!node) return;

        // Recurse into children first (bottom-up)
        if (auto *seq = node->dyn_cast<SSeq>()) {
            for (size_t i = 0; i < seq->Size(); ++i)
                RefineWhileTrueToDoWhile((*seq)[i], factory, ctx);
        } else if (auto *ite = node->dyn_cast<SIfThenElse>()) {
            RefineWhileTrueToDoWhile(ite->ThenBranch(), factory, ctx);
            RefineWhileTrueToDoWhile(ite->ElseBranch(), factory, ctx);
        } else if (auto *w = node->dyn_cast<SWhile>()) {
            RefineWhileTrueToDoWhile(w->Body(), factory, ctx);
        } else if (auto *dw = node->dyn_cast<SDoWhile>()) {
            RefineWhileTrueToDoWhile(dw->Body(), factory, ctx);
        } else if (auto *f = node->dyn_cast<SFor>()) {
            RefineWhileTrueToDoWhile(f->Body(), factory, ctx);
        } else if (auto *sw = node->dyn_cast<SSwitch>()) {
            for (auto &c : sw->Cases())
                RefineWhileTrueToDoWhile(c.body, factory, ctx);
            RefineWhileTrueToDoWhile(sw->DefaultBody(), factory, ctx);
        } else if (auto *lbl = node->dyn_cast<SLabel>()) {
            RefineWhileTrueToDoWhile(lbl->Body(), factory, ctx);
        }

        // Now check if this node is a parent SSeq containing a while(true)
        auto *parent_seq = node->dyn_cast<SSeq>();
        if (!parent_seq) return;

        for (size_t wi = 0; wi < parent_seq->Size(); ++wi) {
            auto *w = (*parent_seq)[wi]->dyn_cast<SWhile>();
            if (!w || !IsLiteralOne(w->Cond())) continue;

            // Skip self-loop patterns already handled by FoldDoWhileLoop.
            // If the body is a single SDoWhile, this while(true) wraps
            // a do-while that was already properly structured.
            if (w->Body() && w->Body()->isa<SDoWhile>()) continue;

            auto *body = w->Body() ? w->Body()->dyn_cast<SSeq>() : nullptr;
            if (!body || body->Size() < 2) continue;

            // Find the first SIfThenElse(cond, SBreak, null)
            for (size_t k = 0; k < body->Size(); ++k) {
                auto *ite = (*body)[k]->dyn_cast<SIfThenElse>();
                if (!ite) continue;
                if (!ite->ThenBranch() || !ite->ThenBranch()->isa<SBreak>())
                    continue;
                if (ite->ElseBranch()) continue;

                clang::Expr *exit_cond = ite->Cond();
                if (!exit_cond) continue;

                // Trivial: break is last child
                if (k == body->Size() - 1) {
                    body->RemoveChild(k);
                    auto *dowhile = factory.Make<SDoWhile>(
                        body, NegateCond(exit_cond, ctx));
                    parent_seq->ReplaceChild(wi, dowhile);
                    break;
                }

                // Middle break: check def-use safety.
                // Collect vars read by the exit condition.
                std::unordered_set<clang::VarDecl *> cond_vars;
                CollectReadVars(exit_cond, cond_vars);
                if (cond_vars.empty()) break; // can't verify, skip

                // Collect vars written by stmts AFTER the break.
                std::unordered_set<clang::VarDecl *> body_writes;
                for (size_t j = k + 1; j < body->Size(); ++j)
                    CollectSNodeWrites((*body)[j], body_writes);

                // Check intersection
                bool safe = true;
                for (auto *v : cond_vars) {
                    if (body_writes.count(v)) { safe = false; break; }
                }
                if (!safe) break;

                // Safe: wrap post-break stmts in if(!cond) { ... }
                auto *guarded = factory.Make<SSeq>();
                while (body->Size() > k + 1) {
                    guarded->AddChild((*body)[k + 1]);
                    body->RemoveChild(k + 1);
                }
                body->RemoveChild(k); // remove if-break

                auto *guard = factory.Make<SIfThenElse>(
                    NegateCond(exit_cond, ctx), guarded, nullptr);
                body->AddChild(guard);

                auto *dowhile = factory.Make<SDoWhile>(
                    body, NegateCond(exit_cond, ctx));
                parent_seq->ReplaceChild(wi, dowhile);
                break;
            }
        }
    }


} // namespace detail
} // namespace patchestry::ast
