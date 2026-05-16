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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/PrettyPrinter.h>
#include <clang/AST/Stmt.h>

#include <llvm/Support/raw_ostream.h>

namespace patchestry::ast {

namespace detail {
    static clang::CompoundStmt *MakeCompound(
        clang::ASTContext &ctx, const std::vector< clang::Stmt * > &stmts) {
        auto loc = VirtualLoc(ctx);
        return clang::CompoundStmt::Create(ctx, stmts, clang::FPOptionsOverride(), loc, loc);
    }
} // namespace detail

    // Collect all LabelDecls referenced by GotoStmts in a Stmt tree.
    static void CollectGotoTargets(clang::Stmt *s,
                                   std::unordered_set< clang::LabelDecl * > &targets,
                                   std::unordered_set< clang::Stmt * > &seen) {
        if (!s || !seen.insert(s).second) return;
        if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
            targets.insert(gs->getLabel());
            return;
        }
        for (auto *child : s->children()) {
            CollectGotoTargets(child, targets, seen);
        }
    }
    // ---- Pretty-print cleanup (patchir-decomp only) ----

    namespace {
        // If stmt is a LabelStmt wrapping a CompoundStmt, push the label inside:
        //   LabelStmt(CompoundStmt{s1, s2, ...}) → CompoundStmt{LabelStmt(s1), s2, ...}
        // Otherwise return the stmt unchanged.
        clang::Stmt *PushLabelInside(clang::ASTContext &ctx, clang::Stmt *s) {
            auto *ls = llvm::dyn_cast_or_null< clang::LabelStmt >(s);
            if (!ls) return s;
            auto *inner = llvm::dyn_cast_or_null< clang::CompoundStmt >(ls->getSubStmt());
            if (!inner || inner->body_empty()) return s;

            auto it = inner->body_begin();
            ls->setSubStmt(*it);
            std::vector< clang::Stmt * > stmts;
            stmts.push_back(ls);
            for (++it; it != inner->body_end(); ++it)
                stmts.push_back(*it);
            return detail::MakeCompound(ctx, stmts);
        }

        // Replace a trailing GotoStmt in a case body with break or continue.
        // Returns the modified stmt, or the original if no replacement was made.
        clang::Stmt *ReplaceTrailingGoto(clang::ASTContext &ctx, clang::Stmt *s,
                                          const std::string &break_label,
                                          const std::string &continue_label) {
            if (!s) return s;

            // Direct GotoStmt
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
                std::string name = gs->getLabel()->getName().str();
                if (!break_label.empty() && name == break_label)
                    return new (ctx) clang::BreakStmt(VirtualLoc(ctx));
                if (!continue_label.empty() && name == continue_label)
                    return new (ctx) clang::ContinueStmt(VirtualLoc(ctx));
                return s;
            }

            // CompoundStmt — check/replace last stmt
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                if (cs->body_empty()) return s;
                auto *last = *(cs->body_end() - 1);
                auto *replaced = ReplaceTrailingGoto(ctx, last, break_label, continue_label);
                if (replaced == last) return s;

                std::vector< clang::Stmt * > stmts;
                for (auto it = cs->body_begin(); std::next(it) != cs->body_end(); ++it)
                    stmts.push_back(*it);
                stmts.push_back(replaced);
                return detail::MakeCompound(ctx, stmts);
            }

            return s;
        }

        // Walk case/default bodies in a SwitchStmt and convert trailing gotos
        // to break (if targeting break_label) or continue (if targeting continue_label).
        void ConvertSwitchCaseGotos(clang::ASTContext &ctx, clang::SwitchStmt *sw,
                                     const std::string &break_label,
                                     const std::string &continue_label) {
            auto *body = sw->getBody();
            auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(body);
            if (!cs) return;

            for (auto *child : cs->body()) {
                if (auto *case_s = llvm::dyn_cast< clang::CaseStmt >(child)) {
                    auto *sub = case_s->getSubStmt();
                    auto *r = ReplaceTrailingGoto(ctx, sub, break_label, continue_label);
                    if (r != sub) case_s->setSubStmt(r);
                } else if (auto *def_s = llvm::dyn_cast< clang::DefaultStmt >(child)) {
                    auto *sub = def_s->getSubStmt();
                    auto *r = ReplaceTrailingGoto(ctx, sub, break_label, continue_label);
                    if (r != sub) def_s->setSubStmt(r);
                }
            }
        }

        // Check if ALL case/default bodies in a switch end with goto to the same
        // label (and that label is NOT the break/continue label). Returns the
        // common label name, or empty string if not uniform.
        std::string FindCommonTrailingGoto(clang::SwitchStmt *sw) {
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
        clang::Stmt *CleanupStmtTree(clang::ASTContext &ctx, clang::Stmt *s,
                                      const std::string &continue_label = "") {
            if (!s) return nullptr;

            // Handle IfStmt: recurse into then/else, push labels inside
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
                ifs->setThen(PushLabelInside(ctx,
                    CleanupStmtTree(ctx, ifs->getThen(), continue_label)));
                if (ifs->getElse())
                    ifs->setElse(PushLabelInside(ctx,
                        CleanupStmtTree(ctx, ifs->getElse(), continue_label)));
                return s;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
                ws->setBody(PushLabelInside(ctx,
                    CleanupStmtTree(ctx, ws->getBody(), continue_label)));
                return s;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(s)) {
                ds->setBody(PushLabelInside(ctx,
                    CleanupStmtTree(ctx, ds->getBody(), continue_label)));
                return s;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
                fs->setBody(PushLabelInside(ctx,
                    CleanupStmtTree(ctx, fs->getBody(), continue_label)));
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
                ls->setSubStmt(CleanupStmtTree(ctx, sub,
                    new_cont.empty() ? continue_label : new_cont));
                return s;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(s)) {
                sw->setBody(CleanupStmtTree(ctx, sw->getBody(), continue_label));
                return s;
            }
            if (auto *cs_node = llvm::dyn_cast< clang::CaseStmt >(s)) {
                cs_node->setSubStmt(CleanupStmtTree(ctx, cs_node->getSubStmt(), continue_label));
                return s;
            }
            if (auto *def = llvm::dyn_cast< clang::DefaultStmt >(s)) {
                def->setSubStmt(CleanupStmtTree(ctx, def->getSubStmt(), continue_label));
                return s;
            }

            // CompoundStmt: recurse, flatten nested compounds, push labels inside,
            // then convert switch case gotos to break/continue.
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                // --- First pass: recurse, flatten, push labels ---
                std::vector< clang::Stmt * > children;
                for (auto *child : cs->body()) {
                    auto *cleaned = CleanupStmtTree(ctx, child, continue_label);
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
                        ConvertSwitchCaseGotos(ctx, sw, break_label, continue_label);
                    }

                    // Hoist: if ALL cases goto the same label (not break/continue
                    // target), replace with breaks and add goto after switch.
                    if (break_label.empty()) {
                        std::string common = FindCommonTrailingGoto(sw);
                        if (!common.empty() && common != continue_label) {
                            // Replace all trailing gotos with break
                            ConvertSwitchCaseGotos(ctx, sw, common, "");
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
                                auto loc = VirtualLoc(ctx);
                                auto *hoisted_goto = new (ctx) clang::GotoStmt(
                                    target_decl, loc, loc);
                                children.insert(children.begin() + static_cast< long >(i) + 1,
                                                hoisted_goto);
                                ++i; // skip the inserted goto
                            }
                        }
                    }
                }

                return detail::MakeCompound(ctx, children);
            }

            return s;
        }
    } // namespace

    // Remove LabelStmts that are not the target of any GotoStmt.
    // Replaces dead LabelStmt with its sub-statement.
    static clang::Stmt *RemoveDeadLabels(clang::ASTContext &ctx, clang::Stmt *s,
                                          const std::unordered_set< clang::LabelDecl * > &live) {
        if (!s) return nullptr;

        // Guarantee a non-null Stmt* for set* methods that require one.
        auto safe = [&](clang::Stmt *r) -> clang::Stmt * {
            return r ? r : new (ctx) clang::NullStmt(VirtualLoc(ctx));
        };

        if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
            auto *sub = RemoveDeadLabels(ctx, ls->getSubStmt(), live);
            if (!live.count(ls->getDecl())) {
                return sub;
            }
            ls->setSubStmt(safe(sub));
            return ls;
        }

        if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
            std::vector< clang::Stmt * > children;
            for (auto *child : cs->body()) {
                auto *cleaned = RemoveDeadLabels(ctx, child, live);
                if (cleaned) children.push_back(cleaned);
            }

            // Drop unreachable children after a terminator.  When
            // RemoveDeadLabels strips a dead label, the body remains as
            // a bare stmt.  If it follows a return/goto/break/continue
            // and contains no live labels, it is unreachable and can
            // be removed.
            auto is_terminator = [](clang::Stmt *st) -> bool {
                return detail::EndsWithTerminator(st);
            };

            for (size_t i = 1; i < children.size();) {
                if (!is_terminator(children[i - 1])) {
                    ++i;
                    continue;
                }
                bool has_live_label                        = false;
                std::function< void(clang::Stmt *) > check = [&](clang::Stmt *st) {
                    if (!st || has_live_label) {
                        return;
                    }
                    if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st)) {
                        if (live.count(ls->getDecl())) {
                            has_live_label = true;
                        }
                    }
                    for (auto *c : st->children()) {
                        check(c);
                    }
                };
                check(children[i]);
                if (has_live_label) {
                    ++i;
                    continue;
                }
                children.erase(children.begin() + static_cast< ptrdiff_t >(i));
            }

            return detail::MakeCompound(ctx, children);
        }

        if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
            ifs->setThen(safe(RemoveDeadLabels(ctx, ifs->getThen(), live)));
            if (ifs->getElse())
                ifs->setElse(safe(RemoveDeadLabels(ctx, ifs->getElse(), live)));
            return s;
        }
        if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
            ws->setBody(safe(RemoveDeadLabels(ctx, ws->getBody(), live)));
            return s;
        }
        if (auto *ds = llvm::dyn_cast< clang::DoStmt >(s)) {
            ds->setBody(safe(RemoveDeadLabels(ctx, ds->getBody(), live)));
            return s;
        }
        if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
            fs->setBody(safe(RemoveDeadLabels(ctx, fs->getBody(), live)));
            return s;
        }
        if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(s)) {
            sw->setBody(safe(RemoveDeadLabels(ctx, sw->getBody(), live)));
            return s;
        }
        if (auto *cs_node = llvm::dyn_cast< clang::CaseStmt >(s)) {
            cs_node->setSubStmt(safe(RemoveDeadLabels(ctx, cs_node->getSubStmt(), live)));
            return s;
        }
        if (auto *def = llvm::dyn_cast< clang::DefaultStmt >(s)) {
            def->setSubStmt(safe(RemoveDeadLabels(ctx, def->getSubStmt(), live)));
            return s;
        }

        return s;
    }

    // Helper: extract if(cond) goto L pattern. Returns {label, cond} or nulls.
    static clang::Stmt *UnwrapSingleCompoundStmt(clang::Stmt *s) {
        auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(s);
        if (!cs || cs->size() != 1) {
            return s;
        }
        return cs->body_front();
    }

    static clang::GotoStmt *AsGotoStmt(clang::Stmt *s) {
        return llvm::dyn_cast_or_null< clang::GotoStmt >(
            UnwrapSingleCompoundStmt(s));
    }

    static std::pair< clang::LabelDecl *, clang::Expr * >
    ExtractIfGotoPattern(clang::Stmt *s) {
        auto *ifs = llvm::dyn_cast_or_null< clang::IfStmt >(s);
        if (!ifs || ifs->getElse()) {
            return { nullptr, nullptr };
        }
        auto *gs = AsGotoStmt(ifs->getThen());
        if (!gs) {
            return { nullptr, nullptr };
        }
        return { gs->getLabel(), ifs->getCond() };
    }

    // Recursively remove empty CompoundStmts, NullStmts, and merge
    // consecutive if(c1) goto L; if(c2) goto L; into if(c1||c2) goto L;
    static clang::Stmt *RemoveEmptyBlocks(clang::ASTContext &ctx, clang::Stmt *s) {
        if (!s) {
            return nullptr;
        }

        auto safe = [&](clang::Stmt *r) -> clang::Stmt * {
            return r ? r : new (ctx) clang::NullStmt(VirtualLoc(ctx));
        };

        if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
            std::vector< clang::Stmt * > children;
            for (auto *child : cs->body()) {
                auto *cleaned = RemoveEmptyBlocks(ctx, child);
                if (!cleaned) {
                    continue;
                }
                if (llvm::isa< clang::NullStmt >(cleaned)) {
                    continue;
                }
                if (auto *inner = llvm::dyn_cast< clang::CompoundStmt >(cleaned)) {
                    if (inner->body_empty()) {
                        continue;
                    }
                }
                children.push_back(cleaned);
            }

            // Merge consecutive if(c1) goto L; if(c2) goto L;
            {
                size_t i = 0;
                while (i + 1 < children.size()) {
                    auto [l1, c1] = ExtractIfGotoPattern(children[i]);
                    if (!l1) {
                        ++i;
                        continue;
                    }
                    auto [l2, c2] = ExtractIfGotoPattern(children[i + 1]);
                    if (!l2 || l1->getName() != l2->getName()) {
                        ++i;
                        continue;
                    }

                    auto *merged = clang::BinaryOperator::Create(
                        ctx, EnsureRValue(ctx, c1), EnsureRValue(ctx, c2),
                        clang::BO_LOr, ctx.BoolTy, clang::VK_PRValue,
                        clang::OK_Ordinary, VirtualLoc(ctx),
                        clang::FPOptionsOverride()
                    );
                    llvm::cast< clang::IfStmt >(children[i])->setCond(merged);
                    children.erase(children.begin() + static_cast< long >(i) + 1);
                    // Don't advance i — re-check for third consecutive
                }
            }

            return detail::MakeCompound(ctx, children);
        }
        if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
            ifs->setThen(safe(RemoveEmptyBlocks(ctx, ifs->getThen())));
            if (ifs->getElse()) {
                auto *new_else = safe(RemoveEmptyBlocks(ctx, ifs->getElse()));
                // Drop an else clause that cleaned up to nothing — the
                // pretty-printer would otherwise render `else { }`.
                auto *else_cs = llvm::dyn_cast< clang::CompoundStmt >(new_else);
                bool else_empty = llvm::isa< clang::NullStmt >(new_else)
                    || (else_cs && else_cs->body_empty());
                if (else_empty) {
                    return clang::IfStmt::Create(
                        ctx, ifs->getIfLoc(), ifs->getStatementKind(),
                        ifs->getInit(), ifs->getConditionVariable(),
                        ifs->getCond(), ifs->getLParenLoc(),
                        ifs->getRParenLoc(), ifs->getThen());
                }
                if (auto *inner = llvm::dyn_cast_or_null< clang::IfStmt >(
                        UnwrapSingleCompoundStmt(new_else))) {
                    auto *outer_goto = AsGotoStmt(ifs->getThen());
                    auto *inner_goto = AsGotoStmt(inner->getThen());
                    if (!ifs->getInit() && !ifs->getConditionVariable()
                        && !inner->getInit()
                        && !inner->getConditionVariable()
                        && outer_goto && inner_goto
                        && outer_goto->getLabel()->getName()
                               == inner_goto->getLabel()->getName()) {
                        auto *merged = clang::BinaryOperator::Create(
                            ctx, EnsureRValue(ctx, ifs->getCond()),
                            EnsureRValue(ctx, inner->getCond()),
                            clang::BO_LOr, ctx.BoolTy, clang::VK_PRValue,
                            clang::OK_Ordinary, VirtualLoc(ctx),
                            clang::FPOptionsOverride());
                        return clang::IfStmt::Create(
                            ctx, ifs->getIfLoc(), ifs->getStatementKind(),
                            nullptr, nullptr, merged, ifs->getLParenLoc(),
                            ifs->getRParenLoc(), ifs->getThen(),
                            inner->getElseLoc(), inner->getElse());
                    }
                }
                ifs->setElse(new_else);
            }
            return s;
        }
        if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
            ws->setBody(safe(RemoveEmptyBlocks(ctx, ws->getBody())));
            return s;
        }
        if (auto *ds = llvm::dyn_cast< clang::DoStmt >(s)) {
            ds->setBody(safe(RemoveEmptyBlocks(ctx, ds->getBody())));
            return s;
        }
        if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
            fs->setBody(safe(RemoveEmptyBlocks(ctx, fs->getBody())));
            return s;
        }
        if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
            ls->setSubStmt(safe(RemoveEmptyBlocks(ctx, ls->getSubStmt())));
            return s;
        }
        if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(s)) {
            sw->setBody(safe(RemoveEmptyBlocks(ctx, sw->getBody())));
            return s;
        }
        if (auto *cs_node = llvm::dyn_cast< clang::CaseStmt >(s)) {
            cs_node->setSubStmt(
                safe(RemoveEmptyBlocks(ctx, cs_node->getSubStmt())));
            return s;
        }
        if (auto *def = llvm::dyn_cast< clang::DefaultStmt >(s)) {
            def->setSubStmt(
                safe(RemoveEmptyBlocks(ctx, def->getSubStmt())));
            return s;
        }
        return s;
    }

    // ---------------------------------------------------------------
    // EliminateGotoToNextLabel — recursively drop gotos whose target
    // is the immediately following LabelStmt in the same CompoundStmt.
    //
    // The goto may be deeply nested: inside an IfStmt else-arm, inside
    // a LabelStmt body, inside a CompoundStmt.  We chase through
    // nesting to find the "deepest trailing stmt" and check if it's a
    // goto to the next sibling label.
    // ---------------------------------------------------------------

    namespace {

        /// Get label name from a LabelStmt, empty otherwise.
        llvm::StringRef GotoElimGetLabel(clang::Stmt *s) {
            if (auto *ls = llvm::dyn_cast_or_null< clang::LabelStmt >(s)) {
                return ls->getDecl()->getName();
            }
            if (auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(s)) {
                if (!cs->body_empty())
                    return GotoElimGetLabel(*cs->body_begin());
            }
            return {};
        }

        /// Get goto target name, empty if not a GotoStmt.
        llvm::StringRef GotoElimGetTarget(clang::Stmt *s) {
            if (auto *gs = llvm::dyn_cast_or_null< clang::GotoStmt >(s)) {
                return gs->getLabel()->getName();
            }
            return {};
        }

        /// Recursively find the deepest trailing stmt — chasing through
        /// CompoundStmt (last child) and LabelStmt (sub-stmt).
        /// IfStmts and other nodes are returned as-is so the caller
        /// can inspect their arms via IfStmtGotoArm.
        clang::Stmt *DeepTrailingStmt(clang::Stmt *s) {
            if (!s) {
                return nullptr;
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                if (cs->body_empty()) {
                    return nullptr;
                }
                return DeepTrailingStmt(*(cs->body_end() - 1));
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
                return DeepTrailingStmt(ls->getSubStmt());
            }
            return s; // leaf: GotoStmt, IfStmt, etc.
        }

        /// Returns 0/1/2 = no match / else arm / then arm.  Arm 1
        /// matches the else's deepest trailing goto (handler strips
        /// just that goto).  Arm 2 only matches when then IS the
        /// goto — the handler discards the whole then arm, so a deep
        /// match here would drop preceding stmts.
        int IfStmtGotoArm(clang::IfStmt *ifs, llvm::StringRef target) {
            if (!ifs) {
                return 0;
            }
            auto et = GotoElimGetTarget(DeepTrailingStmt(ifs->getElse()));
            if (!et.empty() && et == target) {
                return 1;
            }
            auto tt = GotoElimGetTarget(ifs->getThen());
            if (!tt.empty() && tt == target && ifs->getElse()) {
                return 2;
            }
            return 0;
        }

        /// Strip the trailing `goto target` from `st`, walking through
        /// the last child of nested CompoundStmts and LabelStmt
        /// sub-stmts.  Returns NullStmt when the stripped stmt itself
        /// is the matching goto.
        clang::Stmt *StripTrailingGoto(
            clang::ASTContext &ctx, clang::Stmt *st, llvm::StringRef target
        ) {
            if (!st) {
                return st;
            }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(st)) {
                if (gs->getLabel()->getName() == target) {
                    return new (ctx) clang::NullStmt(VirtualLoc(ctx));
                }
                return st;
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                if (cs->body_empty()) {
                    return st;
                }
                std::vector< clang::Stmt * > b(cs->body_begin(), cs->body_end());
                auto *last_stripped = StripTrailingGoto(ctx, b.back(), target);
                if (llvm::isa< clang::NullStmt >(last_stripped)
                    && llvm::isa< clang::GotoStmt >(b.back()))
                {
                    b.pop_back();
                } else {
                    b.back() = last_stripped;
                }
                return detail::MakeCompound(ctx, b);
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st)) {
                ls->setSubStmt(StripTrailingGoto(ctx, ls->getSubStmt(), target));
                return st;
            }
            return st;
        }

        clang::Stmt *StripGotoToFollowingLabelFromTailPosition(
            clang::ASTContext &ctx, clang::Stmt *st, llvm::StringRef target,
            unsigned &stripped
        ) {
            if (!st) {
                return st;
            }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(st)) {
                if (gs->getLabel()->getName() == target) {
                    ++stripped;
                    return new (ctx) clang::NullStmt(VirtualLoc(ctx));
                }
                return st;
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                if (cs->body_empty()) {
                    return st;
                }
                std::vector< clang::Stmt * > b(cs->body_begin(), cs->body_end());
                unsigned before = stripped;
                b.back() = StripGotoToFollowingLabelFromTailPosition(
                    ctx, b.back(), target, stripped);
                return stripped != before ? detail::MakeCompound(ctx, b) : st;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st)) {
                ls->setSubStmt(StripGotoToFollowingLabelFromTailPosition(
                    ctx, ls->getSubStmt(), target, stripped));
                return st;
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(st)) {
                unsigned before = stripped;
                ifs->setThen(StripGotoToFollowingLabelFromTailPosition(
                    ctx, ifs->getThen(), target, stripped));
                if (ifs->getElse()) {
                    ifs->setElse(StripGotoToFollowingLabelFromTailPosition(
                        ctx, ifs->getElse(), target, stripped));
                }
                return stripped != before ? ifs : st;
            }
            return st;
        }

        bool ContinuationIsSafeToMoveBeforeJoin(clang::Stmt *st) {
            if (!st) {
                return true;
            }
            if (llvm::isa< clang::LabelStmt >(st)
                || llvm::isa< clang::SwitchStmt >(st)
                || llvm::isa< clang::WhileStmt >(st)
                || llvm::isa< clang::DoStmt >(st)
                || llvm::isa< clang::ForStmt >(st)
                || llvm::isa< clang::DeclStmt >(st)
                || llvm::isa< clang::ReturnStmt >(st)
                || llvm::isa< clang::BreakStmt >(st)
                || llvm::isa< clang::ContinueStmt >(st)) {
                return false;
            }
            for (clang::Stmt *child : st->children()) {
                if (!ContinuationIsSafeToMoveBeforeJoin(child)) {
                    return false;
                }
            }
            return true;
        }

        unsigned CountGotosToTargetName(clang::Stmt *st, llvm::StringRef target) {
            if (!st) {
                return 0;
            }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(st)) {
                return gs->getLabel()->getName() == target ? 1U : 0U;
            }
            unsigned count = 0;
            for (clang::Stmt *child : st->children()) {
                count += CountGotosToTargetName(child, target);
            }
            return count;
        }

        unsigned CountTailFallthroughLeaves(clang::Stmt *st, llvm::StringRef skip_target) {
            if (!st) {
                return 1;
            }
            if (llvm::isa< clang::GotoStmt >(st)) {
                (void) skip_target;
                return 0;
            }
            if (llvm::isa< clang::ReturnStmt >(st)
                || llvm::isa< clang::BreakStmt >(st)
                || llvm::isa< clang::ContinueStmt >(st)
                || llvm::isa< clang::SwitchStmt >(st)
                || llvm::isa< clang::WhileStmt >(st)
                || llvm::isa< clang::DoStmt >(st)
                || llvm::isa< clang::ForStmt >(st)) {
                return 0;
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                if (cs->body_empty()) {
                    return 1;
                }
                return CountTailFallthroughLeaves(cs->body_back(), skip_target);
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st)) {
                return CountTailFallthroughLeaves(ls->getSubStmt(), skip_target);
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(st)) {
                unsigned count = CountTailFallthroughLeaves(ifs->getThen(), skip_target);
                count += ifs->getElse()
                    ? CountTailFallthroughLeaves(ifs->getElse(), skip_target)
                    : 1U;
                return count;
            }
            return 1;
        }

        clang::Stmt *MoveContinuationIntoSingleFallthrough(
            clang::ASTContext &ctx, clang::Stmt *st, llvm::StringRef skip_target,
            clang::Stmt *continuation, unsigned &removed_gotos,
            unsigned &inserted
        ) {
            if (!st) {
                ++inserted;
                return continuation;
            }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(st)) {
                if (gs->getLabel()->getName() == skip_target) {
                    ++removed_gotos;
                    return new (ctx) clang::NullStmt(VirtualLoc(ctx));
                }
                return st;
            }
            if (llvm::isa< clang::ReturnStmt >(st)
                || llvm::isa< clang::BreakStmt >(st)
                || llvm::isa< clang::ContinueStmt >(st)
                || llvm::isa< clang::SwitchStmt >(st)
                || llvm::isa< clang::WhileStmt >(st)
                || llvm::isa< clang::DoStmt >(st)
                || llvm::isa< clang::ForStmt >(st)) {
                return st;
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                if (cs->body_empty()) {
                    ++inserted;
                    return continuation;
                }
                std::vector< clang::Stmt * > b(cs->body_begin(), cs->body_end());
                b.back() = MoveContinuationIntoSingleFallthrough(
                    ctx, b.back(), skip_target, continuation, removed_gotos,
                    inserted);
                return detail::MakeCompound(ctx, b);
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st)) {
                ls->setSubStmt(MoveContinuationIntoSingleFallthrough(
                    ctx, ls->getSubStmt(), skip_target, continuation,
                    removed_gotos, inserted));
                return ls;
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(st)) {
                clang::Stmt *new_then = MoveContinuationIntoSingleFallthrough(
                    ctx, ifs->getThen(), skip_target, continuation,
                    removed_gotos, inserted);
                if (ifs->getElse()) {
                    ifs->setThen(new_then);
                    ifs->setElse(MoveContinuationIntoSingleFallthrough(
                        ctx, ifs->getElse(), skip_target, continuation,
                        removed_gotos, inserted));
                    return ifs;
                }

                ++inserted;
                return clang::IfStmt::Create(
                    ctx, ifs->getIfLoc(), ifs->getStatementKind(),
                    ifs->getInit(), ifs->getConditionVariable(),
                    ifs->getCond(), ifs->getLParenLoc(), ifs->getRParenLoc(),
                    new_then, ifs->getIfLoc(), continuation);
            }

            ++inserted;
            return detail::MakeCompound(ctx, {st, continuation});
        }

        clang::Stmt *UnwrapDeadLeadingLabelForMove(
            clang::ASTContext &ctx, clang::Stmt *st,
            const std::unordered_set< clang::LabelDecl * > *live
        ) {
            if (!st || !live) {
                return st;
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(st)) {
                return live->count(label->getDecl()) ? st : label->getSubStmt();
            }
            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(st);
            if (!compound || compound->body_empty()) {
                return st;
            }
            auto it = compound->body_begin();
            auto *label = llvm::dyn_cast< clang::LabelStmt >(*it);
            if (!label || live->count(label->getDecl())) {
                return st;
            }

            std::vector< clang::Stmt * > unwrapped;
            unwrapped.push_back(label->getSubStmt());
            for (++it; it != compound->body_end(); ++it) {
                unwrapped.push_back(*it);
            }
            return detail::MakeCompound(ctx, unwrapped);
        }

        bool HasDeadLeadingLabelForMove(
            clang::Stmt *st, const std::unordered_set< clang::LabelDecl * > *live
        ) {
            if (!st || !live) {
                return false;
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(st)) {
                return !live->count(label->getDecl());
            }
            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(st);
            if (!compound || compound->body_empty()) {
                return false;
            }
            auto *label = llvm::dyn_cast< clang::LabelStmt >(compound->body_front());
            return label && !live->count(label->getDecl());
        }

        bool TryMoveTailContinuationIntoDispatch(
            clang::ASTContext &ctx, std::vector< clang::Stmt * > &body,
            llvm::StringRef next_label,
            const std::unordered_set< clang::LabelDecl * > *live
        ) {
            if (body.size() < 2) {
                return false;
            }
            size_t continuation_idx = body.size() - 1;
            size_t dispatch_idx     = body.size() - 2;
            if (!HasDeadLeadingLabelForMove(body[continuation_idx], live)) {
                return false;
            }
            clang::Stmt *continuation =
                UnwrapDeadLeadingLabelForMove(ctx, body[continuation_idx], live);
            if (!GotoElimGetLabel(continuation).empty()
                || !ContinuationIsSafeToMoveBeforeJoin(continuation)
                || CountGotosToTargetName(body[dispatch_idx], next_label) == 0
                || CountTailFallthroughLeaves(body[dispatch_idx], next_label) != 1) {
                return false;
            }

            unsigned removed_gotos = 0;
            unsigned inserted      = 0;
            auto *rewritten = MoveContinuationIntoSingleFallthrough(
                ctx, body[dispatch_idx], next_label, continuation, removed_gotos,
                inserted);
            if (removed_gotos == 0 || inserted != 1) {
                return false;
            }

            body[dispatch_idx] = rewritten;
            body.erase(body.begin() + static_cast< ptrdiff_t >(continuation_idx));
            return true;
        }

        clang::Stmt *MoveTailContinuationBeforeFollowingLabel(
            clang::ASTContext &ctx, clang::Stmt *st, llvm::StringRef next_label,
            const std::unordered_set< clang::LabelDecl * > *live, unsigned &moved
        ) {
            if (!st) {
                return st;
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                std::vector< clang::Stmt * > b(cs->body_begin(), cs->body_end());
                if (TryMoveTailContinuationIntoDispatch(ctx, b, next_label, live)) {
                    ++moved;
                    return detail::MakeCompound(ctx, b);
                }
                if (!b.empty()) {
                    unsigned before = moved;
                    b.back() = MoveTailContinuationBeforeFollowingLabel(
                        ctx, b.back(), next_label, live, moved);
                    if (moved != before) {
                        return detail::MakeCompound(ctx, b);
                    }
                }
                return st;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st)) {
                ls->setSubStmt(MoveTailContinuationBeforeFollowingLabel(
                    ctx, ls->getSubStmt(), next_label, live, moved));
                return st;
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(st)) {
                unsigned before = moved;
                ifs->setThen(MoveTailContinuationBeforeFollowingLabel(
                    ctx, ifs->getThen(), next_label, live, moved));
                if (ifs->getElse()) {
                    ifs->setElse(MoveTailContinuationBeforeFollowingLabel(
                        ctx, ifs->getElse(), next_label, live, moved));
                }
                return moved != before ? ifs : st;
            }
            return st;
        }

        clang::Stmt *ReplaceTrailingDeepStmt(
            clang::ASTContext &ctx,
            clang::Stmt *st,
            clang::Stmt *target,
            clang::Stmt *replacement
        ) {
            if (!st || st == target)
                return st == target ? replacement : st;
            if (auto *inner = llvm::dyn_cast<clang::CompoundStmt>(st)) {
                if (inner->body_empty())
                    return st;
                std::vector<clang::Stmt *> b(
                    inner->body_begin(), inner->body_end());
                if (DeepTrailingStmt(b.back()) == target)
                    b.back() = ReplaceTrailingDeepStmt(
                        ctx, b.back(), target, replacement);
                return detail::MakeCompound(ctx, b);
            }
            if (auto *ls = llvm::dyn_cast<clang::LabelStmt>(st)) {
                ls->setSubStmt(ReplaceTrailingDeepStmt(
                    ctx, ls->getSubStmt(), target, replacement));
                return ls;
            }
            if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(st)) {
                if (DeepTrailingStmt(ifs->getThen()) == target) {
                    ifs->setThen(ReplaceTrailingDeepStmt(
                        ctx, ifs->getThen(), target, replacement));
                    return ifs;
                }
                if (ifs->getElse()
                    && DeepTrailingStmt(ifs->getElse()) == target) {
                    ifs->setElse(ReplaceTrailingDeepStmt(
                        ctx, ifs->getElse(), target, replacement));
                    return ifs;
                }
            }
            return st;
        }

        clang::IfStmt *BuildIfGotoArmReplacement(
            clang::ASTContext &ctx,
            clang::IfStmt *ifs,
            llvm::StringRef target
        ) {
            int arm = IfStmtGotoArm(ifs, target);
            if (arm == 0)
                return nullptr;

            auto loc = ifs->getIfLoc();
            if (arm == 1) {
                auto *new_else =
                    StripTrailingGoto(ctx, ifs->getElse(), target);
                if (llvm::isa<clang::NullStmt>(new_else))
                    new_else = nullptr;
                return clang::IfStmt::Create(
                    ctx, loc, clang::IfStatementKind::Ordinary,
                    nullptr, nullptr, ifs->getCond(), loc, loc,
                    ifs->getThen(), loc, new_else);
            }

            auto *neg = NegateExpr(ctx, ifs->getCond());
            return clang::IfStmt::Create(
                ctx, loc, clang::IfStatementKind::Ordinary, nullptr,
                nullptr, neg, loc, loc, ifs->getElse(), loc, nullptr);
        }

        /// Process a CompoundStmt: for each pair of adjacent stmts where
        /// the second is a LabelStmt, check if the first's deepest trailing
        /// stmt is a goto to that label.  Returns new stmt if changed.
        clang::Stmt *EliminateGotoToNextLabel(
            clang::ASTContext &ctx, clang::Stmt *s,
            const std::unordered_set< clang::LabelDecl * > *live = nullptr);

        clang::Stmt *ProcessCompound(
            clang::ASTContext &ctx, clang::CompoundStmt *cs,
            const std::unordered_set< clang::LabelDecl * > *live) {
            std::vector< clang::Stmt * > body(cs->body_begin(), cs->body_end());

            // First: recurse into all children
            for (auto *&child : body) {
                child = EliminateGotoToNextLabel(ctx, child, live);
            }

            // Then: find goto-to-next-label patterns
            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t i = 0; i + 1 < body.size(); ++i) {
                    size_t next_idx = i + 1;
                    while (next_idx < body.size()
                           && llvm::isa< clang::NullStmt >(body[next_idx]))
                        ++next_idx;
                    if (next_idx >= body.size())
                        continue;

                    auto next_label = GotoElimGetLabel(body[next_idx]);
                    if (next_label.empty()) {
                        continue;
                    }

                    // Find the deepest trailing stmt of body[i]
                    auto *deep = DeepTrailingStmt(body[i]);
                    if (!deep) {
                        continue;
                    }

                    // Pattern 1: deepest trailing is goto L; next is L:
                    auto tgt = GotoElimGetTarget(deep);
                    if (!tgt.empty() && tgt == next_label) {
                        // Simple case: body[i] IS the goto
                        if (deep == body[i]) {
                            body.erase(body.begin() + static_cast< ptrdiff_t >(i));
                            changed = true;
                            break;
                        }
                        // Otherwise: rebuild without the trailing goto.
                        std::function< clang::Stmt *(clang::Stmt *) > strip_tail;
                        strip_tail = [&](clang::Stmt *st) -> clang::Stmt * {
                            if (auto *inner = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                                if (inner->body_empty()) {
                                    return st;
                                }
                                auto *last = *(inner->body_end() - 1);
                                if (last == deep) {
                                    std::vector< clang::Stmt * > b(
                                        inner->body_begin(), inner->body_end() - 1
                                    );
                                    if (b.empty()) {
                                        return new (ctx)
                                            clang::NullStmt(VirtualLoc(ctx));
                                    }
                                    return detail::MakeCompound(ctx, b);
                                }
                                std::vector< clang::Stmt * > b(
                                    inner->body_begin(), inner->body_end()
                                );
                                b.back() = strip_tail(b.back());
                                return detail::MakeCompound(ctx, b);
                            }
                            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st)) {
                                ls->setSubStmt(strip_tail(ls->getSubStmt()));
                                return st;
                            }
                            // Reached the goto itself — replace with NullStmt
                            return new (ctx) clang::NullStmt(VirtualLoc(ctx));
                        };
                        body[i] = strip_tail(body[i]);
                        changed = true;
                        break;
                    }

                    // Pattern 2: deepest trailing is IfStmt with goto arm.
                    // Arm 1: strip trailing goto from else; drop the else
                    //   if it collapses to NullStmt.
                    // Arm 2: drop the then arm and flip to `if(!c) else`
                    //   (safe because IfStmtGotoArm requires then to be
                    //   exactly the goto — see its docstring).
                    if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(deep)) {
                        clang::IfStmt *new_if =
                            BuildIfGotoArmReplacement(ctx, ifs, next_label);
                        if (new_if) {
                            if (deep == body[i]) {
                                body[i] = new_if;
                            } else {
                                body[i] = ReplaceTrailingDeepStmt(
                                    ctx, body[i], deep, new_if);
                            }
                            changed = true;
                            break;
                        }

                        if (!ifs->getElse()) {
                            if (auto *nested_if =
                                    llvm::dyn_cast_or_null<clang::IfStmt>(
                                        DeepTrailingStmt(ifs->getThen()))) {
                                clang::IfStmt *new_nested =
                                    BuildIfGotoArmReplacement(
                                        ctx, nested_if, next_label);
                                if (new_nested) {
                                    body[i] = ReplaceTrailingDeepStmt(
                                        ctx, body[i], nested_if, new_nested);
                                    changed = true;
                                    break;
                                }
                            }
                        }
                    }

                    unsigned moved = 0;
                    auto *moved_tail = MoveTailContinuationBeforeFollowingLabel(
                        ctx, body[i], next_label, live, moved);
                    if (moved != 0) {
                        body[i] = moved_tail;
                        changed = true;
                        break;
                    }

                    unsigned stripped = 0;
                    auto *rewritten = StripGotoToFollowingLabelFromTailPosition(
                        ctx, body[i], next_label, stripped);
                    if (stripped != 0) {
                        body[i] = rewritten;
                        changed = true;
                        break;
                    }
                }
            }

            // Pattern: a conditional dispatch skips the immediately following
            // continuation by jumping to the next label:
            //
            //   if (...) goto Join; else if (...) ; else goto Join;
            //   if (side) goto Side;
            // Join:
            //
            // Move the continuation into the dispatch's single fallthrough
            // path, delete the skip-gotos, and let RemoveDeadLabels erase
            // Join if no other gotos remain.
            for (size_t label_idx = 2; label_idx < body.size(); ++label_idx) {
                auto next_label = GotoElimGetLabel(body[label_idx]);
                if (next_label.empty()) {
                    continue;
                }
                size_t continuation_idx = label_idx - 1;
                size_t dispatch_idx     = label_idx - 2;
                (void) continuation_idx;
                (void) dispatch_idx;
                std::vector< clang::Stmt * > tail_pair = {
                    body[label_idx - 2],
                    body[label_idx - 1],
                };
                if (TryMoveTailContinuationIntoDispatch(
                        ctx, tail_pair, next_label, live)) {
                    body[label_idx - 2] = tail_pair.front();
                    body.erase(
                        body.begin()
                        + static_cast< ptrdiff_t >(label_idx - 1));
                    changed = true;
                    break;
                }
            }

            // Pattern: switch case goto L; } L: → replace goto with break.
            // When a switch stmt is followed by a LabelStmt, any case
            // body ending in goto-to-that-label can use break instead.
            for (size_t i = 0; i + 1 < body.size(); ++i) {
                auto *sw = llvm::dyn_cast< clang::SwitchStmt >(body[i]);
                if (!sw) {
                    continue;
                }
                auto next_label = GotoElimGetLabel(body[i + 1]);
                if (next_label.empty()) {
                    continue;
                }

                auto *sw_body = llvm::dyn_cast_or_null< clang::CompoundStmt >(sw->getBody());
                if (!sw_body) {
                    continue;
                }

                std::vector< clang::Stmt * > sw_stmts(
                    sw_body->body_begin(), sw_body->body_end()
                );
                bool sw_changed = false;

                std::function< clang::Stmt *(clang::Stmt *) > replace_goto_break;
                replace_goto_break = [&](clang::Stmt *st) -> clang::Stmt * {
                    if (!st) {
                        return st;
                    }
                    if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(st)) {
                        if (gs->getLabel()->getName() == next_label) {
                            sw_changed = true;
                            return new (ctx) clang::BreakStmt(VirtualLoc(ctx));
                        }
                        return st;
                    }
                    if (auto *cs2 = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                        if (cs2->body_empty()) {
                            return st;
                        }
                        std::vector< clang::Stmt * > cb(cs2->body_begin(), cs2->body_end());
                        cb.back() = replace_goto_break(cb.back());
                        return detail::MakeCompound(ctx, cb);
                    }
                    return st;
                };

                for (auto *&case_stmt : sw_stmts) {
                    if (auto *cs2 = llvm::dyn_cast< clang::CaseStmt >(case_stmt)) {
                        cs2->setSubStmt(replace_goto_break(cs2->getSubStmt()));
                    } else if (auto *ds = llvm::dyn_cast< clang::DefaultStmt >(case_stmt)) {
                        ds->setSubStmt(replace_goto_break(ds->getSubStmt()));
                    }
                }

                if (sw_changed) {
                    sw->setBody(detail::MakeCompound(ctx, sw_stmts));
                    changed = true;
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        clang::Stmt *EliminateGotoToNextLabel(
            clang::ASTContext &ctx, clang::Stmt *s,
            const std::unordered_set< clang::LabelDecl * > *live) {
            if (!s) {
                return s;
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                return ProcessCompound(ctx, cs, live);
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
                ifs->setThen(EliminateGotoToNextLabel(ctx, ifs->getThen(), live));
                if (ifs->getElse()) {
                    ifs->setElse(EliminateGotoToNextLabel(ctx, ifs->getElse(), live));
                }
                return s;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
                ws->setBody(EliminateGotoToNextLabel(ctx, ws->getBody(), live));
                return s;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(s)) {
                ds->setBody(EliminateGotoToNextLabel(ctx, ds->getBody(), live));
                return s;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
                fs->setBody(EliminateGotoToNextLabel(ctx, fs->getBody(), live));
                return s;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
                ls->setSubStmt(EliminateGotoToNextLabel(ctx, ls->getSubStmt(), live));
                return s;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(s)) {
                sw->setBody(EliminateGotoToNextLabel(ctx, sw->getBody(), live));
                return s;
            }
            return s;
        }

        // ---------------------------------------------------------------
        // ScopeifyIfGotos — convert if(c) goto L; stmts; L: into
        // if(!c) { stmts; } L:
        //
        // Only fires when no intermediate LabelStmts exist between
        // the if-goto and the target label (labels are goto targets
        // from elsewhere and can't be moved into a scope).
        // ---------------------------------------------------------------

        void CountGotoDeclRefs(
            clang::Stmt *stmt,
            std::unordered_map<clang::LabelDecl *, unsigned> &refs
        );

        clang::GotoStmt *SingleGotoStmt(clang::Stmt *stmt);

        clang::LabelDecl *LeadingLabelDecl(clang::Stmt *stmt) {
            if (auto *label = llvm::dyn_cast_or_null<clang::LabelStmt>(stmt))
                return label->getDecl();
            auto *compound =
                llvm::dyn_cast_or_null<clang::CompoundStmt>(stmt);
            if (!compound || compound->body_empty())
                return nullptr;
            return LeadingLabelDecl(compound->body_front());
        }

        bool ClangScopeifyLabelsAreRegionLocal(
            const std::vector<clang::Stmt *> &region,
            const std::unordered_map<clang::LabelDecl *, unsigned> &refs
        ) {
            std::unordered_set<clang::LabelDecl *> labels;
            std::unordered_map<clang::LabelDecl *, unsigned> region_refs;

            std::function<void(clang::Stmt *)> collect_labels =
                [&](clang::Stmt *stmt) {
                    if (!stmt) return;
                    if (auto *label = llvm::dyn_cast<clang::LabelStmt>(stmt))
                        labels.insert(label->getDecl());
                    for (clang::Stmt *child : stmt->children())
                        collect_labels(child);
                };

            for (clang::Stmt *stmt : region) {
                collect_labels(stmt);
                CountGotoDeclRefs(stmt, region_refs);
            }

            for (clang::LabelDecl *label : labels) {
                unsigned global_count = 0;
                if (auto it = refs.find(label); it != refs.end())
                    global_count = it->second;

                unsigned local_count = 0;
                if (auto it = region_refs.find(label);
                    it != region_refs.end())
                    local_count = it->second;

                if (global_count != local_count)
                    return false;
            }
            return true;
        }

        bool IsEffectivelyEmptyStmt(clang::Stmt *stmt) {
            if (!stmt) return true;
            if (llvm::isa<clang::NullStmt>(stmt))
                return true;
            auto *compound =
                llvm::dyn_cast_or_null<clang::CompoundStmt>(stmt);
            if (!compound)
                return false;
            for (clang::Stmt *child : compound->body())
                if (!IsEffectivelyEmptyStmt(child))
                    return false;
            return true;
        }

        clang::Stmt *ScopeifyIfGotos(
            clang::ASTContext &ctx,
            clang::Stmt *s,
            const std::unordered_map<clang::LabelDecl *, unsigned> &refs
        ) {
            if (!s) {
                return s;
            }

            // Recurse into structured bodies first
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
                ifs->setThen(ScopeifyIfGotos(ctx, ifs->getThen(), refs));
                if (ifs->getElse()) {
                    ifs->setElse(ScopeifyIfGotos(ctx, ifs->getElse(), refs));
                }
                return s;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
                ws->setBody(ScopeifyIfGotos(ctx, ws->getBody(), refs));
                return s;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(s)) {
                ds->setBody(ScopeifyIfGotos(ctx, ds->getBody(), refs));
                return s;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
                fs->setBody(ScopeifyIfGotos(ctx, fs->getBody(), refs));
                return s;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
                ls->setSubStmt(ScopeifyIfGotos(ctx, ls->getSubStmt(), refs));
                return s;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(s)) {
                sw->setBody(ScopeifyIfGotos(ctx, sw->getBody(), refs));
                return s;
            }

            auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s);
            if (!cs) {
                return s;
            }

            std::vector< clang::Stmt * > body(cs->body_begin(), cs->body_end());

            // Recurse into children first
            for (auto *&child : body) {
                child = ScopeifyIfGotos(ctx, child, refs);
            }

            // Find if(c) goto L; ... L: patterns
            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t i = 0; i < body.size(); ++i) {
                    auto *ifs = llvm::dyn_cast< clang::IfStmt >(body[i]);
                    if (!ifs) {
                        continue;
                    }

                    if (ifs->getElse() && i + 1 < body.size()) {
                        clang::LabelDecl *next_label =
                            LeadingLabelDecl(body[i + 1]);
                        if (next_label) {
                            auto *then_goto = SingleGotoStmt(ifs->getThen());
                            auto *else_goto = SingleGotoStmt(ifs->getElse());
                            clang::Stmt *replacement = nullptr;
                            auto loc = ifs->getIfLoc();

                            if (then_goto
                                && then_goto->getLabel() == next_label
                                && !IsEffectivelyEmptyStmt(ifs->getElse())) {
                                replacement = clang::IfStmt::Create(
                                    ctx, loc,
                                    clang::IfStatementKind::Ordinary,
                                    nullptr, nullptr,
                                    NegateExpr(ctx, ifs->getCond()), loc, loc,
                                    ifs->getElse(), loc, nullptr);
                            } else if (
                                else_goto
                                && else_goto->getLabel() == next_label
                                && !IsEffectivelyEmptyStmt(ifs->getThen())) {
                                replacement = clang::IfStmt::Create(
                                    ctx, loc,
                                    clang::IfStatementKind::Ordinary,
                                    nullptr, nullptr, ifs->getCond(), loc,
                                    loc, ifs->getThen(), loc, nullptr);
                            }

                            if (replacement) {
                                body[i] = replacement;
                                changed = true;
                                break;
                            }
                        }
                    }

                    if (ifs->getElse()) {
                        continue;
                    }
                    auto *tail_goto = llvm::dyn_cast_or_null<clang::GotoStmt>(
                        DeepTrailingStmt(ifs->getThen()));
                    if (tail_goto && tail_goto->getLabel()) {
                        auto *target_decl = tail_goto->getLabel();
                        auto target_name = target_decl->getName();

                        size_t label_idx = body.size();
                        for (size_t j = i + 1; j < body.size(); ++j) {
                            clang::LabelDecl *leading_label =
                                LeadingLabelDecl(body[j]);
                            if (leading_label
                                && leading_label->getName() == target_name) {
                                label_idx = j;
                                break;
                            }
                        }
                        if (label_idx >= body.size())
                            continue;

                        clang::Stmt *then_without_goto =
                            StripTrailingGoto(ctx, ifs->getThen(), target_name);
                        if (!IsEffectivelyEmptyStmt(then_without_goto)) {
                            std::vector<clang::Stmt *> scoped;
                            for (size_t j = i + 1; j < label_idx; ++j)
                                scoped.push_back(body[j]);
                            if (!ClangScopeifyLabelsAreRegionLocal(
                                    scoped, refs))
                                continue;

                            auto loc = ifs->getIfLoc();
                            auto *new_if = clang::IfStmt::Create(
                                ctx, loc, clang::IfStatementKind::Ordinary,
                                nullptr, nullptr, ifs->getCond(), loc, loc,
                                then_without_goto, loc,
                                detail::MakeCompound(ctx, scoped));

                            body.erase(
                                body.begin() + static_cast<ptrdiff_t>(i),
                                body.begin()
                                    + static_cast<ptrdiff_t>(label_idx));
                            body.insert(
                                body.begin() + static_cast<ptrdiff_t>(i),
                                new_if);
                            changed = true;
                            break;
                        }
                    }

                    auto *gs = SingleGotoStmt(ifs->getThen());
                    if (!gs) {
                        continue;
                    }
                    auto *target_decl = gs->getLabel();
                    if (!target_decl) {
                        continue;
                    }
                    auto target_name = target_decl->getName();

                    // Find the target LabelStmt in the same CompoundStmt
                    size_t label_idx = body.size();
                    for (size_t j = i + 1; j < body.size(); ++j) {
                        clang::LabelDecl *leading_label =
                            LeadingLabelDecl(body[j]);
                        if (leading_label
                            && leading_label->getName() == target_name) {
                            label_idx = j;
                            break;
                        }
                    }
                    if (label_idx >= body.size()) {
                        continue;
                    }
                    // Skip adjacent if-goto (label_idx == i+1) — the
                    // goto looks dead but the condition may be a guard
                    // for code after the label.  Let it remain as a
                    // no-op if-goto.
                    if (label_idx == i + 1) {
                        continue;
                    }

                    // Collect intermediate stmts
                    std::vector< clang::Stmt * > scoped;
                    for (size_t j = i + 1; j < label_idx; ++j) {
                        scoped.push_back(body[j]);
                    }
                    // Intermediate labels are safe only when every reference
                    // to them is also inside the region being scoped.  This
                    // permits local label traffic while still rejecting gotos
                    // from outside into the moved block.
                    if (!ClangScopeifyLabelsAreRegionLocal(scoped, refs)) {
                        continue;
                    }

                    // Build: if(!cond) { scoped_stmts }
                    auto *neg        = NegateExpr(ctx, ifs->getCond());
                    auto *scope_body = detail::MakeCompound(ctx, scoped);
                    auto loc         = ifs->getIfLoc();
                    auto *new_if     = clang::IfStmt::Create(
                        ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr, neg,
                        loc, loc, scope_body, loc, nullptr
                    );

                    // Replace: remove if-goto + intermediates, insert new if
                    body.erase(
                        body.begin() + static_cast< ptrdiff_t >(i),
                        body.begin() + static_cast< ptrdiff_t >(label_idx)
                    );
                    body.insert(body.begin() + static_cast< ptrdiff_t >(i), new_if);

                    changed = true;
                    break;
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        void CountGotoDeclRefs(
            clang::Stmt *stmt,
            std::unordered_map<clang::LabelDecl *, unsigned> &refs,
            std::unordered_set<clang::Stmt *> &seen
        ) {
            if (!stmt || !seen.insert(stmt).second) return;
            if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(stmt)) {
                refs[gs->getLabel()]++;
                return;
            }
            for (clang::Stmt *child : stmt->children())
                CountGotoDeclRefs(child, refs, seen);
        }

        void CountGotoDeclRefs(
            clang::Stmt *stmt,
            std::unordered_map<clang::LabelDecl *, unsigned> &refs
        ) {
            std::unordered_set<clang::Stmt *> seen;
            CountGotoDeclRefs(stmt, refs, seen);
        }

        clang::GotoStmt *SingleGotoStmt(clang::Stmt *stmt) {
            if (!stmt) return nullptr;
            if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(stmt))
                return gs;
            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
                if (compound->body_empty())
                    return nullptr;
                auto it = compound->body_begin();
                ++it;
                if (it != compound->body_end())
                    return nullptr;
                return SingleGotoStmt(compound->body_front());
            }
            return nullptr;
        }

        struct NestedClangEntryLabel {
            clang::Stmt *entry_stmt = nullptr;
            clang::IfStmt *owner_if = nullptr;
            bool is_then = false;
        };

        bool FindDirectNestedEntryLabel(
            clang::ASTContext &ctx,
            clang::Stmt *stmt,
            clang::LabelDecl *target,
            NestedClangEntryLabel &out
        ) {
            auto *ifs = llvm::dyn_cast_or_null<clang::IfStmt>(stmt);
            if (!ifs)
                return false;

            auto match_arm = [&](clang::Stmt *arm, bool is_then) -> bool {
                auto *label = llvm::dyn_cast_or_null<clang::LabelStmt>(arm);
                if (label && label->getDecl() == target) {
                    out = {label->getSubStmt(), ifs, is_then};
                    return true;
                }

                auto *compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(arm);
                if (!compound || compound->body_empty())
                    return false;
                auto it = compound->body_begin();
                label = llvm::dyn_cast_or_null<clang::LabelStmt>(*it);
                if (!label || label->getDecl() != target)
                    return false;

                std::vector<clang::Stmt *> unwrapped;
                unwrapped.push_back(label->getSubStmt());
                for (++it; it != compound->body_end(); ++it)
                    unwrapped.push_back(*it);
                out = {detail::MakeCompound(ctx, unwrapped), ifs, is_then};
                return true;
            };

            return match_arm(ifs->getThen(), true)
                || match_arm(ifs->getElse(), false);
        }

        size_t CountEntryStmtUnits(const clang::Stmt *stmt) {
            if (!stmt) return 0;
            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
                size_t count = 0;
                for (const clang::Stmt *child : compound->body()) {
                    count += CountEntryStmtUnits(child);
                    if (count > 8)
                        return count;
                }
                return count;
            }
            return 1;
        }

        bool EntryStmtIsCloneSafe(const clang::Stmt *stmt) {
            if (!stmt || CountEntryStmtUnits(stmt) > 8)
                return false;
            if (llvm::isa<clang::LabelStmt>(stmt)
                || llvm::isa<clang::GotoStmt>(stmt)
                || llvm::isa<clang::BreakStmt>(stmt)
                || llvm::isa<clang::ContinueStmt>(stmt)
                || llvm::isa<clang::SwitchStmt>(stmt)
                || llvm::isa<clang::WhileStmt>(stmt)
                || llvm::isa<clang::DoStmt>(stmt)
                || llvm::isa<clang::ForStmt>(stmt))
                return false;
            for (const clang::Stmt *child : stmt->children())
                if (!EntryStmtIsCloneSafe(child))
                    return false;
            return true;
        }

        clang::Stmt *RepairCrossScopeLabelEntries(
            clang::ASTContext &ctx,
            clang::Stmt *stmt,
            const std::unordered_map<clang::LabelDecl *, unsigned> &refs
        ) {
            if (!stmt)
                return stmt;

            if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(stmt)) {
                ifs->setThen(RepairCrossScopeLabelEntries(
                    ctx, ifs->getThen(), refs));
                if (ifs->getElse())
                    ifs->setElse(RepairCrossScopeLabelEntries(
                        ctx, ifs->getElse(), refs));
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast<clang::WhileStmt>(stmt)) {
                ws->setBody(RepairCrossScopeLabelEntries(
                    ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast<clang::DoStmt>(stmt)) {
                ds->setBody(RepairCrossScopeLabelEntries(
                    ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast<clang::ForStmt>(stmt)) {
                fs->setBody(RepairCrossScopeLabelEntries(
                    ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast<clang::LabelStmt>(stmt)) {
                ls->setSubStmt(RepairCrossScopeLabelEntries(
                    ctx, ls->getSubStmt(), refs));
                return ls;
            }
            if (auto *sw = llvm::dyn_cast<clang::SwitchStmt>(stmt)) {
                sw->setBody(RepairCrossScopeLabelEntries(
                    ctx, sw->getBody(), refs));
                return sw;
            }

            auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt);
            if (!compound)
                return stmt;

            std::vector<clang::Stmt *> body(
                compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body)
                child = RepairCrossScopeLabelEntries(ctx, child, refs);

            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t i = 0; i + 1 < body.size(); ++i) {
                    auto *ifs = llvm::dyn_cast<clang::IfStmt>(body[i]);
                    if (!ifs || ifs->getElse())
                        continue;
                    clang::GotoStmt *guard_goto =
                        SingleGotoStmt(ifs->getThen());
                    if (!guard_goto || !guard_goto->getLabel())
                        continue;
                    clang::LabelDecl *target = guard_goto->getLabel();
                    auto ref_it = refs.find(target);
                    if (ref_it == refs.end() || ref_it->second != 1)
                        continue;

                    size_t carrier_idx = body.size();
                    NestedClangEntryLabel loc;
                    for (size_t j = i + 1; j < body.size(); ++j) {
                        if (FindDirectNestedEntryLabel(
                                ctx, body[j], target, loc)) {
                            carrier_idx = j;
                            break;
                        }
                    }
                    if (carrier_idx >= body.size() || !loc.entry_stmt
                        || !loc.owner_if)
                        continue;

                    clang::Stmt *entry_stmt = loc.entry_stmt;
                    if (!EntryStmtIsCloneSafe(entry_stmt))
                        continue;

                    if (loc.is_then)
                        loc.owner_if->setThen(entry_stmt);
                    else
                        loc.owner_if->setElse(entry_stmt);

                    std::vector<clang::Stmt *> else_body;
                    else_body.reserve(carrier_idx - i);
                    for (size_t j = i + 1; j <= carrier_idx; ++j)
                        else_body.push_back(body[j]);

                    auto loc_if = ifs->getIfLoc();
                    auto *new_if = clang::IfStmt::Create(
                        ctx, loc_if, clang::IfStatementKind::Ordinary,
                        nullptr, nullptr, CloneExpr(ctx, ifs->getCond()),
                        loc_if, loc_if, entry_stmt, loc_if,
                        detail::MakeCompound(ctx, else_body));

                    body.erase(
                        body.begin() + static_cast<ptrdiff_t>(i),
                        body.begin() + static_cast<ptrdiff_t>(carrier_idx)
                            + 1);
                    body.insert(
                        body.begin() + static_cast<ptrdiff_t>(i), new_if);
                    changed = true;
                    break;
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        clang::Stmt *SwitchLocalLabelEntryStmt(
            clang::ASTContext &ctx,
            clang::Stmt *stmt,
            clang::LabelDecl *target
        ) {
            if (auto *label = llvm::dyn_cast_or_null<clang::LabelStmt>(stmt)) {
                if (label->getDecl() == target)
                    return label->getSubStmt();
                return nullptr;
            }

            auto *compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(stmt);
            if (!compound || compound->body_empty())
                return nullptr;

            auto it = compound->body_begin();
            auto *label = llvm::dyn_cast_or_null<clang::LabelStmt>(*it);
            if (!label || label->getDecl() != target)
                return nullptr;

            std::vector<clang::Stmt *> unwrapped;
            unwrapped.push_back(label->getSubStmt());
            for (++it; it != compound->body_end(); ++it)
                unwrapped.push_back(*it);
            return detail::MakeCompound(ctx, unwrapped);
        }

        bool StmtStartsWithLabel(clang::Stmt *stmt) {
            if (llvm::isa_and_nonnull<clang::LabelStmt>(stmt))
                return true;
            auto *compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(stmt);
            return compound && !compound->body_empty()
                && llvm::isa<clang::LabelStmt>(compound->body_front());
        }

        bool BuildSwitchLocalLabelTail(
            clang::ASTContext &ctx,
            const std::vector<clang::Stmt *> &body,
            size_t label_idx,
            clang::LabelDecl *target,
            clang::Stmt *&entry,
            size_t &erase_end
        ) {
            entry = nullptr;
            erase_end = label_idx;
            clang::Stmt *first =
                SwitchLocalLabelEntryStmt(ctx, body[label_idx], target);
            if (!first)
                return false;

            std::vector<clang::Stmt *> tail;
            tail.push_back(first);
            erase_end = label_idx + 1;

            auto build_tail_stmt = [&]() -> clang::Stmt * {
                if (tail.size() == 1)
                    return tail.front();
                return detail::MakeCompound(ctx, tail);
            };

            while (!detail::EndsWithTerminator(build_tail_stmt())) {
                if (erase_end >= body.size())
                    return false;
                if (StmtStartsWithLabel(body[erase_end]))
                    return false;
                tail.push_back(body[erase_end]);
                ++erase_end;
            }

            entry = build_tail_stmt();
            return EntryStmtIsCloneSafe(entry);
        }

        clang::Stmt *ReplaceGotoToSwitchLocalLabel(
            clang::ASTContext &ctx,
            clang::Stmt *stmt,
            clang::LabelDecl *target,
            clang::Stmt *replacement,
            unsigned &replaced
        ) {
            if (!stmt) return stmt;
            if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(stmt)) {
                if (gs->getLabel() == target) {
                    ++replaced;
                    return replacement;
                }
                return stmt;
            }

            // Do not rewrite gotos in nested control scopes; this pass is for
            // the current switch case body only.
            if (llvm::isa<clang::SwitchStmt>(stmt)
                || llvm::isa<clang::WhileStmt>(stmt)
                || llvm::isa<clang::DoStmt>(stmt)
                || llvm::isa<clang::ForStmt>(stmt))
                return stmt;

            if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
                std::vector<clang::Stmt *> children;
                unsigned before = replaced;
                for (clang::Stmt *child : compound->body()) {
                    clang::Stmt *rewritten = ReplaceGotoToSwitchLocalLabel(
                        ctx, child, target, replacement, replaced);
                    children.push_back(rewritten);
                }
                if (replaced != before)
                    return detail::MakeCompound(ctx, children);
                return stmt;
            }

            if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(stmt)) {
                ifs->setThen(ReplaceGotoToSwitchLocalLabel(
                    ctx, ifs->getThen(), target, replacement, replaced));
                if (ifs->getElse())
                    ifs->setElse(ReplaceGotoToSwitchLocalLabel(
                        ctx, ifs->getElse(), target, replacement,
                        replaced));
                return stmt;
            }

            return stmt;
        }

        bool ReplaceGotoToSwitchLocalLabelInSwitch(
            clang::ASTContext &ctx,
            clang::SwitchStmt *sw,
            clang::LabelDecl *target,
            clang::Stmt *replacement,
            unsigned &replaced
        ) {
            auto *body = llvm::dyn_cast_or_null<clang::CompoundStmt>(
                sw->getBody());
            if (!body) return false;

            unsigned before = replaced;
            for (clang::Stmt *child : body->body()) {
                if (auto *case_stmt = llvm::dyn_cast<clang::CaseStmt>(child)) {
                    case_stmt->setSubStmt(ReplaceGotoToSwitchLocalLabel(
                        ctx, case_stmt->getSubStmt(), target, replacement,
                        replaced));
                } else if (auto *default_stmt =
                               llvm::dyn_cast<clang::DefaultStmt>(child)) {
                    default_stmt->setSubStmt(ReplaceGotoToSwitchLocalLabel(
                        ctx, default_stmt->getSubStmt(), target, replacement,
                        replaced));
                }
            }
            return replaced != before;
        }

        unsigned CountSwitchLocalGotosToLabel(
            clang::Stmt *stmt,
            clang::LabelDecl *target
        ) {
            if (!stmt) return 0;
            if (auto *gs = llvm::dyn_cast<clang::GotoStmt>(stmt))
                return gs->getLabel() == target ? 1U : 0U;
            if (llvm::isa<clang::SwitchStmt>(stmt)
                || llvm::isa<clang::WhileStmt>(stmt)
                || llvm::isa<clang::DoStmt>(stmt)
                || llvm::isa<clang::ForStmt>(stmt))
                return 0;

            unsigned count = 0;
            for (clang::Stmt *child : stmt->children())
                count += CountSwitchLocalGotosToLabel(child, target);
            return count;
        }

        unsigned CountSwitchLocalGotosToLabel(
            clang::SwitchStmt *sw,
            clang::LabelDecl *target
        ) {
            auto *body = llvm::dyn_cast_or_null<clang::CompoundStmt>(
                sw->getBody());
            if (!body) return 0;

            unsigned count = 0;
            for (clang::Stmt *child : body->body()) {
                if (auto *case_stmt = llvm::dyn_cast<clang::CaseStmt>(child))
                    count += CountSwitchLocalGotosToLabel(
                        case_stmt->getSubStmt(), target);
                else if (auto *default_stmt =
                             llvm::dyn_cast<clang::DefaultStmt>(child))
                    count += CountSwitchLocalGotosToLabel(
                        default_stmt->getSubStmt(), target);
            }
            return count;
        }

        clang::Stmt *FoldClangSwitchLocalCaseTargets(
            clang::ASTContext &ctx,
            clang::Stmt *stmt,
            const std::unordered_map<clang::LabelDecl *, unsigned> &refs
        ) {
            if (!stmt) return stmt;

            if (auto *ifs = llvm::dyn_cast<clang::IfStmt>(stmt)) {
                ifs->setThen(FoldClangSwitchLocalCaseTargets(
                    ctx, ifs->getThen(), refs));
                if (ifs->getElse())
                    ifs->setElse(FoldClangSwitchLocalCaseTargets(
                        ctx, ifs->getElse(), refs));
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast<clang::WhileStmt>(stmt)) {
                ws->setBody(FoldClangSwitchLocalCaseTargets(
                    ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast<clang::DoStmt>(stmt)) {
                ds->setBody(FoldClangSwitchLocalCaseTargets(
                    ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast<clang::ForStmt>(stmt)) {
                fs->setBody(FoldClangSwitchLocalCaseTargets(
                    ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast<clang::LabelStmt>(stmt)) {
                ls->setSubStmt(FoldClangSwitchLocalCaseTargets(
                    ctx, ls->getSubStmt(), refs));
                return ls;
            }

            auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt);
            if (!compound)
                return stmt;

            std::vector<clang::Stmt *> body(
                compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body)
                child = FoldClangSwitchLocalCaseTargets(ctx, child, refs);

            for (size_t i = 0; i < body.size(); ++i) {
                auto *sw = llvm::dyn_cast<clang::SwitchStmt>(body[i]);
                if (!sw) continue;

                bool changed = true;
                while (changed) {
                    changed = false;
                    for (size_t j = i + 1; j < body.size(); ++j) {
                        clang::LabelDecl *target = nullptr;
                        if (auto *label =
                                llvm::dyn_cast<clang::LabelStmt>(body[j])) {
                            target = label->getDecl();
                        } else if (auto *label_body =
                                       llvm::dyn_cast<clang::CompoundStmt>(
                                           body[j])) {
                            if (!label_body->body_empty()) {
                                if (auto *label = llvm::dyn_cast<clang::LabelStmt>(
                                        label_body->body_front()))
                                    target = label->getDecl();
                            }
                        }
                        if (!target) continue;

                        auto ref_it = refs.find(target);
                        if (ref_it == refs.end() || ref_it->second == 0)
                            continue;
                        if (j == 0 || !detail::EndsWithTerminator(body[j - 1]))
                            continue;

                        clang::Stmt *entry = nullptr;
                        size_t erase_end = j;
                        if (!BuildSwitchLocalLabelTail(
                                ctx, body, j, target, entry, erase_end))
                            continue;
                        if (CountSwitchLocalGotosToLabel(sw, target)
                            != ref_it->second)
                            continue;

                        unsigned replaced = 0;
                        if (!ReplaceGotoToSwitchLocalLabelInSwitch(
                                ctx, sw, target, entry, replaced))
                            continue;
                        if (replaced != ref_it->second)
                            continue;

                        body.erase(body.begin() + static_cast<ptrdiff_t>(j),
                                   body.begin()
                                       + static_cast<ptrdiff_t>(erase_end));
                        changed = true;
                        break;
                    }
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        constexpr size_t kMaxConditionalFallthroughBodyStmts = 8;
        constexpr size_t kMaxTerminalPrefixStmts             = 12;

        void AppendStmtSequence(clang::Stmt *stmt, std::vector< clang::Stmt * > &out) {
            if (!stmt) { return; }
            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                for (clang::Stmt *child : compound->body()) { out.push_back(child); }
                return;
            }
            out.push_back(stmt);
        }

        struct LocalLabelBlock
        {
            clang::LabelDecl *decl = nullptr;
            size_t begin           = 0;
            size_t end             = 0;
            std::vector< clang::Stmt * > stmts;
        };

        bool ExtractLocalLabelBlock(
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &body, size_t label_idx,
            LocalLabelBlock &block
        ) {
            if (label_idx >= body.size()) { return false; }

            clang::LabelDecl *decl = LeadingLabelDecl(body[label_idx]);
            if (!decl) { return false; }

            clang::Stmt *entry = SwitchLocalLabelEntryStmt(ctx, body[label_idx], decl);
            if (!entry) { return false; }

            block       = {};
            block.decl  = decl;
            block.begin = label_idx;
            AppendStmtSequence(entry, block.stmts);

            size_t end = label_idx + 1;
            while (end < body.size()) {
                if (LeadingLabelDecl(body[end])) { break; }
                block.stmts.push_back(body[end]);
                ++end;
            }
            block.end = end;
            return !block.stmts.empty();
        }

        bool SeqEndsWithTerminator(
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &stmts
        ) {
            if (stmts.empty()) { return false; }
            return detail::EndsWithTerminator(detail::MakeCompound(ctx, stmts));
        }

        bool SeqHasUnsafeStructure(clang::Stmt *stmt, bool allow_goto, bool allow_return) {
            if (!stmt) { return false; }
            if (llvm::isa< clang::LabelStmt >(stmt) || llvm::isa< clang::SwitchStmt >(stmt)
                || llvm::isa< clang::WhileStmt >(stmt) || llvm::isa< clang::DoStmt >(stmt)
                || llvm::isa< clang::ForStmt >(stmt) || llvm::isa< clang::DeclStmt >(stmt)
                || llvm::isa< clang::BreakStmt >(stmt)
                || llvm::isa< clang::ContinueStmt >(stmt))
            {
                return true;
            }
            if (!allow_goto && llvm::isa< clang::GotoStmt >(stmt)) { return true; }
            if (!allow_return && llvm::isa< clang::ReturnStmt >(stmt)) { return true; }
            for (clang::Stmt *child : stmt->children()) {
                if (SeqHasUnsafeStructure(child, allow_goto, allow_return)) { return true; }
            }
            return false;
        }

        bool SeqHasUnsafeStructure(
            const std::vector< clang::Stmt * > &stmts, bool allow_goto, bool allow_return
        ) {
            for (clang::Stmt *stmt : stmts) {
                if (SeqHasUnsafeStructure(stmt, allow_goto, allow_return)) { return true; }
            }
            return false;
        }

        unsigned CountGotosToLabel(clang::Stmt *stmt, clang::LabelDecl *target) {
            if (!stmt || !target) { return 0; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                return gs->getLabel() == target ? 1U : 0U;
            }
            unsigned count = 0;
            for (clang::Stmt *child : stmt->children()) {
                count += CountGotosToLabel(child, target);
            }
            return count;
        }

        clang::Stmt *
        StmtFromSeq(clang::ASTContext &ctx, const std::vector< clang::Stmt * > &stmts) {
            if (stmts.empty()) { return new (ctx) clang::NullStmt(VirtualLoc(ctx)); }
            if (stmts.size() == 1) { return stmts.front(); }
            return detail::MakeCompound(ctx, stmts);
        }

        clang::Stmt *ReplaceGotoWithStmt(
            clang::ASTContext &ctx, clang::Stmt *stmt, clang::LabelDecl *target,
            clang::Stmt *replacement, unsigned &replaced
        ) {
            if (!stmt) { return stmt; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                if (gs->getLabel() == target) {
                    ++replaced;
                    return replacement;
                }
                return stmt;
            }

            if (llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::WhileStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt))
            {
                return stmt;
            }

            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > children;
                unsigned before = replaced;
                for (clang::Stmt *child : compound->body()) {
                    children.push_back(
                        ReplaceGotoWithStmt(ctx, child, target, replacement, replaced)
                    );
                }
                return replaced != before ? detail::MakeCompound(ctx, children) : stmt;
            }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(
                    ReplaceGotoWithStmt(ctx, ifs->getThen(), target, replacement, replaced)
                );
                if (ifs->getElse()) {
                    ifs->setElse(
                        ReplaceGotoWithStmt(ctx, ifs->getElse(), target, replacement, replaced)
                    );
                }
                return stmt;
            }

            return stmt;
        }

        clang::Stmt *FoldConditionalFallthroughChains(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(FoldConditionalFallthroughChains(ctx, ifs->getThen(), refs));
                if (ifs->getElse()) {
                    ifs->setElse(FoldConditionalFallthroughChains(ctx, ifs->getElse(), refs));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(FoldConditionalFallthroughChains(ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(FoldConditionalFallthroughChains(ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(FoldConditionalFallthroughChains(ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(FoldConditionalFallthroughChains(ctx, ls->getSubStmt(), refs));
                return ls;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(FoldConditionalFallthroughChains(ctx, sw->getBody(), refs));
                return sw;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = FoldConditionalFallthroughChains(ctx, child, refs);
            }

            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t label_idx = 1; label_idx < body.size(); ++label_idx) {
                    clang::LabelDecl *target = LeadingLabelDecl(body[label_idx]);
                    if (!target) { continue; }
                    auto ref_it = refs.find(target);
                    if (ref_it == refs.end() || ref_it->second != 1) { continue; }
                    if (!detail::EndsWithTerminator(body[label_idx - 1])) { continue; }
                    if (CountGotosToLabel(body[label_idx - 1], target) != 1) { continue; }

                    LocalLabelBlock block;
                    if (!ExtractLocalLabelBlock(ctx, body, label_idx, block)) { continue; }
                    if (block.stmts.size() > kMaxConditionalFallthroughBodyStmts) { continue; }
                    if (SeqEndsWithTerminator(ctx, block.stmts)) { continue; }
                    if (SeqHasUnsafeStructure(
                            block.stmts, /*allow_goto=*/true,
                            /*allow_return=*/false
                        ))
                    {
                        continue;
                    }

                    clang::Stmt *replacement = StmtFromSeq(ctx, block.stmts);
                    unsigned replaced        = 0;
                    body[label_idx - 1]      = ReplaceGotoWithStmt(
                        ctx, body[label_idx - 1], target, replacement, replaced
                    );
                    if (replaced != 1) { continue; }

                    body.erase(
                        body.begin() + static_cast< ptrdiff_t >(block.begin),
                        body.begin() + static_cast< ptrdiff_t >(block.end)
                    );
                    changed = true;
                    break;
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        std::string StmtToStableString(clang::ASTContext &ctx, clang::Stmt *stmt) {
            std::string out;
            llvm::raw_string_ostream os(out);
            clang::PrintingPolicy policy(ctx.getLangOpts());
            policy.SuppressTagKeyword = true;
            if (stmt) { stmt->printPretty(os, nullptr, policy); }
            os.flush();
            return out;
        }

        size_t CommonTerminalSuffixSize(
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &lhs,
            const std::vector< clang::Stmt * > &rhs
        ) {
            size_t count = 0;
            while (count < lhs.size() && count < rhs.size()) {
                clang::Stmt *a = lhs[lhs.size() - count - 1];
                clang::Stmt *b = rhs[rhs.size() - count - 1];
                if (StmtToStableString(ctx, a) != StmtToStableString(ctx, b)) { break; }
                ++count;
            }
            return count;
        }

        bool LabelHasNoFallthroughPredecessor(
            const std::vector< clang::Stmt * > &body, size_t label_idx
        ) {
            return label_idx == 0 || detail::EndsWithTerminator(body[label_idx - 1]);
        }

        bool FindLabelBlockByDecl(
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &body,
            clang::LabelDecl *decl, LocalLabelBlock &block
        ) {
            for (size_t i = 0; i < body.size(); ++i) {
                if (LeadingLabelDecl(body[i]) != decl) { continue; }
                return ExtractLocalLabelBlock(ctx, body, i, block);
            }
            return false;
        }

        struct TerminalEpilogueRewrite
        {
            clang::IfStmt *site      = nullptr;
            clang::Stmt *replacement = nullptr;
            LocalLabelBlock then_block;
            LocalLabelBlock else_block;
        };

        bool TryBuildTerminalEpilogueRewriteForIf(
            clang::ASTContext &ctx, clang::IfStmt *ifs,
            const std::vector< clang::Stmt * > &body,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs,
            TerminalEpilogueRewrite &rewrite
        ) {
            if (!ifs || !ifs->getElse()) { return false; }
            clang::GotoStmt *then_goto = SingleGotoStmt(ifs->getThen());
            clang::GotoStmt *else_goto = SingleGotoStmt(ifs->getElse());
            if (!then_goto || !else_goto) { return false; }

            clang::LabelDecl *then_label = then_goto->getLabel();
            clang::LabelDecl *else_label = else_goto->getLabel();
            if (!then_label || !else_label || then_label == else_label) { return false; }
            auto then_ref_it = refs.find(then_label);
            auto else_ref_it = refs.find(else_label);
            if (then_ref_it == refs.end() || else_ref_it == refs.end()
                || then_ref_it->second != 1 || else_ref_it->second != 1)
            {
                return false;
            }

            LocalLabelBlock then_block;
            LocalLabelBlock else_block;
            if (!FindLabelBlockByDecl(ctx, body, then_label, then_block)
                || !FindLabelBlockByDecl(ctx, body, else_label, else_block))
            {
                return false;
            }
            if (!LabelHasNoFallthroughPredecessor(body, then_block.begin)
                || !LabelHasNoFallthroughPredecessor(body, else_block.begin))
            {
                return false;
            }
            if (!SeqEndsWithTerminator(ctx, then_block.stmts)
                || !SeqEndsWithTerminator(ctx, else_block.stmts))
            {
                return false;
            }
            if (SeqHasUnsafeStructure(
                    then_block.stmts, /*allow_goto=*/false,
                    /*allow_return=*/true
                )
                || SeqHasUnsafeStructure(
                    else_block.stmts, /*allow_goto=*/false,
                    /*allow_return=*/true
                ))
            {
                return false;
            }

            size_t suffix = CommonTerminalSuffixSize(ctx, then_block.stmts, else_block.stmts);
            if (suffix == 0) { return false; }

            std::vector< clang::Stmt * > common_suffix(
                then_block.stmts.end() - static_cast< ptrdiff_t >(suffix),
                then_block.stmts.end()
            );
            if (!SeqEndsWithTerminator(ctx, common_suffix)) { return false; }

            std::vector< clang::Stmt * > then_prefix(
                then_block.stmts.begin(),
                then_block.stmts.end() - static_cast< ptrdiff_t >(suffix)
            );
            std::vector< clang::Stmt * > else_prefix(
                else_block.stmts.begin(),
                else_block.stmts.end() - static_cast< ptrdiff_t >(suffix)
            );
            if (then_prefix.size() > kMaxTerminalPrefixStmts
                || else_prefix.size() > kMaxTerminalPrefixStmts)
            {
                return false;
            }
            if (SeqHasUnsafeStructure(
                    then_prefix, /*allow_goto=*/false,
                    /*allow_return=*/false
                )
                || SeqHasUnsafeStructure(
                    else_prefix, /*allow_goto=*/false,
                    /*allow_return=*/false
                ))
            {
                return false;
            }
            if (!then_prefix.empty() && detail::EndsWithTerminator(then_prefix.back())) {
                return false;
            }
            if (!else_prefix.empty() && detail::EndsWithTerminator(else_prefix.back())) {
                return false;
            }

            auto loc               = ifs->getIfLoc();
            clang::Stmt *then_stmt = StmtFromSeq(ctx, then_prefix);
            clang::Stmt *else_stmt = StmtFromSeq(ctx, else_prefix);
            auto *new_if           = clang::IfStmt::Create(
                ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr, ifs->getCond(),
                loc, loc, then_stmt, loc, else_stmt
            );

            std::vector< clang::Stmt * > replacement_stmts;
            replacement_stmts.push_back(new_if);
            replacement_stmts.insert(
                replacement_stmts.end(), common_suffix.begin(), common_suffix.end()
            );

            rewrite.site        = ifs;
            rewrite.replacement = detail::MakeCompound(ctx, replacement_stmts);
            rewrite.then_block  = then_block;
            rewrite.else_block  = else_block;
            return true;
        }

        bool FindTerminalEpilogueRewriteInStmt(
            clang::ASTContext &ctx, clang::Stmt *stmt, const std::vector< clang::Stmt * > &body,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs,
            TerminalEpilogueRewrite &rewrite
        ) {
            if (!stmt) { return false; }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                if (TryBuildTerminalEpilogueRewriteForIf(ctx, ifs, body, refs, rewrite)) {
                    return true;
                }
                if (FindTerminalEpilogueRewriteInStmt(ctx, ifs->getThen(), body, refs, rewrite))
                {
                    return true;
                }
                if (FindTerminalEpilogueRewriteInStmt(ctx, ifs->getElse(), body, refs, rewrite))
                {
                    return true;
                }
                return false;
            }
            if (llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::WhileStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt))
            {
                return false;
            }
            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                for (clang::Stmt *child : compound->body()) {
                    if (FindTerminalEpilogueRewriteInStmt(ctx, child, body, refs, rewrite)) {
                        return true;
                    }
                }
            }
            return false;
        }

        clang::Stmt *ReplaceStmtPtr(
            clang::ASTContext &ctx, clang::Stmt *stmt, clang::Stmt *target,
            clang::Stmt *replacement, bool &changed
        ) {
            if (!stmt) { return stmt; }
            if (stmt == target) {
                changed = true;
                return replacement;
            }
            if (llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::WhileStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt))
            {
                return stmt;
            }
            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > children;
                bool before = changed;
                for (clang::Stmt *child : compound->body()) {
                    children.push_back(
                        ReplaceStmtPtr(ctx, child, target, replacement, changed)
                    );
                }
                return changed != before ? detail::MakeCompound(ctx, children) : stmt;
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(ReplaceStmtPtr(ctx, ifs->getThen(), target, replacement, changed));
                if (ifs->getElse()) {
                    ifs->setElse(
                        ReplaceStmtPtr(ctx, ifs->getElse(), target, replacement, changed)
                    );
                }
                return stmt;
            }
            return stmt;
        }

        void EraseLabelBlockRanges(
            std::vector< clang::Stmt * > &body, const LocalLabelBlock &a,
            const LocalLabelBlock &b
        ) {
            std::vector< std::pair< size_t, size_t > > ranges = {
                { a.begin, a.end },
                { b.begin, b.end },
            };
            if (ranges[0].first > ranges[1].first) { std::swap(ranges[0], ranges[1]); }
            if (ranges[0].second > ranges[1].first) { return; }
            for (auto it = ranges.rbegin(); it != ranges.rend(); ++it) {
                body.erase(
                    body.begin() + static_cast< ptrdiff_t >(it->first),
                    body.begin() + static_cast< ptrdiff_t >(it->second)
                );
            }
        }

        clang::Stmt *SinkCommonTerminalEpilogues(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(SinkCommonTerminalEpilogues(ctx, ifs->getThen(), refs));
                if (ifs->getElse()) {
                    ifs->setElse(SinkCommonTerminalEpilogues(ctx, ifs->getElse(), refs));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(SinkCommonTerminalEpilogues(ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(SinkCommonTerminalEpilogues(ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(SinkCommonTerminalEpilogues(ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(SinkCommonTerminalEpilogues(ctx, ls->getSubStmt(), refs));
                return ls;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(SinkCommonTerminalEpilogues(ctx, sw->getBody(), refs));
                return sw;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = SinkCommonTerminalEpilogues(ctx, child, refs);
            }

            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t i = 0; i < body.size(); ++i) {
                    if (LeadingLabelDecl(body[i])) { continue; }
                    TerminalEpilogueRewrite rewrite;
                    if (!FindTerminalEpilogueRewriteInStmt(ctx, body[i], body, refs, rewrite)) {
                        continue;
                    }

                    bool replaced = false;
                    body[i]       = ReplaceStmtPtr(
                        ctx, body[i], rewrite.site, rewrite.replacement, replaced
                    );
                    if (!replaced) { continue; }

                    EraseLabelBlockRanges(body, rewrite.then_block, rewrite.else_block);
                    changed = true;
                    break;
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        // Collect all LabelDecls that have a LabelStmt definition in the tree.
        void CollectDefinedLabels(clang::Stmt *s,
                                  std::unordered_set< clang::LabelDecl * > &defined) {
            llvm::SmallVector< clang::Stmt *, 16 > worklist;
            if (s) worklist.push_back(s);

            while (!worklist.empty()) {
                auto *cur = worklist.pop_back_val();
                if (!cur) continue;
                if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(cur))
                    defined.insert(ls->getDecl());
                for (auto *child : cur->children())
                    if (child) worklist.push_back(child);
            }
        }

        // Replace GotoStmts whose target label has no LabelStmt in the
        // function body with NullStmt.  When the orphaned goto is inside
        // a switch case body (in_switch_case=true), replace with BreakStmt
        // instead to prevent unintended fallthrough.
        clang::Stmt *RemoveOrphanedGotos(
            clang::ASTContext &ctx, clang::Stmt *s,
            const std::unordered_set< clang::LabelDecl * > &defined,
            unsigned depth = 0, bool in_switch_case = false
        ) {
            if (!s) return nullptr;
            if (depth > 256) {
                LOG(ERROR) << "RemoveOrphanedGotos: recursion depth exceeded "
                              "(depth=" << depth << "). Possible malformed AST "
                              "or unexpectedly deep nesting — skipping subtree.\n";
                return nullptr;
            }

            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
                if (!defined.count(gs->getLabel())) {
                    LOG(ERROR) << "ORPHANED GOTO: removing 'goto "
                               << gs->getLabel()->getName()
                               << "' with no matching LabelStmt in function body. "
                                  "This may indicate a structuring rule bug that "
                                  "dropped the target label — verify emitted output.\n";
                    if (in_switch_case) {
                        return new (ctx) clang::BreakStmt(gs->getGotoLoc());
                    }
                    return new (ctx) clang::NullStmt(gs->getGotoLoc());
                }
                return nullptr;
            }

            // Track whether children are inside a switch case body.
            bool child_in_case = in_switch_case
                || llvm::isa< clang::CaseStmt >(s)
                || llvm::isa< clang::DefaultStmt >(s);

            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                std::vector< clang::Stmt * > children;
                bool changed = false;
                for (auto *child : cs->body()) {
                    auto *repl = RemoveOrphanedGotos(
                        ctx, child, defined, depth + 1, child_in_case);
                    children.push_back(repl ? repl : child);
                    if (repl) changed = true;
                }
                return changed ? detail::MakeCompound(ctx, children) : nullptr;
            }

            // Recurse into IfStmt, LabelStmt, etc. via child iteration.
            bool changed = false;
            for (auto it = s->child_begin(); it != s->child_end(); ++it) {
                if (!*it) continue;
                auto *repl = RemoveOrphanedGotos(
                    ctx, *it, defined, depth + 1, child_in_case);
                if (repl) {
                    *it = repl;
                    changed = true;
                }
            }
            return changed ? s : nullptr;
        }

        // ---------------------------------------------------------------
        // NormalizeBoolExpr — fold negations on a boolean-context expr.
        //
        //   !(a == b)  → a != b          (always safe)
        //   !(a != b)  → a == b          (always safe)
        //   !(a <  b)  → a >= b ]
        //   !(a >  b)  → a <= b ]        (only when neither operand is
        //   !(a <= b)  → a >  b ]         floating-point — NaN breaks the
        //   !(a >= b)  → a <  b ]         identity for ordered comparisons)
        //   !!x        → x
        //
        // Recurses through `&&` / `||` operands and ParenExpr so the whole
        // boolean spine of a condition is normalized.  Only ever invoked on
        // expressions used in a boolean context (if/while/do/for conditions
        // and the operands of `&&`/`||`), so dropping the `!!` wrapper — which
        // changes a bool-typed expr to its int-typed inner — is value-safe.
        // ---------------------------------------------------------------
        clang::Expr *NormalizeBoolExpr(clang::ASTContext &ctx, clang::Expr *e) {
            if (!e) {
                return e;
            }

            if (auto *pe = llvm::dyn_cast< clang::ParenExpr >(e)) {
                auto *inner = NormalizeBoolExpr(ctx, pe->getSubExpr());
                if (inner == pe->getSubExpr()) {
                    return pe;
                }
                return new (ctx) clang::ParenExpr(
                    pe->getLParen(), pe->getRParen(), inner);
            }

            if (auto *bo = llvm::dyn_cast< clang::BinaryOperator >(e)) {
                if (bo->getOpcode() == clang::BO_LAnd
                    || bo->getOpcode() == clang::BO_LOr) {
                    bo->setLHS(NormalizeBoolExpr(ctx, bo->getLHS()));
                    bo->setRHS(NormalizeBoolExpr(ctx, bo->getRHS()));
                }
                return bo;
            }

            auto *uo = llvm::dyn_cast< clang::UnaryOperator >(e);
            if (!uo || uo->getOpcode() != clang::UO_LNot) {
                return e;
            }

            clang::Expr *sub = uo->getSubExpr()->IgnoreParens();

            // !!x → x
            if (auto *inner_uo = llvm::dyn_cast< clang::UnaryOperator >(sub)) {
                if (inner_uo->getOpcode() == clang::UO_LNot) {
                    return NormalizeBoolExpr(ctx, inner_uo->getSubExpr());
                }
            }

            // !(a OP b) → a FLIP(OP) b
            if (auto *inner_bo = llvm::dyn_cast< clang::BinaryOperator >(sub)) {
                auto op = inner_bo->getOpcode();
                clang::BinaryOperatorKind flipped = op;
                bool can_flip = false;
                if (inner_bo->isEqualityOp()) {
                    flipped   = (op == clang::BO_EQ) ? clang::BO_NE : clang::BO_EQ;
                    can_flip  = true;
                } else if (inner_bo->isRelationalOp()) {
                    bool is_fp =
                        inner_bo->getLHS()->getType()->isFloatingType()
                        || inner_bo->getRHS()->getType()->isFloatingType();
                    if (!is_fp) {
                        switch (op) {
                            case clang::BO_LT: flipped = clang::BO_GE; break;
                            case clang::BO_GT: flipped = clang::BO_LE; break;
                            case clang::BO_LE: flipped = clang::BO_GT; break;
                            case clang::BO_GE: flipped = clang::BO_LT; break;
                            default: break;
                        }
                        can_flip = (flipped != op);
                    }
                }
                if (can_flip) {
                    return clang::BinaryOperator::Create(
                        ctx, inner_bo->getLHS(), inner_bo->getRHS(), flipped,
                        inner_bo->getType(), inner_bo->getValueKind(),
                        inner_bo->getObjectKind(), inner_bo->getOperatorLoc(),
                        clang::FPOptionsOverride());
                }
            }

            // No fold applies — normalize inside the `!` and keep it.
            uo->setSubExpr(NormalizeBoolExpr(ctx, uo->getSubExpr()));
            return uo;
        }

        // Walk the Stmt tree, normalizing every if/while/do/for condition.
        void NormalizeConditions(clang::ASTContext &ctx, clang::Stmt *s) {
            if (!s) {
                return;
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
                if (ifs->getCond()) {
                    ifs->setCond(NormalizeBoolExpr(ctx, ifs->getCond()));
                }
                NormalizeConditions(ctx, ifs->getThen());
                NormalizeConditions(ctx, ifs->getElse());
                return;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
                if (ws->getCond()) {
                    ws->setCond(NormalizeBoolExpr(ctx, ws->getCond()));
                }
                NormalizeConditions(ctx, ws->getBody());
                return;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(s)) {
                if (ds->getCond()) {
                    ds->setCond(NormalizeBoolExpr(ctx, ds->getCond()));
                }
                NormalizeConditions(ctx, ds->getBody());
                return;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
                if (fs->getCond()) {
                    fs->setCond(NormalizeBoolExpr(ctx, fs->getCond()));
                }
                NormalizeConditions(ctx, fs->getBody());
                return;
            }
            for (auto *child : s->children()) {
                NormalizeConditions(ctx, child);
            }
        }

    } // anonymous namespace

    void CleanupPrettyPrint(clang::FunctionDecl *fn, clang::ASTContext &ctx) {
        if (!fn || !fn->hasBody()) {
            return;
        }
        auto *body = CleanupStmtTree(ctx, fn->getBody());
        if (body) {
            fn->setBody(body);
        }

        // Eliminate gotos to immediately following labels.  Iterates
        // to handle cascading patterns.
        for (int pass = 0; pass < kMaxGotoEliminationPasses; ++pass) {
            std::unordered_set< clang::LabelDecl * > goto_targets;
            std::unordered_set< clang::Stmt * > seen;
            CollectGotoTargets(fn->getBody(), goto_targets, seen);
            body = EliminateGotoToNextLabel(ctx, fn->getBody(), &goto_targets);
            if (body) {
                fn->setBody(body);
            } else {
                break;
            }
        }

        // Scope creation + goto-to-next-label cascade.  ScopeifyIfGotos
        // converts if(c) goto L; stmts; L: → if(!c) { stmts; }, which
        // may create new goto-to-next-label adjacencies, so iterate.
        for (int pass = 0; pass < kMaxGotoEliminationPasses; ++pass) {
            auto *prev = fn->getBody();
            std::unordered_map<clang::LabelDecl *, unsigned> refs;
            CountGotoDeclRefs(fn->getBody(), refs);
            body = RepairCrossScopeLabelEntries(ctx, fn->getBody(), refs);
            if (body) {
                fn->setBody(body);
            }
            refs.clear();
            CountGotoDeclRefs(fn->getBody(), refs);
            body = FoldClangSwitchLocalCaseTargets(
                ctx, fn->getBody(), refs);
            if (body) {
                fn->setBody(body);
            }
            refs.clear();
            CountGotoDeclRefs(fn->getBody(), refs);
            body = ScopeifyIfGotos(ctx, fn->getBody(), refs);
            if (body) {
                fn->setBody(body);
            }
            refs.clear();
            CountGotoDeclRefs(fn->getBody(), refs);
            body = FoldConditionalFallthroughChains(
                ctx, fn->getBody(), refs);
            if (body) {
                fn->setBody(body);
            }
            refs.clear();
            CountGotoDeclRefs(fn->getBody(), refs);
            body = SinkCommonTerminalEpilogues(
                ctx, fn->getBody(), refs);
            if (body) {
                fn->setBody(body);
            }
            std::unordered_set< clang::LabelDecl * > goto_targets;
            std::unordered_set< clang::Stmt * > seen;
            CollectGotoTargets(fn->getBody(), goto_targets, seen);
            body = EliminateGotoToNextLabel(ctx, fn->getBody(), &goto_targets);
            if (body) {
                fn->setBody(body);
            }
            refs.clear();
            CountGotoDeclRefs(fn->getBody(), refs);
            body = FoldClangSwitchLocalCaseTargets(
                ctx, fn->getBody(), refs);
            if (body) {
                fn->setBody(body);
            }
            if (fn->getBody() == prev) {
                break;
            }
        }

        // Remove labels that are not the target of any goto.
        // Run after CleanupStmtTree which may convert gotos to break/continue.
        std::unordered_set< clang::LabelDecl * > goto_targets;
        std::unordered_set< clang::Stmt * > seen;
        CollectGotoTargets(fn->getBody(), goto_targets, seen);
        body = RemoveDeadLabels(ctx, fn->getBody(), goto_targets);
        if (body) {
            fn->setBody(body);
        }

        // Remove gotos whose target label was never emitted (orphaned
        // by structuring rules that absorbed the target block).
        std::unordered_set< clang::LabelDecl * > defined;
        CollectDefinedLabels(fn->getBody(), defined);
        body = RemoveOrphanedGotos(ctx, fn->getBody(), defined);
        if (body) {
            fn->setBody(body);
        }

        // Final pass: remove empty CompoundStmts and NullStmts.
        body = RemoveEmptyBlocks(ctx, fn->getBody());
        if (body) {
            fn->setBody(body);
        }

        // Cosmetic: fold double negations and `!(a OP b)` comparisons in
        // if/while/do/for conditions.  Runs last — purely a readability
        // pass, no effect on goto/label structure.
        NormalizeConditions(ctx, fn->getBody());
    }

} // namespace patchestry::ast
