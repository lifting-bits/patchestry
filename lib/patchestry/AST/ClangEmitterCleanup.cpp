/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/ClangEmitter.hpp>
#include <patchestry/AST/Utils.hpp>

#include <functional>
#include <string>
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
                    return new (ctx) clang::BreakStmt(clang::SourceLocation());
                if (!continue_label.empty() && name == continue_label)
                    return new (ctx) clang::ContinueStmt(clang::SourceLocation());
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
            return r ? r : new (ctx) clang::NullStmt(clang::SourceLocation());
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
                std::function< bool(clang::Stmt *) > check;
                check = [&](clang::Stmt *s) -> bool {
                    if (!s) return false;
                    if (llvm::isa< clang::ReturnStmt >(s)
                        || llvm::isa< clang::GotoStmt >(s)
                        || llvm::isa< clang::BreakStmt >(s)
                        || llvm::isa< clang::ContinueStmt >(s))
                        return true;
                    if (auto *cs_inner = llvm::dyn_cast< clang::CompoundStmt >(s))
                        return !cs_inner->body_empty() && check(cs_inner->body_back());
                    if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s))
                        return ifs->getThen() && ifs->getElse()
                            && check(ifs->getThen()) && check(ifs->getElse());
                    return false;
                };
                return check(st);
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
    static std::pair< clang::LabelDecl *, clang::Expr * >
    ExtractIfGotoPattern(clang::Stmt *s) {
        auto *ifs = llvm::dyn_cast_or_null< clang::IfStmt >(s);
        if (!ifs || ifs->getElse()) {
            return { nullptr, nullptr };
        }
        auto *then_s = ifs->getThen();
        if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(then_s)) {
            if (cs->size() == 1) {
                then_s = cs->body_front();
            }
        }
        auto *gs = llvm::dyn_cast_or_null< clang::GotoStmt >(then_s);
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
            return r ? r : new (ctx) clang::NullStmt(clang::SourceLocation());
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
                        clang::OK_Ordinary, clang::SourceLocation(),
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
                ifs->setElse(safe(RemoveEmptyBlocks(ctx, ifs->getElse())));
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

        /// Check if an IfStmt has a goto to `target` in one of its arms.
        /// Returns: 0=no match, 1=else arm matches, 2=then arm matches.
        int IfStmtGotoArm(clang::IfStmt *ifs, llvm::StringRef target) {
            if (!ifs) {
                return 0;
            }
            auto et = GotoElimGetTarget(ifs->getElse());
            if (!et.empty() && et == target) {
                return 1;
            }
            auto tt = GotoElimGetTarget(ifs->getThen());
            if (!tt.empty() && tt == target && ifs->getElse()) {
                return 2;
            }
            return 0;
        }

        /// Recursively check whether stmt tree contains any LabelStmt.
        bool ContainsAnyLabelStmt(clang::Stmt *s) {
            if (!s) {
                return false;
            }
            if (llvm::isa< clang::LabelStmt >(s)) {
                return true;
            }
            for (auto *child : s->children()) {
                if (ContainsAnyLabelStmt(child)) {
                    return true;
                }
            }
            return false;
        }

        /// Process a CompoundStmt: for each pair of adjacent stmts where
        /// the second is a LabelStmt, check if the first's deepest trailing
        /// stmt is a goto to that label.  Returns new stmt if changed.
        clang::Stmt *EliminateGotoToNextLabel(clang::ASTContext &ctx, clang::Stmt *s);

        clang::Stmt *ProcessCompound(clang::ASTContext &ctx, clang::CompoundStmt *cs) {
            std::vector< clang::Stmt * > body(cs->body_begin(), cs->body_end());

            // First: recurse into all children
            for (auto *&child : body) {
                child = EliminateGotoToNextLabel(ctx, child);
            }

            // Then: find goto-to-next-label patterns
            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t i = 0; i + 1 < body.size(); ++i) {
                    auto next_label = GotoElimGetLabel(body[i + 1]);
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
                                            clang::NullStmt(clang::SourceLocation());
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
                            return new (ctx) clang::NullStmt(clang::SourceLocation());
                        };
                        body[i] = strip_tail(body[i]);
                        changed = true;
                        break;
                    }

                    // Pattern 2: deepest trailing is IfStmt with goto arm
                    if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(deep)) {
                        int arm = IfStmtGotoArm(ifs, next_label);
                        if (arm == 1) {
                            // else goto L; L: → drop else
                            auto loc     = ifs->getIfLoc();
                            auto *new_if = clang::IfStmt::Create(
                                ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                                ifs->getCond(), loc, loc, ifs->getThen(), loc, nullptr
                            );
                            if (deep == body[i]) {
                                body[i] = new_if;
                            } else {
                                std::function< clang::Stmt *(clang::Stmt *) > replace_deep;
                                replace_deep = [&](clang::Stmt *st) -> clang::Stmt * {
                                    if (auto *inner =
                                            llvm::dyn_cast< clang::CompoundStmt >(st)) {
                                        std::vector< clang::Stmt * > b(
                                            inner->body_begin(), inner->body_end()
                                        );
                                        if (!b.empty()
                                            && DeepTrailingStmt(b.back()) == deep)
                                        {
                                            b.back() = replace_deep(b.back());
                                        }
                                        return detail::MakeCompound(ctx, b);
                                    }
                                    if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st))
                                    {
                                        ls->setSubStmt(replace_deep(ls->getSubStmt()));
                                        return st;
                                    }
                                    return new_if;
                                };
                                body[i] = replace_deep(body[i]);
                            }
                            changed = true;
                            break;
                        }
                        if (arm == 2) {
                            // if(c) goto L; else S; L: → if(!c) S
                            auto *neg    = NegateExpr(ctx, ifs->getCond());
                            auto loc     = ifs->getIfLoc();
                            auto *new_if = clang::IfStmt::Create(
                                ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                                neg, loc, loc, ifs->getElse(), loc, nullptr
                            );
                            if (deep == body[i]) {
                                body[i] = new_if;
                            } else {
                                std::function< clang::Stmt *(clang::Stmt *) > replace_deep;
                                replace_deep = [&](clang::Stmt *st) -> clang::Stmt * {
                                    if (auto *inner =
                                            llvm::dyn_cast< clang::CompoundStmt >(st)) {
                                        std::vector< clang::Stmt * > b(
                                            inner->body_begin(), inner->body_end()
                                        );
                                        if (!b.empty()
                                            && DeepTrailingStmt(b.back()) == deep)
                                        {
                                            b.back() = replace_deep(b.back());
                                        }
                                        return detail::MakeCompound(ctx, b);
                                    }
                                    if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st))
                                    {
                                        ls->setSubStmt(replace_deep(ls->getSubStmt()));
                                        return st;
                                    }
                                    return new_if;
                                };
                                body[i] = replace_deep(body[i]);
                            }
                            changed = true;
                            break;
                        }
                    }
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
                            return new (ctx) clang::BreakStmt(clang::SourceLocation());
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

        clang::Stmt *EliminateGotoToNextLabel(clang::ASTContext &ctx, clang::Stmt *s) {
            if (!s) {
                return s;
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                return ProcessCompound(ctx, cs);
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
                ifs->setThen(EliminateGotoToNextLabel(ctx, ifs->getThen()));
                if (ifs->getElse()) {
                    ifs->setElse(EliminateGotoToNextLabel(ctx, ifs->getElse()));
                }
                return s;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
                ws->setBody(EliminateGotoToNextLabel(ctx, ws->getBody()));
                return s;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(s)) {
                ds->setBody(EliminateGotoToNextLabel(ctx, ds->getBody()));
                return s;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
                fs->setBody(EliminateGotoToNextLabel(ctx, fs->getBody()));
                return s;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
                ls->setSubStmt(EliminateGotoToNextLabel(ctx, ls->getSubStmt()));
                return s;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(s)) {
                sw->setBody(EliminateGotoToNextLabel(ctx, sw->getBody()));
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

        clang::Stmt *ScopeifyIfGotos(clang::ASTContext &ctx, clang::Stmt *s) {
            if (!s) {
                return s;
            }

            // Recurse into structured bodies first
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
                ifs->setThen(ScopeifyIfGotos(ctx, ifs->getThen()));
                if (ifs->getElse()) {
                    ifs->setElse(ScopeifyIfGotos(ctx, ifs->getElse()));
                }
                return s;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
                ws->setBody(ScopeifyIfGotos(ctx, ws->getBody()));
                return s;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(s)) {
                ds->setBody(ScopeifyIfGotos(ctx, ds->getBody()));
                return s;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
                fs->setBody(ScopeifyIfGotos(ctx, fs->getBody()));
                return s;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
                ls->setSubStmt(ScopeifyIfGotos(ctx, ls->getSubStmt()));
                return s;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(s)) {
                sw->setBody(ScopeifyIfGotos(ctx, sw->getBody()));
                return s;
            }

            auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s);
            if (!cs) {
                return s;
            }

            std::vector< clang::Stmt * > body(cs->body_begin(), cs->body_end());

            // Recurse into children first
            for (auto *&child : body) {
                child = ScopeifyIfGotos(ctx, child);
            }

            // Find if(c) goto L; ... L: patterns
            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t i = 0; i < body.size(); ++i) {
                    auto *ifs = llvm::dyn_cast< clang::IfStmt >(body[i]);
                    if (!ifs || ifs->getElse()) {
                        continue;
                    }
                    auto *gs = llvm::dyn_cast_or_null< clang::GotoStmt >(ifs->getThen());
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
                        if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(body[j])) {
                            if (ls->getDecl()->getName() == target_name) {
                                label_idx = j;
                                break;
                            }
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

                    // Check: no intermediate labels at any depth. Labels are
                    // function-scope goto targets; moving nested labels into
                    // a conditional block can change control-flow semantics.
                    bool has_label = false;
                    for (size_t j = i + 1; j < label_idx; ++j) {
                        if (ContainsAnyLabelStmt(body[j])) {
                            has_label = true;
                            break;
                        }
                    }
                    if (has_label) {
                        continue;
                    }

                    // Collect intermediate stmts
                    std::vector< clang::Stmt * > scoped;
                    for (size_t j = i + 1; j < label_idx; ++j) {
                        scoped.push_back(body[j]);
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

        // Collect all LabelDecls that have a LabelStmt definition in the tree.
        void CollectDefinedLabels(clang::Stmt *s,
                                  std::unordered_set< clang::LabelDecl * > &defined,
                                  std::unordered_set< clang::Stmt * > &seen) {
            if (!s || !seen.insert(s).second) return;
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
                defined.insert(ls->getDecl());
            }
            for (auto *child : s->children()) {
                CollectDefinedLabels(child, defined, seen);
            }
        }

        // Replace GotoStmts whose target label has no LabelStmt in the
        // function body with NullStmt.  Returns a new tree if changes
        // were made, nullptr otherwise.
        clang::Stmt *RemoveOrphanedGotos(
            clang::ASTContext &ctx, clang::Stmt *s,
            const std::unordered_set< clang::LabelDecl * > &defined
        ) {
            if (!s) return nullptr;

            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
                if (!defined.count(gs->getLabel())) {
                    return new (ctx) clang::NullStmt(gs->getGotoLoc());
                }
                return nullptr;
            }

            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                std::vector< clang::Stmt * > children;
                bool changed = false;
                for (auto *child : cs->body()) {
                    auto *repl = RemoveOrphanedGotos(ctx, child, defined);
                    children.push_back(repl ? repl : child);
                    if (repl) changed = true;
                }
                return changed ? detail::MakeCompound(ctx, children) : nullptr;
            }

            // Recurse into IfStmt, LabelStmt, etc. via child iteration.
            bool changed = false;
            for (auto it = s->child_begin(); it != s->child_end(); ++it) {
                if (!*it) continue;
                auto *repl = RemoveOrphanedGotos(ctx, *it, defined);
                if (repl) {
                    *it = repl;
                    changed = true;
                }
            }
            return changed ? s : nullptr;
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
            body = EliminateGotoToNextLabel(ctx, fn->getBody());
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
            body       = ScopeifyIfGotos(ctx, fn->getBody());
            if (body) {
                fn->setBody(body);
            }
            body = EliminateGotoToNextLabel(ctx, fn->getBody());
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
        {
            std::unordered_set< clang::LabelDecl * > defined;
            std::unordered_set< clang::Stmt * > seen2;
            CollectDefinedLabels(fn->getBody(), defined, seen2);
            body = RemoveOrphanedGotos(ctx, fn->getBody(), defined);
            if (body) {
                fn->setBody(body);
            }
        }

        // Final pass: remove empty CompoundStmts and NullStmts.
        body = RemoveEmptyBlocks(ctx, fn->getBody());
        if (body) {
            fn->setBody(body);
        }
    }

} // namespace patchestry::ast
