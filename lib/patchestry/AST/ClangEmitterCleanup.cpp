/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/ClangEmitter.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Util/Log.hpp>

#include <algorithm>
#include <cctype>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/PrettyPrinter.h>
#include <clang/AST/Stmt.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

namespace patchestry::ast {

    namespace detail {
        static clang::CompoundStmt *
        MakeCompound(clang::ASTContext &ctx, const std::vector< clang::Stmt * > &stmts) {
            auto loc = VirtualLoc(ctx);
            return clang::CompoundStmt::Create(
                ctx, stmts, clang::FPOptionsOverride(), loc, loc
            );
        }
    } // namespace detail

    // Collect all LabelDecls referenced by GotoStmts in a Stmt tree.
    static void CollectGotoTargets(
        clang::Stmt *s, std::unordered_set< clang::LabelDecl * > &targets,
        std::unordered_set< clang::Stmt * > &seen
    ) {
        if (!s || !seen.insert(s).second) { return; }
        if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
            targets.insert(gs->getLabel());
            return;
        }
        for (auto *child : s->children()) { CollectGotoTargets(child, targets, seen); }
    }

    // ---- Pretty-print cleanup (patchir-decomp only) ----

    namespace {
        // If stmt is a LabelStmt wrapping a CompoundStmt, push the label inside:
        //   LabelStmt(CompoundStmt{s1, s2, ...}) → CompoundStmt{LabelStmt(s1), s2, ...}
        // Otherwise return the stmt unchanged.
        clang::Stmt *PushLabelInside(clang::ASTContext &ctx, clang::Stmt *s) {
            auto *ls = llvm::dyn_cast_or_null< clang::LabelStmt >(s);
            if (!ls) { return s; }
            auto *inner = llvm::dyn_cast_or_null< clang::CompoundStmt >(ls->getSubStmt());
            if (!inner || inner->body_empty()) { return s; }

            auto it = inner->body_begin();
            ls->setSubStmt(*it);
            std::vector< clang::Stmt * > stmts;
            stmts.push_back(ls);
            for (++it; it != inner->body_end(); ++it) { stmts.push_back(*it); }
            return detail::MakeCompound(ctx, stmts);
        }

        // Replace a trailing GotoStmt in a case body with break or continue.
        // Returns the modified stmt, or the original if no replacement was made.
        clang::Stmt *ReplaceTrailingGoto(
            clang::ASTContext &ctx, clang::Stmt *s, const std::string &break_label,
            const std::string &continue_label
        ) {
            if (!s) { return s; }

            // Direct GotoStmt
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
                std::string name = gs->getLabel()->getName().str();
                if (!break_label.empty() && name == break_label) {
                    return new (ctx) clang::BreakStmt(VirtualLoc(ctx));
                }
                if (!continue_label.empty() && name == continue_label) {
                    return new (ctx) clang::ContinueStmt(VirtualLoc(ctx));
                }
                return s;
            }

            // CompoundStmt — check/replace last stmt
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                if (cs->body_empty()) { return s; }
                auto *last     = *(cs->body_end() - 1);
                auto *replaced = ReplaceTrailingGoto(ctx, last, break_label, continue_label);
                if (replaced == last) { return s; }

                std::vector< clang::Stmt * > stmts;
                for (auto it = cs->body_begin(); std::next(it) != cs->body_end(); ++it) {
                    stmts.push_back(*it);
                }
                stmts.push_back(replaced);
                return detail::MakeCompound(ctx, stmts);
            }

            return s;
        }

        // Walk case/default bodies in a SwitchStmt and convert trailing gotos
        // to break (if targeting break_label) or continue (if targeting continue_label).
        void ConvertSwitchCaseGotos(
            clang::ASTContext &ctx, clang::SwitchStmt *sw, const std::string &break_label,
            const std::string &continue_label
        ) {
            auto *body = sw->getBody();
            auto *cs   = llvm::dyn_cast_or_null< clang::CompoundStmt >(body);
            if (!cs) { return; }

            for (auto *child : cs->body()) {
                if (auto *case_s = llvm::dyn_cast< clang::CaseStmt >(child)) {
                    auto *sub = case_s->getSubStmt();
                    auto *r   = ReplaceTrailingGoto(ctx, sub, break_label, continue_label);
                    if (r != sub) { case_s->setSubStmt(r); }
                } else if (auto *def_s = llvm::dyn_cast< clang::DefaultStmt >(child)) {
                    auto *sub = def_s->getSubStmt();
                    auto *r   = ReplaceTrailingGoto(ctx, sub, break_label, continue_label);
                    if (r != sub) { def_s->setSubStmt(r); }
                }
            }
        }

        // Check if ALL case/default bodies in a switch end with goto to the same
        // label (and that label is NOT the break/continue label). Returns the
        // common label name, or empty string if not uniform.
        std::string FindCommonTrailingGoto(clang::SwitchStmt *sw) {
            auto *body = sw->getBody();
            auto *cs   = llvm::dyn_cast_or_null< clang::CompoundStmt >(body);
            if (!cs) { return {}; }

            std::string common;
            auto getTrailingGotoLabel = [](clang::Stmt *s) -> std::string {
                if (!s) { return {}; }
                if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
                    return gs->getLabel()->getName().str();
                }
                if (auto *c = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                    if (!c->body_empty()) {
                        if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(c->body_back())) {
                            return gs->getLabel()->getName().str();
                        }
                    }
                }
                return {};
            };

            for (auto *child : cs->body()) {
                clang::Stmt *sub = nullptr;
                if (auto *case_s = llvm::dyn_cast< clang::CaseStmt >(child)) {
                    sub = case_s->getSubStmt();
                } else if (auto *def_s = llvm::dyn_cast< clang::DefaultStmt >(child)) {
                    sub = def_s->getSubStmt();
                } else {
                    continue;
                }

                auto label = getTrailingGotoLabel(sub);
                if (label.empty()) { return {}; }
                if (common.empty()) {
                    common = label;
                } else if (common != label) {
                    return {};
                }
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
        clang::Stmt *CleanupStmtTree(
            clang::ASTContext &ctx, clang::Stmt *s, const std::string &continue_label = ""
        ) {
            if (!s) { return nullptr; }

            // Handle IfStmt: recurse into then/else, push labels inside
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
                ifs->setThen(
                    PushLabelInside(ctx, CleanupStmtTree(ctx, ifs->getThen(), continue_label))
                );
                if (ifs->getElse()) {
                    ifs->setElse(PushLabelInside(
                        ctx, CleanupStmtTree(ctx, ifs->getElse(), continue_label)
                    ));
                }
                return s;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
                ws->setBody(
                    PushLabelInside(ctx, CleanupStmtTree(ctx, ws->getBody(), continue_label))
                );
                return s;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(s)) {
                ds->setBody(
                    PushLabelInside(ctx, CleanupStmtTree(ctx, ds->getBody(), continue_label))
                );
                return s;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
                fs->setBody(
                    PushLabelInside(ctx, CleanupStmtTree(ctx, fs->getBody(), continue_label))
                );
                return s;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
                // If this label wraps a loop, set it as the continue target
                auto *sub = ls->getSubStmt();
                std::string new_cont;
                if (sub
                    && (llvm::isa< clang::WhileStmt >(sub) || llvm::isa< clang::DoStmt >(sub)
                        || llvm::isa< clang::ForStmt >(sub)))
                {
                    new_cont = ls->getDecl()->getName().str();
                }
                ls->setSubStmt(
                    CleanupStmtTree(ctx, sub, new_cont.empty() ? continue_label : new_cont)
                );
                return s;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(s)) {
                sw->setBody(CleanupStmtTree(ctx, sw->getBody(), continue_label));
                return s;
            }
            if (auto *cs_node = llvm::dyn_cast< clang::CaseStmt >(s)) {
                cs_node->setSubStmt(
                    CleanupStmtTree(ctx, cs_node->getSubStmt(), continue_label)
                );
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
                    if (!cleaned) { continue; }

                    // Flatten nested CompoundStmts
                    if (auto *inner_cs = llvm::dyn_cast< clang::CompoundStmt >(cleaned)) {
                        for (auto *gc : inner_cs->body()) { children.push_back(gc); }
                    }
                    // Push label inside compound
                    else if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(cleaned))
                    {
                        if (auto *lcs = llvm::dyn_cast< clang::CompoundStmt >(ls->getSubStmt()))
                        {
                            auto it = lcs->body_begin();
                            if (it != lcs->body_end()) {
                                ls->setSubStmt(*it);
                                children.push_back(ls);
                                for (++it; it != lcs->body_end(); ++it) {
                                    children.push_back(*it);
                                }
                            } else {
                                children.push_back(ls);
                            }
                        } else {
                            children.push_back(cleaned);
                        }
                    } else {
                        children.push_back(cleaned);
                    }
                }

                // --- Second pass: convert gotos in switch case bodies ---
                for (size_t i = 0; i < children.size(); ++i) {
                    auto *sw = llvm::dyn_cast< clang::SwitchStmt >(children[i]);
                    if (!sw) { continue; }

                    // Find label immediately after switch → break target
                    std::string break_label;
                    if (i + 1 < children.size()) {
                        if (auto *next_ls = llvm::dyn_cast< clang::LabelStmt >(children[i + 1]))
                        {
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
                                    if (!st || target_decl) { return; }
                                    if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(st)) {
                                        if (gs->getLabel()->getName().str() == common) {
                                            target_decl = gs->getLabel();
                                        }
                                        return;
                                    }
                                    if (auto *ls2 = llvm::dyn_cast< clang::LabelStmt >(st)) {
                                        if (ls2->getDecl()->getName().str() == common) {
                                            target_decl = ls2->getDecl();
                                        }
                                    }
                                    for (auto *c : st->children()) { findLabel(c); }
                                };
                            // Scan the entire children vector for the label
                            for (auto *c : children) { findLabel(c); }
                            if (target_decl) {
                                auto loc = VirtualLoc(ctx);
                                auto *hoisted_goto =
                                    new (ctx) clang::GotoStmt(target_decl, loc, loc);
                                children.insert(
                                    children.begin() + static_cast< long >(i) + 1, hoisted_goto
                                );
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
    static clang::Stmt *RemoveDeadLabels(
        clang::ASTContext &ctx, clang::Stmt *s,
        const std::unordered_set< clang::LabelDecl * > &live
    ) {
        if (!s) { return nullptr; }

        // Guarantee a non-null Stmt* for set* methods that require one.
        auto safe = [&](clang::Stmt *r) -> clang::Stmt * {
            return r ? r : new (ctx) clang::NullStmt(VirtualLoc(ctx));
        };

        if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(s)) {
            auto *sub = RemoveDeadLabels(ctx, ls->getSubStmt(), live);
            if (!live.count(ls->getDecl())) { return sub; }
            ls->setSubStmt(safe(sub));
            return ls;
        }

        if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
            std::vector< clang::Stmt * > children;
            for (auto *child : cs->body()) {
                auto *cleaned = RemoveDeadLabels(ctx, child, live);
                if (cleaned) { children.push_back(cleaned); }
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
                    if (!st || has_live_label) { return; }
                    if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st)) {
                        if (live.count(ls->getDecl())) { has_live_label = true; }
                    }
                    for (auto *c : st->children()) { check(c); }
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
            if (ifs->getElse()) {
                ifs->setElse(safe(RemoveDeadLabels(ctx, ifs->getElse(), live)));
            }
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
        if (!cs || cs->size() != 1) { return s; }
        return cs->body_front();
    }

    static clang::GotoStmt *AsGotoStmt(clang::Stmt *s) {
        return llvm::dyn_cast_or_null< clang::GotoStmt >(UnwrapSingleCompoundStmt(s));
    }

    static clang::Expr *ParenConditionOperand(clang::ASTContext &ctx, clang::Expr *expr) {
        return new (ctx)
            clang::ParenExpr(VirtualLoc(ctx), VirtualLoc(ctx), EnsureRValue(ctx, expr));
    }

    static std::pair< clang::LabelDecl *, clang::Expr * > ExtractIfGotoPattern(clang::Stmt *s) {
        auto *ifs = llvm::dyn_cast_or_null< clang::IfStmt >(s);
        if (!ifs || ifs->getElse()) { return { nullptr, nullptr }; }
        auto *gs = AsGotoStmt(ifs->getThen());
        if (!gs) { return { nullptr, nullptr }; }
        return { gs->getLabel(), ifs->getCond() };
    }

    // Recursively remove empty CompoundStmts, NullStmts, and merge
    // consecutive if(c1) goto L; if(c2) goto L; into if(c1||c2) goto L;
    static clang::Stmt *RemoveEmptyBlocks(clang::ASTContext &ctx, clang::Stmt *s) {
        if (!s) { return nullptr; }

        auto safe = [&](clang::Stmt *r) -> clang::Stmt * {
            return r ? r : new (ctx) clang::NullStmt(VirtualLoc(ctx));
        };

        auto contains_decl_stmt = [](clang::Stmt *stmt) {
            std::function< bool(clang::Stmt *) > walk = [&](clang::Stmt *cur) -> bool {
                if (!cur) { return false; }
                if (llvm::isa< clang::DeclStmt >(cur)) { return true; }
                for (clang::Stmt *sub : cur->children()) {
                    if (walk(sub)) { return true; }
                }
                return false;
            };
            return walk(stmt);
        };

        auto contains_label_stmt = [](clang::Stmt *stmt) {
            std::function< bool(clang::Stmt *) > walk = [&](clang::Stmt *cur) -> bool {
                if (!cur) { return false; }
                if (llvm::isa< clang::LabelStmt >(cur) || llvm::isa< clang::CaseStmt >(cur)
                    || llvm::isa< clang::DefaultStmt >(cur))
                {
                    return true;
                }
                for (clang::Stmt *sub : cur->children()) {
                    if (walk(sub)) { return true; }
                }
                return false;
            };
            return walk(stmt);
        };

        auto append_stmt_sequence = [&](clang::Stmt *stmt, std::vector< clang::Stmt * > &out) {
            if (auto *compound = llvm::dyn_cast_or_null< clang::CompoundStmt >(stmt)) {
                for (clang::Stmt *nested : compound->body()) { out.push_back(nested); }
            } else if (stmt && !llvm::isa< clang::NullStmt >(stmt)) {
                out.push_back(stmt);
            }
        };

        auto flatten_if_else_after_terminator = [&](clang::IfStmt *ifs, clang::Stmt *then_stmt,
                                                    clang::Stmt *else_stmt) -> clang::Stmt * {
            if (!ifs || !else_stmt || ifs->getInit() || ifs->getConditionVariable()
                || !detail::EndsWithTerminator(then_stmt) || contains_decl_stmt(else_stmt))
            {
                return nullptr;
            }

            auto *flattened_if = clang::IfStmt::Create(
                ctx, ifs->getIfLoc(), ifs->getStatementKind(), nullptr, nullptr, ifs->getCond(),
                ifs->getLParenLoc(), ifs->getRParenLoc(), then_stmt
            );
            std::vector< clang::Stmt * > stmts;
            stmts.push_back(flattened_if);
            append_stmt_sequence(else_stmt, stmts);
            return detail::MakeCompound(ctx, stmts);
        };

        auto fold_else_leading_terminating_if = [&](clang::Stmt *else_stmt) -> clang::Stmt * {
            auto *compound = llvm::dyn_cast_or_null< clang::CompoundStmt >(else_stmt);
            if (!compound || compound->size() < 2) { return nullptr; }

            std::vector< clang::Stmt * > children(compound->body_begin(), compound->body_end());
            auto *lead = llvm::dyn_cast_or_null< clang::IfStmt >(children.front());
            if (!lead || lead->getElse() || lead->getInit() || lead->getConditionVariable()
                || !detail::EndsWithTerminator(lead->getThen()))
            {
                return nullptr;
            }

            std::vector< clang::Stmt * > rest(std::next(children.begin()), children.end());
            auto *rest_stmt = rest.size() == 1 ? rest.front() : detail::MakeCompound(ctx, rest);
            if (contains_decl_stmt(rest_stmt) || contains_label_stmt(rest_stmt)) {
                return nullptr;
            }

            return clang::IfStmt::Create(
                ctx, lead->getIfLoc(), lead->getStatementKind(), nullptr, nullptr,
                lead->getCond(), lead->getLParenLoc(), lead->getRParenLoc(), lead->getThen(),
                VirtualLoc(ctx), rest_stmt
            );
        };

        if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
            std::vector< clang::Stmt * > children;
            for (auto *child : cs->body()) {
                auto *cleaned = RemoveEmptyBlocks(ctx, child);
                if (!cleaned) { continue; }
                if (llvm::isa< clang::NullStmt >(cleaned)) { continue; }
                if (auto *inner = llvm::dyn_cast< clang::CompoundStmt >(cleaned)) {
                    if (inner->body_empty()) { continue; }
                    if (!contains_decl_stmt(inner)) {
                        append_stmt_sequence(inner, children);
                        continue;
                    }
                }

                if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(cleaned)) {
                    if (auto *flattened = flatten_if_else_after_terminator(
                            ifs, ifs->getThen(), ifs->getElse()
                        ))
                    {
                        append_stmt_sequence(flattened, children);
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
                        ctx, ParenConditionOperand(ctx, c1), ParenConditionOperand(ctx, c2),
                        clang::BO_LOr, ctx.BoolTy, clang::VK_PRValue, clang::OK_Ordinary,
                        VirtualLoc(ctx), clang::FPOptionsOverride()
                    );
                    llvm::cast< clang::IfStmt >(children[i])->setCond(merged);
                    children.erase(children.begin() + static_cast< long >(i) + 1);
                    // Don't advance i — re-check for third consecutive
                }
            }

            return detail::MakeCompound(ctx, children);
        }
        if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
            auto stmt_is_empty = [](clang::Stmt *stmt) -> bool {
                auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(stmt);
                return !stmt || llvm::isa< clang::NullStmt >(stmt) || (cs && cs->body_empty());
            };

            auto *new_then = safe(RemoveEmptyBlocks(ctx, ifs->getThen()));
            ifs->setThen(new_then);
            if (ifs->getElse()) {
                auto *new_else  = safe(RemoveEmptyBlocks(ctx, ifs->getElse()));
                // Drop an else clause that cleaned up to nothing — the
                // pretty-printer would otherwise render `else { }`.
                bool else_empty = stmt_is_empty(new_else);
                if (else_empty) {
                    return clang::IfStmt::Create(
                        ctx, ifs->getIfLoc(), ifs->getStatementKind(), ifs->getInit(),
                        ifs->getConditionVariable(), ifs->getCond(), ifs->getLParenLoc(),
                        ifs->getRParenLoc(), ifs->getThen()
                    );
                }
                // Prefer `if (!cond) { body }` over `if (cond) ; else { body }`.
                // Keep this to plain if-statements so we do not move a scoped
                // init/condition variable outside its original shape.
                if (stmt_is_empty(new_then) && !ifs->getInit() && !ifs->getConditionVariable())
                {
                    return clang::IfStmt::Create(
                        ctx, ifs->getIfLoc(), ifs->getStatementKind(), nullptr, nullptr,
                        NegateExpr(ctx, ifs->getCond()), ifs->getLParenLoc(),
                        ifs->getRParenLoc(), new_else
                    );
                }
                if (auto *flattened = flatten_if_else_after_terminator(ifs, new_then, new_else))
                {
                    return flattened;
                }
                if (auto *inner = llvm::dyn_cast_or_null< clang::IfStmt >(
                        UnwrapSingleCompoundStmt(new_then)
                    ))
                {
                    auto *inner_goto = AsGotoStmt(inner->getThen());
                    auto *else_goto  = AsGotoStmt(new_else);
                    if (!ifs->getInit() && !ifs->getConditionVariable() && !inner->getInit()
                        && !inner->getConditionVariable() && !inner->getElse() && inner_goto
                        && else_goto
                        && inner_goto->getLabel()->getName()
                            == else_goto->getLabel()->getName())
                    {
                        auto *merged = clang::BinaryOperator::Create(
                            ctx, ParenConditionOperand(ctx, NegateExpr(ctx, ifs->getCond())),
                            ParenConditionOperand(ctx, inner->getCond()), clang::BO_LOr,
                            ctx.BoolTy, clang::VK_PRValue, clang::OK_Ordinary, VirtualLoc(ctx),
                            clang::FPOptionsOverride()
                        );
                        return clang::IfStmt::Create(
                            ctx, ifs->getIfLoc(), ifs->getStatementKind(), nullptr, nullptr,
                            merged, ifs->getLParenLoc(), ifs->getRParenLoc(), inner->getThen()
                        );
                    }
                }
                if (auto *folded = fold_else_leading_terminating_if(new_else)) {
                    new_else = safe(RemoveEmptyBlocks(ctx, folded));
                }
                if (auto *inner = llvm::dyn_cast_or_null< clang::IfStmt >(
                        UnwrapSingleCompoundStmt(new_else)
                    ))
                {
                    auto *outer_goto = AsGotoStmt(ifs->getThen());
                    auto *inner_goto = AsGotoStmt(inner->getThen());
                    if (!ifs->getInit() && !ifs->getConditionVariable() && !inner->getInit()
                        && !inner->getConditionVariable() && outer_goto && inner_goto
                        && outer_goto->getLabel()->getName()
                            == inner_goto->getLabel()->getName())
                    {
                        auto *merged = clang::BinaryOperator::Create(
                            ctx, ParenConditionOperand(ctx, ifs->getCond()),
                            ParenConditionOperand(ctx, inner->getCond()), clang::BO_LOr,
                            ctx.BoolTy, clang::VK_PRValue, clang::OK_Ordinary, VirtualLoc(ctx),
                            clang::FPOptionsOverride()
                        );
                        return clang::IfStmt::Create(
                            ctx, ifs->getIfLoc(), ifs->getStatementKind(), nullptr, nullptr,
                            merged, ifs->getLParenLoc(), ifs->getRParenLoc(), ifs->getThen(),
                            inner->getElseLoc(), inner->getElse()
                        );
                    }

                    // Cosmetic only: `else { if (...) ... }` has the same
                    // scope and control behavior as `else if (...) ...` when
                    // the compound contains only the nested if.
                    if (new_else != inner) { new_else = inner; }
                }
                return clang::IfStmt::Create(
                    ctx, ifs->getIfLoc(), ifs->getStatementKind(), ifs->getInit(),
                    ifs->getConditionVariable(), ifs->getCond(), ifs->getLParenLoc(),
                    ifs->getRParenLoc(), ifs->getThen(), ifs->getElseLoc(), new_else
                );
            } else if (
                auto *inner =
                    llvm::dyn_cast_or_null< clang::IfStmt >(UnwrapSingleCompoundStmt(new_then))
            )
            {
                // Cosmetic only: `if (a) { if (b) body; }` becomes
                // `if (a && b) body;`.  Keep this to plain if-statements with
                // no else arm so scoped condition variables and dangling-else
                // behavior cannot change.
                if (!ifs->getInit() && !ifs->getConditionVariable() && !inner->getInit()
                    && !inner->getConditionVariable() && !inner->getElse())
                {
                    auto *merged = clang::BinaryOperator::Create(
                        ctx, ParenConditionOperand(ctx, ifs->getCond()),
                        ParenConditionOperand(ctx, inner->getCond()), clang::BO_LAnd,
                        ctx.BoolTy, clang::VK_PRValue, clang::OK_Ordinary, VirtualLoc(ctx),
                        clang::FPOptionsOverride()
                    );
                    return clang::IfStmt::Create(
                        ctx, ifs->getIfLoc(), ifs->getStatementKind(), nullptr, nullptr, merged,
                        ifs->getLParenLoc(), ifs->getRParenLoc(), inner->getThen()
                    );
                }
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
            cs_node->setSubStmt(safe(RemoveEmptyBlocks(ctx, cs_node->getSubStmt())));
            return s;
        }
        if (auto *def = llvm::dyn_cast< clang::DefaultStmt >(s)) {
            def->setSubStmt(safe(RemoveEmptyBlocks(ctx, def->getSubStmt())));
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
                if (!cs->body_empty()) { return GotoElimGetLabel(*cs->body_begin()); }
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
            if (!s) { return nullptr; }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                if (cs->body_empty()) { return nullptr; }
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
            if (!ifs) { return 0; }
            auto et = GotoElimGetTarget(DeepTrailingStmt(ifs->getElse()));
            if (!et.empty() && et == target) { return 1; }
            auto tt = GotoElimGetTarget(ifs->getThen());
            if (!tt.empty() && tt == target && ifs->getElse()) { return 2; }
            return 0;
        }

        /// Strip the trailing `goto target` from `st`, walking through
        /// the last child of nested CompoundStmts and LabelStmt
        /// sub-stmts.  Returns NullStmt when the stripped stmt itself
        /// is the matching goto.
        clang::Stmt *
        StripTrailingGoto(clang::ASTContext &ctx, clang::Stmt *st, llvm::StringRef target) {
            if (!st) { return st; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(st)) {
                if (gs->getLabel()->getName() == target) {
                    return new (ctx) clang::NullStmt(VirtualLoc(ctx));
                }
                return st;
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                if (cs->body_empty()) { return st; }
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
            clang::ASTContext &ctx, clang::Stmt *st, llvm::StringRef target, unsigned &stripped
        ) {
            if (!st) { return st; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(st)) {
                if (gs->getLabel()->getName() == target) {
                    ++stripped;
                    return new (ctx) clang::NullStmt(VirtualLoc(ctx));
                }
                return st;
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                if (cs->body_empty()) { return st; }
                std::vector< clang::Stmt * > b(cs->body_begin(), cs->body_end());
                unsigned before = stripped;
                b.back() =
                    StripGotoToFollowingLabelFromTailPosition(ctx, b.back(), target, stripped);
                return stripped != before ? detail::MakeCompound(ctx, b) : st;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st)) {
                ls->setSubStmt(StripGotoToFollowingLabelFromTailPosition(
                    ctx, ls->getSubStmt(), target, stripped
                ));
                return st;
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(st)) {
                unsigned before = stripped;
                ifs->setThen(StripGotoToFollowingLabelFromTailPosition(
                    ctx, ifs->getThen(), target, stripped
                ));
                if (ifs->getElse()) {
                    ifs->setElse(StripGotoToFollowingLabelFromTailPosition(
                        ctx, ifs->getElse(), target, stripped
                    ));
                }
                return stripped != before ? ifs : st;
            }
            return st;
        }

        bool ContinuationIsSafeToMoveBeforeJoin(clang::Stmt *st) {
            if (!st) { return true; }
            if (llvm::isa< clang::LabelStmt >(st) || llvm::isa< clang::SwitchStmt >(st)
                || llvm::isa< clang::WhileStmt >(st) || llvm::isa< clang::DoStmt >(st)
                || llvm::isa< clang::ForStmt >(st) || llvm::isa< clang::DeclStmt >(st)
                || llvm::isa< clang::ReturnStmt >(st) || llvm::isa< clang::BreakStmt >(st)
                || llvm::isa< clang::ContinueStmt >(st))
            {
                return false;
            }
            for (clang::Stmt *child : st->children()) {
                if (!ContinuationIsSafeToMoveBeforeJoin(child)) { return false; }
            }
            return true;
        }

        bool StraightLineContinuationIsSafeToMove(clang::Stmt *st) {
            if (!st) { return true; }
            if (!ContinuationIsSafeToMoveBeforeJoin(st) || llvm::isa< clang::IfStmt >(st)
                || llvm::isa< clang::GotoStmt >(st) || llvm::isa< clang::CaseStmt >(st)
                || llvm::isa< clang::DefaultStmt >(st))
            {
                return false;
            }
            for (clang::Stmt *child : st->children()) {
                if (!StraightLineContinuationIsSafeToMove(child)) { return false; }
            }
            return true;
        }

        constexpr size_t kMaxExternalContinuationCloneStmts     = 8;
        constexpr unsigned kMaxExternalContinuationFallthroughs = 4;

        size_t CountStraightLineStmtUnits(clang::Stmt *stmt) {
            if (!stmt) { return 0; }
            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                size_t count = 0;
                for (clang::Stmt *child : compound->body()) {
                    count += CountStraightLineStmtUnits(child);
                    if (count > kMaxExternalContinuationCloneStmts) { return count; }
                }
                return count;
            }
            return llvm::isa< clang::NullStmt >(stmt) ? 0 : 1;
        }

        clang::Stmt *CloneStraightLineStmt(clang::ASTContext &ctx, clang::Stmt *stmt) {
            if (!stmt) { return new (ctx) clang::NullStmt(VirtualLoc(ctx)); }
            if (llvm::isa< clang::NullStmt >(stmt)) {
                return new (ctx) clang::NullStmt(VirtualLoc(ctx));
            }
            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > cloned;
                cloned.reserve(compound->size());
                for (clang::Stmt *child : compound->body()) {
                    clang::Stmt *copy = CloneStraightLineStmt(ctx, child);
                    if (!copy) { return nullptr; }
                    cloned.push_back(copy);
                }
                return detail::MakeCompound(ctx, cloned);
            }
            if (auto *expr = llvm::dyn_cast< clang::Expr >(stmt)) {
                return CloneExpr(ctx, expr);
            }
            return nullptr;
        }

        bool StraightLineContinuationSeqIsSafe(const std::vector< clang::Stmt * > &stmts) {
            size_t count = 0;
            for (clang::Stmt *stmt : stmts) {
                if (!StraightLineContinuationIsSafeToMove(stmt)) { return false; }
                count += CountStraightLineStmtUnits(stmt);
                if (count > kMaxExternalContinuationCloneStmts) { return false; }
            }
            return count != 0;
        }

        bool ContinuationStmtIsSafeToMoveOnce(clang::Stmt *stmt) {
            if (!stmt) { return true; }
            if (llvm::isa< clang::WhileStmt >(stmt) || llvm::isa< clang::DoStmt >(stmt)
                || llvm::isa< clang::ForStmt >(stmt) || llvm::isa< clang::SwitchStmt >(stmt))
            {
                std::function< bool(clang::Stmt *) > has_label_or_goto =
                    [&](clang::Stmt *cur) -> bool {
                    if (!cur) { return false; }
                    if (llvm::isa< clang::LabelStmt >(cur) || llvm::isa< clang::GotoStmt >(cur))
                    {
                        return true;
                    }
                    for (clang::Stmt *child : cur->children()) {
                        if (has_label_or_goto(child)) { return true; }
                    }
                    return false;
                };
                return !has_label_or_goto(stmt);
            }
            if (llvm::isa< clang::LabelStmt >(stmt) || llvm::isa< clang::GotoStmt >(stmt)
                || llvm::isa< clang::DeclStmt >(stmt) || llvm::isa< clang::CaseStmt >(stmt)
                || llvm::isa< clang::DefaultStmt >(stmt) || llvm::isa< clang::BreakStmt >(stmt)
                || llvm::isa< clang::ContinueStmt >(stmt))
            {
                return false;
            }
            for (clang::Stmt *child : stmt->children()) {
                if (!ContinuationStmtIsSafeToMoveOnce(child)) { return false; }
            }
            return true;
        }

        bool ContinuationSeqIsSafeToMoveOnce(const std::vector< clang::Stmt * > &stmts) {
            if (stmts.empty()) { return false; }
            for (clang::Stmt *stmt : stmts) {
                if (!ContinuationStmtIsSafeToMoveOnce(stmt)) { return false; }
            }
            return true;
        }

        clang::Stmt *MoveContinuationSeqAsStmt(
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &stmts
        ) {
            if (stmts.empty()) { return new (ctx) clang::NullStmt(VirtualLoc(ctx)); }
            if (stmts.size() == 1) { return stmts.front(); }
            return detail::MakeCompound(ctx, stmts);
        }

        clang::Stmt *CloneContinuationSeq(
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &stmts
        ) {
            std::vector< clang::Stmt * > cloned;
            cloned.reserve(stmts.size());
            for (clang::Stmt *stmt : stmts) {
                clang::Stmt *copy = CloneStraightLineStmt(ctx, stmt);
                if (!copy) { return nullptr; }
                if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(copy)) {
                    for (clang::Stmt *child : compound->body()) { cloned.push_back(child); }
                } else if (!llvm::isa< clang::NullStmt >(copy)) {
                    cloned.push_back(copy);
                }
            }
            return cloned.empty()
                ? static_cast< clang::Stmt * >(new (ctx) clang::NullStmt(VirtualLoc(ctx)))
                : detail::MakeCompound(ctx, cloned);
        }

        unsigned CountGotosToTargetName(clang::Stmt *st, llvm::StringRef target) {
            if (!st) { return 0; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(st)) {
                return gs->getLabel()->getName() == target ? 1U : 0U;
            }
            unsigned count = 0;
            for (clang::Stmt *child : st->children()) {
                count += CountGotosToTargetName(child, target);
            }
            return count;
        }

        unsigned CountAllGotos(clang::Stmt *st) {
            if (!st) { return 0; }
            if (llvm::isa< clang::GotoStmt >(st)) { return 1; }
            unsigned count = 0;
            for (clang::Stmt *child : st->children()) { count += CountAllGotos(child); }
            return count;
        }

        bool ContainsSwitchStmt(clang::Stmt *st) {
            if (!st) { return false; }
            if (llvm::isa< clang::SwitchStmt >(st)) { return true; }
            for (clang::Stmt *child : st->children()) {
                if (ContainsSwitchStmt(child)) { return true; }
            }
            return false;
        }

        unsigned CountTailFallthroughLeaves(clang::Stmt *st, llvm::StringRef skip_target) {
            if (!st) { return 1; }
            if (llvm::isa< clang::GotoStmt >(st)) {
                (void) skip_target;
                return 0;
            }
            if (llvm::isa< clang::ReturnStmt >(st) || llvm::isa< clang::BreakStmt >(st)
                || llvm::isa< clang::ContinueStmt >(st) || llvm::isa< clang::SwitchStmt >(st)
                || llvm::isa< clang::WhileStmt >(st) || llvm::isa< clang::DoStmt >(st)
                || llvm::isa< clang::ForStmt >(st))
            {
                return 0;
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                if (cs->body_empty()) { return 1; }
                return CountTailFallthroughLeaves(cs->body_back(), skip_target);
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st)) {
                return CountTailFallthroughLeaves(ls->getSubStmt(), skip_target);
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(st)) {
                unsigned count  = CountTailFallthroughLeaves(ifs->getThen(), skip_target);
                count          += ifs->getElse()
                    ? CountTailFallthroughLeaves(ifs->getElse(), skip_target)
                    : 1U;
                return count;
            }
            return 1;
        }

        clang::Stmt *MoveContinuationIntoSingleFallthrough(
            clang::ASTContext &ctx, clang::Stmt *st, llvm::StringRef skip_target,
            clang::Stmt *continuation, unsigned &removed_gotos, unsigned &inserted
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
            if (llvm::isa< clang::ReturnStmt >(st) || llvm::isa< clang::BreakStmt >(st)
                || llvm::isa< clang::ContinueStmt >(st) || llvm::isa< clang::SwitchStmt >(st)
                || llvm::isa< clang::WhileStmt >(st) || llvm::isa< clang::DoStmt >(st)
                || llvm::isa< clang::ForStmt >(st))
            {
                return st;
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                if (cs->body_empty()) {
                    ++inserted;
                    return continuation;
                }
                std::vector< clang::Stmt * > b(cs->body_begin(), cs->body_end());
                b.back() = MoveContinuationIntoSingleFallthrough(
                    ctx, b.back(), skip_target, continuation, removed_gotos, inserted
                );
                return detail::MakeCompound(ctx, b);
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st)) {
                ls->setSubStmt(MoveContinuationIntoSingleFallthrough(
                    ctx, ls->getSubStmt(), skip_target, continuation, removed_gotos, inserted
                ));
                return ls;
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(st)) {
                clang::Stmt *new_then = MoveContinuationIntoSingleFallthrough(
                    ctx, ifs->getThen(), skip_target, continuation, removed_gotos, inserted
                );
                if (ifs->getElse()) {
                    ifs->setThen(new_then);
                    ifs->setElse(MoveContinuationIntoSingleFallthrough(
                        ctx, ifs->getElse(), skip_target, continuation, removed_gotos, inserted
                    ));
                    return ifs;
                }

                ++inserted;
                return clang::IfStmt::Create(
                    ctx, ifs->getIfLoc(), ifs->getStatementKind(), ifs->getInit(),
                    ifs->getConditionVariable(), ifs->getCond(), ifs->getLParenLoc(),
                    ifs->getRParenLoc(), new_then, ifs->getIfLoc(), continuation
                );
            }

            ++inserted;
            return detail::MakeCompound(ctx, { st, continuation });
        }

        clang::Stmt *RewriteExternalTargetGotoWithContinuation(
            clang::ASTContext &ctx, clang::Stmt *st, llvm::StringRef target,
            const std::vector< clang::Stmt * > &continuation, unsigned &removed_gotos,
            unsigned &inserted, bool &failed
        ) {
            if (!st || failed) { return st; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(st)) {
                if (gs->getLabel()->getName() == target) {
                    ++removed_gotos;
                    return new (ctx) clang::NullStmt(VirtualLoc(ctx));
                }
                return st;
            }
            if (llvm::isa< clang::ReturnStmt >(st) || llvm::isa< clang::BreakStmt >(st)
                || llvm::isa< clang::ContinueStmt >(st) || llvm::isa< clang::SwitchStmt >(st)
                || llvm::isa< clang::WhileStmt >(st) || llvm::isa< clang::DoStmt >(st)
                || llvm::isa< clang::ForStmt >(st) || llvm::isa< clang::LabelStmt >(st)
                || llvm::isa< clang::CaseStmt >(st) || llvm::isa< clang::DefaultStmt >(st))
            {
                return st;
            }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                if (cs->body_empty()) {
                    clang::Stmt *copy = CloneContinuationSeq(ctx, continuation);
                    if (!copy) {
                        failed = true;
                        return st;
                    }
                    ++inserted;
                    return copy;
                }
                std::vector< clang::Stmt * > children(cs->body_begin(), cs->body_end());
                children.back() = RewriteExternalTargetGotoWithContinuation(
                    ctx, children.back(), target, continuation, removed_gotos, inserted, failed
                );
                return failed ? st : detail::MakeCompound(ctx, children);
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(st)) {
                ifs->setThen(RewriteExternalTargetGotoWithContinuation(
                    ctx, ifs->getThen(), target, continuation, removed_gotos, inserted, failed
                ));
                if (failed) { return st; }
                if (ifs->getElse()) {
                    ifs->setElse(RewriteExternalTargetGotoWithContinuation(
                        ctx, ifs->getElse(), target, continuation, removed_gotos, inserted,
                        failed
                    ));
                    return st;
                }

                clang::Stmt *copy = CloneContinuationSeq(ctx, continuation);
                if (!copy) {
                    failed = true;
                    return st;
                }
                ++inserted;
                return clang::IfStmt::Create(
                    ctx, ifs->getIfLoc(), ifs->getStatementKind(), ifs->getInit(),
                    ifs->getConditionVariable(), ifs->getCond(), ifs->getLParenLoc(),
                    ifs->getRParenLoc(), ifs->getThen(), ifs->getIfLoc(), copy
                );
            }

            clang::Stmt *copy = CloneContinuationSeq(ctx, continuation);
            if (!copy) {
                failed = true;
                return st;
            }
            ++inserted;
            return detail::MakeCompound(ctx, { st, copy });
        }

        clang::Stmt *FoldExternalTargetContinuations(
            clang::ASTContext &ctx, clang::Stmt *stmt, llvm::StringRef target, unsigned &folded
        ) {
            if (!stmt || target.empty()) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(
                    FoldExternalTargetContinuations(ctx, ifs->getThen(), target, folded)
                );
                if (ifs->getElse()) {
                    ifs->setElse(
                        FoldExternalTargetContinuations(ctx, ifs->getElse(), target, folded)
                    );
                }
                return ifs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(
                    FoldExternalTargetContinuations(ctx, ls->getSubStmt(), target, folded)
                );
                return ls;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(
                    FoldExternalTargetContinuations(ctx, ws->getBody(), target, folded)
                );
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(
                    FoldExternalTargetContinuations(ctx, ds->getBody(), target, folded)
                );
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(
                    FoldExternalTargetContinuations(ctx, fs->getBody(), target, folded)
                );
                return fs;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(
                    FoldExternalTargetContinuations(ctx, sw->getBody(), target, folded)
                );
                return sw;
            }
            if (auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(stmt)) {
                case_stmt->setSubStmt(FoldExternalTargetContinuations(
                    ctx, case_stmt->getSubStmt(), target, folded
                ));
                return case_stmt;
            }
            if (auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(stmt)) {
                default_stmt->setSubStmt(FoldExternalTargetContinuations(
                    ctx, default_stmt->getSubStmt(), target, folded
                ));
                return default_stmt;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = FoldExternalTargetContinuations(ctx, child, target, folded);
            }

            bool local_changed = true;
            while (local_changed) {
                local_changed = false;
                for (size_t i = 0; i + 1 < body.size(); ++i) {
                    const unsigned target_gotos = CountGotosToTargetName(body[i], target);
                    if (target_gotos == 0 || CountAllGotos(body[i]) != target_gotos) {
                        continue;
                    }

                    const unsigned fallthroughs = CountTailFallthroughLeaves(body[i], target);
                    if (fallthroughs == 0
                        || fallthroughs > kMaxExternalContinuationFallthroughs)
                    {
                        continue;
                    }

                    std::vector< clang::Stmt * > continuation(
                        body.begin() + static_cast< ptrdiff_t >(i + 1), body.end()
                    );

                    if (fallthroughs == 1 && ContinuationSeqIsSafeToMoveOnce(continuation)) {
                        unsigned removed       = 0;
                        unsigned inserted      = 0;
                        clang::Stmt *rewritten = MoveContinuationIntoSingleFallthrough(
                            ctx, body[i], target, MoveContinuationSeqAsStmt(ctx, continuation),
                            removed, inserted
                        );
                        if (removed == target_gotos && inserted == 1) {
                            body[i] = rewritten;
                            body.erase(
                                body.begin() + static_cast< ptrdiff_t >(i + 1), body.end()
                            );
                            ++folded;
                            local_changed = true;
                            break;
                        }
                    }

                    if (!StraightLineContinuationSeqIsSafe(continuation)) { continue; }

                    unsigned removed       = 0;
                    unsigned inserted      = 0;
                    bool failed            = false;
                    clang::Stmt *rewritten = RewriteExternalTargetGotoWithContinuation(
                        ctx, body[i], target, continuation, removed, inserted, failed
                    );
                    if (failed || removed != target_gotos || inserted != fallthroughs) {
                        continue;
                    }

                    body[i] = rewritten;
                    body.erase(body.begin() + static_cast< ptrdiff_t >(i + 1), body.end());
                    ++folded;
                    local_changed = true;
                    break;
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        clang::Stmt *UnwrapDeadLeadingLabelForMove(
            clang::ASTContext &ctx, clang::Stmt *st,
            const std::unordered_set< clang::LabelDecl * > *live
        ) {
            if (!st || !live) { return st; }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(st)) {
                return live->count(label->getDecl()) ? st : label->getSubStmt();
            }
            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(st);
            if (!compound || compound->body_empty()) { return st; }
            auto it     = compound->body_begin();
            auto *label = llvm::dyn_cast< clang::LabelStmt >(*it);
            if (!label || live->count(label->getDecl())) { return st; }

            std::vector< clang::Stmt * > unwrapped;
            unwrapped.push_back(label->getSubStmt());
            for (++it; it != compound->body_end(); ++it) { unwrapped.push_back(*it); }
            return detail::MakeCompound(ctx, unwrapped);
        }

        bool HasDeadLeadingLabelForMove(
            clang::Stmt *st, const std::unordered_set< clang::LabelDecl * > *live
        ) {
            if (!st || !live) { return false; }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(st)) {
                return !live->count(label->getDecl());
            }
            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(st);
            if (!compound || compound->body_empty()) { return false; }
            auto *label = llvm::dyn_cast< clang::LabelStmt >(compound->body_front());
            return label && !live->count(label->getDecl());
        }

        bool TryMoveTailContinuationIntoDispatch(
            clang::ASTContext &ctx, std::vector< clang::Stmt * > &body,
            llvm::StringRef next_label, const std::unordered_set< clang::LabelDecl * > *live
        ) {
            if (body.size() < 2) { return false; }
            size_t continuation_idx = body.size() - 1;
            size_t dispatch_idx     = body.size() - 2;
            if (!HasDeadLeadingLabelForMove(body[continuation_idx], live)) { return false; }
            clang::Stmt *continuation =
                UnwrapDeadLeadingLabelForMove(ctx, body[continuation_idx], live);
            if (!GotoElimGetLabel(continuation).empty()
                || !ContinuationIsSafeToMoveBeforeJoin(continuation)
                || CountGotosToTargetName(body[dispatch_idx], next_label) == 0
                || CountTailFallthroughLeaves(body[dispatch_idx], next_label) != 1)
            {
                return false;
            }

            unsigned removed_gotos = 0;
            unsigned inserted      = 0;
            auto *rewritten        = MoveContinuationIntoSingleFallthrough(
                ctx, body[dispatch_idx], next_label, continuation, removed_gotos, inserted
            );
            if (removed_gotos == 0 || inserted != 1) { return false; }

            body[dispatch_idx] = rewritten;
            body.erase(body.begin() + static_cast< ptrdiff_t >(continuation_idx));
            return true;
        }

        bool TryMoveContinuationSeqIntoDispatch(
            clang::ASTContext &ctx, std::vector< clang::Stmt * > &body, size_t dispatch_idx,
            size_t label_idx, llvm::StringRef next_label
        ) {
            if (dispatch_idx + 1 >= label_idx || label_idx > body.size()) { return false; }

            std::vector< clang::Stmt * > continuation_stmts(
                body.begin() + static_cast< ptrdiff_t >(dispatch_idx + 1),
                body.begin() + static_cast< ptrdiff_t >(label_idx)
            );
            clang::Stmt *continuation = detail::MakeCompound(ctx, continuation_stmts);
            if (!StraightLineContinuationIsSafeToMove(continuation)
                || CountGotosToTargetName(body[dispatch_idx], next_label) == 0
                || CountAllGotos(body[dispatch_idx])
                    != CountGotosToTargetName(body[dispatch_idx], next_label)
                || CountTailFallthroughLeaves(body[dispatch_idx], next_label) != 1)
            {
                return false;
            }

            unsigned removed_gotos = 0;
            unsigned inserted      = 0;
            auto *rewritten        = MoveContinuationIntoSingleFallthrough(
                ctx, body[dispatch_idx], next_label, continuation, removed_gotos, inserted
            );
            if (removed_gotos == 0 || inserted != 1) { return false; }

            body[dispatch_idx] = rewritten;
            body.erase(
                body.begin() + static_cast< ptrdiff_t >(dispatch_idx + 1),
                body.begin() + static_cast< ptrdiff_t >(label_idx)
            );
            return true;
        }

        clang::Stmt *MoveTailContinuationBeforeFollowingLabel(
            clang::ASTContext &ctx, clang::Stmt *st, llvm::StringRef next_label,
            const std::unordered_set< clang::LabelDecl * > *live, unsigned &moved
        ) {
            if (!st) { return st; }
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                std::vector< clang::Stmt * > b(cs->body_begin(), cs->body_end());
                if (TryMoveTailContinuationIntoDispatch(ctx, b, next_label, live)) {
                    ++moved;
                    return detail::MakeCompound(ctx, b);
                }
                if (!b.empty()) {
                    unsigned before = moved;
                    b.back()        = MoveTailContinuationBeforeFollowingLabel(
                        ctx, b.back(), next_label, live, moved
                    );
                    if (moved != before) { return detail::MakeCompound(ctx, b); }
                }
                return st;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st)) {
                ls->setSubStmt(MoveTailContinuationBeforeFollowingLabel(
                    ctx, ls->getSubStmt(), next_label, live, moved
                ));
                return st;
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(st)) {
                unsigned before = moved;
                ifs->setThen(MoveTailContinuationBeforeFollowingLabel(
                    ctx, ifs->getThen(), next_label, live, moved
                ));
                if (ifs->getElse()) {
                    ifs->setElse(MoveTailContinuationBeforeFollowingLabel(
                        ctx, ifs->getElse(), next_label, live, moved
                    ));
                }
                return moved != before ? ifs : st;
            }
            return st;
        }

        clang::Stmt *ReplaceTrailingDeepStmt(
            clang::ASTContext &ctx, clang::Stmt *st, clang::Stmt *target,
            clang::Stmt *replacement
        ) {
            if (!st || st == target) { return st == target ? replacement : st; }
            if (auto *inner = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                if (inner->body_empty()) { return st; }
                std::vector< clang::Stmt * > b(inner->body_begin(), inner->body_end());
                if (DeepTrailingStmt(b.back()) == target) {
                    b.back() = ReplaceTrailingDeepStmt(ctx, b.back(), target, replacement);
                }
                return detail::MakeCompound(ctx, b);
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(st)) {
                ls->setSubStmt(
                    ReplaceTrailingDeepStmt(ctx, ls->getSubStmt(), target, replacement)
                );
                return ls;
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(st)) {
                if (DeepTrailingStmt(ifs->getThen()) == target) {
                    ifs->setThen(
                        ReplaceTrailingDeepStmt(ctx, ifs->getThen(), target, replacement)
                    );
                    return ifs;
                }
                if (ifs->getElse() && DeepTrailingStmt(ifs->getElse()) == target) {
                    ifs->setElse(
                        ReplaceTrailingDeepStmt(ctx, ifs->getElse(), target, replacement)
                    );
                    return ifs;
                }
            }
            return st;
        }

        clang::IfStmt *BuildIfGotoArmReplacement(
            clang::ASTContext &ctx, clang::IfStmt *ifs, llvm::StringRef target
        ) {
            int arm = IfStmtGotoArm(ifs, target);
            if (arm == 0) { return nullptr; }

            auto loc = ifs->getIfLoc();
            if (arm == 1) {
                auto *new_else = StripTrailingGoto(ctx, ifs->getElse(), target);
                if (llvm::isa< clang::NullStmt >(new_else)) { new_else = nullptr; }
                return clang::IfStmt::Create(
                    ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                    ifs->getCond(), loc, loc, ifs->getThen(), loc, new_else
                );
            }

            auto *neg = NegateExpr(ctx, ifs->getCond());
            return clang::IfStmt::Create(
                ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr, neg, loc, loc,
                ifs->getElse(), loc, nullptr
            );
        }

        /// Process a CompoundStmt: for each pair of adjacent stmts where
        /// the second is a LabelStmt, check if the first's deepest trailing
        /// stmt is a goto to that label.  Returns new stmt if changed.
        clang::Stmt *EliminateGotoToNextLabel(
            clang::ASTContext &ctx, clang::Stmt *s,
            const std::unordered_set< clang::LabelDecl * > *live = nullptr
        );

        clang::Stmt *ProcessCompound(
            clang::ASTContext &ctx, clang::CompoundStmt *cs,
            const std::unordered_set< clang::LabelDecl * > *live
        ) {
            std::vector< clang::Stmt * > body(cs->body_begin(), cs->body_end());

            // First: recurse into all children
            for (auto *&child : body) { child = EliminateGotoToNextLabel(ctx, child, live); }

            // Then: find goto-to-next-label patterns
            bool local_changed = true;
            while (local_changed) {
                local_changed = false;
                for (size_t i = 0; i + 1 < body.size(); ++i) {
                    size_t next_idx = i + 1;
                    while (next_idx < body.size()
                           && llvm::isa< clang::NullStmt >(body[next_idx]))
                    {
                        ++next_idx;
                    }
                    if (next_idx >= body.size()) { continue; }

                    auto next_label = GotoElimGetLabel(body[next_idx]);
                    if (next_label.empty()) { continue; }

                    // Find the deepest trailing stmt of body[i]
                    auto *deep = DeepTrailingStmt(body[i]);
                    if (!deep) { continue; }

                    // Pattern 1: deepest trailing is goto L; next is L:
                    auto tgt = GotoElimGetTarget(deep);
                    if (!tgt.empty() && tgt == next_label) {
                        // Simple case: body[i] IS the goto
                        if (deep == body[i]) {
                            body.erase(body.begin() + static_cast< ptrdiff_t >(i));
                            local_changed = true;
                            break;
                        }
                        // Otherwise: rebuild without the trailing goto.
                        std::function< clang::Stmt *(clang::Stmt *) > strip_tail;
                        strip_tail = [&](clang::Stmt *st) -> clang::Stmt * {
                            if (auto *inner = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                                if (inner->body_empty()) { return st; }
                                auto *last = *(inner->body_end() - 1);
                                if (last == deep) {
                                    std::vector< clang::Stmt * > b(
                                        inner->body_begin(), inner->body_end() - 1
                                    );
                                    if (b.empty()) {
                                        return new (ctx) clang::NullStmt(VirtualLoc(ctx));
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
                        local_changed = true;
                        break;
                    }

                    // Pattern 2: deepest trailing is IfStmt with goto arm.
                    // Arm 1: strip trailing goto from else; drop the else
                    //   if it collapses to NullStmt.
                    // Arm 2: drop the then arm and flip to `if(!c) else`
                    //   (safe because IfStmtGotoArm requires then to be
                    //   exactly the goto — see its docstring).
                    if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(deep)) {
                        clang::IfStmt *new_if = BuildIfGotoArmReplacement(ctx, ifs, next_label);
                        if (new_if) {
                            if (deep == body[i]) {
                                body[i] = new_if;
                            } else {
                                body[i] = ReplaceTrailingDeepStmt(ctx, body[i], deep, new_if);
                            }
                            local_changed = true;
                            break;
                        }

                        if (!ifs->getElse()) {
                            if (auto *nested_if = llvm::dyn_cast_or_null< clang::IfStmt >(
                                    DeepTrailingStmt(ifs->getThen())
                                ))
                            {
                                clang::IfStmt *new_nested =
                                    BuildIfGotoArmReplacement(ctx, nested_if, next_label);
                                if (new_nested) {
                                    body[i] = ReplaceTrailingDeepStmt(
                                        ctx, body[i], nested_if, new_nested
                                    );
                                    local_changed = true;
                                    break;
                                }
                            }
                        }
                    }

                    unsigned folded_external = 0;
                    auto *external_folded    = FoldExternalTargetContinuations(
                        ctx, body[i], next_label, folded_external
                    );
                    if (folded_external != 0) {
                        body[i] = external_folded;
                        local_changed = true;
                        break;
                    }

                    unsigned moved   = 0;
                    auto *moved_tail = MoveTailContinuationBeforeFollowingLabel(
                        ctx, body[i], next_label, live, moved
                    );
                    if (moved != 0) {
                        body[i] = moved_tail;
                        local_changed = true;
                        break;
                    }

                    unsigned stripped = 0;
                    auto *rewritten   = StripGotoToFollowingLabelFromTailPosition(
                        ctx, body[i], next_label, stripped
                    );
                    if (stripped != 0) {
                        body[i] = rewritten;
                        local_changed = true;
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
                if (next_label.empty()) { continue; }
                size_t continuation_idx = label_idx - 1;
                size_t dispatch_idx     = label_idx - 2;
                (void) continuation_idx;
                (void) dispatch_idx;
                for (size_t candidate = label_idx - 1; candidate > 0; --candidate) {
                    dispatch_idx = candidate - 1;
                    if (TryMoveContinuationSeqIntoDispatch(
                            ctx, body, dispatch_idx, label_idx, next_label
                        ))
                    {
                        local_changed = true;
                        break;
                    }
                }
                if (local_changed) { break; }

                std::vector< clang::Stmt * > tail_pair = {
                    body[label_idx - 2],
                    body[label_idx - 1],
                };
                if (TryMoveTailContinuationIntoDispatch(ctx, tail_pair, next_label, live)) {
                    body[label_idx - 2] = tail_pair.front();
                    body.erase(body.begin() + static_cast< ptrdiff_t >(label_idx - 1));
                    local_changed = true;
                    break;
                }
            }

            // Pattern: switch case goto L; } L: → replace goto with break.
            // When a switch stmt is followed by a LabelStmt, any case
            // body ending in goto-to-that-label can use break instead.
            for (size_t i = 0; i + 1 < body.size(); ++i) {
                auto *sw = llvm::dyn_cast< clang::SwitchStmt >(body[i]);
                if (!sw) { continue; }
                auto next_label = GotoElimGetLabel(body[i + 1]);
                if (next_label.empty()) { continue; }

                auto *sw_body = llvm::dyn_cast_or_null< clang::CompoundStmt >(sw->getBody());
                if (!sw_body) { continue; }

                std::vector< clang::Stmt * > sw_stmts(
                    sw_body->body_begin(), sw_body->body_end()
                );
                bool sw_changed = false;

                std::function< clang::Stmt *(clang::Stmt *) > replace_goto_break;
                replace_goto_break = [&](clang::Stmt *st) -> clang::Stmt * {
                    if (!st) { return st; }
                    if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(st)) {
                        if (gs->getLabel()->getName() == next_label) {
                            sw_changed = true;
                            return new (ctx) clang::BreakStmt(VirtualLoc(ctx));
                        }
                        return st;
                    }
                    if (auto *cs2 = llvm::dyn_cast< clang::CompoundStmt >(st)) {
                        if (cs2->body_empty()) { return st; }
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
                    local_changed = true;
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        clang::Stmt *EliminateGotoToNextLabel(
            clang::ASTContext &ctx, clang::Stmt *s,
            const std::unordered_set< clang::LabelDecl * > *live
        ) {
            if (!s) { return s; }
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
            clang::Stmt *stmt, std::unordered_map< clang::LabelDecl *, unsigned > &refs
        );

        clang::GotoStmt *SingleGotoStmt(clang::Stmt *stmt);

        clang::LabelDecl *LeadingLabelDecl(clang::Stmt *stmt) {
            if (auto *label = llvm::dyn_cast_or_null< clang::LabelStmt >(stmt)) {
                return label->getDecl();
            }
            auto *compound = llvm::dyn_cast_or_null< clang::CompoundStmt >(stmt);
            if (!compound || compound->body_empty()) { return nullptr; }
            return LeadingLabelDecl(compound->body_front());
        }

        bool ClangScopeifyLabelsAreRegionLocal(
            const std::vector< clang::Stmt * > &region,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            std::unordered_set< clang::LabelDecl * > labels;
            std::unordered_map< clang::LabelDecl *, unsigned > region_refs;

            std::function< void(clang::Stmt *) > collect_labels = [&](clang::Stmt *stmt) {
                if (!stmt) { return; }
                if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                    labels.insert(label->getDecl());
                }
                for (clang::Stmt *child : stmt->children()) { collect_labels(child); }
            };

            for (clang::Stmt *stmt : region) {
                collect_labels(stmt);
                CountGotoDeclRefs(stmt, region_refs);
            }

            for (clang::LabelDecl *label : labels) {
                unsigned global_count = 0;
                if (auto it = refs.find(label); it != refs.end()) { global_count = it->second; }

                unsigned local_count = 0;
                if (auto it = region_refs.find(label); it != region_refs.end()) {
                    local_count = it->second;
                }

                if (global_count != local_count) { return false; }
            }
            return true;
        }

        bool IsEffectivelyEmptyStmt(clang::Stmt *stmt) {
            if (!stmt) { return true; }
            if (llvm::isa< clang::NullStmt >(stmt)) { return true; }
            auto *compound = llvm::dyn_cast_or_null< clang::CompoundStmt >(stmt);
            if (!compound) { return false; }
            for (clang::Stmt *child : compound->body()) {
                if (!IsEffectivelyEmptyStmt(child)) { return false; }
            }
            return true;
        }

        clang::Stmt *ScopeifyIfGotos(
            clang::ASTContext &ctx, clang::Stmt *s,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (!s) { return s; }

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
            if (!cs) { return s; }

            std::vector< clang::Stmt * > body(cs->body_begin(), cs->body_end());

            // Recurse into children first
            for (auto *&child : body) { child = ScopeifyIfGotos(ctx, child, refs); }

            // Find if(c) goto L; ... L: patterns
            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t i = 0; i < body.size(); ++i) {
                    auto *ifs = llvm::dyn_cast< clang::IfStmt >(body[i]);
                    if (!ifs) { continue; }

                    if (ifs->getElse() && i + 1 < body.size()) {
                        clang::LabelDecl *next_label = LeadingLabelDecl(body[i + 1]);
                        if (next_label) {
                            auto *then_goto          = SingleGotoStmt(ifs->getThen());
                            auto *else_goto          = SingleGotoStmt(ifs->getElse());
                            clang::Stmt *replacement = nullptr;
                            auto loc                 = ifs->getIfLoc();

                            if (then_goto && then_goto->getLabel() == next_label
                                && !IsEffectivelyEmptyStmt(ifs->getElse()))
                            {
                                replacement = clang::IfStmt::Create(
                                    ctx, loc, clang::IfStatementKind::Ordinary, nullptr,
                                    nullptr, NegateExpr(ctx, ifs->getCond()), loc, loc,
                                    ifs->getElse(), loc, nullptr
                                );
                            } else if (
                                else_goto && else_goto->getLabel() == next_label
                                && !IsEffectivelyEmptyStmt(ifs->getThen())
                            )
                            {
                                replacement = clang::IfStmt::Create(
                                    ctx, loc, clang::IfStatementKind::Ordinary, nullptr,
                                    nullptr, ifs->getCond(), loc, loc, ifs->getThen(), loc,
                                    nullptr
                                );
                            }

                            if (replacement) {
                                body[i] = replacement;
                                changed = true;
                                break;
                            }
                        }
                    }

                    if (ifs->getElse()) { continue; }
                    auto *tail_goto = llvm::dyn_cast_or_null< clang::GotoStmt >(
                        DeepTrailingStmt(ifs->getThen())
                    );
                    if (tail_goto && tail_goto->getLabel()) {
                        auto *target_decl = tail_goto->getLabel();
                        auto target_name  = target_decl->getName();

                        size_t label_idx = body.size();
                        for (size_t j = i + 1; j < body.size(); ++j) {
                            clang::LabelDecl *leading_label = LeadingLabelDecl(body[j]);
                            if (leading_label && leading_label->getName() == target_name) {
                                label_idx = j;
                                break;
                            }
                        }
                        if (label_idx >= body.size()) { continue; }

                        clang::Stmt *then_without_goto =
                            StripTrailingGoto(ctx, ifs->getThen(), target_name);
                        if (!IsEffectivelyEmptyStmt(then_without_goto)) {
                            std::vector< clang::Stmt * > scoped;
                            for (size_t j = i + 1; j < label_idx; ++j) {
                                scoped.push_back(body[j]);
                            }
                            if (!ClangScopeifyLabelsAreRegionLocal(scoped, refs)) { continue; }

                            auto loc     = ifs->getIfLoc();
                            auto *new_if = clang::IfStmt::Create(
                                ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                                ifs->getCond(), loc, loc, then_without_goto, loc,
                                detail::MakeCompound(ctx, scoped)
                            );

                            body.erase(
                                body.begin() + static_cast< ptrdiff_t >(i),
                                body.begin() + static_cast< ptrdiff_t >(label_idx)
                            );
                            body.insert(body.begin() + static_cast< ptrdiff_t >(i), new_if);
                            changed = true;
                            break;
                        }
                    }

                    auto *gs = SingleGotoStmt(ifs->getThen());
                    if (!gs) { continue; }
                    auto *target_decl = gs->getLabel();
                    if (!target_decl) { continue; }
                    auto target_name = target_decl->getName();

                    // Find the target LabelStmt in the same CompoundStmt
                    size_t label_idx = body.size();
                    for (size_t j = i + 1; j < body.size(); ++j) {
                        clang::LabelDecl *leading_label = LeadingLabelDecl(body[j]);
                        if (leading_label && leading_label->getName() == target_name) {
                            label_idx = j;
                            break;
                        }
                    }
                    if (label_idx >= body.size()) { continue; }
                    // Skip adjacent if-goto (label_idx == i+1) — the
                    // goto looks dead but the condition may be a guard
                    // for code after the label.  Let it remain as a
                    // no-op if-goto.
                    if (label_idx == i + 1) { continue; }

                    // Collect intermediate stmts
                    std::vector< clang::Stmt * > scoped;
                    for (size_t j = i + 1; j < label_idx; ++j) { scoped.push_back(body[j]); }
                    // Intermediate labels are safe only when every reference
                    // to them is also inside the region being scoped.  This
                    // permits local label traffic while still rejecting gotos
                    // from outside into the moved block.
                    if (!ClangScopeifyLabelsAreRegionLocal(scoped, refs)) { continue; }

                    // Build: if(!cond) { scoped_stmts }
                    auto *neg        = NegateExpr(ctx, ifs->getCond());
                    auto *scope_body = detail::MakeCompound(ctx, scoped);
                    auto loc         = ifs->getIfLoc();
                    auto *new_if     = clang::IfStmt::Create(
                        ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr, neg, loc,
                        loc, scope_body, loc, nullptr
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
            clang::Stmt *stmt, std::unordered_map< clang::LabelDecl *, unsigned > &refs,
            std::unordered_set< clang::Stmt * > &seen
        ) {
            if (!stmt || !seen.insert(stmt).second) { return; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                refs[gs->getLabel()]++;
                return;
            }
            for (clang::Stmt *child : stmt->children()) {
                CountGotoDeclRefs(child, refs, seen);
            }
        }

        void CountGotoDeclRefs(
            clang::Stmt *stmt, std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            std::unordered_set< clang::Stmt * > seen;
            CountGotoDeclRefs(stmt, refs, seen);
        }

        clang::GotoStmt *SingleGotoStmt(clang::Stmt *stmt) {
            if (!stmt) { return nullptr; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) { return gs; }
            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                if (compound->body_empty()) { return nullptr; }
                auto it = compound->body_begin();
                ++it;
                if (it != compound->body_end()) { return nullptr; }
                return SingleGotoStmt(compound->body_front());
            }
            return nullptr;
        }

        struct NestedClangEntryLabel
        {
            clang::Stmt *entry_stmt = nullptr;
            clang::IfStmt *owner_if = nullptr;
            bool is_then            = false;
        };

        bool FindDirectNestedEntryLabel(
            clang::ASTContext &ctx, clang::Stmt *stmt, clang::LabelDecl *target,
            NestedClangEntryLabel &out
        ) {
            auto *ifs = llvm::dyn_cast_or_null< clang::IfStmt >(stmt);
            if (!ifs) { return false; }

            auto match_arm = [&](clang::Stmt *arm, bool is_then) -> bool {
                auto *label = llvm::dyn_cast_or_null< clang::LabelStmt >(arm);
                if (label && label->getDecl() == target) {
                    out = { label->getSubStmt(), ifs, is_then };
                    return true;
                }

                auto *compound = llvm::dyn_cast_or_null< clang::CompoundStmt >(arm);
                if (!compound || compound->body_empty()) { return false; }
                auto it = compound->body_begin();
                label   = llvm::dyn_cast_or_null< clang::LabelStmt >(*it);
                if (!label || label->getDecl() != target) { return false; }

                std::vector< clang::Stmt * > unwrapped;
                unwrapped.push_back(label->getSubStmt());
                for (++it; it != compound->body_end(); ++it) { unwrapped.push_back(*it); }
                out = { detail::MakeCompound(ctx, unwrapped), ifs, is_then };
                return true;
            };

            return match_arm(ifs->getThen(), true) || match_arm(ifs->getElse(), false);
        }

        size_t CountEntryStmtUnits(const clang::Stmt *stmt) {
            if (!stmt) { return 0; }
            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                size_t count = 0;
                for (const clang::Stmt *child : compound->body()) {
                    count += CountEntryStmtUnits(child);
                    if (count > 8) { return count; }
                }
                return count;
            }
            return 1;
        }

        bool EntryStmtIsCloneSafe(const clang::Stmt *stmt) {
            if (!stmt || CountEntryStmtUnits(stmt) > 8) { return false; }
            if (llvm::isa< clang::LabelStmt >(stmt) || llvm::isa< clang::GotoStmt >(stmt)
                || llvm::isa< clang::BreakStmt >(stmt) || llvm::isa< clang::ContinueStmt >(stmt)
                || llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::WhileStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt)
                || llvm::isa< clang::DeclStmt >(stmt))
            {
                return false;
            }
            for (const clang::Stmt *child : stmt->children()) {
                if (!EntryStmtIsCloneSafe(child)) { return false; }
            }
            return true;
        }

        struct ScopeFrame
        {
            const clang::Stmt *owner = nullptr;
            unsigned slot            = 0;
        };

        using ScopePath = std::vector< ScopeFrame >;

        struct ScopedGotoPlacement
        {
            clang::GotoStmt *go = nullptr;
            ScopePath scope;
        };

        struct ScopedControlTransferState
        {
            std::unordered_map< clang::LabelDecl *, ScopePath > labels;
            std::vector< ScopedGotoPlacement > gotos;
        };

        bool IsScopePrefix(const ScopePath &prefix, const ScopePath &path) {
            if (prefix.size() > path.size()) { return false; }
            for (size_t i = 0; i < prefix.size(); ++i) {
                if (prefix[i].owner != path[i].owner || prefix[i].slot != path[i].slot) {
                    return false;
                }
            }
            return true;
        }

        ScopePath WithScope(const ScopePath &scope, const clang::Stmt *owner, unsigned slot) {
            ScopePath result = scope;
            result.push_back(ScopeFrame{ owner, slot });
            return result;
        }

        void CollectScopedControlTransfers(
            clang::Stmt *stmt, ScopedControlTransferState &state, const ScopePath &scope
        ) {
            if (!stmt) { return; }

            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                state.labels.try_emplace(label->getDecl(), scope);
                CollectScopedControlTransfers(label->getSubStmt(), state, scope);
                return;
            }

            if (auto *go = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                state.gotos.push_back(ScopedGotoPlacement{ go, scope });
                return;
            }

            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                ScopePath compound_scope = WithScope(scope, compound, 0);
                for (auto *child : compound->body()) {
                    CollectScopedControlTransfers(child, state, compound_scope);
                }
                return;
            }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                CollectScopedControlTransfers(ifs->getThen(), state, WithScope(scope, ifs, 1));
                CollectScopedControlTransfers(ifs->getElse(), state, WithScope(scope, ifs, 2));
                return;
            }

            if (auto *while_stmt = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                CollectScopedControlTransfers(
                    while_stmt->getBody(), state, WithScope(scope, while_stmt, 1)
                );
                return;
            }

            if (auto *do_stmt = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                CollectScopedControlTransfers(
                    do_stmt->getBody(), state, WithScope(scope, do_stmt, 1)
                );
                return;
            }

            if (auto *for_stmt = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                CollectScopedControlTransfers(
                    for_stmt->getBody(), state, WithScope(scope, for_stmt, 1)
                );
                return;
            }

            if (auto *switch_stmt = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                CollectScopedControlTransfers(
                    switch_stmt->getBody(), state, WithScope(scope, switch_stmt, 1)
                );
                return;
            }

            if (auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(stmt)) {
                CollectScopedControlTransfers(
                    case_stmt->getSubStmt(), state, WithScope(scope, case_stmt, 1)
                );
                return;
            }

            if (auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(stmt)) {
                CollectScopedControlTransfers(
                    default_stmt->getSubStmt(), state, WithScope(scope, default_stmt, 1)
                );
                return;
            }

            for (auto *child : stmt->children()) {
                CollectScopedControlTransfers(child, state, scope);
            }
        }

        std::unordered_set< clang::LabelDecl * >
        CollectCrossScopeGotoTargets(clang::Stmt *stmt) {
            ScopedControlTransferState state;
            CollectScopedControlTransfers(stmt, state, ScopePath{});

            std::unordered_set< clang::LabelDecl * > targets;
            for (const ScopedGotoPlacement &placement : state.gotos) {
                if (!placement.go) { continue; }
                auto label_it = state.labels.find(placement.go->getLabel());
                if (label_it == state.labels.end()) { continue; }
                if (!IsScopePrefix(label_it->second, placement.scope)) {
                    targets.insert(placement.go->getLabel());
                }
            }
            return targets;
        }

        bool HoistedLabelBodyIsSafe(const clang::Stmt *stmt) {
            if (!stmt) { return true; }
            if (llvm::isa< clang::LabelStmt >(stmt) || llvm::isa< clang::GotoStmt >(stmt)
                || llvm::isa< clang::BreakStmt >(stmt) || llvm::isa< clang::ContinueStmt >(stmt)
                || llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::CaseStmt >(stmt)
                || llvm::isa< clang::DefaultStmt >(stmt) || llvm::isa< clang::WhileStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt)
                || llvm::isa< clang::DeclStmt >(stmt))
            {
                return false;
            }
            for (const clang::Stmt *child : stmt->children()) {
                if (!HoistedLabelBodyIsSafe(child)) { return false; }
            }
            return true;
        }

        struct ExtractedNestedLabel
        {
            clang::LabelDecl *decl = nullptr;
            clang::Stmt *body      = nullptr;
        };

        clang::Stmt *ExtractFirstNestedTargetLabel(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_set< clang::LabelDecl * > &targets,
            bool under_structured_scope, ExtractedNestedLabel &extracted, bool &changed
        ) {
            if (!stmt || changed) { return stmt; }

            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                if (under_structured_scope && targets.contains(label->getDecl())
                    && HoistedLabelBodyIsSafe(label->getSubStmt()))
                {
                    extracted.decl = label->getDecl();
                    extracted.body = label->getSubStmt() ? label->getSubStmt()
                                                         : new (ctx)
                                                               clang::NullStmt(VirtualLoc(ctx));
                    changed        = true;
                    return new (ctx)
                        clang::GotoStmt(extracted.decl, VirtualLoc(ctx), VirtualLoc(ctx));
                }

                label->setSubStmt(ExtractFirstNestedTargetLabel(
                    ctx, label->getSubStmt(), targets, under_structured_scope, extracted,
                    changed
                ));
                return label;
            }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(ExtractFirstNestedTargetLabel(
                    ctx, ifs->getThen(), targets, /*under_structured_scope=*/true, extracted,
                    changed
                ));
                if (changed) { return ifs; }
                if (ifs->getElse()) {
                    ifs->setElse(ExtractFirstNestedTargetLabel(
                        ctx, ifs->getElse(), targets, /*under_structured_scope=*/true,
                        extracted, changed
                    ));
                }
                return ifs;
            }

            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > children;
                for (auto *child : compound->body()) {
                    children.push_back(ExtractFirstNestedTargetLabel(
                        ctx, child, targets, under_structured_scope, extracted, changed
                    ));
                    if (changed) {
                        for (auto it = std::next(
                                 compound->body_begin(),
                                 static_cast< ptrdiff_t >(children.size())
                             );
                             it != compound->body_end(); ++it)
                        {
                            children.push_back(*it);
                        }
                        break;
                    }
                }
                return changed ? detail::MakeCompound(ctx, children) : stmt;
            }

            // A jump from outside directly into a loop or switch body is an
            // irreducible entry.  This pass only hoists labels out of if/else
            // scopes where fallthrough can be preserved with a synthetic join.
            if (llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::WhileStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt)
                || llvm::isa< clang::CaseStmt >(stmt) || llvm::isa< clang::DefaultStmt >(stmt))
            {
                return stmt;
            }

            return stmt;
        }

        clang::LabelDecl *CreateSyntheticJoinLabel(
            clang::ASTContext &ctx, clang::FunctionDecl *fn, clang::LabelDecl *target
        ) {
            static unsigned counter = 0;
            std::string name        = "__patchestry_scope_join_" + std::to_string(counter++);
            if (target && !target->getName().empty()) {
                name += "_";
                name += target->getName().str();
            }

            auto *decl =
                clang::LabelDecl::Create(ctx, fn, VirtualLoc(ctx), &ctx.Idents.get(name));
            if (fn) {
                decl->setDeclContext(fn);
                fn->addDecl(decl);
            }
            return decl;
        }

        clang::Stmt *HoistCrossScopeLabelEntries(
            clang::ASTContext &ctx, clang::FunctionDecl *fn, clang::Stmt *stmt
        ) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(HoistCrossScopeLabelEntries(ctx, fn, ifs->getThen()));
                if (ifs->getElse()) {
                    ifs->setElse(HoistCrossScopeLabelEntries(ctx, fn, ifs->getElse()));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(HoistCrossScopeLabelEntries(ctx, fn, ws->getBody()));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(HoistCrossScopeLabelEntries(ctx, fn, ds->getBody()));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(HoistCrossScopeLabelEntries(ctx, fn, fs->getBody()));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(HoistCrossScopeLabelEntries(ctx, fn, ls->getSubStmt()));
                return ls;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(HoistCrossScopeLabelEntries(ctx, fn, sw->getBody()));
                return sw;
            }
            if (auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(stmt)) {
                case_stmt->setSubStmt(
                    HoistCrossScopeLabelEntries(ctx, fn, case_stmt->getSubStmt())
                );
                return case_stmt;
            }
            if (auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(stmt)) {
                default_stmt->setSubStmt(
                    HoistCrossScopeLabelEntries(ctx, fn, default_stmt->getSubStmt())
                );
                return default_stmt;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = HoistCrossScopeLabelEntries(ctx, fn, child);
            }

            bool changed = true;
            while (changed) {
                changed       = false;
                auto *scratch = detail::MakeCompound(ctx, body);
                auto targets  = CollectCrossScopeGotoTargets(scratch);
                if (targets.empty()) { break; }

                for (size_t i = 0; i < body.size(); ++i) {
                    ExtractedNestedLabel extracted;
                    bool extracted_one     = false;
                    clang::Stmt *rewritten = ExtractFirstNestedTargetLabel(
                        ctx, body[i], targets, /*under_structured_scope=*/false, extracted,
                        extracted_one
                    );
                    if (!extracted_one || !extracted.decl || !extracted.body) { continue; }

                    body[i] = rewritten;

                    auto *join_decl = CreateSyntheticJoinLabel(ctx, fn, extracted.decl);
                    std::vector< clang::Stmt * > injected;
                    injected.push_back(
                        new (ctx) clang::GotoStmt(join_decl, VirtualLoc(ctx), VirtualLoc(ctx))
                    );
                    injected.push_back(new (ctx) clang::LabelStmt(
                        VirtualLoc(ctx), extracted.decl, extracted.body
                    ));
                    injected.push_back(new (ctx) clang::LabelStmt(
                        VirtualLoc(ctx), join_decl, new (ctx) clang::NullStmt(VirtualLoc(ctx))
                    ));

                    body.insert(
                        body.begin() + static_cast< ptrdiff_t >(i + 1), injected.begin(),
                        injected.end()
                    );
                    changed = true;
                    break;
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        clang::Stmt *RepairCrossScopeLabelEntries(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(RepairCrossScopeLabelEntries(ctx, ifs->getThen(), refs));
                if (ifs->getElse()) {
                    ifs->setElse(RepairCrossScopeLabelEntries(ctx, ifs->getElse(), refs));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(RepairCrossScopeLabelEntries(ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(RepairCrossScopeLabelEntries(ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(RepairCrossScopeLabelEntries(ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(RepairCrossScopeLabelEntries(ctx, ls->getSubStmt(), refs));
                return ls;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(RepairCrossScopeLabelEntries(ctx, sw->getBody(), refs));
                return sw;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = RepairCrossScopeLabelEntries(ctx, child, refs);
            }

            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t i = 0; i + 1 < body.size(); ++i) {
                    auto *ifs = llvm::dyn_cast< clang::IfStmt >(body[i]);
                    if (!ifs || ifs->getElse()) { continue; }
                    clang::GotoStmt *guard_goto = SingleGotoStmt(ifs->getThen());
                    if (!guard_goto || !guard_goto->getLabel()) { continue; }
                    clang::LabelDecl *target = guard_goto->getLabel();
                    auto ref_it              = refs.find(target);
                    if (ref_it == refs.end() || ref_it->second != 1) { continue; }

                    size_t carrier_idx = body.size();
                    NestedClangEntryLabel loc;
                    for (size_t j = i + 1; j < body.size(); ++j) {
                        if (FindDirectNestedEntryLabel(ctx, body[j], target, loc)) {
                            carrier_idx = j;
                            break;
                        }
                    }
                    if (carrier_idx >= body.size() || !loc.entry_stmt || !loc.owner_if) {
                        continue;
                    }

                    clang::Stmt *entry_stmt = loc.entry_stmt;
                    if (!EntryStmtIsCloneSafe(entry_stmt)) { continue; }

                    if (loc.is_then) {
                        loc.owner_if->setThen(entry_stmt);
                    } else {
                        loc.owner_if->setElse(entry_stmt);
                    }

                    std::vector< clang::Stmt * > else_body;
                    else_body.reserve(carrier_idx - i);
                    for (size_t j = i + 1; j <= carrier_idx; ++j) {
                        else_body.push_back(body[j]);
                    }

                    auto loc_if  = ifs->getIfLoc();
                    auto *new_if = clang::IfStmt::Create(
                        ctx, loc_if, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                        CloneExpr(ctx, ifs->getCond()), loc_if, loc_if, entry_stmt, loc_if,
                        detail::MakeCompound(ctx, else_body)
                    );

                    body.erase(
                        body.begin() + static_cast< ptrdiff_t >(i),
                        body.begin() + static_cast< ptrdiff_t >(carrier_idx) + 1
                    );
                    body.insert(body.begin() + static_cast< ptrdiff_t >(i), new_if);
                    changed = true;
                    break;
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        clang::Stmt *SwitchLocalLabelEntryStmt(
            clang::ASTContext &ctx, clang::Stmt *stmt, clang::LabelDecl *target
        ) {
            if (auto *label = llvm::dyn_cast_or_null< clang::LabelStmt >(stmt)) {
                if (label->getDecl() == target) { return label->getSubStmt(); }
                return nullptr;
            }

            auto *compound = llvm::dyn_cast_or_null< clang::CompoundStmt >(stmt);
            if (!compound || compound->body_empty()) { return nullptr; }

            auto it     = compound->body_begin();
            auto *label = llvm::dyn_cast_or_null< clang::LabelStmt >(*it);
            if (!label || label->getDecl() != target) { return nullptr; }

            std::vector< clang::Stmt * > unwrapped;
            unwrapped.push_back(label->getSubStmt());
            for (++it; it != compound->body_end(); ++it) { unwrapped.push_back(*it); }
            return detail::MakeCompound(ctx, unwrapped);
        }

        bool StmtStartsWithLabel(clang::Stmt *stmt) {
            if (llvm::isa_and_nonnull< clang::LabelStmt >(stmt)) { return true; }
            auto *compound = llvm::dyn_cast_or_null< clang::CompoundStmt >(stmt);
            return compound && !compound->body_empty()
                && llvm::isa< clang::LabelStmt >(compound->body_front());
        }

        bool BuildSwitchLocalLabelTail(
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &body, size_t label_idx,
            clang::LabelDecl *target, clang::Stmt *&entry, size_t &erase_end
        ) {
            entry              = nullptr;
            erase_end          = label_idx;
            clang::Stmt *first = SwitchLocalLabelEntryStmt(ctx, body[label_idx], target);
            if (!first) { return false; }

            std::vector< clang::Stmt * > tail;
            tail.push_back(first);
            erase_end = label_idx + 1;

            auto build_tail_stmt = [&]() -> clang::Stmt * {
                if (tail.size() == 1) { return tail.front(); }
                return detail::MakeCompound(ctx, tail);
            };

            while (!detail::EndsWithTerminator(build_tail_stmt())) {
                if (erase_end >= body.size()) { return false; }
                if (StmtStartsWithLabel(body[erase_end])) { return false; }
                tail.push_back(body[erase_end]);
                ++erase_end;
            }

            entry = build_tail_stmt();
            return EntryStmtIsCloneSafe(entry);
        }

        bool SwitchMoveTailHasUnsafeLocalControl(clang::Stmt *stmt) {
            if (!stmt) { return false; }
            if (llvm::isa< clang::LabelStmt >(stmt) || llvm::isa< clang::CaseStmt >(stmt)
                || llvm::isa< clang::DefaultStmt >(stmt) || llvm::isa< clang::BreakStmt >(stmt)
                || llvm::isa< clang::ContinueStmt >(stmt))
            {
                return true;
            }

            // Moving a whole nested loop/switch is safe for this check: any
            // break/continue/case/default inside it remains scoped to that
            // nested construct.
            if (llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::WhileStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt))
            {
                return false;
            }

            for (clang::Stmt *child : stmt->children()) {
                if (SwitchMoveTailHasUnsafeLocalControl(child)) { return true; }
            }
            return false;
        }

        bool BuildSwitchLocalLabelMoveTail(
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &body, size_t label_idx,
            clang::LabelDecl *target, clang::Stmt *&entry, size_t &erase_end
        ) {
            entry              = nullptr;
            erase_end          = label_idx;
            clang::Stmt *first = SwitchLocalLabelEntryStmt(ctx, body[label_idx], target);
            if (!first) { return false; }

            std::vector< clang::Stmt * > tail;
            tail.push_back(first);
            erase_end = label_idx + 1;

            auto build_tail_stmt = [&]() -> clang::Stmt * {
                return tail.size() == 1 ? tail.front() : detail::MakeCompound(ctx, tail);
            };

            while (!detail::EndsWithTerminator(build_tail_stmt())) {
                if (erase_end >= body.size()) { return false; }
                if (StmtStartsWithLabel(body[erase_end])) { return false; }
                tail.push_back(body[erase_end]);
                ++erase_end;
            }

            for (clang::Stmt *stmt : tail) {
                if (SwitchMoveTailHasUnsafeLocalControl(stmt)) { return false; }
            }

            entry = build_tail_stmt();
            return true;
        }

        clang::Stmt *ReplaceGotoToSwitchLocalLabel(
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

            // Do not rewrite gotos in nested control scopes; this pass is for
            // the current switch case body only.
            if (llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::WhileStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt))
            {
                return stmt;
            }

            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > children;
                unsigned before = replaced;
                for (clang::Stmt *child : compound->body()) {
                    clang::Stmt *rewritten = ReplaceGotoToSwitchLocalLabel(
                        ctx, child, target, replacement, replaced
                    );
                    children.push_back(rewritten);
                }
                if (replaced != before) { return detail::MakeCompound(ctx, children); }
                return stmt;
            }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(ReplaceGotoToSwitchLocalLabel(
                    ctx, ifs->getThen(), target, replacement, replaced
                ));
                if (ifs->getElse()) {
                    ifs->setElse(ReplaceGotoToSwitchLocalLabel(
                        ctx, ifs->getElse(), target, replacement, replaced
                    ));
                }
                return stmt;
            }

            return stmt;
        }

        bool ReplaceGotoToSwitchLocalLabelInSwitch(
            clang::ASTContext &ctx, clang::SwitchStmt *sw, clang::LabelDecl *target,
            clang::Stmt *replacement, unsigned &replaced
        ) {
            auto *body = llvm::dyn_cast_or_null< clang::CompoundStmt >(sw->getBody());
            if (!body) { return false; }

            unsigned before = replaced;
            for (clang::Stmt *child : body->body()) {
                if (auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(child)) {
                    case_stmt->setSubStmt(ReplaceGotoToSwitchLocalLabel(
                        ctx, case_stmt->getSubStmt(), target, replacement, replaced
                    ));
                } else if (auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(child)) {
                    default_stmt->setSubStmt(ReplaceGotoToSwitchLocalLabel(
                        ctx, default_stmt->getSubStmt(), target, replacement, replaced
                    ));
                }
            }
            return replaced != before;
        }

        unsigned CountSwitchLocalGotosToLabel(clang::Stmt *stmt, clang::LabelDecl *target) {
            if (!stmt) { return 0; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                return gs->getLabel() == target ? 1U : 0U;
            }
            if (llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::WhileStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt))
            {
                return 0;
            }

            unsigned count = 0;
            for (clang::Stmt *child : stmt->children()) {
                count += CountSwitchLocalGotosToLabel(child, target);
            }
            return count;
        }

        unsigned CountSwitchLocalGotosToLabel(clang::SwitchStmt *sw, clang::LabelDecl *target) {
            auto *body = llvm::dyn_cast_or_null< clang::CompoundStmt >(sw->getBody());
            if (!body) { return 0; }

            unsigned count = 0;
            for (clang::Stmt *child : body->body()) {
                if (auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(child)) {
                    count += CountSwitchLocalGotosToLabel(case_stmt->getSubStmt(), target);
                } else if (auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(child)) {
                    count += CountSwitchLocalGotosToLabel(default_stmt->getSubStmt(), target);
                }
            }
            return count;
        }

        clang::Stmt *ReplaceGotoWithTerminalStmtInAllScopes(
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

            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > children;
                unsigned before = replaced;
                for (clang::Stmt *child : compound->body()) {
                    children.push_back(ReplaceGotoWithTerminalStmtInAllScopes(
                        ctx, child, target, replacement, replaced
                    ));
                }
                return replaced != before ? detail::MakeCompound(ctx, children) : stmt;
            }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(ReplaceGotoWithTerminalStmtInAllScopes(
                    ctx, ifs->getThen(), target, replacement, replaced
                ));
                if (ifs->getElse()) {
                    ifs->setElse(ReplaceGotoWithTerminalStmtInAllScopes(
                        ctx, ifs->getElse(), target, replacement, replaced
                    ));
                }
                return stmt;
            }

            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(ReplaceGotoWithTerminalStmtInAllScopes(
                    ctx, sw->getBody(), target, replacement, replaced
                ));
                return stmt;
            }
            if (auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(stmt)) {
                case_stmt->setSubStmt(ReplaceGotoWithTerminalStmtInAllScopes(
                    ctx, case_stmt->getSubStmt(), target, replacement, replaced
                ));
                return stmt;
            }
            if (auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(stmt)) {
                default_stmt->setSubStmt(ReplaceGotoWithTerminalStmtInAllScopes(
                    ctx, default_stmt->getSubStmt(), target, replacement, replaced
                ));
                return stmt;
            }

            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(ReplaceGotoWithTerminalStmtInAllScopes(
                    ctx, ws->getBody(), target, replacement, replaced
                ));
                return stmt;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(ReplaceGotoWithTerminalStmtInAllScopes(
                    ctx, ds->getBody(), target, replacement, replaced
                ));
                return stmt;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(ReplaceGotoWithTerminalStmtInAllScopes(
                    ctx, fs->getBody(), target, replacement, replaced
                ));
                return stmt;
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                label->setSubStmt(ReplaceGotoWithTerminalStmtInAllScopes(
                    ctx, label->getSubStmt(), target, replacement, replaced
                ));
                return stmt;
            }

            return stmt;
        }

        clang::Stmt *FoldClangSwitchLocalCaseTargets(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(FoldClangSwitchLocalCaseTargets(ctx, ifs->getThen(), refs));
                if (ifs->getElse()) {
                    ifs->setElse(FoldClangSwitchLocalCaseTargets(ctx, ifs->getElse(), refs));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(FoldClangSwitchLocalCaseTargets(ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(FoldClangSwitchLocalCaseTargets(ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(FoldClangSwitchLocalCaseTargets(ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(FoldClangSwitchLocalCaseTargets(ctx, ls->getSubStmt(), refs));
                return ls;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = FoldClangSwitchLocalCaseTargets(ctx, child, refs);
            }

            for (size_t i = 0; i < body.size(); ++i) {
                auto *sw = llvm::dyn_cast< clang::SwitchStmt >(body[i]);
                if (!sw) { continue; }

                bool changed = true;
                while (changed) {
                    changed = false;
                    for (size_t j = i + 1; j < body.size(); ++j) {
                        clang::LabelDecl *target = nullptr;
                        if (auto *label = llvm::dyn_cast< clang::LabelStmt >(body[j])) {
                            target = label->getDecl();
                        } else if (
                            auto *label_body = llvm::dyn_cast< clang::CompoundStmt >(body[j])
                        )
                        {
                            if (!label_body->body_empty()) {
                                if (auto *label = llvm::dyn_cast< clang::LabelStmt >(
                                        label_body->body_front()
                                    ))
                                {
                                    target = label->getDecl();
                                }
                            }
                        }
                        if (!target) { continue; }

                        auto ref_it = refs.find(target);
                        if (ref_it == refs.end() || ref_it->second == 0) { continue; }
                        if (j == 0 || !detail::EndsWithTerminator(body[j - 1])) { continue; }

                        clang::Stmt *entry = nullptr;
                        size_t erase_end   = j;
                        if (!BuildSwitchLocalLabelTail(ctx, body, j, target, entry, erase_end))
                        {
                            if (ref_it->second != 1
                                || !BuildSwitchLocalLabelMoveTail(
                                    ctx, body, j, target, entry, erase_end
                                ))
                            {
                                continue;
                            }
                        }
                        if (CountSwitchLocalGotosToLabel(sw, target) != ref_it->second) {
                            continue;
                        }

                        unsigned replaced = 0;
                        if (!ReplaceGotoToSwitchLocalLabelInSwitch(
                                ctx, sw, target, entry, replaced
                            ))
                        {
                            continue;
                        }
                        if (replaced != ref_it->second) { continue; }

                        body.erase(
                            body.begin() + static_cast< ptrdiff_t >(j),
                            body.begin() + static_cast< ptrdiff_t >(erase_end)
                        );
                        changed = true;
                        break;
                    }
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        constexpr size_t kMaxConditionalFallthroughBodyStmts = 8;
        constexpr size_t kMaxTerminalPrefixStmts             = 12;
        constexpr unsigned kMaxClonedTailUses                = 4;

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

        clang::Stmt *FoldForwardSingleRefLabelRegions(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(FoldForwardSingleRefLabelRegions(ctx, ifs->getThen(), refs));
                if (ifs->getElse()) {
                    ifs->setElse(FoldForwardSingleRefLabelRegions(ctx, ifs->getElse(), refs));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(FoldForwardSingleRefLabelRegions(ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(FoldForwardSingleRefLabelRegions(ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(FoldForwardSingleRefLabelRegions(ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(FoldForwardSingleRefLabelRegions(ctx, ls->getSubStmt(), refs));
                return ls;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(FoldForwardSingleRefLabelRegions(ctx, sw->getBody(), refs));
                return sw;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = FoldForwardSingleRefLabelRegions(ctx, child, refs);
            }

            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t if_idx = 0; if_idx + 1 < body.size(); ++if_idx) {
                    auto *ifs = llvm::dyn_cast< clang::IfStmt >(body[if_idx]);
                    if (!ifs) { continue; }
                    if (ifs->getElse() && !IsEffectivelyEmptyStmt(ifs->getElse())) { continue; }

                    clang::GotoStmt *then_goto = SingleGotoStmt(ifs->getThen());
                    if (!then_goto || !then_goto->getLabel()) { continue; }

                    clang::LabelDecl *target = then_goto->getLabel();
                    auto ref_it              = refs.find(target);
                    if (ref_it == refs.end() || ref_it->second != 1) { continue; }

                    auto build_if_from_sequence =
                        [&](const std::vector< clang::Stmt * > &sequence, size_t label_idx,
                            clang::Stmt *&new_if,
                            std::vector< clang::Stmt * > &remainder) -> bool {
                        if (label_idx == 0 || label_idx >= sequence.size()) { return false; }

                        std::vector< clang::Stmt * > skipped;
                        for (size_t j = 0; j < label_idx; ++j) {
                            if (LeadingLabelDecl(sequence[j])) { return false; }
                            skipped.push_back(sequence[j]);
                        }
                        if (skipped.empty() || !SeqEndsWithTerminator(ctx, skipped)) {
                            return false;
                        }
                        if (SeqHasUnsafeStructure(
                                skipped, /*allow_goto=*/true,
                                /*allow_return=*/true
                            ))
                        {
                            return false;
                        }

                        LocalLabelBlock block;
                        if (!ExtractLocalLabelBlock(ctx, sequence, label_idx, block)) {
                            return false;
                        }
                        if (block.stmts.size() > kMaxConditionalFallthroughBodyStmts) {
                            return false;
                        }
                        if (SeqHasUnsafeStructure(
                                block.stmts, /*allow_goto=*/true,
                                /*allow_return=*/false
                            ))
                        {
                            return false;
                        }

                        auto loc = ifs->getIfLoc();
                        new_if   = clang::IfStmt::Create(
                            ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                            ifs->getCond(), loc, loc, StmtFromSeq(ctx, block.stmts), loc,
                            StmtFromSeq(ctx, skipped)
                        );
                        remainder.assign(
                            sequence.begin() + static_cast< ptrdiff_t >(block.end),
                            sequence.end()
                        );
                        return true;
                    };

                    std::vector< clang::Stmt * > tail(
                        body.begin() + static_cast< ptrdiff_t >(if_idx + 1), body.end()
                    );
                    size_t tail_label_idx = tail.size();
                    for (size_t j = 0; j < tail.size(); ++j) {
                        if (LeadingLabelDecl(tail[j]) == target) {
                            tail_label_idx = j;
                            break;
                        }
                    }

                    clang::Stmt *new_if = nullptr;
                    std::vector< clang::Stmt * > remainder;
                    if (build_if_from_sequence(tail, tail_label_idx, new_if, remainder)) {
                        body.erase(body.begin() + static_cast< ptrdiff_t >(if_idx), body.end());
                        body.push_back(new_if);
                        body.insert(body.end(), remainder.begin(), remainder.end());
                        changed = true;
                        break;
                    }

                    auto *next_compound =
                        llvm::dyn_cast< clang::CompoundStmt >(body[if_idx + 1]);
                    if (!next_compound || next_compound->body_empty()) { continue; }

                    std::vector< clang::Stmt * > nested(
                        next_compound->body_begin(), next_compound->body_end()
                    );
                    size_t nested_label_idx = nested.size();
                    for (size_t j = 0; j < nested.size(); ++j) {
                        if (LeadingLabelDecl(nested[j]) == target) {
                            nested_label_idx = j;
                            break;
                        }
                    }
                    if (!build_if_from_sequence(nested, nested_label_idx, new_if, remainder)) {
                        continue;
                    }
                    body[if_idx] = new_if;
                    if (remainder.empty()) {
                        body.erase(body.begin() + static_cast< ptrdiff_t >(if_idx + 1));
                    } else {
                        body[if_idx + 1] = StmtFromSeq(ctx, remainder);
                    }

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

        clang::Stmt *InlineSingleRefTerminalLabelBlocks(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(InlineSingleRefTerminalLabelBlocks(ctx, ifs->getThen(), refs));
                if (ifs->getElse()) {
                    ifs->setElse(InlineSingleRefTerminalLabelBlocks(ctx, ifs->getElse(), refs));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(InlineSingleRefTerminalLabelBlocks(ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(InlineSingleRefTerminalLabelBlocks(ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(InlineSingleRefTerminalLabelBlocks(ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(InlineSingleRefTerminalLabelBlocks(ctx, ls->getSubStmt(), refs));
                return ls;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(InlineSingleRefTerminalLabelBlocks(ctx, sw->getBody(), refs));
                return sw;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = InlineSingleRefTerminalLabelBlocks(ctx, child, refs);
            }

            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t label_idx = 0; label_idx < body.size(); ++label_idx) {
                    clang::LabelDecl *target = LeadingLabelDecl(body[label_idx]);
                    if (!target) { continue; }
                    auto ref_it = refs.find(target);
                    if (ref_it == refs.end() || ref_it->second != 1) { continue; }
                    if (!LabelHasNoFallthroughPredecessor(body, label_idx)) { continue; }

                    LocalLabelBlock block;
                    if (!ExtractLocalLabelBlock(ctx, body, label_idx, block)) { continue; }
                    if (block.stmts.size() > kMaxTerminalPrefixStmts) { continue; }
                    if (!SeqEndsWithTerminator(ctx, block.stmts)) { continue; }

                    clang::Stmt *block_stmt = StmtFromSeq(ctx, block.stmts);
                    unsigned block_gotos    = CountAllGotos(block_stmt);
                    bool safe_control       = false;
                    if (block_gotos == 0) {
                        safe_control = !SeqHasUnsafeStructure(
                            block.stmts, /*allow_goto=*/false, /*allow_return=*/true
                        );
                    } else if (
                        block_gotos == 1
                        && !GotoElimGetTarget(DeepTrailingStmt(block_stmt)).empty()
                        && GotoElimGetTarget(DeepTrailingStmt(block_stmt)) != target->getName()
                    )
                    {
                        safe_control = !SeqHasUnsafeStructure(
                            block.stmts, /*allow_goto=*/true, /*allow_return=*/true
                        );
                    }
                    if (!safe_control) { continue; }

                    unsigned replaced        = 0;
                    clang::Stmt *replacement = block_stmt;
                    for (size_t i = 0; i < body.size(); ++i) {
                        if (i >= block.begin && i < block.end) { continue; }
                        body[i] = ReplaceGotoWithTerminalStmtInAllScopes(
                            ctx, body[i], target, replacement, replaced
                        );
                    }
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

        clang::Stmt *CloneStraightLineSeqAsStmt(
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &stmts,
            clang::LabelDecl *trailing_goto = nullptr
        ) {
            std::vector< clang::Stmt * > cloned;
            for (clang::Stmt *stmt : stmts) {
                clang::Stmt *copy = CloneStraightLineStmt(ctx, stmt);
                if (!copy) { return nullptr; }
                AppendStmtSequence(copy, cloned);
            }
            if (trailing_goto) {
                auto loc = VirtualLoc(ctx);
                cloned.push_back(new (ctx) clang::GotoStmt(trailing_goto, loc, loc));
            }
            return StmtFromSeq(ctx, cloned);
        }

        clang::Stmt *CloneGotoFreeStmt(clang::ASTContext &ctx, clang::Stmt *stmt) {
            if (!stmt) { return new (ctx) clang::NullStmt(VirtualLoc(ctx)); }
            if (llvm::isa< clang::NullStmt >(stmt)) {
                return new (ctx) clang::NullStmt(VirtualLoc(ctx));
            }
            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > cloned;
                for (clang::Stmt *child : compound->body()) {
                    clang::Stmt *copy = CloneGotoFreeStmt(ctx, child);
                    if (!copy) { return nullptr; }
                    AppendStmtSequence(copy, cloned);
                }
                return StmtFromSeq(ctx, cloned);
            }
            if (auto *ret = llvm::dyn_cast< clang::ReturnStmt >(stmt)) {
                return clang::ReturnStmt::Create(
                    ctx, ret->getReturnLoc(), CloneExpr(ctx, ret->getRetValue()), nullptr
                );
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                if (ifs->getInit() || ifs->getConditionVariable()) { return nullptr; }
                clang::Stmt *then_copy = CloneGotoFreeStmt(ctx, ifs->getThen());
                if (!then_copy) { return nullptr; }
                clang::Stmt *else_copy = nullptr;
                if (ifs->getElse()) {
                    else_copy = CloneGotoFreeStmt(ctx, ifs->getElse());
                    if (!else_copy) { return nullptr; }
                }
                auto loc = ifs->getIfLoc();
                return clang::IfStmt::Create(
                    ctx, loc, ifs->getStatementKind(), nullptr, nullptr,
                    CloneExpr(ctx, ifs->getCond()), ifs->getLParenLoc(), ifs->getRParenLoc(),
                    then_copy, loc, else_copy
                );
            }
            if (auto *expr = llvm::dyn_cast< clang::Expr >(stmt)) {
                return CloneExpr(ctx, expr);
            }
            return nullptr;
        }

        clang::Stmt *CloneGotoFreeSeqAsStmt(
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &stmts
        ) {
            std::vector< clang::Stmt * > cloned;
            for (clang::Stmt *stmt : stmts) {
                clang::Stmt *copy = CloneGotoFreeStmt(ctx, stmt);
                if (!copy) { return nullptr; }
                AppendStmtSequence(copy, cloned);
            }
            return StmtFromSeq(ctx, cloned);
        }

        clang::Stmt *ReplaceGotoWithClonedJoinTail(
            clang::ASTContext &ctx, clang::Stmt *stmt, clang::LabelDecl *target,
            clang::LabelDecl *join, const std::vector< clang::Stmt * > &tail,
            unsigned &replaced, bool &failed
        ) {
            if (!stmt || failed) { return stmt; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                if (gs->getLabel() != target) { return stmt; }
                clang::Stmt *replacement = CloneStraightLineSeqAsStmt(ctx, tail, join);
                if (!replacement) {
                    failed = true;
                    return stmt;
                }
                ++replaced;
                return replacement;
            }

            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > children;
                unsigned before = replaced;
                for (clang::Stmt *child : compound->body()) {
                    children.push_back(ReplaceGotoWithClonedJoinTail(
                        ctx, child, target, join, tail, replaced, failed
                    ));
                    if (failed) { return stmt; }
                }
                return replaced != before ? detail::MakeCompound(ctx, children) : stmt;
            }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(ReplaceGotoWithClonedJoinTail(
                    ctx, ifs->getThen(), target, join, tail, replaced, failed
                ));
                if (failed) { return stmt; }
                if (ifs->getElse()) {
                    ifs->setElse(ReplaceGotoWithClonedJoinTail(
                        ctx, ifs->getElse(), target, join, tail, replaced, failed
                    ));
                }
                return stmt;
            }

            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                label->setSubStmt(ReplaceGotoWithClonedJoinTail(
                    ctx, label->getSubStmt(), target, join, tail, replaced, failed
                ));
                return label;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(ReplaceGotoWithClonedJoinTail(
                    ctx, sw->getBody(), target, join, tail, replaced, failed
                ));
                return sw;
            }
            if (auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(stmt)) {
                case_stmt->setSubStmt(ReplaceGotoWithClonedJoinTail(
                    ctx, case_stmt->getSubStmt(), target, join, tail, replaced, failed
                ));
                return case_stmt;
            }
            if (auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(stmt)) {
                default_stmt->setSubStmt(ReplaceGotoWithClonedJoinTail(
                    ctx, default_stmt->getSubStmt(), target, join, tail, replaced, failed
                ));
                return default_stmt;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(ReplaceGotoWithClonedJoinTail(
                    ctx, ws->getBody(), target, join, tail, replaced, failed
                ));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(ReplaceGotoWithClonedJoinTail(
                    ctx, ds->getBody(), target, join, tail, replaced, failed
                ));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(ReplaceGotoWithClonedJoinTail(
                    ctx, fs->getBody(), target, join, tail, replaced, failed
                ));
                return fs;
            }

            return stmt;
        }

        clang::Stmt *ReplaceGotoWithClonedGotoFreeSeq(
            clang::ASTContext &ctx, clang::Stmt *stmt, clang::LabelDecl *target,
            const std::vector< clang::Stmt * > &replacement_stmts, unsigned &replaced,
            bool &failed
        ) {
            if (!stmt || failed) { return stmt; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                if (gs->getLabel() != target) { return stmt; }
                clang::Stmt *replacement = CloneGotoFreeSeqAsStmt(ctx, replacement_stmts);
                if (!replacement) {
                    failed = true;
                    return stmt;
                }
                ++replaced;
                return replacement;
            }

            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > children;
                unsigned before = replaced;
                for (clang::Stmt *child : compound->body()) {
                    children.push_back(ReplaceGotoWithClonedGotoFreeSeq(
                        ctx, child, target, replacement_stmts, replaced, failed
                    ));
                    if (failed) { return stmt; }
                }
                return replaced != before ? detail::MakeCompound(ctx, children) : stmt;
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(ReplaceGotoWithClonedGotoFreeSeq(
                    ctx, ifs->getThen(), target, replacement_stmts, replaced, failed
                ));
                if (failed) { return stmt; }
                if (ifs->getElse()) {
                    ifs->setElse(ReplaceGotoWithClonedGotoFreeSeq(
                        ctx, ifs->getElse(), target, replacement_stmts, replaced, failed
                    ));
                }
                return stmt;
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                label->setSubStmt(ReplaceGotoWithClonedGotoFreeSeq(
                    ctx, label->getSubStmt(), target, replacement_stmts, replaced, failed
                ));
                return label;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(ReplaceGotoWithClonedGotoFreeSeq(
                    ctx, sw->getBody(), target, replacement_stmts, replaced, failed
                ));
                return sw;
            }
            if (auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(stmt)) {
                case_stmt->setSubStmt(ReplaceGotoWithClonedGotoFreeSeq(
                    ctx, case_stmt->getSubStmt(), target, replacement_stmts, replaced, failed
                ));
                return case_stmt;
            }
            if (auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(stmt)) {
                default_stmt->setSubStmt(ReplaceGotoWithClonedGotoFreeSeq(
                    ctx, default_stmt->getSubStmt(), target, replacement_stmts, replaced, failed
                ));
                return default_stmt;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(ReplaceGotoWithClonedGotoFreeSeq(
                    ctx, ws->getBody(), target, replacement_stmts, replaced, failed
                ));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(ReplaceGotoWithClonedGotoFreeSeq(
                    ctx, ds->getBody(), target, replacement_stmts, replaced, failed
                ));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(ReplaceGotoWithClonedGotoFreeSeq(
                    ctx, fs->getBody(), target, replacement_stmts, replaced, failed
                ));
                return fs;
            }
            return stmt;
        }

        clang::Stmt *CloneFallthroughTerminalLabelGotos(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (!stmt) { return stmt; }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(CloneFallthroughTerminalLabelGotos(ctx, ifs->getThen(), refs));
                if (ifs->getElse()) {
                    ifs->setElse(CloneFallthroughTerminalLabelGotos(ctx, ifs->getElse(), refs));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(CloneFallthroughTerminalLabelGotos(ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(CloneFallthroughTerminalLabelGotos(ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(CloneFallthroughTerminalLabelGotos(ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                label->setSubStmt(
                    CloneFallthroughTerminalLabelGotos(ctx, label->getSubStmt(), refs)
                );
                return label;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(CloneFallthroughTerminalLabelGotos(ctx, sw->getBody(), refs));
                return sw;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = CloneFallthroughTerminalLabelGotos(ctx, child, refs);
            }

            for (size_t label_idx = 0; label_idx < body.size(); ++label_idx) {
                clang::LabelDecl *target = LeadingLabelDecl(body[label_idx]);
                if (!target) { continue; }
                auto ref_it = refs.find(target);
                if (ref_it == refs.end() || ref_it->second == 0) { continue; }
                if (LabelHasNoFallthroughPredecessor(body, label_idx)) { continue; }

                LocalLabelBlock block;
                if (!ExtractLocalLabelBlock(ctx, body, label_idx, block)) { continue; }
                if (block.stmts.size() > kMaxTerminalPrefixStmts) { continue; }
                if (!SeqEndsWithTerminator(ctx, block.stmts)) { continue; }
                if (CountAllGotos(StmtFromSeq(ctx, block.stmts)) != 0) { continue; }
                if (SeqHasUnsafeStructure(
                        block.stmts, /*allow_goto=*/false, /*allow_return=*/true
                    ))
                {
                    continue;
                }

                unsigned replaced = 0;
                bool failed       = false;
                for (size_t i = 0; i < body.size(); ++i) {
                    if (i >= block.begin && i < block.end) { continue; }
                    body[i] = ReplaceGotoWithClonedGotoFreeSeq(
                        ctx, body[i], target, block.stmts, replaced, failed
                    );
                    if (failed) { break; }
                }
                if (!failed && replaced == ref_it->second) {
                    return detail::MakeCompound(ctx, body);
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        clang::Stmt *CloneNoFallthroughTerminalLabelGotos(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (!stmt) { return stmt; }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(CloneNoFallthroughTerminalLabelGotos(ctx, ifs->getThen(), refs));
                if (ifs->getElse()) {
                    ifs->setElse(
                        CloneNoFallthroughTerminalLabelGotos(ctx, ifs->getElse(), refs)
                    );
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(CloneNoFallthroughTerminalLabelGotos(ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(CloneNoFallthroughTerminalLabelGotos(ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(CloneNoFallthroughTerminalLabelGotos(ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                label->setSubStmt(
                    CloneNoFallthroughTerminalLabelGotos(ctx, label->getSubStmt(), refs)
                );
                return label;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(CloneNoFallthroughTerminalLabelGotos(ctx, sw->getBody(), refs));
                return sw;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = CloneNoFallthroughTerminalLabelGotos(ctx, child, refs);
            }

            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t label_idx = 0; label_idx < body.size(); ++label_idx) {
                    clang::LabelDecl *target = LeadingLabelDecl(body[label_idx]);
                    if (!target) { continue; }

                    auto ref_it = refs.find(target);
                    if (ref_it == refs.end() || ref_it->second == 0
                        || ref_it->second > kMaxClonedTailUses)
                    {
                        continue;
                    }
                    if (!LabelHasNoFallthroughPredecessor(body, label_idx)) { continue; }

                    LocalLabelBlock block;
                    if (!ExtractLocalLabelBlock(ctx, body, label_idx, block)) { continue; }
                    if (block.stmts.size() > kMaxTerminalPrefixStmts) { continue; }
                    if (!SeqEndsWithTerminator(ctx, block.stmts)) { continue; }
                    if (CountAllGotos(StmtFromSeq(ctx, block.stmts)) != 0) { continue; }
                    if (SeqHasUnsafeStructure(
                            block.stmts, /*allow_goto=*/false, /*allow_return=*/true
                        ))
                    {
                        continue;
                    }

                    unsigned replaced = 0;
                    bool failed       = false;
                    for (size_t i = 0; i < body.size(); ++i) {
                        if (i >= block.begin && i < block.end) { continue; }
                        body[i] = ReplaceGotoWithClonedGotoFreeSeq(
                            ctx, body[i], target, block.stmts, replaced, failed
                        );
                        if (failed) { break; }
                    }
                    if (failed || replaced != ref_it->second) { continue; }

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

        bool CleanupStmtHasCleanupLikeCall(clang::Stmt *stmt) {
            if (!stmt) { return false; }
            if (auto *call = llvm::dyn_cast< clang::CallExpr >(stmt)) {
                if (const auto *callee = call->getDirectCallee()) {
                    std::string name = callee->getNameAsString();
                    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) {
                        return static_cast< char >(std::tolower(ch));
                    });
                    if (name.find("dealloc") != std::string::npos
                        || name.find("delete") != std::string::npos
                        || name.find("dtor") != std::string::npos
                        || name.find("unref") != std::string::npos
                        || name.find("free") != std::string::npos)
                    {
                        return true;
                    }
                }
            }
            for (clang::Stmt *child : stmt->children()) {
                if (CleanupStmtHasCleanupLikeCall(child)) { return true; }
            }
            return false;
        }

        bool CleanupSeqHasCleanupLikeCall(const std::vector< clang::Stmt * > &stmts) {
            for (clang::Stmt *stmt : stmts) {
                if (CleanupStmtHasCleanupLikeCall(stmt)) { return true; }
            }
            return false;
        }

        bool StmtHasCallExpr(clang::Stmt *stmt) {
            if (!stmt) { return false; }
            if (llvm::isa< clang::CallExpr >(stmt)) { return true; }
            for (clang::Stmt *child : stmt->children()) {
                if (StmtHasCallExpr(child)) { return true; }
            }
            return false;
        }

        bool SeqHasCallExpr(const std::vector< clang::Stmt * > &stmts) {
            for (clang::Stmt *stmt : stmts) {
                if (StmtHasCallExpr(stmt)) { return true; }
            }
            return false;
        }

        bool StmtHasUnsafeGuardedJoinMoveControl(clang::Stmt *stmt) {
            if (!stmt) { return false; }
            if (llvm::isa< clang::LabelStmt >(stmt) || llvm::isa< clang::GotoStmt >(stmt)
                || llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::WhileStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt)
                || llvm::isa< clang::CaseStmt >(stmt) || llvm::isa< clang::DefaultStmt >(stmt)
                || llvm::isa< clang::DeclStmt >(stmt))
            {
                return true;
            }
            for (clang::Stmt *child : stmt->children()) {
                if (StmtHasUnsafeGuardedJoinMoveControl(child)) { return true; }
            }
            return false;
        }

        bool SeqHasUnsafeGuardedJoinMoveControl(
            const std::vector< clang::Stmt * > &stmts
        ) {
            for (clang::Stmt *stmt : stmts) {
                if (StmtHasUnsafeGuardedJoinMoveControl(stmt)) { return true; }
            }
            return false;
        }

        bool IsSyntheticScopeJoinLabel(clang::LabelDecl *label) {
            return label && label->getName().starts_with("__patchestry_scope_join");
        }

        clang::Stmt *ReplaceGotoWithClonedCleanupThenJoin(
            clang::ASTContext &ctx, clang::Stmt *stmt, clang::LabelDecl *cleanup_label,
            clang::LabelDecl *join_label, const std::vector< clang::Stmt * > &cleanup_stmts,
            unsigned &replaced, bool &failed
        ) {
            if (!stmt || failed) { return stmt; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                if (gs->getLabel() != cleanup_label) { return stmt; }
                clang::Stmt *replacement =
                    CloneStraightLineSeqAsStmt(ctx, cleanup_stmts, join_label);
                if (!replacement) {
                    failed = true;
                    return stmt;
                }
                ++replaced;
                return replacement;
            }

            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > children;
                unsigned before = replaced;
                for (clang::Stmt *child : compound->body()) {
                    children.push_back(ReplaceGotoWithClonedCleanupThenJoin(
                        ctx, child, cleanup_label, join_label, cleanup_stmts, replaced, failed
                    ));
                    if (failed) { return stmt; }
                }
                return replaced != before ? detail::MakeCompound(ctx, children) : stmt;
            }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(ReplaceGotoWithClonedCleanupThenJoin(
                    ctx, ifs->getThen(), cleanup_label, join_label, cleanup_stmts, replaced,
                    failed
                ));
                if (failed) { return stmt; }
                if (ifs->getElse()) {
                    ifs->setElse(ReplaceGotoWithClonedCleanupThenJoin(
                        ctx, ifs->getElse(), cleanup_label, join_label, cleanup_stmts, replaced,
                        failed
                    ));
                }
                return stmt;
            }

            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                label->setSubStmt(ReplaceGotoWithClonedCleanupThenJoin(
                    ctx, label->getSubStmt(), cleanup_label, join_label, cleanup_stmts, replaced,
                    failed
                ));
                return label;
            }

            // Do not clone cleanup logic into nested loop/switch scopes; a
            // goto there may have different structured-control ownership.
            if (llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::WhileStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt))
            {
                return stmt;
            }

            return stmt;
        }

        clang::Stmt *CloneCleanupLabelBeforeJoinGotos(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs, bool &changed
        ) {
            if (!stmt) { return stmt; }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(CloneCleanupLabelBeforeJoinGotos(ctx, ifs->getThen(), refs, changed));
                if (ifs->getElse()) {
                    ifs->setElse(
                        CloneCleanupLabelBeforeJoinGotos(ctx, ifs->getElse(), refs, changed)
                    );
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(CloneCleanupLabelBeforeJoinGotos(ctx, ws->getBody(), refs, changed));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(CloneCleanupLabelBeforeJoinGotos(ctx, ds->getBody(), refs, changed));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(CloneCleanupLabelBeforeJoinGotos(ctx, fs->getBody(), refs, changed));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(
                    CloneCleanupLabelBeforeJoinGotos(ctx, ls->getSubStmt(), refs, changed)
                );
                return ls;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(CloneCleanupLabelBeforeJoinGotos(ctx, sw->getBody(), refs, changed));
                return sw;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                bool child_changed = false;
                child              = CloneCleanupLabelBeforeJoinGotos(ctx, child, refs, child_changed);
                if (child_changed) { changed = true; }
            }

            bool body_changed  = changed;
            bool local_changed = true;
            while (local_changed) {
                local_changed = false;
                for (size_t label_idx = 0; label_idx + 1 < body.size(); ++label_idx) {
                    clang::LabelDecl *cleanup_label = LeadingLabelDecl(body[label_idx]);
                    if (!cleanup_label) { continue; }
                    auto ref_it = refs.find(cleanup_label);
                    if (ref_it == refs.end() || ref_it->second == 0
                        || ref_it->second > kMaxClonedTailUses)
                    {
                        continue;
                    }

                    LocalLabelBlock cleanup_block;
                    if (!ExtractLocalLabelBlock(ctx, body, label_idx, cleanup_block)) {
                        continue;
                    }
                    if (cleanup_block.end >= body.size()) { continue; }
                    clang::LabelDecl *join_label = LeadingLabelDecl(body[cleanup_block.end]);
                    if (!join_label || join_label == cleanup_label) { continue; }
                    if (cleanup_block.stmts.size() > kMaxTerminalPrefixStmts) { continue; }
                    if (!CleanupSeqHasCleanupLikeCall(cleanup_block.stmts)) { continue; }
                    if (SeqEndsWithTerminator(ctx, cleanup_block.stmts)) { continue; }
                    if (!StraightLineContinuationSeqIsSafe(cleanup_block.stmts)) { continue; }

                    unsigned replaced = 0;
                    bool failed       = false;
                    for (size_t i = 0; i < cleanup_block.begin; ++i) {
                        body[i] = ReplaceGotoWithClonedCleanupThenJoin(
                            ctx, body[i], cleanup_label, join_label, cleanup_block.stmts,
                            replaced, failed
                        );
                        if (failed) { break; }
                    }
                    if (failed || replaced != ref_it->second) { continue; }

                    body.erase(
                        body.begin() + static_cast< ptrdiff_t >(cleanup_block.begin),
                        body.begin() + static_cast< ptrdiff_t >(cleanup_block.end)
                    );
                    changed      = true;
                    body_changed = true;
                    local_changed = true;
                    break;
                }
            }

            return body_changed ? detail::MakeCompound(ctx, body) : stmt;
        }

        clang::Stmt *CloneSmallStraightLineLabelBeforeJoinGotos(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs, bool &changed
        ) {
            if (!stmt) { return stmt; }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(
                    CloneSmallStraightLineLabelBeforeJoinGotos(ctx, ifs->getThen(), refs, changed)
                );
                if (ifs->getElse()) {
                    ifs->setElse(CloneSmallStraightLineLabelBeforeJoinGotos(
                        ctx, ifs->getElse(), refs, changed
                    ));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(
                    CloneSmallStraightLineLabelBeforeJoinGotos(ctx, ws->getBody(), refs, changed)
                );
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(
                    CloneSmallStraightLineLabelBeforeJoinGotos(ctx, ds->getBody(), refs, changed)
                );
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(
                    CloneSmallStraightLineLabelBeforeJoinGotos(ctx, fs->getBody(), refs, changed)
                );
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(CloneSmallStraightLineLabelBeforeJoinGotos(
                    ctx, ls->getSubStmt(), refs, changed
                ));
                return ls;
            }
            if (llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::CaseStmt >(stmt)
                || llvm::isa< clang::DefaultStmt >(stmt))
            {
                return stmt;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                bool child_changed = false;
                child = CloneSmallStraightLineLabelBeforeJoinGotos(ctx, child, refs, child_changed);
                if (child_changed) { changed = true; }
            }

            bool body_changed  = changed;
            bool local_changed = true;
            while (local_changed) {
                local_changed = false;
                for (size_t label_idx = 0; label_idx + 1 < body.size(); ++label_idx) {
                    clang::LabelDecl *target_label = LeadingLabelDecl(body[label_idx]);
                    if (!target_label) { continue; }
                    auto ref_it = refs.find(target_label);
                    if (ref_it == refs.end() || ref_it->second == 0
                        || ref_it->second > kMaxClonedTailUses)
                    {
                        continue;
                    }

                    LocalLabelBlock target_block;
                    if (!ExtractLocalLabelBlock(ctx, body, label_idx, target_block)) {
                        continue;
                    }
                    if (target_block.end >= body.size()) { continue; }
                    clang::LabelDecl *join_label = LeadingLabelDecl(body[target_block.end]);
                    if (!join_label || join_label == target_label) { continue; }

                    if (target_block.stmts.size() > kMaxConditionalFallthroughBodyStmts) {
                        continue;
                    }
                    if (SeqEndsWithTerminator(ctx, target_block.stmts)) { continue; }
                    if (!StraightLineContinuationSeqIsSafe(target_block.stmts)) { continue; }
                    if (SeqHasCallExpr(target_block.stmts)) { continue; }

                    unsigned replaced = 0;
                    bool failed       = false;
                    for (size_t i = 0; i < target_block.begin; ++i) {
                        body[i] = ReplaceGotoWithClonedCleanupThenJoin(
                            ctx, body[i], target_label, join_label, target_block.stmts,
                            replaced, failed
                        );
                        if (failed) { break; }
                    }
                    if (failed || replaced != ref_it->second) { continue; }

                    body.erase(
                        body.begin() + static_cast< ptrdiff_t >(target_block.begin),
                        body.begin() + static_cast< ptrdiff_t >(target_block.end)
                    );
                    changed       = true;
                    body_changed  = true;
                    local_changed = true;
                    break;
                }
            }

            return body_changed ? detail::MakeCompound(ctx, body) : stmt;
        }

        bool TryFoldThreeGuardJoinAt(
            clang::ASTContext &ctx, std::vector< clang::Stmt * > &body, size_t guard_idx,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (guard_idx + 3 >= body.size()) { return false; }

            auto *skip_guard = llvm::dyn_cast< clang::IfStmt >(body[guard_idx]);
            auto *alt_guard  = llvm::dyn_cast< clang::IfStmt >(body[guard_idx + 1]);
            auto *join_guard = llvm::dyn_cast< clang::IfStmt >(body[guard_idx + 2]);
            if (!skip_guard || !alt_guard || !join_guard) { return false; }
            if (skip_guard->getElse() || alt_guard->getElse() || join_guard->getElse()
                || skip_guard->getInit() || alt_guard->getInit() || join_guard->getInit()
                || skip_guard->getConditionVariable() || alt_guard->getConditionVariable()
                || join_guard->getConditionVariable())
            {
                return false;
            }

            clang::GotoStmt *skip_goto = SingleGotoStmt(skip_guard->getThen());
            clang::GotoStmt *alt_goto  = SingleGotoStmt(alt_guard->getThen());
            clang::GotoStmt *join_goto = SingleGotoStmt(join_guard->getThen());
            if (!skip_goto || !alt_goto || !join_goto) { return false; }

            clang::LabelDecl *join_label = skip_goto->getLabel();
            clang::LabelDecl *alt_label  = alt_goto->getLabel();
            if (!join_label || !alt_label || join_goto->getLabel() != join_label
                || join_label == alt_label)
            {
                return false;
            }

            auto alt_ref = refs.find(alt_label);
            auto join_ref = refs.find(join_label);
            if (alt_ref == refs.end() || alt_ref->second != 1
                || join_ref == refs.end() || join_ref->second != 2)
            {
                return false;
            }

            LocalLabelBlock alt_block;
            if (!FindLabelBlockByDecl(ctx, body, alt_label, alt_block)) { return false; }
            if (alt_block.begin <= guard_idx + 2) { return false; }
            if (alt_block.stmts.empty() || alt_block.stmts.size() > kMaxTerminalPrefixStmts) {
                return false;
            }

            if (alt_block.end >= body.size()) { return false; }
            if (LeadingLabelDecl(body[alt_block.end]) != join_label) { return false; }

            std::vector< clang::Stmt * > normal_region(
                body.begin() + static_cast< ptrdiff_t >(guard_idx + 3),
                body.begin() + static_cast< ptrdiff_t >(alt_block.begin)
            );
            if (normal_region.empty() || !SeqEndsWithTerminator(ctx, normal_region)) {
                return false;
            }
            if (SeqHasUnsafeGuardedJoinMoveControl(normal_region)
                || SeqHasUnsafeGuardedJoinMoveControl(alt_block.stmts))
            {
                return false;
            }
            if (SeqEndsWithTerminator(ctx, alt_block.stmts)) { return false; }

            auto loc = skip_guard->getIfLoc();
            auto *normal_if = clang::IfStmt::Create(
                ctx, join_guard->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr,
                nullptr, NegateExpr(ctx, join_guard->getCond()), join_guard->getLParenLoc(),
                join_guard->getRParenLoc(), StmtFromSeq(ctx, normal_region)
            );
            auto *alt_if = clang::IfStmt::Create(
                ctx, alt_guard->getIfLoc(), clang::IfStatementKind::Ordinary, nullptr,
                nullptr, alt_guard->getCond(), alt_guard->getLParenLoc(),
                alt_guard->getRParenLoc(), StmtFromSeq(ctx, alt_block.stmts),
                alt_guard->getElseLoc(), normal_if
            );
            auto *outer_if = clang::IfStmt::Create(
                ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                NegateExpr(ctx, skip_guard->getCond()), skip_guard->getLParenLoc(),
                skip_guard->getRParenLoc(), alt_if
            );

            body.erase(
                body.begin() + static_cast< ptrdiff_t >(guard_idx),
                body.begin() + static_cast< ptrdiff_t >(alt_block.end)
            );
            body.insert(body.begin() + static_cast< ptrdiff_t >(guard_idx), outer_if);
            return true;
        }

        bool StripRegionTrailingGotoToJoin(
            clang::ASTContext &ctx, const std::vector< clang::Stmt * > &region,
            clang::LabelDecl *join_label, std::vector< clang::Stmt * > &stripped
        ) {
            if (region.empty() || !join_label) { return false; }
            clang::GotoStmt *join_goto = llvm::dyn_cast_or_null< clang::GotoStmt >(
                DeepTrailingStmt(StmtFromSeq(ctx, region))
            );
            if (!join_goto || join_goto->getLabel() != join_label) { return false; }

            clang::Stmt *stripped_stmt =
                StripTrailingGoto(ctx, StmtFromSeq(ctx, region), join_label->getName());
            stripped.clear();
            AppendStmtSequence(stripped_stmt, stripped);
            return !SeqHasUnsafeGuardedJoinMoveControl(stripped);
        }

        bool TryFoldIfGotoAltThenJoinAt(
            clang::ASTContext &ctx, std::vector< clang::Stmt * > &body, size_t if_idx,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (if_idx + 2 >= body.size()) { return false; }
            auto *ifs = llvm::dyn_cast< clang::IfStmt >(body[if_idx]);
            if (!ifs || ifs->getElse() || ifs->getInit() || ifs->getConditionVariable()) {
                return false;
            }

            clang::GotoStmt *alt_goto = SingleGotoStmt(ifs->getThen());
            if (!alt_goto || !alt_goto->getLabel()) { return false; }
            clang::LabelDecl *alt_label = alt_goto->getLabel();
            auto alt_ref                = refs.find(alt_label);
            if (alt_ref == refs.end() || alt_ref->second != 1) { return false; }

            LocalLabelBlock alt_block;
            if (!FindLabelBlockByDecl(ctx, body, alt_label, alt_block)) { return false; }
            if (alt_block.begin <= if_idx + 1 || alt_block.end >= body.size()) {
                return false;
            }

            clang::LabelDecl *join_label = LeadingLabelDecl(body[alt_block.end]);
            if (!IsSyntheticScopeJoinLabel(join_label)) { return false; }
            if (alt_block.stmts.empty() || alt_block.stmts.size() > kMaxTerminalPrefixStmts) {
                return false;
            }
            if (SeqHasUnsafeGuardedJoinMoveControl(alt_block.stmts)) { return false; }
            if (SeqEndsWithTerminator(ctx, alt_block.stmts)) { return false; }

            std::vector< clang::Stmt * > false_region(
                body.begin() + static_cast< ptrdiff_t >(if_idx + 1),
                body.begin() + static_cast< ptrdiff_t >(alt_block.begin)
            );
            std::vector< clang::Stmt * > false_stripped;
            if (!StripRegionTrailingGotoToJoin(ctx, false_region, join_label, false_stripped)) {
                return false;
            }

            auto loc = ifs->getIfLoc();
            auto *new_if = clang::IfStmt::Create(
                ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr, ifs->getCond(),
                ifs->getLParenLoc(), ifs->getRParenLoc(), StmtFromSeq(ctx, alt_block.stmts),
                ifs->getElseLoc(), StmtFromSeq(ctx, false_stripped)
            );

            body.erase(
                body.begin() + static_cast< ptrdiff_t >(if_idx),
                body.begin() + static_cast< ptrdiff_t >(alt_block.end)
            );
            body.insert(body.begin() + static_cast< ptrdiff_t >(if_idx), new_if);
            return true;
        }

        bool TryFoldIfThenJoinElseAltAt(
            clang::ASTContext &ctx, std::vector< clang::Stmt * > &body, size_t if_idx
        ) {
            if (if_idx + 1 >= body.size()) { return false; }
            auto *ifs = llvm::dyn_cast< clang::IfStmt >(body[if_idx]);
            if (!ifs || ifs->getElse() || ifs->getInit() || ifs->getConditionVariable()) {
                return false;
            }

            clang::LabelDecl *alt_label = LeadingLabelDecl(body[if_idx + 1]);
            if (!alt_label) { return false; }
            LocalLabelBlock alt_block;
            if (!ExtractLocalLabelBlock(ctx, body, if_idx + 1, alt_block)) { return false; }
            if (alt_block.end >= body.size()) { return false; }

            clang::LabelDecl *join_label = LeadingLabelDecl(body[alt_block.end]);
            if (!IsSyntheticScopeJoinLabel(join_label)) { return false; }
            if (alt_block.stmts.empty() || alt_block.stmts.size() > kMaxTerminalPrefixStmts) {
                return false;
            }
            if (SeqHasUnsafeGuardedJoinMoveControl(alt_block.stmts)) { return false; }
            if (SeqEndsWithTerminator(ctx, alt_block.stmts)) { return false; }

            std::vector< clang::Stmt * > then_region;
            AppendStmtSequence(ifs->getThen(), then_region);
            std::vector< clang::Stmt * > then_stripped;
            if (GotoElimGetTarget(DeepTrailingStmt(ifs->getThen())) != join_label->getName()) {
                return false;
            }
            clang::Stmt *stripped_then =
                StripTrailingGoto(ctx, ifs->getThen(), join_label->getName());
            then_stripped.clear();
            AppendStmtSequence(stripped_then, then_stripped);
            if (SeqHasUnsafeGuardedJoinMoveControl(then_stripped)) { return false; }

            auto loc = ifs->getIfLoc();
            auto *new_if = clang::IfStmt::Create(
                ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr, ifs->getCond(),
                ifs->getLParenLoc(), ifs->getRParenLoc(), StmtFromSeq(ctx, then_stripped),
                ifs->getElseLoc(), StmtFromSeq(ctx, alt_block.stmts)
            );

            body.erase(
                body.begin() + static_cast< ptrdiff_t >(if_idx),
                body.begin() + static_cast< ptrdiff_t >(alt_block.end)
            );
            body.insert(body.begin() + static_cast< ptrdiff_t >(if_idx), new_if);
            return true;
        }

        clang::Stmt *FoldGuardedJoinLabelChains(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs, bool &changed
        ) {
            if (!stmt) { return stmt; }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(
                    FoldGuardedJoinLabelChains(ctx, ifs->getThen(), refs, changed)
                );
                if (ifs->getElse()) {
                    ifs->setElse(
                        FoldGuardedJoinLabelChains(ctx, ifs->getElse(), refs, changed)
                    );
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(FoldGuardedJoinLabelChains(ctx, ws->getBody(), refs, changed));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(FoldGuardedJoinLabelChains(ctx, ds->getBody(), refs, changed));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(FoldGuardedJoinLabelChains(ctx, fs->getBody(), refs, changed));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(
                    FoldGuardedJoinLabelChains(ctx, ls->getSubStmt(), refs, changed)
                );
                return ls;
            }
            if (llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::CaseStmt >(stmt)
                || llvm::isa< clang::DefaultStmt >(stmt))
            {
                return stmt;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = FoldGuardedJoinLabelChains(ctx, child, refs, changed);
            }

            bool body_changed  = changed;
            bool local_changed = true;
            while (local_changed) {
                local_changed = false;
                for (size_t i = 0; i < body.size(); ++i) {
                    if (TryFoldThreeGuardJoinAt(ctx, body, i, refs)
                        || TryFoldIfGotoAltThenJoinAt(ctx, body, i, refs)
                        || TryFoldIfThenJoinElseAltAt(ctx, body, i))
                    {
                        changed       = true;
                        body_changed  = true;
                        local_changed = true;
                        break;
                    }
                }
            }

            return body_changed ? detail::MakeCompound(ctx, body) : stmt;
        }

        clang::Stmt *FoldCrossCompoundIfLabelDiamonds(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(FoldCrossCompoundIfLabelDiamonds(ctx, ifs->getThen(), refs));
                if (ifs->getElse()) {
                    ifs->setElse(FoldCrossCompoundIfLabelDiamonds(ctx, ifs->getElse(), refs));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(FoldCrossCompoundIfLabelDiamonds(ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(FoldCrossCompoundIfLabelDiamonds(ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(FoldCrossCompoundIfLabelDiamonds(ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(FoldCrossCompoundIfLabelDiamonds(ctx, ls->getSubStmt(), refs));
                return ls;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(FoldCrossCompoundIfLabelDiamonds(ctx, sw->getBody(), refs));
                return sw;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = FoldCrossCompoundIfLabelDiamonds(ctx, child, refs);
            }

            for (size_t if_idx = 0; if_idx + 2 < body.size(); ++if_idx) {
                auto *ifs = llvm::dyn_cast< clang::IfStmt >(body[if_idx]);
                if (!ifs || ifs->getElse() || ifs->getInit() || ifs->getConditionVariable()) {
                    continue;
                }

                clang::LabelDecl *alt_label = LeadingLabelDecl(body[if_idx + 1]);
                if (!alt_label) { continue; }

                LocalLabelBlock alt_block;
                if (!ExtractLocalLabelBlock(ctx, body, if_idx + 1, alt_block)) { continue; }
                if (alt_block.end >= body.size()) { continue; }
                clang::LabelDecl *join_label = LeadingLabelDecl(body[alt_block.end]);
                if (!join_label || join_label == alt_label) { continue; }

                const unsigned alt_gotos  = CountGotosToLabel(ifs->getThen(), alt_label);
                const unsigned join_gotos = CountGotosToLabel(ifs->getThen(), join_label);
                if (alt_gotos == 0 || join_gotos == 0) { continue; }

                auto alt_ref = refs.find(alt_label);
                if (alt_ref == refs.end() || alt_ref->second != alt_gotos) { continue; }

                if (GotoElimGetTarget(DeepTrailingStmt(ifs->getThen()))
                    != join_label->getName())
                {
                    continue;
                }
                if (CountAllGotos(ifs->getThen()) != alt_gotos + join_gotos) { continue; }
                if (!StraightLineContinuationSeqIsSafe(alt_block.stmts)) { continue; }
                if (SeqEndsWithTerminator(ctx, alt_block.stmts)) { continue; }

                unsigned replaced           = 0;
                bool failed                 = false;
                clang::Stmt *then_rewritten = ReplaceGotoWithClonedJoinTail(
                    ctx, ifs->getThen(), alt_label, join_label, alt_block.stmts, replaced,
                    failed
                );
                if (failed || replaced != alt_gotos) { continue; }

                then_rewritten = StripTrailingGoto(ctx, then_rewritten, join_label->getName());
                clang::Stmt *else_stmt = StmtFromSeq(ctx, alt_block.stmts);

                auto loc     = ifs->getIfLoc();
                auto *new_if = clang::IfStmt::Create(
                    ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                    ifs->getCond(), loc, loc, then_rewritten, loc, else_stmt
                );

                body[if_idx] = new_if;
                body.erase(
                    body.begin() + static_cast< ptrdiff_t >(alt_block.begin),
                    body.begin() + static_cast< ptrdiff_t >(alt_block.end)
                );
                return detail::MakeCompound(ctx, body);
            }

            return detail::MakeCompound(ctx, body);
        }

        struct DispatchGuard
        {
            clang::IfStmt *ifs       = nullptr;
            clang::LabelDecl *target = nullptr;
        };

        clang::Stmt *BuildDispatchGuardChain(
            clang::ASTContext &ctx, const std::vector< DispatchGuard > &guards,
            const std::unordered_map< clang::LabelDecl *, std::vector< clang::Stmt * > > &arms,
            clang::Stmt *default_stmt
        ) {
            clang::Stmt *tail = default_stmt;
            for (size_t i = guards.size(); i != 0; --i) {
                const DispatchGuard &guard = guards[i - 1];
                auto arm_it                = arms.find(guard.target);
                if (arm_it == arms.end()) { return nullptr; }
                auto loc = guard.ifs->getIfLoc();
                tail     = clang::IfStmt::Create(
                    ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                    guard.ifs->getCond(), loc, loc, StmtFromSeq(ctx, arm_it->second), loc, tail
                );
            }
            return tail;
        }

        bool TryFoldCompoundDispatchChainAt(
            clang::ASTContext &ctx, std::vector< clang::Stmt * > &body, size_t guard_idx,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            std::vector< DispatchGuard > candidate_guards;
            size_t cursor = guard_idx;
            while (cursor < body.size()) {
                auto *ifs = llvm::dyn_cast< clang::IfStmt >(body[cursor]);
                if (!ifs || ifs->getElse() || ifs->getInit() || ifs->getConditionVariable()) {
                    break;
                }
                clang::GotoStmt *go = SingleGotoStmt(ifs->getThen());
                if (!go || !go->getLabel()) { break; }
                candidate_guards.push_back({ ifs, go->getLabel() });
                ++cursor;
            }
            if (candidate_guards.size() < 2) { return false; }

            auto try_prefix = [&](size_t guard_count) -> bool {
                std::vector< DispatchGuard > guards(
                    candidate_guards.begin(),
                    candidate_guards.begin() + static_cast< ptrdiff_t >(guard_count)
                );
                std::unordered_map< clang::LabelDecl *, unsigned > guard_refs;
                for (const DispatchGuard &guard : guards) { ++guard_refs[guard.target]; }

                for (const auto &[label, count] : guard_refs) {
                    auto ref_it = refs.find(label);
                    if (ref_it == refs.end() || ref_it->second != count) {
                        return false;
                    }
                }

                size_t default_start = guard_idx + guard_count;
                if (default_start >= body.size()) {
                    return false;
                }

                std::unordered_map< clang::LabelDecl *, LocalLabelBlock > target_blocks;
                size_t first_label_idx = body.size();
                size_t join_idx        = 0;
                for (const DispatchGuard &guard : guards) {
                    if (target_blocks.count(guard.target) != 0) { continue; }
                    LocalLabelBlock block;
                    if (!FindLabelBlockByDecl(ctx, body, guard.target, block)) {
                        return false;
                    }
                    if (block.begin <= default_start) {
                        return false;
                    }
                    first_label_idx = std::min(first_label_idx, block.begin);
                    join_idx        = std::max(join_idx, block.end);
                    target_blocks.emplace(guard.target, block);
                }
                if (first_label_idx == body.size() || join_idx >= body.size()) {
                    return false;
                }

                for (size_t i = default_start; i < first_label_idx; ++i) {
                    if (LeadingLabelDecl(body[i])) {
                        return false;
                    }
                }
                for (size_t i = first_label_idx; i < join_idx; ++i) {
                    clang::LabelDecl *decl = LeadingLabelDecl(body[i]);
                    if (!decl) { continue; }
                    if (target_blocks.find(decl) == target_blocks.end()) {
                        return false;
                    }
                }

                clang::LabelDecl *join_label = LeadingLabelDecl(body[join_idx]);
                if (!join_label) {
                    return false;
                }

                std::unordered_map< clang::LabelDecl *, std::vector< clang::Stmt * > > arms;
                unsigned stripped_join_gotos = 0;
                for (const DispatchGuard &guard : guards) {
                    const LocalLabelBlock &block = target_blocks.find(guard.target)->second;
                    std::vector< clang::Stmt * > stmts = block.stmts;
                    if (stmts.empty() || stmts.size() > kMaxTerminalPrefixStmts) {
                        return false;
                    }

                    clang::GotoStmt *tail_go = llvm::dyn_cast_or_null< clang::GotoStmt >(
                        DeepTrailingStmt(StmtFromSeq(ctx, stmts))
                    );
                    if (tail_go && tail_go->getLabel() == join_label) {
                        clang::Stmt *stripped =
                            StripTrailingGoto(ctx, StmtFromSeq(ctx, stmts), join_label->getName());
                        stmts.clear();
                        AppendStmtSequence(stripped, stmts);
                        ++stripped_join_gotos;
                    } else if (block.end != join_idx) {
                        return false;
                    }

                    if (SeqHasUnsafeStructure(
                            stmts, /*allow_goto=*/false, /*allow_return=*/false
                        ))
                    {
                        return false;
                    }
                    arms.emplace(guard.target, std::move(stmts));
                }
                if (stripped_join_gotos == 0) {
                    return false;
                }

                std::vector< clang::Stmt * > default_region(
                    body.begin() + static_cast< ptrdiff_t >(default_start),
                    body.begin() + static_cast< ptrdiff_t >(first_label_idx)
                );
                if (default_region.empty()) {
                    return false;
                }
                if (SeqHasUnsafeStructure(
                        default_region, /*allow_goto=*/true, /*allow_return=*/true
                    ))
                {
                    return false;
                }

                clang::Stmt *replacement =
                    BuildDispatchGuardChain(ctx, guards, arms, StmtFromSeq(ctx, default_region));
                if (!replacement) { return false; }

                body.erase(
                    body.begin() + static_cast< ptrdiff_t >(guard_idx),
                    body.begin() + static_cast< ptrdiff_t >(join_idx)
                );
                body.insert(body.begin() + static_cast< ptrdiff_t >(guard_idx), replacement);
                return true;
            };

            for (size_t guard_count = candidate_guards.size(); guard_count >= 2; --guard_count) {
                if (try_prefix(guard_count)) { return true; }
                if (guard_count == 2) { break; }
            }
            return false;
        }

        bool TryFoldCrossCompoundDispatchArmAt(
            clang::ASTContext &ctx, std::vector< clang::Stmt * > &parent_body,
            size_t carrier_idx, clang::Stmt *&arm,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            auto *compound = llvm::dyn_cast_or_null< clang::CompoundStmt >(arm);
            if (!compound || carrier_idx + 1 >= parent_body.size()) { return false; }

            std::vector< clang::Stmt * > arm_body(
                compound->body_begin(), compound->body_end()
            );
            for (size_t guard_idx = 0; guard_idx + 1 < arm_body.size(); ++guard_idx) {
                std::vector< DispatchGuard > candidate_guards;
                size_t cursor = guard_idx;
                while (cursor < arm_body.size()) {
                    auto *ifs = llvm::dyn_cast< clang::IfStmt >(arm_body[cursor]);
                    if (!ifs || ifs->getElse() || ifs->getInit()
                        || ifs->getConditionVariable())
                    {
                        break;
                    }
                    clang::GotoStmt *go = SingleGotoStmt(ifs->getThen());
                    if (!go || !go->getLabel()) { break; }
                    candidate_guards.push_back({ ifs, go->getLabel() });
                    ++cursor;
                }
                if (candidate_guards.size() < 2) { continue; }

                auto try_prefix = [&](size_t guard_count) -> bool {
                    std::vector< DispatchGuard > guards(
                        candidate_guards.begin(),
                        candidate_guards.begin() + static_cast< ptrdiff_t >(guard_count)
                    );
                    std::unordered_map< clang::LabelDecl *, unsigned > guard_refs;
                    for (const DispatchGuard &guard : guards) { ++guard_refs[guard.target]; }
                    for (const auto &[label, count] : guard_refs) {
                        auto ref_it = refs.find(label);
                        if (ref_it == refs.end() || ref_it->second != count) {
                            return false;
                        }
                    }

                    size_t default_start = guard_idx + guard_count;
                    if (default_start >= arm_body.size()) { return false; }

                    std::unordered_map< clang::LabelDecl *, LocalLabelBlock > target_blocks;
                    size_t first_label_idx = parent_body.size();
                    size_t join_idx        = 0;
                    for (const DispatchGuard &guard : guards) {
                        if (target_blocks.count(guard.target) != 0) { continue; }
                        LocalLabelBlock block;
                        if (!FindLabelBlockByDecl(ctx, parent_body, guard.target, block)) {
                            return false;
                        }
                        if (block.begin <= carrier_idx) { return false; }
                        first_label_idx = std::min(first_label_idx, block.begin);
                        join_idx        = std::max(join_idx, block.end);
                        target_blocks.emplace(guard.target, block);
                    }
                    if (first_label_idx != carrier_idx + 1 || join_idx >= parent_body.size()) {
                        return false;
                    }
                    for (size_t i = first_label_idx; i < join_idx; ++i) {
                        clang::LabelDecl *decl = LeadingLabelDecl(parent_body[i]);
                        if (!decl) { continue; }
                        if (target_blocks.find(decl) == target_blocks.end()) { return false; }
                    }

                    clang::LabelDecl *join_label = LeadingLabelDecl(parent_body[join_idx]);
                    if (!join_label) { return false; }

                    std::unordered_map< clang::LabelDecl *, std::vector< clang::Stmt * > > arms;
                    unsigned stripped_join_gotos = 0;
                    for (const DispatchGuard &guard : guards) {
                        const LocalLabelBlock &block = target_blocks.find(guard.target)->second;
                        std::vector< clang::Stmt * > stmts = block.stmts;
                        if (stmts.empty() || stmts.size() > kMaxTerminalPrefixStmts) {
                            return false;
                        }

                        clang::GotoStmt *tail_go =
                            llvm::dyn_cast_or_null< clang::GotoStmt >(
                                DeepTrailingStmt(StmtFromSeq(ctx, stmts))
                            );
                        if (tail_go && tail_go->getLabel() == join_label) {
                            clang::Stmt *stripped = StripTrailingGoto(
                                ctx, StmtFromSeq(ctx, stmts), join_label->getName()
                            );
                            stmts.clear();
                            AppendStmtSequence(stripped, stmts);
                            ++stripped_join_gotos;
                        } else if (block.end != join_idx) {
                            return false;
                        }
                        if (SeqHasUnsafeStructure(
                                stmts, /*allow_goto=*/false, /*allow_return=*/false
                            ))
                        {
                            return false;
                        }
                        arms.emplace(guard.target, std::move(stmts));
                    }
                    if (stripped_join_gotos == 0) { return false; }

                    std::vector< clang::Stmt * > default_region(
                        arm_body.begin() + static_cast< ptrdiff_t >(default_start),
                        arm_body.end()
                    );
                    if (default_region.empty()) { return false; }
                    if (SeqHasUnsafeStructure(
                            default_region, /*allow_goto=*/true, /*allow_return=*/true
                        ))
                    {
                        return false;
                    }

                    clang::Stmt *replacement = BuildDispatchGuardChain(
                        ctx, guards, arms, StmtFromSeq(ctx, default_region)
                    );
                    if (!replacement) { return false; }

                    std::vector< clang::Stmt * > new_arm(
                        arm_body.begin(), arm_body.begin() + static_cast< ptrdiff_t >(guard_idx)
                    );
                    new_arm.push_back(replacement);
                    arm = StmtFromSeq(ctx, new_arm);

                    parent_body.erase(
                        parent_body.begin() + static_cast< ptrdiff_t >(first_label_idx),
                        parent_body.begin() + static_cast< ptrdiff_t >(join_idx)
                    );
                    return true;
                };

                for (size_t guard_count = candidate_guards.size(); guard_count >= 2; --guard_count)
                {
                    if (try_prefix(guard_count)) { return true; }
                    if (guard_count == 2) { break; }
                }
            }
            return false;
        }

        clang::Stmt *FoldCrossCompoundDispatchChains(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(FoldCrossCompoundDispatchChains(ctx, ifs->getThen(), refs));
                if (ifs->getElse()) {
                    ifs->setElse(FoldCrossCompoundDispatchChains(ctx, ifs->getElse(), refs));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(FoldCrossCompoundDispatchChains(ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(FoldCrossCompoundDispatchChains(ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(FoldCrossCompoundDispatchChains(ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(FoldCrossCompoundDispatchChains(ctx, ls->getSubStmt(), refs));
                return ls;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(FoldCrossCompoundDispatchChains(ctx, sw->getBody(), refs));
                return sw;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = FoldCrossCompoundDispatchChains(ctx, child, refs);
            }

            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t i = 0; i < body.size(); ++i) {
                    if (TryFoldCompoundDispatchChainAt(ctx, body, i, refs)) {
                        changed = true;
                        break;
                    }
                    if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(body[i])) {
                        clang::Stmt *then_arm = ifs->getThen();
                        if (TryFoldCrossCompoundDispatchArmAt(ctx, body, i, then_arm, refs)) {
                            ifs->setThen(then_arm);
                            changed = true;
                            break;
                        }
                        clang::Stmt *else_arm = ifs->getElse();
                        if (TryFoldCrossCompoundDispatchArmAt(ctx, body, i, else_arm, refs)) {
                            ifs->setElse(else_arm);
                            changed = true;
                            break;
                        }
                    }
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        bool IsLoopStmt(clang::Stmt *stmt) {
            return llvm::isa< clang::WhileStmt >(stmt) || llvm::isa< clang::DoStmt >(stmt)
                || llvm::isa< clang::ForStmt >(stmt);
        }

        clang::Expr *CreateTrueExpr(clang::ASTContext &ctx) {
            return clang::IntegerLiteral::Create(
                ctx, llvm::APInt(ctx.getIntWidth(ctx.IntTy), 1), ctx.IntTy, VirtualLoc(ctx)
            );
        }

        clang::Stmt *GetLoopBody(clang::Stmt *stmt) {
            if (auto *ws = llvm::dyn_cast_or_null< clang::WhileStmt >(stmt)) {
                return ws->getBody();
            }
            if (auto *ds = llvm::dyn_cast_or_null< clang::DoStmt >(stmt)) {
                return ds->getBody();
            }
            if (auto *fs = llvm::dyn_cast_or_null< clang::ForStmt >(stmt)) {
                return fs->getBody();
            }
            return nullptr;
        }

        void SetLoopBody(clang::Stmt *stmt, clang::Stmt *body) {
            if (auto *ws = llvm::dyn_cast_or_null< clang::WhileStmt >(stmt)) {
                ws->setBody(body);
            } else if (auto *ds = llvm::dyn_cast_or_null< clang::DoStmt >(stmt)) {
                ds->setBody(body);
            } else if (auto *fs = llvm::dyn_cast_or_null< clang::ForStmt >(stmt)) {
                fs->setBody(body);
            }
        }

        clang::Stmt *ReplaceGotoWithBreakInCurrentLoop(
            clang::ASTContext &ctx, clang::Stmt *stmt, clang::LabelDecl *target,
            unsigned &replaced
        ) {
            if (!stmt) { return stmt; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                if (gs->getLabel() == target) {
                    ++replaced;
                    return new (ctx) clang::BreakStmt(gs->getGotoLoc());
                }
                return stmt;
            }

            // `break` inside a nested loop or switch would exit the nested scope,
            // not the loop whose exit label we are rewriting.
            if (llvm::isa< clang::WhileStmt >(stmt) || llvm::isa< clang::DoStmt >(stmt)
                || llvm::isa< clang::ForStmt >(stmt) || llvm::isa< clang::SwitchStmt >(stmt)
                || llvm::isa< clang::CaseStmt >(stmt) || llvm::isa< clang::DefaultStmt >(stmt))
            {
                return stmt;
            }

            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > children;
                unsigned before = replaced;
                for (clang::Stmt *child : compound->body()) {
                    children.push_back(
                        ReplaceGotoWithBreakInCurrentLoop(ctx, child, target, replaced)
                    );
                }
                return replaced != before ? detail::MakeCompound(ctx, children) : stmt;
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(
                    ReplaceGotoWithBreakInCurrentLoop(ctx, ifs->getThen(), target, replaced)
                );
                if (ifs->getElse()) {
                    ifs->setElse(
                        ReplaceGotoWithBreakInCurrentLoop(ctx, ifs->getElse(), target, replaced)
                    );
                }
                return stmt;
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                label->setSubStmt(ReplaceGotoWithBreakInCurrentLoop(
                    ctx, label->getSubStmt(), target, replaced
                ));
                return stmt;
            }
            return stmt;
        }

        clang::Stmt *
        ConvertImmediateLoopExitGotosToBreak(clang::ASTContext &ctx, clang::Stmt *stmt) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(ConvertImmediateLoopExitGotosToBreak(ctx, ifs->getThen()));
                if (ifs->getElse()) {
                    ifs->setElse(ConvertImmediateLoopExitGotosToBreak(ctx, ifs->getElse()));
                }
                return ifs;
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                label->setSubStmt(
                    ConvertImmediateLoopExitGotosToBreak(ctx, label->getSubStmt())
                );
                return label;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(ConvertImmediateLoopExitGotosToBreak(ctx, ws->getBody()));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(ConvertImmediateLoopExitGotosToBreak(ctx, ds->getBody()));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(ConvertImmediateLoopExitGotosToBreak(ctx, fs->getBody()));
                return fs;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(ConvertImmediateLoopExitGotosToBreak(ctx, sw->getBody()));
                return sw;
            }
            if (auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(stmt)) {
                case_stmt->setSubStmt(
                    ConvertImmediateLoopExitGotosToBreak(ctx, case_stmt->getSubStmt())
                );
                return case_stmt;
            }
            if (auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(stmt)) {
                default_stmt->setSubStmt(
                    ConvertImmediateLoopExitGotosToBreak(ctx, default_stmt->getSubStmt())
                );
                return default_stmt;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = ConvertImmediateLoopExitGotosToBreak(ctx, child);
            }

            for (size_t i = 0; i + 1 < body.size(); ++i) {
                if (!IsLoopStmt(body[i])) { continue; }
                size_t next_idx = i + 1;
                while (next_idx < body.size() && llvm::isa< clang::NullStmt >(body[next_idx])) {
                    ++next_idx;
                }
                if (next_idx >= body.size()) { continue; }
                clang::LabelDecl *exit_label = LeadingLabelDecl(body[next_idx]);
                if (!exit_label) { continue; }

                unsigned replaced           = 0;
                clang::Stmt *rewritten_body = ReplaceGotoWithBreakInCurrentLoop(
                    ctx, GetLoopBody(body[i]), exit_label, replaced
                );
                if (replaced != 0) { SetLoopBody(body[i], rewritten_body); }
            }

            return detail::MakeCompound(ctx, body);
        }

        bool LocalLoopRegionHasUnsafeControl(
            clang::Stmt *stmt, clang::LabelDecl *backedge_target, unsigned &backedges
        ) {
            if (!stmt) { return false; }
            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                if (gs->getLabel() == backedge_target) {
                    ++backedges;
                    return false;
                }
                return true;
            }
            if (llvm::isa< clang::LabelStmt >(stmt) || llvm::isa< clang::SwitchStmt >(stmt)
                || llvm::isa< clang::WhileStmt >(stmt) || llvm::isa< clang::DoStmt >(stmt)
                || llvm::isa< clang::ForStmt >(stmt) || llvm::isa< clang::CaseStmt >(stmt)
                || llvm::isa< clang::DefaultStmt >(stmt) || llvm::isa< clang::BreakStmt >(stmt)
                || llvm::isa< clang::ContinueStmt >(stmt) || llvm::isa< clang::DeclStmt >(stmt))
            {
                return true;
            }
            for (clang::Stmt *child : stmt->children()) {
                if (LocalLoopRegionHasUnsafeControl(child, backedge_target, backedges)) {
                    return true;
                }
            }
            return false;
        }

        clang::Stmt *PromoteLocalBackwardGotoLoops(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (!stmt) { return stmt; }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(PromoteLocalBackwardGotoLoops(ctx, ifs->getThen(), refs));
                if (ifs->getElse()) {
                    ifs->setElse(PromoteLocalBackwardGotoLoops(ctx, ifs->getElse(), refs));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(PromoteLocalBackwardGotoLoops(ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(PromoteLocalBackwardGotoLoops(ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(PromoteLocalBackwardGotoLoops(ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                label->setSubStmt(
                    PromoteLocalBackwardGotoLoops(ctx, label->getSubStmt(), refs)
                );
                return label;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(PromoteLocalBackwardGotoLoops(ctx, sw->getBody(), refs));
                return sw;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = PromoteLocalBackwardGotoLoops(ctx, child, refs);
            }

            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t label_idx = 0; label_idx < body.size(); ++label_idx) {
                    clang::LabelDecl *target = LeadingLabelDecl(body[label_idx]);
                    if (!target) { continue; }
                    auto ref_it = refs.find(target);
                    if (ref_it == refs.end() || ref_it->second != 1) { continue; }
                    if (label_idx > 0 && detail::EndsWithTerminator(body[label_idx - 1])) {
                        continue;
                    }

                    LocalLabelBlock block;
                    if (!ExtractLocalLabelBlock(ctx, body, label_idx, block)) { continue; }
                    if (block.stmts.empty()) { continue; }
                    if (GotoElimGetTarget(DeepTrailingStmt(block.stmts.back()))
                        != target->getName())
                    {
                        continue;
                    }

                    unsigned backedges = 0;
                    bool unsafe        = false;
                    for (clang::Stmt *child : block.stmts) {
                        if (LocalLoopRegionHasUnsafeControl(child, target, backedges)) {
                            unsafe = true;
                            break;
                        }
                    }
                    if (unsafe || backedges != 1) { continue; }

                    clang::Stmt *loop_body = StripTrailingGoto(
                        ctx, StmtFromSeq(ctx, block.stmts), target->getName()
                    );
                    auto *while_stmt = clang::WhileStmt::Create(
                        ctx, nullptr, CreateTrueExpr(ctx), loop_body, VirtualLoc(ctx),
                        VirtualLoc(ctx), VirtualLoc(ctx)
                    );
                    body.erase(
                        body.begin() + static_cast< ptrdiff_t >(block.begin),
                        body.begin() + static_cast< ptrdiff_t >(block.end)
                    );
                    body.insert(
                        body.begin() + static_cast< ptrdiff_t >(block.begin), while_stmt
                    );
                    changed = true;
                    break;
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        bool SeqHasLocalLabelOrControl(
            const std::vector< clang::Stmt * > &stmts, bool allow_return
        ) {
            return SeqHasUnsafeStructure(
                stmts, /*allow_goto=*/false, /*allow_return=*/allow_return
            );
        }

        clang::Stmt *FoldLocalGotoDiamonds(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            const std::unordered_map< clang::LabelDecl *, unsigned > &refs
        ) {
            if (!stmt) { return stmt; }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(FoldLocalGotoDiamonds(ctx, ifs->getThen(), refs));
                if (ifs->getElse()) {
                    ifs->setElse(FoldLocalGotoDiamonds(ctx, ifs->getElse(), refs));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(FoldLocalGotoDiamonds(ctx, ws->getBody(), refs));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(FoldLocalGotoDiamonds(ctx, ds->getBody(), refs));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(FoldLocalGotoDiamonds(ctx, fs->getBody(), refs));
                return fs;
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                label->setSubStmt(FoldLocalGotoDiamonds(ctx, label->getSubStmt(), refs));
                return label;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(FoldLocalGotoDiamonds(ctx, sw->getBody(), refs));
                return sw;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > body(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : body) {
                child = FoldLocalGotoDiamonds(ctx, child, refs);
            }

            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t if_idx = 0; if_idx + 2 < body.size(); ++if_idx) {
                    auto *ifs = llvm::dyn_cast< clang::IfStmt >(body[if_idx]);
                    if (!ifs) { continue; }

                    if (ifs->getElse()) {
                        auto try_fold_if_else_goto_arm = [&](bool goto_is_then) -> bool {
                            clang::Stmt *goto_arm =
                                goto_is_then ? ifs->getThen() : ifs->getElse();
                            clang::Stmt *body_arm =
                                goto_is_then ? ifs->getElse() : ifs->getThen();
                            clang::GotoStmt *alt_goto = SingleGotoStmt(goto_arm);
                            clang::GotoStmt *join_goto =
                                llvm::dyn_cast_or_null< clang::GotoStmt >(
                                    DeepTrailingStmt(body_arm)
                                );
                            if (!alt_goto || !join_goto || !alt_goto->getLabel()
                                || !join_goto->getLabel())
                            {
                                return false;
                            }

                            clang::LabelDecl *alt_label  = alt_goto->getLabel();
                            clang::LabelDecl *join_label = join_goto->getLabel();
                            auto alt_ref                 = refs.find(alt_label);
                            if (alt_ref == refs.end() || alt_ref->second != 1) { return false; }

                            size_t alt_label_idx  = body.size();
                            size_t join_label_idx = body.size();
                            for (size_t j = if_idx + 1; j < body.size(); ++j) {
                                clang::LabelDecl *leading = LeadingLabelDecl(body[j]);
                                if (leading == alt_label && alt_label_idx == body.size()) {
                                    alt_label_idx = j;
                                }
                                if (leading == join_label) {
                                    join_label_idx = j;
                                    break;
                                }
                            }
                            if (alt_label_idx >= body.size() || join_label_idx >= body.size()
                                || alt_label_idx >= join_label_idx)
                            {
                                return false;
                            }

                            LocalLabelBlock alt_block;
                            if (!ExtractLocalLabelBlock(ctx, body, alt_label_idx, alt_block)) {
                                return false;
                            }
                            if (alt_block.end > join_label_idx) { return false; }

                            std::vector< clang::Stmt * > alt_region = alt_block.stmts;
                            if (alt_block.end < join_label_idx) {
                                alt_region.insert(
                                    alt_region.end(),
                                    body.begin() + static_cast< ptrdiff_t >(alt_block.end),
                                    body.begin() + static_cast< ptrdiff_t >(join_label_idx)
                                );
                            }
                            if (SeqHasLocalLabelOrControl(alt_region, /*allow_return=*/true)) {
                                return false;
                            }

                            clang::Stmt *stripped_body =
                                StripTrailingGoto(ctx, body_arm, join_label->getName());
                            std::vector< clang::Stmt * > body_region;
                            AppendStmtSequence(stripped_body, body_region);
                            if (SeqHasLocalLabelOrControl(body_region, /*allow_return=*/true)) {
                                return false;
                            }

                            auto loc              = ifs->getIfLoc();
                            clang::Stmt *new_then = goto_is_then
                                ? StmtFromSeq(ctx, alt_region)
                                : StmtFromSeq(ctx, body_region);
                            clang::Stmt *new_else = goto_is_then ? StmtFromSeq(ctx, body_region)
                                                                 : StmtFromSeq(ctx, alt_region);
                            auto *new_if          = clang::IfStmt::Create(
                                ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                                ifs->getCond(), loc, loc, new_then, loc, new_else
                            );

                            body.erase(
                                body.begin() + static_cast< ptrdiff_t >(if_idx),
                                body.begin() + static_cast< ptrdiff_t >(join_label_idx)
                            );
                            body.insert(
                                body.begin() + static_cast< ptrdiff_t >(if_idx), new_if
                            );
                            return true;
                        };

                        if (try_fold_if_else_goto_arm(/*goto_is_then=*/true)
                            || try_fold_if_else_goto_arm(/*goto_is_then=*/false))
                        {
                            changed = true;
                            break;
                        }
                        continue;
                    }

                    clang::GotoStmt *then_goto = SingleGotoStmt(ifs->getThen());
                    if (!then_goto || !then_goto->getLabel()) { continue; }
                    clang::LabelDecl *then_label = then_goto->getLabel();
                    auto then_ref                = refs.find(then_label);
                    if (then_ref == refs.end() || then_ref->second != 1) { continue; }

                    size_t then_label_idx = body.size();
                    for (size_t j = if_idx + 1; j < body.size(); ++j) {
                        if (LeadingLabelDecl(body[j]) == then_label) {
                            then_label_idx = j;
                            break;
                        }
                    }
                    if (then_label_idx >= body.size() || then_label_idx == if_idx + 1) {
                        continue;
                    }

                    std::vector< clang::Stmt * > false_region(
                        body.begin() + static_cast< ptrdiff_t >(if_idx + 1),
                        body.begin() + static_cast< ptrdiff_t >(then_label_idx)
                    );
                    clang::GotoStmt *join_goto = llvm::dyn_cast_or_null< clang::GotoStmt >(
                        DeepTrailingStmt(StmtFromSeq(ctx, false_region))
                    );
                    if (!join_goto || !join_goto->getLabel()) { continue; }
                    clang::LabelDecl *join_label = join_goto->getLabel();
                    size_t join_label_idx        = body.size();
                    for (size_t j = then_label_idx + 1; j < body.size(); ++j) {
                        if (LeadingLabelDecl(body[j]) == join_label) {
                            join_label_idx = j;
                            break;
                        }
                    }
                    if (join_label_idx >= body.size()) { continue; }

                    LocalLabelBlock then_block;
                    if (!ExtractLocalLabelBlock(ctx, body, then_label_idx, then_block)) {
                        continue;
                    }
                    if (then_block.end > join_label_idx) { continue; }

                    std::vector< clang::Stmt * > true_region = then_block.stmts;
                    if (then_block.end < join_label_idx) {
                        true_region.insert(
                            true_region.end(),
                            body.begin() + static_cast< ptrdiff_t >(then_block.end),
                            body.begin() + static_cast< ptrdiff_t >(join_label_idx)
                        );
                    }

                    clang::Stmt *false_stmt = StripTrailingGoto(
                        ctx, StmtFromSeq(ctx, false_region), join_label->getName()
                    );
                    false_region.clear();
                    AppendStmtSequence(false_stmt, false_region);
                    if (SeqHasLocalLabelOrControl(false_region, /*allow_return=*/true)
                        || SeqHasLocalLabelOrControl(true_region, /*allow_return=*/true))
                    {
                        continue;
                    }

                    auto loc     = ifs->getIfLoc();
                    auto *new_if = clang::IfStmt::Create(
                        ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                        ifs->getCond(), loc, loc, StmtFromSeq(ctx, true_region), loc,
                        StmtFromSeq(ctx, false_region)
                    );

                    body.erase(
                        body.begin() + static_cast< ptrdiff_t >(if_idx),
                        body.begin() + static_cast< ptrdiff_t >(join_label_idx)
                    );
                    body.insert(body.begin() + static_cast< ptrdiff_t >(if_idx), new_if);
                    changed = true;
                    break;
                }
            }

            return detail::MakeCompound(ctx, body);
        }

        // Collect all LabelDecls that have a LabelStmt definition in the tree.
        void CollectDefinedLabels(
            clang::Stmt *s, std::unordered_set< clang::LabelDecl * > &defined
        ) {
            llvm::SmallVector< clang::Stmt *, 16 > worklist;
            if (s) { worklist.push_back(s); }

            while (!worklist.empty()) {
                auto *cur = worklist.pop_back_val();
                if (!cur) { continue; }
                if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(cur)) {
                    defined.insert(ls->getDecl());
                }
                for (auto *child : cur->children()) {
                    if (child) { worklist.push_back(child); }
                }
            }
        }

        // Replace GotoStmts whose target label has no LabelStmt in the
        // function body with NullStmt.  When the orphaned goto is inside
        // a switch case body (in_switch_case=true), replace with BreakStmt
        // instead to prevent unintended fallthrough.
        clang::Stmt *RemoveOrphanedGotos(
            clang::ASTContext &ctx, clang::Stmt *s,
            const std::unordered_set< clang::LabelDecl * > &defined, unsigned depth = 0,
            bool in_switch_case = false
        ) {
            if (!s) { return nullptr; }
            if (depth > 256) {
                LOG(ERROR) << "RemoveOrphanedGotos: recursion depth exceeded "
                              "(depth="
                           << depth
                           << "). Possible malformed AST "
                              "or unexpectedly deep nesting — skipping subtree.\n";
                return nullptr;
            }

            if (auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
                if (!defined.count(gs->getLabel())) {
                    LOG(ERROR) << "ORPHANED GOTO: removing 'goto " << gs->getLabel()->getName()
                               << "' with no matching LabelStmt in function body. "
                                  "This may indicate a structuring rule bug that "
                                  "dropped the target label — verify emitted output.\n";
                    if (in_switch_case) { return new (ctx) clang::BreakStmt(gs->getGotoLoc()); }
                    return new (ctx) clang::NullStmt(gs->getGotoLoc());
                }
                return nullptr;
            }

            // Track whether children are inside a switch case body.
            bool child_in_case = in_switch_case || llvm::isa< clang::CaseStmt >(s)
                || llvm::isa< clang::DefaultStmt >(s);

            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(s)) {
                std::vector< clang::Stmt * > children;
                bool changed = false;
                for (auto *child : cs->body()) {
                    auto *repl =
                        RemoveOrphanedGotos(ctx, child, defined, depth + 1, child_in_case);
                    children.push_back(repl ? repl : child);
                    if (repl) { changed = true; }
                }
                return changed ? detail::MakeCompound(ctx, children) : nullptr;
            }

            // Recurse into IfStmt, LabelStmt, etc. via child iteration.
            bool changed = false;
            for (auto it = s->child_begin(); it != s->child_end(); ++it) {
                if (!*it) { continue; }
                auto *repl = RemoveOrphanedGotos(ctx, *it, defined, depth + 1, child_in_case);
                if (repl) {
                    *it     = repl;
                    changed = true;
                }
            }
            return changed ? s : nullptr;
        }

        clang::DeclRefExpr *AsDeclRef(clang::Expr *expr) {
            if (!expr) { return nullptr; }
            return llvm::dyn_cast< clang::DeclRefExpr >(expr->IgnoreParenImpCasts());
        }

        clang::VarDecl *AsVarRef(clang::Expr *expr) {
            auto *decl_ref = AsDeclRef(expr);
            if (!decl_ref) { return nullptr; }
            return llvm::dyn_cast< clang::VarDecl >(decl_ref->getDecl());
        }

        clang::Expr *AsExprStmt(clang::Stmt *stmt) {
            return llvm::dyn_cast_or_null< clang::Expr >(stmt);
        }

        clang::VarDecl *AssignmentLHSVar(clang::Stmt *stmt) {
            auto *expr = AsExprStmt(stmt);
            auto *bo   = llvm::dyn_cast_or_null< clang::BinaryOperator >(expr);
            if (!bo || bo->getOpcode() != clang::BO_Assign) { return nullptr; }
            return AsVarRef(bo->getLHS());
        }

        bool IsSimpleCounterStep(clang::Expr *expr, const clang::VarDecl *var) {
            if (!expr || !var) { return false; }

            if (auto *unary = llvm::dyn_cast< clang::UnaryOperator >(expr)) {
                if (unary->isIncrementDecrementOp()) {
                    return AsVarRef(unary->getSubExpr()) == var;
                }
            }

            auto *assign = llvm::dyn_cast< clang::BinaryOperator >(expr);
            if (!assign || assign->getOpcode() != clang::BO_Assign) { return false; }
            if (AsVarRef(assign->getLHS()) != var) { return false; }

            auto *rhs = llvm::dyn_cast_or_null< clang::BinaryOperator >(
                assign->getRHS()->IgnoreParenImpCasts()
            );
            if (!rhs) { return false; }

            if (rhs->getOpcode() != clang::BO_Add && rhs->getOpcode() != clang::BO_Sub) {
                return false;
            }
            if (AsVarRef(rhs->getLHS()) == var) { return true; }
            return rhs->getOpcode() == clang::BO_Add && AsVarRef(rhs->getRHS()) == var;
        }

        bool IsSimpleCounterCondition(clang::Expr *cond, const clang::VarDecl *var) {
            if (!cond || !var) { return false; }
            auto *bo = llvm::dyn_cast< clang::BinaryOperator >(cond->IgnoreParenImpCasts());
            if (!bo || !(bo->isRelationalOp() || bo->isEqualityOp())) { return false; }
            return AsVarRef(bo->getLHS()) == var || AsVarRef(bo->getRHS()) == var;
        }

        bool StmtWritesVar(clang::Stmt *stmt, const clang::VarDecl *var) {
            if (!stmt || !var) { return false; }
            if (auto *unary = llvm::dyn_cast< clang::UnaryOperator >(stmt)) {
                if (unary->isIncrementDecrementOp() && AsVarRef(unary->getSubExpr()) == var) {
                    return true;
                }
            }
            if (auto *bo = llvm::dyn_cast< clang::BinaryOperator >(stmt)) {
                if (bo->isAssignmentOp() && AsVarRef(bo->getLHS()) == var) { return true; }
            }
            for (clang::Stmt *child : stmt->children()) {
                if (StmtWritesVar(child, var)) { return true; }
            }
            return false;
        }

        bool HasUnsafeForPromotionControl(clang::Stmt *stmt, clang::ContinueStmt *allowed) {
            if (!stmt) { return false; }
            if (auto *cont = llvm::dyn_cast< clang::ContinueStmt >(stmt)) {
                return cont != allowed;
            }
            if (llvm::isa< clang::LabelStmt >(stmt) || llvm::isa< clang::GotoStmt >(stmt)
                || llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::CaseStmt >(stmt)
                || llvm::isa< clang::DefaultStmt >(stmt) || llvm::isa< clang::WhileStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt)
                || llvm::isa< clang::DeclStmt >(stmt))
            {
                return true;
            }
            for (clang::Stmt *child : stmt->children()) {
                if (HasUnsafeForPromotionControl(child, allowed)) { return true; }
            }
            return false;
        }

        bool IsNestedLoopOrSwitch(clang::Stmt *stmt) {
            return llvm::isa< clang::SwitchStmt >(stmt) || llvm::isa< clang::WhileStmt >(stmt)
                || llvm::isa< clang::DoStmt >(stmt) || llvm::isa< clang::ForStmt >(stmt);
        }

        struct LatchRewritePlan
        {
            std::string inc_text;
            clang::Expr *inc            = nullptr;
            size_t continues            = 0;
            size_t latch_writes         = 0;
            size_t other_counter_writes = 0;
        };

        bool AnalyzeSharedLatchInStmt(
            clang::ASTContext &ctx, clang::Stmt *stmt, const clang::VarDecl *counter,
            LatchRewritePlan &plan, bool nested_root = false
        );

        bool AnalyzeSharedLatchChildren(
            clang::ASTContext &ctx, clang::CompoundStmt *compound,
            const clang::VarDecl *counter, LatchRewritePlan &plan
        ) {
            if (!compound) { return false; }
            std::vector< clang::Stmt * > children(compound->body_begin(), compound->body_end());
            for (size_t i = 0; i < children.size(); ++i) {
                if (llvm::isa< clang::ContinueStmt >(children[i])) {
                    ++plan.continues;
                    if (i == 0) { return false; }
                    clang::Expr *inc = AsExprStmt(children[i - 1]);
                    if (!IsSimpleCounterStep(inc, counter)) { return false; }

                    std::string inc_text = StmtToStableString(ctx, inc);
                    if (plan.inc_text.empty()) {
                        plan.inc_text = inc_text;
                        plan.inc      = inc;
                    } else if (plan.inc_text != inc_text) {
                        return false;
                    }
                    ++plan.latch_writes;
                    continue;
                }

                if (auto *nested = llvm::dyn_cast< clang::CompoundStmt >(children[i])) {
                    if (!AnalyzeSharedLatchChildren(ctx, nested, counter, plan)) {
                        return false;
                    }
                    continue;
                }
                if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(children[i])) {
                    if (!AnalyzeSharedLatchInStmt(ctx, ifs->getThen(), counter, plan, true)
                        || !AnalyzeSharedLatchInStmt(ctx, ifs->getElse(), counter, plan, true))
                    {
                        return false;
                    }
                    continue;
                }
                if (auto *label = llvm::dyn_cast< clang::LabelStmt >(children[i])) {
                    if (!AnalyzeSharedLatchInStmt(
                            ctx, label->getSubStmt(), counter, plan, true
                        ))
                    {
                        return false;
                    }
                    continue;
                }
                if (IsNestedLoopOrSwitch(children[i])) { return false; }

                if (StmtWritesVar(children[i], counter)) {
                    if (i + 1 < children.size()
                        && llvm::isa< clang::ContinueStmt >(children[i + 1])
                        && IsSimpleCounterStep(AsExprStmt(children[i]), counter))
                    {
                        continue;
                    }
                    ++plan.other_counter_writes;
                    return false;
                }

                if (!AnalyzeSharedLatchInStmt(ctx, children[i], counter, plan, true)) {
                    return false;
                }
            }
            return true;
        }

        bool AnalyzeSharedLatchInStmt(
            clang::ASTContext &ctx, clang::Stmt *stmt, const clang::VarDecl *counter,
            LatchRewritePlan &plan, bool nested_root
        ) {
            if (!stmt) { return true; }
            if (nested_root && IsNestedLoopOrSwitch(stmt)) { return false; }
            if (llvm::isa< clang::ContinueStmt >(stmt)) { return false; }

            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                return AnalyzeSharedLatchChildren(ctx, compound, counter, plan);
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                return AnalyzeSharedLatchInStmt(ctx, ifs->getThen(), counter, plan, true)
                    && AnalyzeSharedLatchInStmt(ctx, ifs->getElse(), counter, plan, true);
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                return AnalyzeSharedLatchInStmt(ctx, label->getSubStmt(), counter, plan, true);
            }

            return true;
        }

        clang::Stmt *RemoveSharedLatchIncrements(
            clang::ASTContext &ctx, clang::Stmt *stmt, const clang::VarDecl *counter,
            const std::string &inc_text
        );

        clang::Stmt *RemoveSharedLatchIncrementsFromCompound(
            clang::ASTContext &ctx, clang::CompoundStmt *compound,
            const clang::VarDecl *counter, const std::string &inc_text
        ) {
            std::vector< clang::Stmt * > children(compound->body_begin(), compound->body_end());
            std::vector< clang::Stmt * > rewritten;
            rewritten.reserve(children.size());
            for (size_t i = 0; i < children.size(); ++i) {
                if (i + 1 < children.size() && llvm::isa< clang::ContinueStmt >(children[i + 1])
                    && IsSimpleCounterStep(AsExprStmt(children[i]), counter)
                    && StmtToStableString(ctx, children[i]) == inc_text)
                {
                    continue;
                }
                rewritten.push_back(
                    RemoveSharedLatchIncrements(ctx, children[i], counter, inc_text)
                );
            }
            return detail::MakeCompound(ctx, rewritten);
        }

        clang::Stmt *RemoveSharedLatchIncrements(
            clang::ASTContext &ctx, clang::Stmt *stmt, const clang::VarDecl *counter,
            const std::string &inc_text
        ) {
            if (!stmt) { return stmt; }
            if (IsNestedLoopOrSwitch(stmt)) { return stmt; }
            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                return RemoveSharedLatchIncrementsFromCompound(
                    ctx, compound, counter, inc_text
                );
            }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(
                    RemoveSharedLatchIncrements(ctx, ifs->getThen(), counter, inc_text)
                );
                if (ifs->getElse()) {
                    ifs->setElse(
                        RemoveSharedLatchIncrements(ctx, ifs->getElse(), counter, inc_text)
                    );
                }
                return ifs;
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                label->setSubStmt(
                    RemoveSharedLatchIncrements(ctx, label->getSubStmt(), counter, inc_text)
                );
                return label;
            }
            return stmt;
        }

        bool TryPromoteWhileAt(
            clang::ASTContext &ctx, std::vector< clang::Stmt * > &body, size_t while_idx
        ) {
            if (while_idx == 0 || while_idx >= body.size()) { return false; }

            auto *while_stmt = llvm::dyn_cast< clang::WhileStmt >(body[while_idx]);
            if (!while_stmt || !while_stmt->getCond() || !while_stmt->getBody()) {
                return false;
            }

            clang::VarDecl *counter = AssignmentLHSVar(body[while_idx - 1]);
            if (!counter || !IsSimpleCounterCondition(while_stmt->getCond(), counter)) {
                return false;
            }

            std::vector< clang::Stmt * > loop_body;
            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(while_stmt->getBody())) {
                loop_body.assign(compound->body_begin(), compound->body_end());
            } else {
                loop_body.push_back(while_stmt->getBody());
            }
            if (loop_body.empty()) { return false; }

            clang::ContinueStmt *terminal_continue = nullptr;
            size_t update_idx                      = loop_body.size() - 1;
            if (auto *cont = llvm::dyn_cast< clang::ContinueStmt >(loop_body.back())) {
                terminal_continue = cont;
                if (loop_body.size() < 2) { return false; }
                update_idx = loop_body.size() - 2;
            }

            clang::Expr *inc = AsExprStmt(loop_body[update_idx]);
            if (!IsSimpleCounterStep(inc, counter)) { return false; }

            for (size_t i = 0; i < loop_body.size(); ++i) {
                if (i == update_idx) { continue; }
                if (HasUnsafeForPromotionControl(loop_body[i], terminal_continue)) {
                    LatchRewritePlan plan;
                    if (!AnalyzeSharedLatchInStmt(
                            ctx, while_stmt->getBody(), counter, plan, false
                        )
                        || !plan.inc || plan.continues == 0
                        || plan.continues != plan.latch_writes
                        || plan.other_counter_writes != 0)
                    {
                        return false;
                    }

                    clang::Stmt *rewritten_body = RemoveSharedLatchIncrements(
                        ctx, while_stmt->getBody(), counter, plan.inc_text
                    );
                    auto *for_stmt = new (ctx) clang::ForStmt(
                        ctx, body[while_idx - 1], while_stmt->getCond(), nullptr, plan.inc,
                        rewritten_body, VirtualLoc(ctx), VirtualLoc(ctx), VirtualLoc(ctx)
                    );
                    body.erase(
                        body.begin() + static_cast< ptrdiff_t >(while_idx - 1),
                        body.begin() + static_cast< ptrdiff_t >(while_idx + 1)
                    );
                    body.insert(
                        body.begin() + static_cast< ptrdiff_t >(while_idx - 1), for_stmt
                    );
                    return true;
                }
                if (StmtWritesVar(loop_body[i], counter)) { return false; }
            }

            std::vector< clang::Stmt * > promoted_body;
            promoted_body.reserve(loop_body.size() - 1);
            for (size_t i = 0; i < loop_body.size(); ++i) {
                if (i == update_idx) { continue; }
                promoted_body.push_back(loop_body[i]);
            }

            auto *for_body = promoted_body.empty()
                ? static_cast< clang::Stmt * >(new (ctx) clang::NullStmt(VirtualLoc(ctx)))
                : static_cast< clang::Stmt * >(detail::MakeCompound(ctx, promoted_body));

            auto *for_stmt = new (ctx) clang::ForStmt(
                ctx, body[while_idx - 1], while_stmt->getCond(), nullptr, inc, for_body,
                VirtualLoc(ctx), VirtualLoc(ctx), VirtualLoc(ctx)
            );

            body.erase(
                body.begin() + static_cast< ptrdiff_t >(while_idx - 1),
                body.begin() + static_cast< ptrdiff_t >(while_idx + 1)
            );
            body.insert(body.begin() + static_cast< ptrdiff_t >(while_idx - 1), for_stmt);
            return true;
        }

        // Phase 2a: `mutated` is set true only when a real while→for
        // promotion fires.  The recursive walk still rebuilds CompoundStmts,
        // but the caller relies on `mutated` (not the non-null return) to
        // decide whether anything actually changed.
        clang::Stmt *PromoteSimpleCounterWhileToFor(
            clang::ASTContext &ctx, clang::Stmt *stmt, bool &mutated
        ) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(PromoteSimpleCounterWhileToFor(ctx, ifs->getThen(), mutated));
                if (ifs->getElse()) {
                    ifs->setElse(PromoteSimpleCounterWhileToFor(ctx, ifs->getElse(), mutated));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(PromoteSimpleCounterWhileToFor(ctx, ws->getBody(), mutated));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(PromoteSimpleCounterWhileToFor(ctx, ds->getBody(), mutated));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(PromoteSimpleCounterWhileToFor(ctx, fs->getBody(), mutated));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(PromoteSimpleCounterWhileToFor(ctx, ls->getSubStmt(), mutated));
                return ls;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(PromoteSimpleCounterWhileToFor(ctx, sw->getBody(), mutated));
                return sw;
            }
            if (auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(stmt)) {
                case_stmt->setSubStmt(
                    PromoteSimpleCounterWhileToFor(ctx, case_stmt->getSubStmt(), mutated)
                );
                return case_stmt;
            }
            if (auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(stmt)) {
                default_stmt->setSubStmt(
                    PromoteSimpleCounterWhileToFor(ctx, default_stmt->getSubStmt(), mutated)
                );
                return default_stmt;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > children(compound->body_begin(), compound->body_end());
            for (clang::Stmt *&child : children) {
                child = PromoteSimpleCounterWhileToFor(ctx, child, mutated);
            }

            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t i = 1; i < children.size(); ++i) {
                    if (TryPromoteWhileAt(ctx, children, i)) {
                        changed = true;
                        mutated = true;
                        break;
                    }
                }
            }

            return detail::MakeCompound(ctx, children);
        }

        // Phase 2a: `mutated` is set true only when a trailing `continue`
        // is actually dropped from a for-loop body.
        clang::Stmt *RemoveRedundantTerminalForContinues(
            clang::ASTContext &ctx, clang::Stmt *stmt, bool &mutated
        ) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(RemoveRedundantTerminalForContinues(ctx, ifs->getThen(), mutated));
                if (ifs->getElse()) {
                    ifs->setElse(
                        RemoveRedundantTerminalForContinues(ctx, ifs->getElse(), mutated)
                    );
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(RemoveRedundantTerminalForContinues(ctx, ws->getBody(), mutated));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(RemoveRedundantTerminalForContinues(ctx, ds->getBody(), mutated));
                return ds;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(
                    RemoveRedundantTerminalForContinues(ctx, ls->getSubStmt(), mutated)
                );
                return ls;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(RemoveRedundantTerminalForContinues(ctx, sw->getBody(), mutated));
                return sw;
            }
            if (auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(stmt)) {
                case_stmt->setSubStmt(
                    RemoveRedundantTerminalForContinues(ctx, case_stmt->getSubStmt(), mutated)
                );
                return case_stmt;
            }
            if (auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(stmt)) {
                default_stmt->setSubStmt(RemoveRedundantTerminalForContinues(
                    ctx, default_stmt->getSubStmt(), mutated
                ));
                return default_stmt;
            }
            if (auto *for_stmt = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                clang::Stmt *for_body =
                    RemoveRedundantTerminalForContinues(ctx, for_stmt->getBody(), mutated);
                if (auto *compound = llvm::dyn_cast_or_null< clang::CompoundStmt >(for_body)) {
                    std::vector< clang::Stmt * > children(
                        compound->body_begin(), compound->body_end()
                    );
                    if (!children.empty() && llvm::isa< clang::ContinueStmt >(children.back()))
                    {
                        children.pop_back();
                        mutated  = true;
                        for_body = children.empty()
                            ? static_cast< clang::Stmt * >(new (ctx)
                                                               clang::NullStmt(VirtualLoc(ctx)))
                            : static_cast< clang::Stmt * >(detail::MakeCompound(ctx, children));
                    }
                } else if (llvm::isa_and_nonnull< clang::ContinueStmt >(for_body)) {
                    for_body = new (ctx) clang::NullStmt(VirtualLoc(ctx));
                    mutated  = true;
                }
                for_stmt->setBody(for_body);
                return for_stmt;
            }
            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > children;
                children.reserve(compound->size());
                for (clang::Stmt *child : compound->body()) {
                    children.push_back(
                        RemoveRedundantTerminalForContinues(ctx, child, mutated)
                    );
                }
                return detail::MakeCompound(ctx, children);
            }
            return stmt;
        }

        // Phase 2a: `mutated` is set true when a label/case/default body
        // compound is unwrapped, or a nested compound is spliced flat.
        clang::Stmt *PushLabelsIntoCompounds(
            clang::ASTContext &ctx, clang::Stmt *stmt, bool &mutated
        ) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(PushLabelsIntoCompounds(ctx, ifs->getThen(), mutated));
                if (ifs->getElse()) {
                    ifs->setElse(PushLabelsIntoCompounds(ctx, ifs->getElse(), mutated));
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(PushLabelsIntoCompounds(ctx, ws->getBody(), mutated));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(PushLabelsIntoCompounds(ctx, ds->getBody(), mutated));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(PushLabelsIntoCompounds(ctx, fs->getBody(), mutated));
                return fs;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(PushLabelsIntoCompounds(ctx, sw->getBody(), mutated));
                return sw;
            }
            if (auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(stmt)) {
                auto *sub      = PushLabelsIntoCompounds(ctx, case_stmt->getSubStmt(), mutated);
                auto *compound = llvm::dyn_cast_or_null< clang::CompoundStmt >(sub);
                if (!compound) {
                    case_stmt->setSubStmt(
                        sub ? sub : new (ctx) clang::NullStmt(VirtualLoc(ctx))
                    );
                    return case_stmt;
                }
                mutated = true;

                std::vector< clang::Stmt * > children(
                    compound->body_begin(), compound->body_end()
                );
                if (children.empty()) {
                    case_stmt->setSubStmt(new (ctx) clang::NullStmt(VirtualLoc(ctx)));
                    return case_stmt;
                }

                case_stmt->setSubStmt(children.front());
                if (children.size() == 1) { return case_stmt; }

                std::vector< clang::Stmt * > flattened;
                flattened.reserve(children.size());
                flattened.push_back(case_stmt);
                flattened.insert(flattened.end(), std::next(children.begin()), children.end());
                return detail::MakeCompound(ctx, flattened);
            }
            if (auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(stmt)) {
                auto *sub =
                    PushLabelsIntoCompounds(ctx, default_stmt->getSubStmt(), mutated);
                auto *compound = llvm::dyn_cast_or_null< clang::CompoundStmt >(sub);
                if (!compound) {
                    default_stmt->setSubStmt(
                        sub ? sub : new (ctx) clang::NullStmt(VirtualLoc(ctx))
                    );
                    return default_stmt;
                }
                mutated = true;

                std::vector< clang::Stmt * > children(
                    compound->body_begin(), compound->body_end()
                );
                if (children.empty()) {
                    default_stmt->setSubStmt(new (ctx) clang::NullStmt(VirtualLoc(ctx)));
                    return default_stmt;
                }

                default_stmt->setSubStmt(children.front());
                if (children.size() == 1) { return default_stmt; }

                std::vector< clang::Stmt * > flattened;
                flattened.reserve(children.size());
                flattened.push_back(default_stmt);
                flattened.insert(flattened.end(), std::next(children.begin()), children.end());
                return detail::MakeCompound(ctx, flattened);
            }
            if (auto *label = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                auto *sub      = PushLabelsIntoCompounds(ctx, label->getSubStmt(), mutated);
                auto *compound = llvm::dyn_cast_or_null< clang::CompoundStmt >(sub);
                if (!compound) {
                    label->setSubStmt(sub ? sub : new (ctx) clang::NullStmt(VirtualLoc(ctx)));
                    return label;
                }
                mutated = true;

                std::vector< clang::Stmt * > children(
                    compound->body_begin(), compound->body_end()
                );
                if (children.empty()) {
                    label->setSubStmt(new (ctx) clang::NullStmt(VirtualLoc(ctx)));
                    return label;
                }

                label->setSubStmt(children.front());
                if (children.size() == 1) { return label; }

                std::vector< clang::Stmt * > flattened;
                flattened.reserve(children.size());
                flattened.push_back(label);
                flattened.insert(flattened.end(), std::next(children.begin()), children.end());
                return detail::MakeCompound(ctx, flattened);
            }
            if (auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > children;
                children.reserve(compound->size());
                for (clang::Stmt *child : compound->body()) {
                    auto *cleaned = PushLabelsIntoCompounds(ctx, child, mutated);
                    if (auto *nested = llvm::dyn_cast_or_null< clang::CompoundStmt >(cleaned)) {
                        for (clang::Stmt *nested_child : nested->body()) {
                            children.push_back(nested_child);
                        }
                        mutated = true;
                        continue;
                    }
                    children.push_back(cleaned);
                }
                return detail::MakeCompound(ctx, children);
            }
            return stmt;
        }

        // Phase 2a: `mutated` is set true only when an empty label is
        // actually attached to its following statement.
        clang::Stmt *AttachEmptyLabelsToFollowingStmt(
            clang::ASTContext &ctx, clang::Stmt *stmt, bool &mutated
        ) {
            if (!stmt) { return stmt; }

            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                ifs->setThen(AttachEmptyLabelsToFollowingStmt(ctx, ifs->getThen(), mutated));
                if (ifs->getElse()) {
                    ifs->setElse(
                        AttachEmptyLabelsToFollowingStmt(ctx, ifs->getElse(), mutated)
                    );
                }
                return ifs;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(stmt)) {
                ws->setBody(AttachEmptyLabelsToFollowingStmt(ctx, ws->getBody(), mutated));
                return ws;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(stmt)) {
                ds->setBody(AttachEmptyLabelsToFollowingStmt(ctx, ds->getBody(), mutated));
                return ds;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(stmt)) {
                fs->setBody(AttachEmptyLabelsToFollowingStmt(ctx, fs->getBody(), mutated));
                return fs;
            }
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                ls->setSubStmt(
                    AttachEmptyLabelsToFollowingStmt(ctx, ls->getSubStmt(), mutated)
                );
                return ls;
            }
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                sw->setBody(AttachEmptyLabelsToFollowingStmt(ctx, sw->getBody(), mutated));
                return sw;
            }
            if (auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(stmt)) {
                case_stmt->setSubStmt(
                    AttachEmptyLabelsToFollowingStmt(ctx, case_stmt->getSubStmt(), mutated)
                );
                return case_stmt;
            }
            if (auto *default_stmt = llvm::dyn_cast< clang::DefaultStmt >(stmt)) {
                default_stmt->setSubStmt(AttachEmptyLabelsToFollowingStmt(
                    ctx, default_stmt->getSubStmt(), mutated
                ));
                return default_stmt;
            }

            auto *compound = llvm::dyn_cast< clang::CompoundStmt >(stmt);
            if (!compound) { return stmt; }

            std::vector< clang::Stmt * > children;
            children.reserve(compound->size());
            for (clang::Stmt *child : compound->body()) {
                children.push_back(AttachEmptyLabelsToFollowingStmt(ctx, child, mutated));
            }

            for (size_t i = 0; i + 1 < children.size(); ++i) {
                auto *label = llvm::dyn_cast_or_null< clang::LabelStmt >(children[i]);
                if (!label || !llvm::isa_and_nonnull< clang::NullStmt >(label->getSubStmt())) {
                    continue;
                }

                clang::Stmt *next = children[i + 1];
                if (!next || llvm::isa< clang::DeclStmt >(next)
                    || llvm::isa< clang::CaseStmt >(next)
                    || llvm::isa< clang::DefaultStmt >(next))
                {
                    continue;
                }

                label->setSubStmt(next);
                children.erase(children.begin() + static_cast< ptrdiff_t >(i + 1));
                mutated = true;
            }

            return detail::MakeCompound(ctx, children);
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
        std::string NormalizeExprKey(clang::ASTContext &ctx, clang::Expr *expr) {
            std::string text = StmtToStableString(ctx, expr);
            text.erase(
                std::remove_if(
                    text.begin(), text.end(),
                    [](unsigned char ch) { return std::isspace(ch) != 0; }
                ),
                text.end()
            );
            while (text.size() >= 2 && text.front() == '(' && text.back() == ')') {
                text = text.substr(1, text.size() - 2);
            }
            return text;
        }

        std::string NormalizeIntegerText(std::string text) {
            while (!text.empty() && text.front() == '(' && text.back() == ')') {
                text = text.substr(1, text.size() - 2);
            }
            while (!text.empty()) {
                char ch = text.back();
                if (ch != 'u' && ch != 'U' && ch != 'l' && ch != 'L') { break; }
                text.pop_back();
            }
            return text;
        }

        bool IsZeroIntegerText(const std::string &text) {
            return NormalizeIntegerText(text) == "0";
        }

        bool IsNonzeroIntegerText(const std::string &text) {
            std::string normalized = NormalizeIntegerText(text);
            if (normalized.empty() || normalized == "0") { return false; }
            size_t begin = normalized.front() == '-' ? 1 : 0;
            if (begin == normalized.size()) { return false; }
            return std::all_of(
                normalized.begin() + static_cast< ptrdiff_t >(begin), normalized.end(),
                [](char ch) { return std::isdigit(static_cast< unsigned char >(ch)) != 0; }
            );
        }

        struct SimpleEqualityCompare
        {
            std::string lhs;
            std::string rhs;
            clang::BinaryOperatorKind op = clang::BO_Comma;
            clang::Expr *expr            = nullptr;
        };

        bool ExtractSimpleEqualityCompare(
            clang::ASTContext &ctx, clang::Expr *expr, SimpleEqualityCompare &out
        ) {
            auto *bo = llvm::dyn_cast_or_null< clang::BinaryOperator >(expr->IgnoreParens());
            if (!bo || !bo->isEqualityOp()) { return false; }

            out.lhs  = NormalizeExprKey(ctx, bo->getLHS());
            out.rhs  = NormalizeExprKey(ctx, bo->getRHS());
            out.op   = bo->getOpcode();
            out.expr = expr;

            if (IsNonzeroIntegerText(out.lhs) || IsZeroIntegerText(out.lhs)) {
                std::swap(out.lhs, out.rhs);
            }
            return true;
        }

        clang::Expr *SimplifyRedundantNonzeroGuard(
            clang::ASTContext &ctx, clang::Expr *lhs, clang::Expr *rhs
        ) {
            SimpleEqualityCompare left;
            SimpleEqualityCompare right;
            if (!ExtractSimpleEqualityCompare(ctx, lhs, left)
                || !ExtractSimpleEqualityCompare(ctx, rhs, right) || left.lhs != right.lhs)
            {
                return nullptr;
            }

            auto is_eq_nonzero = [](const SimpleEqualityCompare &cmp) {
                return cmp.op == clang::BO_EQ && IsNonzeroIntegerText(cmp.rhs);
            };
            auto is_ne_zero = [](const SimpleEqualityCompare &cmp) {
                return cmp.op == clang::BO_NE && IsZeroIntegerText(cmp.rhs);
            };

            if (is_eq_nonzero(left) && is_ne_zero(right)) { return left.expr; }
            if (is_ne_zero(left) && is_eq_nonzero(right)) { return right.expr; }
            return nullptr;
        }

        clang::Expr *NormalizeBoolExpr(clang::ASTContext &ctx, clang::Expr *e) {
            if (!e) { return e; }

            if (auto *pe = llvm::dyn_cast< clang::ParenExpr >(e)) {
                auto *inner = NormalizeBoolExpr(ctx, pe->getSubExpr());
                if (inner == pe->getSubExpr()) { return pe; }
                return new (ctx) clang::ParenExpr(pe->getLParen(), pe->getRParen(), inner);
            }

            if (auto *bo = llvm::dyn_cast< clang::BinaryOperator >(e)) {
                if (bo->getOpcode() == clang::BO_And) {
                    auto build_bitmask_compare = [&](clang::Expr *value,
                                                     clang::Expr *maybe_cmp) -> clang::Expr * {
                        auto *cmp = llvm::dyn_cast_or_null< clang::BinaryOperator >(
                            maybe_cmp ? maybe_cmp->IgnoreParens() : nullptr
                        );
                        if (!cmp || !cmp->isEqualityOp()) { return nullptr; }
                        if (!value || value->getType()->isBooleanType()
                            || !value->getType()->isIntegerType())
                        {
                            return nullptr;
                        }

                        clang::Expr *mask = cmp->getLHS();
                        clang::Expr *rhs  = cmp->getRHS();
                        if (!mask || !rhs || mask->getType()->isBooleanType()
                            || !mask->getType()->isIntegerType())
                        {
                            return nullptr;
                        }

                        auto loc     = VirtualLoc(ctx);
                        auto *masked = clang::BinaryOperator::Create(
                            ctx, EnsureRValue(ctx, value), EnsureRValue(ctx, mask),
                            clang::BO_And, value->getType(), clang::VK_PRValue,
                            clang::OK_Ordinary, loc, clang::FPOptionsOverride()
                        );
                        auto *paren_masked = new (ctx) clang::ParenExpr(loc, loc, masked);
                        return clang::BinaryOperator::Create(
                            ctx, paren_masked, EnsureRValue(ctx, rhs), cmp->getOpcode(),
                            ctx.BoolTy, clang::VK_PRValue, clang::OK_Ordinary, loc,
                            clang::FPOptionsOverride()
                        );
                    };

                    if (auto *rewritten = build_bitmask_compare(bo->getLHS(), bo->getRHS())) {
                        return NormalizeBoolExpr(ctx, rewritten);
                    }
                    if (auto *rewritten = build_bitmask_compare(bo->getRHS(), bo->getLHS())) {
                        return NormalizeBoolExpr(ctx, rewritten);
                    }
                }
                if (bo->isEqualityOp() || bo->isRelationalOp()) {
                    auto parenthesize_bitwise = [&](clang::Expr *expr) -> clang::Expr * {
                        auto *child = llvm::dyn_cast_or_null< clang::BinaryOperator >(
                            expr->IgnoreParens()
                        );
                        if (!child) { return expr; }
                        if (child->getOpcode() != clang::BO_And
                            && child->getOpcode() != clang::BO_Or
                            && child->getOpcode() != clang::BO_Xor)
                        {
                            return expr;
                        }
                        return new (ctx)
                            clang::ParenExpr(VirtualLoc(ctx), VirtualLoc(ctx), expr);
                    };
                    bo->setLHS(parenthesize_bitwise(bo->getLHS()));
                    bo->setRHS(parenthesize_bitwise(bo->getRHS()));
                    return bo;
                }
                if (bo->getOpcode() == clang::BO_LAnd || bo->getOpcode() == clang::BO_LOr) {
                    bo->setLHS(NormalizeBoolExpr(ctx, bo->getLHS()));
                    bo->setRHS(NormalizeBoolExpr(ctx, bo->getRHS()));
                    if (bo->getOpcode() == clang::BO_LAnd) {
                        if (auto *simplified =
                                SimplifyRedundantNonzeroGuard(ctx, bo->getLHS(), bo->getRHS()))
                        {
                            return simplified;
                        }
                    }
                }
                return bo;
            }

            auto *uo = llvm::dyn_cast< clang::UnaryOperator >(e);
            if (!uo || uo->getOpcode() != clang::UO_LNot) { return e; }

            clang::Expr *sub = uo->getSubExpr()->IgnoreParens();

            // !!x → x
            if (auto *inner_uo = llvm::dyn_cast< clang::UnaryOperator >(sub)) {
                if (inner_uo->getOpcode() == clang::UO_LNot) {
                    return NormalizeBoolExpr(ctx, inner_uo->getSubExpr());
                }
            }

            clang::Expr *normalized_sub = NormalizeBoolExpr(ctx, uo->getSubExpr());
            if (normalized_sub != uo->getSubExpr()) {
                uo->setSubExpr(normalized_sub);
                sub = normalized_sub->IgnoreParens();
            }

            // !(a || b) -> !a && !b, !(a && b) -> !a || !b.
            // In boolean conditions this preserves short-circuit order and
            // exposes comparison flips handled below.
            if (auto *inner_logic = llvm::dyn_cast< clang::BinaryOperator >(sub)) {
                if (inner_logic->getOpcode() == clang::BO_LAnd
                    || inner_logic->getOpcode() == clang::BO_LOr)
                {
                    auto flipped = inner_logic->getOpcode() == clang::BO_LAnd ? clang::BO_LOr
                                                                              : clang::BO_LAnd;
                    auto *lhs = NormalizeBoolExpr(ctx, NegateExpr(ctx, inner_logic->getLHS()));
                    auto *rhs = NormalizeBoolExpr(ctx, NegateExpr(ctx, inner_logic->getRHS()));
                    return clang::BinaryOperator::Create(
                        ctx, ParenConditionOperand(ctx, lhs), ParenConditionOperand(ctx, rhs),
                        flipped, ctx.BoolTy, clang::VK_PRValue, clang::OK_Ordinary,
                        VirtualLoc(ctx), clang::FPOptionsOverride()
                    );
                }
            }

            // !(a OP b) → a FLIP(OP) b
            if (auto *inner_bo = llvm::dyn_cast< clang::BinaryOperator >(sub)) {
                auto op                           = inner_bo->getOpcode();
                clang::BinaryOperatorKind flipped = op;
                bool can_flip                     = false;
                if (inner_bo->isEqualityOp()) {
                    flipped  = (op == clang::BO_EQ) ? clang::BO_NE : clang::BO_EQ;
                    can_flip = true;
                } else if (inner_bo->isRelationalOp()) {
                    bool is_fp = inner_bo->getLHS()->getType()->isFloatingType()
                        || inner_bo->getRHS()->getType()->isFloatingType();
                    if (!is_fp) {
                        switch (op) {
                            case clang::BO_LT:
                                flipped = clang::BO_GE;
                                break;
                            case clang::BO_GT:
                                flipped = clang::BO_LE;
                                break;
                            case clang::BO_LE:
                                flipped = clang::BO_GT;
                                break;
                            case clang::BO_GE:
                                flipped = clang::BO_LT;
                                break;
                            default:
                                break;
                        }
                        can_flip = (flipped != op);
                    }
                }
                if (can_flip) {
                    return clang::BinaryOperator::Create(
                        ctx, inner_bo->getLHS(), inner_bo->getRHS(), flipped,
                        inner_bo->getType(), inner_bo->getValueKind(),
                        inner_bo->getObjectKind(), inner_bo->getOperatorLoc(),
                        clang::FPOptionsOverride()
                    );
                }
            }

            // No fold applies — normalize inside the `!` and keep it.
            uo->setSubExpr(NormalizeBoolExpr(ctx, uo->getSubExpr()));
            return uo;
        }

        // Walk the Stmt tree, normalizing every if/while/do/for condition.
        void NormalizeConditions(clang::ASTContext &ctx, clang::Stmt *s) {
            if (!s) { return; }
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(s)) {
                if (ifs->getCond()) { ifs->setCond(NormalizeBoolExpr(ctx, ifs->getCond())); }
                NormalizeConditions(ctx, ifs->getThen());
                NormalizeConditions(ctx, ifs->getElse());
                return;
            }
            if (auto *ws = llvm::dyn_cast< clang::WhileStmt >(s)) {
                if (ws->getCond()) { ws->setCond(NormalizeBoolExpr(ctx, ws->getCond())); }
                NormalizeConditions(ctx, ws->getBody());
                return;
            }
            if (auto *ds = llvm::dyn_cast< clang::DoStmt >(s)) {
                if (ds->getCond()) { ds->setCond(NormalizeBoolExpr(ctx, ds->getCond())); }
                NormalizeConditions(ctx, ds->getBody());
                return;
            }
            if (auto *fs = llvm::dyn_cast< clang::ForStmt >(s)) {
                if (fs->getCond()) { fs->setCond(NormalizeBoolExpr(ctx, fs->getCond())); }
                NormalizeConditions(ctx, fs->getBody());
                return;
            }
            for (auto *child : s->children()) { NormalizeConditions(ctx, child); }
        }

        struct ClangCleanupMetrics
        {
            size_t gotos       = 0;
            size_t labels      = 0;
            size_t dangling    = 0;
            size_t cross_scope = 0;
        };

        ClangCleanupMetrics MeasureClangCleanup(clang::Stmt *stmt) {
            ClangCleanupMetrics metrics;

            std::unordered_map< clang::LabelDecl *, unsigned > refs;
            CountGotoDeclRefs(stmt, refs);
            for (const auto &[label, count] : refs) {
                (void)label;
                metrics.gotos += count;
            }

            std::unordered_set< clang::LabelDecl * > defined;
            CollectDefinedLabels(stmt, defined);
            metrics.labels = defined.size();

            for (const auto &[label, count] : refs) {
                if (!defined.contains(label)) { metrics.dangling += count; }
            }

            metrics.cross_scope = CollectCrossScopeGotoTargets(stmt).size();
            return metrics;
        }

        llvm::StringRef CleanupReportName(std::string_view name) {
            if (name.empty()) { return "<unknown>"; }
            return llvm::StringRef(name.data(), name.size());
        }

    } // anonymous namespace

    void CleanupPrettyPrint(
        clang::FunctionDecl *fn, clang::ASTContext &ctx, bool report_cleanup,
        std::string_view function_name
    ) {
        if (!fn || !fn->hasBody()) { return; }
        auto initial_metrics = MeasureClangCleanup(fn->getBody());
        auto *body = CleanupStmtTree(ctx, fn->getBody());
        if (body) { fn->setBody(body); }

        auto apply_body = [&](clang::Stmt *next) {
            if (next) { fn->setBody(next); }
        };
        auto run_with_refs = [&](const auto &rewrite) {
            std::unordered_map< clang::LabelDecl *, unsigned > refs;
            CountGotoDeclRefs(fn->getBody(), refs);
            apply_body(rewrite(refs));
        };
        auto run_goto_to_next_label_fixed_point = [&]() {
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
        };
        auto run_goto_to_next_label_once = [&]() {
            std::unordered_set< clang::LabelDecl * > goto_targets;
            std::unordered_set< clang::Stmt * > seen;
            CollectGotoTargets(fn->getBody(), goto_targets, seen);
            body = EliminateGotoToNextLabel(ctx, fn->getBody(), &goto_targets);
            if (body) { fn->setBody(body); }
        };
        auto run_remove_dead_labels = [&]() {
            std::unordered_set< clang::LabelDecl * > goto_targets;
            std::unordered_set< clang::Stmt * > seen;
            CollectGotoTargets(fn->getBody(), goto_targets, seen);
            body = RemoveDeadLabels(ctx, fn->getBody(), goto_targets);
            if (body) { fn->setBody(body); }
        };
        auto run_remove_empty_blocks = [&]() {
            body = RemoveEmptyBlocks(ctx, fn->getBody());
            if (body) { fn->setBody(body); }
        };
        auto run_late_join_fixups = [&]() {
            run_goto_to_next_label_fixed_point();
            run_remove_dead_labels();
            run_remove_empty_blocks();
        };

        // Eliminate gotos to immediately following labels.  Iterates
        // to handle cascading patterns.
        run_goto_to_next_label_fixed_point();

        // Scope creation + goto-to-next-label cascade.  ScopeifyIfGotos
        // converts if(c) goto L; stmts; L: → if(!c) { stmts; }, which
        // may create new goto-to-next-label adjacencies, so iterate.
        std::vector< std::function< void() > > fixed_point_cleanup_schedule = {
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return RepairCrossScopeLabelEntries(ctx, fn->getBody(), refs);
                });
            },
            [&]() { apply_body(HoistCrossScopeLabelEntries(ctx, fn, fn->getBody())); },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return FoldClangSwitchLocalCaseTargets(ctx, fn->getBody(), refs);
                });
            },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return ScopeifyIfGotos(ctx, fn->getBody(), refs);
                });
            },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return FoldConditionalFallthroughChains(ctx, fn->getBody(), refs);
                });
            },
            [&]() { apply_body(ConvertImmediateLoopExitGotosToBreak(ctx, fn->getBody())); },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return PromoteLocalBackwardGotoLoops(ctx, fn->getBody(), refs);
                });
            },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return FoldLocalGotoDiamonds(ctx, fn->getBody(), refs);
                });
            },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return FoldCrossCompoundIfLabelDiamonds(ctx, fn->getBody(), refs);
                });
            },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return CloneFallthroughTerminalLabelGotos(ctx, fn->getBody(), refs);
                });
            },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return CloneNoFallthroughTerminalLabelGotos(ctx, fn->getBody(), refs);
                });
            },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return FoldForwardSingleRefLabelRegions(ctx, fn->getBody(), refs);
                });
            },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return SinkCommonTerminalEpilogues(ctx, fn->getBody(), refs);
                });
            },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return CloneFallthroughTerminalLabelGotos(ctx, fn->getBody(), refs);
                });
            },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return CloneNoFallthroughTerminalLabelGotos(ctx, fn->getBody(), refs);
                });
            },
            [&]() { run_goto_to_next_label_once(); },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return FoldClangSwitchLocalCaseTargets(ctx, fn->getBody(), refs);
                });
            },
        };
        for (int pass = 0; pass < kMaxGotoEliminationPasses; ++pass) {
            auto *prev = fn->getBody();
            for (const auto &step : fixed_point_cleanup_schedule) { step(); }
            if (fn->getBody() == prev) { break; }
        }

        // Remove labels that are not the target of any goto.
        // Run after CleanupStmtTree which may convert gotos to break/continue.
        run_remove_dead_labels();
        run_goto_to_next_label_fixed_point();
        run_remove_dead_labels();

        std::unordered_map< clang::LabelDecl *, unsigned > refs;
        CountGotoDeclRefs(fn->getBody(), refs);
        body = ScopeifyIfGotos(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        run_goto_to_next_label_once();
        run_remove_dead_labels();

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = FoldForwardSingleRefLabelRegions(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = SinkCommonTerminalEpilogues(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = InlineSingleRefTerminalLabelBlocks(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        run_remove_dead_labels();

        // Remove gotos whose target label was never emitted (orphaned
        // by structuring rules that absorbed the target block).
        std::unordered_set< clang::LabelDecl * > defined;
        CollectDefinedLabels(fn->getBody(), defined);
        body = RemoveOrphanedGotos(ctx, fn->getBody(), defined);
        if (body) { fn->setBody(body); }

        // Final pass: remove empty CompoundStmts and NullStmts.
        body = RemoveEmptyBlocks(ctx, fn->getBody());
        if (body) { fn->setBody(body); }

        body = HoistCrossScopeLabelEntries(ctx, fn, fn->getBody());
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = FoldConditionalFallthroughChains(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }
        body = ConvertImmediateLoopExitGotosToBreak(ctx, fn->getBody());
        if (body) { fn->setBody(body); }
        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = PromoteLocalBackwardGotoLoops(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }
        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = FoldLocalGotoDiamonds(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = FoldCrossCompoundIfLabelDiamonds(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = FoldCrossCompoundDispatchChains(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = CloneFallthroughTerminalLabelGotos(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = CloneNoFallthroughTerminalLabelGotos(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = FoldForwardSingleRefLabelRegions(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = FoldCrossCompoundDispatchChains(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = SinkCommonTerminalEpilogues(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = CloneFallthroughTerminalLabelGotos(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = CloneNoFallthroughTerminalLabelGotos(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        run_goto_to_next_label_fixed_point();
        run_remove_dead_labels();

        body = RemoveEmptyBlocks(ctx, fn->getBody());
        if (body) { fn->setBody(body); }

        bool promoted_counter_for = false;
        body = PromoteSimpleCounterWhileToFor(ctx, fn->getBody(), promoted_counter_for);
        if (promoted_counter_for && body) { fn->setBody(body); }

        bool removed_terminal_continue = false;
        body = RemoveRedundantTerminalForContinues(
            ctx, fn->getBody(), removed_terminal_continue
        );
        if (removed_terminal_continue && body) { fn->setBody(body); }

        body = RemoveEmptyBlocks(ctx, fn->getBody());
        if (body) { fn->setBody(body); }

        bool pushed_labels = false;
        body = PushLabelsIntoCompounds(ctx, fn->getBody(), pushed_labels);
        if (pushed_labels && body) { fn->setBody(body); }

        bool attached_empty_labels = false;
        body = AttachEmptyLabelsToFollowingStmt(ctx, fn->getBody(), attached_empty_labels);
        if (attached_empty_labels && body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = FoldLocalGotoDiamonds(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = FoldCrossCompoundIfLabelDiamonds(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        if (!ContainsSwitchStmt(fn->getBody())) {
            refs.clear();
            CountGotoDeclRefs(fn->getBody(), refs);
            body = CloneFallthroughTerminalLabelGotos(ctx, fn->getBody(), refs);
            if (body) { fn->setBody(body); }

            run_remove_dead_labels();

            body = RemoveEmptyBlocks(ctx, fn->getBody());
            if (body) { fn->setBody(body); }
        }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        bool folded_guarded_join = false;
        body = FoldGuardedJoinLabelChains(ctx, fn->getBody(), refs, folded_guarded_join);
        if (body) { fn->setBody(body); }

        if (folded_guarded_join) {
            run_late_join_fixups();
        }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        bool cloned_small_join = false;
        body = CloneSmallStraightLineLabelBeforeJoinGotos(
            ctx, fn->getBody(), refs, cloned_small_join
        );
        if (body) { fn->setBody(body); }

        if (cloned_small_join) {
            run_late_join_fixups();
        }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        bool cloned_cleanup_join = false;
        body = CloneCleanupLabelBeforeJoinGotos(ctx, fn->getBody(), refs, cloned_cleanup_join);
        if (body) { fn->setBody(body); }

        if (cloned_cleanup_join) {
            run_late_join_fixups();
        }

        // Cosmetic: fold double negations and `!(a OP b)` comparisons in
        // if/while/do/for conditions.  Runs last — purely a readability
        // pass, no effect on goto/label structure.
        NormalizeConditions(ctx, fn->getBody());

        if (report_cleanup) {
            auto final_metrics = MeasureClangCleanup(fn->getBody());
            llvm::errs() << "CLANG_CLEANUP_SUMMARY function="
                         << CleanupReportName(function_name)
                         << " initial_gotos=" << initial_metrics.gotos
                         << " final_gotos=" << final_metrics.gotos
                         << " initial_labels=" << initial_metrics.labels
                         << " final_labels=" << final_metrics.labels
                         << " initial_dangling=" << initial_metrics.dangling
                         << " final_dangling=" << final_metrics.dangling
                         << " initial_cross_scope=" << initial_metrics.cross_scope
                         << " final_cross_scope=" << final_metrics.cross_scope << "\n";
        }
    }

} // namespace patchestry::ast
