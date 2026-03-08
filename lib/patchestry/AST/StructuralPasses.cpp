/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/StructuralPasses.hpp>
#include <patchestry/Util/Log.hpp>

#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/Stmt.h>

#include "NormalizationPipelineInternal.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace patchestry::ast {

    namespace {

        // Build a label → position map for children of an SSeq
        std::unordered_map< std::string, size_t >
        buildLabelMap(const SSeq *seq) {
            std::unordered_map< std::string, size_t > map;
            for (size_t i = 0; i < seq->size(); ++i) {
                if (auto *lbl = seq->children()[i]->dyn_cast< SLabel >()) {
                    map[std::string(lbl->name())] = i;
                }
            }
            return map;
        }

        // Count goto references inside Clang Stmt* trees (inside SBlocks)
        unsigned countClangGotoRefs(const clang::Stmt *s, std::string_view label) {
            if (!s) return 0;
            if (auto *g = llvm::dyn_cast< clang::GotoStmt >(s)) {
                if (g->getLabel()->getName() ==
                    llvm::StringRef(label.data(), label.size())) return 1;
            }
            unsigned count = 0;
            for (auto *child : s->children()) {
                count += countClangGotoRefs(child, label);
            }
            return count;
        }

        // Count goto references to a label in an SNode tree
        // Also checks clang::GotoStmt inside SBlock stmts
        unsigned countGotoRefs(const SNode *node, std::string_view label) {
            if (!node) return 0;
            if (auto *g = node->dyn_cast< SGoto >()) {
                return g->target() == label ? 1 : 0;
            }
            unsigned count = 0;
            switch (node->kind()) {
            case SNodeKind::Seq: {
                auto *seq = node->as< SSeq >();
                for (auto *child : seq->children()) count += countGotoRefs(child, label);
                break;
            }
            case SNodeKind::Block: {
                auto *blk = node->as< SBlock >();
                for (auto *s : blk->stmts()) {
                    count += countClangGotoRefs(s, label);
                }
                break;
            }
            case SNodeKind::IfThenElse: {
                auto *ite = node->as< SIfThenElse >();
                count += countGotoRefs(ite->thenBranch(), label);
                count += countGotoRefs(ite->elseBranch(), label);
                break;
            }
            case SNodeKind::While: {
                auto *w = node->as< SWhile >();
                count += countGotoRefs(w->body(), label);
                break;
            }
            case SNodeKind::DoWhile: {
                auto *dw = node->as< SDoWhile >();
                count += countGotoRefs(dw->body(), label);
                break;
            }
            case SNodeKind::For: {
                auto *f = node->as< SFor >();
                count += countGotoRefs(f->body(), label);
                break;
            }
            case SNodeKind::Switch: {
                auto *sw = node->as< SSwitch >();
                for (auto &c : sw->cases()) count += countGotoRefs(c.body, label);
                count += countGotoRefs(sw->defaultBody(), label);
                break;
            }
            case SNodeKind::Label: {
                auto *l = node->as< SLabel >();
                count += countGotoRefs(l->body(), label);
                break;
            }
            default: break;
            }
            return count;
        }

        // Apply a transformation recursively to all SSeq nodes (bottom-up)
        bool transformSeqs(SNode *node, SNodeFactory &factory,
                           std::function< bool(SSeq *, SNodeFactory &) > fn) {
            if (!node) return false;
            bool changed = false;

            switch (node->kind()) {
            case SNodeKind::Seq: {
                auto *seq = node->as< SSeq >();
                for (auto *child : seq->children()) {
                    changed |= transformSeqs(child, factory, fn);
                }
                changed |= fn(seq, factory);
                break;
            }
            case SNodeKind::IfThenElse: {
                auto *ite = node->as< SIfThenElse >();
                changed |= transformSeqs(ite->thenBranch(), factory, fn);
                changed |= transformSeqs(ite->elseBranch(), factory, fn);
                break;
            }
            case SNodeKind::While: {
                auto *w = node->as< SWhile >();
                changed |= transformSeqs(w->body(), factory, fn);
                break;
            }
            case SNodeKind::DoWhile: {
                auto *dw = node->as< SDoWhile >();
                changed |= transformSeqs(dw->body(), factory, fn);
                break;
            }
            case SNodeKind::For: {
                auto *f = node->as< SFor >();
                changed |= transformSeqs(f->body(), factory, fn);
                break;
            }
            case SNodeKind::Switch: {
                auto *sw = node->as< SSwitch >();
                for (auto &c : sw->cases()) {
                    changed |= transformSeqs(c.body, factory, fn);
                }
                changed |= transformSeqs(sw->defaultBody(), factory, fn);
                break;
            }
            case SNodeKind::Label: {
                auto *l = node->as< SLabel >();
                changed |= transformSeqs(l->body(), factory, fn);
                break;
            }
            default: break;
            }
            return changed;
        }

    } // namespace

    // ========================================================================
    // Step 8: Sequence Collapse Pass
    // ========================================================================

    namespace {
        // Extract the inner SGoto from a node that might be SGoto directly
        // or SSeq containing a single SGoto
        SGoto *extractSGoto(SNode *node) {
            if (!node) return nullptr;
            if (auto *g = node->dyn_cast< SGoto >()) return g;
            if (auto *seq = node->dyn_cast< SSeq >()) {
                if (seq->size() == 1) return extractSGoto(seq->children()[0]);
            }
            return nullptr;
        }
    } // namespace

    bool SequenceCollapsePass::run(SNode *root, SNodeFactory &factory,
                                    clang::ASTContext & /*ctx*/) {
        // Pre-pass: unwrap singleton SSeqs inside if-then-else branches
        // if(cond) { SSeq[X] } → if(cond) { X }
        std::function< bool(SNode *) > unwrapSingletons = [&](SNode *node) -> bool {
            if (!node) return false;
            bool changed = false;
            switch (node->kind()) {
            case SNodeKind::Seq: {
                auto *seq = node->as< SSeq >();
                for (size_t i = 0; i < seq->size(); ++i) {
                    auto *child_seq = seq->children()[i]->dyn_cast< SSeq >();
                    if (child_seq && child_seq->size() == 1) {
                        seq->replaceChild(i, child_seq->children()[0]);
                        changed = true;
                    }
                }
                for (auto *child : seq->children()) changed |= unwrapSingletons(child);
                break;
            }
            case SNodeKind::IfThenElse: {
                auto *ite = node->as< SIfThenElse >();
                if (ite->thenBranch()) {
                    auto *then_seq = ite->thenBranch()->dyn_cast< SSeq >();
                    if (then_seq && then_seq->size() == 1) {
                        ite->setThenBranch(then_seq->children()[0]);
                        changed = true;
                    }
                    changed |= unwrapSingletons(ite->thenBranch());
                }
                if (ite->elseBranch()) {
                    auto *else_seq = ite->elseBranch()->dyn_cast< SSeq >();
                    if (else_seq && else_seq->size() == 1) {
                        ite->setElseBranch(else_seq->children()[0]);
                        changed = true;
                    }
                    changed |= unwrapSingletons(ite->elseBranch());
                }
                break;
            }
            case SNodeKind::While:
                changed |= unwrapSingletons(node->as< SWhile >()->body());
                break;
            case SNodeKind::DoWhile:
                changed |= unwrapSingletons(node->as< SDoWhile >()->body());
                break;
            case SNodeKind::For:
                changed |= unwrapSingletons(node->as< SFor >()->body());
                break;
            case SNodeKind::Label:
                changed |= unwrapSingletons(node->as< SLabel >()->body());
                break;
            default: break;
            }
            return changed;
        };
        unwrapSingletons(root);

        return transformSeqs(root, factory, [&](SSeq *seq, SNodeFactory & /*fac*/) -> bool {
            bool any_changed = false;
            bool changed;
            do {
            changed = false;

            // Pass 0: Remove empty SBlocks
            for (size_t i = 0; i < seq->size(); ) {
                auto *blk = seq->children()[i]->dyn_cast< SBlock >();
                if (blk && blk->empty() && blk->label().empty()) {
                    seq->removeChild(i);
                    changed = true;
                    continue;
                }
                ++i;
            }

            // Pass 1: Remove trivial gotos (goto L where L is the next label)
            for (size_t i = 0; i + 1 < seq->size(); ) {
                auto *g = seq->children()[i]->dyn_cast< SGoto >();
                if (g && i + 1 < seq->size()) {
                    auto *next_label = seq->children()[i + 1]->dyn_cast< SLabel >();
                    if (next_label && g->target() == next_label->name()) {
                        seq->removeChild(i);
                        changed = true;
                        continue;
                    }
                }
                ++i;
            }

            // Pass 1b: Remove trivial if(cond) goto L where L is the next label
            for (size_t i = 0; i + 1 < seq->size(); ) {
                auto *ite = seq->children()[i]->dyn_cast< SIfThenElse >();
                if (ite && !ite->elseBranch()) {
                    auto *then_goto = extractSGoto(ite->thenBranch());
                    if (then_goto && i + 1 < seq->size()) {
                        auto *next_label = seq->children()[i + 1]->dyn_cast< SLabel >();
                        if (next_label && then_goto->target() == next_label->name()) {
                            seq->removeChild(i);
                            changed = true;
                            continue;
                        }
                    }
                }
                // Also handle: if(cond) goto Lt else goto Le; Le:
                // → if(cond) goto Lt; (drop the trivial else-goto)
                if (ite && ite->elseBranch()) {
                    auto *else_goto = ite->elseBranch()->dyn_cast< SGoto >();
                    if (else_goto) {
                        auto *next_label = seq->children()[i + 1]->dyn_cast< SLabel >();
                        if (next_label && else_goto->target() == next_label->name()) {
                            ite->setElseBranch(nullptr);
                            changed = true;
                        }
                    }
                    auto *then_goto = ite->thenBranch()
                        ? ite->thenBranch()->dyn_cast< SGoto >() : nullptr;
                    if (then_goto) {
                        auto *next_label = seq->children()[i + 1]->dyn_cast< SLabel >();
                        if (next_label && then_goto->target() == next_label->name()) {
                            // Then-goto is trivial; if there's an else, swap
                            if (ite->elseBranch()) {
                                // if(cond) goto next; else X → if(!cond) X
                                // but we'd need ctx here for negation — skip for now
                            } else {
                                // No else either — just remove the if
                                seq->removeChild(i);
                                changed = true;
                                continue;
                            }
                        }
                    }
                }
                ++i;
            }

            // Pass 2: Merge adjacent SBlocks with no labels between
            for (size_t i = 0; i + 1 < seq->size(); ) {
                auto *blk1 = seq->children()[i]->dyn_cast< SBlock >();
                auto *blk2 = seq->children()[i + 1]->dyn_cast< SBlock >();
                if (blk1 && blk2 && blk2->label().empty()) {
                    // Merge blk2 into blk1
                    for (auto *s : blk2->stmts()) {
                        blk1->addStmt(s);
                    }
                    seq->removeChild(i + 1);
                    changed = true;
                    continue;
                }
                ++i;
            }

            // Pass 3: Remove dead labels (labels with no goto references)
            for (size_t i = 0; i < seq->size(); ) {
                auto *lbl = seq->children()[i]->dyn_cast< SLabel >();
                if (lbl && countGotoRefs(root, lbl->name()) == 0) {
                    // Replace label with its body
                    if (lbl->body()) {
                        seq->replaceChild(i, lbl->body());
                    } else {
                        seq->removeChild(i);
                    }
                    changed = true;
                    continue;
                }
                ++i;
            }

            any_changed |= changed;
            } while (changed);
            return any_changed;
        });
    }

    // ========================================================================
    // Step 9: If-Then-Else Recovery Pass
    // ========================================================================

    bool IfThenElseRecoveryPass::run(SNode *root, SNodeFactory &factory,
                                      clang::ASTContext &ctx) {
        return transformSeqs(root, factory, [&](SSeq *seq, SNodeFactory &fac) -> bool {
            bool changed = false;
            auto label_map = buildLabelMap(seq);

            for (size_t i = 0; i < seq->size(); ++i) {
                auto *ite = seq->children()[i]->dyn_cast< SIfThenElse >();
                if (!ite) continue;

                // Pattern: if(cond) { goto Lt } else { goto Le }
                auto *then_goto = ite->thenBranch()
                    ? ite->thenBranch()->dyn_cast< SGoto >() : nullptr;
                auto *else_goto = ite->elseBranch()
                    ? ite->elseBranch()->dyn_cast< SGoto >() : nullptr;

                if (!then_goto) continue;

                std::string then_target(then_goto->target());
                auto it_then = label_map.find(then_target);
                if (it_then == label_map.end()) continue;
                size_t then_idx = it_then->second;

                if (else_goto) {
                    // Two-sided diamond: if(cond) goto Lt; else goto Le;
                    std::string else_target(else_goto->target());
                    auto it_else = label_map.find(else_target);
                    if (it_else == label_map.end()) continue;
                    size_t else_idx = it_else->second;

                    // Need both targets to be after the if
                    if (then_idx <= i || else_idx <= i) continue;

                    // Determine which target comes first/second in layout
                    size_t first_idx = std::min(then_idx, else_idx);
                    size_t second_idx = std::max(then_idx, else_idx);

                    // Check for diamond: the first body ends with goto to
                    // a join label that's after the second body.
                    // Layout: [first_body...] goto Lj; [second_body...] Lj:
                    size_t join_idx = 0;
                    bool has_diamond = false;
                    if (second_idx > 0) {
                        auto *mid_goto = seq->children()[second_idx - 1]
                            ->dyn_cast< SGoto >();
                        if (mid_goto) {
                            auto jit = label_map.find(
                                std::string(mid_goto->target()));
                            if (jit != label_map.end()
                                && jit->second > second_idx) {
                                join_idx = jit->second;
                                has_diamond = true;
                            }
                        }
                    }

                    if (has_diamond) {
                        // Diamond pattern with common join point.
                        // first_body = [first_idx, second_idx-1) (strip goto)
                        // second_body = [second_idx, join_idx)
                        //   (strip trailing goto to join if present)
                        auto *first_seq = fac.make< SSeq >();
                        for (size_t j = first_idx; j < second_idx - 1; ++j)
                            first_seq->addChild(seq->children()[j]);

                        size_t second_end = join_idx;
                        if (second_end > second_idx) {
                            auto *tail = seq->children()[second_end - 1]
                                ->dyn_cast< SGoto >();
                            if (tail && label_map.count(
                                    std::string(tail->target()))
                                && label_map[std::string(tail->target())]
                                    == join_idx) {
                                second_end--;  // strip trailing goto to join
                            }
                        }
                        auto *second_seq = fac.make< SSeq >();
                        for (size_t j = second_idx; j < second_end; ++j)
                            second_seq->addChild(seq->children()[j]);

                        SNode *then_body, *else_body;
                        clang::Expr *cond_expr = ite->cond();
                        if (then_idx < else_idx) {
                            then_body = first_seq;
                            else_body = second_seq;
                        } else {
                            then_body = second_seq;
                            else_body = first_seq;
                        }

                        auto *new_ite = fac.make< SIfThenElse >(
                            cond_expr, then_body,
                            (else_body->dyn_cast<SSeq>() && else_body->dyn_cast<SSeq>()->empty()) ? nullptr : else_body
                        );
                        seq->replaceRange(i, join_idx, new_ite);
                        changed = true;
                    } else if (then_idx < else_idx) {
                        // No diamond join — single-sided: then comes first
                        auto *then_seq = fac.make< SSeq >();
                        for (size_t j = then_idx; j < else_idx; ++j)
                            then_seq->addChild(seq->children()[j]);

                        auto *new_ite = fac.make< SIfThenElse >(
                            ite->cond(), then_seq, nullptr
                        );
                        seq->replaceRange(i, else_idx, new_ite);
                        changed = true;
                    } else {
                        // No diamond join — single-sided: else comes first
                        auto *else_seq = fac.make< SSeq >();
                        for (size_t j = else_idx; j < then_idx; ++j)
                            else_seq->addChild(seq->children()[j]);

                        auto *neg_cond = detail::negateConditionWithParens(
                            ctx, ite->cond(), clang::SourceLocation()
                        );
                        auto *new_ite = fac.make< SIfThenElse >(
                            neg_cond, else_seq, nullptr
                        );
                        seq->replaceRange(i, then_idx, new_ite);
                        changed = true;
                    }
                    // Rebuild label map after modification
                    label_map = buildLabelMap(seq);
                } else {
                    // Single-sided: if(cond) goto Lt; <fall-through>; Lt: ...
                    if (then_idx <= i + 1) continue;

                    // Check for diamond: fallthrough ends with goto to a join
                    // label that's after then_idx.
                    // Pattern: if(cond) goto Lt; <else-code>; goto Lj; Lt: <then-code>; Lj:
                    // → if(cond) { then-code } else { else-code } join-code
                    auto *pre_then_last = seq->children()[then_idx - 1];
                    auto *join_goto = pre_then_last->dyn_cast< SGoto >();
                    if (join_goto) {
                        auto join_it = label_map.find(std::string(join_goto->target()));
                        if (join_it != label_map.end()
                            && join_it->second > then_idx) {
                            size_t join_idx = join_it->second;

                            // Else-branch = [i+1, then_idx-1) (skip trailing goto)
                            auto *else_seq = fac.make< SSeq >();
                            for (size_t j = i + 1; j < then_idx - 1; ++j) {
                                else_seq->addChild(seq->children()[j]);
                            }

                            // Then-branch = [then_idx, join_idx)
                            auto *then_seq = fac.make< SSeq >();
                            for (size_t j = then_idx; j < join_idx; ++j) {
                                then_seq->addChild(seq->children()[j]);
                            }

                            auto *new_ite = fac.make< SIfThenElse >(
                                ite->cond(), then_seq,
                                else_seq->empty() ? nullptr : else_seq
                            );
                            seq->replaceRange(i, join_idx, new_ite);
                            changed = true;
                            label_map = buildLabelMap(seq);
                            continue;
                        }
                    }

                    // Simple single-sided: if(!cond) { fallthrough stmts }
                    auto *body_seq = fac.make< SSeq >();
                    for (size_t j = i + 1; j < then_idx; ++j) {
                        body_seq->addChild(seq->children()[j]);
                    }

                    auto *neg_cond = detail::negateConditionWithParens(
                        ctx, ite->cond(), clang::SourceLocation()
                    );
                    auto *new_if = fac.make< SIfThenElse >(neg_cond, body_seq);
                    seq->replaceRange(i, then_idx, new_if);
                    changed = true;
                    label_map = buildLabelMap(seq);
                }
            }

            return changed;
        });
    }

    // ========================================================================
    // Step 10: While Loop Recovery Pass
    // ========================================================================

    bool WhileLoopRecoveryPass::run(SNode *root, SNodeFactory &factory,
                                     clang::ASTContext &ctx) {
        return transformSeqs(root, factory, [&](SSeq *seq, SNodeFactory &fac) -> bool {
            bool changed = false;
            auto label_map = buildLabelMap(seq);

            for (const auto &loop : loop_info_.loops) {
                // Find the header label in the sequence
                if (loop.header >= cfg_.blocks.size()) continue;
                auto header_label = cfg_.blocks[loop.header].label;
                if (header_label.empty()) continue;

                auto it = label_map.find(header_label);
                if (it == label_map.end()) continue;
                size_t header_pos = it->second;

                // Find the end of the loop in the sequence
                size_t loop_end = header_pos + 1;
                for (size_t b : loop.body) {
                    if (b >= cfg_.blocks.size()) continue;
                    auto blabel = cfg_.blocks[b].label;
                    if (blabel.empty()) continue;
                    auto bit = label_map.find(blabel);
                    if (bit != label_map.end() && bit->second >= loop_end) {
                        loop_end = bit->second + 1;
                    }
                }
                if (loop_end > seq->size()) loop_end = seq->size();

                // Extend loop_end to include trailing non-label nodes that are
                // part of the loop (back-edge gotos, conditionals targeting header)
                while (loop_end < seq->size()) {
                    auto *trail = seq->children()[loop_end];
                    if (auto *g = trail->dyn_cast< SGoto >()) {
                        if (g->target() == header_label) {
                            ++loop_end;
                            continue;
                        }
                    }
                    if (auto *ite = trail->dyn_cast< SIfThenElse >()) {
                        auto *tg = extractSGoto(ite->thenBranch());
                        if (tg && tg->target() == header_label) {
                            ++loop_end;
                            continue;
                        }
                    }
                    break;
                }

                // Check the header for a condition
                if (header_pos >= seq->size()) continue;
                if (loop_end <= header_pos + 1) continue;

                // === Try while pattern FIRST ===
                // Pattern: header_label → if(cond) goto target; ...
                auto *header_node = seq->children()[header_pos];
                auto *header_label_node = header_node->dyn_cast< SLabel >();
                bool matched_while = false;

                if (header_label_node && header_pos + 1 < loop_end) {
                    auto *cond_ite = seq->children()[header_pos + 1]->dyn_cast< SIfThenElse >();
                    if (cond_ite && !cond_ite->elseBranch()) {
                        auto *then_goto = extractSGoto(cond_ite->thenBranch());
                        if (then_goto) {
                            auto target = std::string(then_goto->target());
                            auto target_it = label_map.find(target);

                            if (target_it != label_map.end()
                                && target_it->second >= loop_end) {
                                // Pattern A: if(exit_cond) goto exit; body;
                                // → while(!exit_cond) { body }
                                auto *body_seq = fac.make< SSeq >();
                                for (size_t j = header_pos + 2; j < loop_end; ++j) {
                                    auto *child = seq->children()[j];
                                    if (auto *g = child->dyn_cast< SGoto >()) {
                                        if (g->target() == header_label) continue;
                                    }
                                    body_seq->addChild(child);
                                }
                                auto *neg_cond = detail::negateConditionWithParens(
                                    ctx, cond_ite->cond(), clang::SourceLocation()
                                );
                                auto *w = fac.make< SWhile >(neg_cond, body_seq);
                                seq->replaceRange(header_pos, loop_end, w);
                                changed = true;
                                matched_while = true;
                            } else if (target_it != label_map.end()
                                       && target_it->second > header_pos + 1
                                       && target_it->second < loop_end) {
                                // Pattern B: if(cont_cond) goto body_label;
                                //            exit_code; body_label: body;
                                // → while(cont_cond) { body } exit_code
                                size_t body_start = target_it->second;

                                auto *body_seq = fac.make< SSeq >();
                                for (size_t j = body_start; j < loop_end; ++j) {
                                    auto *child = seq->children()[j];
                                    if (auto *g = child->dyn_cast< SGoto >()) {
                                        if (g->target() == header_label) continue;
                                    }
                                    body_seq->addChild(child);
                                }

                                std::vector< SNode * > exit_nodes;
                                for (size_t j = header_pos + 2; j < body_start; ++j) {
                                    exit_nodes.push_back(seq->children()[j]);
                                }

                                auto *w = fac.make< SWhile >(
                                    cond_ite->cond(), body_seq
                                );

                                std::vector< SNode * > replacements;
                                replacements.push_back(w);
                                for (auto *en : exit_nodes) {
                                    replacements.push_back(en);
                                }
                                seq->replaceRange(header_pos, loop_end, replacements);
                                changed = true;
                                matched_while = true;
                            }
                        }
                    }
                }

                if (matched_while) {
                    label_map = buildLabelMap(seq);
                    continue;
                }

                // === Try do-while pattern ===
                auto *last = seq->children()[loop_end - 1];

                // Conditional back-edge: if(cond) goto header → do-while(cond)
                auto *last_ite = last->dyn_cast< SIfThenElse >();
                if (last_ite) {
                    auto *then_goto = extractSGoto(last_ite->thenBranch());
                    if (then_goto && then_goto->target() == header_label) {
                        auto *body_seq = fac.make< SSeq >();
                        for (size_t j = header_pos; j < loop_end - 1; ++j) {
                            body_seq->addChild(seq->children()[j]);
                        }
                        auto *dw = fac.make< SDoWhile >(
                            body_seq, last_ite->cond()
                        );
                        seq->replaceRange(header_pos, loop_end, dw);
                        changed = true;
                        label_map = buildLabelMap(seq);
                        continue;
                    }
                }

                // Unconditional back-edge: goto header → do { body } while (1)
                auto *last_goto = last->dyn_cast< SGoto >();
                if (last_goto && last_goto->target() == header_label) {
                    auto *body_seq = fac.make< SSeq >();
                    for (size_t j = header_pos; j < loop_end - 1; ++j) {
                        body_seq->addChild(seq->children()[j]);
                    }
                    auto *one = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1),
                        ctx.UnsignedIntTy, clang::SourceLocation()
                    );
                    auto *dw = fac.make< SDoWhile >(body_seq, one);
                    seq->replaceRange(header_pos, loop_end, dw);
                    changed = true;
                    label_map = buildLabelMap(seq);
                    continue;
                }
            }

            return changed;
        });
    }

    // ========================================================================
    // Step 11: Forward Goto Elimination Pass
    // ========================================================================

    bool ForwardGotoEliminationPass::run(SNode *root, SNodeFactory &factory,
                                          clang::ASTContext &ctx) {
        return transformSeqs(root, factory, [&](SSeq *seq, SNodeFactory &fac) -> bool {
            bool changed = false;
            auto label_map = buildLabelMap(seq);

            for (size_t i = 0; i < seq->size(); ++i) {
                // Pattern: if(cond) goto L; <stmts>; L: ...
                //       → if(!cond) { <stmts> }
                auto *ite = seq->children()[i]->dyn_cast< SIfThenElse >();
                if (!ite || ite->elseBranch()) continue;

                auto *then_goto = extractSGoto(ite->thenBranch());
                if (!then_goto) continue;

                std::string target(then_goto->target());
                auto it = label_map.find(target);
                if (it == label_map.end()) continue;

                size_t target_idx = it->second;
                if (target_idx <= i) continue; // not forward

                if (target_idx == i + 1) {
                    // Goto targets the immediately next label — the if is a no-op
                    seq->removeChild(i);
                    changed = true;
                    label_map = buildLabelMap(seq);
                    continue;
                }

                // Wrap [i+1, target_idx) in if(!cond)
                auto *body_seq = fac.make< SSeq >();
                for (size_t j = i + 1; j < target_idx; ++j) {
                    body_seq->addChild(seq->children()[j]);
                }

                auto *neg_cond = detail::negateConditionWithParens(
                    ctx, ite->cond(), clang::SourceLocation()
                );
                auto *new_if = fac.make< SIfThenElse >(neg_cond, body_seq);
                seq->replaceRange(i, target_idx, new_if);
                changed = true;
                label_map = buildLabelMap(seq);
            }

            return changed;
        });
    }

    // ========================================================================
    // Step 12: Backward Goto to Do-While Pass
    // ========================================================================

    bool BackwardGotoToDoWhilePass::run(SNode *root, SNodeFactory &factory,
                                         clang::ASTContext &ctx) {
        return transformSeqs(root, factory, [&](SSeq *seq, SNodeFactory &fac) -> bool {
            bool changed = false;
            auto label_map = buildLabelMap(seq);

            for (size_t i = 0; i < seq->size(); ++i) {
                // Pattern: ... if(cond) goto L; where L is before i
                //       → do { L: ...; } while(cond);
                auto *ite = seq->children()[i]->dyn_cast< SIfThenElse >();
                SGoto *back_goto = nullptr;
                clang::Expr *cond = nullptr;

                if (ite && !ite->elseBranch()) {
                    back_goto = ite->thenBranch()
                        ? ite->thenBranch()->dyn_cast< SGoto >() : nullptr;
                    cond = ite->cond();
                }

                // Also check plain SGoto (unconditional back-edge)
                if (!back_goto) {
                    auto *plain_goto = seq->children()[i]->dyn_cast< SGoto >();
                    if (plain_goto) {
                        std::string target(plain_goto->target());
                        auto it = label_map.find(target);
                        if (it != label_map.end() && it->second < i) {
                            // Unconditional backward goto → do { ... } while(true)
                            size_t label_idx = it->second;
                            auto *body_seq = fac.make< SSeq >();
                            for (size_t j = label_idx; j < i; ++j) {
                                body_seq->addChild(seq->children()[j]);
                            }
                            auto *true_cond = detail::makeBoolTrue(ctx, clang::SourceLocation());
                            auto *dw = fac.make< SDoWhile >(body_seq, true_cond);
                            seq->replaceRange(label_idx, i + 1, dw);
                            changed = true;
                            label_map = buildLabelMap(seq);
                            continue;
                        }
                    }
                    continue;
                }

                std::string target(back_goto->target());
                auto it = label_map.find(target);
                if (it == label_map.end()) continue;

                size_t label_idx = it->second;
                if (label_idx >= i) continue; // not backward

                // Wrap [label_idx, i) in do { ... } while(cond)
                auto *body_seq = fac.make< SSeq >();
                for (size_t j = label_idx; j < i; ++j) {
                    body_seq->addChild(seq->children()[j]);
                }

                auto *dw = fac.make< SDoWhile >(body_seq, cond);
                seq->replaceRange(label_idx, i + 1, dw);
                changed = true;
                label_map = buildLabelMap(seq);
            }

            return changed;
        });
    }

    // ========================================================================
    // Step 13: Switch Recovery Pass
    // ========================================================================

    bool SwitchRecoveryPass::run(SNode *root, SNodeFactory &factory,
                                  clang::ASTContext & /*ctx*/) {
        return transformSeqs(root, factory, [&](SSeq *seq, SNodeFactory &fac) -> bool {
            bool changed = false;

            // Look for cascaded equality comparisons:
            // if(x == 1) goto L1; if(x == 2) goto L2; ...
            for (size_t i = 0; i < seq->size(); ++i) {
                auto *ite = seq->children()[i]->dyn_cast< SIfThenElse >();
                if (!ite || ite->elseBranch()) continue;

                auto *then_goto = ite->thenBranch()
                    ? ite->thenBranch()->dyn_cast< SGoto >() : nullptr;
                if (!then_goto) continue;

                // Check if condition is x == constant
                auto *binop = llvm::dyn_cast_or_null< clang::BinaryOperator >(ite->cond());
                if (!binop || binop->getOpcode() != clang::BO_EQ) continue;

                clang::Expr *discriminant = binop->getLHS();
                clang::Expr *first_value = binop->getRHS();

                // Collect consecutive cases
                struct CaseInfo {
                    clang::Expr *value;
                    std::string target;
                };
                std::vector< CaseInfo > cases;
                cases.push_back({first_value, std::string(then_goto->target())});

                size_t end = i + 1;
                while (end < seq->size()) {
                    auto *next_ite = seq->children()[end]->dyn_cast< SIfThenElse >();
                    if (!next_ite || next_ite->elseBranch()) break;

                    auto *next_goto = next_ite->thenBranch()
                        ? next_ite->thenBranch()->dyn_cast< SGoto >() : nullptr;
                    if (!next_goto) break;

                    auto *next_binop = llvm::dyn_cast_or_null< clang::BinaryOperator >(
                        next_ite->cond()
                    );
                    if (!next_binop || next_binop->getOpcode() != clang::BO_EQ) break;

                    // TODO: check that LHS is the same discriminant
                    cases.push_back({next_binop->getRHS(),
                                    std::string(next_goto->target())});
                    ++end;
                }

                // Need at least 3 cases to form a switch
                if (cases.size() < 3) continue;

                auto label_map = buildLabelMap(seq);

                auto *sw = fac.make< SSwitch >(discriminant);
                for (const auto &c : cases) {
                    // Build case body from target label
                    auto it = label_map.find(c.target);
                    SNode *case_body = nullptr;
                    if (it != label_map.end()) {
                        case_body = fac.make< SGoto >(
                            fac.intern(c.target)
                        );
                    }
                    sw->addCase(c.value, case_body);
                }

                seq->replaceRange(i, end, sw);
                changed = true;
            }

            return changed;
        });
    }

    // ========================================================================
    // Step 14: Short-Circuit Boolean Recovery Pass
    // ========================================================================

    bool ShortCircuitRecoveryPass::run(SNode *root, SNodeFactory &factory,
                                        clang::ASTContext &ctx) {
        return transformSeqs(root, factory, [&](SSeq *seq, SNodeFactory &fac) -> bool {
            bool changed = false;

            for (size_t i = 0; i + 1 < seq->size(); ++i) {
                auto *ite1 = seq->children()[i]->dyn_cast< SIfThenElse >();
                auto *ite2 = seq->children()[i + 1]->dyn_cast< SIfThenElse >();
                if (!ite1 || !ite2) continue;
                // First if must be single-sided (no else); second may have else
                if (ite1->elseBranch()) continue;

                auto *goto1 = ite1->thenBranch()
                    ? ite1->thenBranch()->dyn_cast< SGoto >() : nullptr;
                auto *goto2 = ite2->thenBranch()
                    ? ite2->thenBranch()->dyn_cast< SGoto >() : nullptr;
                if (!goto1 || !goto2) continue;

                // Pattern: if(a) goto L; if(b) goto L; [else goto M;]
                // → if(a||b) goto L; [else goto M;]
                if (goto1->target() == goto2->target()) {
                    // Create || of the two conditions (both jump to same target)
                    auto *lor = clang::BinaryOperator::Create(
                        ctx, detail::ensureRValue(ctx, ite1->cond()),
                        detail::ensureRValue(ctx, ite2->cond()),
                        clang::BO_LOr, ctx.IntTy, clang::VK_PRValue,
                        clang::OK_Ordinary, clang::SourceLocation(),
                        clang::FPOptionsOverride()
                    );

                    auto *combined = fac.make< SIfThenElse >(
                        lor, fac.make< SGoto >(goto1->target()),
                        ite2->elseBranch()  // preserve else from second if
                    );
                    seq->replaceChild(i, combined);
                    seq->removeChild(i + 1);
                    changed = true;
                    --i; // re-check in case there's a third consecutive if
                }
            }

            return changed;
        });
    }

    // ========================================================================
    // Step 15: Multi-Exit Break Pass
    // ========================================================================

    bool MultiExitBreakPass::run(SNode *root, SNodeFactory &factory,
                                  clang::ASTContext & /*ctx*/) {
        // Walk the tree looking for gotos inside loops that target labels after the loop
        // Replace them with break statements

        std::function< bool(SNode *) > visit = [&](SNode *node) -> bool {
            if (!node) return false;
            bool changed = false;

            auto processLoopBody = [&](SNode *body, SNode * /*loop_node*/) -> bool {
                if (!body) return false;
                auto *seq = body->dyn_cast< SSeq >();
                if (!seq) return false;

                bool mod = false;
                for (size_t i = 0; i < seq->size(); ++i) {
                    // if(cond) goto L; where L is after loop → if(cond) break;
                    auto *ite = seq->children()[i]->dyn_cast< SIfThenElse >();
                    if (ite && !ite->elseBranch()) {
                        auto *g = ite->thenBranch()
                            ? ite->thenBranch()->dyn_cast< SGoto >() : nullptr;
                        if (g) {
                            // Check if target is not inside this loop body
                            // (simple heuristic: label not in this seq)
                            auto label_map = buildLabelMap(seq);
                            if (label_map.find(std::string(g->target())) == label_map.end()) {
                                ite->setThenBranch(factory.make< SBreak >());
                                mod = true;
                            }
                        }
                    }

                    // Plain goto L; where L is after loop → break;
                    auto *g = seq->children()[i]->dyn_cast< SGoto >();
                    if (g) {
                        auto label_map = buildLabelMap(seq);
                        if (label_map.find(std::string(g->target())) == label_map.end()) {
                            seq->replaceChild(i, factory.make< SBreak >());
                            mod = true;
                        }
                    }
                }
                return mod;
            };

            switch (node->kind()) {
            case SNodeKind::While: {
                auto *w = node->as< SWhile >();
                changed |= processLoopBody(w->body(), node);
                changed |= visit(w->body());
                break;
            }
            case SNodeKind::DoWhile: {
                auto *dw = node->as< SDoWhile >();
                changed |= processLoopBody(dw->body(), node);
                changed |= visit(dw->body());
                break;
            }
            case SNodeKind::For: {
                auto *f = node->as< SFor >();
                changed |= processLoopBody(f->body(), node);
                changed |= visit(f->body());
                break;
            }
            case SNodeKind::Seq: {
                auto *seq = node->as< SSeq >();
                for (auto *child : seq->children()) changed |= visit(child);
                break;
            }
            case SNodeKind::IfThenElse: {
                auto *ite = node->as< SIfThenElse >();
                changed |= visit(ite->thenBranch());
                changed |= visit(ite->elseBranch());
                break;
            }
            case SNodeKind::Label: {
                auto *l = node->as< SLabel >();
                changed |= visit(l->body());
                break;
            }
            default: break;
            }

            return changed;
        };

        return visit(root);
    }

    // ========================================================================
    // SwitchBackedgeLoop Pass
    // Detects switch statements inside SBlocks whose case arms contain
    // GotoStmt back-edges to the enclosing label, and wraps them in
    // while(true) with continue/break replacing the gotos.
    // ========================================================================

    namespace {
        // Recursively replace GotoStmt targeting `target_label` with
        // `replacement` inside a Clang Stmt tree.  Returns a new Stmt if
        // the top-level node itself is a matching goto.
        clang::Stmt *replaceClangGoto(
            clang::ASTContext &ctx, clang::Stmt *stmt,
            llvm::StringRef target_label, clang::Stmt *replacement
        ) {
            if (!stmt) return stmt;

            // If this stmt IS the goto we're looking for, return the replacement
            if (auto *g = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                if (g->getLabel()->getName() == target_label) {
                    return replacement;
                }
                return stmt;
            }

            // For CompoundStmt, rebuild with replaced children
            if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(stmt)) {
                std::vector< clang::Stmt * > new_stmts;
                bool any = false;
                for (auto *child : cs->body()) {
                    auto *replaced = replaceClangGoto(ctx, child, target_label, replacement);
                    new_stmts.push_back(replaced);
                    if (replaced != child) any = true;
                }
                if (any) {
                    return clang::CompoundStmt::Create(
                        ctx, new_stmts, clang::FPOptionsOverride(),
                        cs->getLBracLoc(), cs->getRBracLoc()
                    );
                }
                return stmt;
            }

            // For CaseStmt, replace the sub-statement
            if (auto *cs = llvm::dyn_cast< clang::CaseStmt >(stmt)) {
                auto *sub = cs->getSubStmt();
                auto *new_sub = replaceClangGoto(ctx, sub, target_label, replacement);
                if (new_sub != sub) {
                    cs->setSubStmt(new_sub);
                }
                return stmt;
            }

            // For DefaultStmt, replace the sub-statement
            if (auto *ds = llvm::dyn_cast< clang::DefaultStmt >(stmt)) {
                auto *sub = ds->getSubStmt();
                auto *new_sub = replaceClangGoto(ctx, sub, target_label, replacement);
                if (new_sub != sub) {
                    ds->setSubStmt(new_sub);
                }
                return stmt;
            }

            // For LabelStmt, replace the sub-statement
            if (auto *ls = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                auto *sub = ls->getSubStmt();
                auto *new_sub = replaceClangGoto(ctx, sub, target_label, replacement);
                if (new_sub != sub) {
                    ls->setSubStmt(new_sub);
                }
                return stmt;
            }

            // For IfStmt, replace then/else
            if (auto *is = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                auto *t = is->getThen();
                auto *e = is->getElse();
                auto *new_t = replaceClangGoto(ctx, t, target_label, replacement);
                auto *new_e = replaceClangGoto(ctx, e, target_label, replacement);
                if (new_t != t) is->setThen(new_t);
                if (new_e != e) is->setElse(new_e);
                return stmt;
            }

            // For SwitchStmt, replace the body
            if (auto *sw = llvm::dyn_cast< clang::SwitchStmt >(stmt)) {
                auto *body = sw->getBody();
                auto *new_body = replaceClangGoto(ctx, body, target_label, replacement);
                if (new_body != body) sw->setBody(new_body);
                return stmt;
            }

            return stmt;
        }

        // Check if a Clang Stmt tree contains a GotoStmt to `target_label`
        bool hasClangGotoTo(const clang::Stmt *stmt, llvm::StringRef target_label) {
            if (!stmt) return false;
            if (auto *g = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                return g->getLabel()->getName() == target_label;
            }
            for (auto *child : stmt->children()) {
                if (hasClangGotoTo(child, target_label)) return true;
            }
            return false;
        }
    } // namespace

    bool SwitchBackedgeLoopPass::run(SNode *root, SNodeFactory &factory,
                                      clang::ASTContext &ctx) {
        return transformSeqs(root, factory, [&](SSeq *seq, SNodeFactory &fac) -> bool {
            bool changed = false;

            for (size_t i = 0; i < seq->size(); ++i) {
                // Look for SLabel whose body is an SBlock containing a SwitchStmt
                auto *lbl = seq->children()[i]->dyn_cast< SLabel >();
                if (!lbl) continue;

                auto *body_block = lbl->body() ? lbl->body()->dyn_cast< SBlock >() : nullptr;
                if (!body_block) continue;

                // Find a SwitchStmt in the block's stmts
                clang::SwitchStmt *sw = nullptr;
                for (auto *s : body_block->stmts()) {
                    if (auto *ss = llvm::dyn_cast< clang::SwitchStmt >(s)) {
                        sw = ss;
                        break;
                    }
                }
                if (!sw) continue;

                // Check if any case arm has a goto back to this label (backedge)
                std::string label_name(lbl->name());
                llvm::StringRef label_ref(label_name);
                if (!hasClangGotoTo(sw, label_ref)) continue;

                // Found switch-dispatch loop pattern.
                // Determine exit label: check if next sibling is an SLabel
                std::string exit_label;
                if (i + 1 < seq->size()) {
                    if (auto *next_lbl = seq->children()[i + 1]->dyn_cast< SLabel >()) {
                        exit_label = std::string(next_lbl->name());
                    }
                }

                // Replace backedge gotos with ContinueStmt
                auto *cont_stmt = new (ctx) clang::ContinueStmt(clang::SourceLocation());
                replaceClangGoto(ctx, sw, label_ref, cont_stmt);

                // Replace exit gotos with BreakStmt
                if (!exit_label.empty()) {
                    auto *break_stmt = new (ctx) clang::BreakStmt(clang::SourceLocation());
                    replaceClangGoto(ctx, sw, llvm::StringRef(exit_label), break_stmt);
                }

                // Build while(true) { <block stmts> }
                auto *true_cond = detail::makeBoolTrue(ctx, clang::SourceLocation());
                auto *inner_block = fac.make< SBlock >();
                for (auto *s : body_block->stmts()) {
                    inner_block->addStmt(s);
                }
                auto *inner_seq = fac.make< SSeq >();
                inner_seq->addChild(inner_block);
                auto *w = fac.make< SWhile >(true_cond, inner_seq);

                // Replace the SLabel with the while loop
                seq->replaceChild(i, w);
                changed = true;
            }

            return changed;
        });
    }

    // ========================================================================
    // SwitchGotoInlining Pass
    // Inlines label bodies into switch case arms when case arms are
    // goto-to-label dispatch patterns. Also replaces trailing gotos to
    // the join/continue label with break.
    // ========================================================================

    namespace {
        // Collect goto targets from case arms of a SwitchStmt
        struct CaseGotoInfo {
            clang::SwitchCase *case_stmt;  // CaseStmt or DefaultStmt
            std::string goto_target;
        };

        std::vector< CaseGotoInfo > collectCaseGotos(clang::SwitchStmt *sw) {
            std::vector< CaseGotoInfo > result;
            for (auto *sc = sw->getSwitchCaseList(); sc; sc = sc->getNextSwitchCase()) {
                auto *sub = sc->getSubStmt();
                // Case body might be a GotoStmt directly
                if (auto *g = llvm::dyn_cast_or_null< clang::GotoStmt >(sub)) {
                    result.push_back({sc, g->getLabel()->getName().str()});
                }
                // Or a CompoundStmt containing a single GotoStmt
                else if (auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(sub)) {
                    if (cs->size() == 1) {
                        if (auto *g2 = llvm::dyn_cast< clang::GotoStmt >(cs->body_front())) {
                            result.push_back({sc, g2->getLabel()->getName().str()});
                        }
                    }
                }
            }
            return result;
        }

        // Build label → position map for SSeq, looking inside SLabel nodes
        // AND also plain SBlocks that start with a LabelStmt
        std::unordered_map< std::string, size_t >
        buildFullLabelMap(const SSeq *seq) {
            std::unordered_map< std::string, size_t > map;
            for (size_t i = 0; i < seq->size(); ++i) {
                if (auto *lbl = seq->children()[i]->dyn_cast< SLabel >()) {
                    map[std::string(lbl->name())] = i;
                }
            }
            return map;
        }

        // Collect all stmts from an SNode tree (SBlock stmts, recursing into SSeq/SLabel)
        void collectStmtsFromSNode(SNode *node, std::vector< clang::Stmt * > &out) {
            if (!node) return;
            if (auto *blk = node->dyn_cast< SBlock >()) {
                for (auto *s : blk->stmts()) out.push_back(s);
            } else if (auto *seq = node->dyn_cast< SSeq >()) {
                for (auto *child : seq->children()) {
                    collectStmtsFromSNode(child, out);
                }
            } else if (auto *lbl = node->dyn_cast< SLabel >()) {
                collectStmtsFromSNode(lbl->body(), out);
            } else if (auto *g = node->dyn_cast< SGoto >()) {
                // Cannot convert SGoto directly to Clang — keep as marker
                (void) g;
            }
        }

        // Replace trailing GotoStmt to `join_label` with BreakStmt in a stmt list
        void replaceTrailingGotoWithBreak(
            clang::ASTContext &ctx, std::vector< clang::Stmt * > &stmts,
            llvm::StringRef join_label
        ) {
            if (stmts.empty()) return;
            auto *last = stmts.back();
            if (auto *g = llvm::dyn_cast< clang::GotoStmt >(last)) {
                if (g->getLabel()->getName() == join_label) {
                    stmts.back() = new (ctx) clang::BreakStmt(clang::SourceLocation());
                }
            }
        }
    } // namespace

    bool SwitchGotoInliningPass::run(SNode *root, SNodeFactory &factory,
                                      clang::ASTContext &ctx) {
        return transformSeqs(root, factory, [&](SSeq *seq, SNodeFactory & /*fac*/) -> bool {
            bool changed = false;

            for (size_t i = 0; i < seq->size(); ++i) {
                // Find SBlock containing a SwitchStmt with goto-to-label case arms.
                // The SBlock may be bare, or wrapped in an SLabel.
                SBlock *blk = seq->children()[i]->dyn_cast< SBlock >();
                if (!blk) {
                    auto *lbl = seq->children()[i]->dyn_cast< SLabel >();
                    if (lbl && lbl->body()) {
                        blk = lbl->body()->dyn_cast< SBlock >();
                    }
                }
                if (!blk) continue;

                clang::SwitchStmt *sw = nullptr;
                for (auto *s : blk->stmts()) {
                    if (auto *ss = llvm::dyn_cast< clang::SwitchStmt >(s)) {
                        sw = ss;
                        break;
                    }
                }
                if (!sw) continue;

                auto case_gotos = collectCaseGotos(sw);
                if (case_gotos.empty()) continue;

                // Check that at least some targets are labels in the same SSeq
                auto label_map = buildFullLabelMap(seq);
                bool any_target_found = false;
                for (auto &cg : case_gotos) {
                    if (label_map.find(cg.goto_target) != label_map.end()) {
                        any_target_found = true;
                        break;
                    }
                }
                if (!any_target_found) continue;

                // Determine the "join" label: look for a label that all case bodies
                // eventually goto (the continuation point after the switch).
                // For each target label, check what its trailing goto points to.
                std::unordered_map< std::string, int > join_candidates;
                for (auto &cg : case_gotos) {
                    auto it_cg = label_map.find(cg.goto_target);
                    if (it_cg == label_map.end()) continue;
                    size_t target_pos = it_cg->second;
                    auto *target_node = seq->children()[target_pos];
                    auto *target_lbl = target_node->dyn_cast< SLabel >();
                    if (!target_lbl || !target_lbl->body()) continue;

                    // Walk the body to find trailing goto
                    std::function< std::string(SNode *) > findTrailingGoto;
                    findTrailingGoto = [&](SNode *n) -> std::string {
                        if (!n) return "";
                        if (auto *g = n->dyn_cast< SGoto >()) return std::string(g->target());
                        if (auto *s = n->dyn_cast< SSeq >()) {
                            if (!s->empty()) return findTrailingGoto(s->children().back());
                        }
                        if (auto *b = n->dyn_cast< SBlock >()) {
                            if (!b->empty()) {
                                auto *last = b->stmts().back();
                                if (auto *g = llvm::dyn_cast< clang::GotoStmt >(last)) {
                                    return g->getLabel()->getName().str();
                                }
                            }
                        }
                        return "";
                    };
                    auto join = findTrailingGoto(target_lbl->body());
                    if (!join.empty()) {
                        join_candidates[join]++;
                    }
                }

                // Find the most common join target
                std::string join_label;
                int max_count = 0;
                for (auto &[label, count] : join_candidates) {
                    if (count > max_count) {
                        max_count = count;
                        join_label = label;
                    }
                }

                // Inline each case body and track inlined labels for removal
                bool any_inlined = false;
                std::unordered_set< std::string > inlined_labels;
                for (auto &cg : case_gotos) {
                    auto it_target = label_map.find(cg.goto_target);
                    if (it_target == label_map.end()) continue;
                    size_t target_pos = it_target->second;
                    auto *target_lbl = seq->children()[target_pos]->dyn_cast< SLabel >();
                    if (!target_lbl) continue;

                    // Collect statements from the label body
                    std::vector< clang::Stmt * > body_stmts;
                    collectStmtsFromSNode(target_lbl->body(), body_stmts);

                    // Replace trailing goto-to-join with break
                    if (!join_label.empty()) {
                        replaceTrailingGotoWithBreak(ctx, body_stmts,
                                                     llvm::StringRef(join_label));
                    }

                    // Add a BreakStmt at the end if there isn't one already
                    if (!body_stmts.empty()) {
                        if (!llvm::isa< clang::BreakStmt >(body_stmts.back()) &&
                            !llvm::isa< clang::ReturnStmt >(body_stmts.back()) &&
                            !llvm::isa< clang::ContinueStmt >(body_stmts.back())) {
                            body_stmts.push_back(
                                new (ctx) clang::BreakStmt(clang::SourceLocation())
                            );
                        }
                    }

                    // Build compound stmt for the case body
                    auto *compound = detail::makeCompound(ctx, body_stmts);
                    if (auto *cs = llvm::dyn_cast< clang::CaseStmt >(cg.case_stmt)) {
                        cs->setSubStmt(compound);
                    } else if (auto *ds = llvm::dyn_cast< clang::DefaultStmt >(cg.case_stmt)) {
                        ds->setSubStmt(compound);
                    }
                    inlined_labels.insert(cg.goto_target);
                    any_inlined = true;
                }

                if (any_inlined) {
                    // Collect positions of inlined labels for range-based removal
                    std::unordered_set< size_t > remove_indices;
                    label_map = buildFullLabelMap(seq); // rebuild after potential changes
                    for (auto &lab : inlined_labels) {
                        auto it2 = label_map.find(lab);
                        if (it2 != label_map.end()) {
                            remove_indices.insert(it2->second);
                        }
                    }
                    // Also mark SGoto nodes targeting join_label that sit
                    // immediately after an inlined label (trailing gotos)
                    for (size_t idx = 0; idx < seq->size(); ++idx) {
                        auto *g = seq->children()[idx]->dyn_cast< SGoto >();
                        if (g && !join_label.empty() &&
                            std::string(g->target()) == join_label) {
                            // Only remove if previous node was an inlined label
                            if (idx > 0 && remove_indices.count(idx - 1)) {
                                remove_indices.insert(idx);
                            }
                        }
                    }
                    // Remove in reverse order
                    std::vector< size_t > sorted_indices(remove_indices.begin(),
                                                          remove_indices.end());
                    std::sort(sorted_indices.rbegin(), sorted_indices.rend());
                    for (size_t idx : sorted_indices) {
                        if (idx < seq->size()) {
                            seq->removeChild(idx);
                        }
                    }
                    changed = true;
                }
            }

            return changed;
        });
    }

    // ========================================================================
    // IndirectGotoSwitch Pass
    // Converts GNU computed gotos (goto *tbl[sel]) with address-of-label
    // tables into switch statements.
    // ========================================================================

    namespace {
        // Find IndirectGotoStmt inside an SBlock
        clang::IndirectGotoStmt *findIndirectGoto(SBlock *blk) {
            for (auto *s : blk->stmts()) {
                if (auto *ig = llvm::dyn_cast< clang::IndirectGotoStmt >(s)) {
                    return ig;
                }
            }
            return nullptr;
        }

        // Try to extract the address-of-label table and index from an IndirectGotoStmt.
        // Expects: goto *tbl[sel] where tbl is an array of &&label
        struct IndirectGotoTableInfo {
            clang::Expr *index_expr = nullptr;
            std::vector< std::string > label_names;
            clang::VarDecl *table_var = nullptr;
        };

        std::optional< IndirectGotoTableInfo >
        analyzeIndirectGoto(clang::IndirectGotoStmt *ig) {
            // The target expression is usually a deref of an array subscript:
            // *tbl[sel] or tbl[sel] (since tbl elements are void*)
            auto *target = ig->getTarget();
            if (!target) return std::nullopt;

            // Strip implicit casts
            target = target->IgnoreParenImpCasts();

            // Look for ArraySubscriptExpr
            auto *subscript = llvm::dyn_cast< clang::ArraySubscriptExpr >(target);
            if (!subscript) return std::nullopt;

            auto *base = subscript->getBase()->IgnoreParenImpCasts();
            auto *idx = subscript->getIdx()->IgnoreParenImpCasts();

            // Base should be a DeclRefExpr to the table variable
            auto *base_ref = llvm::dyn_cast< clang::DeclRefExpr >(base);
            if (!base_ref) return std::nullopt;

            auto *var = llvm::dyn_cast< clang::VarDecl >(base_ref->getDecl());
            if (!var || !var->hasInit()) return std::nullopt;

            // The initializer should be an InitListExpr with AddrLabelExpr entries
            auto *init = var->getInit()->IgnoreParenImpCasts();
            auto *init_list = llvm::dyn_cast< clang::InitListExpr >(init);
            if (!init_list) return std::nullopt;

            IndirectGotoTableInfo info;
            info.index_expr = idx;
            info.table_var = var;

            for (unsigned j = 0; j < init_list->getNumInits(); ++j) {
                auto *init_elem = init_list->getInit(j)->IgnoreParenImpCasts();
                auto *addr_label = llvm::dyn_cast< clang::AddrLabelExpr >(init_elem);
                if (!addr_label) return std::nullopt;
                info.label_names.push_back(addr_label->getLabel()->getName().str());
            }

            return info;
        }
    } // namespace

    bool IndirectGotoSwitchPass::run(SNode *root, SNodeFactory &factory,
                                      clang::ASTContext &ctx) {
        return transformSeqs(root, factory, [&](SSeq *seq, SNodeFactory &fac) -> bool {
            bool changed = false;

            for (size_t i = 0; i < seq->size(); ++i) {
                // Look for SBlock or SLabel(SBlock) containing IndirectGotoStmt
                SBlock *blk = seq->children()[i]->dyn_cast< SBlock >();
                if (!blk) {
                    auto *lbl = seq->children()[i]->dyn_cast< SLabel >();
                    if (lbl) {
                        blk = lbl->body() ? lbl->body()->dyn_cast< SBlock >() : nullptr;
                    }
                }
                if (!blk) continue;

                auto *ig = findIndirectGoto(blk);
                if (!ig) continue;

                auto info = analyzeIndirectGoto(ig);
                if (!info) continue;

                // Found the pattern. Build an SSwitch.
                auto *sw = fac.make< SSwitch >(info->index_expr);

                for (size_t j = 0; j < info->label_names.size(); ++j) {
                    auto *case_val = detail::makeIntLiteral(
                        ctx, j, ctx.IntTy, clang::SourceLocation()
                    );
                    auto *case_goto = fac.make< SGoto >(
                        fac.intern(info->label_names[j])
                    );
                    sw->addCase(case_val, case_goto);
                }

                // Check if there's a bounds-check before: if(sel >= N) goto default
                // Look at the previous sibling
                SIfThenElse *bounds_check = nullptr;
                size_t bounds_idx = i;
                if (i > 0) {
                    bounds_check = seq->children()[i - 1]->dyn_cast< SIfThenElse >();
                    if (bounds_check) {
                        auto *then_goto = bounds_check->thenBranch()
                            ? bounds_check->thenBranch()->dyn_cast< SGoto >()
                            : nullptr;
                        if (then_goto) {
                            // Add default case targeting the bounds-check goto target
                            sw->setDefaultBody(
                                fac.make< SGoto >(then_goto->target())
                            );
                            bounds_idx = i - 1;
                        } else {
                            bounds_check = nullptr;
                        }
                    }
                }

                // Also remove the IndirectGotoStmt from the block, replace the
                // SBlock+bounds-check range with the SSwitch
                // Remove the indirect goto from the block's stmts
                auto &stmts = blk->stmts();
                stmts.erase(
                    std::remove_if(stmts.begin(), stmts.end(),
                        [ig](clang::Stmt *s) { return s == ig; }),
                    stmts.end()
                );

                if (blk->empty()) {
                    // Replace the entire range [bounds_idx, i+1) with switch
                    seq->replaceRange(bounds_idx, i + 1, sw);
                } else {
                    // Keep the block (has other stmts), insert switch after
                    seq->replaceChild(i, sw);
                    if (bounds_check) {
                        seq->removeChild(bounds_idx);
                        // i shifted by one since we removed bounds_idx before i
                    }
                }
                changed = true;
            }

            return changed;
        });
    }

    // ========================================================================
    // Step 16: Irreducible Handling Pass
    // ========================================================================

    bool IrreducibleHandlingPass::run(SNode *root, SNodeFactory & /*factory*/,
                                       clang::ASTContext & /*ctx*/) {
        // Count all gotos by walking the tree
        std::function< unsigned(const SNode *) > countGotos = [&](const SNode *n) -> unsigned {
            if (!n) return 0;
            if (n->isa< SGoto >()) return 1;
            unsigned count = 0;
            switch (n->kind()) {
            case SNodeKind::Seq: {
                auto *seq = n->as< SSeq >();
                for (auto *child : seq->children()) count += countGotos(child);
                break;
            }
            case SNodeKind::IfThenElse: {
                auto *ite = n->as< SIfThenElse >();
                count += countGotos(ite->thenBranch());
                count += countGotos(ite->elseBranch());
                break;
            }
            case SNodeKind::While:
                count += countGotos(n->as< SWhile >()->body());
                break;
            case SNodeKind::DoWhile:
                count += countGotos(n->as< SDoWhile >()->body());
                break;
            case SNodeKind::For:
                count += countGotos(n->as< SFor >()->body());
                break;
            case SNodeKind::Switch: {
                auto *sw = n->as< SSwitch >();
                for (auto &c : sw->cases()) count += countGotos(c.body);
                count += countGotos(sw->defaultBody());
                break;
            }
            case SNodeKind::Label:
                count += countGotos(n->as< SLabel >()->body());
                break;
            default: break;
            }
            return count;
        };

        unsigned goto_count = countGotos(root);
        if (goto_count > 0) {
            LOG(WARNING) << "Irreducible handling: " << goto_count
                         << " gotos remain after structural analysis\n";
        }

        return false; // No modifications
    }

    // ========================================================================
    // Step 18: Cleanup Pass
    // ========================================================================

    bool CleanupPass::run(SNode *root, SNodeFactory &factory,
                           clang::ASTContext & /*ctx*/) {
        return transformSeqs(root, factory, [&](SSeq *seq, SNodeFactory & /*fac*/) -> bool {
            bool changed = false;

            // 1. Remove empty SBlocks
            for (size_t i = 0; i < seq->size(); ) {
                auto *blk = seq->children()[i]->dyn_cast< SBlock >();
                if (blk && blk->empty() && blk->label().empty()) {
                    seq->removeChild(i);
                    changed = true;
                    continue;
                }
                ++i;
            }

            // 2. Else-if chaining: if { } else { if { } } → if { } else if { }
            for (size_t i = 0; i < seq->size(); ++i) {
                auto *ite = seq->children()[i]->dyn_cast< SIfThenElse >();
                if (!ite || !ite->elseBranch()) continue;

                auto *else_seq = ite->elseBranch()->dyn_cast< SSeq >();
                if (else_seq && else_seq->size() == 1) {
                    auto *inner_if = else_seq->children()[0]->dyn_cast< SIfThenElse >();
                    if (inner_if) {
                        ite->setElseBranch(inner_if);
                        changed = true;
                    }
                }
            }

            // 3. Remove empty else branches
            for (size_t i = 0; i < seq->size(); ++i) {
                auto *ite = seq->children()[i]->dyn_cast< SIfThenElse >();
                if (!ite || !ite->elseBranch()) continue;

                auto *else_seq = ite->elseBranch()->dyn_cast< SSeq >();
                if (else_seq && else_seq->empty()) {
                    ite->setElseBranch(nullptr);
                    changed = true;
                }
                auto *else_blk = ite->elseBranch()
                    ? ite->elseBranch()->dyn_cast< SBlock >() : nullptr;
                if (else_blk && else_blk->empty()) {
                    ite->setElseBranch(nullptr);
                    changed = true;
                }
            }

            // 4. Degenerate loop unwrap: while(true) { ...; break; } → { ... }
            for (size_t i = 0; i < seq->size(); ++i) {
                auto *w = seq->children()[i]->dyn_cast< SWhile >();
                if (!w) continue;

                // Check if condition is always true
                if (!detail::isAlwaysTrue(w->cond())) continue;

                auto *body_seq = w->body() ? w->body()->dyn_cast< SSeq >() : nullptr;
                if (!body_seq || body_seq->empty()) continue;

                // Check if last statement is a break
                auto *last = body_seq->children()[body_seq->size() - 1];
                if (last->isa< SBreak >()) {
                    // Remove the break and replace while with body
                    body_seq->removeChild(body_seq->size() - 1);
                    seq->replaceChild(i, body_seq);
                    changed = true;
                }
            }

            // 4b. Degenerate do-while unwrap:
            // do { body } while (false) → body
            // do { } while (false) → remove
            for (size_t i = 0; i < seq->size(); ) {
                auto *dw = seq->children()[i]->dyn_cast< SDoWhile >();
                if (!dw || !detail::isAlwaysFalse(dw->cond())) {
                    ++i;
                    continue;
                }

                // Body executes exactly once — unwrap it
                if (dw->body()) {
                    auto *body_seq = dw->body()->dyn_cast< SSeq >();
                    if (body_seq && body_seq->empty()) {
                        seq->removeChild(i);
                    } else {
                        seq->replaceChild(i, dw->body());
                        ++i;
                    }
                } else {
                    seq->removeChild(i);
                }
                changed = true;
            }

            // 5. Dead label removal
            for (size_t i = 0; i < seq->size(); ) {
                auto *lbl = seq->children()[i]->dyn_cast< SLabel >();
                if (lbl && countGotoRefs(root, lbl->name()) == 0) {
                    if (lbl->body()) {
                        seq->replaceChild(i, lbl->body());
                    } else {
                        seq->removeChild(i);
                    }
                    changed = true;
                    continue;
                }
                ++i;
            }

            // 6. Trailing goto elimination (goto L where L is next)
            for (size_t i = 0; i + 1 < seq->size(); ) {
                auto *g = seq->children()[i]->dyn_cast< SGoto >();
                if (g) {
                    auto *next_label = seq->children()[i + 1]->dyn_cast< SLabel >();
                    if (next_label && g->target() == next_label->name()) {
                        seq->removeChild(i);
                        changed = true;
                        continue;
                    }
                }
                ++i;
            }

            // 7. Dead code after unconditional terminators (return/break/continue)
            for (size_t i = 0; i + 1 < seq->size(); ) {
                auto *node = seq->children()[i];
                bool is_terminator = node->isa< SReturn >() ||
                                     node->isa< SBreak >() ||
                                     node->isa< SContinue >();
                // Also check SBlock ending with return/break
                if (!is_terminator) {
                    if (auto *blk = node->dyn_cast< SBlock >()) {
                        if (!blk->empty()) {
                            auto *last = blk->stmts().back();
                            is_terminator = llvm::isa< clang::ReturnStmt >(last) ||
                                            llvm::isa< clang::BreakStmt >(last) ||
                                            llvm::isa< clang::ContinueStmt >(last);
                        }
                    }
                }
                if (is_terminator) {
                    // Remove everything after this until end or next label with refs
                    bool removed_any = false;
                    while (i + 1 < seq->size()) {
                        auto *next = seq->children()[i + 1];
                        // Don't remove labels that have goto references
                        if (auto *lbl = next->dyn_cast< SLabel >()) {
                            if (countGotoRefs(root, lbl->name()) > 0) break;
                        }
                        seq->removeChild(i + 1);
                        removed_any = true;
                    }
                    if (removed_any) changed = true;
                }
                ++i;
            }

            return changed;
        });
    }

} // namespace patchestry::ast
