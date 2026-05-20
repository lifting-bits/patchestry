/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/ClangEmitter.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Util/Log.hpp>

#include "ClangEmitterPostPassesInternal.hpp"

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
#include <llvm/ADT/FoldingSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

namespace patchestry::ast {

    void CleanupPrettyPrint(clang::FunctionDecl *fn, clang::ASTContext &ctx) {
        if (!fn || !fn->hasBody()) { return; }
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
        // EliminateGotoToNextLabel always returns a non-null Stmt* on
        // non-null input (it rebuilds the CompoundStmt unconditionally),
        // so we can't use nullptr as a fixed-point sentinel.  Detect
        // convergence via Stmt::Profile, matching the outer schedule
        // loop below.  Without this check the loop would run the full
        // kMaxGotoEliminationPasses on every function regardless of
        // whether work remained — wasted cycles, and a cap-hit warning
        // that fires for everyone.
        auto run_goto_to_next_label_fixed_point = [&]() {
            bool reached_fixed_point = false;
            for (int pass = 0; pass < kMaxGotoEliminationPasses; ++pass) {
                llvm::FoldingSetNodeID before;
                fn->getBody()->Profile(before, ctx, /*Canonical=*/false);

                std::unordered_set< clang::LabelDecl * > goto_targets;
                std::unordered_set< clang::Stmt * > seen;
                CollectGotoTargets(fn->getBody(), goto_targets, seen);
                body = EliminateGotoToNextLabel(ctx, fn->getBody(), &goto_targets);
                if (body) { fn->setBody(body); }

                llvm::FoldingSetNodeID after;
                fn->getBody()->Profile(after, ctx, /*Canonical=*/false);
                if (before == after) {
                    reached_fixed_point = true;
                    break;
                }
            }
            // Mirror the SNode-cleanup cap-hit diagnostic at
            // ASTConsumer.cpp:308-333.  With the Profile-based check
            // above, this only fires on genuine pass oscillation.
            if (!reached_fixed_point) {
                LOG(WARNING)
                    << "Clang-AST goto-to-next-label fixed-point hit cap of "
                    << kMaxGotoEliminationPasses << " passes for "
                    << fn->getNameAsString()
                    << " — possible pass oscillation\n";
            }
        };
        auto run_goto_to_next_label_once = [&]() {
            std::unordered_set< clang::LabelDecl * > goto_targets;
            std::unordered_set< clang::Stmt * > seen;
            CollectGotoTargets(fn->getBody(), goto_targets, seen);
            body = EliminateGotoToNextLabel(ctx, fn->getBody(), &goto_targets);
            if (body) { fn->setBody(body); }
        };
        // Consolidated dead-control-flow cleanup (strip dead
        // labels, drop orphaned gotos, remove empty blocks).
        auto run_dead_control_flow = [&]() {
            bool mutated = false;
            apply_body(RemoveDeadControlFlow(ctx, fn->getBody(), mutated));
            (void) mutated;
        };
        // Dead-label sweep only.  A few legacy tail positions run this
        // without the empty-block merge: running RemoveEmptyBlocks there
        // reshapes if/else ahead of downstream goto-elimination passes
        // and regresses delicate fixtures (cve_2016_6563_fun_0000b920).
        // Folds back into run_dead_control_flow once Phase 4 removes the
        // downstream order-dependence.
        auto run_remove_dead_labels = [&]() {
            std::unordered_set< clang::LabelDecl * > goto_targets;
            std::unordered_set< clang::Stmt * > seen;
            CollectGotoTargets(fn->getBody(), goto_targets, seen);
            apply_body(RemoveDeadLabels(ctx, fn->getBody(), goto_targets));
        };
        auto run_late_join_fixups = [&]() {
            run_goto_to_next_label_fixed_point();
            run_dead_control_flow();
        };

        // Eliminate gotos to immediately following labels.  Iterates
        // to handle cascading patterns.
        run_goto_to_next_label_fixed_point();

        // Scope creation + goto-to-next-label cascade.  ScopeifyIfGotos
        // converts if(c) goto L; stmts; L: → if(!c) { stmts; }, which
        // may create new goto-to-next-label adjacencies, so iterate.
        //
        // Each schedule lambda discards its `mutated` flag — the outer
        // fixed-point loop below uses Stmt::Profile to detect
        // structural convergence across the whole schedule, so a
        // per-pass mutation flag is redundant here.  Single-shot
        // callers (further down) DO gate body reassignment on
        // `mutated` to avoid pointer-churn when the pass is a no-op.
        std::vector< std::function< void() > > fixed_point_cleanup_schedule = {
            [&]() {
                bool mutated = false;
                apply_body(HoistCrossScopeLabels(ctx, fn, fn->getBody(), mutated));
                (void) mutated;
            },
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
            [&]() {
                bool mutated = false;
                apply_body(RecoverLoop(ctx, fn->getBody(), mutated));
                (void) mutated;
            },
            [&]() {
                bool mutated = false;
                apply_body(FoldGotoDiamonds(ctx, fn->getBody(), mutated));
                (void) mutated;
            },
            [&]() {
                bool mutated = false;
                apply_body(CloneTerminalLabelGotos(ctx, fn->getBody(), mutated));
                (void) mutated;
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
                bool mutated = false;
                apply_body(CloneTerminalLabelGotos(ctx, fn->getBody(), mutated));
                (void) mutated;
            },
            [&]() { run_goto_to_next_label_once(); },
            [&]() {
                run_with_refs([&](const auto &refs) {
                    return FoldClangSwitchLocalCaseTargets(ctx, fn->getBody(), refs);
                });
            },
        };
        // Phase 4 — real fixed point for the cleanup schedule: iterate until a
        // full schedule pass leaves the body structurally unchanged
        // (Stmt::Profile), bounded by kMaxGotoEliminationPasses against
        // pathological oscillation.  Replaces a body-pointer-identity check
        // that never converged — every always-rebuild pass bumped the pointer,
        // so the loop always ran the full cap.  Output-identical: a schedule
        // pass reporting no structural change is a true fixed point, and the
        // schedule is deterministic, so any further passes are no-ops.
        bool schedule_reached_fixed_point = false;
        for (int pass = 0; pass < kMaxGotoEliminationPasses; ++pass) {
            llvm::FoldingSetNodeID before;
            fn->getBody()->Profile(before, ctx, /*Canonical=*/false);
            for (const auto &step : fixed_point_cleanup_schedule) { step(); }
            llvm::FoldingSetNodeID after;
            fn->getBody()->Profile(after, ctx, /*Canonical=*/false);
            if (before == after) {
                schedule_reached_fixed_point = true;
                break;
            }
        }
        if (!schedule_reached_fixed_point) {
            LOG(WARNING)
                << "Clang-AST cleanup schedule hit cap of "
                << kMaxGotoEliminationPasses << " iterations for "
                << fn->getNameAsString()
                << " — possible pass oscillation\n";
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
        bool inlined_single_ref = false;
        body = InlineSingleRefTerminalLabelBlocks(ctx, fn->getBody(), refs, inlined_single_ref);
        if (inlined_single_ref && body) { fn->setBody(body); }

        // Strip dead labels, orphaned gotos (targets absorbed by
        // structuring rules), and empty CompoundStmts/NullStmts.
        run_dead_control_flow();

        bool hoisted_cross_scope = false;
        body = HoistCrossScopeLabels(ctx, fn, fn->getBody(), hoisted_cross_scope);
        if (hoisted_cross_scope && body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = FoldConditionalFallthroughChains(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }
        bool recovered_loop = false;
        body = RecoverLoop(ctx, fn->getBody(), recovered_loop);
        if (body) { fn->setBody(body); }
        bool folded_diamonds = false;
        body = FoldGotoDiamonds(ctx, fn->getBody(), folded_diamonds);
        if (body) { fn->setBody(body); }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        body = FoldCrossCompoundDispatchChains(ctx, fn->getBody(), refs);
        if (body) { fn->setBody(body); }

        bool cloned_terminal = false;
        body = CloneTerminalLabelGotos(ctx, fn->getBody(), cloned_terminal);
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

        bool cloned_terminal_late = false;
        body = CloneTerminalLabelGotos(ctx, fn->getBody(), cloned_terminal_late);
        if (body) { fn->setBody(body); }

        run_goto_to_next_label_fixed_point();
        run_dead_control_flow();

        bool promoted_counter_for = false;
        body = PromoteSimpleCounterWhileToFor(ctx, fn->getBody(), promoted_counter_for);
        if (promoted_counter_for && body) { fn->setBody(body); }

        bool removed_terminal_continue = false;
        body = RemoveRedundantTerminalForContinues(
            ctx, fn->getBody(), removed_terminal_continue
        );
        if (removed_terminal_continue && body) { fn->setBody(body); }

        run_dead_control_flow();

        bool pushed_labels = false;
        body = PushLabelsIntoCompounds(ctx, fn->getBody(), pushed_labels);
        if (pushed_labels && body) { fn->setBody(body); }

        bool attached_empty_labels = false;
        body = AttachEmptyLabelsToFollowingStmt(ctx, fn->getBody(), attached_empty_labels);
        if (attached_empty_labels && body) { fn->setBody(body); }

        bool folded_diamonds_late = false;
        body = FoldGotoDiamonds(ctx, fn->getBody(), folded_diamonds_late);
        if (body) { fn->setBody(body); }

        if (!ContainsSwitchStmt(fn->getBody())) {
            refs.clear();
            CountGotoDeclRefs(fn->getBody(), refs);
            body = CloneFallthroughTerminalLabelGotos(ctx, fn->getBody(), refs);
            if (body) { fn->setBody(body); }

            run_dead_control_flow();
        }

        refs.clear();
        CountGotoDeclRefs(fn->getBody(), refs);
        bool folded_guarded_join = false;
        body = FoldGuardedJoinLabelChains(ctx, fn->getBody(), refs, folded_guarded_join);
        if (folded_guarded_join && body) { fn->setBody(body); }

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
        if (cloned_cleanup_join && body) { fn->setBody(body); }

        if (cloned_cleanup_join) {
            run_late_join_fixups();
        }

        // Cosmetic: fold double negations and `!(a OP b)` comparisons in
        // if/while/do/for conditions.  Runs last — purely a readability
        // pass, no effect on goto/label structure.
        NormalizeConditions(ctx, fn->getBody());
    }

} // namespace patchestry::ast
