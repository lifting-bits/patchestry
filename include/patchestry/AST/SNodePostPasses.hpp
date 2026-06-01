/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/CGraph.hpp>
#include <patchestry/AST/SNode.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace clang { class ASTContext; }

namespace patchestry::ast {

    /// Lift raw clang control flow embedded in opaque SStmt leaves into
    /// first-class SNodes so later SNode repair passes can see it.
    /// Unsupported clang statement shapes remain opaque, and goto/label
    /// names are preserved verbatim.
    void NormalizeRawControlFlow(std::vector< SNode * > &root,
                                 SNodeFactory &factory,
                                 clang::ASTContext &ctx);

    /// Eliminate gotos whose target label immediately follows in the
    /// same sibling sequence.  Chases through SLabel→SStmt nesting to
    /// find deeply buried gotos (DeepTrailingStmt pattern).
    bool EliminateGotoToNextLabel(std::vector< SNode * > &root,
                                  SNodeFactory &factory,
                                  clang::ASTContext &ctx);

    /// Absorb unreferenced SLabel siblings after if-then (no else) into
    /// the else branch.  Prevents spurious fallthrough from the false path
    /// into code that was goto-only in the original CFG.
    bool AbsorbFallthroughIntoElse(std::vector< SNode * > &root,
                                   SNodeFactory &factory);

    /// Convert if(cond) goto L; stmts; L: patterns into if(!cond) { stmts }
    /// within a sibling sequence.  Only fires when label L has a single
    /// goto reference and no intermediate SLabel nodes exist between the
    /// goto and its target.  Recurses into all nested SNode bodies.
    bool ScopeifyIfGotos(std::vector< SNode * > &root, SNodeFactory &factory,
                         clang::ASTContext &ctx);

    /// Cross-scope ifgoto absorption: fold
    ///     if (outer) { if (inner) goto L; goto L2; } ... L: body
    /// into
    ///     if (outer && !inner) goto L2; ... body
    /// Only fires when L has a single goto reference and the children
    /// between the outer-if and the L: label are label-free (no goto
    /// targets stranded by the absorption). Works on the residuals that
    /// ScopeifyIfGotos leaves behind (where the inner-goto sits in a
    /// nested if-then rather than at the outer-body tail).
    bool AbsorbCrossScopeIfGoto(std::vector< SNode * > &root,
                                SNodeFactory &factory,
                                clang::ASTContext &ctx);

    /// Post-structuring cleanup: inline residual goto-to-label pairs.
    /// When an SGoto's target SLabel is a sibling referenced only by that
    /// goto, replace the goto with the label's body and drop the label.
    /// Returns true if any inlining was performed.
    bool InlineResidualGotos(std::vector< SNode * > &root, SNodeFactory &factory);

    /// Replace goto-to-break/continue: when a trailing clang::GotoStmt or
    /// SGoto targets an enclosing loop's exit label (→ break) or header
    /// label (→ continue), replace it with the structured equivalent.
    ///
    /// Returns true if any replacement was performed.
    bool ConvertGotoToBreakContinue(std::vector< SNode * > &root,
                                    SNodeFactory &factory);

    /// Replace goto-to-return: when an SStmt holding a clang::GotoStmt
    /// (or an SGoto SNode) targets a label whose body ends with a
    /// clang::ReturnStmt, replace the goto with the label's return stmts.
    /// Also handles gotos inside clang::IfStmt arms held by SStmt nodes
    /// — the most common residual pattern.
    /// Dead labels are cleaned up by a subsequent InlineResidualGotos pass.
    ///
    /// Returns true if any replacement was performed.
    bool ConvertGotoToReturn(std::vector< SNode * > &root, SNodeFactory &factory,
                             clang::ASTContext &ctx);

    /// Redirect gotos through labels whose body is only `goto other_label`.
    /// This removes label pass-through chains (alias elimination) and is run
    /// early, before the regular goto cleanup.
    bool CollapsePassThroughLabels(std::vector< SNode * > &root,
                                   SNodeFactory &factory);

    /// Fold empty label wrappers onto the following sibling and simplify
    /// empty if/else shells: drop side-effect-free `if (c) {}` and flip
    /// `if (c) {} else { B }` into `if (!c) { B }`.
    bool SimplifyEmptyControlFlow(std::vector< SNode * > &root,
                                  clang::ASTContext &ctx);

    /// Merge adjacent, nested, and else-if guarded gotos that target the
    /// same label, preserving short-circuit condition order.
    bool MergeRedundantGotoGuards(std::vector< SNode * > &root,
                                  SNodeFactory &factory,
                                  clang::ASTContext &ctx);

    /// Duplicate small, side-effect-contained label targets into switch
    /// case arms that end in `SGoto L`, making the case bodies goto-free.
    /// Handles goto-into-another-switch by recursively cloning the inner
    /// switch.  Refuses to clone subtrees containing labels or loops so
    /// that goto/label pairing stays consistent.  Outbound gotos from
    /// the clone are allowed — they reference an already-live label.
    ///
    /// Returns true if any duplication was performed.
    bool DuplicateSwitchCaseTargets(std::vector< SNode * > &root,
                                    SNodeFactory &factory);

    /// Fold guarded fallthrough-to-join patterns:
    ///
    ///   if (A && B && C) goto body; else goto join;
    /// body:
    ///   ...
    /// join:
    ///
    /// into:
    ///
    ///   if (A && B && C) { ... }
    /// join:
    ///
    /// The pass only moves a single-ref immediately-following label body whose
    /// tail falls through to the next direct label, and only when all other
    /// guard exits already target that join label.
    bool FoldGuardedFallthroughTargets(std::vector< SNode * > &root,
                                       SNodeFactory &factory,
                                       clang::ASTContext &ctx);

    /// Repair one-way gotos that enter a label nested inside a later
    /// structured if/else region by duplicating a small label body at the
    /// outer guard site and moving the skipped region into the guard's else.
    ///
    /// This is intentionally conservative: it only handles single-ref labels
    /// nested under if/else nodes, not loops/switches, and only when the label
    /// is the last child along the nested path so the cloned entry has the
    /// same fallthrough continuation as the original label entry.
    bool RepairCrossScopeLabelEntries(std::vector< SNode * > &root,
                                      SNodeFactory &factory,
                                      clang::ASTContext &ctx);

    /// Repair sibling-arm label entries:
    ///
    ///   if (A) { L: body; } else { ...; if (B) goto L; }
    ///
    /// into:
    ///
    ///   if (A) { body; } else { ...; if (B) { body; } }
    ///
    /// Only fires for single-ref labels, terminal goto paths in the sibling
    /// arm, and small clone-safe tails.  It also sinks an identical trailing
    /// statement out of both arms before cloning the target prefix.  This
    /// covers shared cleanup blocks that Ghidra represents as a label in one
    /// if/else arm plus a goto from the other arm.
    bool FoldSiblingArmLabelEntries(std::vector< SNode * > &root,
                                    SNodeFactory &factory,
                                    clang::ASTContext &ctx);

    /// Move single-ref label targets into switch case arms.
    ///
    /// This complements DuplicateSwitchCaseTargets for switch-local labels that
    /// contain calls and therefore must not be cloned.  The target label must
    /// have exactly one goto reference, no fallthrough predecessor, a bounded
    /// label-free body, and every path through the moved body must terminate.
    bool FoldSwitchLocalCaseTargets(std::vector< SNode * > &root,
                                    SNodeFactory &factory,
                                    clang::ASTContext &ctx);

    /// Duplicate small, terminating, label-free target bodies at ordinary
    /// residual goto sites.  This is the general form of the switch-case target
    /// duplication rule: it clones only bounded assignment/control tails that
    /// always terminate and rejects labels, loops, calls, and break/continue
    /// whose target would depend on the original lexical scope.
    ///
    /// Returns true if any duplication was performed.
    bool DuplicateSmallTerminatingTargets(std::vector< SNode * > &root,
                                          SNodeFactory &factory);

    /// Split a label's owning sequence at the label and duplicate the
    /// terminating, goto-free entry suffix at residual goto sites.
    ///
    /// Unlike DuplicateSmallTerminatingTargets, this pass is allowed to copy
    /// call-containing statement tails when doing so preserves the original
    /// edge semantics: the copied suffix replaces the goto edge, all local
    /// live-ins must already be available at the goto site, and the original
    /// label remains for fallthrough paths until later dead-label cleanup.
    bool SplitAndCloneCrossScopeEntries(std::vector< SNode * > &root,
                                        SNodeFactory &factory,
                                        clang::ASTContext &ctx);

    /// Duplicate small producer->epilogue error tails at residual goto sites.
    /// A direct goto to the epilogue is cloned only when all local variables
    /// read by the epilogue are definitely assigned earlier in the same SNode
    /// sequence.  A goto to a producer label may clone the producer plus one
    /// terminal epilogue hop when the combined body is self-contained.
    bool DuplicateSmallEpilogueTargets(std::vector< SNode * > &root,
                                       SNodeFactory &factory,
                                       clang::ASTContext &ctx);

    /// Duplicate small switch-local fallthrough label tails into case arms.
    /// A clone is emitted as the label body plus an explicit SBreak, and only
    /// when the target is small, label-free, call-free, and has no local
    /// live-ins.  Rewrites are restricted to the same SSwitch so the cloned
    /// break targets the same structured scope as the original fallthrough.
    bool DuplicateSwitchFallthroughTargets(std::vector< SNode * > &root,
                                           SNodeFactory &factory,
                                           clang::ASTContext &ctx);

    /// Duplicate loop-local continue/latch label tails into goto sites.
    /// The pass only clones small, goto-free, label-free tails that end in
    /// continue, and only rewrites gotos inside the same loop body.  Nested
    /// loops are treated as separate continue scopes.
    bool DuplicateLoopContinueTargets(std::vector< SNode * > &root,
                                      SNodeFactory &factory,
                                      clang::ASTContext &ctx);

    /// Duplicate stack-protector return epilogues at residual goto sites.
    /// The cloned body may contain only stack_chk_fail-like calls, must
    /// terminate on every path, and is still gated by local live-in checks
    /// before being placed at a goto site.  Label wrappers are dropped from
    /// the clone so the pass does not duplicate label definitions.
    bool DuplicateStackGuardReturnTargets(std::vector< SNode * > &root,
                                          SNodeFactory &factory,
                                          clang::ASTContext &ctx);

    /// Duplicate small cleanup/logging return tails at residual goto sites.
    /// The pass follows bounded cleanup chains, drops cloned label/goto
    /// wrappers, and only clones direct calls that look like cleanup or
    /// diagnostic calls.  Local live-ins must be definitely available at
    /// the goto site.
    bool DuplicateCleanupReturnTargets(std::vector< SNode * > &root,
                                       SNodeFactory &factory,
                                       clang::ASTContext &ctx);

    /// Cross-scope version of InlineResidualGotos: when a goto's target
    /// label has exactly one reference, the label's body always terminates,
    /// and the label's preceding sibling also terminates (no fallthrough
    /// reaches it), move the body into the goto's slot and drop the label.
    /// Safe because no other goto observes the label, no fallthrough
    /// reaches it, and the moved body terminates.
    /// Returns true if any label was inlined.
    bool InlineCrossScopeSingleRef(std::vector< SNode * > &root,
                                   SNodeFactory &factory);

    /// Remove unreachable siblings that follow a terminating sibling.
    /// Preserves SLabel children (may be goto targets from other scopes).
    /// Bottom-up recursive.
    bool RemoveDeadSSeqChildren(std::vector< SNode * > &root);

    /// Final pre-emission repair fixed point.  This runs the conservative
    /// goto-repair passes one last time after the main structuring pipeline,
    /// so any region entry/exit patterns exposed late by earlier cleanup have
    /// a chance to become structured break/continue/return or cloned targets
    /// before lowering observes the tree.
    bool FinalizeRegionRepairs(std::vector< SNode * > &root,
                               SNodeFactory &factory,
                               clang::ASTContext &ctx);

    struct RegionRepairVerifierResult
    {
        size_t goto_refs                = 0;
        size_t same_scope_gotos         = 0;
        size_t outward_gotos            = 0;
        size_t cross_scope_entry_gotos  = 0;
        size_t unresolved_gotos         = 0;
        std::vector< std::string > diagnostics;

        bool hasFatalErrors() const { return unresolved_gotos != 0; }

        bool ok() const {
            return cross_scope_entry_gotos == 0 && unresolved_gotos == 0;
        }
    };

    /// Verify the SNode tree before lowering.  Every residual goto must
    /// resolve to a live label.  Cross-scope entries are counted separately:
    /// the repair pipeline should remove reducible cases, but irreducible
    /// residual gotos are still representable by the lowering path.
    RegionRepairVerifierResult
    VerifyRegionRepairedBeforeLowering(const std::vector< SNode * > &root);

} // namespace patchestry::ast
