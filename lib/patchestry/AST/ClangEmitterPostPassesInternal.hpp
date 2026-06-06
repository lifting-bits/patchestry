/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

// Internal header for the post-emission Clang-AST cleanup layer.
//
// The transform passes live in ClangEmitterPostPasses.cpp; the driver
// `CleanupPrettyPrint` lives in ClangEmitterCleanup.cpp.  This header
// declares the pass/helper functions the driver calls across that file
// boundary.  It is included only by those two translation units.

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>

namespace patchestry::ast {

    // --- prologue -----------------------------------------------------------
    clang::Stmt *CleanupStmtTree(
        clang::ASTContext &ctx, clang::Stmt *s, const std::string &continue_label = ""
    );

    // --- shared analysis helpers --------------------------------------------
    void CollectGotoTargets(
        clang::Stmt *s, std::unordered_set< clang::LabelDecl * > &targets,
        std::unordered_set< clang::Stmt * > &seen
    );
    void CountGotoDeclRefs(
        clang::Stmt *stmt, std::unordered_map< clang::LabelDecl *, unsigned > &refs
    );
    void CollectDefinedLabels(
        clang::Stmt *s, std::unordered_set< clang::LabelDecl * > &defined
    );
    std::unordered_set< clang::LabelDecl * > CollectCrossScopeGotoTargets(clang::Stmt *stmt);
    bool ContainsSwitchStmt(clang::Stmt *st);

    // --- inline a terminal label's payload to its goto site ----------------
    clang::Stmt *CloneTerminalLabelGotos(
        clang::ASTContext &ctx, clang::Stmt *body, bool &mutated
    );
    clang::Stmt *CloneFallthroughTerminalLabelGotos(
        clang::ASTContext &ctx, clang::Stmt *stmt,
        const std::unordered_map< clang::LabelDecl *, unsigned > &refs
    );
    clang::Stmt *InlineSingleRefTerminalLabelBlocks(
        clang::ASTContext &ctx, clang::Stmt *stmt,
        const std::unordered_map< clang::LabelDecl *, unsigned > &refs, bool &mutated
    );
    clang::Stmt *CloneSmallStraightLineLabelBeforeJoinGotos(
        clang::ASTContext &ctx, clang::Stmt *stmt,
        const std::unordered_map< clang::LabelDecl *, unsigned > &refs, bool &changed
    );
    clang::Stmt *CloneCleanupLabelBeforeJoinGotos(
        clang::ASTContext &ctx, clang::Stmt *stmt,
        const std::unordered_map< clang::LabelDecl *, unsigned > &refs, bool &changed
    );
    clang::Stmt *FoldSmallCrossScopeGotoTargets(
        clang::ASTContext &ctx, clang::FunctionDecl *fn, clang::Stmt *body, bool &mutated
    );

    // --- fold goto forwarders / diamonds ------------------------------------
    clang::Stmt *FoldGotoDiamonds(clang::ASTContext &ctx, clang::Stmt *body, bool &mutated);
    clang::Stmt *FoldConditionalFallthroughChains(
        clang::ASTContext &ctx, clang::Stmt *stmt,
        const std::unordered_map< clang::LabelDecl *, unsigned > &refs
    );
    clang::Stmt *FoldForwardSingleRefLabelRegions(
        clang::ASTContext &ctx, clang::Stmt *stmt,
        const std::unordered_map< clang::LabelDecl *, unsigned > &refs
    );
    clang::Stmt *SinkCommonTerminalEpilogues(
        clang::ASTContext &ctx, clang::Stmt *stmt,
        const std::unordered_map< clang::LabelDecl *, unsigned > &refs
    );
    clang::Stmt *FoldCrossCompoundDispatchChains(
        clang::ASTContext &ctx, clang::Stmt *stmt,
        const std::unordered_map< clang::LabelDecl *, unsigned > &refs
    );
    clang::Stmt *FoldGuardedJoinLabelChains(
        clang::ASTContext &ctx, clang::Stmt *stmt,
        const std::unordered_map< clang::LabelDecl *, unsigned > &refs, bool &changed
    );

    // --- eliminate goto to the immediately-following label -----------------
    clang::Stmt *EliminateGotoToNextLabel(
        clang::ASTContext &ctx, clang::Stmt *s,
        const std::unordered_set< clang::LabelDecl * > *live
    );

    // --- remove dead control flow -------------------------------------------
    // Collapse clone-induced duplicate same-named labels (re-point gotos onto
    // the canonical LabelStmt, unwrap duplicates) so decl-keyed liveness can't
    // desync.  No-op when there are no duplicate names.
    clang::Stmt *DedupSameNameLabels(clang::ASTContext &ctx, clang::Stmt *body);
    clang::Stmt *RemoveDeadControlFlow(
        clang::ASTContext &ctx, clang::Stmt *body, bool &mutated
    );
    clang::Stmt *RemoveDeadLabels(
        clang::ASTContext &ctx, clang::Stmt *s,
        const std::unordered_set< clang::LabelDecl * > &live
    );
    clang::Stmt *RemoveEmptyBlocks(clang::ASTContext &ctx, clang::Stmt *s);
    // RemoveOrphanedGotos is internal to ClangEmitterPostPasses.cpp —
    // it's reached only via RemoveDeadControlFlow.  Kept file-local so
    // its default arguments (depth = 0, in_switch_case = false) are not
    // out of sync with a header declaration that lacks them.

    // --- scope-ify conditional gotos ----------------------------------------
    clang::Stmt *ScopeifyIfGotos(
        clang::ASTContext &ctx, clang::Stmt *s,
        const std::unordered_map< clang::LabelDecl *, unsigned > &refs
    );

    // --- hoist cross-scope label entries ------------------------------------
    clang::Stmt *HoistCrossScopeLabels(
        clang::ASTContext &ctx, clang::FunctionDecl *fn, clang::Stmt *body, bool &mutated
    );
    clang::Stmt *FoldScopeJoinIfElseChain(
        clang::ASTContext &ctx, clang::Stmt *body, bool &mutated
    );

    // --- recover loops ------------------------------------------------------
    clang::Stmt *RecoverLoop(clang::ASTContext &ctx, clang::Stmt *body, bool &mutated);

    // --- fold switch-case-target gotos --------------------------------------
    clang::Stmt *FoldClangSwitchLocalCaseTargets(
        clang::ASTContext &ctx, clang::Stmt *stmt,
        const std::unordered_map< clang::LabelDecl *, unsigned > &refs
    );

    // --- condition / loop / label readability passes ------------------------
    clang::Stmt *PromoteSimpleCounterWhileToFor(
        clang::ASTContext &ctx, clang::Stmt *stmt, bool &mutated
    );
    clang::Stmt *RemoveRedundantTerminalForContinues(
        clang::ASTContext &ctx, clang::Stmt *stmt, bool &mutated
    );
    clang::Stmt *PushLabelsIntoCompounds(
        clang::ASTContext &ctx, clang::Stmt *stmt, bool &mutated
    );
    clang::Stmt *AttachEmptyLabelsToFollowingStmt(
        clang::ASTContext &ctx, clang::Stmt *stmt, bool &mutated
    );
    void NormalizeConditions(clang::ASTContext &ctx, clang::Stmt *s);

} // namespace patchestry::ast
