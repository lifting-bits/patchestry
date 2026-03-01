/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// Orchestrates the full AST normalization pipeline by composing pass groups
// defined in NormalizationCfgPasses.cpp, NormalizationConditionalPasses.cpp,
// NormalizationLoopPasses.cpp, NormalizationSwitchPasses.cpp, and
// NormalizationCleanupPasses.cpp.

#include <patchestry/AST/ASTNormalizationPipeline.hpp>
#include <patchestry/AST/ASTPassManager.hpp>
#include <patchestry/Util/Log.hpp>

#include "NormalizationPipelineInternal.hpp"

namespace patchestry::ast {

    bool runASTNormalizationPipeline(
        clang::ASTContext &ctx, const patchestry::Options &options
    ) {
        ASTPassManager pass_manager;
        detail::PipelineState state;

        // Steps 1–2: extract CFG and reorder basic blocks to RPO.
        detail::addCfgPasses(pass_manager, state);

        // Step 3: prune unreachable dead code exposed by RPO ordering.
        detail::addDeadCfgPruningPass(pass_manager, state);

        // Steps 4–5: loop structuring.
        // WhileLoopStructurizePass must run before ConditionalStructurizePass.
        // RPO places the loop exit block at i+1 (immediately after the while-head),
        // so ConditionalStructurizePass would otherwise misinterpret the head's
        // IfStmt as a single-sided if.
        detail::addWhileLoopStructurizePass(pass_manager, state);
        detail::addLoopStructurizePass(pass_manager, state);

        // Step 6: strip CFG-block labels that have no incoming gotos.
        // These labels are inserted by CfgExtractPass as basic-block markers; any
        // that remain unreferenced after loop structuring would otherwise cause
        // containsLabelInRange to block single-sided-if structuring in the next step.
        detail::addAstCleanupPass(pass_manager, state);

        // Steps 7–13: conditional and switch structuring.
        detail::addConditionalStructurizePass(pass_manager, state);   // 7
        detail::addSwitchRecoveryPass(pass_manager, state);           // 8
        detail::addIfElseRegionFormationPass(pass_manager, state);    // 9
        detail::addIrreducibleFallbackPass(pass_manager, state);      // 10
        detail::addSwitchGotoInliningPass(pass_manager, state);       // 11
        // Step 11.5 — Control-equivalence hoisting.  Pipeline placement: run
        // after switch-goto inlining because that pass exposes new structure
        // (reduced gotos, clearer loop boundaries).  Catches loop-exit and
        // diamond patterns at the top level before DegenerateLoopUnwrap and
        // late loop recovery (steps 12–19) modify the CFG.  If we ran later,
        // some hoisting opportunities would be lost or require different
        // pattern matching.
        detail::addHoistControlEquivalentStmtsPass(pass_manager, state); // 11.5
        detail::addDegenerateLoopUnwrapPass(pass_manager, state);     // 12
        detail::addLoopConditionRecoveryPass(pass_manager, state);    // 13

        // Steps 14–16: dead-code / trailing-jump / AST cleanup round.
        detail::addDeadCfgPruningPass(pass_manager, state);           // 14
        detail::addTrailingJumpElimPass(pass_manager, state);         // 15
        detail::addAstCleanupPass(pass_manager, state);               // 16

        // Step 17: degenerate while elimination group.
        detail::addDegenerateWhileElimGroup(pass_manager, state);

        // Step 18: backedge loop structuring.
        detail::addBackedgeLoopStructurizePass(pass_manager, state);

        // Step 19: recover natural loops after backedge structuring.
        detail::addNaturalLoopRecoveryPass(pass_manager, state);

        // Step 20: nested diamonds in late loops.
        detail::addConditionalStructurizePass(pass_manager, state);

        // Step 20.5 — Second control-equivalence hoisting round.  Pipeline
        // placement: backedge loop structuring (step 18) and nested diamond
        // structuring (step 20) create new control-flow structure.  Regions
        // that were not control-equivalent before may become so after these
        // passes.  Running HoistControlEquivalentStmtsPass again catches
        // patterns that were invisible in the first round (step 11.5).
        detail::addHoistControlEquivalentStmtsPass(pass_manager, state); // 20.5

        // Step 21: cleanup after late structuring.
        detail::addAstCleanupPass(pass_manager, state);

        // Step 22: upgrade while loops to for loops.
        detail::addWhileToForUpgradePass(pass_manager, state);

        // Step 23: goto canonicalization.
        detail::addGotoCanonicalizePass(pass_manager, state);

        // Step 24: final cleanup.
        detail::addAstCleanupPass(pass_manager, state);

        // Step 25: verification — report any remaining gotos.
        detail::addNoGotoVerificationPass(pass_manager, state);
        return pass_manager.run(ctx, options);
    }

} // namespace patchestry::ast
