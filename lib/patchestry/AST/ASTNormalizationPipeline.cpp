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
        detail::addCfgPasses(pass_manager, state);            // CfgExtract + BasicBlockReorder

        // Step 3: prune unreachable dead code exposed by RPO ordering.
        detail::addDeadCfgPruningPass(pass_manager, state);

        // Step 3.5: canonicalize gotos before loop/conditional structuring.
        detail::addGotoCanonicalizePass(pass_manager, state);

        // Steps 4–5: loop structuring.
        // WhileLoopStructurizePass must run before ConditionalStructurizePass.
        // RPO places the loop exit block at i+1 (immediately after the while-head),
        // so ConditionalStructurizePass would otherwise misinterpret the head's
        // IfStmt as a single-sided if.
        detail::addWhileLoopStructurizePass(pass_manager, state);
        detail::addLoopStructurizePass(pass_manager, state);

        // Step 5.5: strip CFG-block labels that have no incoming gotos.
        // These labels are inserted by CfgExtractPass as basic-block markers; any
        // that remain unreferenced after loop structuring would otherwise cause
        // containsLabelInRange to block single-sided-if structuring in the next step.
        detail::addAstCleanupPass(pass_manager, state);

        // Steps 6–7: conditional and switch structuring.
        detail::addConditionalStructurizePass(pass_manager, state);
        detail::addSwitchRecoveryPass(pass_manager, state);
        detail::addIfElseRegionFormationPass(pass_manager, state);
        detail::addIrreducibleFallbackPass(pass_manager, state);
        detail::addSwitchGotoInliningPass(pass_manager, state);
        // detail::addHoistControlEquivalentStmtsIntoLoopPass(pass_manager, state);
        detail::addDegenerateLoopUnwrapPass(pass_manager, state);
        detail::addLoopConditionRecoveryPass(pass_manager, state);

        //  Steps 17–19: another dead-code / trailing-jump / AST cleanup round.
        detail::addDeadCfgPruningPass(pass_manager, state);
        detail::addTrailingJumpElimPass(pass_manager, state);
        detail::addAstCleanupPass(pass_manager, state);

        // Steps 20–28: ast_improvements branch passes.
        detail::addDegenerateWhileElimPass(pass_manager, state);      // 20
        detail::addDeadCfgPruningPass(
            pass_manager, state
        ); // 20.5 – prune nulls from break replacement
        detail::addAstCleanupPass(
            pass_manager, state
        ); // 20.6 – strip empty null-stmt if-branches
        detail::addBackedgeLoopStructurizePass(pass_manager, state); // 21

        // Step 22: recover natural loops after backedge structuring.
        detail::addNaturalLoopRecoveryPass(pass_manager, state);      // 22
        detail::addAstCleanupPass(pass_manager, state);               // 23
        detail::addWhileToForUpgradePass(pass_manager, state);        // 24.75
        detail::addGotoCanonicalizePass(pass_manager, state);         // 25
        detail::addAstCleanupPass(pass_manager, state);               // 27

        // Final verification: report any remaining gotos.
        detail::addNoGotoVerificationPass(pass_manager, state);
        return pass_manager.run(ctx, options);
    }

} // namespace patchestry::ast
