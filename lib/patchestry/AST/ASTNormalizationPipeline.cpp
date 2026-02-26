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
        // NOTE: GotoCanonicalizePass is intentionally NOT run before
        // BasicBlockReorderPass.  In the original binary order, loop back-edges
        // (e.g. "goto Lhead" at the end of a body block) may appear as
        // trivially-adjacent forward jumps because the block order does not match
        // execution order.  Running GotoCanonicalizePass here would silently
        // remove those back-edge gotos, making WhileLoopStructurizePass unable to
        // detect the loop later.  BasicBlockReorderPass already runs
        // GotoCanonicalizePass internally after reordering to RPO, where genuine
        // back-edges are no longer adjacent to their targets and are therefore
        // preserved.
        detail::addCfgPasses(pass_manager, state);            // CfgExtract + BasicBlockReorder

        // Step 3: prune unreachable dead code exposed by RPO ordering.
        detail::addDeadCfgPruningPass(pass_manager, state);

        // Step 3.5: inline single-use temporaries before loop/conditional passes
        // so that loop condition recognizers see bare comparison expressions, not
        // wrapped temps.
        detail::addSingleUseTempInliningPass(pass_manager, state);
        detail::addGotoCanonicalizePass(pass_manager, state);
        detail::addIfElseRegionFormationPass(pass_manager, state);
        detail::addLoopPasses(pass_manager, state);

        // detail::addConditionalStructurizePass(pass_manager, state);
#if 0
        // Steps 4–5: loop structuring.
        // WhileLoopStructurizePass must run before ConditionalStructurizePass.
        // RPO places the loop exit block at i+1 (immediately after the while-head),
        // so ConditionalStructurizePass would otherwise misinterpret the head's
        // IfStmt as a single-sided if.
        detail::addWhileLoopStructurizePass(pass_manager, state);
        detail::addLoopStructurizePass(pass_manager, state);

        // Steps 6–7: conditional and switch structuring.
        detail::addConditionalStructurizePass(pass_manager, state);
        detail::addSwitchRecoveryPass(pass_manager, state);

        // Steps 8–11: cleanup and another conditional/goto-canonicalize round.
        detail::addAstCleanupPass(pass_manager, state);
        detail::addDeadLabelElimPass(pass_manager, state);
        detail::addDeadCfgPruningPass(pass_manager, state);
        detail::addGotoCanonicalizePass(pass_manager, state);
        detail::addConditionalStructurizePass(pass_manager, state);

        // Steps 12–16: if-else region formation and irreducible-flow handling,
        // then degenerate-loop and condition-recovery passes.
        detail::addIfElseRegionFormationPass(pass_manager, state);

        // Step 12.5: merge chains of if-goto-common / else-goto-next-label into a
        // single if (C0 || C1 || ...) goto COMMON; else goto FINAL;
        // Must run after IfElseRegionFormationPass so that loops are already
        // structured and the sentinel-check chains are exposed at the compound level.
        detail::addIfGotoChainMergePass(pass_manager, state);

        // detail::addIrreducibleFallbackPass(pass_manager, state);
        detail::addSwitchGotoInliningPass(pass_manager, state);
        // detail::addDegenerateLoopUnwrapPass(pass_manager, state);
        // detail::addLoopConditionRecoveryPass(pass_manager, state);

        // Steps 17–19: another dead-code / trailing-jump / AST cleanup round.
        detail::addDeadCfgPruningPass(pass_manager, state);
        detail::addTrailingJumpElimPass(pass_manager, state);
        detail::addAstCleanupPass(pass_manager, state);
        detail::addDeadLabelElimPass(pass_manager, state);

        // Steps 20–28: ast_improvements branch passes.
        detail::addDegenerateWhileElimPass(pass_manager, state);      // 20
        detail::addDeadCfgPruningPass(
            pass_manager, state
        ); // 20.5 – prune nulls from break replacement
        detail::addAstCleanupPass(
            pass_manager, state
        ); // 20.6 – strip empty null-stmt if-branches
        detail::addBackedgeLoopStructurizePass(pass_manager, state);  // 21
        detail::addCfgExtractPass(pass_manager, state);               // 22

        // Step 22.5: inline temps again just before NaturalLoopRecoveryPass so
        // that the for-loop recognizer sees bare comparisons in loop conditions.
        detail::addSingleUseTempInliningPass(pass_manager, state);    // 22.5
        detail::addNaturalLoopRecoveryPass(pass_manager, state);      // 23
        detail::addAstCleanupPass(pass_manager, state);               // 24
        detail::addDeadLabelElimPass(pass_manager, state);            // 24.5
        detail::addWhileToForUpgradePass(pass_manager, state);        // 24.75
        detail::addGotoCanonicalizePass(pass_manager, state);         // 25
        detail::addIfElseRegionFormationPass(pass_manager, state);    // 26
        detail::addAstCleanupPass(pass_manager, state);               // 27
        detail::addDeadLabelElimPass(pass_manager, state);            // 27.5
        detail::addCleanupTailExtractionPass(pass_manager, state);    // 28

        // Final verification: report any remaining gotos.
        detail::addNoGotoVerificationPass(pass_manager, state);
#endif
        return pass_manager.run(ctx, options);
    }

} // namespace patchestry::ast
