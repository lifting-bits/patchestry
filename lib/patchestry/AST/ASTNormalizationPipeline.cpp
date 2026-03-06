/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// Orchestrates the full AST normalization pipeline.
// Uses SNode wrapper IR for structural analysis, then emits back to Clang AST.

#include <patchestry/AST/ASTNormalizationPipeline.hpp>
#include <patchestry/AST/ASTPassManager.hpp>
#include <patchestry/AST/CfgBuilder.hpp>
#include <patchestry/AST/ClangEmitter.hpp>
#include <patchestry/AST/DomTree.hpp>
#include <patchestry/AST/LoopInfo.hpp>
#include <patchestry/AST/SNode.hpp>
#include <patchestry/AST/SNodeBuilder.hpp>
#include <patchestry/AST/SNodeDebug.hpp>
#include <patchestry/AST/SNodePass.hpp>
#include <patchestry/AST/StructuralPasses.hpp>
#include <patchestry/Util/Log.hpp>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include "NormalizationPipelineInternal.hpp"

namespace patchestry::ast {

    bool runASTNormalizationPipeline(
        clang::ASTContext &ctx, const patchestry::Options &options
    ) {
        if (!options.enable_goto_elimination) {
            LOG(INFO) << "Goto elimination disabled, skipping AST normalization\n";
            return true;
        }

        LOG(INFO) << "Starting SNode-based AST normalization pipeline\n";

        // Step 4: Build CFGs from Clang AST
        auto cfgs = buildCfgs(ctx);
        if (cfgs.empty()) {
            LOG(INFO) << "No functions to normalize\n";
            return true;
        }

        LOG(INFO) << "Built CFGs for " << cfgs.size() << " functions\n";

        // Process each function
        for (auto &cfg : cfgs) {
            if (cfg.blocks.empty()) continue;

            auto *fn = const_cast< clang::FunctionDecl * >(cfg.function);
            LOG(INFO) << "Processing function: "
                      << fn->getNameAsString() << " ("
                      << cfg.blockCount() << " blocks)\n";

            // Step 5: Build initial SNode tree from CFG
            SNodeFactory factory;
            SNode *root = buildSNodeTree(cfg, factory);

            // Dump initial SNode tree if verbose
            if (options.verbose && options.print_tu && !options.output_file.empty()) {
                std::string prefix = options.output_file + "_snode_initial_"
                    + fn->getNameAsString();
                {
                    std::error_code ec;
                    llvm::raw_fd_ostream out(prefix + ".c", ec,
                                             llvm::sys::fs::OF_Text);
                    if (!ec) {
                        out << "// Initial SNode tree for "
                            << fn->getNameAsString() << "\n\n";
                        printPseudoC(root, out, &ctx);
                    }
                }
                {
                    std::error_code ec;
                    llvm::raw_fd_ostream out(prefix + ".dot", ec,
                                             llvm::sys::fs::OF_Text);
                    if (!ec) emitDOT(root, out);
                }
            }

            // Step 6: Compute dominator trees
            auto dom = DomTree::buildDom(cfg);
            auto post_dom = DomTree::buildPostDom(cfg);

            // Step 7: Detect natural loops
            auto loop_info = detectLoops(cfg, dom);
            if (options.verbose) {
                LOG(INFO) << "  Detected " << loop_info.loops.size()
                          << " natural loops\n";
            }

            // Build SNode pass pipeline
            SNodePassManager snode_pm;

            // 1. Sequence collapse (trivial goto removal, block merging)
            snode_pm.addPass(std::make_unique< SequenceCollapsePass >());

            // 2. Indirect goto → switch (resolve computed gotos early)
            snode_pm.addPass(std::make_unique< IndirectGotoSwitchPass >());

            // 3. While/do-while loop recovery
            snode_pm.addPass(std::make_unique< WhileLoopRecoveryPass >(
                cfg, loop_info));

            // 4. Short-circuit boolean recovery (before if-then-else!)
            snode_pm.addPass(std::make_unique< ShortCircuitRecoveryPass >());

            // 5. If-then-else recovery
            snode_pm.addPass(std::make_unique< IfThenElseRecoveryPass >(
                cfg, post_dom));

            // 6. Switch recovery (cascaded if-else → switch)
            snode_pm.addPass(std::make_unique< SwitchRecoveryPass >());

            // 7. Forward goto elimination
            snode_pm.addPass(std::make_unique< ForwardGotoEliminationPass >());

            // 8. Switch backedge loop (switch with backedge gotos → while)
            snode_pm.addPass(std::make_unique< SwitchBackedgeLoopPass >());

            // 9. Switch goto inlining (inline label bodies into case arms)
            snode_pm.addPass(std::make_unique< SwitchGotoInliningPass >());

            // 10. Backward goto to do-while
            snode_pm.addPass(std::make_unique< BackwardGotoToDoWhilePass >());

            // 11. Multi-exit break insertion
            snode_pm.addPass(std::make_unique< MultiExitBreakPass >());

            // 12. Irreducible handling (logs remaining gotos)
            snode_pm.addPass(std::make_unique< IrreducibleHandlingPass >());

            // 13. Cleanup passes
            snode_pm.addPass(std::make_unique< CleanupPass >());

            // 14-15. Second round of sequence collapse and cleanup
            snode_pm.addPass(std::make_unique< SequenceCollapsePass >());
            snode_pm.addPass(std::make_unique< CleanupPass >());

            // Run all SNode passes
            snode_pm.run(root, factory, ctx, options);

            // Step 17: Emit back to Clang AST
            emitClangAST(root, fn, ctx);

            if (options.verbose) {
                LOG(INFO) << "  Completed normalization for "
                          << fn->getNameAsString() << "\n";
            }
        }

        // Goto diagnostic summary
        auto [gotos, indirect_gotos, first_loc] =
            detail::countRemainingGotos(ctx.getTranslationUnitDecl());
        LOG(INFO) << "AST normalization complete: "
                  << gotos << " gotos, "
                  << indirect_gotos << " indirect gotos remaining\n";

        return true;
    }

} // namespace patchestry::ast
