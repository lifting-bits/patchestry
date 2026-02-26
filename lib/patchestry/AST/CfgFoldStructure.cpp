/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "CfgFoldInternal.hpp"

namespace patchestry::ast {

    SNode *CfgFoldStructure(const Cfg &cfg, SNodeFactory &factory,
                             clang::ASTContext &ctx,
                             const patchestry::Options &options) {
        if (cfg.blocks.empty()) {
            return factory.Make<SSeq>();
        }

        // Set up DOT tracer — output to same directory as output_file
        // (or input_file's directory as fallback)
        CGraphDotTracer tracer;
        tracer.enabled = options.emit_dot_cfg;
        tracer.audit = options.verify_structuring;
        if (tracer.enabled || tracer.audit) {
            std::string dot_dir;
            const auto &base = options.output_file.empty()
                ? options.input_file : options.output_file;
            auto slash = base.find_last_of("/\\");
            if (slash != std::string::npos) {
                dot_dir = base.substr(0, slash + 1);
            }
            std::string name = cfg.function
                ? cfg.function->getName().str() : "unknown";
            tracer.fn_name = dot_dir + name;
            tracer.original_stmt_count = CountCfgStmts(cfg);
            CollectCfgStmtPtrs(cfg, tracer.baseline_stmts);
        }

        // Step 0: Emit input CFG as the first step DOT
        if (tracer.enabled) {
            std::ostringstream filename;
            filename << tracer.fn_name << ".step_"
                     << std::setfill('0') << std::setw(3) << tracer.step
                     << ".CfgInput.dot";
            std::error_code ec;
            llvm::raw_fd_ostream out(filename.str(), ec, llvm::sys::fs::OF_Text);
            if (!ec) EmitCfgDot(cfg, out);
            ++tracer.step;
        }

        // 1. Build the collapse graph
        detail::CGraph g = detail::BuildCGraph(cfg);
        tracer.Dump(g, "BuildCGraph", true);

        // 1b. CFG simplification
        // Fold constant branches, remove unreachable blocks, eliminate
        // empty blocks, remove dead stmts after gotos/returns.
        detail::SimplifyCGraph(g);
        tracer.Dump(g, "SimplifyCGraph", true);

        // 2. Mark back-edges
        detail::MarkBackEdges(g);

        // 2b. Discover loops, compute bodies/nesting/exits, order innermost-first
        std::list<detail::LoopBody> loopbody;
        detail::OrderLoopBodies(g, loopbody);
        LOG(INFO) << "CfgFoldStructure: found " << loopbody.size() << " loop(s)\n";
        tracer.Dump(g, "MarkBackEdges", true);

        // 2c. BackEdgePrePass removed — FoldGoto now handles non-conditional
        // 2-successor blocks with back-edges inline during the main loop
        // (Ghidra-style: ruleBlockGoto handles all goto patterns).

        // 2d. Collapse AND/OR conditions before main collapse loop
        detail::ResolveAllConditionChains(g, factory, ctx);
        LOG(INFO) << "CfgFoldStructure: condition collapsing complete\n";
        tracer.Dump(g, "ConditionChains", true);

        // 2e. Absorb switch guard chains — mark guard→fallback edges as goto
        // so FoldGoto + FoldSequence chain guards into the switch block,
        // reducing the fallback block's sizeIn for FoldSwitch.
        detail::ResolveSwitchGuards(g);
        tracer.Dump(g, "SwitchGuards", true);

        // 2f. Multi-way exit goto marking moved to stuck-resolution loop
        // (step 4) so FoldIfElseChain gets first chance at structuring.

        // 2g. Switch-skip goto conversion is integrated into FoldSwitch
        // directly: cases that target the exit/merge block get empty
        // bodies instead of blocking the fold (Ghidra's checkSwitchSkips).

        // 3. Main collapse loop
        size_t isolated = detail::FoldMainLoop(g, factory, ctx, tracer);
        tracer.Dump(g, "AfterMainLoop", true);

        // 3b. Try control-equivalence hoisting before falling back to gotos.
        // This duplicates or absorbs small shared blocks to unblock
        // FoldIfThen / FoldIfElse.
        while (isolated < g.ActiveCount()) {
            if (!detail::ResolveControlEquivHoist(g)) break;
            tracer.Dump(g, "ControlEquivHoist", true);
            isolated = detail::FoldMainLoop(g, factory, ctx, tracer);
        }

        // 4. When stuck, select gotos via TraceDAG and retry
        size_t max_iterations = g.nodes.size() * 4;  // safety bound
        size_t iter = 0;
        while (isolated < g.ActiveCount() && iter < max_iterations) {
            if (!detail::ResolveGotoSelection(g, loopbody)) {
                // TraceDAG failed (likely iteration limit) — try fallbacks:
                // 1. Control-equivalence hoisting (duplicate/absorb small blocks)
                if (detail::ResolveControlEquivHoist(g)) {
                    tracer.Dump(g, "ControlEquivHoist_Fallback", true);
                    isolated = detail::FoldMainLoop(g, factory, ctx, tracer);
                    ++iter;
                    continue;
                }
                // 2. Multi-way exit goto marking (deferred from pre-pass
                //    so FoldIfElseChain gets first chance at structuring)
                if (detail::ResolveMultiWayExitGotos(g)) {
                    tracer.Dump(g, "MultiWayExitGotos", true);
                    isolated = detail::FoldMainLoop(g, factory, ctx, tracer);
                    ++iter;
                    continue;
                }
                // 3. Simpler merge-point heuristic
                if (detail::ResolveMergePointGotos(g)) {
                    tracer.Dump(g, "MergePointGotos", true);
                    isolated = detail::FoldMainLoop(g, factory, ctx, tracer);
                    ++iter;
                    continue;
                }
                LOG(WARNING) << "CfgFoldStructure: could not select goto, "
                             << g.ActiveCount() - isolated << " blocks remaining\n";
                break;
            }
            tracer.Dump(g, "GotoSelection", true);
            isolated = detail::FoldMainLoop(g, factory, ctx, tracer);
            ++iter;
        }

        // 4b. Wrap remaining edges as gotos.
        //
        // Mark ALL remaining edges on unstructured blocks as kGoto,
        // then run FoldMainLoop one final time.  FoldGoto will wrap
        // each kGoto edge in an SGoto SNode, preserving control flow
        // that CfgBuilder::resolveEdges consumed from the original
        // goto stmts.  This is the Ghidra "ruleBlockGoto-first"
        // approach: every edge becomes either structured (if/while/
        // switch) or explicitly a goto — nothing is silently dropped.
        {
            bool marked = false;
            for (auto &n : g.nodes) {
                if (n.collapsed) continue;
                if (n.SizeOut() == 0) continue;
                if (n.structured) continue;  // handled by step 4c
                for (size_t i = 0; i < n.succs.size(); ++i) {
                    if (!n.IsGotoOut(i)) {
                        n.SetGoto(i);
                        marked = true;
                    }
                }
            }
            if (marked) {
                detail::FoldMainLoop(g, factory, ctx, tracer);
            }
        }

        // 4c. Append explicit gotos to partially-folded blocks.
        //
        // Blocks with structured SNode content (from a prior fold rule)
        // may still carry remaining outgoing edges that weren't wrapped.
        // Step 4b skipped them (routing through FoldMainLoop can cause
        // cascading interactions on large functions).  Instead, directly
        // append SGoto nodes for each remaining edge and remove the edges.
        // This prevents step 5's sequential concatenation from creating
        // wrong fallthrough chains.
        {
            for (auto &n : g.nodes) {
                if (n.collapsed) continue;
                if (!n.structured) continue;  // unstructured handled by 4b
                if (n.SizeOut() == 0) continue;

                // Append a goto for each remaining outgoing edge.
                // Typically there's just one (the "else" fallthrough of a
                // conditional goto), but handle all for completeness.
                while (n.SizeOut() > 0) {
                    size_t target_id = n.succs[0];
                    std::string target_label =
                        detail::ResolveTargetLabel(g, target_id);

                    auto *goto_node = factory.Make<SGoto>(
                        factory.Intern(target_label));

                    // Wrap existing structured content + goto in SSeq
                    auto *body = detail::LeafFromNode(n, factory);
                    auto *seq  = factory.Make<SSeq>();
                    seq->AddChild(body);
                    seq->AddChild(goto_node);
                    n.structured = seq;
                    n.label.clear();

                    g.RemoveEdge(n.id, target_id);
                }
            }
        }

        // 5. Collect the final structured tree.
        SNode *root = nullptr;
        for (auto &n : g.nodes) {
            if (n.collapsed) continue;
            auto *block = detail::LeafFromNode(n, factory);
            if (!root) {
                root = block;
            } else {
                auto *seq = root->dyn_cast<SSeq>();
                if (!seq) {
                    seq = factory.Make<SSeq>();
                    seq->AddChild(root);
                    root = seq;
                }
                seq->AddChild(block);
            }
        }

        if (!root) root = factory.Make<SSeq>();

        // 6. Post-collapse transforms (order matters per research)
        auto audit_refine = [&](const char *name) {
            if (!tracer.audit) return;
            size_t c = CountSNodeStmts(root);
            if (c < tracer.original_stmt_count) {
                LOG(WARNING) << "REFINE DROP after " << name << ": "
                             << c << " / " << tracer.original_stmt_count
                             << " (lost " << (tracer.original_stmt_count - c) << ")\n";
                // Pointer-level diff to identify exactly which stmts were lost
                if (!tracer.baseline_stmts.empty()) {
                    std::unordered_set<const clang::Stmt *> current;
                    CollectSNodeStmtPtrs(root, current);
                    ReportMissingStmts(tracer.baseline_stmts, current, name);
                }
            }
        };
        detail::RefineGotoElseNesting(root, factory);          audit_refine("GotoElseNesting");
        std::unordered_set<std::string> hoisted_labels;
        detail::RefineHoistLabel(root, factory, hoisted_labels);               audit_refine("HoistLabel");
        detail::RefineAddSkipGotos(root, factory, hoisted_labels);             audit_refine("AddSkipGotos");
        detail::RefineFallthroughGoto(root, factory, ctx);          audit_refine("FallthroughGoto");
        detail::ScopeBreak(root, "", "", factory);             audit_refine("ScopeBreak");
        detail::RefineFallthroughGoto(root, factory, ctx);          audit_refine("FallthroughGoto2");
        detail::RefineWhileTrueToDoWhile(root, factory, ctx);  audit_refine("WhileTrueToDoWhile");
        detail::RefineWhileToFor(root, factory, ctx);       audit_refine("WhileToFor");
        detail::RefineGotoToDoWhile(root, factory, ctx);  audit_refine("GotoToDoWhile");
        detail::RefineGotoEndToBreak(root, factory);     audit_refine("GotoEndToBreak");
        detail::ScopeBreak(root, "", "", factory);           audit_refine("ScopeBreak2");
        // NOTE: RefineGotoSkipTrailing distributes trailing code into
        // non-goto if-else branches and removes it from the seq.  The
        // `distributed` guard inside prevents removal when no branch
        // accepted the trailing code.  If stmts are still lost, the
        // audit_refine lambda will catch and report them.
        detail::RefineGotoSkipTrailing(root, factory); audit_refine("GotoSkipTrailing");
        detail::RefineRedundantGoto(root);             audit_refine("RedundantGoto");
        detail::RefineSwitchCaseInline(root, factory, ctx); audit_refine("SwitchCaseInline");
        detail::RefineDeadLabels(root);                audit_refine("DeadLabels");

        // Final step: emit SNode tree as the last numbered DOT
        if (tracer.enabled) {
            std::ostringstream filename;
            filename << tracer.fn_name << ".step_"
                     << std::setfill('0') << std::setw(3) << tracer.step
                     << ".SNodeOutput.dot";
            std::error_code ec;
            llvm::raw_fd_ostream out(filename.str(), ec, llvm::sys::fs::OF_Text);
            if (!ec) EmitDot(root, out);
        }

        return root;
    }

} // namespace patchestry::ast
