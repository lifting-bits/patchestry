/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// CFG-level normalization passes:
//   CfgExtractPass         – extract per-function CFG edges from the AST
//   GotoCanonicalizePass   – remove trivial goto → adjacent-label jumps
//   DeadLabelElimPass      – strip LabelStmt wrappers that are no longer jump targets
//   BasicBlockReorderPass  – reorder labeled blocks into RPO (DFS) order

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <patchestry/AST/ASTPassManager.hpp>
#include <patchestry/Util/Log.hpp>

#include "NormalizationPipelineInternal.hpp"

namespace patchestry::ast {
    namespace {

        using namespace detail;

        // =========================================================================
        // File-local helpers used only by CfgExtractPass
        // =========================================================================

        // Record a single normalized edge from a top-level terminator statement.
        // `lexical_successor` is the next top-level statement in the function body,
        // used as the target for implicit fallthrough edges.
        void recordNormalizedTerminatorEdges(
            FunctionCfg &cfg, const clang::Stmt *stmt, const clang::Stmt *lexical_successor
        ) {
            if (stmt == nullptr) {
                return;
            }

            if (auto *goto_stmt = llvm::dyn_cast< clang::GotoStmt >(stmt);
                goto_stmt != nullptr)
            {
                cfg.edges.push_back(CfgEdge{
                    .kind = EdgeKind::Unconditional,
                    .from = goto_stmt,
                    .to   = cfg.labels.contains(goto_stmt->getLabel())
                               ? cfg.labels[goto_stmt->getLabel()]
                               : nullptr
                });
                return;
            }

            if (auto *indirect = llvm::dyn_cast< clang::IndirectGotoStmt >(stmt);
                indirect != nullptr)
            {
                cfg.edges.push_back(CfgEdge{
                    .kind = EdgeKind::Indirect,
                    .from = indirect,
                    .to   = nullptr
                });
                return;
            }

            if (auto *ret_stmt = llvm::dyn_cast< clang::ReturnStmt >(stmt);
                ret_stmt != nullptr)
            {
                cfg.edges.push_back(CfgEdge{
                    .kind = EdgeKind::Exit,
                    .from = ret_stmt,
                    .to   = nullptr
                });
                return;
            }

            if (auto *if_stmt = llvm::dyn_cast< clang::IfStmt >(stmt); if_stmt != nullptr) {
                const clang::Stmt *then_stmt = if_stmt->getThen();
                const clang::Stmt *else_stmt = if_stmt->getElse();

                const auto normalize_if_edge = [&](EdgeKind kind,
                                                   const clang::Stmt *arm) -> void {
                    if (auto *arm_goto = llvm::dyn_cast_or_null< clang::GotoStmt >(arm)) {
                        cfg.edges.push_back(CfgEdge{
                            .kind = kind,
                            .from = if_stmt,
                            .to   = cfg.labels.contains(arm_goto->getLabel())
                                       ? cfg.labels[arm_goto->getLabel()]
                                       : nullptr
                        });
                    } else if (llvm::isa_and_nonnull< clang::ReturnStmt >(arm)) {
                        cfg.edges.push_back(CfgEdge{
                            .kind = EdgeKind::Exit,
                            .from = if_stmt,
                            .to   = nullptr
                        });
                    } else {
                        cfg.edges.push_back(CfgEdge{
                            .kind = kind == EdgeKind::TrueBranch ? EdgeKind::TrueBranch
                                                                 : EdgeKind::FalseBranch,
                            .from = if_stmt,
                            .to   = lexical_successor
                        });
                    }
                };

                normalize_if_edge(EdgeKind::TrueBranch, then_stmt);
                normalize_if_edge(EdgeKind::FalseBranch, else_stmt);
                return;
            }

            cfg.edges.push_back(CfgEdge{
                .kind = EdgeKind::Fallthrough,
                .from = stmt,
                .to   = lexical_successor
            });
        }

        // Walk every top-level statement in the function body and record a normalized
        // outgoing edge for each terminator.
        void normalizeFunctionEdges(FunctionCfg &cfg) {
            auto *body = llvm::dyn_cast_or_null< clang::CompoundStmt >(cfg.function->getBody());
            if (body == nullptr) {
                return;
            }

            for (auto it = body->body_begin(); it != body->body_end(); ++it) {
                const clang::Stmt *stmt = *it;
                if (stmt == nullptr) {
                    continue;
                }

                const clang::Stmt *next_stmt = nullptr;
                auto next                    = std::next(it);
                if (next != body->body_end()) {
                    next_stmt = *next;
                }

                recordNormalizedTerminatorEdges(cfg, stmt, next_stmt);
            }
        }

        // =========================================================================
        // CfgExtractPass
        // =========================================================================

        class CfgExtractPass final : public ASTPass
        {
          public:
            explicit CfgExtractPass(PipelineState &state)
                : state(state) {}

            const char *name(void) const override { return "CfgExtractPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                state.cfgs.clear();
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }

                    FunctionCfg cfg{ .function = func };
                    FunctionJumpCollector collector(cfg);
                    collector.TraverseStmt(func->getBody());
                    normalizeFunctionEdges(cfg);
                    state.cfgs.emplace_back(std::move(cfg));
                }

                if (options.verbose) {
                    unsigned total_gotos          = 0;
                    unsigned total_indirect_gotos = 0;
                    for (const auto &cfg : state.cfgs) {
                        total_gotos += cfg.goto_count;
                        total_indirect_gotos += cfg.indirect_goto_count;
                    }
                    LOG(DEBUG) << "CfgExtractPass extracted " << state.cfgs.size()
                               << " function CFG(s), gotos=" << total_gotos
                               << ", indirect_gotos=" << total_indirect_gotos << "\n";
                }

                return true;
            }

          private:
            PipelineState &state;
        };

        // =========================================================================
        // GotoCanonicalizePass
        // =========================================================================

        class GotoCanonicalizePass final : public ASTPass
        {
          public:
            explicit GotoCanonicalizePass(PipelineState &state)
                : state(state) {}

            const char *name(void) const override { return "GotoCanonicalizePass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                state.trivial_gotos_removed = 0;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }

                    auto *body = llvm::dyn_cast< clang::CompoundStmt >(func->getBody());
                    if (body == nullptr) {
                        continue;
                    }

                    std::vector< clang::Stmt * > rewritten;
                    rewritten.reserve(body->size());

                    for (auto it = body->body_begin(); it != body->body_end(); ++it) {
                        auto *stmt = *it;
                        if (stmt == nullptr) {
                            continue;
                        }

                        auto next = std::next(it);

                        // Case 1: standalone `goto L;` where L is the immediately following
                        // label — the jump is a no-op, remove it.
                        if (auto *goto_stmt = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                            if (next != body->body_end()) {
                                auto *next_label = llvm::dyn_cast< clang::LabelStmt >(*next);
                                if (next_label != nullptr
                                    && next_label->getDecl() == goto_stmt->getLabel())
                                {
                                    ++state.trivial_gotos_removed;
                                    continue; // drop the trivial goto
                                }
                            }
                            rewritten.push_back(stmt);
                            continue;
                        }

                        // Case 2: `if (cond) goto L1; else goto L2;`
                        // If L2 (else target) is the immediately following label, the else arm
                        // is a trivial fallthrough — strip it, leaving `if (cond) goto L1;`.
                        // If L1 (then target) is the immediately following label, the then arm
                        // is trivial — strip it and negate the condition, leaving
                        // `if (!cond) goto L2;`.
                        if (auto *if_stmt = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                            auto *then_goto =
                                llvm::dyn_cast_or_null< clang::GotoStmt >(if_stmt->getThen());
                            auto *else_goto =
                                llvm::dyn_cast_or_null< clang::GotoStmt >(if_stmt->getElse());

                            if (then_goto != nullptr && else_goto != nullptr
                                && next != body->body_end())
                            {
                                auto *next_label = llvm::dyn_cast< clang::LabelStmt >(*next);
                                if (next_label != nullptr) {
                                    const auto *next_decl = next_label->getDecl();

                                    if (next_decl == else_goto->getLabel()) {
                                        // else arm is trivial:
                                        // `if (cond) goto L1; else goto L2;` → `if (cond) goto L1;`
                                        ++state.trivial_gotos_removed;
                                        stmt = clang::IfStmt::Create(
                                            ctx, if_stmt->getIfLoc(),
                                            clang::IfStatementKind::Ordinary, nullptr, nullptr,
                                            if_stmt->getCond(), if_stmt->getLParenLoc(),
                                            then_goto->getBeginLoc(), then_goto,
                                            clang::SourceLocation(), nullptr
                                        );
                                    } else if (next_decl == then_goto->getLabel()) {
                                        // then arm is trivial: negate condition, swap arms.
                                        // `if (cond) goto L1; else goto L2;` → `if (!cond) goto L2;`
                                        ++state.trivial_gotos_removed;
                                        auto *neg_cond = clang::UnaryOperator::Create(
                                            ctx,
                                            ensureRValue(
                                                ctx,
                                                const_cast< clang::Expr * >(if_stmt->getCond())
                                            ),
                                            clang::UO_LNot, ctx.IntTy, clang::VK_PRValue,
                                            clang::OK_Ordinary, if_stmt->getIfLoc(),
                                            /*canOverflow=*/false, clang::FPOptionsOverride()
                                        );
                                        stmt = clang::IfStmt::Create(
                                            ctx, if_stmt->getIfLoc(),
                                            clang::IfStatementKind::Ordinary, nullptr, nullptr,
                                            neg_cond, if_stmt->getLParenLoc(),
                                            else_goto->getBeginLoc(), else_goto,
                                            clang::SourceLocation(), nullptr
                                        );
                                    }
                                }
                            }
                        }

                        rewritten.push_back(stmt);
                    }

                    if (rewritten.size() != body->size()) {
                        auto *new_body = clang::CompoundStmt::Create(
                            ctx, rewritten, clang::FPOptionsOverride(),
                            body->getLBracLoc(), body->getRBracLoc()
                        );
                        func->setBody(new_body);
                    }
                }

                // Recompute CFG edges after canonicalization to keep normalized state in-sync.
                CfgExtractPass(state).run(ctx, options);

                if (options.verbose) {
                    LOG(DEBUG) << "GotoCanonicalizePass removed "
                               << state.trivial_gotos_removed << " trivial goto(s)\n";
                }

                return true;
            }

          private:
            PipelineState &state;
        };

        // =========================================================================
        // DeadLabelElimPass
        // =========================================================================

        // Remove LabelStmt wrappers for labels that are not the target of any GotoStmt
        // within the same function.  Such labels arise after structuring transforms
        // (while/if-else recovery) that replace gotos referencing those labels.
        class DeadLabelElimPass final : public ASTPass
        {
          public:
            explicit DeadLabelElimPass(PipelineState &state)
                : state(state) {}

            const char *name(void) const override { return "DeadLabelElimPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                unsigned local_removed = 0;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }
                    auto *body = llvm::dyn_cast< clang::CompoundStmt >(func->getBody());
                    if (body == nullptr) {
                        continue;
                    }

                    std::vector< const clang::LabelDecl * > targets;
                    collectGotoTargets(body, targets);
                    std::unordered_set< const clang::LabelDecl * > used_labels(
                        targets.begin(), targets.end()
                    );

                    unsigned func_removed = 0;
                    auto *new_body =
                        stripUnreferencedLabels(ctx, body, used_labels, func_removed);
                    if (func_removed > 0 && new_body != nullptr) {
                        func->setBody(new_body);
                        local_removed += func_removed;
                    }
                }

                state.dead_labels_removed += local_removed;
                if (local_removed > 0 && options.verbose) {
                    LOG(DEBUG) << "DeadLabelElimPass: removed " << local_removed
                               << " unreferenced label(s)\n";
                }
                return true;
            }

          private:
            PipelineState &state;
        };

        // =========================================================================
        // BasicBlockReorderPass
        // =========================================================================

        // Reorder top-level labeled blocks in a function body so they appear in DFS
        // (RPO) order starting from the entry block, rather than binary address order.
        class BasicBlockReorderPass final : public ASTPass
        {
          public:
            explicit BasicBlockReorderPass(PipelineState &state)
                : state(state) {}

            const char *name(void) const override { return "BasicBlockReorderPass"; }

            bool run(clang::ASTContext &ctx, const patchestry::Options &options) override {
                if (options.verbose) {
                    LOG(DEBUG) << "Running AST pass: " << name() << "\n";
                }

                unsigned local_reordered = 0;
                for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                    auto *func = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (func == nullptr || !func->doesThisDeclarationHaveABody()) {
                        continue;
                    }
                    auto *body = llvm::dyn_cast< clang::CompoundStmt >(func->getBody());
                    if (body == nullptr) {
                        continue;
                    }
                    if (reorderFunctionBlocks(ctx, func, body)) {
                        ++local_reordered;
                    }
                }

                state.blocks_reordered += local_reordered;

                if (local_reordered > 0) {
                    if (options.verbose) {
                        LOG(DEBUG) << "BasicBlockReorderPass: reordered " << local_reordered
                                   << " function(s)\n";
                    }
                    // After reordering, new trivial goto→label adjacencies may have appeared.
                    GotoCanonicalizePass(state).run(ctx, options);
                    CfgExtractPass(state).run(ctx, options);
                }

                return true;
            }

          private:
            PipelineState &state;

            // A contiguous run of top-level statements belonging to one labeled block.
            // stmts[0] is always the LabelStmt itself.
            struct BlockSegment
            {
                clang::LabelDecl *label;
                std::vector< clang::Stmt * > stmts;
            };

            // Partition body into (unlabeled_prefix, labeled_blocks).
            static std::pair< std::vector< clang::Stmt * >, std::vector< BlockSegment > >
            partitionBody(clang::CompoundStmt *body) {
                std::vector< clang::Stmt * > prefix;
                std::vector< BlockSegment > blocks;

                for (auto *stmt : body->body()) {
                    if (auto *ls = llvm::dyn_cast_or_null< clang::LabelStmt >(stmt)) {
                        blocks.push_back(BlockSegment{ ls->getDecl(), { stmt } });
                    } else if (!blocks.empty()) {
                        blocks.back().stmts.push_back(stmt);
                    } else {
                        prefix.push_back(stmt);
                    }
                }

                return { std::move(prefix), std::move(blocks) };
            }

            // True if a statement is a definitive control-flow terminator
            // (i.e., nothing after it executes in the same block).
            static bool isBlockTerminator(const clang::Stmt *stmt) {
                if (stmt == nullptr) {
                    return false;
                }
                return llvm::isa< clang::ReturnStmt >(stmt)
                    || llvm::isa< clang::BreakStmt >(stmt)
                    || llvm::isa< clang::ContinueStmt >(stmt)
                    || llvm::isa< clang::GotoStmt >(stmt);
            }

            // Build label → successor-label adjacency (preserving first-occurrence order).
            // Includes implicit fallthrough edges for blocks whose last statement is not
            // a terminator (this handles trivial gotos already removed by GotoCanonicalizePass).
            static std::unordered_map<
                const clang::LabelDecl *, std::vector< const clang::LabelDecl * > >
            buildSuccessorMap(
                const std::vector< BlockSegment > &blocks,
                const std::unordered_map< const clang::LabelDecl *, std::size_t > &label_to_idx
            ) {
                std::unordered_map<
                    const clang::LabelDecl *, std::vector< const clang::LabelDecl * > >
                    succ_map;

                for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
                    const auto &block = blocks[bi];
                    std::vector< const clang::LabelDecl * > raw_targets;
                    for (const auto *stmt : block.stmts) {
                        collectGotoTargets(stmt, raw_targets);
                    }

                    // If block has no explicit goto targets and does not end with a
                    // terminator, it falls through to the next block in the current flat
                    // order (this recovers edges removed by GotoCanonicalizePass).
                    if (raw_targets.empty() && bi + 1 < blocks.size()) {
                        const auto *last = block.stmts.empty() ? nullptr : block.stmts.back();
                        if (!isBlockTerminator(last)) {
                            raw_targets.push_back(blocks[bi + 1].label);
                        }
                    }

                    // Deduplicate, keeping first occurrence and only known labels.
                    std::vector< const clang::LabelDecl * > unique_targets;
                    std::unordered_set< const clang::LabelDecl * > seen;
                    for (const auto *t : raw_targets) {
                        if (label_to_idx.contains(t) && seen.insert(t).second) {
                            unique_targets.push_back(t);
                        }
                    }

                    succ_map[block.label] = std::move(unique_targets);
                }

                return succ_map;
            }

            // Identify the entry label using two heuristics in priority order:
            //   1. The first GotoStmt target found in the unlabeled prefix.
            //   2. The first block (in original order) whose in-degree from other blocks is 0.
            //   3. Fallback: the very first block.
            static const clang::LabelDecl *findEntryLabel(
                const std::vector< clang::Stmt * > &prefix,
                const std::vector< BlockSegment > &blocks,
                const std::unordered_map< const clang::LabelDecl *, std::size_t > &label_to_idx,
                const std::unordered_map<
                    const clang::LabelDecl *, std::vector< const clang::LabelDecl * > >
                    &succ_map
            ) {
                // Heuristic 1: goto in the unlabeled prefix → that is the entry.
                for (const auto *stmt : prefix) {
                    std::vector< const clang::LabelDecl * > targets;
                    collectGotoTargets(stmt, targets);
                    for (const auto *t : targets) {
                        if (label_to_idx.contains(t)) {
                            return t;
                        }
                    }
                }

                // Heuristic 2: block with in-degree 0 among labeled blocks.
                std::unordered_map< const clang::LabelDecl *, unsigned > in_degree;
                for (const auto &block : blocks) {
                    in_degree.try_emplace(block.label, 0U);
                }
                for (const auto &[from, succs] : succ_map) {
                    for (const auto *t : succs) {
                        auto it = in_degree.find(t);
                        if (it != in_degree.end()) {
                            ++it->second;
                        }
                    }
                }

                // Return the first zero-in-degree block in original order.
                for (const auto &block : blocks) {
                    if (in_degree.at(block.label) == 0U) {
                        return block.label;
                    }
                }

                // Heuristic 3: all blocks have incoming edges (cycle) — first block wins.
                return blocks.empty() ? nullptr : blocks.front().label;
            }

            // Reverse Post-Order (RPO) DFS from entry.
            //
            // RPO is the canonical linearisation used by structured decompilers.
            // It guarantees three properties that BFS cannot:
            //
            //   1. A block always appears before all of its forward-edge successors
            //      → ConditionalStructurizePass sees then/else/join in the correct order.
            //
            //   2. For a natural loop whose back-edge is B→H, the exit block E
            //      (else-branch of the condition at H) is placed immediately after H's
            //      condition IfStmt → LoopStructurizePass's `exit_idx == tail_pos + 1`
            //      check passes without needing special loop-awareness.
            //
            //   3. Post-loop epilog blocks discovered via a short path (e.g. 0→10→11)
            //      are NOT pulled in front of the loop body; they remain at their
            //      natural RPO position (after all blocks that must precede them in a
            //      topological sort of the forward-edge DAG).
            //
            // Back-edges are handled by the standard DFS "already visited" guard: when
            // a successor is already on the visited set we simply skip it, so the
            // back-edge is recorded as a backward goto in the flat list (which is
            // exactly what LoopStructurizePass and IrreducibleFallbackPass expect).
            //
            // Implementation: iterative post-order DFS, then reverse.
            static std::vector< std::size_t > rpoOrder(
                const clang::LabelDecl *entry,
                const std::unordered_map< const clang::LabelDecl *, std::size_t > &label_to_idx,
                const std::unordered_map<
                    const clang::LabelDecl *, std::vector< const clang::LabelDecl * > >
                    &succ_map
            ) {
                std::vector< std::size_t > post_order;
                std::unordered_set< const clang::LabelDecl * > visited;

                std::vector< std::pair< const clang::LabelDecl *, std::size_t > > stack;
                stack.push_back({ entry, 0 });
                visited.insert(entry);

                while (!stack.empty()) {
                    auto &[label, succ_idx] = stack.back();

                    auto sit          = succ_map.find(label);
                    const auto *succs = (sit != succ_map.end()) ? &sit->second : nullptr;

                    bool pushed = false;
                    while (succs != nullptr && succ_idx < succs->size()) {
                        const auto *next = (*succs)[succ_idx++];
                        if (visited.insert(next).second) {
                            stack.push_back({ next, 0 });
                            pushed = true;
                            break;
                        }
                    }

                    if (!pushed) {
                        auto it = label_to_idx.find(label);
                        if (it != label_to_idx.end()) {
                            post_order.push_back(it->second);
                        }
                        stack.pop_back();
                    }
                }

                std::reverse(post_order.begin(), post_order.end());
                return post_order;
            }

            bool reorderFunctionBlocks(
                clang::ASTContext &ctx, clang::FunctionDecl *func, clang::CompoundStmt *body
            ) {
                auto [prefix, blocks] = partitionBody(body);

                if (blocks.size() < 2U) {
                    return false; // Nothing to reorder.
                }

                std::unordered_map< const clang::LabelDecl *, std::size_t > label_to_idx;
                label_to_idx.reserve(blocks.size());
                for (std::size_t i = 0; i < blocks.size(); ++i) {
                    label_to_idx[blocks[i].label] = i;
                }

                auto succ_map = buildSuccessorMap(blocks, label_to_idx);
                auto *entry   = findEntryLabel(prefix, blocks, label_to_idx, succ_map);
                if (entry == nullptr) {
                    return false;
                }

                auto dfs_indices = rpoOrder(entry, label_to_idx, succ_map);

                // Append unreachable blocks (dead code) at the end.
                std::unordered_set< std::size_t > visited_set(
                    dfs_indices.begin(), dfs_indices.end()
                );
                for (std::size_t i = 0; i < blocks.size(); ++i) {
                    if (!visited_set.contains(i)) {
                        dfs_indices.push_back(i);
                    }
                }

                // Check whether the order actually changed.
                bool changed = false;
                for (std::size_t i = 0; i < dfs_indices.size(); ++i) {
                    if (dfs_indices[i] != i) {
                        changed = true;
                        break;
                    }
                }
                if (!changed) {
                    return false;
                }

                // Rebuild body: prefix first, then blocks in DFS order.
                std::vector< clang::Stmt * > new_stmts;
                new_stmts.reserve(body->size());
                new_stmts.insert(new_stmts.end(), prefix.begin(), prefix.end());
                for (std::size_t idx : dfs_indices) {
                    const auto &seg = blocks[idx];
                    new_stmts.insert(new_stmts.end(), seg.stmts.begin(), seg.stmts.end());
                }

                func->setBody(makeCompound(ctx, new_stmts, body->getLBracLoc(), body->getRBracLoc())
                );
                return true;
            }
        };

    } // anonymous namespace

    namespace detail {

        void runCfgExtractPass(
            PipelineState &state, clang::ASTContext &ctx, const patchestry::Options &options
        ) {
            CfgExtractPass(state).run(ctx, options);
        }

        void runGotoCanonicalizePass(
            PipelineState &state, clang::ASTContext &ctx, const patchestry::Options &options
        ) {
            GotoCanonicalizePass(state).run(ctx, options);
        }

        void addCfgPasses(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< CfgExtractPass >(state));
            pm.add_pass(std::make_unique< BasicBlockReorderPass >(state));
        }

        void addCfgExtractPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< CfgExtractPass >(state));
        }

        void addGotoCanonicalizePass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< GotoCanonicalizePass >(state));
        }

        void addDeadLabelElimPass(ASTPassManager &pm, PipelineState &state) {
            pm.add_pass(std::make_unique< DeadLabelElimPass >(state));
        }

    } // namespace detail

} // namespace patchestry::ast
