/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

// Internal header for CfgFoldStructure split files.
// NOT part of the public API — only included by lib/patchestry/AST/*.cpp.

#include <patchestry/AST/CfgFoldStructure.hpp>
#include <patchestry/AST/CfgBuilder.hpp>
#include <patchestry/AST/CfgDotEmitter.hpp>
#include <patchestry/AST/SNode.hpp>
#include <patchestry/AST/SNodeDebug.hpp>
#include <patchestry/Util/Log.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <list>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>

namespace patchestry::ast {

    // All internal functions live in the detail namespace alongside
    // CGraph/CNode/LoopBody already declared in CfgFoldStructure.hpp.
    namespace detail {

    // ---------------------------------------------------------------
    // Shared helpers (CfgFoldRules.cpp)
    // ---------------------------------------------------------------

    clang::Expr *NegateCond(clang::Expr *cond, clang::ASTContext &ctx);

    SNode *LeafFromNode(CNode &n, SNodeFactory &factory);

    std::string ResolveTargetLabel(const CGraph &g, size_t target_id);

    SNode *BuildInlineOrGoto(
        CGraph &g, size_t target_id,
        std::unordered_set<size_t> &id_set,
        size_t exit_id,
        std::vector<size_t> &extra_ids,
        SNodeFactory &factory, clang::ASTContext &ctx,
        size_t depth = 0,
        const CNode *source = nullptr, size_t edge_idx = 0);

    SNode *EmitExitGotos(
        CGraph &g, SNode *body, CNode &tail,
        std::unordered_set<size_t> &id_set,
        size_t exit_id,
        std::vector<size_t> &extra_ids,
        SNodeFactory &factory, clang::ASTContext &ctx,
        size_t depth = 0);

    // ---------------------------------------------------------------
    // Fold rules (CfgFoldRules.cpp)
    // ---------------------------------------------------------------

    bool FoldSequence(CGraph &g, size_t id, SNodeFactory &factory);
    bool FoldIfThen(CGraph &g, size_t id, SNodeFactory &factory, clang::ASTContext &ctx);
    bool FoldIfElse(CGraph &g, size_t id, SNodeFactory &factory, clang::ASTContext &ctx);
    bool FoldIfElseChain(CGraph &g, size_t id, SNodeFactory &factory, clang::ASTContext &ctx);
    bool FoldWhileLoop(CGraph &g, size_t id, SNodeFactory &factory, clang::ASTContext &ctx);
    bool FoldDoWhileLoop(CGraph &g, size_t id, SNodeFactory &factory, clang::ASTContext &ctx);
    bool FoldInfiniteLoop(CGraph &g, size_t id, SNodeFactory &factory, clang::ASTContext &ctx);
    bool FoldIfForcedGoto(CGraph &g, size_t id, SNodeFactory &factory, clang::ASTContext &ctx);
    bool FoldIfThenGoto(CGraph &g, size_t id, SNodeFactory &factory, clang::ASTContext &ctx);
    bool FoldSwitch(CGraph &g, size_t id, SNodeFactory &factory, clang::ASTContext &ctx);
    bool FoldGoto(CGraph &g, size_t id, SNodeFactory &factory, clang::ASTContext &ctx);
    bool FoldCaseFallthrough(CGraph &g, size_t id);

    // ---------------------------------------------------------------
    // Resolve functions (CfgFoldResolve.cpp)
    // ---------------------------------------------------------------

    /// CFG simplification — constant branch folding, unreachable block
    /// removal, empty block elimination, dead-stmt cleanup.  Runs before
    /// structural analysis (Ghidra-style pre-structuring normalization).
    void SimplifyCGraph(CGraph &g);

    void ResolveAllConditionChains(CGraph &g, SNodeFactory &factory, clang::ASTContext &ctx);
    bool ResolveGotoSelection(CGraph &g, std::list<LoopBody> &loopbody);
    bool ResolveMergePointGotos(CGraph &g);
    bool ResolveSwitchGuards(CGraph &g);
    bool ResolveMultiWayExitGotos(CGraph &g);
    bool ResolveControlEquivHoist(CGraph &g);

    size_t FoldMainLoop(CGraph &g, SNodeFactory &factory,
                        clang::ASTContext &ctx, CGraphDotTracer &tracer);

    // ---------------------------------------------------------------
    // Refine functions (CfgFoldRefine.cpp)
    // ---------------------------------------------------------------

    void CollectGotoTargets(SNode *node, std::unordered_set<std::string> &targets);
    bool EndsWithTransfer(SNode *n);
    bool FallsThroughToLabel(SNode *node, std::string_view target);
    SNode *AppendGoto(SNode *branch, std::string_view label, SNodeFactory &factory);

    [[maybe_unused]]
    void RefineBreakContinue(SNode *node, std::string_view exit,
                             std::string_view header, SNodeFactory &factory);
    void ScopeBreak(SNode *node, std::string_view exit_label,
                    std::string_view header_label, SNodeFactory &factory);
    void RefineWhileToFor(SNode *node, SNodeFactory &factory, clang::ASTContext &ctx);
    void RefineGotoElseNesting(SNode *node, SNodeFactory &factory,
                               const std::unordered_set<std::string> &ancestor_gotos = {});
    void RefineHoistLabel(SNode *root, SNodeFactory &factory,
                          std::unordered_set<std::string> &hoisted_labels);
    void RefineAddSkipGotos(SNode *node, SNodeFactory &factory,
                            const std::unordered_set<std::string> &hoisted_labels);
    void RefineGotoToDoWhile(SNode *node, SNodeFactory &factory, clang::ASTContext &ctx);
    void RefineGotoEndToBreak(SNode *root, SNodeFactory &factory);
    void RefineGotoSkipTrailing(SNode *node, SNodeFactory &factory);
    void RefineRedundantGoto(SNode *node);
    void RefineSwitchCaseInline(SNode *node, SNodeFactory &factory, clang::ASTContext &ctx);
    void RefineFallthroughGoto(SNode *node, SNodeFactory &factory, clang::ASTContext &ctx);
    void RefineEpilogueReturn(SNode *root, SNodeFactory &factory);
    void RefineCommonGotoHoist(SNode *node, SNodeFactory &factory);
    void RefineSmallBlockInline(SNode *root, SNodeFactory &factory);
    void RefineDeadCode(SNode *node);
    void RefineDeadLabels(SNode *root);
    void RefineWhileTrueToDoWhile(SNode *node, SNodeFactory &factory, clang::ASTContext &ctx);

    } // namespace detail

    // Statement counting (in CfgDotEmitter.cpp, patchestry::ast namespace)
    size_t CountSNodeStmts(const SNode *root);
    void CollectSNodeStmtPtrs(const SNode *root,
                              std::unordered_set<const clang::Stmt *> &ptrs);
    size_t ReportMissingStmts(const std::unordered_set<const clang::Stmt *> &baseline,
                              const std::unordered_set<const clang::Stmt *> &current,
                              llvm::StringRef context);

} // namespace patchestry::ast
