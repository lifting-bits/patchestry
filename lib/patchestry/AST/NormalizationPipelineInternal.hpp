/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// Internal header for the AST normalization pipeline.
// This file is NOT installed; it is only included by the normalization pass
// .cpp files under lib/patchestry/AST/.  All symbols live in the
// patchestry::ast::detail namespace so they do not pollute the public API.

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/Diagnostic.h>
#include <llvm/Support/Casting.h>

namespace patchestry::ast::detail {

    // =========================================================================
    // CFG edge representation
    // =========================================================================

    enum class EdgeKind
    {
        TrueBranch,
        FalseBranch,
        Unconditional,
        Fallthrough,
        Exit,
        Indirect
    };

    struct CfgEdge
    {
        EdgeKind kind;
        const clang::Stmt *from;
        const clang::Stmt *to;
    };

    struct FunctionCfg
    {
        const clang::FunctionDecl *function = nullptr;
        std::unordered_map< const clang::LabelDecl *, const clang::LabelStmt * > labels;
        std::vector< const clang::AddrLabelExpr * > addr_label_exprs;
        std::vector< CfgEdge > edges;
        unsigned goto_count          = 0;
        unsigned indirect_goto_count = 0;
    };

    // =========================================================================
    // Pipeline-wide mutable state passed to every pass
    // =========================================================================

    struct PipelineState
    {
        std::vector< FunctionCfg > cfgs;
        unsigned trivial_gotos_removed        = 0;
        unsigned conditional_structurized     = 0;
        unsigned loops_structurized           = 0;
        unsigned indirect_switches_built      = 0;
        unsigned fallback_rewrites            = 0;
        bool used_irreducible_fallback        = false;
        unsigned dead_stmts_pruned            = 0;
        unsigned blocks_reordered             = 0;
        unsigned degenerate_while_eliminated  = 0;
        unsigned natural_loops_recovered      = 0;
        unsigned for_loops_upgraded           = 0;
        unsigned cleanup_tails_extracted      = 0;
        unsigned dead_labels_removed          = 0;
        unsigned backedge_loops_structured    = 0;
        unsigned single_use_temps_inlined     = 0;
        unsigned switch_cases_inlined         = 0;
    };

    // =========================================================================
    // Visitor helpers
    // =========================================================================

    // Collects all goto/indirect-goto edges and label declarations in a function.
    class FunctionJumpCollector final
        : public clang::RecursiveASTVisitor< FunctionJumpCollector >
    {
      public:
        explicit FunctionJumpCollector(FunctionCfg &cfg)
            : cfg(cfg) {}

        bool VisitLabelStmt(clang::LabelStmt *stmt) {
            cfg.labels[stmt->getDecl()] = stmt;
            return true;
        }

        bool VisitGotoStmt(clang::GotoStmt *stmt) {
            ++cfg.goto_count;
            cfg.edges.push_back(CfgEdge{
                .kind = EdgeKind::Unconditional,
                .from = stmt,
                .to   = cfg.labels.contains(stmt->getLabel())
                           ? cfg.labels[stmt->getLabel()]
                           : nullptr
            });
            return true;
        }

        bool VisitIndirectGotoStmt(clang::IndirectGotoStmt *stmt) {
            ++cfg.indirect_goto_count;
            cfg.edges.push_back(CfgEdge{
                .kind = EdgeKind::Indirect,
                .from = stmt,
                .to   = nullptr
            });
            return true;
        }

        bool VisitAddrLabelExpr(clang::AddrLabelExpr *expr) {
            cfg.addr_label_exprs.push_back(expr);
            return true;
        }

      private:
        FunctionCfg &cfg;
    };

    // Counts remaining gotos after each pipeline iteration.
    class RemainingGotoCollector final
        : public clang::RecursiveASTVisitor< RemainingGotoCollector >
    {
      public:
        bool VisitGotoStmt(clang::GotoStmt *stmt) {
            ++gotos;
            if (!first_location.has_value()) {
                first_location = stmt->getBeginLoc();
            }
            return true;
        }

        bool VisitIndirectGotoStmt(clang::IndirectGotoStmt *stmt) {
            ++indirect_gotos;
            if (!first_location.has_value()) {
                first_location = stmt->getBeginLoc();
            }
            return true;
        }

        unsigned gotos                              = 0;
        unsigned indirect_gotos                     = 0;
        std::optional< clang::SourceLocation > first_location;
    };

    // Collects all label declarations that are actually referenced by a goto or
    // address-of-label expression; used by AstCleanupPass.
    class LabelUseCollector final
        : public clang::RecursiveASTVisitor< LabelUseCollector >
    {
      public:
        bool VisitGotoStmt(clang::GotoStmt *stmt) {
            live_labels.insert(stmt->getLabel());
            return true;
        }

        bool VisitAddrLabelExpr(clang::AddrLabelExpr *expr) {
            live_labels.insert(expr->getLabel());
            return true;
        }

        std::unordered_set< const clang::LabelDecl * > live_labels;
    };

    // =========================================================================
    // Trivially-inline AST construction utilities
    // =========================================================================

    inline void emitDiagnostic(
        clang::ASTContext &ctx, clang::SourceLocation loc,
        clang::DiagnosticsEngine::Level level, const std::string &message
    ) {
        auto &diag = ctx.getDiagnostics();
        auto id    = diag.getCustomDiagID(level, "goto-elimination: %0");
        diag.Report(loc, id) << message;
    }

    // Wrap an lvalue expression in an implicit LValueToRValue cast so it can
    // be used as a prvalue operand (e.g., for UO_LNot or as an IfStmt condition).
    // Returns the expression unchanged when it is already a prvalue.
    inline clang::Expr *ensureRValue(clang::ASTContext &ctx, clang::Expr *expr) {
        if (expr == nullptr || !expr->isGLValue()) {
            return expr;
        }
        return clang::ImplicitCastExpr::Create(
            ctx, expr->getType(), clang::CK_LValueToRValue, expr, nullptr,
            clang::VK_PRValue, clang::FPOptionsOverride()
        );
    }

    inline clang::CompoundStmt *makeCompound(
        clang::ASTContext &ctx, const std::vector< clang::Stmt * > &stmts,
        clang::SourceLocation l_brace = clang::SourceLocation(),
        clang::SourceLocation r_brace = clang::SourceLocation()
    ) {
        return clang::CompoundStmt::Create(
            ctx, stmts, clang::FPOptionsOverride(), l_brace, r_brace
        );
    }

    inline clang::Expr *makeIntLiteral(
        clang::ASTContext &ctx, std::uint64_t value, clang::QualType type,
        clang::SourceLocation loc = clang::SourceLocation()
    ) {
        return clang::IntegerLiteral::Create(
            ctx, llvm::APInt(static_cast< unsigned >(ctx.getTypeSize(type)), value), type, loc
        );
    }

    inline clang::Expr *makeBoolTrue(clang::ASTContext &ctx, clang::SourceLocation loc) {
        return makeIntLiteral(ctx, 1, ctx.IntTy, loc);
    }

    inline bool isNullStmt(const clang::Stmt *stmt) {
        return llvm::isa_and_nonnull< clang::NullStmt >(stmt);
    }

    // =========================================================================
    // Declarations of non-inline shared helper functions
    // (implemented in NormalizationPipelineHelpers.cpp)
    // =========================================================================

    // Build a map from each top-level LabelDecl in `body` to its flat index.
    std::unordered_map< const clang::LabelDecl *, std::size_t >
    topLevelLabelIndex(const clang::CompoundStmt *body);

    // Returns true if any statement in [begin, end] within `stmts` is a LabelStmt.
    bool containsLabelInRange(
        const std::vector< clang::Stmt * > &stmts, std::size_t begin, std::size_t end
    );

    // Recursively collect all GotoStmt label targets reachable from `stmt`.
    void collectGotoTargets(
        const clang::Stmt *stmt, std::vector< const clang::LabelDecl * > &targets
    );

    // Returns true if any GotoStmt reachable from `stmt` targets `target`.
    bool containsGotoTo(const clang::Stmt *stmt, const clang::LabelDecl *target);

    // Recursively strip LabelStmt wrappers whose LabelDecl is not referenced by
    // any GotoStmt.  When a label is stripped the sub-statement takes its place.
    clang::Stmt *stripUnreferencedLabels(
        clang::ASTContext &ctx, clang::Stmt *stmt,
        const std::unordered_set< const clang::LabelDecl * > &used_labels, unsigned &removed
    );

    // Count remaining goto/indirect-goto stmts in the translation unit.
    std::tuple< unsigned, unsigned, std::optional< clang::SourceLocation > >
    countRemainingGotos(clang::TranslationUnitDecl *tu);

    // Returns true if `cond` is a compile-time constant with a true (non-zero) value.
    bool isAlwaysTrue(const clang::Expr *cond);

    // Returns true if `cond` is a compile-time constant with a false (zero) value.
    bool isAlwaysFalse(const clang::Expr *cond);

    // Returns true if `stmt` unconditionally exits the current scope (break/continue/
    // return/goto, or an if-else where both branches are unconditional terminators).
    bool isUnconditionalTerminator(const clang::Stmt *stmt);

    // =========================================================================
    // CFG infrastructure types and declarations
    // (SimpleBlockDomTree is defined here; builders declared, implemented in
    //  NormalizationPipelineHelpers.cpp)
    // =========================================================================

    struct SimpleBlockDomTree
    {
        static constexpr std::size_t UNDEF = std::numeric_limits< std::size_t >::max();
        std::vector< std::size_t > idom;    // immediate dominator per block
        std::vector< std::size_t > rpo_num; // RPO position per block
        std::size_t entry;
        std::size_t N; // virtual exit = N

        // Returns true if block `a` dominates block `b`.
        bool dominates(std::size_t a, std::size_t b) const {
            if (a == b) {
                return true;
            }
            if (b == UNDEF || b > N || rpo_num[b] == UNDEF) {
                return false;
            }
            std::size_t x  = b;
            unsigned guard = 0;
            while (x != entry && x != UNDEF && guard < 1024U) {
                if (idom[x] == UNDEF) {
                    return false;
                }
                x = idom[x];
                if (x == a) {
                    return true;
                }
                ++guard;
            }
            return x == a;
        }
    };

    // Build flat adjacency list: succ_of[i] = successor indices in flat stmts order.
    // N = stmts.size() is used as the virtual exit node index.
    std::vector< std::vector< std::size_t > > buildSuccOf(
        const std::vector< clang::Stmt * > &stmts,
        const std::unordered_map< const clang::LabelDecl *, std::size_t > &label_index,
        std::size_t N
    );

    // Detect backward edges (i→j) where j ≤ i in flat statement order.
    std::vector< std::pair< std::size_t, std::size_t > > detectBackedges(
        std::size_t N, const std::vector< std::vector< std::size_t > > &succ_of
    );

    // Build a dominator tree for the flat-indexed block graph using the
    // Cooper/Harvey/Kennedy iterative algorithm.
    SimpleBlockDomTree buildDomTree(
        const std::vector< std::vector< std::size_t > > &succ_of, std::size_t entry_idx,
        std::size_t N
    );

} // namespace patchestry::ast::detail

// =========================================================================
// Pass-group registration functions and cross-file pass invocation helpers
// (defined in their respective NormalizationXxxPasses.cpp files)
// =========================================================================

#include <patchestry/Util/Options.hpp>

namespace patchestry::ast {
    class ASTPassManager;
} // namespace patchestry::ast

namespace patchestry::ast::detail {

    // Helpers called by passes in one file that need to invoke a pass defined in
    // another file (e.g., ConditionalStructurizePass calls CfgExtractPass).
    void runCfgExtractPass(PipelineState &, clang::ASTContext &, const patchestry::Options &);
    void runGotoCanonicalizePass(PipelineState &, clang::ASTContext &, const patchestry::Options &);
    void runAstCleanupPass(PipelineState &, clang::ASTContext &, const patchestry::Options &);

    // ---- CFG passes (NormalizationCfgPasses.cpp) ----
    void addCfgPasses(patchestry::ast::ASTPassManager &, PipelineState &);
    void addCfgExtractPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addGotoCanonicalizePass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addDeadLabelElimPass(patchestry::ast::ASTPassManager &, PipelineState &);

    // ---- Conditional passes (NormalizationConditionalPasses.cpp) ----
    void addConditionalStructurizePass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addIfElseRegionFormationPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addConditionalPasses(patchestry::ast::ASTPassManager &, PipelineState &);

    // ---- Loop passes (NormalizationLoopPasses.cpp) ----
    void addWhileLoopStructurizePass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addLoopStructurizePass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addDegenerateLoopUnwrapPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addLoopConditionRecoveryPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addDegenerateWhileElimPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addNaturalLoopRecoveryPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addBackedgeLoopStructurizePass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addWhileToForUpgradePass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addLoopPasses(patchestry::ast::ASTPassManager &, PipelineState &);

    // ---- Switch/Irreducible passes (NormalizationSwitchPasses.cpp) ----
    void addSwitchRecoveryPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addIrreducibleFallbackPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addSwitchGotoInliningPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addSwitchPasses(patchestry::ast::ASTPassManager &, PipelineState &);

    // ---- Cleanup passes (NormalizationCleanupPasses.cpp) ----
    void addAstCleanupPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addDeadCfgPruningPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addTrailingJumpElimPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addSingleUseTempInliningPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addCleanupTailExtractionPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addNoGotoVerificationPass(patchestry::ast::ASTPassManager &, PipelineState &);
    void addCleanupPasses(patchestry::ast::ASTPassManager &, PipelineState &);

} // namespace patchestry::ast::detail
