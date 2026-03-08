/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/SNodePass.hpp>
#include <patchestry/AST/CfgBuilder.hpp>
#include <patchestry/AST/DomTree.hpp>
#include <patchestry/AST/LoopInfo.hpp>

namespace patchestry::ast {

    // Step 8: Collapse consecutive non-branching blocks and remove trivial gotos
    class SequenceCollapsePass : public SNodePass {
      public:
        std::string_view name() const override { return "SequenceCollapse"; }
        bool run(SNode *root, SNodeFactory &factory, clang::ASTContext &ctx) override;
    };

    // Step 9: Recover if-then-else from conditional gotos using post-dom convergence
    class IfThenElseRecoveryPass : public SNodePass {
      public:
        IfThenElseRecoveryPass(const Cfg & /*cfg*/, const DomTree & /*post_dom*/) {}
        std::string_view name() const override { return "IfThenElseRecovery"; }
        bool run(SNode *root, SNodeFactory &factory, clang::ASTContext &ctx) override;
    };

    // Step 10: Recover while/do-while loops from natural loop info
    class WhileLoopRecoveryPass : public SNodePass {
      public:
        WhileLoopRecoveryPass(const Cfg &cfg, const LoopInfo &loop_info)
            : cfg_(cfg), loop_info_(loop_info) {}
        std::string_view name() const override { return "WhileLoopRecovery"; }
        bool run(SNode *root, SNodeFactory &factory, clang::ASTContext &ctx) override;
      private:
        const Cfg &cfg_;
        const LoopInfo &loop_info_;
    };

    // Step 11: Convert forward gotos into if statements
    class ForwardGotoEliminationPass : public SNodePass {
      public:
        std::string_view name() const override { return "ForwardGotoElimination"; }
        bool run(SNode *root, SNodeFactory &factory, clang::ASTContext &ctx) override;
    };

    // Step 12: Convert backward gotos into do-while loops
    class BackwardGotoToDoWhilePass : public SNodePass {
      public:
        std::string_view name() const override { return "BackwardGotoToDoWhile"; }
        bool run(SNode *root, SNodeFactory &factory, clang::ASTContext &ctx) override;
    };

    // Step 13: Recover switch statements from cascaded if-else chains
    class SwitchRecoveryPass : public SNodePass {
      public:
        std::string_view name() const override { return "SwitchRecovery"; }
        bool run(SNode *root, SNodeFactory &factory, clang::ASTContext &ctx) override;
    };

    // Step 14: Recover short-circuit && and || from nested if-goto patterns
    class ShortCircuitRecoveryPass : public SNodePass {
      public:
        std::string_view name() const override { return "ShortCircuitRecovery"; }
        bool run(SNode *root, SNodeFactory &factory, clang::ASTContext &ctx) override;
    };

    // Step 15: Convert cross-jump gotos within loops to break statements
    class MultiExitBreakPass : public SNodePass {
      public:
        std::string_view name() const override { return "MultiExitBreak"; }
        bool run(SNode *root, SNodeFactory &factory, clang::ASTContext &ctx) override;
    };

    // Step 16: Handle irreducible control flow
    class IrreducibleHandlingPass : public SNodePass {
      public:
        std::string_view name() const override { return "IrreducibleHandling"; }
        bool run(SNode *root, SNodeFactory &factory, clang::ASTContext &ctx) override;
    };

    // Switch-dispatch loop: switch inside SBlock with backedge gotos → while(true) { switch }
    class SwitchBackedgeLoopPass : public SNodePass {
      public:
        std::string_view name() const override { return "SwitchBackedgeLoop"; }
        bool run(SNode *root, SNodeFactory &factory, clang::ASTContext &ctx) override;
    };

    // Inline goto targets from switch case arms into the case bodies
    class SwitchGotoInliningPass : public SNodePass {
      public:
        std::string_view name() const override { return "SwitchGotoInlining"; }
        bool run(SNode *root, SNodeFactory &factory, clang::ASTContext &ctx) override;
    };

    // Convert GNU computed gotos (goto *tbl[sel]) with address-of-label tables to switch
    class IndirectGotoSwitchPass : public SNodePass {
      public:
        std::string_view name() const override { return "IndirectGotoSwitch"; }
        bool run(SNode *root, SNodeFactory &factory, clang::ASTContext &ctx) override;
    };

    // Step 18: Cleanup passes for readability
    class CleanupPass : public SNodePass {
      public:
        std::string_view name() const override { return "Cleanup"; }
        bool run(SNode *root, SNodeFactory &factory, clang::ASTContext &ctx) override;
    };

} // namespace patchestry::ast
