/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/BuildSNodeFromRegion.hpp>

#include <patchestry/AST/CGraph.hpp>
#include <patchestry/AST/SNode.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLFunctionalExtras.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace patchestry::ast {
    namespace {

        constexpr int kGhidraBreakGoto = 2;
        constexpr int kGhidraContinueGoto = 4;

        void SetLoopLabel(
            SNodeFactory &factory, std::string_view label,
            llvm::function_ref< void(std::string_view) > set_label
        ) {
            if (!label.empty()) {
                set_label(factory.Intern(label));
            }
        }

        bool EndsWith(std::string_view value, std::string_view suffix) {
            return value.size() >= suffix.size()
                && value.substr(value.size() - suffix.size()) == suffix;
        }

        bool IsConditionOr(std::string_view opcode) {
            return opcode == "OR" || EndsWith(opcode, "_OR");
        }

        bool IsConditionAnd(std::string_view opcode) {
            return opcode == "AND" || EndsWith(opcode, "_AND");
        }

        clang::Expr *MakeLogicalCondition(
            clang::ASTContext &ctx,
            clang::Expr *lhs,
            clang::Expr *rhs,
            clang::BinaryOperatorKind opcode
        ) {
            auto loc = lhs != nullptr ? lhs->getExprLoc() : VirtualLoc(ctx);
            return clang::BinaryOperator::Create(
                ctx, EnsureRValue(ctx, CloneExpr(ctx, lhs)),
                EnsureRValue(ctx, CloneExpr(ctx, rhs)), opcode, ctx.BoolTy,
                clang::VK_PRValue, clang::OK_Ordinary, loc,
                clang::FPOptionsOverride());
        }

        class ScopedLoopContext {
          public:
            ScopedLoopContext(
                std::vector< size_t > &active_headers,
                std::vector< size_t > &active_break_targets,
                std::optional< size_t > header_idx,
                std::optional< size_t > exit_idx
            )
                : active_headers_(active_headers)
                , active_break_targets_(active_break_targets)
                , has_header_(header_idx.has_value())
                , has_exit_(exit_idx.has_value()) {
                if (header_idx.has_value()) {
                    active_headers_.push_back(*header_idx);
                }
                if (exit_idx.has_value()) {
                    active_break_targets_.push_back(*exit_idx);
                }
            }

            ~ScopedLoopContext() {
                if (has_exit_) {
                    active_break_targets_.pop_back();
                }
                if (has_header_) {
                    active_headers_.pop_back();
                }
            }

          private:
            std::vector< size_t > &active_headers_;
            std::vector< size_t > &active_break_targets_;
            bool has_header_ = false;
            bool has_exit_ = false;
        };

        class ScopedBreakTarget {
          public:
            ScopedBreakTarget(
                std::vector< size_t > &active_break_targets,
                std::optional< size_t > break_idx
            )
                : active_break_targets_(active_break_targets)
                , active_(break_idx.has_value()) {
                if (break_idx.has_value()) {
                    active_break_targets_.push_back(*break_idx);
                }
            }

            ~ScopedBreakTarget() {
                if (active_) {
                    active_break_targets_.pop_back();
                }
            }

          private:
            std::vector< size_t > &active_break_targets_;
            bool active_ = false;
        };

    } // namespace

    BuildSNodeFromRegion::BuildSNodeFromRegion(
        const ghidra::Function &function, CGraph &graph,
        SNodeFactory &factory, clang::ASTContext &ctx
    )
        : function_(function), graph_(graph), factory_(factory), ctx_(ctx) {}

    bool BuildSNodeFromRegion::TryBuild() {
        if (!function_.region.has_value() || graph_.nodes.empty()) {
            return false;
        }

        covered_cnodes_.clear();
        shared_plain_blocks_.clear();
        PrescanSharedPlainBlocks();
        auto body = Translate(*function_.region);
        if (!body.has_value()) {
            return false;
        }

        // Synthetic `:entry` CNode (function prologue) isn't in Ghidra's
        // region tree; prepend its stmts so the prologue isn't dropped.
        const size_t entry_id = graph_.entry;
        const auto &entry_cnode = graph_.Node(entry_id);
        if (!covered_cnodes_.count(entry_id)
            && !function_.entry_block.empty()
            && entry_cnode.source_key == function_.entry_block
            && entry_cnode.succs.size() == 1)
        {
            std::vector< SNode * > prelude;
            prelude.reserve(entry_cnode.stmts.size() + body->size());
            for (clang::Stmt *stmt : entry_cnode.stmts) {
                if (stmt != nullptr) {
                    prelude.push_back(factory_.Make< SStmt >(stmt));
                }
            }
            for (SNode *s : *body) {
                prelude.push_back(s);
            }
            *body = std::move(prelude);
            covered_cnodes_.insert(entry_id);
        }

        // Uncovered CNodes would silently drop statements — fall back loudly.
        // Only reachable CNodes matter: CGraphBuilder appends unreachable blocks
        // to graph_.nodes (see CGraphBuilder.cpp compute_rpo_from_function), but
        // Ghidra's region tree only covers blocks reachable from the entry.
        // Comparing against the full node count would spuriously fall back on any
        // function containing dead code, so count nodes reachable from the entry.
        std::unordered_set< size_t > reachable;
        std::vector< size_t > worklist{ graph_.entry };
        reachable.insert(graph_.entry);
        while (!worklist.empty()) {
            const size_t id = worklist.back();
            worklist.pop_back();
            for (size_t succ : graph_.Node(id).succs) {
                if (reachable.insert(succ).second) {
                    worklist.push_back(succ);
                }
            }
        }
        for (size_t id : reachable) {
            if (!covered_cnodes_.count(id)) {
                LOG(WARNING) << "Ghidra region tree covers "
                             << covered_cnodes_.size() << "/" << reachable.size()
                             << " reachable CNodes for " << function_.name
                             << "; falling back to CFGStructure\n";
                return false;
            }
        }

        // Collapse every CNode into the entry; entry-first so IdentifyInternal
        // picks it as the representative.
        std::vector< size_t > ids;
        ids.reserve(graph_.nodes.size());
        ids.push_back(graph_.entry);
        for (const auto &n : graph_.nodes) {
            if (n.id != graph_.entry) {
                ids.push_back(n.id);
            }
        }

        graph_.IdentifyInternal(ids, CNode::BlockType::kSequence, std::move(*body));
        return true;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::Translate(const ghidra::RegionNode &node) {
        // Kind vocabulary: PcodeBlock::typeToName (see PcodeOperations.hpp).
        if (node.kind == "plain") {
            return TranslatePlain(node);
        }
        if (node.kind == "graph" || node.kind == "list") {
            return TranslateChildSeq(node);
        }
        if (node.kind == "switch") {
            return TranslateSwitch(node);
        }
        if (node.kind == "properif") {
            return TranslateProperIf(node);
        }
        if (node.kind == "ifgoto") {
            return TranslateIfGoto(node);
        }
        if (node.kind == "ifelse") {
            return TranslateIfElse(node);
        }
        if (node.kind == "whiledo") {
            return TranslateWhileDo(node);
        }
        if (node.kind == "dowhile") {
            return TranslateDoWhile(node);
        }
        if (node.kind == "infloop") {
            return TranslateInfLoop(node);
        }
        if (node.kind == "goto") {
            return TranslateGoto(node);
        }
        // multigoto/condition: no trusted transformation — fall back loudly.
        if (node.kind == "multigoto" || node.kind == "condition") {
            LOG(WARNING) << "BuildSNodeFromRegion: kind '" << node.kind
                         << "' not handled in " << function_.name
                         << "; falling back to CFGStructure\n";
            return std::nullopt;
        }
        return std::nullopt;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslatePlain(const ghidra::RegionNode &node) {
        if (!node.block.has_value()) {
            LOG(WARNING) << "plain region node missing block label in "
                         << function_.name << "\n";
            return std::nullopt;
        }
        auto cnode_idx = FindCNode(*node.block);
        if (!cnode_idx.has_value()) {
            return std::nullopt;
        }
        const bool is_shared = shared_plain_blocks_.count(*cnode_idx) > 0;
        const auto &cnode = graph_.Node(*cnode_idx);
        if (!covered_cnodes_.insert(*cnode_idx).second) {
            // Shared block re-emitted: goto the first emission's label
            // (body + terminal were emitted on the first encounter).
            if (is_shared && !cnode.original_label.empty()) {
                std::vector< SNode * > out;
                out.push_back(factory_.Make< SGoto >(
                    factory_.Intern(cnode.original_label)));
                return out;
            }
            LOG(WARNING) << "Ghidra region tree references CNode "
                         << *cnode_idx << " twice in " << function_.name
                         << "; falling back to CFGStructure\n";
            return std::nullopt;
        }
        std::vector< SNode * > stmts;
        stmts.reserve(cnode.stmts.size() + 1);
        for (clang::Stmt *stmt : cnode.stmts) {
            if (stmt != nullptr) {
                stmts.push_back(factory_.Make< SStmt >(stmt));
            }
        }
        // Shared blocks need the terminal so the goto from the
        // duplicate site reaches the right downstream block.
        if (is_shared && cnode.terminal != nullptr) {
            stmts.push_back(factory_.Make< SStmt >(cnode.terminal));
        }

        // Wrap in SLabel so SGoto references elsewhere resolve;
        // RemoveDeadLabels strips unreferenced wraps so over-wrapping is safe.
        std::vector< SNode * > out;
        if (!cnode.original_label.empty()) {
            out.push_back(factory_.Make< SLabel >(
                factory_.Intern(cnode.original_label), std::move(stmts)));
        } else {
            out = std::move(stmts);
        }
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateChildSeq(const ghidra::RegionNode &node) {
        std::vector< SNode * > out;
        for (const auto &child : node.children) {
            auto child_seq = Translate(child);
            if (!child_seq.has_value()) {
                return std::nullopt;
            }
            for (SNode *s : *child_seq) {
                out.push_back(s);
            }
        }
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateSwitch(const ghidra::RegionNode &node) {
        if (node.children.empty()) {
            return std::nullopt;
        }
        if (!CheckRegionBoundaryResolvable(node)) {
            return std::nullopt;
        }

        // child[0] = dispatcher block (plain leaf w/ BRANCHIND terminal,
        // switch_cases, branch_cond).
        const auto &disp_struct = node.children[0];

        // Multi-block unstructured dispatch wrapped in a switch — emit
        // raw labeled+terminal blocks and let arms translate as gotos.
        if (disp_struct.kind == "multigoto") {
            auto disp_seq = TranslateMultigotoDispatch(disp_struct);
            if (!disp_seq.has_value()) {
                return std::nullopt;
            }
            std::vector< SNode * > out = std::move(*disp_seq);
            for (size_t i = 1; i < node.children.size(); ++i) {
                auto arm = Translate(node.children[i]);
                if (!arm.has_value()) {
                    return std::nullopt;
                }
                for (SNode *s : *arm) {
                    out.push_back(s);
                }
            }
            return out;
        }

        if (disp_struct.kind != "plain" || !disp_struct.block.has_value()) {
            return std::nullopt;
        }
        auto disp_idx = FindCNode(*disp_struct.block);
        if (!disp_idx.has_value()) {
            return std::nullopt;
        }
        const auto &disp = graph_.Node(*disp_idx);
        if (disp.branch_cond == nullptr || disp.switch_cases.empty()) {
            return std::nullopt;
        }

        std::optional< std::string > fallback_block;
        bool serialized_has_default = false;
        if (auto bb_it = function_.basic_blocks.find(disp.source_key);
            bb_it != function_.basic_blocks.end())
        {
            const auto &bb = bb_it->second;
            for (const auto &op_key : bb.ordered_operations) {
                auto op_it = bb.operations.find(op_key);
                if (op_it == bb.operations.end()) {
                    continue;
                }
                const auto &op = op_it->second;
                if (op.switch_cases.empty() && !op.fallback_block.has_value()) {
                    continue;
                }
                fallback_block = op.fallback_block;
                serialized_has_default = std::any_of(
                    op.switch_cases.begin(), op.switch_cases.end(),
                    [](const ghidra::SwitchCase &sc) {
                        return sc.is_default;
                    });
                break;
            }
        }
        std::optional< size_t > fallback_idx;
        if (fallback_block.has_value()) {
            fallback_idx = FindCNode(*fallback_block);
        }

        // Emit pre-discriminant stmts and mark dispatcher covered.
        auto pre = TranslatePlain(disp_struct);
        if (!pre.has_value()) {
            return std::nullopt;
        }

        std::unordered_set< size_t > region_arm_entries;
        for (size_t i = 1; i < node.children.size(); ++i) {
            auto entry_lbl = FirstBlockLabel(node.children[i]);
            if (!entry_lbl.has_value()) {
                continue;
            }
            auto entry_cnode = FindCNode(*entry_lbl);
            if (entry_cnode.has_value()) {
                region_arm_entries.insert(*entry_cnode);
            }
        }

        // Group switch_cases by successor — one group per arm body.
        struct ArmInfo {
            std::vector< std::int64_t > values;
            bool has_default = false;
        };
        std::unordered_map< size_t, ArmInfo > succ_to_arm;
        for (const auto &sc : disp.switch_cases) {
            if (sc.succ_index >= disp.succs.size()) {
                LOG(WARNING) << "switch case " << sc.value
                             << " succ_index " << sc.succ_index
                             << " out of range (succs=" << disp.succs.size()
                             << ") in " << function_.name
                             << "; falling back to CFGStructure\n";
                return std::nullopt;
            }
            const size_t target = disp.succs[sc.succ_index];
            const bool synthetic_external_fallback =
                sc.is_default && !serialized_has_default
                && fallback_idx.has_value() && target == *fallback_idx
                && !region_arm_entries.count(*fallback_idx);
            if (synthetic_external_fallback) {
                continue;
            }

            auto &ai = succ_to_arm[target];
            if (sc.is_default) {
                ai.has_default = true;
            } else {
                ai.values.push_back(sc.value);
            }
        }

        std::vector< size_t > preliminary_non_loop_orphans;
        for (const auto &kv : succ_to_arm) {
            if (region_arm_entries.count(kv.first)) {
                continue;
            }
            if (IsActiveLoopHeader(kv.first)) {
                continue;
            }
            preliminary_non_loop_orphans.push_back(kv.first);
        }
        if (preliminary_non_loop_orphans.size() > 1) {
            LOG(WARNING) << "switch in " << function_.name
                         << " has " << preliminary_non_loop_orphans.size()
                         << " distinct non-loop orphan dispatch targets; "
                         << "falling back to CFGStructure\n";
            return std::nullopt;
        }
        std::optional< size_t > switch_break_target;
        if (!preliminary_non_loop_orphans.empty()) {
            switch_break_target = preliminary_non_loop_orphans.front();
        }
        ScopedBreakTarget switch_break_scope(
            active_break_targets_, switch_break_target);

        clang::QualType case_type = disp.branch_cond->getType();
        if (case_type->isEnumeralType()) {
            case_type = case_type->castAs< clang::EnumType >()
                            ->getDecl()->getIntegerType();
        }

        auto *sw = factory_.Make< SSwitch >(disp.branch_cond);

        // Track arm-consumed dispatcher succs; the rest are "orphans"
        // (dispatch targets sitting outside the switch in Ghidra's tree —
        // typically the post-switch merge) that need `case X: break;`.
        std::unordered_set< size_t > consumed_succs;

        for (size_t i = 1; i < node.children.size(); ++i) {
            const auto &arm_struct = node.children[i];

            auto body_seq = Translate(arm_struct);
            if (!body_seq.has_value()) {
                return std::nullopt;
            }

            auto entry_lbl = FirstBlockLabel(arm_struct);
            if (!entry_lbl.has_value()) {
                return std::nullopt;
            }
            auto entry_cnode = FindCNode(*entry_lbl);
            if (!entry_cnode.has_value()) {
                return std::nullopt;
            }

            auto arm_it = succ_to_arm.find(*entry_cnode);
            if (arm_it == succ_to_arm.end()) {
                LOG(WARNING) << "switch arm entry " << *entry_lbl
                             << " has no matching switch_case in "
                             << function_.name << "; falling back\n";
                return std::nullopt;
            }
            consumed_succs.insert(*entry_cnode);
            const auto &arm = arm_it->second;

            const size_t total_labels = arm.values.size()
                                        + (arm.has_default ? 1u : 0u);
            if (total_labels == 0) {
                continue;
            }

            // ClangEmitter places default after all case labels, so the
            // body sits at the last position (default if any, else last value).
            const size_t body_pos = total_labels - 1;
            size_t pos = 0;
            for (std::int64_t v : arm.values) {
                auto *val = MakeIntLiteralPrintable(
                    ctx_, static_cast< uint64_t >(v), case_type,
                    /*is_signed=*/true, VirtualLoc(ctx_));
                if (pos == body_pos) {
                    sw->AddCase(val, *body_seq);
                } else {
                    sw->AddCase(val, std::vector< SNode * >{});
                }
                ++pos;
            }
            if (arm.has_default) {
                if (pos == body_pos) {
                    sw->SetDefaultBody(*body_seq);
                } else {
                    sw->SetDefaultBody(std::vector< SNode * >{});
                }
            }
        }

        // Orphans (targets outside the switch region, typically the
        // post-switch merge) get `case X: break;`. Earlier labels fall
        // through to a tail SBreak so emit-order changes can't silently
        // bleed into the next case. Sort by case value for stable output.
        std::vector< std::pair< size_t, const ArmInfo * > > orphans;
        for (const auto &kv : succ_to_arm) {
            if (!consumed_succs.count(kv.first)) {
                orphans.emplace_back(kv.first, &kv.second);
            }
        }
        std::sort(orphans.begin(), orphans.end(),
                  [](const auto &a, const auto &b) {
                      auto av = a.second->values.empty() ? 0 : a.second->values.front();
                      auto bv = b.second->values.empty() ? 0 : b.second->values.front();
                      return av < bv;
                  });
        auto emit_orphan_arm = [&](const ArmInfo &arm, SNode *terminator) {
            const size_t total_labels = arm.values.size()
                                        + (arm.has_default ? 1u : 0u);
            if (total_labels == 0) {
                return;
            }
            const size_t body_pos = total_labels - 1;
            size_t pos = 0;
            for (std::int64_t v : arm.values) {
                auto *val = MakeIntLiteralPrintable(
                    ctx_, static_cast< uint64_t >(v), case_type,
                    /*is_signed=*/true, VirtualLoc(ctx_));
                if (pos == body_pos) {
                    sw->AddCase(val, std::vector< SNode * >{terminator});
                } else {
                    sw->AddCase(val, std::vector< SNode * >{});
                }
                ++pos;
            }
            if (arm.has_default && sw->DefaultBodyList().empty()) {
                if (pos == body_pos) {
                    sw->SetDefaultBody(std::vector< SNode * >{terminator});
                } else {
                    sw->SetDefaultBody(std::vector< SNode * >{});
                }
            }
        };

        std::vector< std::pair< size_t, const ArmInfo * > > non_loop_orphans;
        for (const auto &kv : orphans) {
            const auto &target_node = graph_.Node(kv.first);
            std::string target_label = target_node.original_label.empty()
                                           ? target_node.source_key
                                           : target_node.original_label;
            if (IsActiveLoopHeader(kv.first)
                || IsInnermostBreakTarget(kv.first))
            {
                emit_orphan_arm(
                    *kv.second,
                    MakeStructuredExitNode(
                        std::nullopt, kv.first, target_label));
            } else {
                non_loop_orphans.push_back(kv);
            }
        }

        // At most one non-loop orphan is the post-switch merge; >1 means
        // the rest would silently jump to the wrong continuation.
        if (non_loop_orphans.size() > 1) {
            LOG(WARNING) << "switch in " << function_.name
                         << " has " << non_loop_orphans.size()
                         << " distinct non-loop orphan dispatch targets; "
                         << "falling back to CFGStructure\n";
            return std::nullopt;
        }
        for (const auto &kv : non_loop_orphans) {
            const auto &target_node = graph_.Node(kv.first);
            std::string target_label = target_node.original_label.empty()
                                           ? target_node.source_key
                                           : target_node.original_label;
            emit_orphan_arm(
                *kv.second,
                MakeStructuredExitNode(std::nullopt, kv.first, target_label));
        }

        auto out = std::move(*pre);
        out.push_back(sw);
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateMultigotoDispatch(
        const ghidra::RegionNode &mg)
    {
        // Each plain child's CBRANCH/BRANCH terminal is kept as raw
        // SStmt so NormalizeRawControlFlow can lift it later.
        std::vector< SNode * > out;
        for (const auto &child : mg.children) {
            if (child.kind != "plain" || !child.block.has_value()) {
                LOG(WARNING) << "multigoto in " << function_.name
                             << " has non-plain child kind '" << child.kind
                             << "'; falling back\n";
                return std::nullopt;
            }
            auto idx = FindCNode(*child.block);
            if (!idx.has_value()) {
                return std::nullopt;
            }
            if (!covered_cnodes_.insert(*idx).second) {
                LOG(WARNING) << "multigoto plain child " << *child.block
                             << " referenced twice in " << function_.name
                             << "; falling back\n";
                return std::nullopt;
            }
            const auto &cnode = graph_.Node(*idx);
            std::vector< SNode * > body;
            body.reserve(cnode.stmts.size() + 1);
            for (clang::Stmt *s : cnode.stmts) {
                if (s != nullptr) {
                    body.push_back(factory_.Make< SStmt >(s));
                }
            }
            if (cnode.terminal != nullptr) {
                body.push_back(factory_.Make< SStmt >(cnode.terminal));
            }
            if (!cnode.original_label.empty()) {
                out.push_back(factory_.Make< SLabel >(
                    factory_.Intern(cnode.original_label), std::move(body)));
            } else {
                for (SNode *s : body) {
                    out.push_back(s);
                }
            }
        }
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateProperIf(const ghidra::RegionNode &node) {
        if (node.children.size() != 2) {
            return std::nullopt;
        }
        if (!CheckRegionBoundaryResolvable(node)) {
            return std::nullopt;
        }
        const auto &cond_struct = node.children[0];
        const auto &body_struct = node.children[1];

        auto head = TranslateCondHead(cond_struct);
        if (!head.has_value()) {
            return std::nullopt;
        }
        if (head->succs.size() != 2 || head->branch_cond == nullptr) {
            return std::nullopt;
        }

        auto body_seq = Translate(body_struct);
        if (!body_seq.has_value()) {
            return std::nullopt;
        }

        auto body_entry_lbl = FirstBlockLabel(body_struct);
        if (!body_entry_lbl.has_value()) {
            return std::nullopt;
        }
        auto body_entry = FindCNode(*body_entry_lbl);
        if (!body_entry.has_value()) {
            return std::nullopt;
        }

        // succs[0]=not-taken, succs[1]=taken. Body on succs[1] uses cond
        // as-is; body on succs[0] negates.
        clang::Expr *if_cond = head->branch_cond;
        if (head->succs[1] == *body_entry) {
        } else if (head->succs[0] == *body_entry) {
            if_cond = NegateExpr(ctx_, head->branch_cond);
        } else {
            LOG(WARNING) << "properif body entry " << *body_entry_lbl
                         << " matches neither succ of cond block in "
                         << function_.name << "; falling back\n";
            return std::nullopt;
        }

        auto *if_node = factory_.Make< SIfThenElse >(
            if_cond, std::move(*body_seq), std::vector< SNode * >{});
        auto out = std::move(head->pre);
        out.push_back(if_node);
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateIfGoto(const ghidra::RegionNode &node) {
        if (node.children.size() != 1) {
            return std::nullopt;
        }
        const auto &cond_struct = node.children[0];

        auto head = TranslateCondHead(cond_struct);
        if (!head.has_value()) {
            return std::nullopt;
        }
        if (head->succs.size() != 2 || head->branch_cond == nullptr) {
            return std::nullopt;
        }

        // Prefer Ghidra's explicit BlockIfGoto target; older JSON
        // lacks it, fall back to successor-order heuristic.
        size_t goto_succ_idx = 1;
        clang::Expr *if_cond = head->branch_cond;
        std::optional< std::string > explicit_target_label;
        std::optional< std::string > normalized_target_label;
        if (node.goto_targets.size() == 1
            && !node.goto_targets[0].empty())
        {
            explicit_target_label = node.goto_targets[0];
        }

        if (explicit_target_label.has_value()) {
            auto target_idx = FindCNode(*explicit_target_label);
            if (!target_idx.has_value()) {
                LOG(WARNING) << "ifgoto explicit target "
                             << *explicit_target_label << " not found in "
                             << function_.name << "; falling back\n";
                return std::nullopt;
            }
            const auto &target_cnode = graph_.Node(*target_idx);
            if (target_cnode.original_label.empty()) {
                LOG(WARNING) << "ifgoto explicit target "
                             << *explicit_target_label
                             << " has no original_label in "
                             << function_.name << "; falling back\n";
                return std::nullopt;
            }
            normalized_target_label = target_cnode.original_label;
            if (head->succs[1] == *target_idx) {
                goto_succ_idx = 1;
            } else if (head->succs[0] == *target_idx) {
                goto_succ_idx = 0;
                if_cond = NegateExpr(ctx_, head->branch_cond);
            } else {
                LOG(WARNING) << "ifgoto explicit target "
                             << *explicit_target_label
                             << " matches neither succ of cond block in "
                             << function_.name << "; falling back\n";
                return std::nullopt;
            }
        } else {
            // ifgoto polarity isn't uniform. Default succs[1]=goto;
            // when ifgoto is a properif body and succs[1] matches the
            // properif fallthrough, succs[1] is the in-region next-block
            // and the goto must flip to succs[0] (see ProperIfFallthroughBlock).
            auto fallthrough = ProperIfFallthroughBlock(node);
            if (fallthrough.has_value() && head->succs[1] == *fallthrough) {
                goto_succ_idx = 0;
                if_cond = NegateExpr(ctx_, head->branch_cond);
            }
        }

        std::string target_label = normalized_target_label.value_or(
            graph_.Node(head->succs[goto_succ_idx]).original_label);
        if (target_label.empty()) {
            LOG(WARNING) << "ifgoto goto target lacks label in "
                         << function_.name << "; falling back\n";
            return std::nullopt;
        }

        const size_t target_idx = head->succs[goto_succ_idx];
        auto *goto_node = MakeStructuredExitNode(
            node.goto_type, target_idx, target_label);
        auto *if_node = factory_.Make< SIfThenElse >(
            if_cond, goto_node, nullptr);
        auto out = std::move(head->pre);
        out.push_back(if_node);
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateIfElse(const ghidra::RegionNode &node) {
        if (node.children.size() != 3) {
            return std::nullopt;
        }
        if (!CheckRegionBoundaryResolvable(node)) {
            return std::nullopt;
        }
        const auto &cond_struct = node.children[0];
        const auto &arm1_struct = node.children[1];
        const auto &arm2_struct = node.children[2];

        auto head = TranslateCondHead(cond_struct);
        if (!head.has_value()) {
            return std::nullopt;
        }
        if (head->succs.size() != 2 || head->branch_cond == nullptr) {
            return std::nullopt;
        }

        auto arm1_seq = Translate(arm1_struct);
        if (!arm1_seq.has_value()) {
            return std::nullopt;
        }
        auto arm2_seq = Translate(arm2_struct);
        if (!arm2_seq.has_value()) {
            return std::nullopt;
        }

        auto arm1_lbl = FirstBlockLabel(arm1_struct);
        auto arm2_lbl = FirstBlockLabel(arm2_struct);
        if (!arm1_lbl.has_value() || !arm2_lbl.has_value()) {
            return std::nullopt;
        }
        auto arm1_entry = FindCNode(*arm1_lbl);
        auto arm2_entry = FindCNode(*arm2_lbl);
        if (!arm1_entry.has_value() || !arm2_entry.has_value()) {
            return std::nullopt;
        }

        // Pair succs[1]=then, succs[0]=else; preserves CBRANCH cond shape.
        std::vector< SNode * > then_body;
        std::vector< SNode * > else_body;
        if (*arm1_entry == head->succs[1]
            && *arm2_entry == head->succs[0])
        {
            then_body = std::move(*arm1_seq);
            else_body = std::move(*arm2_seq);
        } else if (*arm2_entry == head->succs[1]
                   && *arm1_entry == head->succs[0])
        {
            then_body = std::move(*arm2_seq);
            else_body = std::move(*arm1_seq);
        } else {
            LOG(WARNING) << "ifelse arms (" << *arm1_lbl << ", " << *arm2_lbl
                         << ") do not cover both cond successors in "
                         << function_.name << "; falling back\n";
            return std::nullopt;
        }

        auto *if_node = factory_.Make< SIfThenElse >(
            head->branch_cond, std::move(then_body), std::move(else_body));
        auto out = std::move(head->pre);
        out.push_back(if_node);
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateWhileDo(const ghidra::RegionNode &node) {
        if (node.children.size() != 2) {
            return std::nullopt;
        }
        if (!CheckRegionBoundaryResolvable(node)) {
            return std::nullopt;
        }
        const auto &cond_struct = node.children[0];
        const auto &body_struct = node.children[1];

        // Fast path: plain cond with no stmts → simple `while (cond) body`.
        if (cond_struct.kind == "plain" && cond_struct.block.has_value()) {
            auto cond_idx = FindCNode(*cond_struct.block);
            if (cond_idx.has_value()) {
                const auto &cond = graph_.Node(*cond_idx);
                if (cond.is_conditional && cond.succs.size() == 2
                    && cond.branch_cond != nullptr && cond.stmts.empty())
                {
                    if (!covered_cnodes_.insert(*cond_idx).second) {
                        LOG(WARNING) << "whiledo cond block "
                                     << *cond_struct.block
                                     << " referenced twice in "
                                     << function_.name
                                     << "; falling back\n";
                        return std::nullopt;
                    }

                    auto body_lbl = FirstBlockLabel(body_struct);
                    if (!body_lbl.has_value()) {
                        return std::nullopt;
                    }
                    auto body_entry = FindCNode(*body_lbl);
                    if (!body_entry.has_value()) {
                        return std::nullopt;
                    }

                    clang::Expr *while_cond = cond.branch_cond;
                    size_t exit_succ_idx = 0;
                    if (cond.succs[1] == *body_entry) {
                        exit_succ_idx = 0;
                    } else if (cond.succs[0] == *body_entry) {
                        while_cond = NegateExpr(ctx_, cond.branch_cond);
                        exit_succ_idx = 1;
                    } else {
                        LOG(WARNING) << "whiledo body entry " << *body_lbl
                                     << " doesn't match cond succs in "
                                     << function_.name << "; falling back\n";
                        return std::nullopt;
                    }

                    ScopedLoopContext loop_scope(
                        active_loop_headers_, active_break_targets_,
                        cond_idx, cond.succs[exit_succ_idx]);
                    auto body_seq = Translate(body_struct);
                    if (!body_seq.has_value()) {
                        return std::nullopt;
                    }

                    auto *w = factory_.Make< SWhile >(
                        while_cond, std::move(*body_seq));
                    SetLoopLabel(factory_, cond.original_label, [&](std::string_view l) {
                        w->SetHeaderLabel(l);
                    });
                    const auto &exit_node = graph_.Node(cond.succs[exit_succ_idx]);
                    SetLoopLabel(factory_, exit_node.original_label, [&](std::string_view l) {
                        w->SetExitLabel(l);
                    });
                    std::vector< SNode * > out;
                    // SLabel wrap so external gotos resolve; RemoveDeadLabels strips if unused.
                    if (!cond.original_label.empty()) {
                        out.push_back(factory_.Make< SLabel >(
                            factory_.Intern(cond.original_label),
                            std::vector< SNode * >{ w }));
                    } else {
                        out.push_back(w);
                    }
                    return out;
                }
            }
        }

        // Slow path (list cond or plain cond with stmts):
        //   while (1) { head.pre; if (break_cond) break; body }
        // so the pre-cond re-executes each iteration.
        auto head = TranslateCondHead(cond_struct);
        if (!head.has_value()) {
            return std::nullopt;
        }
        if (head->succs.size() != 2 || head->branch_cond == nullptr) {
            return std::nullopt;
        }

        auto body_lbl = FirstBlockLabel(body_struct);
        if (!body_lbl.has_value()) {
            return std::nullopt;
        }
        auto body_entry = FindCNode(*body_lbl);
        if (!body_entry.has_value()) {
            return std::nullopt;
        }

        // Compute directly from branch_cond to avoid emitting `!!cond`.
        clang::Expr *break_cond = nullptr;
        size_t exit_succ_idx = 0;
        if (head->succs[1] == *body_entry) {
            break_cond = NegateExpr(ctx_, head->branch_cond);
            exit_succ_idx = 0;
        } else if (head->succs[0] == *body_entry) {
            break_cond = head->branch_cond;
            exit_succ_idx = 1;
        } else {
            LOG(WARNING) << "whiledo body entry " << *body_lbl
                         << " doesn't match cond succs in "
                         << function_.name << "; falling back\n";
            return std::nullopt;
        }

        ScopedLoopContext loop_scope(
            active_loop_headers_, active_break_targets_, head->entry_idx,
            head->succs[exit_succ_idx]);
        auto body_seq = Translate(body_struct);
        if (!body_seq.has_value()) {
            return std::nullopt;
        }

        std::vector< SNode * > w_body = std::move(head->pre);
        auto *break_node = factory_.Make< SBreak >();
        auto *if_break = factory_.Make< SIfThenElse >(
            break_cond, break_node, /*else=*/nullptr);
        w_body.push_back(if_break);
        for (SNode *s : *body_seq) {
            w_body.push_back(s);
        }

        auto *one = clang::IntegerLiteral::Create(
            ctx_, llvm::APInt(ctx_.getIntWidth(ctx_.IntTy), 1, true),
            ctx_.IntTy, VirtualLoc(ctx_));
        auto *w = factory_.Make< SWhile >(one, std::move(w_body));
        SetLoopLabel(factory_, head->original_label, [&](std::string_view l) {
            w->SetHeaderLabel(l);
        });
        const auto &exit_node = graph_.Node(head->succs[exit_succ_idx]);
        SetLoopLabel(factory_, exit_node.original_label, [&](std::string_view l) {
            w->SetExitLabel(l);
        });
        std::vector< SNode * > out;
        out.push_back(w);
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateDoWhile(const ghidra::RegionNode &node) {
        // CBRANCH tail = rightmost plain leaf; loop-back succ must
        // target body entry (entry==tail for self-loops).
        if (node.children.size() != 1) {
            return std::nullopt;
        }
        if (!CheckRegionBoundaryResolvable(node)) {
            return std::nullopt;
        }
        const auto &body_struct = node.children[0];

        std::vector< const ghidra::RegionNode * > pre_regions;
        const ghidra::RegionNode *cur = &body_struct;
        while (cur->kind == "list") {
            if (cur->children.empty()) {
                return std::nullopt;
            }
            for (size_t i = 0; i + 1 < cur->children.size(); ++i) {
                pre_regions.push_back(&cur->children[i]);
            }
            cur = &cur->children.back();
        }
        if (cur->kind != "plain" || !cur->block.has_value()) {
            return std::nullopt;
        }

        auto tail_idx = FindCNode(*cur->block);
        if (!tail_idx.has_value()) {
            return std::nullopt;
        }
        const auto &tail_cnode = graph_.Node(*tail_idx);
        if (!tail_cnode.is_conditional || tail_cnode.succs.size() != 2
            || tail_cnode.branch_cond == nullptr)
        {
            return std::nullopt;
        }

        auto entry_lbl = FirstBlockLabel(body_struct);
        if (!entry_lbl.has_value()) {
            return std::nullopt;
        }
        auto entry_idx = FindCNode(*entry_lbl);
        if (!entry_idx.has_value()) {
            return std::nullopt;
        }

        bool s0_loop = tail_cnode.succs[0] == *entry_idx;
        bool s1_loop = tail_cnode.succs[1] == *entry_idx;
        if (!s0_loop && !s1_loop) {
            LOG(WARNING) << "dowhile tail block " << *cur->block
                         << " doesn't loop back to body entry in "
                         << function_.name << "; falling back\n";
            return std::nullopt;
        }
        size_t exit_succ_idx = s1_loop ? 0 : 1;

        ScopedLoopContext loop_scope(
            active_loop_headers_, active_break_targets_, tail_idx,
            tail_cnode.succs[exit_succ_idx]);
        std::vector< SNode * > body_seq;
        for (const auto *r : pre_regions) {
            auto seq = Translate(*r);
            if (!seq.has_value()) {
                return std::nullopt;
            }
            for (SNode *s : *seq) {
                body_seq.push_back(s);
            }
        }
        auto tail_seq = TranslatePlain(*cur);
        if (!tail_seq.has_value()) {
            return std::nullopt;
        }
        for (SNode *s : *tail_seq) {
            body_seq.push_back(s);
        }

        clang::Expr *do_cond = s1_loop ? tail_cnode.branch_cond
                                       : NegateExpr(ctx_, tail_cnode.branch_cond);

        auto *dw = factory_.Make< SDoWhile >(std::move(body_seq), do_cond);
        SetLoopLabel(factory_, tail_cnode.original_label, [&](std::string_view l) {
            dw->SetHeaderLabel(l);
        });
        const auto &exit_node = graph_.Node(tail_cnode.succs[exit_succ_idx]);
        SetLoopLabel(factory_, exit_node.original_label, [&](std::string_view l) {
            dw->SetExitLabel(l);
        });
        std::vector< SNode * > out;
        out.push_back(dw);
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateInfLoop(const ghidra::RegionNode &node) {
        if (node.children.size() != 1) {
            return std::nullopt;
        }
        if (!CheckRegionBoundaryResolvable(node)) {
            return std::nullopt;
        }
        const auto &body_struct = node.children[0];

        std::optional< size_t > header_idx;
        if (auto header_lbl = FirstBlockLabel(body_struct)) {
            header_idx = FindCNode(*header_lbl);
        }

        ScopedLoopContext loop_scope(
            active_loop_headers_, active_break_targets_, header_idx,
            std::nullopt);
        auto body_seq = Translate(body_struct);
        if (!body_seq.has_value()) {
            return std::nullopt;
        }

        auto *one = clang::IntegerLiteral::Create(
            ctx_, llvm::APInt(ctx_.getIntWidth(ctx_.IntTy), 1, true),
            ctx_.IntTy, VirtualLoc(ctx_));
        auto *w = factory_.Make< SWhile >(one, std::move(*body_seq));
        if (header_idx.has_value()) {
            SetLoopLabel(
                factory_, graph_.Node(*header_idx).original_label,
                [&](std::string_view l) { w->SetHeaderLabel(l); });
        }
        std::vector< SNode * > out;
        out.push_back(w);
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateGoto(const ghidra::RegionNode &node) {
        if (node.children.size() != 1) {
            return std::nullopt;
        }
        const auto &src_struct = node.children[0];

        // Prefer Ghidra's explicit goto_targets; older JSON requires
        // inferring from a plain child's sole successor.
        std::optional< std::string > target_label;
        if (node.goto_targets.size() == 1
            && !node.goto_targets[0].empty())
        {
            target_label = node.goto_targets[0];
        } else {
            if (src_struct.kind != "plain"
                || !src_struct.block.has_value())
            {
                return std::nullopt;
            }
            auto src_idx = FindCNode(*src_struct.block);
            if (!src_idx.has_value()) {
                return std::nullopt;
            }
            const auto &src = graph_.Node(*src_idx);
            if (src.succs.size() != 1) {
                LOG(WARNING) << "goto source " << *src_struct.block
                             << " has " << src.succs.size()
                             << " successors in " << function_.name
                             << "; falling back\n";
                return std::nullopt;
            }
            const auto &target = graph_.Node(src.succs[0]);
            if (target.original_label.empty()) {
                LOG(WARNING) << "goto target " << target.source_key
                             << " lacks label in " << function_.name
                             << "; falling back\n";
                return std::nullopt;
            }
            // FindCNode matches by source_key (colon form), not original_label.
            target_label = target.source_key;
        }

        // Normalize to original_label (underscore form) so the SGoto
        // name matches the SLabel from TranslatePlain.
        auto target_cnode_idx = FindCNode(*target_label);
        if (!target_cnode_idx.has_value()) {
            LOG(WARNING) << "goto target " << *target_label
                         << " is not a known CNode in " << function_.name
                         << "; falling back\n";
            return std::nullopt;
        }
        const auto &target_cnode = graph_.Node(*target_cnode_idx);
        if (target_cnode.original_label.empty()) {
            LOG(WARNING) << "goto target " << *target_label
                         << " has no original_label in " << function_.name
                         << "; falling back\n";
            return std::nullopt;
        }
        std::string goto_name = target_cnode.original_label;

        auto body = Translate(src_struct);
        if (!body.has_value()) {
            return std::nullopt;
        }

        auto *goto_node = MakeStructuredExitNode(
            node.goto_type, target_cnode_idx, goto_name);
        auto out = std::move(*body);
        out.push_back(goto_node);
        return out;
    }

    std::optional< BuildSNodeFromRegion::CondHead >
    BuildSNodeFromRegion::TranslateCondHead(
        const ghidra::RegionNode &cond_struct)
    {
        if (cond_struct.kind == "condition") {
            return TranslateConditionHead(cond_struct);
        }

        if (cond_struct.kind == "plain") {
            if (!cond_struct.block.has_value()) {
                return std::nullopt;
            }
            auto idx = FindCNode(*cond_struct.block);
            if (!idx.has_value()) {
                return std::nullopt;
            }
            auto plain_seq = TranslatePlain(cond_struct);
            if (!plain_seq.has_value()) {
                return std::nullopt;
            }
            const auto &cond = graph_.Node(*idx);
            if (!cond.is_conditional || cond.succs.size() != 2
                || cond.branch_cond == nullptr)
            {
                return std::nullopt;
            }
            return CondHead{
                std::move(*plain_seq),
                *idx,
                *idx,
                cond.branch_cond,
                cond.succs,
                cond.original_label};
        }
        // `list` cond: pre-cond regions + terminating CBRANCH plain leaf.
        if (cond_struct.kind != "list") {
            return std::nullopt;
        }
        std::vector< const ghidra::RegionNode * > pre_regions;
        const ghidra::RegionNode *cur = &cond_struct;
        while (cur->kind == "list") {
            if (cur->children.size() < 2) {
                return std::nullopt;
            }
            for (size_t i = 0; i + 1 < cur->children.size(); ++i) {
                pre_regions.push_back(&cur->children[i]);
            }
            cur = &cur->children.back();
        }
        if (cur->kind != "plain" || !cur->block.has_value()) {
            return std::nullopt;
        }
        auto cond_idx = FindCNode(*cur->block);
        if (!cond_idx.has_value()) {
            return std::nullopt;
        }

        std::vector< SNode * > pre;
        for (const auto *region : pre_regions) {
            auto seq = Translate(*region);
            if (!seq.has_value()) {
                return std::nullopt;
            }
            for (SNode *s : *seq) {
                pre.push_back(s);
            }
        }
        auto cond_seq = TranslatePlain(*cur);
        if (!cond_seq.has_value()) {
            return std::nullopt;
        }
        for (SNode *s : *cond_seq) {
            pre.push_back(s);
        }
        const auto &cond = graph_.Node(*cond_idx);
        if (!cond.is_conditional || cond.succs.size() != 2
            || cond.branch_cond == nullptr)
        {
            return std::nullopt;
        }
        auto entry_lbl = FirstBlockLabel(cond_struct);
        if (!entry_lbl.has_value()) {
            return std::nullopt;
        }
        auto entry_idx = FindCNode(*entry_lbl);
        if (!entry_idx.has_value()) {
            return std::nullopt;
        }
        return CondHead{
            std::move(pre),
            *cond_idx,
            *entry_idx,
            cond.branch_cond,
            cond.succs,
            cond.original_label};
    }

    std::optional< BuildSNodeFromRegion::CondHead >
    BuildSNodeFromRegion::TranslateConditionHead(
        const ghidra::RegionNode &cond_struct)
    {
        if (cond_struct.children.size() != 2
            || !cond_struct.condition_opcode.has_value())
        {
            return std::nullopt;
        }

        auto lhs = TranslateCondHead(cond_struct.children[0]);
        auto rhs = TranslateCondHead(cond_struct.children[1]);
        if (!lhs.has_value() || !rhs.has_value()
            || lhs->succs.size() != 2 || rhs->succs.size() != 2
            || lhs->branch_cond == nullptr || rhs->branch_cond == nullptr)
        {
            return std::nullopt;
        }

        std::vector< SNode * > pre = std::move(lhs->pre);
        for (SNode *s : rhs->pre) {
            pre.push_back(s);
        }

        if (IsConditionOr(*cond_struct.condition_opcode)) {
            if (lhs->succs[0] != rhs->entry_idx) {
                LOG(WARNING) << "condition OR false edge does not enter rhs "
                             << "condition in " << function_.name
                             << "; falling back\n";
                return std::nullopt;
            }
            // Both A-true (lhs->succs[1]) and B-true (rhs->succs[1]) must
            // reach the same destination for `A || B` to fold into one
            // 2-way conditional; the combined head exports lhs->succs[1].
            if (lhs->succs[1] != rhs->succs[1]) {
                LOG(WARNING) << "condition OR true edges differ in "
                             << function_.name << "; falling back\n";
                return std::nullopt;
            }
            return CondHead{
                std::move(pre),
                rhs->cond_idx,
                lhs->entry_idx,
                MakeLogicalCondition(
                    ctx_, lhs->branch_cond, rhs->branch_cond, clang::BO_LOr),
                {rhs->succs[0], lhs->succs[1]},
                lhs->original_label};
        }

        if (IsConditionAnd(*cond_struct.condition_opcode)) {
            if (lhs->succs[1] != rhs->entry_idx) {
                LOG(WARNING) << "condition AND true edge does not enter rhs "
                             << "condition in " << function_.name
                             << "; falling back\n";
                return std::nullopt;
            }
            // Both A-false (lhs->succs[0]) and B-false (rhs->succs[0]) must
            // reach the same destination for `A && B` to fold into one
            // 2-way conditional; the combined head exports lhs->succs[0].
            if (lhs->succs[0] != rhs->succs[0]) {
                LOG(WARNING) << "condition AND false edges differ in "
                             << function_.name << "; falling back\n";
                return std::nullopt;
            }
            return CondHead{
                std::move(pre),
                rhs->cond_idx,
                lhs->entry_idx,
                MakeLogicalCondition(
                    ctx_, lhs->branch_cond, rhs->branch_cond, clang::BO_LAnd),
                {lhs->succs[0], rhs->succs[1]},
                lhs->original_label};
        }

        LOG(WARNING) << "unsupported condition opcode "
                     << *cond_struct.condition_opcode << " in "
                     << function_.name << "; falling back\n";
        return std::nullopt;
    }

    std::optional< std::string >
    BuildSNodeFromRegion::FirstBlockLabel(const ghidra::RegionNode &node) const {
        if (node.kind == "plain") {
            return node.block;
        }
        for (const auto &c : node.children) {
            if (auto r = FirstBlockLabel(c)) {
                return r;
            }
        }
        return std::nullopt;
    }

    std::optional< size_t >
    BuildSNodeFromRegion::FindCNode(const std::string &block_label) const {
        for (const auto &node : graph_.nodes) {
            if (node.source_key == block_label) {
                return node.id;
            }
        }
        return std::nullopt;
    }

    bool BuildSNodeFromRegion::IsActiveLoopHeader(size_t idx) const {
        return std::find(
                   active_loop_headers_.begin(), active_loop_headers_.end(), idx)
               != active_loop_headers_.end();
    }

    bool BuildSNodeFromRegion::IsInnermostBreakTarget(size_t idx) const {
        return !active_break_targets_.empty()
            && active_break_targets_.back() == idx;
    }

    SNode *BuildSNodeFromRegion::MakeStructuredExitNode(
        std::optional< int > goto_type,
        std::optional< size_t > target_idx,
        std::string_view target_label)
    {
        auto targets_innermost_loop_header = [&]() {
            return target_idx.has_value() && !active_loop_headers_.empty()
                && active_loop_headers_.back() == *target_idx;
        };

        if (target_idx.has_value()) {
            if (targets_innermost_loop_header()) {
                return factory_.Make< SContinue >();
            }
            if (IsInnermostBreakTarget(*target_idx)) {
                return factory_.Make< SBreak >();
            }
        }

        const bool target_unknown = !target_idx.has_value();
        if (goto_type.has_value() && *goto_type == kGhidraContinueGoto
            && (target_unknown || targets_innermost_loop_header()))
        {
            return factory_.Make< SContinue >();
        }
        if (goto_type.has_value() && *goto_type == kGhidraBreakGoto
            && (target_unknown || IsInnermostBreakTarget(*target_idx)))
        {
            return factory_.Make< SBreak >();
        }

        return factory_.Make< SGoto >(factory_.Intern(target_label));
    }

    std::optional< BuildSNodeFromRegion::RegionBoundary >
    BuildSNodeFromRegion::AnalyzeRegionBoundary(
        const ghidra::RegionNode &node) const
    {
        RegionBoundary boundary;

        auto entry_label = FirstBlockLabel(node);
        if (entry_label.has_value()) {
            auto entry_idx = FindCNode(*entry_label);
            if (!entry_idx.has_value()) {
                LOG(WARNING) << "region entry block " << *entry_label
                             << " is not a known CNode in " << function_.name
                             << "\n";
                return std::nullopt;
            }
            boundary.entry = *entry_idx;
        }

        std::function< void(const ghidra::RegionNode &) > collect =
            [&](const ghidra::RegionNode &cur) {
                if (cur.kind == "plain" && cur.block.has_value()) {
                    auto idx = FindCNode(*cur.block);
                    if (idx.has_value()) {
                        boundary.blocks.insert(*idx);
                    }
                }
                for (const auto &child : cur.children) {
                    collect(child);
                }
            };
        collect(node);

        if (boundary.blocks.empty() || !boundary.entry.has_value()) {
            LOG(WARNING) << "region kind '" << node.kind << "' in "
                         << function_.name
                         << " has no resolvable entry block\n";
            return std::nullopt;
        }

        for (size_t idx : boundary.blocks) {
            const auto &cnode = graph_.Node(idx);
            for (size_t pred : cnode.preds) {
                if (!boundary.blocks.count(pred) && idx != *boundary.entry) {
                    boundary.external_entries.emplace_back(pred, idx);
                }
            }
            for (size_t succ : cnode.succs) {
                if (!boundary.blocks.count(succ)) {
                    boundary.exit_edges.emplace_back(idx, succ);
                    boundary.exit_targets.insert(succ);
                }
            }
        }

        return boundary;
    }

    bool BuildSNodeFromRegion::CheckRegionBoundaryResolvable(
        const ghidra::RegionNode &node) const
    {
        auto boundary = AnalyzeRegionBoundary(node);
        return boundary.has_value();
    }

    void BuildSNodeFromRegion::PrescanSharedPlainBlocks() {
        if (!function_.region.has_value()) {
            return;
        }
        std::unordered_map< size_t, int > counts;
        std::function< void(const ghidra::RegionNode &) > walk =
            [&](const ghidra::RegionNode &n) {
                if (n.kind == "plain" && n.block.has_value()) {
                    if (auto idx = FindCNode(*n.block)) {
                        ++counts[*idx];
                    }
                }
                for (const auto &c : n.children) {
                    walk(c);
                }
            };
        walk(*function_.region);
        for (const auto &kv : counts) {
            if (kv.second > 1) {
                shared_plain_blocks_.insert(kv.first);
            }
        }
    }

    std::optional< size_t >
    BuildSNodeFromRegion::ProperIfFallthroughBlock(
        const ghidra::RegionNode &node) const
    {
        if (!function_.region.has_value()) {
            return std::nullopt;
        }
        std::vector<
            std::pair< const ghidra::RegionNode *, size_t > > parents;
        std::function< bool(const ghidra::RegionNode *) > walk;
        walk = [&](const ghidra::RegionNode *cur) -> bool {
            if (cur == &node) {
                return true;
            }
            for (size_t i = 0; i < cur->children.size(); ++i) {
                parents.emplace_back(cur, i);
                if (walk(&cur->children[i])) {
                    return true;
                }
                parents.pop_back();
            }
            return false;
        };
        if (!walk(&*function_.region)) {
            return std::nullopt;
        }
        if (parents.empty()) {
            return std::nullopt;
        }
        // Walk outward for the next lexical-sibling block. Stop at loop
        // boundaries (falling out is "next iteration", not a neighbour).
        // Skip past properif/ifelse/switch parents — their children are
        // mutually exclusive arms; the construct's own lexical successor
        // is at the grandparent.
        for (auto it = parents.rbegin(); it != parents.rend(); ++it) {
            const auto *parent = it->first;
            const auto &kind = parent->kind;
            if (kind == "whiledo" || kind == "dowhile"
                || kind == "infloop")
            {
                return std::nullopt;
            }
            if (kind == "properif" || kind == "ifelse"
                || kind == "switch")
            {
                continue;
            }
            size_t idx = it->second;
            if (idx + 1 < parent->children.size()) {
                auto lbl =
                    FirstBlockLabel(parent->children[idx + 1]);
                if (!lbl.has_value()) {
                    return std::nullopt;
                }
                return FindCNode(*lbl);
            }
        }
        return std::nullopt;
    }

} // namespace patchestry::ast
