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
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <llvm/ADT/APInt.h>
#include <llvm/Support/Casting.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace patchestry::ast {

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
        auto body = Translate(*function_.region);
        if (!body.has_value()) {
            return false;
        }

        // Synthetic-entry prelude: CGraphBuilder seeds a `:entry` CNode
        // that holds the function prologue (allocas, parameter loads)
        // and unconditionally branches to the first real basic block.
        // Ghidra's DecompInterface structureGraph never sees this
        // synthetic block — it operates on the original BlockGraph.
        // Detect the mismatch and prepend the entry's stmts to the
        // translated body so we don't drop the prologue.  The entry's
        // terminal (a goto to its sole successor) is dropped because
        // the structured output emits the successor immediately next —
        // identical to how CFGStructure handles entry-block fallthrough.
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

        // Coverage check: every CNode must be reachable from Ghidra's
        // region tree (after synthetic-entry handling above).
        // CGraphBuilder and Ghidra may disagree on block boundaries
        // for other reasons too (e.g., unreachable blocks).  Silently
        // dropping uncovered CNodes would lose statements; per loud-
        // failure, we fall back instead.
        if (covered_cnodes_.size() != graph_.nodes.size()) {
            LOG(WARNING) << "Ghidra region tree covers "
                         << covered_cnodes_.size() << "/" << graph_.nodes.size()
                         << " CNodes for " << function_.name
                         << "; falling back to CFGStructure\n";
            return false;
        }

        // Seed the whole function: collapse every CNode into the entry,
        // with the translated tree as its structured sequence.  The post-
        // pass chain in ASTConsumer walks the SNode tree directly and is
        // agnostic to which CNode owns it.  We put the entry first so
        // IdentifyInternal picks it as the representative.
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
        // Kind vocabulary (PcodeBlock::typeToName) lives on RegionNode
        // — see Ghidra/PcodeOperations.hpp:243.  Phase 1 handles the
        // straight-line trio; everything else falls back per-function.
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
        // multigoto/condition: rare in practice (4/0 instances in
        // bloodview), no clean transformation we trust without test
        // coverage.  Fall back loudly per loud-failure: the function-
        // level coverage check will redirect to CFGStructure.
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
            // Ghidra named a block we don't have a CNode for.  Treat as
            // failure so the caller falls back to CFGStructure.
            return std::nullopt;
        }
        if (!covered_cnodes_.insert(*cnode_idx).second) {
            // Same CNode named by two plain leaves — Ghidra's tree
            // shouldn't do this for plain blocks.  Bail loudly.
            LOG(WARNING) << "Ghidra region tree references CNode "
                         << *cnode_idx << " twice in " << function_.name
                         << "; falling back to CFGStructure\n";
            return std::nullopt;
        }
        const auto &cnode = graph_.Node(*cnode_idx);
        std::vector< SNode * > stmts;
        stmts.reserve(cnode.stmts.size());
        for (clang::Stmt *stmt : cnode.stmts) {
            if (stmt != nullptr) {
                stmts.push_back(factory_.Make< SStmt >(stmt));
            }
        }

        // Wrap the block in SLabel(original_label, stmts) when it has
        // a label, mirroring what CFGStructure does for active node
        // representatives at the end of StructureAll.  Without this
        // wrap, SGoto references emitted elsewhere (ifgoto's taken
        // arm, the explicit `goto` kind handler) point at no in-tree
        // label and get dropped by the ClangEmitter dead-goto sweep.
        // RemoveDeadLabels strips any label not referenced by a goto,
        // so over-wrapping is safe.
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

        // child[0] = dispatcher block.  Expected to be a `plain` leaf
        // whose CNode carries the BRANCHIND terminal + populated
        // switch_cases + branch_cond (discriminant Expr).
        const auto &disp_struct = node.children[0];
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

        // Emit any pre-discriminant statements via the plain handler;
        // this also marks the dispatcher CNode covered.
        auto pre = TranslatePlain(disp_struct);
        if (!pre.has_value()) {
            return std::nullopt;
        }

        // Group dispatcher's switch_cases by successor CNode id.  Each
        // group becomes one arm body's set of case labels.
        struct ArmInfo {
            std::vector< std::int64_t > values;
            bool has_default = false;
        };
        std::unordered_map< size_t, ArmInfo > succ_to_arm;
        for (const auto &sc : disp.switch_cases) {
            if (sc.succ_index >= disp.succs.size()) {
                continue;
            }
            auto &ai = succ_to_arm[disp.succs[sc.succ_index]];
            if (sc.is_default) {
                ai.has_default = true;
            } else {
                ai.values.push_back(sc.value);
            }
        }

        // Discriminant type for IntegerLiteral case-value construction.
        clang::QualType case_type = disp.branch_cond->getType();
        if (case_type->isEnumeralType()) {
            case_type = case_type->castAs< clang::EnumType >()
                            ->getDecl()->getIntegerType();
        }
        unsigned case_width = ctx_.getIntWidth(case_type);

        auto *sw = factory_.Make< SSwitch >(disp.branch_cond);

        // Track which dispatcher successors are consumed by an arm
        // body so we can detect "orphan" cases — dispatch entries
        // whose target block sits OUTSIDE the switch in Ghidra's tree
        // (typically the post-switch merge block placed as a sibling).
        // Those still need explicit `case X: break;` labels so control
        // exits the switch on dispatch and falls into the merge block
        // emitted immediately after.
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

            // ClangEmitter places the default label after all case
            // labels in the output, so the "last" position (which
            // carries the body — earlier ones fall through to it)
            // is the default if present, else the last case value.
            const size_t body_pos = total_labels - 1;
            size_t pos = 0;
            for (std::int64_t v : arm.values) {
                auto *val = clang::IntegerLiteral::Create(
                    ctx_,
                    llvm::APInt(case_width, static_cast< uint64_t >(v), true),
                    case_type, VirtualLoc(ctx_));
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

        // Emit empty-body cases for orphan dispatch entries.  Sort by
        // case value for stable output (succ_to_arm iteration order
        // would otherwise depend on unordered_map's hash policy).
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
        for (const auto &kv : orphans) {
            const auto &arm = *kv.second;
            for (std::int64_t v : arm.values) {
                auto *val = clang::IntegerLiteral::Create(
                    ctx_,
                    llvm::APInt(case_width, static_cast< uint64_t >(v), true),
                    case_type, VirtualLoc(ctx_));
                sw->AddCase(val, std::vector< SNode * >{});
            }
            if (arm.has_default && sw->DefaultBodyList().empty()) {
                sw->SetDefaultBody(std::vector< SNode * >{});
            }
        }

        auto out = std::move(*pre);
        out.push_back(sw);
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateProperIf(const ghidra::RegionNode &node) {
        if (node.children.size() != 2) {
            return std::nullopt;
        }
        const auto &cond_struct = node.children[0];
        const auto &body_struct = node.children[1];

        auto head = TranslateCondHead(cond_struct, node);
        if (!head.has_value()) {
            return std::nullopt;
        }
        const auto &cond = graph_.Node(head->cond_idx);
        if (!cond.is_conditional || cond.succs.size() != 2
            || cond.branch_cond == nullptr)
        {
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

        // succs[0] = not-taken (cond false), succs[1] = taken (cond true).
        // The body covers ONE of the two arms; the other arm is the merge
        // (the region tree's next sibling, handled by the enclosing
        // list/graph).  Use branch_cond as-is when body is on succs[1]
        // and negate when body is on succs[0].
        clang::Expr *if_cond = cond.branch_cond;
        if (cond.succs[1] == *body_entry) {
            // Body is taken arm — use cond as-is.
        } else if (cond.succs[0] == *body_entry) {
            if_cond = NegateExpr(ctx_, cond.branch_cond);
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
        if (cond_struct.kind != "plain" || !cond_struct.block.has_value()) {
            return std::nullopt;
        }
        auto cond_idx = FindCNode(*cond_struct.block);
        if (!cond_idx.has_value()) {
            return std::nullopt;
        }
        const auto &cond = graph_.Node(*cond_idx);
        if (!cond.is_conditional || cond.succs.size() != 2
            || cond.branch_cond == nullptr)
        {
            return std::nullopt;
        }

        auto pre = TranslatePlain(cond_struct);
        if (!pre.has_value()) {
            return std::nullopt;
        }

        const auto &target = graph_.Node(cond.succs[1]);
        if (target.original_label.empty()) {
            LOG(WARNING) << "ifgoto taken target lacks label in "
                         << function_.name << "; falling back\n";
            return std::nullopt;
        }

        auto *goto_node = factory_.Make< SGoto >(
            factory_.Intern(target.original_label));
        auto *if_node = factory_.Make< SIfThenElse >(
            cond.branch_cond, goto_node, nullptr);
        auto out = std::move(*pre);
        out.push_back(if_node);
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateIfElse(const ghidra::RegionNode &node) {
        if (node.children.size() != 3) {
            return std::nullopt;
        }
        const auto &cond_struct = node.children[0];
        const auto &arm1_struct = node.children[1];
        const auto &arm2_struct = node.children[2];

        auto head = TranslateCondHead(cond_struct, node);
        if (!head.has_value()) {
            return std::nullopt;
        }
        const auto &cond = graph_.Node(head->cond_idx);
        if (!cond.is_conditional || cond.succs.size() != 2
            || cond.branch_cond == nullptr)
        {
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

        // succs[0] = not-taken (cond false), succs[1] = taken (cond true).
        // Assign the arm whose entry is succs[1] to the then-branch and
        // the arm whose entry is succs[0] to the else-branch.  Using the
        // raw branch_cond — no NegateExpr — keeps the condition shape
        // identical to what CGraphBuilder extracted from the CBRANCH.
        std::vector< SNode * > then_body;
        std::vector< SNode * > else_body;
        if (*arm1_entry == cond.succs[1] && *arm2_entry == cond.succs[0]) {
            then_body = std::move(*arm1_seq);
            else_body = std::move(*arm2_seq);
        } else if (*arm2_entry == cond.succs[1]
                   && *arm1_entry == cond.succs[0])
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
            cond.branch_cond, std::move(then_body), std::move(else_body));
        auto out = std::move(head->pre);
        out.push_back(if_node);
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateWhileDo(const ghidra::RegionNode &node) {
        if (node.children.size() != 2) {
            return std::nullopt;
        }
        const auto &cond_struct = node.children[0];
        const auto &body_struct = node.children[1];

        if (cond_struct.kind != "plain" || !cond_struct.block.has_value()) {
            return std::nullopt;
        }
        auto cond_idx = FindCNode(*cond_struct.block);
        if (!cond_idx.has_value()) {
            return std::nullopt;
        }
        const auto &cond = graph_.Node(*cond_idx);
        if (!cond.is_conditional || cond.succs.size() != 2
            || cond.branch_cond == nullptr)
        {
            return std::nullopt;
        }

        // Phase 5 simplification: the cond block must have no
        // non-terminal stmts.  Otherwise those stmts would have to
        // re-execute before each iteration's test — a transformation
        // we don't perform here.  Fall back for those.
        if (!cond.stmts.empty()) {
            LOG(WARNING) << "whiledo cond block " << *cond_struct.block
                         << " has " << cond.stmts.size()
                         << " pre-cond stmts in " << function_.name
                         << "; falling back\n";
            return std::nullopt;
        }

        // Mark cond CNode covered (we drained its stmts as "empty body";
        // TranslatePlain would do this but we want to skip its empty
        // SStmt accumulation).
        if (!covered_cnodes_.insert(*cond_idx).second) {
            LOG(WARNING) << "whiledo cond block " << *cond_struct.block
                         << " referenced twice in " << function_.name
                         << "; falling back\n";
            return std::nullopt;
        }

        auto body_seq = Translate(body_struct);
        if (!body_seq.has_value()) {
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

        // succs[0] = not-taken (cond false, exit), succs[1] = taken
        // (cond true, body).  Negate when body sits on succs[0].
        clang::Expr *while_cond = cond.branch_cond;
        if (cond.succs[1] == *body_entry) {
            // body on taken arm — branch_cond as-is.
        } else if (cond.succs[0] == *body_entry) {
            while_cond = NegateExpr(ctx_, cond.branch_cond);
        } else {
            LOG(WARNING) << "whiledo body entry " << *body_lbl
                         << " doesn't match cond succs in "
                         << function_.name << "; falling back\n";
            return std::nullopt;
        }

        auto *w = factory_.Make< SWhile >(while_cond, std::move(*body_seq));
        std::vector< SNode * > out;
        out.push_back(w);
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateDoWhile(const ghidra::RegionNode &node) {
        // Single-block self-loop: child[0] is the block carrying both
        // body stmts and the CBRANCH whose taken arm is the back-edge.
        if (node.children.size() != 1) {
            return std::nullopt;
        }
        const auto &block_struct = node.children[0];
        if (block_struct.kind != "plain" || !block_struct.block.has_value()) {
            return std::nullopt;
        }
        auto idx = FindCNode(*block_struct.block);
        if (!idx.has_value()) {
            return std::nullopt;
        }
        const auto &cnode = graph_.Node(*idx);
        if (!cnode.is_conditional || cnode.succs.size() != 2
            || cnode.branch_cond == nullptr)
        {
            return std::nullopt;
        }

        // Must be a self-loop on one arm.
        bool s0_self = cnode.succs[0] == *idx;
        bool s1_self = cnode.succs[1] == *idx;
        if (!s0_self && !s1_self) {
            LOG(WARNING) << "dowhile block " << *block_struct.block
                         << " is not a self-loop in " << function_.name
                         << "; falling back\n";
            return std::nullopt;
        }

        // TranslatePlain wraps stmts and registers the CNode as covered.
        auto body_seq = TranslatePlain(block_struct);
        if (!body_seq.has_value()) {
            return std::nullopt;
        }

        // Polarity: taken-arm-is-self → branch_cond keeps the loop
        // running (cond=true continues).  not-taken-self → negate.
        clang::Expr *do_cond = cnode.branch_cond;
        if (s1_self) {
            // already cond as-is
        } else {
            do_cond = NegateExpr(ctx_, cnode.branch_cond);
        }

        auto *dw = factory_.Make< SDoWhile >(std::move(*body_seq), do_cond);
        std::vector< SNode * > out;
        out.push_back(dw);
        return out;
    }

    BuildSNodeFromRegion::SNodeSeq
    BuildSNodeFromRegion::TranslateInfLoop(const ghidra::RegionNode &node) {
        if (node.children.size() != 1) {
            return std::nullopt;
        }
        const auto &body_struct = node.children[0];

        auto body_seq = Translate(body_struct);
        if (!body_seq.has_value()) {
            return std::nullopt;
        }

        // while (1) — the IntegerLiteral type matches `int`.
        auto *one = clang::IntegerLiteral::Create(
            ctx_, llvm::APInt(ctx_.getIntWidth(ctx_.IntTy), 1, true),
            ctx_.IntTy, VirtualLoc(ctx_));
        auto *w = factory_.Make< SWhile >(one, std::move(*body_seq));
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

        // Prefer the explicit goto target from Ghidra's BlockGoto
        // metadata (commit 03e679e) when present — that path supports
        // arbitrary child kinds because the destination is resolved
        // structurally by the Ghidra serializer (walks to the first
        // BlockCopy leaf), not inferred from the child's CFG successor.
        std::optional< std::string > target_label;
        if (node.goto_targets.size() == 1
            && !node.goto_targets[0].empty())
        {
            target_label = node.goto_targets[0];
        } else {
            // Fallback for JSON predating 03e679e: only plain children
            // are translatable, and the target comes from the child
            // CNode's sole successor.
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
            target_label = target.original_label;
        }

        // Validate the target label resolves to a CNode — otherwise the
        // SGoto would dangle and the dead-goto sweep would silently drop it.
        if (!FindCNode(*target_label).has_value()) {
            LOG(WARNING) << "goto target " << *target_label
                         << " is not a known CNode in " << function_.name
                         << "; falling back\n";
            return std::nullopt;
        }

        // Translate the child subtree as any kind — recursive Translate
        // handles plain/list/graph/properif/whiledo/etc.  The structured
        // result executes, then the appended SGoto transfers control to
        // the wrapped block's escape destination.
        auto body = Translate(src_struct);
        if (!body.has_value()) {
            return std::nullopt;
        }

        auto *goto_node = factory_.Make< SGoto >(
            factory_.Intern(*target_label));
        auto out = std::move(*body);
        out.push_back(goto_node);
        return out;
    }

    std::optional< BuildSNodeFromRegion::CondHead >
    BuildSNodeFromRegion::TranslateCondHead(
        const ghidra::RegionNode &cond_struct,
        const ghidra::RegionNode &enclosing)
    {
        // Fast path: a `plain` cond is a single CBRANCH block.  Delegate
        // to TranslatePlain (which registers coverage and wraps in SLabel
        // when the block has an original_label).
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
            return CondHead{std::move(*plain_seq), *idx};
        }
        // `list` cond: one or more pre-cond blocks compute intermediate
        // values, the final child holds the CBRANCH.  Unifying these
        // makes properif/ifelse handlers cover Ghidra's common multi-
        // block-condition output (146 of 176 fallbacks on bloodview).
        if (cond_struct.kind != "list") {
            return std::nullopt;
        }
        if (cond_struct.children.size() < 2) {
            return std::nullopt;
        }
        const auto &cond_child = cond_struct.children.back();
        if (cond_child.kind != "plain" || !cond_child.block.has_value()) {
            return std::nullopt;
        }
        auto cond_idx = FindCNode(*cond_child.block);
        if (!cond_idx.has_value()) {
            return std::nullopt;
        }

        // Option (c): TranslatePlain wraps each pre-cond block in an
        // SLabel when it has a non-empty original_label.  Inside the
        // if/loop construct, that SLabel is at risk of duplication by
        // DuplicateStackGuardReturnTargets.  If a goto from outside the
        // enclosing construct targets one of these labels, duplication
        // leaves the outside reference dangling and crashes the pass.
        // Bail in that case.
        if (PreCondLabelsReferencedFromOutside(cond_struct, enclosing)) {
            LOG(WARNING) << "list cond in " << function_.name
                         << " has pre-cond labels referenced from outside"
                            " the enclosing construct; falling back\n";
            return std::nullopt;
        }

        std::vector< SNode * > pre;
        for (size_t i = 0; i + 1 < cond_struct.children.size(); ++i) {
            auto seq = Translate(cond_struct.children[i]);
            if (!seq.has_value()) {
                return std::nullopt;
            }
            for (SNode *s : *seq) {
                pre.push_back(s);
            }
        }
        auto cond_seq = TranslatePlain(cond_child);
        if (!cond_seq.has_value()) {
            return std::nullopt;
        }
        for (SNode *s : *cond_seq) {
            pre.push_back(s);
        }
        return CondHead{std::move(pre), *cond_idx};
    }

    bool BuildSNodeFromRegion::IsAddressInSubtree(
        const ghidra::RegionNode *needle,
        const ghidra::RegionNode &subtree) const
    {
        if (needle == &subtree) {
            return true;
        }
        for (const auto &c : subtree.children) {
            if (IsAddressInSubtree(needle, c)) {
                return true;
            }
        }
        return false;
    }

    namespace {
        // Walk a clang::Stmt subtree and collect GotoStmt target names.
        void CollectGotoStmtTargets(
            const clang::Stmt *s,
            std::vector< std::string > &out)
        {
            if (s == nullptr) {
                return;
            }
            if (const auto *gs = llvm::dyn_cast< clang::GotoStmt >(s)) {
                if (const auto *ld = gs->getLabel()) {
                    out.push_back(ld->getName().str());
                }
                return;
            }
            for (const clang::Stmt *child : s->children()) {
                CollectGotoStmtTargets(child, out);
            }
        }
    } // namespace

    bool BuildSNodeFromRegion::PreCondLabelsReferencedFromOutside(
        const ghidra::RegionNode &cond_struct,
        const ghidra::RegionNode &enclosing) const
    {
        if (cond_struct.kind != "list" || cond_struct.children.size() < 2) {
            return false;
        }

        // Step 1: collect labels of pre-cond blocks that would emit an
        // SLabel wrap.  Recurse into any non-plain pre-cond children
        // (e.g., nested list) so we don't miss labels inside them.
        std::unordered_set< std::string > pre_labels;
        std::function< void(const ghidra::RegionNode &) > collect_labels =
            [&](const ghidra::RegionNode &n) {
                if (n.kind == "plain" && n.block.has_value()) {
                    if (auto idx = FindCNode(*n.block)) {
                        const auto &cn = graph_.Node(*idx);
                        if (!cn.original_label.empty()) {
                            pre_labels.insert(cn.original_label);
                        }
                    }
                }
                for (const auto &c : n.children) {
                    collect_labels(c);
                }
            };
        for (size_t i = 0; i + 1 < cond_struct.children.size(); ++i) {
            collect_labels(cond_struct.children[i]);
        }
        if (pre_labels.empty()) {
            // No SLabel wrap → no duplication risk.
            return false;
        }

        // Step 2: walk the whole function region; for every goto
        // reference, if its target hits pre_labels AND the source region
        // sits OUTSIDE the enclosing subtree, the reference will dangle.
        if (!function_.region.has_value()) {
            return false;
        }

        bool unsafe = false;
        std::function< void(const ghidra::RegionNode &) > walk =
            [&](const ghidra::RegionNode &root) {
                if (unsafe) {
                    return;
                }
                auto check_target = [&](const std::string &target) {
                    if (target.empty() || !pre_labels.count(target)) {
                        return;
                    }
                    if (!IsAddressInSubtree(&root, enclosing)) {
                        unsafe = true;
                    }
                };

                // Raw clang::GotoStmts embedded in plain-leaf CNode stmts.
                if (root.kind == "plain" && root.block.has_value()) {
                    if (auto idx = FindCNode(*root.block)) {
                        const auto &cn = graph_.Node(*idx);
                        std::vector< std::string > targets;
                        for (const clang::Stmt *s : cn.stmts) {
                            CollectGotoStmtTargets(s, targets);
                        }
                        CollectGotoStmtTargets(cn.terminal, targets);
                        for (const auto &t : targets) {
                            check_target(t);
                            if (unsafe) return;
                        }
                    }
                }
                // Region-level goto/multigoto wrappers.  Goto-target
                // strings come from Ghidra in "address:idx:type" (colon)
                // form, but SLabel names use the "_"-normalized form.
                // Check both since either could match a pre-cond label.
                if (root.kind == "goto" || root.kind == "multigoto") {
                    for (const auto &t : root.goto_targets) {
                        check_target(t);
                        check_target(LabelNameFromKey(t));
                        if (unsafe) return;
                    }
                }
                // ifgoto: synthetic SGoto target is cond.succs[1].original_label.
                if (root.kind == "ifgoto" && !root.children.empty()) {
                    const auto &cond_child = root.children[0];
                    if (cond_child.kind == "plain"
                        && cond_child.block.has_value())
                    {
                        if (auto idx = FindCNode(*cond_child.block)) {
                            const auto &cn = graph_.Node(*idx);
                            if (cn.succs.size() == 2) {
                                const auto &target = graph_.Node(cn.succs[1]);
                                check_target(target.original_label);
                                if (unsafe) return;
                            }
                        }
                    }
                }
                for (const auto &c : root.children) {
                    walk(c);
                    if (unsafe) return;
                }
            };
        walk(*function_.region);
        return unsafe;
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

} // namespace patchestry::ast
