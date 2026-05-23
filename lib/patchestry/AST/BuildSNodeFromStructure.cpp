/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/BuildSNodeFromStructure.hpp>

#include <patchestry/AST/CGraph.hpp>
#include <patchestry/AST/SNode.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Type.h>
#include <llvm/ADT/APInt.h>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace patchestry::ast {

    BuildSNodeFromStructure::BuildSNodeFromStructure(
        const ghidra::Function &function, CGraph &graph,
        SNodeFactory &factory, clang::ASTContext &ctx
    )
        : function_(function), graph_(graph), factory_(factory), ctx_(ctx) {}

    bool BuildSNodeFromStructure::TryBuild() {
        if (!function_.structure.has_value() || graph_.nodes.empty()) {
            return false;
        }

        covered_cnodes_.clear();
        auto body = Translate(*function_.structure);
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
        // structure tree (after synthetic-entry handling above).
        // CGraphBuilder and Ghidra may disagree on block boundaries
        // for other reasons too (e.g., unreachable blocks).  Silently
        // dropping uncovered CNodes would lose statements; per loud-
        // failure, we fall back instead.
        if (covered_cnodes_.size() != graph_.nodes.size()) {
            LOG(WARNING) << "Ghidra structure tree covers "
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

    BuildSNodeFromStructure::SNodeSeq
    BuildSNodeFromStructure::Translate(const ghidra::StructureNode &node) {
        // Kind vocabulary (PcodeBlock::typeToName) lives on StructureNode
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
        return std::nullopt;
    }

    BuildSNodeFromStructure::SNodeSeq
    BuildSNodeFromStructure::TranslatePlain(const ghidra::StructureNode &node) {
        if (!node.block.has_value()) {
            LOG(WARNING) << "plain structure node missing block label in "
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
            LOG(WARNING) << "Ghidra structure tree references CNode "
                         << *cnode_idx << " twice in " << function_.name
                         << "; falling back to CFGStructure\n";
            return std::nullopt;
        }
        const auto &cnode = graph_.Node(*cnode_idx);
        std::vector< SNode * > out;
        out.reserve(cnode.stmts.size());
        for (clang::Stmt *stmt : cnode.stmts) {
            if (stmt != nullptr) {
                out.push_back(factory_.Make< SStmt >(stmt));
            }
        }
        return out;
    }

    BuildSNodeFromStructure::SNodeSeq
    BuildSNodeFromStructure::TranslateChildSeq(const ghidra::StructureNode &node) {
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

    BuildSNodeFromStructure::SNodeSeq
    BuildSNodeFromStructure::TranslateSwitch(const ghidra::StructureNode &node) {
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

    std::optional< std::string >
    BuildSNodeFromStructure::FirstBlockLabel(const ghidra::StructureNode &node) const {
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
    BuildSNodeFromStructure::FindCNode(const std::string &block_label) const {
        for (const auto &node : graph_.nodes) {
            if (node.source_key == block_label) {
                return node.id;
            }
        }
        return std::nullopt;
    }

} // namespace patchestry::ast
