/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/BuildSNodeFromStructure.hpp>

#include <patchestry/Ghidra/PcodeOperations.hpp>

#include <clang/AST/ASTContext.h>

namespace patchestry::ast {

    BuildSNodeFromStructure::BuildSNodeFromStructure(
        const ghidra::Function &function, CGraph &graph,
        SNodeFactory &factory, clang::ASTContext &ctx
    )
        : function_(function), graph_(graph), factory_(factory), ctx_(ctx) {}

    bool BuildSNodeFromStructure::TryBuild() {
        // Phase 0 scaffold: no kind handlers wired yet.  Translate
        // returns nullptr unconditionally, so we always signal failure
        // and the caller falls back to CFGStructure.
        if (!function_.structure.has_value()) {
            return false;
        }
        return Translate(*function_.structure) != nullptr;
    }

    SNode *BuildSNodeFromStructure::Translate(const ghidra::StructureNode & /*node*/) {
        // Phase 0: no kinds implemented.  Phases 1–6 fill this dispatcher
        // in kind by kind; until then every call falls back per-function.
        return nullptr;
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
