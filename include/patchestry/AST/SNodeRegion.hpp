/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/SNode.hpp>

#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace patchestry::ast {

    enum class SRegionKind {
        Root,
        LabelBody,
        IfThen,
        IfElse,
        WhileBody,
        DoWhileBody,
        ForBody,
        SwitchCase,
        SwitchDefault,
    };

    const char *SRegionKindName(SRegionKind kind);

    struct SRegionNode
    {
        static constexpr size_t kNone = std::numeric_limits< size_t >::max();

        size_t id          = kNone;
        size_t parent      = kNone;
        SRegionKind kind   = SRegionKind::Root;
        const SNode *owner = nullptr;
        std::string name;
        std::vector< size_t > children;

        size_t direct_snodes            = 0;
        size_t subtree_snodes           = 0;
        size_t opaque_compound_payloads = 0;
        std::unordered_map< std::string, unsigned > direct_owned_ops;
        std::unordered_map< std::string, unsigned > subtree_owned_ops;
    };

    struct SRegionGraph
    {
        std::vector< SRegionNode > regions;
        size_t root = SRegionNode::kNone;

        bool empty() const { return regions.empty(); }
    };

    struct SRegionValidationReport
    {
        size_t regions                  = 0;
        size_t owned_ops                = 0;
        size_t duplicated_owned_ops     = 0;
        size_t opaque_compound_payloads = 0;
        std::vector< std::string > diagnostics;

        bool ok() const { return diagnostics.empty(); }
    };

    /// Build a region graph from SNode body-list ownership.  Raw Clang
    /// CompoundStmt payloads inside SStmt remain opaque payload atoms; they are
    /// counted but not converted into structured regions.
    SRegionGraph BuildSNodeRegionGraph(const std::vector< SNode * > &root);

    /// Validate parent/child consistency and ownership aggregation for the
    /// SNode region graph.
    SRegionValidationReport ValidateSNodeRegionGraph(const SRegionGraph &graph);

} // namespace patchestry::ast
