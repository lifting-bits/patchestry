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
        std::vector< StmtOrigin > direct_origins;
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

    struct SRegionLegalityReport
    {
        size_t owned_ops               = 0;
        size_t cloned_owned_ops        = 0;
        size_t cross_region_cloned_ops = 0;
        size_t illegal_cloned_ops      = 0;
        std::vector< std::string > diagnostics;

        bool ok() const { return diagnostics.empty(); }
    };

    enum class SRewriteAction {
        Move,
        Clone,
        Hoist,
        Sink,
    };

    enum class SRewriteDecision {
        LeaveGoto,
        Move,
        Clone,
        Hoist,
        Sink,
    };

    struct SRewriteLegalityOptions
    {
        bool allow_calls            = false;
        bool allow_stores           = false;
        bool allow_volatile         = false;
        bool allow_internal_control = false;
    };

    struct SRewriteLegalityReport
    {
        SRewriteAction action           = SRewriteAction::Move;
        size_t payload_origins          = 0;
        size_t rejected_payload_origins = 0;
        std::vector< std::string > diagnostics;

        bool ok() const { return diagnostics.empty(); }
    };

    struct SRewriteProfitabilityOptions
    {
        SRewriteLegalityOptions legality_options;
        size_t max_clone_payload_origins = 64;
        bool single_reference            = false;
        bool source_has_fallthrough      = true;
        bool preserves_region_ownership  = false;
    };

    struct SRewriteProfitabilityReport
    {
        SRewriteAction requested_action = SRewriteAction::Move;
        SRewriteDecision decision       = SRewriteDecision::LeaveGoto;
        size_t payload_origins          = 0;
        std::vector< std::string > diagnostics;

        bool profitable() const { return decision != SRewriteDecision::LeaveGoto; }
    };

    /// Build a region graph from SNode body-list ownership.  Raw Clang
    /// CompoundStmt payloads inside SStmt remain opaque payload atoms; they are
    /// counted but not converted into structured regions.
    SRegionGraph BuildSNodeRegionGraph(const std::vector< SNode * > &root);

    /// Validate parent/child consistency and ownership aggregation for the
    /// SNode region graph.
    SRegionValidationReport ValidateSNodeRegionGraph(const SRegionGraph &graph);

    /// Validate movement legality for payloads placed by structuring cleanup.
    /// A payload operation may appear in multiple regions only when every
    /// placement is explicitly marked cloneable by source-origin metadata.
    SRegionLegalityReport ValidateSNodeRegionLegality(const SRegionGraph &graph);

    /// Validate whether a cleanup pass may move or clone a payload-carrying
    /// SNode subtree.  This is a local preflight check used before mutating
    /// structured regions; the post-pass region verifier remains authoritative.
    SRewriteLegalityReport ValidateSNodeRewriteLegality(
        const SNode &node, SRewriteAction action, const SRewriteLegalityOptions &options = {}
    );

    SRewriteLegalityReport ValidateSNodeRewriteLegality(
        const std::vector< SNode * > &seq, SRewriteAction action,
        const SRewriteLegalityOptions &options = {}
    );

    bool CanCloneSNodePayloads(const SNode &node, const SRewriteLegalityOptions &options = {});

    bool CanCloneSNodePayloads(
        const std::vector< SNode * > &seq, const SRewriteLegalityOptions &options = {}
    );

    bool CanMoveSNodePayloads(const SNode &node, const SRewriteLegalityOptions &options = {});

    bool CanMoveSNodePayloads(
        const std::vector< SNode * > &seq, const SRewriteLegalityOptions &options = {}
    );

    SRewriteProfitabilityReport EvaluateSNodeRewriteProfitability(
        const SNode &node, SRewriteAction action,
        const SRewriteProfitabilityOptions &options = {}
    );

    SRewriteProfitabilityReport EvaluateSNodeRewriteProfitability(
        const std::vector< SNode * > &seq, SRewriteAction action,
        const SRewriteProfitabilityOptions &options = {}
    );

    bool ShouldApplySNodeRewrite(
        const SNode &node, SRewriteAction action,
        const SRewriteProfitabilityOptions &options = {}
    );

    bool ShouldApplySNodeRewrite(
        const std::vector< SNode * > &seq, SRewriteAction action,
        const SRewriteProfitabilityOptions &options = {}
    );

} // namespace patchestry::ast
