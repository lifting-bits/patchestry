/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace patchestry::ast {

    enum class PayloadKind {
        AtomicPayload,
        PayloadExpression,
        PayloadSyntheticDecl,
        TerminalSynthetic,
        RegionSynthetic,
    };

    struct StmtOrigin
    {
        std::string block_key;
        std::string operation_key;
        PayloadKind kind               = PayloadKind::PayloadExpression;
        bool primary                   = false;
        bool movable                   = false;
        bool cloneable                 = false;
        bool may_call                  = false;
        bool may_store                 = false;
        bool may_volatile              = false;
        bool contains_internal_control = false;
    };

    inline const char *PayloadKindName(PayloadKind kind) {
        switch (kind) {
            case PayloadKind::AtomicPayload:
                return "atomic_payload";
            case PayloadKind::PayloadExpression:
                return "payload_expression";
            case PayloadKind::PayloadSyntheticDecl:
                return "payload_synthetic_decl";
            case PayloadKind::TerminalSynthetic:
                return "terminal_synthetic";
            case PayloadKind::RegionSynthetic:
                return "region_synthetic";
        }
        return "unknown";
    }

    inline bool IsPayloadCarrierKind(PayloadKind kind) {
        return kind == PayloadKind::AtomicPayload || kind == PayloadKind::PayloadExpression
            || kind == PayloadKind::PayloadSyntheticDecl;
    }

    inline std::string StmtOriginKey(const StmtOrigin &origin) {
        return origin.block_key + "::" + origin.operation_key;
    }

} // namespace patchestry::ast
