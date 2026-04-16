/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace patchestry::klee_verifier {

    enum PredicateKind {
        PK_Unknown,
        PK_Nonnull,
        PK_RelNeqArgConst,
        PK_RelEqArgConst,
        PK_RelLtArgConst,
        PK_RelLeArgConst,
        PK_RelGtArgConst,
        PK_RelGeArgConst,
        PK_RangeRet,
        PK_RangeArg,
        PK_Alignment
    };

    struct ParsedPredicate {
        PredicateKind kind = PK_Unknown;
        std::string target;
        unsigned arg_index = 0;
        int64_t constant   = 0;
        int64_t min_val    = 0;
        int64_t max_val    = 0;
        uint64_t alignment = 0;
        bool is_precondition = true;
    };

    // Parse a serialized static_contract string into the set of predicates
    // it declares. `dropped` is incremented for every `{...}` block whose
    // contents fail to produce a valid predicate so callers can surface
    // silent drops.
    std::vector< ParsedPredicate >
    parseStaticContractText(const std::string &contract_str, unsigned &dropped);

} // namespace patchestry::klee_verifier
