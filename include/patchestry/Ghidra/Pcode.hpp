/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <string_view>

#include "Pcode.def"

namespace patchestry::ghidra {

    enum class Mnemonic : int {
#define X(name) OP_##name, // NOLINT(cppcoreguidelines-macro-usage)
        PCODE_MNEMONICS
#undef X
            OP_UNKNOWN
    };

    template< typename EnumType, size_t N >
    struct PCodeStringMapper
    {
        std::array< std::pair< EnumType, std::string_view >, N > mappings;

        // Convert enum to string
        constexpr std::string_view to_string(EnumType val) const {
            for (const auto &[pcode, str] : mappings) {
                if (pcode == val) {
                    return str;
                }
            }
            return "UNKNOWN";
        }

        constexpr EnumType from_string(std::string_view s) const {
            for (const auto &[val, str] : mappings) {
                if (str == s) {
                    return val;
                }
            }
            return EnumType::OP_UNKNOWN;
        }
    };

    // Calculate the number of mnemonics
    constexpr size_t num_mnemonics = []() constexpr {
        size_t count = 0;
#define X(name) ++count; // NOLINT(cppcoreguidelines-macro-usage)
        PCODE_MNEMONICS
#undef X
        return count;
    }();

    // Instantiate the EnumStringMapper for PCodeMnemonic
    constexpr PCodeStringMapper< Mnemonic, num_mnemonics > mnemonic_mapper{ {
#define X(name) std::pair{ Mnemonic::OP_##name, #name },
        PCODE_MNEMONICS
#undef X
    } };

    constexpr std::string_view to_string(Mnemonic mnemonic) {
        return mnemonic_mapper.to_string(mnemonic);
    }

    constexpr Mnemonic from_string(const std::string_view &mnemonic_str) {
        return mnemonic_mapper.from_string(mnemonic_str);
    }

} // namespace patchestry::ghidra
