/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/ADT/StringRef.h>
#include <algorithm>
#include <array>
#include <string_view>
#include <optional>

#include "PcodeDef.h"

namespace patchestry::pc {

enum class PCodeMnemonic {
#define X(name) name,
    PCODE_MNEMONIC_LIST
#undef X
    UNKNOWN
};

enum class PCodeVarnodeType {
#define X(name) name##_,
    PCODE_VARNODE_TYPE
#undef X
    UNKNOWN
};

template <typename EnumType, size_t N>
struct PCodeStringMapper {
    std::array<std::pair<EnumType, std::string_view>, N> mappings;

    //Convert enum to string
    constexpr std::string_view to_string(EnumType val) const {
        for (const auto& [pcode, str] : mappings) {
            if (pcode == val) {
                return str;
            }
        }
        return "UNKNOWN";
    }

    constexpr EnumType from_string(std::string_view s) const {
        for (const auto& [val, str] : mappings) {
            if (str == s) {
                return val;
            }
        }
        return EnumType::UNKNOWN;
    }
};

// Calculate the number of mnemonics
constexpr size_t NumPCodeMnemonics = []() constexpr {
    size_t count = 0;
#define X(name) ++count;
    PCODE_MNEMONIC_LIST
#undef X
    return count;
}();

constexpr size_t NumVarNodeType = []() constexpr {
    size_t count = 0;
#define X(name) ++count;
    PCODE_VARNODE_TYPE
#undef X
    return count;
}();

// Instantiate the EnumStringMapper for PCodeMnemonic
constexpr PCodeStringMapper<PCodeMnemonic, NumPCodeMnemonics> PCodeMnemonicMapper{{
#define X(name) std::pair{PCodeMnemonic::name, #name},
    PCODE_MNEMONIC_LIST
#undef X
}};

constexpr PCodeStringMapper<PCodeVarnodeType, NumVarNodeType> PCodeVarNodeMapper{{
#define X(name) std::pair{PCodeVarnodeType::name##_, #name},
    PCODE_VARNODE_TYPE
#undef X
}};

constexpr std::string_view to_string(PCodeMnemonic mnemonic) {
    return PCodeMnemonicMapper.to_string(mnemonic);
}

constexpr PCodeMnemonic from_string(llvm::StringRef mnemonic_str) {
    return PCodeMnemonicMapper.from_string(mnemonic_str);
}

constexpr std::string_view varnode_to_string(PCodeVarnodeType ty) {
    return PCodeVarNodeMapper.to_string(ty);
}

constexpr PCodeVarnodeType varnode_from_string(llvm::StringRef ty_str) {
    return PCodeVarNodeMapper.from_string(ty_str);
}

}