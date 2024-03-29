/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

/**
 * @brief Ghidra Representations
 */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <llvm/Support/JSON.h>
#pragma clang diagnostic pop

namespace patchestry::ghidra {
    using json_arr = llvm::json::Array;
    using json_obj = llvm::json::Object;
    using json_val = llvm::json::Value;

    template< typename T >
    using expected = llvm::Expected< T >;

    struct varnode_t
    {
        /**
         * @brief Varnode
         */
        enum class kind_t { constant, unique, reg, ram } kind;
        int64_t address;
        std::size_t size;

        static auto get_kind(std::string_view str) -> kind_t;
        static auto from_json(const json_arr &varnode_arr) -> expected< varnode_t >;
    };

    struct pcode_t
    {
        /**
         * @brief PCodeOP
         */
        std::string mnemonic;
        std::optional< varnode_t > output;
        std::vector< varnode_t > inputs;

        static auto from_json(const json_obj &pcode_obj) -> expected< pcode_t >;
    };

    struct instruction_t
    {
        /**
         * @brief Instruction
         */
        std::string mnemonic;
        std::vector< pcode_t > semantics;

        static auto from_json(const json_obj &inst_obj) -> expected< instruction_t >;
    };

    struct code_block_t
    {
        /**
         * @brief CodeBlock
         */
        std::string label;
        std::vector< instruction_t > instructions;

        static auto from_json(const json_obj &block_obj) -> expected< code_block_t >;
    };

    struct function_t
    {
        /**
         * @brief Function
         */
        std::string name;
        std::vector< code_block_t > basic_blocks;

        static auto from_json(const json_obj &func_obj) -> expected< function_t >;
    };
} // namespace patchestry::ghidra
