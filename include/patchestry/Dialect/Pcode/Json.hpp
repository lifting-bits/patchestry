/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include "llvm/Support/JSON.h"
#include <optional>
#include <string>
#include <vector>

namespace patchestry::pc {

    struct pcode
    {
        std::string mnemonic;

        struct
        {
            std::string type;
            std::optional< int64_t > offset;
            std::optional< int64_t > size;
        } output;

        struct input
        {
            std::string type;
            std::optional< int64_t > offset;
            std::optional< int64_t > size;
        };

        std::vector< input > inputs;
    };

    struct instruction
    {
        std::string mnemonic;
        std::string address;
        std::vector< pcode > pcodes;
    };

    struct basic_block
    {
        std::string label;
        std::vector< instruction > instructions;
    };

    struct function
    {
        std::string name;
        std::vector< basic_block > basic_blocks;
    };

    struct program
    {
        std::string arch;
        std::string os;
        std::vector< function > functions;
    };

    class json_parser
    {
      public:
        std::optional< program > parse_program(const llvm::json::Object &root);

      private:
        // Function to parse Pcode
        std::optional< pcode > parse_pcode(const llvm::json::Object &pcode_obj);

        // Function to parse Instructions
        std::optional< instruction > parse_instruction(const llvm::json::Object &inst_obj);

        // Function to parse Basic Blocks
        std::optional< basic_block > parse_basic_block(const llvm::json::Object &block_obj);

        // Function to parse Functions
        std::optional< function > parse_function(const llvm::json::Object &func_obj);
    };

} // namespace patchestry::pc
