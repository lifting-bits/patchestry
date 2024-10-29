/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/Support/JSON.h"

namespace patchestry::ghidra {

    using json_arr = llvm::json::Array;
    using json_obj = llvm::json::Object;
    using json_val = llvm::json::Value;

    struct varnode_type
    {
        enum kind {
            vt_invalid = 0,
            vt_integer,
            vt_float,
            vt_pointer,
            vt_array,
            vt_struct,
            vt_union,
            vt_enum,
            vt_typedef
        };

        static varnode_type::kind convert_to_kind(const std::string &kind) {
            static const std::unordered_map< std::string, varnode_type::kind > kind_map = {
                {"integer", vt_integer},
                {  "float",   vt_float},
                {"pointer", vt_pointer},
                {  "array",   vt_array},
                { "struct",  vt_struct},
                {  "union",   vt_union},
                {   "enum",    vt_enum},
                {"typedef", vt_typedef}
            };

            // if kind is not present in the map, return vt_invalid
            auto it = kind_map.find(kind);
            return it != kind_map.end() ? it->second : vt_invalid;
        }

        kind get_kind() const { return type_kind; }

        uint32_t get_bit_width() const { return size; }

        const varnode_type &get_pointee_type() const { return *pointee_type; }

        std::string name;
        kind type_kind;
        uint32_t size;
        bool is_signed;

        std::shared_ptr< varnode_type > pointee_type;
        std::shared_ptr< varnode_type > base_type;
    };

    struct varnode
    {
        enum kind {
            unknown = 0,
            local,
            global,
        };

        static varnode::kind convert_to_kind(std::string kd) {
            if (kd == "local") {
                return local;
            } else if (kd == "global") {
                return global;
            }

            return unknown;
        }

        kind var_kind;
        varnode_type type;
        uint32_t size;
    };

    struct pcode
    {
        std::string mnemonic;
        std::string name;
        std::vector< varnode > output{};
        std::vector< varnode > inputs{};
    };

    struct basic_block
    {
        std::shared_ptr< basic_block > parent;
        std::string label;
        std::vector< pcode > ops;
    };

    struct function_prototype
    {
        std::vector< varnode > parameters;
        varnode_type rttype;
    };

    struct function
    {
        std::string name;
        function_prototype prototype;
        std::vector< basic_block > basic_blocks;
    };

    struct program
    {
        std::string arch;
        std::string format;
        std::vector< function > functions;
        std::unordered_map< std::string, varnode_type > types;
    };

    class json_parser
    {
      public:
        std::optional< program > parse_program(const json_obj &root);

      private:
        std::optional< varnode_type > parse_type(const json_obj &type);

        void process_serialized_types(std::unordered_map< std::string, varnode_type > &types);

        std::optional< varnode > parse_varnode(const json_obj &var_obj);

        // Function to parse Pcode
        std::optional< pcode > parse_pcode(const json_obj &pcode_obj);

        // Function to parse Basic Blocks
        std::optional< basic_block > parse_basic_block(const json_obj &block_obj);

        std::optional< function_prototype >
        parse_function_prototype(const json_obj &prototype_obj);

        // Function to parse Functions
        std::optional< function > parse_function(const json_obj &func_obj);
    };

} // namespace patchestry::ghidra
