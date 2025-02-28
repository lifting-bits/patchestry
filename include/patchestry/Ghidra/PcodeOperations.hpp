/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeTypes.hpp>

namespace patchestry::ghidra {
    struct Varnode;
    struct Variable;
    struct Operation;
    struct BasicBlock;
    struct FunctionPrototype;
    struct Function;
    struct Program;

} // namespace patchestry::ghidra

namespace patchestry::ghidra {
    using TypeMap = std::unordered_map< std::string, std::shared_ptr< VarnodeType > >;

    using FunctionMap = std::unordered_map< std::string, Function >;

    using BasicBlockMap = std::unordered_map< std::string, BasicBlock >;

    using VariableMap = std::unordered_map< std::string, Variable >;

    struct Varnode
    {
        enum Kind {
            VARNODE_UNKNOWN = 0,
            VARNODE_GLOBAL,
            VARNODE_LOCAL,
            VARNODE_PARAM,
            VARNODE_FUNCTION,
            VARNODE_TEMPORARY,
            VARNODE_CONSTANT,
            VARNODE_STRING
        };

        static Varnode::Kind convertToKind(const std::string &kdd) {
            static const std::unordered_map< std::string, Varnode::Kind > kind_map = {
                {  "unknown",   VARNODE_UNKNOWN},
                {   "global",    VARNODE_GLOBAL},
                {    "local",     VARNODE_LOCAL},
                {"parameter",     VARNODE_PARAM},
                { "function",  VARNODE_FUNCTION},
                {"temporary", VARNODE_TEMPORARY},
                { "constant",  VARNODE_CONSTANT},
                {   "string",    VARNODE_STRING}
            };

            // if kind is not present in the map, return varnode_unknown
            auto iter = kind_map.find(kdd);
            return iter != kind_map.end() ? iter->second : VARNODE_UNKNOWN;
        }

        Kind kind;
        uint32_t size;
        std::string type_key;

        std::optional< std::string > operation;
        std::optional< std::string > function;
        std::optional< uint32_t > value;
        std::optional< std::string > string_value;
        std::optional< std::string > global;
    };

    struct Variable
    {
        std::string name;
        std::string type;
        uint32_t size;
        std::string key;
    };

    struct OperationTarget

    {
        Varnode::Kind kind;
        std::optional< std::string > function;
        std::optional< std::string > operation;
        bool is_noreturn;
    };

    struct Operation
    {
        Mnemonic mnemonic;
        std::optional< Varnode > output;
        std::vector< Varnode > inputs;

        std::string key;
        std::string parent_block_key;

        // Parameter/variable declaration
        std::optional< std::string > name;
        std::optional< std::string > type;
        std::optional< uint32_t > index;

        // Call Operation
        std::optional< OperationTarget > target;

        // Branch Operation
        std::optional< std::string > target_block;

        // Cond Branch
        std::optional< std::string > taken_block;
        std::optional< std::string > not_taken_block;
        std::optional< Varnode > condition;
        std::optional< std::string > address;
    };

    struct BasicBlock
    {
        std::shared_ptr< BasicBlock > parent;
        std::string key;
        std::unordered_map< std::string, Operation > operations;
        std::vector< std::string > ordered_operations;
        bool is_entry_block;
    };

    struct FunctionPrototype
    {
        std::vector< std::string > parameters;
        std::string rttype_key;
        bool is_variadic;
        bool is_noreturn;
    };

    struct Function
    {
        std::string name;
        FunctionPrototype prototype;
        std::string key;
        std::string entry_block;
        std::unordered_map< std::string, BasicBlock > basic_blocks;
    };

    struct Program
    {
        std::optional< std::string > arch;
        std::optional< std::string > lang;
        std::optional< std::string > format;
        std::unordered_map< std::string, Function > serialized_functions;
        std::unordered_map< std::string, std::shared_ptr< VarnodeType > > serialized_types;
        std::unordered_map< std::string, Variable > serialized_globals;
    };
} // namespace patchestry::ghidra
