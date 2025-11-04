/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <string>

#include <llvm/Support/YAMLTraits.h>

/* This file contains domain objects that are common to both patches and contracts.
 * These objects express the same kind of YAML input, parsed from different locations
 * in the configuration file. The objects here are in use within the OperationMatcher,
 * so it's convenient for them to be the same for both patches and contracts.
 */
namespace patchestry::passes {

    enum class PatchInfoMode : uint8_t {
        NONE = 0, // No patch
        APPLY_BEFORE,
        APPLY_AFTER,
        REPLACE
    };

    enum class ArgumentSourceType : uint8_t {
        OPERAND = 0, // Reference to operation operand by index
        VARIABLE,    // Reference to variable by name
        SYMBOL,      // Reference to symbol by name
        CONSTANT,    // Literal constant value
        RETURN_VALUE // Return value of function or operation
    };

    struct Metadata
    {
        std::string name;
        std::string description;
        std::string version;
        std::string author;
        std::string created;
        std::string organization;
    };

    enum class MatchKind : uint8_t { NONE = 0, OPERATION, FUNCTION };

    struct ArgumentMatch
    {
        unsigned index;
        std::string name;
        std::string type;
    };

    using OperandMatch = ArgumentMatch;

    struct VariableMatch
    {
        std::string name;
        std::string type;
    };

    using SymbolMatch     = VariableMatch;
    using FunctionContext = VariableMatch;

    struct Parameter
    {
        std::string name;
        std::string type;
        std::string description;
    };
} // namespace patchestry::passes

LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::VariableMatch)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::ArgumentMatch)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::Parameter)

namespace llvm::yaml {
    using namespace patchestry::passes;

    // Parse Metadata
    template<>
    struct MappingTraits< patchestry::passes::Metadata >
    {
        static void mapping(IO &io, patchestry::passes::Metadata &metadata) {
            io.mapOptional("name", metadata.name);
            io.mapOptional("description", metadata.description);
            io.mapOptional("version", metadata.version);
            io.mapOptional("author", metadata.author);
            io.mapOptional("created", metadata.created);
            io.mapOptional("organization", metadata.organization);
        }
    };

    // Prase ArgumentMatch
    template<>
    struct MappingTraits< ArgumentMatch >
    {
        static void mapping(IO &io, ArgumentMatch &arg) {
            io.mapRequired("index", arg.index);
            io.mapRequired("name", arg.name);
            io.mapOptional("type", arg.type);
        }
    };

    // Prase VariableMatch
    template<>
    struct MappingTraits< VariableMatch >
    {
        static void mapping(IO &io, VariableMatch &var) {
            io.mapRequired("name", var.name);
            io.mapOptional("type", var.type);
        }
    };

    // Parse Parameter
    template<>
    struct MappingTraits< Parameter >
    {
        static void mapping(IO &io, Parameter &param) {
            io.mapRequired("name", param.name);
            io.mapOptional("type", param.type);
            io.mapOptional("description", param.description);
        }
    };
} // namespace llvm::yaml
