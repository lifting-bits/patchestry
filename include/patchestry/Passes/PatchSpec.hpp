/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/YAMLTraits.h>

#include <patchestry/Passes/BaseSpec.hpp>
#include <patchestry/Util/Log.hpp>
#include <patchestry/YAML/YAMLParser.hpp>

namespace patchestry::passes {
    namespace patch {
        enum class PatchInfoMode : uint8_t {
            NONE = 0, // No patch
            APPLY_BEFORE,
            APPLY_AFTER,
            REPLACE
        };

        enum class MatchKind : uint8_t { NONE = 0, OPERATION, FUNCTION };

        enum class ArgumentSourceType : uint8_t {
            OPERAND = 0, // Reference to operation operand by index
            VARIABLE,    // Reference to variable by name
            SYMBOL,      // Reference to symbol by name
            CONSTANT,    // Literal constant value
            RETURN_VALUE // Return value of function or operation
        };

        struct ArgumentSource
        {
            ArgumentSourceType source;
            std::string name;                // Descriptive name for the argument
            std::optional< unsigned > index; // Operand/argument index (required for OPERAND type)
            std::optional< std::string > symbol; // Symbol name (required for VARIABLE/SYMBOL type)
            std::optional< std::string > value;  // Constant value (required for CONSTANT type)
        };

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

        struct Implementation
        {
            std::string language;
            std::string code_file;
            std::string function_name;
            std::vector< Parameter > parameters;
            std::vector< std::string > dependencies;
        };

        struct MatchConfig
        {
            std::string name;
            MatchKind kind;
            std::vector< FunctionContext > function_context;
            std::vector< ArgumentMatch > argument_matches;
            std::vector< VariableMatch > variable_matches;
            std::vector< SymbolMatch > symbol_matches;
            std::vector< OperandMatch > operand_matches;
        };

        struct Action
        {
            PatchInfoMode mode = PatchInfoMode::NONE;
            std::string patch_id;
            std::string description;
            std::vector< ArgumentSource > arguments;
        };

        struct PatchAction
        {
            std::string action_id;
            std::string description;
            std::vector< MatchConfig > match;
            std::vector< Action > action;
        };

        struct PatchSpec
        {
            std::string name;
            std::string id;
            std::string description;
            std::string category;
            std::string severity;
            Implementation implementation;
            std::optional< std::string > patch_module;
        };

        struct PatchLibrary
        {
            std::string api_version;
            Metadata metadata;
            std::vector< PatchSpec > patches;
        };

        struct MetaPatchConfig
        {
            std::string name;
            std::string id;
            std::string description;
            std::set< std::string > optimization;
            std::vector< PatchAction > patch_actions;
        };

        [[maybe_unused]] inline std::string_view patchInfoModeToString(PatchInfoMode mode) {
            switch (mode) {
                case PatchInfoMode::NONE:
                    return "NONE";
                case PatchInfoMode::APPLY_BEFORE:
                    return "APPLY_BEFORE";
                case PatchInfoMode::APPLY_AFTER:
                    return "APPLY_AFTER";
                case PatchInfoMode::REPLACE:
                    return "REPLACE";
            }
            return "UNKNOWN";
        }
    } // namespace patchestry::passes::patch
} // namespace patchestry::passes

namespace patchestry::yaml {
    using namespace patchestry::passes::patch;

    namespace utils {
        [[maybe_unused]] static std::optional< PatchLibrary >
        loadPatchLibrary(const std::string &file_path) {
            YAMLParser parser;
            auto result = parser.parse_from_file< PatchLibrary >(file_path);
            if (!result) {
                LOG(ERROR) << "Failed to load patch library: " << file_path << "\n";
                return std::nullopt;
            }
            return result;
        }
    }
}

LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::PatchSpec)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::VariableMatch)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::ArgumentMatch)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::ArgumentSource)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::MatchConfig)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::Parameter)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::Action)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::PatchAction)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::MetaPatchConfig)

namespace llvm::yaml {
    using namespace patchestry::passes;

    // Parse ArgumentSource
    template<>
    struct MappingTraits< patch::ArgumentSource >
    {
        static void mapping(IO &io, patch::ArgumentSource &arg) {
            std::string source_str;
            io.mapRequired("source", source_str);

            if (source_str == "operand") {
                arg.source = patch::ArgumentSourceType::OPERAND;
            } else if (source_str == "argument") {
                arg.source = patch::ArgumentSourceType::OPERAND; // Treat argument
                                                                              // same as operand
            } else if (source_str == "variable") {
                arg.source = patch::ArgumentSourceType::VARIABLE;
            } else if (source_str == "symbol") {
                arg.source = patch::ArgumentSourceType::SYMBOL;
            } else if (source_str == "constant") {
                arg.source = patch::ArgumentSourceType::CONSTANT;
            } else if (source_str == "return_value") {
                arg.source = patch::ArgumentSourceType::RETURN_VALUE;
            }

            io.mapRequired("name", arg.name);
            io.mapOptional("index", arg.index);
            io.mapOptional("symbol", arg.symbol);
            io.mapOptional("value", arg.value);
        }
    };

    // Prase ArgumentMatch
    template<>
    struct MappingTraits< patch::ArgumentMatch >
    {
        static void mapping(IO &io, patch::ArgumentMatch &arg) {
            io.mapRequired("index", arg.index);
            io.mapRequired("name", arg.name);
            io.mapOptional("type", arg.type);
        }
    };

    // Prase VariableMatch
    template<>
    struct MappingTraits< patch::VariableMatch >
    {
        static void mapping(IO &io, patch::VariableMatch &var) {
            io.mapRequired("name", var.name);
            io.mapOptional("type", var.type);
        }
    };

    // Parse Parameter
    template<>
    struct MappingTraits< patch::Parameter >
    {
        static void mapping(IO &io, patch::Parameter &param) {
            io.mapRequired("name", param.name);
            io.mapOptional("type", param.type);
            io.mapOptional("description", param.description);
        }
    };

    // Parse Implementation
    template<>
    struct MappingTraits< patch::Implementation >
    {
        static void mapping(IO &io, patch::Implementation &impl) {
            io.mapOptional("language", impl.language);
            io.mapRequired("code_file", impl.code_file);
            io.mapOptional("function_name", impl.function_name);
            io.mapRequired("parameters", impl.parameters);
            io.mapOptional("dependencies", impl.dependencies);
        }
    };

    // Parse PatchSpec
    template<>
    struct MappingTraits< patch::PatchSpec >
    {
        static void mapping(IO &io, patch::PatchSpec &spec) {
            io.mapRequired("name", spec.name);
            io.mapRequired("id", spec.id);
            io.mapOptional("description", spec.description);
            io.mapOptional("category", spec.category);
            io.mapOptional("severity", spec.severity);
            io.mapRequired("implementation", spec.implementation);
        }
    };

    // Parse Action
    template<>
    struct MappingTraits< patch::Action >
    {
        static void mapping(IO &io, patch::Action &action) {
            io.mapRequired("patch_id", action.patch_id);
            io.mapOptional("description", action.description);
            io.mapOptional("arguments", action.arguments);

            std::string mode_str;
            io.mapRequired("mode", mode_str);
            if (mode_str == "ApplyBefore" || mode_str == "apply_before") {
                action.mode = patch::PatchInfoMode::APPLY_BEFORE;
            } else if (mode_str == "ApplyAfter" || mode_str == "apply_after") {
                action.mode = patch::PatchInfoMode::APPLY_AFTER;
            } else if (mode_str == "Replace" || mode_str == "replace") {
                action.mode = patch::PatchInfoMode::REPLACE;
            } else {
                action.mode = patch::PatchInfoMode::NONE;
            }
        }
    };

    // Parse PatchMatch
    template<>
    struct MappingTraits< patch::MatchConfig >
    {
        static void mapping(IO &io, patch::MatchConfig &match) {
            io.mapRequired("name", match.name);
            io.mapOptional("function_context", match.function_context);
            io.mapOptional("argument_matches", match.argument_matches);
            io.mapOptional("variable_matches", match.variable_matches);
            io.mapOptional("symbol_matches", match.symbol_matches);
            io.mapOptional("operand_matches", match.operand_matches);

            std::string kind_str;
            io.mapRequired("kind", kind_str);
            if (kind_str == "operation") {
                match.kind = patch::MatchKind::OPERATION;
            } else if (kind_str == "function") {
                match.kind = patch::MatchKind::FUNCTION;
            } else { // Default to NONE
                match.kind = patch::MatchKind::NONE;
            }
        }
    };

    // Parse PatchAction
    template<>
    struct MappingTraits< patch::PatchAction >
    {
        static void mapping(IO &io, patch::PatchAction &patch_action) {
            io.mapRequired("id", patch_action.action_id);
            io.mapOptional("description", patch_action.description);
            io.mapRequired("match", patch_action.match);
            io.mapRequired("action", patch_action.action);
        }
    };



    // Parse PatchLibrary
    template<>
    struct MappingTraits< patch::PatchLibrary >
    {
        static void mapping(IO &io, patch::PatchLibrary &library) {
            io.mapOptional("apiVersion", library.api_version);
            io.mapOptional("metadata", library.metadata);
            io.mapRequired("patches", library.patches);
        }
    };

    // Parse MetaPatchConfig
    template<>
    struct MappingTraits< patch::MetaPatchConfig >
    {
        static void mapping(IO &io, patch::MetaPatchConfig &meta_patch) {
            io.mapRequired("name", meta_patch.name);
            io.mapRequired("id", meta_patch.id);
            io.mapOptional("description", meta_patch.description);

            std::vector< std::string > optimization;
            io.mapOptional("optimization", optimization);
            for (const auto &opt : optimization) {
                meta_patch.optimization.insert(opt);
            }
            io.mapRequired("patch_actions", meta_patch.patch_actions);
        }
    };
} // namespace llvm::yaml
