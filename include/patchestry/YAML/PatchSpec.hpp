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

#include <patchestry/Util/Log.hpp>
#include <patchestry/YAML/BaseSpec.hpp>
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
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::MatchConfig)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::Action)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::PatchAction)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::MetaPatchConfig)

namespace llvm::yaml {
    using namespace patchestry::passes;

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
