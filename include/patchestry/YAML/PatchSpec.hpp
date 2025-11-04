/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
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
        struct ArgumentSource
        {
            ArgumentSourceType source;
            std::string name; // Descriptive name for the argument
            std::optional< unsigned >
                index; // Operand/argument index (required for OPERAND type)
            std::optional< std::string >
                symbol; // Symbol name (required for VARIABLE/SYMBOL type)
            std::optional< std::string > value; // Constant value (required for CONSTANT type)
            std::optional< bool > is_reference; // Whether the argument is a reference
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
            std::string description;
            std::string category;
            std::string severity;
            std::string code_file;
            std::string function_name;
            std::vector< Parameter > parameters;
            std::optional< std::string > patch_module;
        };

        struct MetaPatchConfig
        {
            std::string name;
            std::string description;
            std::set< std::string > optimization;
            std::vector< PatchAction > patch_actions;
        };

        [[maybe_unused]] inline std::string_view infoModeToString(PatchInfoMode mode) {
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
    } // namespace patch
} // namespace patchestry::passes

LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::ArgumentSource)
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
            io.mapOptional("description", spec.description);
            io.mapOptional("category", spec.category);
            io.mapOptional("severity", spec.severity);
            io.mapRequired("code_file", spec.code_file);
            io.mapRequired("function_name", spec.function_name);
            io.mapOptional("parameters", spec.parameters);
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
                action.mode = PatchInfoMode::APPLY_BEFORE;
            } else if (mode_str == "ApplyAfter" || mode_str == "apply_after") {
                action.mode = PatchInfoMode::APPLY_AFTER;
            } else if (mode_str == "Replace" || mode_str == "replace") {
                action.mode = PatchInfoMode::REPLACE;
            } else {
                action.mode = PatchInfoMode::NONE;
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
                match.kind = MatchKind::OPERATION;
            } else if (kind_str == "function") {
                match.kind = MatchKind::FUNCTION;
            } else { // Default to NONE
                match.kind = MatchKind::NONE;
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

            // required block doesn't mean the block must be populated, so check.
            io.mapRequired("match", patch_action.match);
            if (patch_action.match.empty()) {
                io.setError(
                    "PatchAction '" + patch_action.action_id
                    + "' must include at least one 'match' entry."
                );
            }

            io.mapRequired("action", patch_action.action);
            if (patch_action.action.empty()) {
                io.setError(
                    "PatchAction '" + patch_action.action_id
                    + "' must include at least one 'action' entry."
                );
            }
        }
    };

    // Parse MetaPatchConfig
    template<>
    struct MappingTraits< patch::MetaPatchConfig >
    {
        static void mapping(IO &io, patch::MetaPatchConfig &meta_patch) {
            io.mapRequired("name", meta_patch.name);
            io.mapOptional("description", meta_patch.description);
            io.mapRequired("patch_actions", meta_patch.patch_actions);

            std::vector< std::string > optimization;
            io.mapOptional("optimization", optimization);
            for (const auto &opt : optimization) {
                meta_patch.optimization.insert(opt);
            }
            io.mapRequired("patch_actions", meta_patch.patch_actions);
            if (meta_patch.patch_actions.empty()) {
                LOG(ERROR) << "Meta patch has no patch actions";
                return;
            }
        }
    };

    // Parse ArgumentSource
    template<>
    struct MappingTraits< patch::ArgumentSource >
    {
        static void mapping(IO &io, patch::ArgumentSource &arg) {
            std::string source_str;
            io.mapRequired("source", source_str);

            if (source_str == "operand") {
                arg.source = ArgumentSourceType::OPERAND;
            } else if (source_str == "argument") {
                arg.source = ArgumentSourceType::OPERAND; // Treat argument
                                                          // same as operand
            } else if (source_str == "variable") {
                arg.source = ArgumentSourceType::VARIABLE;
            } else if (source_str == "symbol") {
                arg.source = ArgumentSourceType::SYMBOL;
            } else if (source_str == "constant") {
                arg.source = ArgumentSourceType::CONSTANT;
            } else if (source_str == "return_value") {
                arg.source = ArgumentSourceType::RETURN_VALUE;
            }

            io.mapRequired("name", arg.name);
            io.mapOptional("index", arg.index);
            io.mapOptional("symbol", arg.symbol);
            io.mapOptional("value", arg.value);
            io.mapOptional("is_reference", arg.is_reference);
        }
    };

} // namespace llvm::yaml
