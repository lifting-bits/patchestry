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

        // Flat patch entry: simplified YAML surface that inflates to MetaPatchConfig.
        struct PatchEntry
        {
            std::string name;
            std::string id;
            std::string description;
            // match (single object, not array)
            std::string match_name;
            MatchKind match_kind = MatchKind::FUNCTION;
            std::vector< std::string > context; // function_context names
            std::vector< ArgumentMatch > argument_matches;
            std::vector< OperandMatch > operand_matches;
            std::vector< SymbolMatch > symbol_matches;
            // action (inlined)
            PatchInfoMode mode = PatchInfoMode::NONE;
            std::string patch_id;
            std::vector< ArgumentSource > arguments;
            // optimization
            std::set< std::string > optimization;
        };

        // Convert a PatchEntry to the canonical MetaPatchConfig representation.
        [[maybe_unused]] inline MetaPatchConfig inflatePatchEntry(const PatchEntry &flat) {
            MetaPatchConfig meta;
            meta.name         = flat.name;
            meta.description  = flat.description;
            meta.optimization = flat.optimization;

            PatchAction pa;
            pa.action_id   = flat.id.empty() ? flat.name + "/0" : flat.id;
            pa.description = flat.description;

            MatchConfig mc;
            mc.name             = flat.match_name;
            mc.kind             = flat.match_kind;
            mc.argument_matches = flat.argument_matches;
            mc.operand_matches  = flat.operand_matches;
            mc.symbol_matches   = flat.symbol_matches;
            for (const auto &ctx_name : flat.context) {
                FunctionContext fc;
                fc.name = ctx_name;
                mc.function_context.push_back(fc);
            }
            pa.match.push_back(std::move(mc));

            Action act;
            act.mode      = flat.mode;
            act.patch_id  = flat.patch_id;
            act.arguments = flat.arguments;
            pa.action.push_back(std::move(act));

            meta.patch_actions.push_back(std::move(pa));
            return meta;
        }

    } // namespace patch
} // namespace patchestry::passes

LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::ArgumentSource)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::PatchSpec)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::MatchConfig)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::Action)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::PatchAction)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::MetaPatchConfig)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::PatchEntry)

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
                LOG(ERROR) << "Unknown patch mode: '" << mode_str
                           << "'. Valid modes: ApplyBefore, ApplyAfter, Replace";
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
            } else {
                io.setError("Unknown argument source type: '" + source_str + "'");
                return;
            }

            io.mapOptional("name", arg.name);
            io.mapOptional("index", arg.index);
            io.mapOptional("symbol", arg.symbol);
            io.mapOptional("value", arg.value);
            io.mapOptional("is_reference", arg.is_reference);
        }
    };

    // Helper struct for parsing the flat `match:` object inside PatchEntry.
    struct PatchMatchObject
    {
        std::string name;
        MatchKind kind = MatchKind::FUNCTION;
        std::vector< std::string > context;
        std::vector< ArgumentMatch > argument_matches;
        std::vector< OperandMatch > operand_matches;
        std::vector< SymbolMatch > symbol_matches;
    };

    template<>
    struct MappingTraits< PatchMatchObject >
    {
        static void mapping(IO &io, PatchMatchObject &m) {
            io.mapRequired("name", m.name);

            std::string kind_str;
            io.mapOptional("kind", kind_str);
            if (kind_str == "operation") {
                m.kind = MatchKind::OPERATION;
            } else {
                m.kind = MatchKind::FUNCTION;
            }

            io.mapOptional("context", m.context);
            io.mapOptional("argument_matches", m.argument_matches);
            io.mapOptional("operand_matches", m.operand_matches);
            io.mapOptional("symbol_matches", m.symbol_matches);
        }
    };

    // Parse PatchEntry — simplified flat YAML surface for patches.
    template<>
    struct MappingTraits< patch::PatchEntry >
    {
        static void mapping(IO &io, patch::PatchEntry &entry) {
            io.mapRequired("name", entry.name);
            io.mapOptional("id", entry.id);
            io.mapOptional("description", entry.description);

            // Parse nested match object
            PatchMatchObject match_obj;
            io.mapRequired("match", match_obj);
            entry.match_name       = match_obj.name;
            entry.match_kind       = match_obj.kind;
            entry.context          = match_obj.context;
            entry.argument_matches = match_obj.argument_matches;
            entry.operand_matches  = match_obj.operand_matches;
            entry.symbol_matches   = match_obj.symbol_matches;

            // Mode
            std::string mode_str;
            io.mapRequired("mode", mode_str);
            if (mode_str == "ApplyBefore" || mode_str == "apply_before") {
                entry.mode = PatchInfoMode::APPLY_BEFORE;
            } else if (mode_str == "ApplyAfter" || mode_str == "apply_after") {
                entry.mode = PatchInfoMode::APPLY_AFTER;
            } else if (mode_str == "Replace" || mode_str == "replace") {
                entry.mode = PatchInfoMode::REPLACE;
            } else {
                io.setError("Unknown patch mode: '" + mode_str + "'");
                entry.mode = PatchInfoMode::NONE;
            }

            io.mapRequired("patch", entry.patch_id);
            io.mapOptional("arguments", entry.arguments);

            std::vector< std::string > optimization;
            io.mapOptional("optimization", optimization);
            for (const auto &opt : optimization) {
                entry.optimization.insert(opt);
            }
        }
    };

} // namespace llvm::yaml
