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

        struct CaptureSpec
        {
            std::string name;
            std::optional< unsigned > operand; // operand index to bind
            std::optional< unsigned > result;  // result index to bind
            std::string type;                  // optional type constraint
        };

        struct MatchConfig
        {
            std::string name;
            MatchKind kind;
            std::optional< std::string > op_kind; // e.g. "mul" for cir.binop
            std::vector< FunctionContext > function_context;
            std::vector< ArgumentMatch > argument_matches;
            std::vector< VariableMatch > variable_matches;
            std::vector< SymbolMatch > symbol_matches;
            std::vector< OperandMatch > operand_matches;
            std::vector< CaptureSpec > captures;
        };

        struct Action
        {
            InstrumentationMode mode = InstrumentationMode::NONE;
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

        [[maybe_unused]] inline std::string_view infoModeToString(InstrumentationMode mode) {
            switch (mode) {
                case InstrumentationMode::NONE:
                    return "NONE";
                case InstrumentationMode::APPLY_BEFORE:
                    return "APPLY_BEFORE";
                case InstrumentationMode::APPLY_AFTER:
                    return "APPLY_AFTER";
                case InstrumentationMode::APPLY_AT_ENTRYPOINT:
                    return "APPLY_AT_ENTRYPOINT";
                case InstrumentationMode::REPLACE:
                    return "REPLACE";
                case InstrumentationMode::ERASE:
                    return "ERASE";
            }
            return "UNKNOWN";
        }

        // Simplified YAML surface that inflates to MetaPatchConfig.
        struct PatchEntry
        {
            std::string name;
            std::string id;
            std::string description;
            // match (single object, not array)
            std::vector< std::string > match_names; // one or more callee names (OR)
            MatchKind match_kind = MatchKind::FUNCTION;
            std::optional< std::string > op_kind; // e.g. "mul" for cir.binop
            std::vector< std::string > context; // function_context names
            std::vector< ArgumentMatch > argument_matches;
            std::vector< OperandMatch > operand_matches;
            std::vector< SymbolMatch > symbol_matches;
            std::vector< CaptureSpec > captures;
            // action (inlined)
            InstrumentationMode mode = InstrumentationMode::NONE;
            std::string patch_id;
            std::vector< ArgumentSource > arguments;
            // optimization
            std::set< std::string > optimization;
        };

        // Convert a PatchEntry to the canonical MetaPatchConfig representation.
        // Multiple match_names produce one PatchAction per name (OR semantics).
        [[maybe_unused]] inline MetaPatchConfig inflatePatchEntry(const PatchEntry &entry) {
            MetaPatchConfig meta;
            meta.name         = entry.name;
            meta.description  = entry.description;
            meta.optimization = entry.optimization;

            for (std::size_t i = 0; i < entry.match_names.size(); ++i) {
                PatchAction pa;
                pa.action_id = entry.id.empty()
                    ? entry.name + "/" + std::to_string(i)
                    : entry.id + "/" + std::to_string(i);
                pa.description = entry.description;

                MatchConfig mc;
                mc.name             = entry.match_names[i];
                mc.kind             = entry.match_kind;
                mc.op_kind          = entry.op_kind;
                mc.argument_matches = entry.argument_matches;
                mc.operand_matches  = entry.operand_matches;
                mc.symbol_matches   = entry.symbol_matches;
                mc.captures         = entry.captures;
                for (const auto &ctx_name : entry.context) {
                    FunctionContext fc;
                    fc.name = ctx_name;
                    mc.function_context.push_back(fc);
                }
                pa.match.push_back(std::move(mc));

                Action act;
                act.mode      = entry.mode;
                act.patch_id  = entry.patch_id;
                act.arguments = entry.arguments;
                pa.action.push_back(std::move(act));

                meta.patch_actions.push_back(std::move(pa));
            }
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
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::patch::CaptureSpec)

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
            io.mapOptional("patch_id", action.patch_id);
            io.mapOptional("description", action.description);
            io.mapOptional("arguments", action.arguments);

            std::string mode_str;
            io.mapRequired("mode", mode_str);
            if (mode_str == "ApplyBefore" || mode_str == "apply_before") {
                action.mode = InstrumentationMode::APPLY_BEFORE;
            } else if (mode_str == "ApplyAfter" || mode_str == "apply_after") {
                action.mode = InstrumentationMode::APPLY_AFTER;
            } else if (mode_str == "ApplyAtEntrypoint" || mode_str == "apply_at_entrypoint") {
                action.mode = InstrumentationMode::APPLY_AT_ENTRYPOINT;
            } else if (mode_str == "Replace" || mode_str == "replace") {
                action.mode = InstrumentationMode::REPLACE;
            } else if (mode_str == "Erase" || mode_str == "erase") {
                action.mode = InstrumentationMode::ERASE;
            } else {
                LOG(ERROR) << "Unknown patch mode: '" << mode_str
                           << "'. Valid modes: ApplyBefore, ApplyAfter, "
                              "ApplyAtEntrypoint, Replace, Erase";
                action.mode = InstrumentationMode::NONE;
            }

            if (action.mode != InstrumentationMode::ERASE && action.patch_id.empty()) {
                io.setError("'patch_id' is required for mode '" + mode_str + "'");
            }
        }
    };

    // Parse CaptureSpec
    template<>
    struct MappingTraits< patch::CaptureSpec >
    {
        static void mapping(IO &io, patch::CaptureSpec &cap) {
            io.mapRequired("name", cap.name);
            io.mapOptional("operand", cap.operand);
            io.mapOptional("result", cap.result);
            io.mapOptional("type", cap.type);

            if (!cap.operand.has_value() && !cap.result.has_value()) {
                io.setError(
                    "capture '" + cap.name
                    + "' must specify either 'operand' or 'result'"
                );
            }
            if (cap.operand.has_value() && cap.result.has_value()) {
                io.setError(
                    "capture '" + cap.name
                    + "' cannot specify both 'operand' and 'result'"
                );
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
            io.mapOptional("captures", match.captures);
            io.mapOptional("op_kind", match.op_kind);

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
            } else if (source_str == "capture") {
                arg.source = ArgumentSourceType::CAPTURE;
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

    // Helper struct for parsing the `match:` object inside PatchEntry.
    // Supports `name:` (single callee) or `names:` (list, OR semantics).
    struct PatchMatchObject
    {
        std::vector< std::string > names;
        MatchKind kind = MatchKind::FUNCTION;
        std::optional< std::string > op_kind;
        std::vector< std::string > context;
        std::vector< ArgumentMatch > argument_matches;
        std::vector< OperandMatch > operand_matches;
        std::vector< SymbolMatch > symbol_matches;
        std::vector< patch::CaptureSpec > captures;
    };

    template<>
    struct MappingTraits< PatchMatchObject >
    {
        static void mapping(IO &io, PatchMatchObject &m) {
            // `name:` for single callee, `names:` for list (OR).
            // Mutually exclusive.
            std::string single_name;
            io.mapOptional("name", single_name);
            io.mapOptional("names", m.names);

            if (!single_name.empty() && !m.names.empty()) {
                io.setError("'name' and 'names' are mutually exclusive in match");
                return;
            }
            if (!single_name.empty()) {
                m.names.push_back(single_name);
            }
            if (m.names.empty()) {
                io.setError("match must include 'name' or 'names'");
                return;
            }

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
            io.mapOptional("captures", m.captures);
            io.mapOptional("op_kind", m.op_kind);
        }
    };

    // Parse PatchEntry — simplified YAML surface for patches.
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
            entry.match_names      = match_obj.names;
            entry.match_kind       = match_obj.kind;
            entry.op_kind          = match_obj.op_kind;
            entry.context          = match_obj.context;
            entry.argument_matches = match_obj.argument_matches;
            entry.operand_matches  = match_obj.operand_matches;
            entry.symbol_matches   = match_obj.symbol_matches;
            entry.captures         = match_obj.captures;

            // Mode
            std::string mode_str;
            io.mapRequired("mode", mode_str);
            if (mode_str == "ApplyBefore" || mode_str == "apply_before") {
                entry.mode = InstrumentationMode::APPLY_BEFORE;
            } else if (mode_str == "ApplyAfter" || mode_str == "apply_after") {
                entry.mode = InstrumentationMode::APPLY_AFTER;
            } else if (mode_str == "ApplyAtEntrypoint" || mode_str == "apply_at_entrypoint") {
                entry.mode = InstrumentationMode::APPLY_AT_ENTRYPOINT;
            } else if (mode_str == "Replace" || mode_str == "replace") {
                entry.mode = InstrumentationMode::REPLACE;
            } else if (mode_str == "Erase" || mode_str == "erase") {
                entry.mode = InstrumentationMode::ERASE;
            } else {
                io.setError("Unknown patch mode: '" + mode_str + "'");
                entry.mode = InstrumentationMode::NONE;
            }

            io.mapOptional("patch", entry.patch_id);
            io.mapOptional("arguments", entry.arguments);

            if (entry.mode != InstrumentationMode::ERASE && entry.patch_id.empty()) {
                io.setError("'patch' field is required for mode '" + mode_str + "'");
            }

            std::vector< std::string > optimization;
            io.mapOptional("optimization", optimization);
            for (const auto &opt : optimization) {
                entry.optimization.insert(opt);
            }
        }
    };

} // namespace llvm::yaml
