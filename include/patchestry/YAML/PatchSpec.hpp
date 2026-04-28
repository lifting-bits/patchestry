/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cassert>
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

        // Helper for parsing the nested `match:` object inside PatchEntry.
        // Kept in the patch namespace (not llvm::yaml) so the struct lives
        // with the types it serves; only its MappingTraits specialization is
        // in llvm::yaml. Supports `name:` (single callee) or `names:` (list,
        // OR semantics).
        struct PatchMatchObject
        {
            std::vector< std::string > names;
            MatchKind kind = MatchKind::FUNCTION;
            std::optional< std::string > op_kind;
            std::vector< std::string > context;
            std::vector< ArgumentMatch > argument_matches;
            std::vector< OperandMatch > operand_matches;
            std::vector< SymbolMatch > symbol_matches;
            std::vector< VariableMatch > variable_matches;
            std::vector< CaptureSpec > captures;
        };

        // Sibling to PatchMatchObject for parsing the nested `action:` block.
        // Carries the "what to do when a match fires" fields that used to sit
        // flat at the top level of a PatchEntry.
        struct PatchActionObject
        {
            InstrumentationMode mode = InstrumentationMode::NONE;
            std::string patch_id;
            std::vector< ArgumentSource > arguments;
            std::set< std::string > optimization;
        };

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
            std::vector< VariableMatch > variable_matches;
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

            // When the parser hits an error (e.g. schema validation failed
            // on this entry), match_names may be empty. llvm::yaml::IO
            // continues calling mapping/inflation after setError so it can
            // accumulate diagnostics; the resulting Configuration gets
            // discarded by the caller. Return an empty MetaPatchConfig
            // here rather than emitting an action with an empty match
            // name (which the matcher would reject silently later).
            if (entry.match_names.empty()) {
                return meta;
            }

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
                mc.variable_matches = entry.variable_matches;
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

    // Helper: parse a YAML mode string into InstrumentationMode, reporting
    // the right error via `io`. Shared between the legacy `meta_patches:`
    // action parser and the flat `patches:` entry parser so the allowed
    // spellings and error messages stay in sync.
    inline InstrumentationMode parsePatchInfoMode(
        IO &io, const std::string &mode_str
    ) {
        if (mode_str == "ApplyBefore" || mode_str == "apply_before") {
            return InstrumentationMode::APPLY_BEFORE;
        }
        if (mode_str == "ApplyAfter" || mode_str == "apply_after") {
            return InstrumentationMode::APPLY_AFTER;
        }
        if (mode_str == "ApplyAtEntrypoint" || mode_str == "apply_at_entrypoint") {
            return InstrumentationMode::APPLY_AT_ENTRYPOINT;
        }
        if (mode_str == "Replace" || mode_str == "replace") {
            return InstrumentationMode::REPLACE;
        }
        if (mode_str == "Erase" || mode_str == "erase") {
            return InstrumentationMode::ERASE;
        }
        io.setError(
            "Unknown patch mode: '" + mode_str
            + "'. Valid modes: ApplyBefore, ApplyAfter, "
              "ApplyAtEntrypoint, Replace, Erase"
        );
        return InstrumentationMode::NONE;
    }

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
            action.mode = parsePatchInfoMode(io, mode_str);

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

            // `name:` is optional for most sources: an ArgumentSource is
            // identified by `source:` plus whichever of `index:` / `symbol:`
            // / `value:` the chosen kind requires, and `name:` is descriptive
            // metadata used for logs. For `source: capture`, however, `name:`
            // is the *only* identifier — it must match an entry in
            // `match.captures`. Enforce it at parse time so the failure mode
            // is a clear spec error, not a runtime LOG(ERROR) per call site.
            io.mapOptional("name", arg.name);
            io.mapOptional("index", arg.index);
            io.mapOptional("symbol", arg.symbol);
            io.mapOptional("value", arg.value);
            io.mapOptional("is_reference", arg.is_reference);

            if (arg.source == ArgumentSourceType::CAPTURE && arg.name.empty()) {
                io.setError(
                    "argument with 'source: capture' requires 'name:' "
                    "(it identifies which entry in match.captures to bind)"
                );
            }
        }
    };

    template<>
    struct MappingTraits< patch::PatchMatchObject >
    {
        static void mapping(IO &io, patch::PatchMatchObject &m) {
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

            // `kind:` must be one of "operation" / "function" (default). A
            // typo like "funciton" previously fell through to FUNCTION
            // silently, masking the spec error.
            std::string kind_str;
            io.mapOptional("kind", kind_str);
            if (kind_str.empty() || kind_str == "function") {
                m.kind = MatchKind::FUNCTION;
            } else if (kind_str == "operation") {
                m.kind = MatchKind::OPERATION;
            } else {
                io.setError(
                    "Unknown match kind: '" + kind_str
                    + "'. Valid kinds: \"function\" (default), \"operation\"."
                );
                return;
            }

            io.mapOptional("context", m.context);
            io.mapOptional("argument_matches", m.argument_matches);
            io.mapOptional("operand_matches", m.operand_matches);
            io.mapOptional("symbol_matches", m.symbol_matches);
            io.mapOptional("variable_matches", m.variable_matches);
            io.mapOptional("captures", m.captures);
            io.mapOptional("op_kind", m.op_kind);
        }
    };

    template<>
    struct MappingTraits< patch::PatchActionObject >
    {
        static void mapping(IO &io, patch::PatchActionObject &a) {
            std::string mode_str;
            io.mapRequired("mode", mode_str);
            a.mode = parsePatchInfoMode(io, mode_str);

            io.mapOptional("patch", a.patch_id);
            io.mapOptional("arguments", a.arguments);

            if (a.mode != InstrumentationMode::ERASE && a.patch_id.empty()) {
                io.setError("'patch' field is required for mode '" + mode_str + "'");
            }

            std::vector< std::string > optimization;
            io.mapOptional("optimization", optimization);
            for (const auto &opt : optimization) {
                a.optimization.insert(opt);
            }
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
            patch::PatchMatchObject match_obj;
            io.mapRequired("match", match_obj);
            entry.match_names      = match_obj.names;
            entry.match_kind       = match_obj.kind;
            entry.op_kind          = match_obj.op_kind;
            entry.context          = match_obj.context;
            entry.argument_matches = match_obj.argument_matches;
            entry.operand_matches  = match_obj.operand_matches;
            entry.symbol_matches   = match_obj.symbol_matches;
            entry.variable_matches = match_obj.variable_matches;
            entry.captures         = match_obj.captures;

            // Parse nested action object — projected back onto the flat
            // PatchEntry fields so inflatePatchEntry keeps working unchanged.
            patch::PatchActionObject action_obj;
            io.mapRequired("action", action_obj);
            entry.mode         = action_obj.mode;
            entry.patch_id     = action_obj.patch_id;
            entry.arguments    = action_obj.arguments;
            entry.optimization = action_obj.optimization;

            // Cross-field validation: `apply_at_entrypoint` inserts the
            // patch call into the enclosing function's entry block. Only
            // the function-kind dispatch path in InstrumentationPass
            // implements it; the operation-kind switch has no case and
            // silently logs an error without signalling pass failure.
            // Reject the combo up-front so the misconfiguration surfaces
            // at spec-load with a clear error pointing at the match
            // rather than disappearing into a "succeeded but did nothing"
            // transform run.
            if (action_obj.mode == InstrumentationMode::APPLY_AT_ENTRYPOINT
                && match_obj.kind != MatchKind::FUNCTION)
            {
                io.setError(
                    "'mode: apply_at_entrypoint' requires 'match.kind: function'; "
                    "the operation-kind dispatch path does not implement it."
                );
            }

            // Cross-field validation: argument sources whose value is
            // only bound at the matched call site cannot be used by an
            // entrypoint-inserted patch call, which runs in the enclosing
            // function's entry block and must satisfy SSA dominance
            // there. The runtime dispatch in
            // `handle_return_value_argument` / `handle_capture_argument`
            // does reject these, but only *after* the rest of
            // `prepare_patch_call_arguments` has decided to build a call
            // — the truncated arg_map then yields a malformed `cir.call`
            // that trips the CIR verifier with an arity error, rather
            // than a clear spec-level diagnostic. Reject up-front so
            // `patchir-yaml-parser --validate` catches the mistake and
            // CI pipelines gating on it don't get a false positive.
            if (action_obj.mode == InstrumentationMode::APPLY_AT_ENTRYPOINT) {
                for (const auto &arg : action_obj.arguments) {
                    if (arg.source == ArgumentSourceType::CAPTURE) {
                        io.setError(
                            "'mode: apply_at_entrypoint' does not accept "
                            "'source: capture'; captures are bound at the "
                            "matched call site and do not dominate the "
                            "enclosing function's entry block. Use "
                            "'operand', 'variable', 'symbol', or "
                            "'constant' instead."
                        );
                    } else if (arg.source == ArgumentSourceType::RETURN_VALUE) {
                        io.setError(
                            "'mode: apply_at_entrypoint' does not accept "
                            "'source: return_value'; the call result is "
                            "only defined at the matched call site and "
                            "cannot be referenced from the enclosing "
                            "function's entry block. Use 'operand', "
                            "'variable', 'symbol', or 'constant' instead."
                        );
                    }
                }
            }

            // Cross-field validation: `op_kind:` only makes sense for
            // `cir.binop` / `cir.cmp` — the matcher stringifies their
            // BinOpKind / CmpOpKind attribute for comparison. Setting it
            // on any other op (or mis-spelling the op name) silently
            // drops every match at runtime with only a DEBUG log. Reject
            // at parse so a typo surfaces loudly instead of yielding a
            // "succeeded but did nothing" transform run.
            if (match_obj.op_kind.has_value()
                && !match_obj.names.empty())
            {
                bool any_supported = false;
                for (const auto &n : match_obj.names) {
                    if (n == "cir.binop" || n == "cir.cmp") {
                        any_supported = true;
                        break;
                    }
                }
                if (!any_supported) {
                    io.setError(
                        "'op_kind:' is only meaningful for 'cir.binop' "
                        "and 'cir.cmp'; setting it on any other op "
                        "silently drops every match at runtime. Remove "
                        "the 'op_kind:' entry, or change 'match.name' "
                        "to 'cir.binop' / 'cir.cmp' if the intent was "
                        "to filter an arithmetic or comparison kind."
                    );
                }
            }

            // Cross-field validation: `mode: replace` on a kinded generic
            // op (cir.binop, cir.cmp) requires an `op_kind` filter.
            // Without it the match is a wildcard that fires on every
            // arithmetic-or-comparison op in scope and substitutes the
            // same patch call for add, sub, mul, div, shl, and/or, etc.
            // The CIR verifier is happy (operand types match) but the
            // semantics are silently wrong — a `patch__replace__int_mul`
            // installed over every `+`/`-`/`/`/`<<` just miscomputes the
            // program. Logging or observational modes (apply_before /
            // apply_after) have a legitimate wildcard use case
            // (operation counters, trace probes) so they are left
            // unrestricted; replace must narrow to a single kind.
            if (action_obj.mode == InstrumentationMode::REPLACE
                && !match_obj.op_kind.has_value())
            {
                for (const auto &n : match_obj.names) {
                    if (n == "cir.binop" || n == "cir.cmp") {
                        io.setError(
                            "'mode: replace' on a kinded generic op ('"
                            + n
                            + "') requires an 'op_kind' filter. A wildcard "
                            "match replaces every arithmetic/comparison op in "
                            "scope with the same patch call — the CIR "
                            "verifier accepts it but the semantics are "
                            "silently wrong across kinds. Narrow the match "
                            "with e.g. 'op_kind: \"mul\"', or use "
                            "'apply_before' / 'apply_after' for observational "
                            "probes that genuinely want the wildcard."
                        );
                        break;
                    }
                }
            }
        }
    };

} // namespace llvm::yaml
