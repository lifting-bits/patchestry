/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <charconv>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/YAMLTraits.h>

#include <patchestry/Dialect/Contracts/ContractsDialect.hpp>
#include <patchestry/Util/Log.hpp>
#include <patchestry/YAML/BaseSpec.hpp>
#include <patchestry/YAML/YAMLParser.hpp>

namespace patchestry::passes {
    namespace contract {

        struct ParseResult {
            std::optional< uint64_t > value;
            std::string error;  // non-empty on failure
        };

        inline auto ParseUint64(std::string_view text) -> ParseResult {
            if (text.empty()) {
                return { std::nullopt, "empty string" };
            }
            uint64_t value = 0;
            auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (ec == std::errc::result_out_of_range) {
                return { std::nullopt, "value out of range for uint64" };
            }
            if (ec != std::errc{} || ptr != text.data() + text.size()) {
                return { std::nullopt, "not a valid integer" };
            }
            return { value, {} };
        }

        // Forward declarations from contract dialect
        namespace attr_types {
            using PredicateKind = ::contracts::PredicateKind;
            using TargetKind    = ::contracts::TargetKind;
            using RelationKind  = ::contracts::RelationKind;
        } // namespace attr_types

        // Contracts attach as MLIR attributes on the matched op. APPLY_BEFORE
        // and APPLY_AFTER currently produce the same attribute placement
        // (both set `contract.static` on the matched op); the distinction is
        // retained on the enum so the YAML surface and the dispatch in
        // `ContractOperationImpl` stay aligned with patches and leave room
        // for later differentiation. There is no runtime-call insertion
        // (that path lives under patches: now), so apply_at_entrypoint is
        // not a valid contract mode.
        enum class InfoMode : uint8_t {
            NONE = 0, // No contract
            APPLY_BEFORE,
            APPLY_AFTER
        };

        // Pin enum ordinals. Mirror the BaseSpec guard: these values end up
        // in MLIR attribute serialization and downstream bytecode, so a
        // silent reorder would break round-trips. Add new enumerators at
        // the end with a matching static_assert line.
        static_assert(static_cast< uint8_t >(InfoMode::NONE) == 0);
        static_assert(static_cast< uint8_t >(InfoMode::APPLY_BEFORE) == 1);
        static_assert(static_cast< uint8_t >(InfoMode::APPLY_AFTER) == 2);

        // Contracts can only match at the function level.
        enum class MatchKind : uint8_t { NONE = 0, FUNCTION };

        struct ArgumentSource
        {
            ArgumentSourceType source;
            std::string name; // Descriptive name for the argument
            std::optional< unsigned >
                index; // Operand/argument index (required for OPERAND type)
            std::optional< std::string >
                symbol; // Symbol name (required for VARIABLE/SYMBOL type)
            std::optional< std::string > value; // Constant value (required for CONSTANT type)
        };

        struct MatchConfig
        {
            std::string name;
            MatchKind kind;
            std::vector< FunctionContext > function_context;
            std::vector< ArgumentMatch > argument_matches;
            std::vector< VariableMatch > variable_matches;
            std::vector< SymbolMatch > symbol_matches;
        };

        struct Action
        {
            InfoMode mode = InfoMode::NONE;
            std::string contract_id;
            std::string description;
            std::vector< ArgumentSource > arguments;
        };

        struct ContractAction
        {
            std::string action_id;
            std::string description;
            std::vector< MatchConfig > match;
            std::vector< Action > action;
        };

        struct Range
        {
            std::string min;
            std::string max;
        };

        struct Value
        {
            std::optional< std::string > constant;
            std::optional< std::string > symbol;
        };

        struct Predicate
        {
            attr_types::PredicateKind kind;
            attr_types::TargetKind target;
            std::optional< uint64_t > arg_index;
            attr_types::RelationKind relation;
            std::optional< std::string >
                value; // For constant values (to be parsed as integers)
            std::optional< std::string > symbol; // Symbol name
            std::optional< uint64_t > align;     // Alignment value
            std::optional< std::string > expr;   // Expression string
            std::optional< Range > range;        // Range constraint
        };

        struct StaticContractRequirement
        {
            std::string id;
            std::optional< std::string > description;
            std::optional< Predicate > pred;
        };

        // Contracts are static only. Runtime contracts have been merged into
        // patches — see patches: in PatchSpec.hpp for C-function-based
        // instrumentation. A ContractSpec carries the declarative
        // pre/postconditions attached as MLIR attributes by the pass.
        struct ContractSpec
        {
            std::string name;
            std::string description;
            std::string category;
            std::string severity;
            std::optional< std::vector< StaticContractRequirement > > preconditions;
            std::optional< std::vector< StaticContractRequirement > > postconditions;
        };

        struct MetaContractConfig
        {
            std::string name;
            std::string description;
            std::vector< ContractAction > contract_actions;
        };

        [[maybe_unused]] inline std::string_view infoModeToString(InfoMode mode) {
            switch (mode) {
                case InfoMode::NONE:
                    return "NONE";
                case InfoMode::APPLY_BEFORE:
                    return "APPLY_BEFORE";
                case InfoMode::APPLY_AFTER:
                    return "APPLY_AFTER";
            }
            return "UNKNOWN";
        }

        // Helper for parsing the nested `match:` object inside ContractEntry.
        // Kept in the contract namespace (not llvm::yaml) so the struct lives
        // with the types it serves; only its MappingTraits specialization is
        // in llvm::yaml.
        struct ContractMatchObject
        {
            std::string name;
            std::vector< std::string > context;
        };

        // Sibling to ContractMatchObject for parsing the nested `action:`
        // block on a contract entry. Arguments are accepted for schema
        // symmetry with patches but warned-and-ignored at parse time
        // (static contracts attach MLIR attributes, not runtime calls).
        struct ContractActionObject
        {
            InfoMode mode = InfoMode::NONE;
            std::string contract_id;
            std::vector< ArgumentSource > arguments;
            bool optimization_warned = false;
        };

        // Simplified contract entry that inflates to MetaContractConfig.
        struct ContractEntry
        {
            std::string name;
            std::string id;
            std::string description;
            // match (single object)
            std::string match_name;
            std::vector< std::string > context; // function_context names
            // action (inlined)
            InfoMode mode = InfoMode::NONE;
            std::string contract_id;
            std::vector< ArgumentSource > arguments;
        };

        // Convert a ContractEntry to the canonical MetaContractConfig representation.
        [[maybe_unused]] inline MetaContractConfig inflateContractEntry(const ContractEntry &entry) {
            MetaContractConfig meta;
            meta.name         = entry.name;
            meta.description  = entry.description;

            ContractAction ca;
            ca.action_id   = entry.id.empty() ? entry.name + "/0" : entry.id;
            ca.description = entry.description;

            MatchConfig mc;
            mc.name = entry.match_name;
            mc.kind = MatchKind::FUNCTION;
            for (const auto &ctx_name : entry.context) {
                FunctionContext fc;
                fc.name = ctx_name;
                mc.function_context.push_back(fc);
            }
            ca.match.push_back(std::move(mc));

            Action act;
            act.mode        = entry.mode;
            act.contract_id = entry.contract_id;
            act.arguments   = entry.arguments;
            ca.action.push_back(std::move(act));

            meta.contract_actions.push_back(std::move(ca));
            return meta;
        }

    } // namespace contract
} // namespace patchestry::passes

LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::ArgumentSource)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::MatchConfig)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::Action)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::ContractAction)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::ContractSpec)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::StaticContractRequirement)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::MetaContractConfig)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::Range)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::Value)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::Predicate)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::ContractEntry)

namespace llvm::yaml {
    using namespace patchestry::passes;

    // Helper: parse a YAML mode string into contract::InfoMode, reporting
    // the right error via `io`. Shared between the legacy `meta_contracts:`
    // action parser and the flat `contracts:` entry parser so the allowed
    // spellings and error messages stay in sync.
    inline contract::InfoMode parseContractInfoMode(
        IO &io, const std::string &mode_str
    ) {
        if (mode_str == "ApplyBefore" || mode_str == "apply_before") {
            return contract::InfoMode::APPLY_BEFORE;
        }
        if (mode_str == "ApplyAfter" || mode_str == "apply_after") {
            return contract::InfoMode::APPLY_AFTER;
        }
        if (mode_str == "ApplyAtEntrypoint" || mode_str == "apply_at_entrypoint"
            || mode_str == "apply_at_entry")
        {
            io.setError(
                "apply_at_entrypoint is not supported for contracts — "
                "contracts are static only. Move the entry under patches: "
                "and use mode: apply_at_entrypoint there."
            );
            return contract::InfoMode::NONE;
        }
        io.setError(
            "Unsupported contract mode: '" + mode_str
            + "'. Valid modes: ApplyBefore, ApplyAfter"
        );
        return contract::InfoMode::NONE;
    }

    // Parse ContractSpec. Contracts are static-only; the legacy
    // type: "RUNTIME" encoding has been merged into patches, and
    // type: is quietly ignored if still present so older YAMLs keep
    // parsing. Any contract carrying code_file / function_name is
    // rejected with a pointer to patches:.
    template<>
    struct MappingTraits< contract::ContractSpec >
    {
        static void mapping(IO &io, contract::ContractSpec &spec) {
            io.mapRequired("name", spec.name);
            io.mapOptional("description", spec.description);
            io.mapOptional("category", spec.category);
            io.mapOptional("severity", spec.severity);

            std::string type_str;
            io.mapOptional("type", type_str);
            if (type_str == "RUNTIME") {
                io.setError(
                    "Runtime contracts are no longer supported — migrate '"
                    + spec.name + "' to a patch entry under patches: (same "
                    "code_file / function_name, applied via apply_before / "
                    "apply_after / apply_at_entrypoint)."
                );
                return;
            }
            if (!type_str.empty() && type_str != "STATIC") {
                io.setError(
                    "Unknown contract type: '" + type_str + "'. The type: "
                    "field is deprecated; contracts are static-only. Remove "
                    "it or set it to \"STATIC\"."
                );
                return;
            }

            // Reject the runtime-only fields so stale YAMLs produce a clear
            // error rather than being silently dropped.
            std::string code_file;
            std::string function_name;
            io.mapOptional("code_file", code_file);
            io.mapOptional("function_name", function_name);
            if (!code_file.empty() || !function_name.empty()) {
                io.setError(
                    "Contract '" + spec.name + "' carries a code_file / "
                    "function_name — runtime contracts have been merged "
                    "into patches. Move the entry under patches:."
                );
                return;
            }

            // Consume `parameters:` if present to keep older library YAMLs
            // parsing cleanly — it's reference-only metadata.
            {
                [[maybe_unused]] std::vector< Parameter > unused_parameters;
                io.mapOptional("parameters", unused_parameters);
            }

            io.mapOptional("preconditions", spec.preconditions);
            io.mapOptional("postconditions", spec.postconditions);
        }
    };

    // Parse Action
    template<>
    struct MappingTraits< contract::Action >
    {
        static void mapping(IO &io, contract::Action &action) {
            io.mapRequired("contract_id", action.contract_id);
            io.mapOptional("description", action.description);
            io.mapOptional("arguments", action.arguments);

            std::string mode_str;
            io.mapRequired("mode", mode_str);
            action.mode = parseContractInfoMode(io, mode_str);

            // Static contracts materialize MLIR attributes; they do not emit
            // a call, so `arguments:` never reaches a consumer. Warn so
            // spec authors migrating from the old runtime-contract shape
            // realize the list is being dropped.
            if (!action.arguments.empty()) {
                LOG(WARNING)
                    << "contract action '" << action.contract_id
                    << "': 'arguments:' is ignored for static contracts. "
                       "Express runtime-style invocation as a patch under "
                       "patches: instead.\n";
            }
        }
    };

    // Parse MatchConfig
    template<>
    struct MappingTraits< contract::MatchConfig >
    {
        static void mapping(IO &io, contract::MatchConfig &match) {
            io.mapRequired("name", match.name);
            io.mapRequired("function_context", match.function_context);
            io.mapOptional("argument_matches", match.argument_matches);
            io.mapOptional("variable_matches", match.variable_matches);
            io.mapOptional("symbol_matches", match.symbol_matches);

            std::string kind_str;
            io.mapRequired("kind", kind_str);
            if (kind_str == "function") {
                match.kind = contract::MatchKind::FUNCTION;
            } else { // Default to NONE
                match.kind = contract::MatchKind::NONE;
            }
        }
    };

    // Parse ContractAction
    template<>
    struct MappingTraits< contract::ContractAction >
    {
        static void mapping(IO &io, contract::ContractAction &contract_action) {
            io.mapRequired("id", contract_action.action_id);
            io.mapOptional("description", contract_action.description);

            // required block doesn't mean the block must be populated, so check.
            io.mapRequired("match", contract_action.match);
            if (contract_action.match.empty()) {
                io.setError(
                    "ContractAction '" + contract_action.action_id
                    + "' must include at least one 'match' entry."
                );
            }

            io.mapRequired("action", contract_action.action);
            if (contract_action.action.empty()) {
                io.setError(
                    "ContractAction '" + contract_action.action_id
                    + "' must include at least one 'action' entry."
                );
            }
        }
    };

    // Parse StaticContractRequirement
    template<>
    struct MappingTraits< contract::StaticContractRequirement >
    {
        static void mapping(IO &io, contract::StaticContractRequirement &requirement) {
            io.mapOptional("id", requirement.id);
            io.mapOptional("description", requirement.description);
            io.mapOptional("pred", requirement.pred);
        }
    };

    // Parse MetaContractConfig. Contracts are static-only, so `optimization:`
    // (e.g. `inline-contracts`) is accepted-but-ignored for backward compat
    // with older YAMLs; a warning is emitted when flags are present.
    template<>
    struct MappingTraits< contract::MetaContractConfig >
    {
        static void mapping(IO &io, contract::MetaContractConfig &meta_contract) {
            io.mapRequired("name", meta_contract.name);
            io.mapOptional("description", meta_contract.description);
            io.mapRequired("contract_actions", meta_contract.contract_actions);

            std::vector< std::string > optimization;
            io.mapOptional("optimization", optimization);
            if (!optimization.empty()) {
                LOG(WARNING)
                    << "meta_contract '" << meta_contract.name
                    << "': 'optimization:' is deprecated for contracts "
                       "(contracts are static-only, nothing to inline) — ignoring.\n";
            }
        }
    };

    // Parse Range
    template<>
    struct MappingTraits< contract::Range >
    {
        static void mapping(IO &io, contract::Range &range) {
            io.mapRequired("min", range.min);
            io.mapRequired("max", range.max);
        }
    };

    // Parse Value
    template<>
    struct MappingTraits< contract::Value >
    {
        static void mapping(IO &io, contract::Value &value) {
            io.mapOptional("constant", value.constant);
            io.mapOptional("symbol", value.symbol);
        }
    };

    // Parse Predicate
    template<>
    struct MappingTraits< contract::Predicate >
    {
        static void mapping(IO &io, contract::Predicate &pred) {
            // Parse kind
            std::string kind_str;
            io.mapRequired("kind", kind_str);
            if (kind_str == "nonnull") {
                pred.kind = ::contracts::PredicateKind::nonnull;
            } else if (kind_str == "relation") {
                pred.kind = ::contracts::PredicateKind::relation;
            } else if (kind_str == "alignment") {
                pred.kind = ::contracts::PredicateKind::alignment;
            } else if (kind_str == "expr") {
                pred.kind = ::contracts::PredicateKind::expr;
            } else if (kind_str == "range") {
                pred.kind = ::contracts::PredicateKind::range;
            } else {
                io.setError("Unsupported predicate kind: " + kind_str);
                return;
            }

            // Parse target - supports formats like "arg0", "arg1", "return_value", "symbol"
            if (pred.kind != ::contracts::PredicateKind::expr) {
                std::string target_str;
                io.mapRequired("target", target_str);
                if (target_str == "return_value") {
                    pred.target = ::contracts::TargetKind::ReturnValue;
                } else if (target_str.starts_with("arg")) {
                    pred.target = ::contracts::TargetKind::Arg;
                    // Extract the index from "arg0", "arg1", etc.
                    auto result =
                        ::patchestry::passes::contract::ParseUint64(target_str.substr(3));
                    if (result.value) {
                        pred.arg_index = *result.value;
                    } else {
                        io.setError("Invalid argument index in target: " + target_str
                                    + " (" + result.error + ")");
                        return;
                    }
                } else if (target_str == "symbol") {
                    pred.target = ::contracts::TargetKind::Symbol;
                } else {
                    io.setError("Unsupported target: " + target_str);
                    return;
                }
            }

            // Parse relation
            std::string relation_str;
            io.mapOptional("relation", relation_str);
            if (relation_str == "eq" || relation_str == "==") {
                pred.relation = ::contracts::RelationKind::eq;
            } else if (relation_str == "neq" || relation_str == "ne" || relation_str == "!=") {
                pred.relation = ::contracts::RelationKind::neq;
            } else if (relation_str == "lt" || relation_str == "<") {
                pred.relation = ::contracts::RelationKind::lt;
            } else if (relation_str == "lte" || relation_str == "le" || relation_str == "<=") {
                pred.relation = ::contracts::RelationKind::lte;
            } else if (relation_str == "gt" || relation_str == ">") {
                pred.relation = ::contracts::RelationKind::gt;
            } else if (relation_str == "gte" || relation_str == "ge" || relation_str == ">=") {
                pred.relation = ::contracts::RelationKind::gte;
            } else {
                pred.relation = ::contracts::RelationKind::none;
            }

            // Parse optional fields
            io.mapOptional("value", pred.value);
            io.mapOptional("symbol", pred.symbol);

            // Parse align as a string, convert to uint64_t
            std::string align_str;
            io.mapOptional("align", align_str);
            if (!align_str.empty()) {
                auto align_result = ::patchestry::passes::contract::ParseUint64(align_str);
                if (align_result.value) {
                    pred.align = *align_result.value;
                } else {
                    io.setError("Invalid alignment value: " + align_str
                                + " (" + align_result.error + ")");
                    return;
                }
            }

            io.mapOptional("expr", pred.expr);
            io.mapOptional("range", pred.range);
        }
    };

    // Parse ArgumentSource
    template<>
    struct MappingTraits< contract::ArgumentSource >
    {
        static void mapping(IO &io, contract::ArgumentSource &arg) {
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
                // Captures are bound at match time and consumed by patch calls.
                // Contracts materialize MLIR attributes and don't run at match
                // time, so nothing ever binds a capture for them — parsing
                // this source used to succeed and then silently drop at emit
                // time. Reject up front with a pointer to patches:.
                io.setError(
                    "'source: capture' is not supported for contracts "
                    "(contracts attach MLIR attributes and have no bound "
                    "match-site values). Use source: operand/variable/"
                    "symbol/constant/return_value, or move the entry "
                    "under patches: if you need capture-driven arguments."
                );
                return;
            } else {
                io.setError("Unknown argument source type: '" + source_str + "'");
                return;
            }

            io.mapOptional("name", arg.name);
            io.mapOptional("index", arg.index);
            io.mapOptional("symbol", arg.symbol);
            io.mapOptional("value", arg.value);
        }
    };

    template<>
    struct MappingTraits< contract::ContractMatchObject >
    {
        static void mapping(IO &io, contract::ContractMatchObject &m) {
            io.mapRequired("name", m.name);
            io.mapOptional("context", m.context);
        }
    };

    template<>
    struct MappingTraits< contract::ContractActionObject >
    {
        static void mapping(IO &io, contract::ContractActionObject &a) {
            std::string mode_str;
            io.mapRequired("mode", mode_str);
            a.mode = parseContractInfoMode(io, mode_str);

            io.mapRequired("contract", a.contract_id);
            io.mapOptional("arguments", a.arguments);

            // Accept-and-warn for backward compat (see MetaContractConfig).
            // The PatchEntry-facing trait doesn't have access to the entry
            // name here, so the warning is emitted there after projection.
            std::vector< std::string > optimization;
            io.mapOptional("optimization", optimization);
            a.optimization_warned = !optimization.empty();
        }
    };

    // Parse ContractEntry — simplified YAML surface for contracts.
    template<>
    struct MappingTraits< contract::ContractEntry >
    {
        static void mapping(IO &io, contract::ContractEntry &entry) {
            io.mapRequired("name", entry.name);
            io.mapOptional("id", entry.id);
            io.mapOptional("description", entry.description);

            // Parse nested match object
            contract::ContractMatchObject match_obj;
            io.mapRequired("match", match_obj);
            entry.match_name = match_obj.name;
            entry.context    = match_obj.context;

            // Parse nested action object — projected onto the flat
            // ContractEntry fields so inflateContractEntry keeps working
            // unchanged.
            contract::ContractActionObject action_obj;
            io.mapRequired("action", action_obj);
            entry.mode        = action_obj.mode;
            entry.contract_id = action_obj.contract_id;
            entry.arguments   = action_obj.arguments;

            // Static-only contracts don't consume `arguments:` at emit time;
            // warn so migrating authors realize the list is being dropped.
            // (inflateContractEntry copies these straight to Action.arguments,
            // bypassing the Action trait that would otherwise warn.)
            if (!entry.arguments.empty()) {
                LOG(WARNING)
                    << "contract '" << entry.name
                    << "': 'arguments:' is ignored for static contracts. "
                       "Express runtime-style invocation as a patch under "
                       "patches: instead.\n";
            }

            if (action_obj.optimization_warned) {
                LOG(WARNING)
                    << "contract '" << entry.name
                    << "': 'optimization:' is deprecated for contracts "
                       "(contracts are static-only, nothing to inline) — ignoring.\n";
            }
        }
    };

} // namespace llvm::yaml
