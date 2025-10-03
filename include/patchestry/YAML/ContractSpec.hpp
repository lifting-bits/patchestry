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

#include <patchestry/Dialect/Contracts/ContractsDialect.hpp>
#include <patchestry/Util/Log.hpp>
#include <patchestry/YAML/BaseSpec.hpp>
#include <patchestry/YAML/YAMLParser.hpp>

namespace patchestry::passes {
    namespace contract {
        enum class InfoMode : uint8_t {
            NONE = 0, // No contract
            APPLY_BEFORE,
            APPLY_AFTER,
            APPLY_AT_ENTRYPOINT
        };

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

        // Forward declarations from contract dialect
        namespace attr_types {
            using PredicateKind = ::contracts::PredicateKind;
            using TargetKind = ::contracts::TargetKind;
            using RelationKind = ::contracts::RelationKind;
        }

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
            std::optional< std::string > value;      // For constant values (to be parsed as integers)
            std::optional< std::string > symbol;     // Symbol name
            std::optional< uint64_t > align;         // Alignment value
            std::optional< std::string > expr;       // Expression string
            std::optional< Range > range;            // Range constraint
        };

        struct StaticContractRequirement
        {
            std::string id;
            std::optional< std::string > description;
            std::optional< Predicate > pred;
        };

        struct ContractSpec
        {
            std::string name;
            std::string id;
            std::string description;
            std::string category;
            std::string severity;
            ContractType type; // "STATIC" or "RUNTIME"
            std::optional< std::vector< StaticContractRequirement > >
                preconditions; // For static contracts
            std::optional< std::vector< StaticContractRequirement > >
                postconditions;                             // For static contracts
            std::optional< Implementation > implementation; // For runtime contracts
            std::optional< std::string > contract_module;   // For runtime contracts
        };

        struct MetaContractConfig
        {
            std::string name;
            std::string id;
            std::string description;
            std::vector< ContractAction > contract_actions;
            std::set< std::string > optimization;
        };

        [[maybe_unused]] inline std::string_view infoModeToString(InfoMode mode) {
            switch (mode) {
                case InfoMode::NONE:
                    return "NONE";
                case InfoMode::APPLY_BEFORE:
                    return "APPLY_BEFORE";
                case InfoMode::APPLY_AFTER:
                    return "APPLY_AFTER";
                case InfoMode::APPLY_AT_ENTRYPOINT:
                    return "APPLY_AT_ENTRYPOINT";
            }
            return "UNKNOWN";
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

namespace llvm::yaml {
    using namespace patchestry::passes;

    // Parse ContractSpec
    template<>
    struct MappingTraits< contract::ContractSpec >
    {
        static void mapping(IO &io, contract::ContractSpec &spec) {
            io.mapRequired("name", spec.name);
            io.mapRequired("id", spec.id);
            io.mapOptional("description", spec.description);
            io.mapOptional("category", spec.category);
            io.mapOptional("severity", spec.severity);

            std::string type_str;
            io.mapRequired("type", type_str);
            if (type_str == "STATIC") {
                spec.type = ContractType::STATIC;
            } else if (type_str == "RUNTIME") {
                spec.type = ContractType::RUNTIME;
            }
            if (spec.type == ContractType::STATIC) {
                io.mapOptional("preconditions", spec.preconditions);
                io.mapOptional("postconditions", spec.postconditions);
            } else if (spec.type == ContractType::RUNTIME) {
                io.mapOptional("implementation", spec.implementation);
            }
            io.mapOptional("contract_module", spec.contract_module);
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
            if (mode_str == "ApplyBefore" || mode_str == "apply_before") {
                action.mode = contract::InfoMode::APPLY_BEFORE;
            } else if (mode_str == "ApplyAfter" || mode_str == "apply_after") {
                action.mode = contract::InfoMode::APPLY_AFTER;
            } else if (mode_str == "ApplyAtEntrypoint" || mode_str == "apply_at_entrypoint") {
                action.mode = contract::InfoMode::APPLY_AT_ENTRYPOINT;
            } else {
                action.mode = contract::InfoMode::NONE;
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

    // Parse StaticContractSpec
    template<>
    struct MappingTraits< contract::StaticContractSpec >
    {
        static void mapping(IO &io, contract::StaticContractSpec &static_contract) {
            io.mapRequired("requires", static_contract.conditions_required);
            if (static_contract.conditions_required.empty()) {
                io.setError("StaticContractSpec must include at least one 'requires' entry.");
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

    // Parse MetaContractConfig
    template<>
    struct MappingTraits< contract::MetaContractConfig >
    {
        static void mapping(IO &io, contract::MetaContractConfig &meta_contract) {
            io.mapRequired("name", meta_contract.name);
            io.mapOptional("id", meta_contract.id);
            io.mapOptional("description", meta_contract.description);
            io.mapRequired("contract_actions", meta_contract.contract_actions);

            std::vector< std::string > optimization;
            io.mapOptional("optimization", optimization);
            for (const auto &opt : optimization) {
                meta_contract.optimization.insert(opt);
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
            std::string target_str;
            io.mapRequired("target", target_str);
            if (target_str == "return_value") {
                pred.target = ::contracts::TargetKind::ReturnValue;
            } else if (target_str.starts_with("arg")) {
                pred.target = ::contracts::TargetKind::Arg;
                // Extract the index from "arg0", "arg1", etc.
                try {
                    pred.arg_index = std::stoull(target_str.substr(3));
                } catch (...) {
                    io.setError("Invalid argument index in target: " + target_str);
                    return;
                }
            } else if (target_str == "symbol") {
                pred.target = ::contracts::TargetKind::Symbol;
            } else {
                io.setError("Unsupported target: " + target_str);
                return;
            }

            // Parse relation
            std::string relation_str;
            io.mapRequired("relation", relation_str);
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
            } else if (relation_str == "none") {
                pred.relation = ::contracts::RelationKind::none;
            } else {
                io.setError("Unsupported relation: " + relation_str);
                return;
            }

            // Parse optional fields
            io.mapOptional("value", pred.value);
            io.mapOptional("symbol", pred.symbol);

            // Parse align as a string, convert to uint64_t
            std::string align_str;
            io.mapOptional("align", align_str);
            if (!align_str.empty()) {
                try {
                    pred.align = std::stoull(align_str);
                } catch (...) {
                    io.setError("Invalid alignment value: " + align_str);
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
            }

            io.mapRequired("name", arg.name);
            io.mapOptional("index", arg.index);
            io.mapOptional("symbol", arg.symbol);
            io.mapOptional("value", arg.value);
        }
    };
} // namespace llvm::yaml
