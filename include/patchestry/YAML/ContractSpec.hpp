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

        struct ContractSpec
        {
            std::string name;
            std::string id;
            std::string description;
            std::string category;
            std::string severity;
            Implementation implementation;
            std::optional< std::string > contract_module;
        };

        struct ContractLibrary
        {
            std::string api_version;
            Metadata metadata;
            std::vector< ContractSpec > contracts;
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

namespace patchestry::yaml {
    using namespace patchestry::passes;

    namespace utils {
        [[maybe_unused]] static std::optional< contract::ContractLibrary >
        loadContractLibrary(const std::string &file_path) {
            YAMLParser parser;
            auto result = parser.parse_from_file< contract::ContractLibrary >(file_path);
            if (!result) {
                LOG(ERROR) << "Failed to load contract library: " << file_path << "\n";
                return std::nullopt;
            }
            return result;
        }

    } // namespace utils
} // namespace patchestry::yaml

LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::ArgumentSource)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::MatchConfig)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::Action)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::ContractAction)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::ContractSpec)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::contract::MetaContractConfig)

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
            io.mapRequired("implementation", spec.implementation);
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

    // Parse ContractLibrary
    template<>
    struct MappingTraits< contract::ContractLibrary >
    {
        static void mapping(IO &io, contract::ContractLibrary &library) {
            io.mapOptional("apiVersion", library.api_version);
            io.mapOptional("metadata", library.metadata);

            // if the contracts: block is included, there must be at least one contract
            io.mapRequired("contracts", library.contracts);
            if (library.contracts.empty()) {
                io.setError(
                    "ContractLibrary '" + library.metadata.name
                    + "' must include at least one 'contracts' entry."
                );
            }
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
        }
    };
} // namespace llvm::yaml
