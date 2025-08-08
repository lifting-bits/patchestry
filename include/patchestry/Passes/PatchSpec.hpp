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
#include <patchestry/YAML/YAMLParser.hpp>

namespace patchestry::passes {

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

    struct Metadata
    {
        std::string name;
        std::string description;
        std::string version;
        std::string author;
        std::string created;
        std::string organization;
    };

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

    struct ContractAction
    {
        std::string action_id;
        std::string description;
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

    struct ContractSpec
    {
        std::string name;
        std::string id;
        std::string description;
    };

    struct PatchLibrary
    {
        std::string api_version;
        Metadata metadata;
        std::vector< PatchSpec > patches;
    };

    struct ContractLibrary
    {
        std::string api_version;
        Metadata metadata;
        std::vector< ContractSpec > contracts;
    };

    struct Libraries
    {
        PatchLibrary patches;
        ContractLibrary contracts;
    };

    struct Target
    {
        std::string binary;
        std::string arch;
    };

    struct MetaPatchConfig
    {
        std::string name;
        std::string id;
        std::string description;
        std::set< std::string > optimization;
        std::vector< PatchAction > patch_actions;
    };

    struct MetaContractConfig
    {
        std::string name;
        std::string id;
        std::string description;
    };

    struct PatchConfiguration
    {
        std::string api_version;
        Metadata metadata;
        Target target;
        Libraries libraries;
        std::vector< std::string > execution_order;
        std::vector< MetaPatchConfig > meta_patches;
        std::vector< MetaContractConfig > meta_contracts;
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

} // namespace patchestry::passes

namespace patchestry::yaml {
    namespace utils {

        [[maybe_unused]] static std::optional< passes::PatchConfiguration >
        loadPatchConfiguration(const std::string &file_path) {
            YAMLParser parser;
            auto result = parser.parse_from_file< passes::PatchConfiguration >(file_path);

            if (!result) {
                LOG(ERROR) << "Failed to load patch configuration: " << file_path << "\n";
                return std::nullopt;
            }

            return result;
        }

        [[maybe_unused]] static std::optional< passes::PatchLibrary >
        loadPatchLibrary(const std::string &file_path) {
            YAMLParser parser;
            auto result = parser.parse_from_file< passes::PatchLibrary >(file_path);
            if (!result) {
                LOG(ERROR) << "Failed to load patch library: " << file_path << "\n";
                return std::nullopt;
            }
            return result;
        }

        [[maybe_unused]] static std::optional< passes::ContractLibrary >
        loadContractLibrary(const std::string &file_path) {
            YAMLParser parser;
            auto result = parser.parse_from_file< passes::ContractLibrary >(file_path);
            if (!result) {
                LOG(ERROR) << "Failed to load contract library: " << file_path << "\n";
                return std::nullopt;
            }
            return result;
        }

        [[maybe_unused]] static bool savePatchConfiguration(
            const passes::PatchConfiguration &config, const std::string &file_path
        ) {
            YAMLParser parser;
            std::string yaml_content = parser.serialize_to_string(config);

            if (yaml_content.empty()) {
                LOG(ERROR) << "Failed to serialize patch configuration";
                return false;
            }

            std::ofstream file(file_path);
            if (!file.is_open()) {
                LOG(ERROR) << "Failed to open file for writing: " << file_path;
                return false;
            }

            file << yaml_content;
            file.close();

            if (file.fail()) {
                LOG(ERROR) << "Failed to write to file: " << file_path;
                return false;
            }

            return true;
        }

        [[maybe_unused]] static bool
        validatePatchConfiguration(const passes::PatchConfiguration &config) {
            if (config.libraries.patches.patches.empty()) {
                LOG(WARNING) << "Patch configuration has no patches";
            }

            // Validate each patch
            for (const auto &patch : config.libraries.patches.patches) {
                if (patch.name.empty()) {
                    LOG(ERROR) << "Patch specification missing name";
                    return false;
                }

                // Validate patch file exists if specified
                if (!patch.implementation.code_file.empty()) {
                    if (!llvm::sys::fs::exists(patch.implementation.code_file)) {
                        LOG(ERROR)
                            << "Patch file does not exist: " << patch.implementation.code_file;
                        return false;
                    }
                }
            }

            return true;
        }

    } // namespace utils

} // namespace patchestry::yaml

LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::PatchSpec)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::VariableMatch)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::ArgumentMatch)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::ArgumentSource)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::MatchConfig)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::Parameter)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::Action)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::PatchAction)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::ContractAction)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::ContractSpec)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::MetaPatchConfig)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::MetaContractConfig)

class PatchSpecContext
{
  public:
    static PatchSpecContext &getInstance() {
        static PatchSpecContext instance;
        return instance;
    }

    void set_spec_path(const std::string &file) {
        auto directory = llvm::sys::path::parent_path(file);
        if (!directory.empty()) {
            spec_path = directory.str();
        }
    }

    std::string resolve_path(const std::string &file) {
        if (llvm::sys::path::is_absolute(file)) {
            return file;
        }

        llvm::SmallVector< char > directory;
        directory.assign(spec_path.begin(), spec_path.end());
        llvm::sys::path::append(directory, file);
        llvm::sys::path::remove_dots(directory);
        return std::string(directory.data(), directory.size());
    }

  private:
    std::string spec_path;
};

namespace llvm::yaml {
    // Parse ArgumentSource
    template<>
    struct MappingTraits< patchestry::passes::ArgumentSource >
    {
        static void mapping(IO &io, patchestry::passes::ArgumentSource &arg) {
            std::string source_str;
            io.mapRequired("source", source_str);

            if (source_str == "operand") {
                arg.source = patchestry::passes::ArgumentSourceType::OPERAND;
            } else if (source_str == "argument") {
                arg.source = patchestry::passes::ArgumentSourceType::OPERAND; // Treat argument
                                                                              // same as operand
            } else if (source_str == "variable") {
                arg.source = patchestry::passes::ArgumentSourceType::VARIABLE;
            } else if (source_str == "symbol") {
                arg.source = patchestry::passes::ArgumentSourceType::SYMBOL;
            } else if (source_str == "constant") {
                arg.source = patchestry::passes::ArgumentSourceType::CONSTANT;
            } else if (source_str == "return_value") {
                arg.source = patchestry::passes::ArgumentSourceType::RETURN_VALUE;
            }

            io.mapRequired("name", arg.name);
            io.mapOptional("index", arg.index);
            io.mapOptional("symbol", arg.symbol);
            io.mapOptional("value", arg.value);
        }
    };

    // Prase ArgumentMatch
    template<>
    struct MappingTraits< patchestry::passes::ArgumentMatch >
    {
        static void mapping(IO &io, patchestry::passes::ArgumentMatch &arg) {
            io.mapRequired("index", arg.index);
            io.mapRequired("name", arg.name);
            io.mapOptional("type", arg.type);
        }
    };

    // Prase VariableMatch
    template<>
    struct MappingTraits< patchestry::passes::VariableMatch >
    {
        static void mapping(IO &io, patchestry::passes::VariableMatch &var) {
            io.mapRequired("name", var.name);
            io.mapOptional("type", var.type);
        }
    };

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

    // Parse Parameter
    template<>
    struct MappingTraits< patchestry::passes::Parameter >
    {
        static void mapping(IO &io, patchestry::passes::Parameter &param) {
            io.mapOptional("name", param.name);
            io.mapOptional("type", param.type);
            io.mapOptional("description", param.description);
        }
    };

    // Parse Implementation
    template<>
    struct MappingTraits< patchestry::passes::Implementation >
    {
        static void mapping(IO &io, patchestry::passes::Implementation &impl) {
            io.mapOptional("language", impl.language);
            io.mapOptional("code_file", impl.code_file);
            io.mapOptional("function_name", impl.function_name);
            io.mapOptional("parameters", impl.parameters);
            io.mapOptional("dependencies", impl.dependencies);
        }
    };

    // Parse PatchSpec
    template<>
    struct MappingTraits< patchestry::passes::PatchSpec >
    {
        static void mapping(IO &io, patchestry::passes::PatchSpec &spec) {
            io.mapRequired("name", spec.name);
            io.mapOptional("id", spec.id);
            io.mapOptional("description", spec.description);
            io.mapOptional("category", spec.category);
            io.mapOptional("severity", spec.severity);
            io.mapOptional("implementation", spec.implementation);
        }
    };

    // Parse ContractSpec
    template<>
    struct MappingTraits< patchestry::passes::ContractSpec >
    {
        static void mapping(IO &io, patchestry::passes::ContractSpec &spec) {
            io.mapRequired("name", spec.name);
            io.mapOptional("id", spec.id);
            io.mapOptional("description", spec.description);
        }
    };

    // Parse Action
    template<>
    struct MappingTraits< patchestry::passes::Action >
    {
        static void mapping(IO &io, patchestry::passes::Action &action) {
            io.mapOptional("patch_id", action.patch_id);
            io.mapOptional("description", action.description);
            io.mapOptional("arguments", action.arguments);

            std::string mode_str;
            io.mapRequired("mode", mode_str);
            if (mode_str == "ApplyBefore" || mode_str == "apply_before") {
                action.mode = patchestry::passes::PatchInfoMode::APPLY_BEFORE;
            } else if (mode_str == "ApplyAfter" || mode_str == "apply_after") {
                action.mode = patchestry::passes::PatchInfoMode::APPLY_AFTER;
            } else if (mode_str == "Replace" || mode_str == "replace") {
                action.mode = patchestry::passes::PatchInfoMode::REPLACE;
            } else {
                action.mode = patchestry::passes::PatchInfoMode::NONE;
            }
        }
    };

    // Parse PatchMatch
    template<>
    struct MappingTraits< patchestry::passes::MatchConfig >
    {
        static void mapping(IO &io, patchestry::passes::MatchConfig &match) {
            io.mapOptional("name", match.name);
            io.mapOptional("function_context", match.function_context);
            io.mapOptional("argument_matches", match.argument_matches);
            io.mapOptional("variable_matches", match.variable_matches);
            io.mapOptional("symbol_matches", match.symbol_matches);
            io.mapOptional("operand_matches", match.operand_matches);

            std::string kind_str;
            io.mapRequired("kind", kind_str);
            if (kind_str == "operation") {
                match.kind = patchestry::passes::MatchKind::OPERATION;
            } else if (kind_str == "function") {
                match.kind = patchestry::passes::MatchKind::FUNCTION;
            } else { // Default to NONE
                match.kind = patchestry::passes::MatchKind::NONE;
            }
        }
    };

    // Parse ContractAction
    template<>
    struct MappingTraits< patchestry::passes::ContractAction >
    {
        static void mapping(IO &io, patchestry::passes::ContractAction &contract_action) {
            io.mapOptional("id", contract_action.action_id);
            io.mapOptional("description", contract_action.description);
        }
    };

    // Parse PatchAction
    template<>
    struct MappingTraits< patchestry::passes::PatchAction >
    {
        static void mapping(IO &io, patchestry::passes::PatchAction &patch_action) {
            io.mapOptional("id", patch_action.action_id);
            io.mapOptional("description", patch_action.description);
            io.mapOptional("match", patch_action.match);
            io.mapOptional("action", patch_action.action);
        }
    };

    // Parse Target
    template<>
    struct MappingTraits< patchestry::passes::Target >
    {
        static void mapping(IO &io, patchestry::passes::Target &target) {
            io.mapOptional("binary", target.binary);
            io.mapRequired("arch", target.arch);
        }
    };

    // Parse PatchLibrary
    template<>
    struct MappingTraits< patchestry::passes::PatchLibrary >
    {
        static void mapping(IO &io, patchestry::passes::PatchLibrary &library) {
            io.mapOptional("apiVersion", library.api_version);
            io.mapOptional("metadata", library.metadata);
            io.mapOptional("patches", library.patches);
        }
    };

    // Parse ContractLibrary
    template<>
    struct MappingTraits< patchestry::passes::ContractLibrary >
    {
        static void mapping(IO &io, patchestry::passes::ContractLibrary &library) {
            io.mapOptional("apiVersion", library.api_version);
            io.mapOptional("metadata", library.metadata);
            io.mapOptional("contracts", library.contracts);
        }
    };

    // Parse Libraries
    template<>
    struct MappingTraits< patchestry::passes::Libraries >
    {
        static void mapping(IO &io, patchestry::passes::Libraries &libraries) {
            // recursively parse libraries yaml files
            std::string patches_file;
            std::string contracts_file;
            io.mapOptional("patches", patches_file);
            io.mapOptional("contracts", contracts_file);
            if (!patches_file.empty()) {
                auto patches_config = patchestry::yaml::utils::loadPatchLibrary(patches_file);
                if (!patches_config) {
                    LOG(ERROR) << "Failed to load patch library: " << patches_file << "\n";
                    return;
                }
                libraries.patches = patches_config.value();
            }
            if (!contracts_file.empty()) {
                auto contracts_config =
                    patchestry::yaml::utils::loadContractLibrary(contracts_file);
                if (!contracts_config) {
                    LOG(ERROR) << "Failed to load contract library: " << contracts_file << "\n";
                    return;
                }
                libraries.contracts = contracts_config.value();
            }
        }
    };

    // Parse MetaPatchConfig
    template<>
    struct MappingTraits< patchestry::passes::MetaPatchConfig >
    {
        static void mapping(IO &io, patchestry::passes::MetaPatchConfig &meta_patch) {
            io.mapOptional("name", meta_patch.name);
            io.mapOptional("id", meta_patch.id);
            io.mapOptional("description", meta_patch.description);

            std::vector< std::string > optimization;
            io.mapOptional("optimization", optimization);
            for (const auto &opt : optimization) {
                meta_patch.optimization.insert(opt);
            }
            io.mapOptional("patch_actions", meta_patch.patch_actions);
        }
    };

    // Parse MetaContractConfig
    template<>
    struct MappingTraits< patchestry::passes::MetaContractConfig >
    {
        static void mapping(IO &io, patchestry::passes::MetaContractConfig &meta_contract) {
            io.mapOptional("name", meta_contract.name);
            io.mapOptional("id", meta_contract.id);
            io.mapOptional("description", meta_contract.description);
        }
    };

    // Parse PatchConfiguration
    template<>
    struct MappingTraits< patchestry::passes::PatchConfiguration >
    {
        static void mapping(IO &io, patchestry::passes::PatchConfiguration &config) {
            io.mapOptional("apiVersion", config.api_version);
            io.mapOptional("metadata", config.metadata);
            io.mapOptional("target", config.target);
            io.mapOptional("libraries", config.libraries);
            io.mapOptional("execution_order", config.execution_order);
            io.mapOptional("meta_patches", config.meta_patches);
            io.mapOptional("meta_contracts", config.meta_contracts);
        }
    };

} // namespace llvm::yaml
