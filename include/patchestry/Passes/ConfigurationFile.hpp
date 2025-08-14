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

#include <patchestry/Passes/ContractSpec.hpp>
#include <patchestry/Passes/PatchSpec.hpp>
#include <patchestry/Util/Log.hpp>
#include <patchestry/YAML/YAMLParser.hpp>

class ConfigurationFile {
  public:
    static ConfigurationFile &getInstance() {
        static ConfigurationFile instance;
        return instance;
    }

    void set_file_path(const std::string &file) {
        auto directory = llvm::sys::path::parent_path(file);
        if (!directory.empty()) {
            file_path = directory.str();
        }
    }

    std::string resolve_path(const std::string &file) {
        if (llvm::sys::path::is_absolute(file)) {
            return file;
        }

        llvm::SmallVector< char > directory;
        directory.assign(file_path.begin(), file_path.end());
        llvm::sys::path::append(directory, file);
        llvm::sys::path::remove_dots(directory);
        return std::string(directory.data(), directory.size());
    }

  private:
    std::string file_path;
};

namespace patchestry::passes {
    struct Target {
        std::string binary;
        std::string arch;
    };

    struct Libraries {
        patch::PatchLibrary patches;
        contract::ContractLibrary contracts;
    };

    struct Metadata {
        std::string name;
        std::string description;
        std::string version;
        std::string author;
        std::string created;
        std::string organization;
    };

    struct Configuration {
        std::string api_version;
        Metadata metadata;
        Target target;
        Libraries libraries;
        std::vector< std::string > execution_order;
        std::vector< patch::MetaPatchConfig > meta_patches;
        std::vector< contract::MetaContractConfig > meta_contracts;
    };
} // namespace patchestry::passes

namespace patchestry::yaml {
    using namespace patchestry::passes;

    namespace utils {
        [[maybe_unused]] static std::optional< Configuration >
        loadConfiguration(const std::string &file_path) {
            YAMLParser parser;
            auto result = parser.parse_from_file< Configuration >(file_path);

            if (!result) {
                LOG(ERROR) << "Failed to load patch configuration: " << file_path << "\n";
                return std::nullopt;
            }

            return result;
        }

        [[maybe_unused]] static bool saveConfiguration(
            const Configuration &config, const std::string &file_path
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
        validateConfiguration(const Configuration &config) {
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

namespace llvm::yaml {
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

    // Parse Target
    template<>
    struct MappingTraits< patchestry::passes::Target >
    {
        static void mapping(IO &io, patchestry::passes::Target &target) {
            io.mapOptional("binary", target.binary);
            io.mapRequired("arch", target.arch);
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

    // Parse Configuration
    template<>
    struct MappingTraits< patchestry::passes::Configuration >
    {
        static void mapping(IO &io, patchestry::passes::Configuration &config) {
            io.mapOptional("apiVersion", config.api_version);
            io.mapOptional("metadata", config.metadata);
            io.mapOptional("target", config.target);
            io.mapOptional("libraries", config.libraries);
            io.mapOptional("execution_order", config.execution_order);
            io.mapOptional("meta_patches", config.meta_patches);
            io.mapOptional("meta_contracts", config.meta_contracts);
        }
    };
}