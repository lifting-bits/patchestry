/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <fstream>
#include <string>
#include <vector>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/YAMLTraits.h>

#include <patchestry/Util/Log.hpp>
#include <patchestry/YAML/BaseSpec.hpp>
#include <patchestry/YAML/ContractSpec.hpp>
#include <patchestry/YAML/PatchSpec.hpp>
#include <patchestry/YAML/YAMLParser.hpp>

class ConfigurationFile
{
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
    struct Target
    {
        std::string binary;
        std::string arch;
    };

    struct Library
    {
        std::string api_version;
        Metadata metadata;
        std::vector< patch::PatchSpec > patches;
        std::vector< contract::ContractSpec > contracts;
    };

    struct Configuration
    {
        std::string api_version;
        Metadata metadata;
        Target target;
        Library libraries;
        std::vector< std::string > execution_order;
        std::vector< patch::MetaPatchConfig > meta_patches;
        std::vector< contract::MetaContractConfig > meta_contracts;
    };
} // namespace patchestry::passes

namespace patchestry::yaml {
    using namespace patchestry::passes;

    namespace utils {

        [[maybe_unused]] static std::optional< Library >
        loadLibrary(const std::string &file_path) {
            YAMLParser parser;
            auto result = parser.parse_from_file< Library >(file_path);
            if (!result) {
                LOG(ERROR) << "Failed to load library: " << file_path << "\n";
                return std::nullopt;
            }
            return result;
        }

        [[maybe_unused]] static std::optional< Configuration >
        loadConfiguration(const std::string &file_path) {
            YAMLParser parser;
            auto result = parser.parse_from_file< Configuration >(file_path);

            if (!result) {
                LOG(ERROR) << "Failed to load Patchestry configuration: " << file_path << "\n";
                return std::nullopt;
            }

            return result;
        }

        [[maybe_unused]] static bool
        saveConfiguration(const Configuration &config, const std::string &file_path) {
            YAMLParser parser;
            std::string yaml_content = parser.serialize_to_string(config);

            if (yaml_content.empty()) {
                LOG(ERROR) << "Failed to serialize Patchestry configuration to: " << file_path
                           << "\n";
                return false;
            }

            std::ofstream file(file_path);
            if (!file.is_open()) {
                LOG(ERROR) << "Failed to open file for writing: " << file_path << "\n";
                return false;
            }

            file << yaml_content;
            file.close();

            if (file.fail()) {
                LOG(ERROR) << "Failed to write to file: " << file_path << "\n";
                return false;
            }

            return true;
        }

        [[maybe_unused]] static bool validateConfiguration(const Configuration &config) {
            if (config.libraries.patches.empty()) {
                LOG(WARNING) << "Patchestry configuration contains no patches\n";
            }

            if (config.libraries.contracts.empty()) {
                LOG(WARNING) << "Patchestry configuration contains no contracts\n";
            }

            // Validate each patch
            for (const auto &patch : config.libraries.patches) {
                if (patch.name.empty()) {
                    LOG(ERROR) << "Patch specification missing name\n";
                    return false;
                }

                // Validate patch file exists if specified
                if (!patch.implementation.code_file.empty()) {
                    if (!llvm::sys::fs::exists(patch.implementation.code_file)) {
                        LOG(ERROR)
                            << "Patch file does not exist: " << patch.implementation.code_file
                            << "\n";
                        return false;
                    }
                }
            }

            return true;
        }
    } // namespace utils
} // namespace patchestry::yaml

namespace llvm::yaml {

    // Parse Target
    template<>
    struct MappingTraits< patchestry::passes::Target >
    {
        static void mapping(IO &io, patchestry::passes::Target &target) {
            io.mapOptional("binary", target.binary);
            io.mapRequired("arch", target.arch);
        }
    };

    // Parse Library
    template<>
    struct MappingTraits< patchestry::passes::Library >
    {
        static void mapping(IO &io, patchestry::passes::Library &library) {
            io.mapOptional("apiVersion", library.api_version);
            io.mapOptional("metadata", library.metadata);
            io.mapOptional("patches", library.patches);
            io.mapOptional("contracts", library.contracts);

            if (library.contracts.empty() && library.patches.empty()) {
                io.setError(
                    "Library '" + library.metadata.name
                    + "' must include at least one 'patches' or 'contracts' entry."
                );
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

            // Parse libraries as array of file paths
            std::vector< std::string > library_files;
            io.mapOptional("libraries", library_files);

            for (const auto &file : library_files) {
                std::string resolved_path = ConfigurationFile::getInstance().resolve_path(file);

                // Try loading as patch library
                auto library = patchestry::yaml::utils::loadLibrary(resolved_path);
                if (!library) {
                    LOG(ERROR) << "Failed to load library: " << resolved_path << "\n";
                    continue;
                }
                // check for api version mismatch
                if (config.api_version != library.value().api_version) {
                    LOG(ERROR) << "API version mismatch: " << config.libraries.api_version
                               << " != " << library.value().api_version << "\n";
                    continue;
                }

                config.libraries.patches.insert(
                    config.libraries.patches.end(), library.value().patches.begin(),
                    library.value().patches.end()
                );
                config.libraries.contracts.insert(
                    config.libraries.contracts.end(), library.value().contracts.begin(),
                    library.value().contracts.end()
                );
            }

            io.mapOptional("execution_order", config.execution_order);
            io.mapOptional("meta_patches", config.meta_patches);
            io.mapOptional("meta_contracts", config.meta_contracts);
        }
    };
} // namespace llvm::yaml
