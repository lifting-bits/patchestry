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
    static ConfigurationFile &GetInstance() {
        static ConfigurationFile instance;
        return instance;
    }

    void SetFilePath(const std::string &file) {
        auto directory = llvm::sys::path::parent_path(file);
        if (!directory.empty()) {
            file_path_ = directory.str();
        }
    }

    std::string ResolvePath(const std::string &file) {
        if (llvm::sys::path::is_absolute(file)) {
            return file;
        }

        llvm::SmallVector< char > directory;
        directory.assign(file_path_.begin(), file_path_.end());
        llvm::sys::path::append(directory, file);
        llvm::sys::path::remove_dots(directory);
        return std::string(directory.data(), directory.size());
    }

  private:
    std::string file_path_;
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
        std::vector< patch::MetaPatchConfig > meta_patches;
        std::vector< contract::MetaContractConfig > meta_contracts;
    };
} // namespace patchestry::passes

namespace patchestry::yaml {
    using namespace patchestry::passes;

    namespace utils {

        [[maybe_unused]] static std::optional< Library >
        LoadLibrary(const std::string &file_path) {
            YAMLParser parser;
            auto result = parser.parse_from_file< Library >(file_path);
            if (!result) {
                LOG(ERROR) << "Failed to load library: " << file_path << "\n";
                return std::nullopt;
            }
            return result;
        }

        [[maybe_unused]] static std::optional< Configuration >
        LoadConfiguration(const std::string &file_path) {
            YAMLParser parser;
            auto result = parser.parse_from_file< Configuration >(file_path);

            if (!result) {
                LOG(ERROR) << "Failed to load Patchestry configuration: " << file_path << "\n";
                return std::nullopt;
            }

            return result;
        }

        [[maybe_unused]] static bool
        SaveConfiguration(const Configuration &config, const std::string &file_path) {
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
            if (!file.good()) {
                LOG(ERROR) << "Error during write to file: " << file_path << "\n";
                return false;
            }
            file.close();
            if (file.fail()) {
                LOG(ERROR) << "Failed to flush/close file: " << file_path << "\n";
                return false;
            }

            return true;
        }

        [[maybe_unused]] static bool ValidateConfiguration(const Configuration &config) {
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
                if (!patch.code_file.empty()) {
                    if (!llvm::sys::fs::exists(patch.code_file)) {
                        LOG(ERROR) << "Patch file does not exist: " << patch.code_file << "\n";
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

    template<>
    struct MappingTraits< patchestry::passes::Library >
    {
        static void mapping(IO &io, patchestry::passes::Library &library) {
            io.mapOptional("apiVersion", library.api_version);
            io.mapOptional("metadata", library.metadata);

            // Reject legacy `patches:` / `contracts:` keys before parsing
            // the new ones — silently accepting either spelling masks
            // partial migrations and lets a stale library shadow a
            // newly-renamed one.
            std::vector< patch::PatchSpec > legacy_patches;
            std::vector< contract::ContractSpec > legacy_contracts;
            io.mapOptional("patch_definitions", library.patches);
            io.mapOptional("contract_definitions", library.contracts);

            if (!library.metadata.kind.empty()
                && library.metadata.kind != "PatchLibrary")
            {
                io.setError(
                    "library '" + library.metadata.name
                    + "' has metadata.kind: '" + library.metadata.kind
                    + "', but only 'PatchLibrary' is valid for files "
                      "loaded via 'libraries:'. Either remove the kind "
                      "field or set it to 'PatchLibrary'."
                );
                return;
            }

            if (library.contracts.empty() && library.patches.empty()) {
                io.setError(
                    "Library '" + library.metadata.name
                    + "' must include at least one 'patch_definitions' "
                      "or 'contract_definitions' entry."
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

            // If `metadata.kind` is set on the deployment file it must
            // identify it as a PatchSpec; the only other accepted value
            // is PatchLibrary, which would be a misuse here (deployments
            // carry rules + targets, not definitions).
            if (!config.metadata.kind.empty()
                && config.metadata.kind != "PatchSpec")
            {
                io.setError(
                    "deployment file '" + config.metadata.name
                    + "' has metadata.kind: '" + config.metadata.kind
                    + "', but only 'PatchSpec' is valid for deployment "
                      "files (the ones carrying 'target:' / "
                      "'libraries:' / rule entries). 'PatchLibrary' "
                      "files belong under 'libraries:'."
                );
                return;
            }

            // A `kind: PatchSpec` deployment must not use the library
            // keys; otherwise the file is structurally a library that
            // happens to be loaded as a deployment, which would silently
            // drop its definitions. Reject loudly.
            std::vector< patch::PatchSpec > stray_patch_defs;
            std::vector< contract::ContractSpec > stray_contract_defs;
            io.mapOptional("patch_definitions", stray_patch_defs);
            io.mapOptional("contract_definitions", stray_contract_defs);
            if (!stray_patch_defs.empty() || !stray_contract_defs.empty()) {
                io.setError(
                    "deployment file '" + config.metadata.name
                    + "' carries 'patch_definitions:' / "
                      "'contract_definitions:' keys, which belong in a "
                      "'kind: PatchLibrary' file loaded via "
                      "'libraries:'. Move these definitions into a "
                      "library YAML and reference it from 'libraries:'."
                );
                return;
            }

            // Parse libraries as array of file paths
            std::vector< std::string > library_files;
            io.mapOptional("libraries", library_files);

            for (const auto &file : library_files) {
                // A library that fails to load is a hard error. Falling
                // through to "continue" silently left the spec missing
                // patches, and the pass would later fail at dispatch time
                // with a confusing "patch specification for ID '…' not
                // found" instead of pointing at the misconfigured path.
                auto library = patchestry::yaml::utils::LoadLibrary(file);
                if (!library) {
                    io.setError("Failed to load library: " + file);
                    return;
                }
                if (config.api_version != library.value().api_version) {
                    io.setError(
                        "API version mismatch in library '" + file + "': expected "
                        + config.api_version + ", got " + library.value().api_version
                    );
                    return;
                }

                // Reject duplicate definition names across libraries —
                // before this check the loader silently shadowed
                // (last-loaded wins), turning a copy/paste mistake into
                // a "patch ran but with the wrong implementation"
                // miscompile. Loud failure with both libraries' names
                // points the spec author at the conflict immediately.
                for (const auto &incoming : library.value().patches) {
                    for (const auto &existing : config.libraries.patches) {
                        if (incoming.name == existing.name) {
                            io.setError(
                                "duplicate patch definition '"
                                + incoming.name + "' loaded from '" + file
                                + "' — a definition with the same name "
                                  "was already loaded from an earlier "
                                  "library. Rename one of the entries or "
                                  "drop the redundant 'libraries:' "
                                  "reference."
                            );
                            return;
                        }
                    }
                }
                for (const auto &incoming : library.value().contracts) {
                    for (const auto &existing : config.libraries.contracts) {
                        if (incoming.name == existing.name) {
                            io.setError(
                                "duplicate contract definition '"
                                + incoming.name + "' loaded from '" + file
                                + "' — a definition with the same name "
                                  "was already loaded from an earlier "
                                  "library. Rename one of the entries or "
                                  "drop the redundant 'libraries:' "
                                  "reference."
                            );
                            return;
                        }
                    }
                }

                config.libraries.patches.insert(
                    config.libraries.patches.end(),
                    std::make_move_iterator(library.value().patches.begin()),
                    std::make_move_iterator(library.value().patches.end())
                );
                config.libraries.contracts.insert(
                    config.libraries.contracts.end(),
                    std::make_move_iterator(library.value().contracts.begin()),
                    std::make_move_iterator(library.value().contracts.end())
                );
            }

            // The legacy `meta_patches:` / `meta_contracts:` YAML
            // surface is removed. Authors use the simplified
            // `patches:` / `contracts:` keys exclusively. Spec files
            // still carrying the old keys hard-error here via LLVM
            // YAMLTraits' "unknown key" path — since neither
            // `meta_patches` nor `meta_contracts` is mapped, strict
            // mode rejects them at parse time.
            std::vector< patch::PatchEntry > patch_entries;
            io.mapOptional("patches", patch_entries);
            for (const auto &entry : patch_entries) {
                config.meta_patches.emplace_back(patch::inflatePatchEntry(entry));
            }

            std::vector< contract::ContractEntry > contract_entries;
            io.mapOptional("contracts", contract_entries);
            for (const auto &entry : contract_entries) {
                config.meta_contracts.emplace_back(contract::inflateContractEntry(entry));
            }
        }
    };
} // namespace llvm::yaml
