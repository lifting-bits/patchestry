/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/YAML/YAMLParser.hpp>

#include <filesystem>
#include <fstream>
#include <optional>

#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/Util/Log.hpp>
#include <patchestry/YAML/ConfigurationFile.hpp>

namespace patchestry::yaml {

    template< typename T >
    std::optional< T > YAMLParser::parse_from_file(const std::string &file_path) {
        // Set the spec path for relative path resolution
        auto file   = ConfigurationFile::getInstance().resolve_path(file_path);
        auto buffer = load_file(file);
        if (!buffer) {
            LOG(ERROR) << "Failed to load file: " << file_path << "\n";
            return std::nullopt;
        }

        return parse_yaml_content< T >(buffer->getBuffer().str());
    }

    template< typename T >
    std::optional< T > YAMLParser::parse_from_string(const std::string &yaml_content) {
        return parse_yaml_content< T >(yaml_content);
    }

    template< typename T >
    std::string YAMLParser::serialize_to_string(const T &object) {
        std::string output;
        llvm::raw_string_ostream stream(output);
        llvm::yaml::Output yaml_output(stream);

        // Create a copy since yaml::Output may modify the object
        T copy = object;
        yaml_output << copy;

        return output;
    }

    template< typename T >
    bool YAMLParser::validate_yaml_file(const std::string &file_path) {
        auto buffer = load_file(file_path);
        if (!buffer) {
            LOG(ERROR) << "Failed to load file: " << file_path << "\n";
            return false;
        }

        // Try to parse as Configuration to validate structure
        auto config = parse_yaml_content< T >(buffer->getBuffer().str());
        if (!config) {
            return false;
        }

        return true;
    }

    std::unique_ptr< llvm::MemoryBuffer > YAMLParser::load_file(const std::string &file_path) {
        if (!llvm::sys::fs::exists(file_path)) {
            LOG(ERROR) << "File does not exist: " << file_path << "\n";
            return nullptr;
        }

        auto bufferOrErr = llvm::MemoryBuffer::getFile(file_path);
        if (!bufferOrErr) {
            LOG(ERROR) << "Failed to read file: " << file_path << " - "
                       << bufferOrErr.getError().message() << "\n";
            return nullptr;
        }

        return std::move(bufferOrErr.get());
    }

    // Explicit template instantiations for commonly used types to avoid linker errors
    template std::optional< patchestry::passes::Configuration >
    YAMLParser::parse_from_file< patchestry::passes::Configuration >(
        const std::string &file_path
    );

    template std::optional< patchestry::passes::patch::PatchLibrary >
    YAMLParser::parse_from_file< patchestry::passes::patch::PatchLibrary >(
        const std::string &file_path
    );

    template std::optional< patchestry::passes::contract::ContractLibrary >
    YAMLParser::parse_from_file< patchestry::passes::contract::ContractLibrary >(
        const std::string &file_path
    );

    template std::optional< patchestry::passes::Configuration >
    YAMLParser::parse_from_string< patchestry::passes::Configuration >(
        const std::string &yaml_content
    );

    template std::optional< patchestry::passes::patch::PatchLibrary >
    YAMLParser::parse_from_string< patchestry::passes::patch::PatchLibrary >(
        const std::string &yaml_content
    );

    template std::optional< patchestry::passes::contract::ContractLibrary >
    YAMLParser::parse_from_string< patchestry::passes::contract::ContractLibrary >(
        const std::string &yaml_content
    );

    template std::string YAMLParser::serialize_to_string< patchestry::passes::Configuration >(
        const patchestry::passes::Configuration &object
    );

    template bool YAMLParser::validate_yaml_file< patchestry::passes::Configuration >(
        const std::string &file_path
    );

} // namespace patchestry::yaml
