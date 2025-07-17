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

#include <patchestry/Passes/PatchSpec.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::yaml {

    template< typename T >
    std::optional< T > YAMLParser::parse_from_file(const std::string &file_path) {
        // Set the spec path for relative path resolution
        auto file   = PatchSpecContext::getInstance().resolve_path(file_path);
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

        // Try to parse as PatchConfiguration to validate structure
        auto config = parse_yaml_content< T >(buffer->getBuffer().str());
        if (!config) {
            return false;
        }

        // Additional validation could be added here
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
    template std::optional< patchestry::passes::PatchConfiguration >
    YAMLParser::parse_from_file< patchestry::passes::PatchConfiguration >(
        const std::string &file_path
    );

    template std::optional< patchestry::passes::PatchLibrary >
    YAMLParser::parse_from_file< patchestry::passes::PatchLibrary >(const std::string &file_path
    );

    template std::optional< patchestry::passes::ContractLibrary >
    YAMLParser::parse_from_file< patchestry::passes::ContractLibrary >(
        const std::string &file_path
    );

    template std::optional< patchestry::passes::PatchConfiguration >
    YAMLParser::parse_from_string< patchestry::passes::PatchConfiguration >(
        const std::string &yaml_content
    );

    template std::optional< patchestry::passes::PatchLibrary >
    YAMLParser::parse_from_string< patchestry::passes::PatchLibrary >(
        const std::string &yaml_content
    );

    template std::optional< patchestry::passes::ContractLibrary >
    YAMLParser::parse_from_string< patchestry::passes::ContractLibrary >(
        const std::string &yaml_content
    );

    template std::string
    YAMLParser::serialize_to_string< patchestry::passes::PatchConfiguration >(
        const patchestry::passes::PatchConfiguration &object
    );

    template bool YAMLParser::validate_yaml_file< patchestry::passes::PatchConfiguration >(
        const std::string &file_path
    );

} // namespace patchestry::yaml
