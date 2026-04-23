/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <llvm/Support/YAMLTraits.h>

#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <patchestry/Util/Log.hpp>

namespace patchestry::yaml {

    class YAMLParser
    {
      public:
        YAMLParser()  = default;
        ~YAMLParser() = default;

        template< typename T >
        std::optional< T > parse_from_file(const std::string &file_path); // NOLINT

        template< typename T >
        std::optional< T > parse_from_string(const std::string &yaml_content); // NOLINT

        // Serialize any YAML-serializable type to string
        template< typename T >
        std::string serialize_to_string(const T &object);

        // Validate YAML file structure
        template< typename T >
        bool validate_yaml_file(const std::string &file_path);

      private:
        // Load file into memory buffer
        std::unique_ptr< llvm::MemoryBuffer > load_file(const std::string &file_path);

        // Parse YAML content with error handling
        template< typename T >
        std::optional< T >
        parse_yaml_content(const std::string &content, const std::string &source = "");
    };

    template< typename T >
    std::optional< T >
    YAMLParser::parse_yaml_content(const std::string &content, const std::string &source) {
        T result;
        llvm::yaml::Input input(content);

        input >> result;

        if (input.error()) {
            if (!source.empty()) {
                LOG(ERROR) << "YAML parse error in '" << source
                           << "': " << input.error().message() << "\n";
            } else {
                LOG(ERROR) << "YAML parse error: " << input.error().message() << "\n";
            }
            return std::nullopt;
        }

        return result;
    }

} // namespace patchestry::yaml
