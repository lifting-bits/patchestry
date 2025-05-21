/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/YAMLTraits.h>

namespace patchestry::passes {

    enum class PatchInfoMode : uint8_t {
        NONE = 0, // No patch
        APPLY_BEFORE,
        APPLY_AFTER,
        REPLACE
    };

    struct ArgumentMatch
    {
        int index;
        std::string name;
        std::string type;
    };

    struct VariableMatch
    {
        std::string name;
        std::string type;
    };

    struct PatchMatch
    {
        std::string symbol;
        std::string kind;
        std::string operation;
        std::string function_name;
        std::vector< ArgumentMatch > argument_matches;
        std::vector< VariableMatch > variable_matches;
    };

    struct PatchInfo
    {
        PatchInfoMode mode = PatchInfoMode::NONE;
        std::string code;
        std::string patch_file;
        std::string patch_function;
        std::optional< std::string > patch_module;
        std::vector< std::string > arguments;
    };

    struct PatchSpec
    {
        std::string name;
        PatchMatch match;
        PatchInfo patch;
        std::vector< std::string > exclude;
    };

    struct PatchConfiguration
    {
        std::string arch;
        std::vector< PatchSpec > patches;
    };

}; // namespace patchestry::passes

LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::PatchSpec)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::VariableMatch)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::ArgumentMatch)

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

    // Prase PatchMatch
    template<>
    struct MappingTraits< patchestry::passes::PatchMatch >
    {
        static void mapping(IO &io, patchestry::passes::PatchMatch &match) {
            io.mapOptional("symbol", match.symbol);
            io.mapOptional("kind", match.kind);
            io.mapOptional("operation", match.operation);
            io.mapOptional("function_name", match.function_name);
            io.mapOptional("argument_matches", match.argument_matches);
            io.mapOptional("variable_matches", match.variable_matches);
        }
    };

    // Prase PatchInfo
    template<>
    struct MappingTraits< patchestry::passes::PatchInfo >
    {
        static void mapping(IO &io, patchestry::passes::PatchInfo &patch) {
            io.mapOptional("code", patch.code);
            io.mapOptional("patch_file", patch.patch_file);
            io.mapOptional("patch_function", patch.patch_function);
            io.mapOptional("arguments", patch.arguments);

            std::string mode_str;
            io.mapRequired("mode", mode_str);
            if (mode_str == "ApplyBefore") {
                patch.mode = patchestry::passes::PatchInfoMode::APPLY_BEFORE;
            } else if (mode_str == "ApplyAfter") {
                patch.mode = patchestry::passes::PatchInfoMode::APPLY_AFTER;
            } else if (mode_str == "Replace") {
                patch.mode = patchestry::passes::PatchInfoMode::REPLACE;
            } else { // Default to NONE
                patch.mode = patchestry::passes::PatchInfoMode::NONE;
            }

            // resolve patch_file path
            if (!patch.patch_file.empty() && !llvm::sys::path::is_absolute(patch.patch_file)) {
                patch.patch_file =
                    PatchSpecContext::getInstance().resolve_path(patch.patch_file);
            }
        }
    };

    // Prase PatchSpec
    template<>
    struct MappingTraits< patchestry::passes::PatchSpec >
    {
        static void mapping(IO &io, patchestry::passes::PatchSpec &spec) {
            io.mapRequired("name", spec.name);
            io.mapRequired("match", spec.match);
            io.mapRequired("patch", spec.patch);
            io.mapOptional("exclude", spec.exclude);
        }
    };

    // Prase PatchConfiguration
    template<>
    struct MappingTraits< patchestry::passes::PatchConfiguration >
    {
        static void mapping(IO &io, patchestry::passes::PatchConfiguration &config) {
            io.mapRequired("arch", config.arch);
            io.mapRequired("patches", config.patches);
        }
    };

} // namespace llvm::yaml
