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

    using SymbolMatch = VariableMatch;

    struct FunctionContext
    {
        std::string name;
        std::string type;
    };

    struct PatchMatch
    {
        std::string name;
        MatchKind kind;
        std::vector< FunctionContext > function_context;
        std::vector< ArgumentMatch > argument_matches;
        std::vector< VariableMatch > variable_matches;
        std::vector< SymbolMatch > symbol_matches;
        std::vector< OperandMatch > operand_matches;
    };

    struct PatchInfo
    {
        PatchInfoMode mode = PatchInfoMode::NONE;
        std::string code;
        std::string patch_file;
        std::string patch_function;
        std::optional< std::string > patch_module;
        std::vector< ArgumentSource >
            argument_sources; // New: structured argument specifications
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

        bool matches_operation(const std::string &operation) const {
            if (operation.empty()) {
                return true;
            }
            return std::any_of(patches.begin(), patches.end(), [&](const PatchSpec &spec) {
                return spec.match.name == operation && spec.match.kind == MatchKind::OPERATION;
            }); // NOLINT
        }

        bool matches_symbol(const std::string &symbol) const {
            if (symbol.empty()) {
                return true;
            }
            return std::any_of(patches.begin(), patches.end(), [&](const PatchSpec &spec) {
                return spec.match.name == symbol && spec.match.kind == MatchKind::FUNCTION;
            }); // NOLINT
        }
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

LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::PatchSpec)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::VariableMatch)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::ArgumentMatch)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::ArgumentSource)
LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::FunctionContext)

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

    // Prase FunctionMatch
    template<>
    struct MappingTraits< patchestry::passes::FunctionContext >
    {
        static void mapping(IO &io, patchestry::passes::FunctionContext &func) {
            io.mapRequired("name", func.name);
            io.mapOptional("type", func.type);
        }
    };

    // Prase PatchMatch
    template<>
    struct MappingTraits< patchestry::passes::PatchMatch >
    {
        static void mapping(IO &io, patchestry::passes::PatchMatch &match) {
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

    // Prase PatchInfo
    template<>
    struct MappingTraits< patchestry::passes::PatchInfo >
    {
        static void mapping(IO &io, patchestry::passes::PatchInfo &patch) {
            io.mapOptional("code", patch.code);
            io.mapOptional("patch_file", patch.patch_file);
            io.mapOptional("patch_function", patch.patch_function);
            io.mapOptional("arguments", patch.argument_sources);

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
