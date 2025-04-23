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

#include <llvm/Support/YAMLTraits.h>

namespace patchestry::passes {

    enum class PatchOperationKind : uint8_t {
        NONE = 0, // No patch
        APPLY_BEFORE_PATCH,
        APPLY_AFTER_PATCH,
        WRAP_AROUND_PATCH
    };

    struct PatchOperation
    {
        PatchOperationKind kind = PatchOperationKind::NONE;
        std::string target;
        std::vector< std::string > arguments;
    };

    struct PatchConfig
    {
        std::string function;
        std::string patch_file;
        std::optional< std::string > patch_module;
        std::vector< std::string > locals;
        std::vector< PatchOperation > operations;
    };

    struct PatchSpec
    {
        std::string arch;
        std::vector< PatchConfig > patches;
    };

}; // namespace patchestry::passes

LLVM_YAML_IS_SEQUENCE_VECTOR(patchestry::passes::PatchConfig)

namespace llvm::yaml {

    // Prase PatchOperation
    template<>
    struct MappingTraits< patchestry::passes::PatchOperation >
    {
        static void mapping(IO &io, patchestry::passes::PatchOperation &arg) {
            io.mapRequired("target", arg.target);
            io.mapOptional("arguments", arg.arguments);
        }
    };

    // Prase PatchConfig
    template<>
    struct MappingTraits< patchestry::passes::PatchConfig >
    {
        static void mapping(IO &io, patchestry::passes::PatchConfig &spec) {
            io.mapRequired("function", spec.function);
            io.mapOptional("patch_file", spec.patch_file);
            io.mapOptional("locals", spec.locals);

            patchestry::passes::PatchOperation before_operation;
            patchestry::passes::PatchOperation after_operation;
            patchestry::passes::PatchOperation wrap_operation;

            io.mapOptional("apply_before", before_operation);
            io.mapOptional("apply_after", after_operation);
            io.mapOptional("wrap_around", wrap_operation);
            bool has_apply_before = before_operation.target != "";
            bool has_apply_after  = after_operation.target != "";
            bool has_wrap_around  = wrap_operation.target != "";

            if ((has_apply_before || has_apply_after) && has_wrap_around) {
                llvm::report_fatal_error(
                    "Wrap around patch cannot be combined with apply before or after patch"
                );
            }

            if (has_apply_before) {
                before_operation.kind =
                    patchestry::passes::PatchOperationKind::APPLY_BEFORE_PATCH;
                spec.operations.emplace_back(before_operation);
            }

            if (has_apply_after) {
                after_operation.kind =
                    patchestry::passes::PatchOperationKind::APPLY_AFTER_PATCH;
                spec.operations.emplace_back(after_operation);
            }

            if (has_wrap_around) {
                wrap_operation.kind = patchestry::passes::PatchOperationKind::WRAP_AROUND_PATCH;
                spec.operations.emplace_back(wrap_operation);
            }

            // If no operations were defined, report an error
            if (spec.operations.empty()) {
                std::string error_message =
                    "No patch operations specified for function: " + spec.function;
                llvm::report_fatal_error(error_message.c_str());
            }
        }
    };

    // Prase PatchSpec
    template<>
    struct MappingTraits< patchestry::passes::PatchSpec >
    {
        static void mapping(IO &io, patchestry::passes::PatchSpec &config) {
            io.mapRequired("arch", config.arch);
            io.mapRequired("patches", config.patches);
        }
    };

} // namespace llvm::yaml
