/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <clang/CIR/Dialect/IR/CIRDialect.h>

namespace patchestry::passes {

    // Forward declarations
    class InstrumentationPass;
    struct PatchInformation;

    /**
     * @brief Implementation class for patch operations.
     *
     * This class provides static methods for applying patches to operations
     * in different modes (before, after, replace).
     */
    class PatchOperationImpl
    {
      public:
        /**
         * @brief Applies a patch before the target operation.
         *
         * This method inserts a call to the patch function immediately before the target
         * operation. It handles module symbol merging, argument preparation, and call creation.
         * The inserted call is added to the inline worklist if inlining is enabled.
         *
         * @param pass The instrumentation pass instance
         * @param op The target operation to be instrumented
         * @param patch The patch information containing the patch function details
         * @param patch_module The module containing the patch function
         * @param inline_patches Whether or not to inline at application
         */
        static void applyPatchBefore(
            InstrumentationPass &pass, mlir::Operation *op, const PatchInformation &patch,
            mlir::ModuleOp patch_module, bool inline_patches
        );

        /**
         * @brief Applies a patch after the target operation.
         *
         * This method inserts a call to the patch function immediately after the target
         * operation. It handles module symbol merging, argument preparation, and call creation.
         * The inserted call is added to the inline worklist if inlining is enabled.
         *
         * @param pass The instrumentation pass instance
         * @param op The target operation to be instrumented
         * @param patch The patch information containing the patch function details
         * @param patch_module The module containing the patch function
         * @param inline_patches Whether or not to inline at application
         */
        static void applyPatchAfter(
            InstrumentationPass &pass, mlir::Operation *op, const PatchInformation &patch,
            mlir::ModuleOp patch_module, bool inline_patches
        );

        /**
         * @brief Replaces a function call with a patch function call.
         *
         * This method completely replaces the original function call with a call to the
         * patch function. It preserves the original call's arguments and return types
         * while redirecting the call to the patch function.
         *
         * @param pass The instrumentation pass instance
         * @param call_op The original call operation to be replaced
         * @param patch The patch information containing the replacement function details
         * @param patch_module The module containing the patch function
         * @param inline_patches Whether or not to inline at application
         */
        static void replaceCall(
            InstrumentationPass &pass, cir::CallOp op, const PatchInformation &patch,
            mlir::ModuleOp patch_module, bool inline_patches
        );
    };

} // namespace patchestry::passes
