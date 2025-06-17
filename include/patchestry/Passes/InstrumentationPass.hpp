/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>
#include <string>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassOptions.h>
#include <mlir/Support/LLVM.h>

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <clang/Frontend/CompilerInstance.h>

#include <patchestry/Passes/OperationMatcher.hpp>
#include <patchestry/Passes/PatchSpec.hpp>

// Forward declarations to minimize header dependencies
namespace mlir {
    class Operation;
    class Pass;
    class Type;
    class Value;
} // namespace mlir

namespace patchestry::passes { // NOLINT

    struct PatchOptions;

    /**
     * @brief Registers instrumentation passes with the MLIR pass registry.
     *
     * This function registers the instrumentation pass with the global MLIR pass registry,
     * making it available for use in pass pipelines. The spec_file parameter specifies
     * the patch specification file that defines the instrumentation rules.
     *
     * @param spec_file Path to the YAML patch specification file containing instrumentation
     * rules
     */
    void registerInstrumentationPasses(std::string spec_file);

    /**
     * @brief Creates a new instance of the InstrumentationPass.
     *
     * Factory function that creates and returns a unique pointer to an InstrumentationPass
     * instance.
     *
     * @param spec_file Path to the YAML patch specification file
     * @param patch_options Configuration options for controlling function inlining behavior
     * @return std::unique_ptr<mlir::Pass> A unique pointer to the created InstrumentationPass
     */
    std::unique_ptr< mlir::Pass >
    createInstrumentationPass(const std::string &spec_file, const PatchOptions &patch_options);

    /**
     * @brief Configuration options for controlling patching behavior.
     */
    struct PatchOptions
    {
        /** @brief Flag to enable or disable function inlining of patch functions */
        bool enable_inlining;
    };

    /**
     * @brief MLIR pass that applies code instrumentation based on patch specifications.
     *
     * The InstrumentationPass is an MLIR transformation pass that modifies MLIR modules
     * by applying patches according to specifications defined in a YAML configuration file.
     * It can instrument function calls and operations by inserting patch code before, after,
     * or replacing the original operations entirely.
     *
     * The pass supports three main instrumentation modes:
     * - APPLY_BEFORE: Insert patch code before the matched operation
     * - APPLY_AFTER: Insert patch code after the matched operation
     * - REPLACE: Replace the matched operation with patch code
     *
     * The pass operates on CIR (Clang IR) dialect operations within MLIR modules and can
     * optionally inline patch functions for performance.
     */
    class InstrumentationPass
        : public mlir::PassWrapper< InstrumentationPass, mlir::OperationPass< mlir::ModuleOp > >
    {
        /** @brief Path to the YAML patch specification file */
        std::string spec_file;

        /** @brief Parsed patch configuration from the specification file */
        std::optional< PatchConfiguration > config;

        /** @brief List of operations to be inlined after instrumentation */
        std::vector< mlir::Operation * > inline_worklists;

        /** @brief Reference to inlining configuration options */
        const PatchOptions &patch_options;

      public:
        /**
         * @brief Constructs an InstrumentationPass with the given specification file and
         * options.
         *
         * The constructor loads and parses the patch specification file, validates patch files,
         * and prepares the pass for execution. If the specification file cannot be loaded or
         * parsed, appropriate error messages are logged.
         *
         * @param spec Path to the YAML patch specification file
         * @param inline_options Reference to inlining configuration options
         */
        explicit InstrumentationPass(std::string spec, const PatchOptions &patch_options);

        /**
         * @brief Main entry point for the pass execution.
         *
         * This method is called by the MLIR pass manager to execute the instrumentation pass
         * on a module. It walks through all functions and operations in the module, applies
         * matching patches according to the specification, and optionally inlines patch
         * functions.
         */
        void runOnOperation() final;

        /**
         * @brief Instruments function calls within a given function.
         *
         * This method walks through all function call operations within the provided function
         * and applies patches that match the call patterns defined in the specification.
         * It handles argument matching and applies the appropriate patch mode (before, after,
         * replace).
         *
         * @param func The function containing calls to be instrumented
         */
        void instrument_function_calls(cir::FuncOp func);

        /**
         * @brief Instruments a specific operation based on patch specifications.
         *
         * This method applies patches to operations that match the operation patterns
         * defined in the patch specification. It supports variable matching and applies
         * before patch modes (replace and after mode is not supported for operations yet).
         *
         * @param op The operation to be instrumented
         */
        void instrument_operation(mlir::Operation *op);

      private:
        /**
         * @brief Prepares arguments for a patch function call.
         *
         * This method handles argument preparation for patch function calls, including
         * type casting when necessary. It supports special argument handling such as
         * passing return values and ensures type compatibility between original and patch
         * functions.
         *
         * @param builder MLIR operation builder for creating new operations
         * @param op The original operation being patched
         * @param patch_func The patch function to be called
         * @param patch Patch information containing argument specifications
         * @param args Output vector to store the prepared arguments
         */
        void prepare_call_arguments(
            mlir::OpBuilder &builder, mlir::Operation *op, cir::FuncOp patch_func,
            const PatchInfo &patch, llvm::SmallVector< mlir::Value > &args
        );

        /**
         * @brief Applies a patch before the target operation.
         *
         * This method inserts a call to the patch function immediately before the target
         * operation. It handles module symbol merging, argument preparation, and call creation.
         * The inserted call is added to the inline worklist if inlining is enabled.
         *
         * @param op The target operation to be instrumented
         * @param match The match information for the operation
         * @param patch The patch information containing the patch function details
         * @param patch_module The module containing the patch function
         */
        void apply_before_patch(
            mlir::Operation *op, const PatchMatch &match, const PatchInfo &patch,
            mlir::ModuleOp patch_module
        );

        /**
         * @brief Applies a patch after the target operation.
         *
         * This method inserts a call to the patch function immediately after the target
         * operation. It handles module symbol merging, argument preparation, and call creation.
         * The inserted call is added to the inline worklist if inlining is enabled.
         *
         * @param op The target operation to be instrumented
         * @param match The match information for the operation
         * @param patch The patch information containing the patch function details
         * @param patch_module The module containing the patch function
         */
        void apply_after_patch(
            mlir::Operation *op, const PatchMatch &match, const PatchInfo &patch,
            mlir::ModuleOp patch_module
        );

        /**
         * @brief Replaces a function call with a patch function call.
         *
         * This method completely replaces the original function call with a call to the
         * patch function. It preserves the original call's arguments and return types
         * while redirecting the call to the patch function.
         *
         * @param call_op The original call operation to be replaced
         * @param match The match information for the call
         * @param patch The patch information containing the replacement function details
         * @param patch_module The module containing the patch function
         */
        void replace_call(
            cir::CallOp op, const PatchMatch &match, const PatchInfo &patch,
            mlir::ModuleOp patch_module
        );

        /**
         * @brief Inlines a function call operation.
         *
         * This method performs function inlining by replacing a call operation with the
         * body of the called function. It handles control flow, argument mapping, and
         * block management to properly integrate the inlined code.
         *
         * @param module The module containing both caller and callee
         * @param call_op The call operation to be inlined
         * @return mlir::LogicalResult Success or failure of the inlining operation
         */
        mlir::LogicalResult inline_call(mlir::ModuleOp module, cir::CallOp call_op);

        /**
         * @brief Loads a patch module from its string representation.
         *
         * This method parses an MLIR module from its textual representation and returns
         * an owning reference to the parsed module. The module contains the patch functions
         * that will be merged into the target module.
         *
         * @param ctx The MLIR context for parsing
         * @param patch_string The textual representation of the patch module
         * @return mlir::OwningOpRef<mlir::ModuleOp> The parsed patch module
         */
        mlir::OwningOpRef< mlir::ModuleOp >
        load_patch_module(mlir::MLIRContext &ctx, const std::string &patch_string);

        /**
         * @brief Merges a specific symbol from source module into destination module.
         *
         * This method copies a named symbol (function, global, etc.) from the source module
         * to the destination module, handling symbol conflicts through renaming when necessary.
         * It also recursively copies any symbols that the target symbol depends on.
         *
         * @param dest The destination module to merge symbols into
         * @param src The source module containing the symbol to merge
         * @param symbol_name The name of the symbol to merge
         * @return mlir::LogicalResult Success or failure of the merge operation
         */
        mlir::LogicalResult merge_module_symbol(
            mlir::ModuleOp dest, mlir::ModuleOp src, const std::string &symbol_name
        );

        /**
         * @brief Determines if a function should be excluded from patching.
         *
         * This method checks whether a given function should be excluded from the
         * patching process based on the patch specification criteria.
         *
         * @param func The function to check for exclusion
         * @param spec The patch specification containing exclusion rules
         * @return bool True if the function should be excluded, false otherwise
         */
        bool exclude_from_patching(cir::FuncOp func, const PatchSpec &spec);

        /**
         * @brief Sets appropriate attributes for the patch call operation.
         *
         * This method handles setting attributes on the patch call based on the
         * type of the original operation being instrumented. It preserves relevant
         * attributes from CallOp operations and adds debugging information.
         *
         * @param patch_call_op The patch call operation to set attributes on
         * @param target_op The original operation being instrumented
         */
        void set_patch_call_attributes(cir::CallOp patch_call_op, mlir::Operation *target_op);
    };

} // namespace patchestry::passes
