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
#include <patchestry/YAML/ConfigurationFile.hpp>

// Forward declarations to minimize header dependencies
namespace mlir {
    class Function;
    class Operation;
    class Pass;
    class Type;
    class Value;
} // namespace mlir

namespace patchestry::passes { // NOLINT

    class ContractOperationImpl;
    class PatchOperationImpl;

    struct InstrumentationOptions;
    struct ContractInformation;

    /**
     * @brief Registers instrumentation passes with the MLIR pass registry.
     *
     * This function registers the instrumentation pass with the global MLIR pass registry,
     * making it available for use in pass pipelines. The configuration_file parameter specifies
     * the Patchestry configuration file that defines the meta-patches, meta-contracts, and
     * other instrumentation rules.
     *
     * @param configuration_file Path to the YAML Patchestry configuration file containing
     * instrumentation rules for applying patches, contracts, or both
     */
    void RegisterInstrumentationPasses(std::string configuration_file);

    /**
     * @brief Creates a new instance of the InstrumentationPass.
     *
     * Factory function that creates and returns a unique pointer to an InstrumentationPass
     * instance.
     *
     * @param configuration_file Path to the YAML configuration file
     * @param options Configuration options for controlling how instrumentation is generally
     * applied
     * @return std::unique_ptr<mlir::Pass> A unique pointer to the created InstrumentationPass
     */
    std::unique_ptr< mlir::Pass > CreateInstrumentationPass(
        const std::string &configuration_file, const InstrumentationOptions &options
    );

    /**
     * @brief Configuration options for controlling instrumentation behavior.
     */
    struct InstrumentationOptions
    {
        /** @brief Flag to enable or disable function inlining of instrumentation functions */
        bool enable_inlining;
    };

    struct PatchInformation
    {
        std::optional< patch::PatchSpec > spec;
        std::optional< patch::PatchAction > patch_action;
        // Named captures bound by `OperationMatcher::patch_action_matches`
        // (operand/result index by name). Assigned onto a per-match-site copy
        // of `PatchInformation` in `apply_patch_action_to_targets`; read by
        // `handle_capture_argument` when a `source: capture` argument is
        // resolved.
        llvm::StringMap< mlir::Value > captures;
    };

    /**
     * @brief MLIR pass that applies code instrumentation based on configuration.
     *
     * The InstrumentationPass is an MLIR transformation pass that modifies MLIR modules
     * by instrumenting according to specifications defined in a YAML configuration file.
     * It can instrument function calls and operations by inserting patch or contract code
     * before or after, redirecting the matched call to a patch, deleting an op entirely,
     * or hoisting a patch call to the caller's entry block.
     *
     * The pass supports five instrumentation modes; the first three are shared between
     * patches and contracts, the last two are patch-only:
     * - APPLY_BEFORE: Insert the patch call / attach the contract attribute before the
     *   matched op.
     * - APPLY_AFTER: Insert the patch call / attach the contract attribute after the
     *   matched op.
     * - APPLY_AT_ENTRYPOINT: Insert the patch call at the enclosing (caller) function's
     *   entry block, after the alloca + parameter-store prologue. Patch-only — contracts
     *   are static and attach predicates as MLIR attributes, so "entrypoint" has no
     *   meaning for them.
     * - REPLACE: Replace the matched op with a call to the patch function (patch-only).
     * - ERASE: Delete the matched op, replacing live results with default values
     *   (patch-only; no patch function invoked).
     *
     * The pass operates on CIR (Clang IR) dialect operations within MLIR modules and can
     * optionally inline patch functions for performance.
     */
    class InstrumentationPass
        : public mlir::PassWrapper< InstrumentationPass, mlir::OperationPass< mlir::ModuleOp > >

    {
        friend class ContractOperationImpl;
        friend class PatchOperationImpl;

        /** @brief Path to the YAML Patchestry configuration file */
        std::string configuration_file;

        /** @brief Parsed patch- or contract-specific configuration from the file */
        std::optional< Configuration > config;

        /** @brief List of operations to be inlined after instrumentation */
        std::set< mlir::Operation * > inline_worklists;

        /** @brief Reference to inlining configuration options */
        const InstrumentationOptions &options;

      public:
        /**
         * @brief Constructs an InstrumentationPass with the given configuration file and
         * options.
         *
         * The constructor loads and parses the configuration YAML file, validates the indicated
         * patches and/or contracts, and prepares the pass for execution. If a file cannot be
         * loaded or parsed, appropriate error messages are logged.
         *
         * @param configuration_file Path to the YAML configuration file
         * @param inline_options Reference to inlining configuration options
         */
        explicit InstrumentationPass(
            std::string configuration_file, const InstrumentationOptions &options
        );

        /**
         * @brief Main entry point for the pass execution.
         *
         * This method is called by the MLIR pass manager to execute the instrumentation pass
         * on a module. It walks through all functions and operations in the module, applies
         * matching patches according to the configuration (meta-patches, etc.), and optionally
         * inlines patch functions.
         */
        void runOnOperation() final;

        /**
         * @brief Applies meta patches in execution order.
         *
         * @param function_worklist List of functions to process
         * @param operation_worklist List of operations to process
         * @param meta_patch_name Name of the meta patch to apply
         */
        void apply_meta_patches(
            llvm::SmallVector< cir::FuncOp, 8 > &function_worklist,
            llvm::SmallVector< mlir::Operation *, 8 > &operation_worklist,
            const std::string &meta_patch_name
        );

        /**
         * @brief Applies meta contracts in execution order.
         *
         * @param function_worklist List of functions to process
         * @param meta_contract_name Name of the meta contract to apply
         */
        void apply_meta_contracts(
            llvm::SmallVector< cir::FuncOp, 8 > &function_worklist,
            const std::string &meta_contract_name
        );

        /**
         * @brief Applies a specific patch action to target functions and operations.
         *
         * @param function_worklist List of functions to process
         * @param operation_worklist List of operations to process
         * @param meta_patch
         * @param modified_patch
         */
        void apply_patch_action_to_targets(
            llvm::SmallVector< cir::FuncOp, 8 > &function_worklist,
            llvm::SmallVector< mlir::Operation *, 8 > &operation_worklist,
            const patch::MetaPatchConfig &meta_patch, const PatchInformation &modified_patch
        );

        /**
         * @brief Applies a specific contract action to target functions.
         *
         * @param function_worklist List of functions to process
         * @param meta_contract
         * @param modified_contract
         */
        void apply_contract_action_to_targets(
            llvm::SmallVector< cir::FuncOp, 8 > &function_worklist,
            const contract::MetaContractConfig &meta_contract,
            const ContractInformation &modified_contract
        );

      private:
        /**
         * @brief Prepares arguments for a patch function call.
         *
         * Dispatches each `arguments:` entry to the matching `handle_*_argument` helper,
         * which resolves the value, inserts casts as needed, and writes the result into
         * `args_map` keyed by the original SSA value (so reference-type writeback in
         * `update_state_after_patch` can find the source).
         *
         * @param builder MLIR operation builder for creating new operations.
         * @param op The matched operation the patch is being wrapped around.
         * @param patch_func The patch function whose arguments are being assembled.
         * @param patch Patch information (spec + action + per-match capture bindings).
         * @param args_map Output map from the original SSA value to the patch-arg value
         *                 (possibly cast). Ordering follows YAML declaration order.
         * @param entrypoint_func When set, the patch call is being emitted at this
         *                        function's entry block (APPLY_AT_ENTRYPOINT mode).
         *                        Semantics:
         *                        - OPERAND index N is remapped to
         *                          `entrypoint_func.getArguments()[N]` so no call-site
         *                          value leaks into the entry block (SSA dominance).
         *                        - RETURN_VALUE and CAPTURE are rejected: both are only
         *                          defined at the matched call site.
         *                        - VARIABLE / SYMBOL / CONSTANT resolve normally.
         */
        void prepare_patch_call_arguments(
            mlir::OpBuilder &builder, mlir::Operation *op, cir::FuncOp patch_func,
            const PatchInformation &patch,
            llvm::MapVector< mlir::Value, mlir::Value > &args_map,
            std::optional< cir::FuncOp > entrypoint_func = std::nullopt
        );

        void update_state_after_patch(
            mlir::OpBuilder &builder, cir::CallOp patch_call_op, mlir::Operation *target_op,
            const PatchInformation &patch, llvm::MapVector< mlir::Value, mlir::Value > &arg_map
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
         * @brief Loads a code module from its string representation. This can be either a
         * patch, or a contract!
         *
         * This method parses an MLIR module from its textual representation and returns
         * an owning reference to the parsed module. The module contains the patch functions
         * that will be merged into the target module.
         *
         * @param ctx The MLIR context for parsing
         * @param module_string The textual representation of the patch module
         * @return mlir::OwningOpRef<mlir::ModuleOp> The parsed patch module
         */
        mlir::OwningOpRef< mlir::ModuleOp >
        load_code_module(mlir::MLIRContext &ctx, const std::string &module_string);

        /**
         * @brief Merges a specific symbol from source module into destination module.
         *
         * This method copies a named symbol (function, global, etc.) from the source module
         * to the destination module, handling symbol conflicts through renaming when necessary.
         * It also recursively copies any symbols that the target symbol depends on.
         *
         * Function definitions (not declarations) are automatically assigned internal linkage
         * to prevent symbol pollution in the final binary, enable better optimizations, and
         * avoid naming conflicts. This makes instrumentation functions (patches/contracts)
         * module-local implementation details rather than externally visible APIs.
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
         * @brief Creates a cast operation if types differ, otherwise returns the original
         * value.
         */
        mlir::Value create_cast_if_needed(
            mlir::OpBuilder &builder, mlir::Operation *call_op, mlir::Value value,
            mlir::Type target_type
        );

        /**
         * @brief Creates a reference (alloca + store) for a given value.
         */
        mlir::Value
        create_reference(mlir::OpBuilder &builder, mlir::Operation *call_op, mlir::Value value);

        /**
         * @brief Handles OPERAND argument source type.
         */
        void handle_operand_argument(
            mlir::OpBuilder &builder, mlir::Operation *call_op,
            const patch::ArgumentSource &arg_spec, mlir::Type patch_arg_type,
            llvm::MapVector< mlir::Value, mlir::Value > &arg_map,
            std::optional< cir::FuncOp > entrypoint_func = std::nullopt
        );

        /**
         * @brief Handles VARIABLE argument source type.
         */
        void handle_variable_argument(
            mlir::OpBuilder &builder, mlir::Operation *call_op,
            const patch::ArgumentSource &arg_spec, mlir::Type patch_arg_type,
            llvm::MapVector< mlir::Value, mlir::Value > &arg_map
        );

        /**
         * @brief Handles SYMBOL argument source type.
         */
        void handle_symbol_argument(
            mlir::OpBuilder &builder, mlir::Operation *call_op,
            const patch::ArgumentSource &arg_spec, mlir::Type patch_arg_type,
            llvm::MapVector< mlir::Value, mlir::Value > &arg_map
        );

        /**
         * @brief Handles RETURN_VALUE argument source type.
         */
        void handle_return_value_argument(
            mlir::OpBuilder &builder, mlir::Operation *call_op,
            const patch::ArgumentSource &arg_spec, mlir::Type patch_arg_type,
            llvm::MapVector< mlir::Value, mlir::Value > &arg_map,
            std::optional< cir::FuncOp > entrypoint_func = std::nullopt
        );

        /**
         * @brief Handles CONSTANT argument source type.
         */
        void handle_constant_argument(
            mlir::OpBuilder &builder, mlir::Operation *call_op,
            const patch::ArgumentSource &arg_spec, mlir::Type patch_arg_type,
            llvm::MapVector< mlir::Value, mlir::Value > &arg_map
        );

        /**
         * @brief Handles CAPTURE argument source type — looks up the named
         *        capture in `patch.captures` and uses the bound `mlir::Value`.
         */
        void handle_capture_argument(
            mlir::OpBuilder &builder, mlir::Operation *call_op,
            const patch::ArgumentSource &arg_spec, mlir::Type patch_arg_type,
            const PatchInformation &patch,
            llvm::MapVector< mlir::Value, mlir::Value > &arg_map,
            std::optional< cir::FuncOp > entrypoint_func = std::nullopt
        );

        /**
         * @brief Parses a constant operand from string based on the target type.
         */
        mlir::Value parse_constant_operand(
            mlir::OpBuilder &builder, mlir::Operation *call_op, const std::string &value,
            mlir::Type target_type
        );

        /**
         * @brief Finds a local variable by name in the current function scope.
         */
        std::optional< mlir::Value >
        find_local_variable(mlir::Operation *call_op, const std::string &var_name);

        /**
         * @brief Finds a global symbol (variable or function) by name.
         */
        std::optional< mlir::Value > find_global_symbol(
            mlir::OpBuilder &builder, mlir::Operation *call_op, const std::string &symbol_name
        );
        void set_instrumentation_call_attributes(
            cir::CallOp instr_call_op, mlir::Operation *target_op
        );
    };

} // namespace patchestry::passes
