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

    struct InstrumentationOptions;

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
    void registerInstrumentationPasses(std::string configuration_file);

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
    std::unique_ptr< mlir::Pass > createInstrumentationPass(
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
    };

    struct ContractInformation
    { // the use of optional here takes care of some typing errors
        std::optional< contract::ContractSpec > spec;
        std::optional< contract::ContractAction > action;
    };

    /**
     * @brief MLIR pass that applies code instrumentation based on configuration.
     *
     * The InstrumentationPass is an MLIR transformation pass that modifies MLIR modules
     * by instrumenting according to specifications defined in a YAML configuration file.
     * It can instrument function calls and operations by inserting patch or contract code
     * before or after, or by replacing an original operation entirely with a patch.
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
        /** @brief Path to the YAML Patchestry configuration file */
        std::string configuration_file;

        /** @brief Parsed patch- or contract-specific configuration from the file */
        std::optional< Configuration > config;

        /** @brief List of operations to be inlined after instrumentation */
        std::vector< mlir::Operation * > inline_worklists;

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
        void prepare_patch_call_arguments(
            mlir::OpBuilder &builder, mlir::Operation *op, cir::FuncOp patch_func,
            const PatchInformation &patch, llvm::MapVector< mlir::Value, mlir::Value > &args_map
        );

        void update_state_after_patch(
            mlir::OpBuilder &builder, cir::CallOp patch_call_op, mlir::Operation *target_op,
            const PatchInformation &patch, llvm::MapVector< mlir::Value, mlir::Value > &arg_map
        );

        /**
         * @brief Prepares arguments for a contract function call.
         *
         * This method handles argument preparation for contract function calls, including
         * type casting when necessary. It supports special argument handling such as
         * passing return values and ensures type compatibility between original and patch
         * functions.
         *
         * @param builder MLIR operation builder for creating new operations
         * @param op The original operation being patched
         * @param contract_func The contract function to be called
         * @param contract Contract information containing argument specifications
         * @param args Output vector to store the prepared arguments
         */
        void prepare_contract_call_arguments(
            mlir::OpBuilder &builder, mlir::Operation *op, cir::FuncOp contract_func,
            const ContractInformation &contract, llvm::SmallVector< mlir::Value > &args
        );

        /**
         * @brief Applies a patch before the target operation.
         *
         * This method inserts a call to the patch function immediately before the target
         * operation. It handles module symbol merging, argument preparation, and call creation.
         * The inserted call is added to the inline worklist if inlining is enabled.
         *
         * @param op The target operation to be instrumented
         * @param patch The patch information containing the patch function details
         * @param patch_module The module containing the patch function
         * @param inline_patches Whether or not to inline at application.
         */
        void apply_before_patch(
            mlir::Operation *op, const PatchInformation &patch, mlir::ModuleOp patch_module,
            bool inline_patches
        );

        /**
         * @brief Applies a patch after the target operation.
         *
         * This method inserts a call to the patch function immediately after the target
         * operation. It handles module symbol merging, argument preparation, and call creation.
         * The inserted call is added to the inline worklist if inlining is enabled.
         *
         * @param op The target operation to be instrumented
         * @param patch The patch information containing the patch function details
         * @param patch_module The module containing the patch function
         */
        void apply_after_patch(
            mlir::Operation *op, const PatchInformation &patch, mlir::ModuleOp patch_module,
            bool inline_patches
        );

        /**
         * @brief Replaces a function call with a patch function call.
         *
         * This method completely replaces the original function call with a call to the
         * patch function. It preserves the original call's arguments and return types
         * while redirecting the call to the patch function.
         *
         * @param call_op The original call operation to be replaced
         * @param patch The patch information containing the replacement function details
         * @param patch_module The module containing the patch function
         */
        void replace_call(
            cir::CallOp op, const PatchInformation &patch, mlir::ModuleOp patch_module,
            bool inline_patches
        );

        /**
         * @brief Applies a contract before the target function.
         *
         * This method inserts a call to the contract function immediately before the target
         * operation. It handles module symbol merging, argument preparation, and call creation.
         * The inserted call is added to the inline worklist if inlining is enabled.
         *
         * @param target_op The target function to be instrumented
         * @param contract The contract information containing the contract function details
         * @param contract_module The module containing the contract function
         * @param should_inline Whether or not to inline at application.
         */
        void apply_contract_before(
            mlir::Operation *target_op, const ContractInformation &contract,
            mlir::ModuleOp contract_module, bool should_inline
        );

        /**
         * @brief Applies a contract after the target function.
         *
         * This method inserts a call to the contract function immediately after the target
         * operation. It handles module symbol merging, argument preparation, and call creation.
         * The inserted call is added to the inline worklist if inlining is enabled.
         *
         * @param op The target function to be instrumented
         * @param contract The contract information containing the contract function details
         * @param contract_module The module containing the contract function
         * @param should_inline Whether or not to inline at application.
         */
        void apply_contract_after(
            mlir::Operation *target_op, const ContractInformation &contract,
            mlir::ModuleOp contract_module, bool should_inline
        );

        /** todo (kaoudis) still thinking about whether this makes sense
         * @brief Applies a contract directly after the target function entrypoint,
         * just "inside" the entrypoint, before the rest of the original function.
         *
         * This method handles module symbol merging, argument preparation, and call creation.
         * The inserted call is added to the inline worklist if inlining is enabled.
         *
         * @param op The target function to be instrumented
         * @param contract The contract information containing the contract function details
         * @param contract_module The module containing the contract function
         * @param should_inline Whether or not to inline at application.
         */
        void apply_contract_at_entrypoint(
            cir::CallOp call_op, const ContractInformation &contract,
            mlir::ModuleOp contract_module, bool should_inline
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
         * @param dest The destination module to merge symbols into
         * @param src The source module containing the symbol to merge
         * @param symbol_name The name of the symbol to merge
         * @return mlir::LogicalResult Success or failure of the merge operation
         */
        mlir::LogicalResult merge_module_symbol(
            mlir::ModuleOp dest, mlir::ModuleOp src, const std::string &symbol_name
        );

        /**
         * @brief Sets appropriate attributes for the instrumentation call operation.
         * This can be a call to a patch or a contract.
         *
         * This function handles setting attributes on the call based on the
         * type of the original operation being instrumented.
         *
         * @param patch_call_op The patch call operation to set attributes on
         * @param target_op The original operation being instrumented
         */
        void set_patch_call_attributes(cir::CallOp patch_call_op, mlir::Operation *target_op);

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
            llvm::MapVector< mlir::Value, mlir::Value > &arg_map
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
            llvm::MapVector< mlir::Value, mlir::Value > &arg_map
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
