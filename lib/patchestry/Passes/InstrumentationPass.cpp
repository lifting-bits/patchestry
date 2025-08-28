/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <string_view>
#include <unordered_map>

#define GET_TYPEDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include <clang/AST/ASTContext.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendOptions.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/YAMLTraits.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LLVM.h>

#include <patchestry/Passes/InstrumentationPass.hpp>
#include <patchestry/Passes/OperationMatcher.hpp>
#include <patchestry/Passes/PatchSpec.hpp>
#include <patchestry/Util/Log.hpp>
#include <patchestry/YAML/YAMLParser.hpp>

namespace patchestry::passes {

    enum class TypeCategory : std::uint8_t {
        None,
        Boolean,
        Integer,
        Float,
        Pointer,
        Array,
        ComplexInt,
        ComplexFloat
    };

    std::optional< std::string >
    emitModuleAsString(const std::string &filename, const std::string &lang); // NOLINT

    namespace {
        /**
         * @brief Converts a string to a valid function name by replacing invalid characters.
         *
         * This function takes a string and converts it to a valid function name by replacing
         * any non-alphanumeric characters (except underscores) with underscores. This is used
         * to ensure patch function names are valid identifiers.
         *
         * @param str The input string to convert
         * @return std::string The converted function name
         */
        std::string namifyPatchFunction(const std::string &str) {
            std::string result;
            for (char c : str) {
                if ((isalnum(c) != 0) || c == '_') {
                    result += c;
                } else {
                    result += '_';
                }
            }
            return result;
        }

        /**
         * @brief Classifies an MLIR type into a category for cast kind determination.
         *
         * This function examines an MLIR type and classifies it into one of several categories
         * (Boolean, Integer, Float, Pointer, Array, ComplexInt, ComplexFloat, or None).
         * This classification is used to determine the appropriate cast kind when converting
         * between different types.
         *
         * @param ty The MLIR type to classify
         * @return TypeCategory The category of the type
         */
        TypeCategory classifyType(mlir::Type ty) {
            if (!ty) {
                return TypeCategory::None;
            }

            if (mlir::isa< cir::BoolType >(ty)) {
                return TypeCategory::Boolean;
            }
            if (mlir::isa< cir::IntType >(ty)) {
                return TypeCategory::Integer;
            }
            if (mlir::isa< cir::SingleType >(ty)) {
                return TypeCategory::Float;
            }
            if (mlir::isa< cir::PointerType >(ty)) {
                return TypeCategory::Pointer;
            }
            if (mlir::isa< cir::ArrayType >(ty)) {
                return TypeCategory::Array;
            }
            if (mlir::isa< cir::ComplexType >(ty)) {
                auto elem_ty = mlir::cast< cir::ComplexType >(ty).getElementTy();
                if (mlir::isa< cir::SingleType >(elem_ty)) {
                    return TypeCategory::ComplexFloat;
                }
                if (mlir::isa< cir::IntType >(elem_ty)) {
                    return TypeCategory::ComplexInt;
                }
            }
            return TypeCategory::None;
        }

        /**
         * @brief Determines the appropriate cast kind between two MLIR types.
         *
         * This function determines the appropriate cast kind between two MLIR types based on
         * their categories. It uses the classifyType function to determine the category of
         * each type and then uses the appropriate cast kind based on the categories.
         *
         * @param from The source MLIR type
         * @param to The destination MLIR type
         * @return cir::CastKind The appropriate cast kind
         */
        cir::CastKind getCastKind(mlir::Type from, mlir::Type to) {
            auto from_category = classifyType(from);
            auto to_category   = classifyType(to);

            if (from_category == TypeCategory::Integer && to_category == TypeCategory::Integer)
            {
                return cir::CastKind::integral;
            }
            if (from_category == TypeCategory::Integer || to_category == TypeCategory::Boolean)
            {
                return cir::CastKind::int_to_bool;
            }
            if (from_category == TypeCategory::Boolean || to_category == TypeCategory::Integer)
            {
                return cir::CastKind::bool_to_int;
            }
            if (from_category == TypeCategory::Float && to_category == TypeCategory::Float) {
                return cir::CastKind::floating;
            }
            if (from_category == TypeCategory::Float && to_category == TypeCategory::Integer) {
                return cir::CastKind::float_to_int;
            }
            if (from_category == TypeCategory::Integer && to_category == TypeCategory::Float) {
                return cir::CastKind::int_to_float;
            }
            if (from_category == TypeCategory::Float && to_category == TypeCategory::Boolean) {
                return cir::CastKind::float_to_bool;
            }
            if (from_category == TypeCategory::Boolean && to_category == TypeCategory::Float) {
                return cir::CastKind::bool_to_float;
            }
            if (from_category == TypeCategory::Pointer && to_category == TypeCategory::Pointer)
            {
                return cir::CastKind::bitcast;
            }

            if (from_category == TypeCategory::Pointer && to_category == TypeCategory::Integer)
            {
                return cir::CastKind::ptr_to_int;
            }
            if (from_category == TypeCategory::Pointer && to_category == TypeCategory::Boolean)
            {
                return cir::CastKind::ptr_to_bool;
            }
            if (from_category == TypeCategory::Integer && to_category == TypeCategory::Pointer)
            {
                return cir::CastKind::int_to_ptr;
            }

            if (from_category == TypeCategory::ComplexInt
                && to_category == TypeCategory::ComplexFloat)
            {
                return cir::CastKind::int_complex_to_float_complex;
            }
            if (from_category == TypeCategory::ComplexFloat
                && to_category == TypeCategory::ComplexInt)
            {
                return cir::CastKind::float_complex_to_int_complex;
            }
            if (from_category == TypeCategory::ComplexInt
                && to_category == TypeCategory::ComplexInt)
            {
                return cir::CastKind::int_complex;
            }
            if (from_category == TypeCategory::ComplexFloat
                && to_category == TypeCategory::ComplexFloat)
            {
                return cir::CastKind::float_complex;
            }
            if (from_category == TypeCategory::Pointer && to_category == TypeCategory::Pointer)
            {
                auto from_ptr = mlir::dyn_cast< cir::PointerType >(from);
                auto to_ptr   = mlir::dyn_cast< cir::PointerType >(to);
                if (from_ptr && to_ptr && from_ptr.getAddrSpace() != to_ptr.getAddrSpace()) {
                    return cir::CastKind::address_space;
                }
                return cir::CastKind::bitcast;
            }

            if (from_category == TypeCategory::Array && to_category == TypeCategory::Pointer) {
                return cir::CastKind::array_to_ptrdecay;
            }
            if ((from_category == TypeCategory::Integer || from_category == TypeCategory::Float)
                && (to_category == TypeCategory::ComplexInt
                    || to_category == TypeCategory::ComplexFloat))
            {
                return (from_category == TypeCategory::Float) ? cir::CastKind::float_to_complex
                                                              : cir::CastKind::int_to_complex;
            }

            if (from_category == TypeCategory::ComplexInt
                && to_category == TypeCategory::Integer)
            {
                return cir::CastKind::int_complex_to_real;
            }

            if (from_category == TypeCategory::ComplexFloat
                && to_category == TypeCategory::Float)
            {
                return cir::CastKind::float_complex_to_real;
            }

            return cir::CastKind::bitcast;
        }

    } // namespace

    /**
     * @brief Creates a new instance of the InstrumentationPass.
     *
     * Factory function that creates and returns a unique pointer to an InstrumentationPass
     * instance. The pass will apply patches according to the specifications in the provided
     * spec_file and use the given inline options for controlling inlining behavior.
     *
     * @param spec_file Path to the YAML patch specification file
     * @param inline_options Configuration options for controlling function inlining behavior
     * @return std::unique_ptr<mlir::Pass> A unique pointer to the created InstrumentationPass
     */
    std::unique_ptr< mlir::Pass >
    createInstrumentationPass(const std::string &spec_file, const PatchOptions &patch_options) {
        return std::make_unique< InstrumentationPass >(spec_file, patch_options);
    }

    template< typename T >
    std::optional< T > lookup_by_name(const std::vector< T > &items, const std::string &name) {
        for (const auto &item : items) {
            if (item.name == name) {
                return item;
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Gets an item from the library based on the name.
     *
     * @param items The vector of items to search through
     * @param name The name of the item to search for
     * @return std::optional<T> The item if found, std::nullopt otherwise
     */
    template< typename T >
    std::optional< T > lookup(const std::vector< T > &items, const std::string &name) {
        for (const auto &item : items) {
            if (item.id == name) {
                return item;
            }
        }
        return lookup_by_name(items, name);
    }

    /**
     * @brief Constructs an InstrumentationPass with the given specification file and options.
     *
     * The constructor loads and parses the patch specification file, validates patch files,
     * and prepares the pass for execution. If the specification file cannot be loaded or
     * parsed, appropriate error messages are logged.
     *
     * @param spec Path to the YAML patch specification file
     * @param patch_options Reference to inlining configuration options
     */
    InstrumentationPass::InstrumentationPass(
        std::string spec_, const PatchOptions &patch_options_
    )
        : spec_file(std::move(spec_)), patch_options(patch_options_) {
        patchestry::yaml::YAMLParser parser;
        PatchSpecContext::getInstance().set_spec_path(spec_file);
        if (!parser.validate_yaml_file< patchestry::passes::PatchConfiguration >(spec_file)) {
            LOG(ERROR) << "Error: Failed to parse patch specification file: " << spec_file
                       << "\n";
            return;
        }

        auto buffer_or_err = llvm::MemoryBuffer::getFile(spec_file);
        if (!buffer_or_err) {
            LOG(ERROR) << "Error: Failed to read patch specification file: " << spec_file
                       << "\n";
            return;
        }

        auto config_or_err = patchestry::yaml::utils::loadPatchConfiguration(
            llvm::sys::path::filename(spec_file).str()
        );
        if (!config_or_err) {
            LOG(ERROR) << "Error: Failed to parse patch specification file: " << spec_file
                       << "\n";
            return;
        }

        config = std::move(config_or_err.value());
        for (auto &spec : config->libraries.patches.patches) {
            auto patches_file_path =
                PatchSpecContext::getInstance().resolve_path(spec.implementation.code_file);
            if (!llvm::sys::fs::exists(patches_file_path)) {
                LOG(ERROR) << "Patch file " << patches_file_path << " does not exist\n";
                continue;
            }

            auto patch_file_path =
                PatchSpecContext::getInstance().resolve_path(spec.implementation.code_file);
            spec.patch_module = emitModuleAsString(patch_file_path, config->target.arch);
            if (!spec.patch_module) {
                LOG(ERROR) << "Failed to emit patch module for " << spec.name << "\n";
                continue;
            }
        }
        (void) patch_options;
    }

    /**
     * @brief Applies instrumentation to the MLIR module based on the patch specifications.
     *        The function follows the execution order and applies meta patches and contracts.
     *
     * @note The function list in module can grow during instrumentation. We collect the
     * list of functions before starting the instrumentation process to avoid issues with
     *       growing functions.
     */
    void InstrumentationPass::runOnOperation() {
        mlir::ModuleOp mod = getOperation();
        llvm::SmallVector< cir::FuncOp, 8 > function_worklist;
        llvm::SmallVector< mlir::Operation *, 8 > operation_worklist;

        // check if the configuration is loaded; if not, return
        if (!config) {
            LOG(ERROR) << "No patch configuration loaded. Skipping instrumentation.\n";
            return;
        }

        // gather all functions for later instrumentation
        mod.walk([&](cir::FuncOp op) { function_worklist.push_back(op); });

        // gather operations for instrumentation
        mod.walk([&](mlir::Operation *op) {
            if (!mlir::isa< cir::FuncOp, mlir::ModuleOp, cir::GlobalOp >(op)) {
                operation_worklist.push_back(op);
            }
        });

        // if the execution order is empty, apply meta patches and contracts in order
        if (config->execution_order.empty()) {
            for (const auto &meta_patch : config->meta_patches) {
                apply_meta_patches(function_worklist, operation_worklist, meta_patch.name);
            }
            for (const auto &meta_contract : config->meta_contracts) {
                apply_meta_contracts(function_worklist, operation_worklist, meta_contract.name);
            }
        } else {
            // process execution order if specified
            for (const auto &execution_item : config->execution_order) {
                // Parse execution item format: "meta_patches: name" or "meta_contracts: name"
                auto colon_pos = execution_item.find("::");
                if (colon_pos == std::string::npos) {
                    LOG(ERROR) << "Invalid execution order format: " << execution_item << "\n";
                    continue;
                }

                std::string type = execution_item.substr(0, colon_pos);
                std::string name = execution_item.substr(colon_pos + 2);

                // Trim whitespace
                name.erase(0, name.find_first_not_of(" \t"));
                name.erase(name.find_last_not_of(" \t") + 1);

                if (type == "meta_patches") {
                    apply_meta_patches(function_worklist, operation_worklist, name);
                } else if (type == "meta_contracts") {
                    apply_meta_contracts(function_worklist, operation_worklist, name);
                } else {
                    LOG(ERROR) << "Unknown execution type: " << type << "\n";
                }
            }

            // Inline inserted call operation
            // if (patch_options.enable_inlining) {
            for (auto *op : inline_worklists) {
                std::ignore = inline_call(mod, mlir::cast< cir::CallOp >(op));
            }

            // clear the worklist after inlining
            inline_worklists.clear();
            //}
        }
    }

    /**
     * @brief Applies meta patches in execution order.
     *
     * @param function_worklist List of functions to process
     * @param operation_worklist List of operations to process
     * @param meta_patch_name Name of the meta patch to apply
     */
    void InstrumentationPass::apply_meta_patches(
        llvm::SmallVector< cir::FuncOp, 8 > &function_worklist,
        llvm::SmallVector< mlir::Operation *, 8 > &operation_worklist,
        const std::string &meta_patch_name
    ) {
        auto target_meta_patch = lookup(config->meta_patches, meta_patch_name);

        if (!target_meta_patch) {
            LOG(ERROR) << "Meta patch '" << meta_patch_name << "' not found\n";
            return;
        }

        LOG(INFO) << "Applying meta patch: " << meta_patch_name << "\n";

        // Process each patch action in the meta patch
        for (const auto &patch_action : target_meta_patch->patch_actions) {
            LOG(INFO) << "Processing patch action: " << patch_action.action_id << "\n";

            auto &action = patch_action.action[0];

            // Find the corresponding patch specification by patch_id
            auto patch_spec = lookup(config->libraries.patches.patches, action.patch_id);
            if (!patch_spec) {
                LOG(ERROR) << "Patch specification for ID '" << action.patch_id
                           << "' not found\n";
                continue;
            }

            // Create a modified patch info with the action's mode and arguments
            PatchInformation patch_to_apply = { .spec         = patch_spec,
                                                .patch_action = patch_action };
            // Apply the patch to matching functions and operations
            apply_patch_action_to_targets(
                function_worklist, operation_worklist, *target_meta_patch, patch_to_apply
            );
        }
    }

    /**
     * @brief Applies meta contracts in execution order.
     *
     * @param function_worklist List of functions to process
     * @param operation_worklist List of operations to process
     * @param meta_contract_name Name of the meta contract to apply
     */
    void InstrumentationPass::apply_meta_contracts(
        llvm::SmallVector< cir::FuncOp, 8 > &function_worklist,
        llvm::SmallVector< mlir::Operation *, 8 > &operation_worklist,
        const std::string &meta_contract_name
    ) {
        // Find the meta contract by name
        auto target_meta_contract = lookup(config->meta_contracts, meta_contract_name);
        if (!target_meta_contract) {
            LOG(ERROR) << "Meta contract '" << meta_contract_name << "' not found\n";
            return;
        }

        LOG(INFO) << "Applying meta contract: " << meta_contract_name << "\n";

        // TODO: Implement meta contract application
        (void) function_worklist;
        (void) operation_worklist;
        (void) target_meta_contract;
    }

    /**
     * @brief Applies a specific patch action to target functions and operations.
     *
     * @param function_worklist List of functions to process
     * @param operation_worklist List of operations to process
     * @param patch_action The patch action containing match criteria
     * @param spec The patch specification to apply
     * @param modified_patch The modified patch info with action-specific settings
     */
    void InstrumentationPass::apply_patch_action_to_targets(
        llvm::SmallVector< cir::FuncOp, 8 > &function_worklist,
        llvm::SmallVector< mlir::Operation *, 8 > &operation_worklist,
        const MetaPatchConfig &meta_patch, const PatchInformation &patch_to_apply
    ) {
        const auto &patch_action = patch_to_apply.patch_action.value();
        const auto &match        = patch_action.match[0];
        const auto &action       = patch_action.action[0];

        if (match.kind == MatchKind::FUNCTION) {
            // Apply to function calls
            for (auto func : function_worklist) {
                func.walk([&](cir::CallOp call_op) {
                    // Create a temporary spec with the patch action match

                    if (OperationMatcher::matches(
                            call_op, func, patch_action, OperationMatcher::Mode::FUNCTION
                        ))
                    {
                        auto patch_module = load_patch_module(
                            *call_op->getContext(), *patch_to_apply.spec->patch_module
                        );
                        if (!patch_module) {
                            LOG(ERROR) << "Failed to load patch module for function: "
                                       << call_op.getCallee()->str() << "\n";
                            return;
                        }

                        switch (action.mode) {
                            case PatchInfoMode::APPLY_BEFORE:
                                apply_before_patch(
                                    call_op, patch_to_apply, patch_module.get(),
                                    meta_patch.optimization.contains("inline-patches")
                                );
                                break;
                            case PatchInfoMode::APPLY_AFTER:
                                apply_after_patch(
                                    call_op, patch_to_apply, patch_module.get(),
                                    meta_patch.optimization.contains("inline-patches")
                                );
                                break;
                            case PatchInfoMode::REPLACE:
                                replace_call(
                                    call_op, patch_to_apply, patch_module.get(),
                                    meta_patch.optimization.contains("inline-patches")
                                );
                                break;
                            default:
                                LOG(ERROR) << "Unsupported patch mode for function call\n";
                                break;
                        }
                    }
                });
            }
        } else if (match.kind == MatchKind::OPERATION) {
            // Apply to operations
            for (auto *op : operation_worklist) {
                auto func = op->getParentOfType< cir::FuncOp >();
                if (!func) {
                    continue;
                }

                if (OperationMatcher::matches(
                        op, func, patch_action, OperationMatcher::Mode::OPERATION
                    ))
                {
                    auto patch_module = load_patch_module(
                        *op->getContext(), *patch_to_apply.spec->patch_module
                    );
                    if (!patch_module) {
                        LOG(ERROR) << "Failed to load patch module for operation: "
                                   << op->getName().getStringRef().str() << "\n";
                        continue;
                    }

                    switch (action.mode) {
                        case PatchInfoMode::APPLY_BEFORE:
                            apply_before_patch(
                                op, patch_to_apply, patch_module.get(),
                                meta_patch.optimization.contains("inline-patches")
                            );
                            break;
                        default:
                            LOG(ERROR) << "Unsupported patch mode for operation: "
                                       << op->getName().getStringRef().str() << "\n";
                            break;
                    }
                }
            }
        }
    }

    /**
     * @brief Prepares the arguments for a function call based on the patch information.
     *        This function handles argument type casting and argument matching using
     *        the new structured ArgumentSource specifications.
     *
     * @param builder The MLIR operation builder.
     * @param op The call operation to be instrumented.
     * @param patch_func The function to be called as a patch.
     * @param patch The patch information.
     * @param args The vector to store the prepared arguments.
     */
    mlir::Value InstrumentationPass::create_cast_if_needed(
        mlir::OpBuilder &builder, mlir::Operation *call_op, mlir::Value value,
        mlir::Type target_type
    ) {
        if (value.getType() == target_type) {
            return value;
        }

        auto cast_op = builder.create< cir::CastOp >(
            call_op->getLoc(), target_type, getCastKind(value.getType(), target_type), value
        );
        return cast_op->getResults().front();
    }

    mlir::Value InstrumentationPass::create_reference(
        mlir::OpBuilder &builder, mlir::Operation *call_op, mlir::Value value
    ) {
        auto abi_align = call_op->getAttrOfType< mlir::IntegerAttr >("abi_align");
        auto addr_type = cir::PointerType::get(builder.getContext(), value.getType());
        auto addr_op   = builder.create< cir::AllocaOp >(
            call_op->getLoc(), addr_type, value.getType(), "", abi_align
        );
        builder.create< cir::StoreOp >(call_op->getLoc(), value, addr_op.getResult());
        return addr_op.getResult();
    }

    void InstrumentationPass::prepare_call_arguments(
        mlir::OpBuilder &builder, mlir::Operation *call_op, cir::FuncOp patch_func,
        const PatchInformation &patch, llvm::DenseMap< mlir::Value, mlir::Value > &arg_map
    ) {
        const auto &patch_action = patch.patch_action.value();

        // Handle structured argument specifications
        for (size_t i = 0;
             i < patch_action.action[0].arguments.size() && i < patch_func.getNumArguments();
             ++i)
        {
            const auto &arg_spec = patch_action.action[0].arguments[i];
            auto patch_arg_type  = patch_func.getArgumentTypes()[i];

            switch (arg_spec.source) {
                case ArgumentSourceType::OPERAND:
                    handle_operand_argument(
                        builder, call_op, arg_spec, patch_arg_type, arg_map
                    );
                    break;
                case ArgumentSourceType::VARIABLE:
                    handle_variable_argument(
                        builder, call_op, arg_spec, patch_arg_type, arg_map
                    );
                    break;
                case ArgumentSourceType::SYMBOL:
                    handle_symbol_argument(builder, call_op, arg_spec, patch_arg_type, arg_map);
                    break;
                case ArgumentSourceType::RETURN_VALUE:
                    handle_return_value_argument(
                        builder, call_op, arg_spec, patch_arg_type, arg_map
                    );
                    break;
                case ArgumentSourceType::CONSTANT:
                    handle_constant_argument(
                        builder, call_op, arg_spec, patch_arg_type, arg_map
                    );
                    break;
            }
        }
    }

    void InstrumentationPass::handle_operand_argument(
        mlir::OpBuilder &builder, mlir::Operation *call_op, const ArgumentSource &arg_spec,
        mlir::Type patch_arg_type, llvm::DenseMap< mlir::Value, mlir::Value > &arg_map
    ) {
        if (!arg_spec.index.has_value()) {
            LOG(ERROR) << "OPERAND source requires index field\n";
            return;
        }
        unsigned idx = arg_spec.index.value();
        mlir::Value operand_value;

        if (auto orig_call_op = mlir::dyn_cast< cir::CallOp >(call_op)) {
            if (idx >= orig_call_op.getArgOperands().size()) {
                LOG(ERROR) << "Operand index " << idx << " out of range\n";
                return;
            }
            operand_value = orig_call_op.getArgOperands()[idx];
        } else {
            if (idx >= call_op->getNumOperands()) {
                LOG(ERROR) << "Operand index " << idx << " out of range\n";
                return;
            }
            operand_value = call_op->getOperand(idx);
        }

        if (arg_spec.is_reference) {
            arg_map[operand_value] = create_cast_if_needed(
                builder, call_op, create_reference(builder, call_op, operand_value),
                patch_arg_type
            );
        } else {
            arg_map[operand_value] =
                create_cast_if_needed(builder, call_op, operand_value, patch_arg_type);
        }
    }

    void InstrumentationPass::handle_variable_argument(
        mlir::OpBuilder &builder, mlir::Operation *call_op, const ArgumentSource &arg_spec,
        mlir::Type patch_arg_type, llvm::DenseMap< mlir::Value, mlir::Value > &arg_map
    ) {
        if (!arg_spec.symbol.has_value()) {
            LOG(ERROR) << "VARIABLE source requires symbol field\n";
            return;
        }

        const std::string &var_name = arg_spec.symbol.value();
        auto variable_ref           = find_local_variable(call_op, var_name);
        if (!variable_ref.has_value()) {
            LOG(WARNING) << "Local variable '" << var_name << "' not found\n";
            return;
        }

        mlir::Value variable_reference = variable_ref.value();
        if (arg_spec.is_reference) {
            arg_map[variable_reference] =
                create_cast_if_needed(builder, call_op, variable_reference, patch_arg_type);
        } else {
            auto load_op = builder.create< cir::LoadOp >(
                call_op->getLoc(), variable_reference, /*isDeref=*/true, /*isVolatile=*/false,
                /*alignment=*/mlir::IntegerAttr{}, /*mem_order=*/cir::MemOrderAttr{},
                /*tbaa=*/mlir::ArrayAttr{}
            );
            arg_map[variable_reference] =
                create_cast_if_needed(builder, call_op, load_op, patch_arg_type);
        }
    }

    void InstrumentationPass::handle_symbol_argument(
        mlir::OpBuilder &builder, mlir::Operation *call_op, const ArgumentSource &arg_spec,
        mlir::Type patch_arg_type, llvm::DenseMap< mlir::Value, mlir::Value > &arg_map
    ) {
        if (!arg_spec.symbol.has_value()) {
            LOG(ERROR) << "SYMBOL source requires symbol field\n";
            return;
        }

        const std::string &symbol_name = arg_spec.symbol.value();
        auto symbol_ref                = find_global_symbol(builder, call_op, symbol_name);
        if (!symbol_ref.has_value()) {
            LOG(WARNING) << "Symbol '" << symbol_name << "' not found in symbol table\n";
            return;
        }

        mlir::Value symbol_reference = symbol_ref.value();
        if (arg_spec.is_reference) {
            arg_map[symbol_reference] =
                create_cast_if_needed(builder, call_op, symbol_reference, patch_arg_type);
        } else {
            auto load_op = builder.create< cir::LoadOp >(
                call_op->getLoc(), symbol_reference, /*isDeref=*/true, /*isVolatile=*/false,
                /*alignment=*/mlir::IntegerAttr{}, /*mem_order=*/cir::MemOrderAttr{},
                /*tbaa=*/mlir::ArrayAttr{}
            );
            arg_map[symbol_reference] =
                create_cast_if_needed(builder, call_op, load_op, patch_arg_type);
        }
    }

    void InstrumentationPass::handle_return_value_argument(
        mlir::OpBuilder &builder, mlir::Operation *call_op, const ArgumentSource &arg_spec,
        mlir::Type patch_arg_type, llvm::DenseMap< mlir::Value, mlir::Value > &arg_map
    ) {
        if (call_op->getNumResults() == 0) {
            LOG(ERROR) << "Operation/function does not have a return value\n";
            return;
        }

        auto arg_value = call_op->getResult(0);
        if (arg_spec.is_reference) {
            arg_map[arg_value] = create_cast_if_needed(
                builder, call_op, create_reference(builder, call_op, arg_value), patch_arg_type
            );
        } else {
            arg_map[arg_value] =
                create_cast_if_needed(builder, call_op, arg_value, patch_arg_type);
        }
    }

    void InstrumentationPass::handle_constant_argument(
        mlir::OpBuilder &builder, mlir::Operation *call_op, const ArgumentSource &arg_spec,
        mlir::Type patch_arg_type, llvm::DenseMap< mlir::Value, mlir::Value > &arg_map
    ) {
        if (!arg_spec.value.has_value()) {
            LOG(ERROR) << "CONSTANT source requires value field\n";
            return;
        }

        const std::string &const_value = arg_spec.value.value();
        mlir::Value arg_value;

        arg_value = parse_constant_operand(builder, call_op, const_value, patch_arg_type);
        if (!arg_value) {
            return;
        }

        arg_map[arg_value] = create_cast_if_needed(builder, call_op, arg_value, patch_arg_type);
    }

    mlir::Value InstrumentationPass::parse_constant_operand(
        mlir::OpBuilder &builder, mlir::Operation *call_op, const std::string &value,
        mlir::Type target_type
    ) {
        if (auto int_type = mlir::dyn_cast< cir::IntType >(target_type)) {
            try {
                int64_t int_val = std::stoll(value, nullptr, 0); // Support hex, oct, dec
                auto attr       = cir::IntAttr::get(
                    cir::IntType::get(
                        builder.getContext(), int_type.getWidth(), int_type.isSigned()
                    ),
                    llvm::APSInt(int_type.getWidth(), int_val)
                );
                return builder.create< cir::ConstantOp >(call_op->getLoc(), int_type, attr);
            } catch (const std::exception &e) {
                LOG(ERROR) << "Failed to parse integer constant '" << value << "': " << e.what()
                           << "\n";
                return nullptr;
            }
        } else if (auto ptr_type = mlir::dyn_cast< cir::PointerType >(target_type)) {
            try {
                uint64_t ptr_val = std::stoull(value, nullptr, 0);
                auto int_type    = cir::IntType::get(builder.getContext(), 64, false);
                auto int_attr    = cir::IntAttr::get(
                    cir::IntType::get(
                        builder.getContext(), int_type.getWidth(), int_type.isSigned()
                    ),
                    llvm::APSInt(int_type.getWidth(), ptr_val)
                );
                auto int_const =
                    builder.create< cir::ConstantOp >(call_op->getLoc(), int_type, int_attr);
                return builder.create< cir::CastOp >(
                    call_op->getLoc(), ptr_type, cir::CastKind::int_to_ptr, int_const
                );
            } catch (const std::exception &e) {
                LOG(ERROR) << "Failed to parse pointer constant '" << value << "': " << e.what()
                           << "\n";
                return nullptr;
            }
        } else {
            LOG(ERROR) << "Unsupported constant type for value '" << value << "'\n";
            return nullptr;
        }
    }

    std::optional< mlir::Value > InstrumentationPass::find_local_variable(
        mlir::Operation *call_op, const std::string &var_name
    ) {
        auto func = call_op->getParentOfType< cir::FuncOp >();
        if (!func) {
            LOG(ERROR) << "Cannot find parent function for local variable lookup\n";
            return std::nullopt;
        }

        mlir::Value variable_reference;
        bool found = false;

        func.walk([&](mlir::Operation *op) {
            if (auto alloca_op = mlir::dyn_cast< cir::AllocaOp >(op)) {
                if (auto name_attr = op->getAttrOfType< mlir::StringAttr >("name")) {
                    if (name_attr.getValue() == var_name) {
                        variable_reference = alloca_op.getResult();
                        found              = true;
                        return mlir::WalkResult::interrupt();
                    }
                }
            }
            return mlir::WalkResult::advance();
        });

        return found ? std::optional< mlir::Value >(variable_reference) : std::nullopt;
    }

    std::optional< mlir::Value > InstrumentationPass::find_global_symbol(
        mlir::OpBuilder &builder, mlir::Operation *call_op, const std::string &symbol_name
    ) {
        auto module = call_op->getParentOfType< mlir::ModuleOp >();
        if (!module) {
            LOG(ERROR) << "Cannot find parent module for symbol lookup\n";
            return std::nullopt;
        }

        // Look for global variables
        if (auto global_op = module.lookupSymbol< cir::GlobalOp >(symbol_name)) {
            auto global_type = global_op.getSymType();
            if (auto global_ptr_type = mlir::dyn_cast< cir::PointerType >(global_type)) {
                return builder.create< cir::GetGlobalOp >(
                    call_op->getLoc(), global_ptr_type, symbol_name
                );
            } else {
                auto ptr_type = cir::PointerType::get(builder.getContext(), global_type);
                return builder.create< cir::GetGlobalOp >(
                    call_op->getLoc(), ptr_type, symbol_name
                );
            }
        }

        // Look for functions
        if (auto func_op = module.lookupSymbol< cir::FuncOp >(symbol_name)) {
            auto func_type     = func_op.getFunctionType();
            auto func_ptr_type = cir::PointerType::get(builder.getContext(), func_type);
            auto symbol_ref = mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol_name);
            return builder.create< cir::GetGlobalOp >(
                call_op->getLoc(), func_ptr_type, symbol_ref
            );
        }

        return std::nullopt;
    }

    /**
     * @brief Applies a patch before the function call. This function inserts a call to the
     * patch function before the original function call.
     *
     * @param op The call operation to be instrumented.
     * @param match The match information for the function call.
     * @param patch The patch information.
     * @param patch_module The module containing the patch function.
     */
    void InstrumentationPass::apply_before_patch(
        mlir::Operation *target_op, const PatchInformation &patch, mlir::ModuleOp patch_module,
        bool inline_patches
    ) {
        if (target_op == nullptr) {
            LOG(ERROR) << "Patch before: Operation is null";
            return;
        }

        const auto &patch_action = patch.patch_action.value();
        const auto &patch_spec   = patch.spec.value();
        (void) patch_action; // Suppress unused warning

        mlir::OpBuilder builder(target_op);
        builder.setInsertionPoint(target_op);
        auto module = target_op->getParentOfType< mlir::ModuleOp >();

        std::string patch_function_name =
            namifyPatchFunction(patch_spec.implementation.function_name);
        auto input_types = llvm::to_vector(target_op->getOperandTypes());
        if (!patch_module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        // check if the patch function is already in the module, if not, merge it
        if (!module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            auto result = merge_module_symbol(module, patch_module, patch_function_name);
            if (mlir::failed(result)) {
                LOG(ERROR) << "Failed to insert symbol into module\n";
                return;
            }
        } else {
            LOG(INFO) << "Patch function " << patch_function_name
                      << " already exists in module, skipping merge\n";
        }

        auto patch_func = module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!patch_func) {
            LOG(ERROR) << "Patch function " << patch_function_name
                       << " not defined. Patching failed...\n";
            return;
        }

        auto symbol_ref =
            mlir::FlatSymbolRefAttr::get(target_op->getContext(), patch_function_name);
        llvm::DenseMap< mlir::Value, mlir::Value > function_args_map;
        prepare_call_arguments(builder, target_op, patch_func, patch, function_args_map);
        llvm::SmallVector< mlir::Value > new_function_args;
        for (auto &[old_arg, new_arg] : function_args_map) {
            new_function_args.push_back(new_arg);
        }

        auto patch_call_op = builder.create< cir::CallOp >(
            target_op->getLoc(), symbol_ref,
            patch_func->getResultTypes().size() != 0 ? patch_func->getResultTypes().front()
                                                     : mlir::Type(),
            new_function_args
        );

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(patch_call_op);
        mlir::DominanceInfo DT(patch_call_op->getParentOfType< mlir::FunctionOpInterface >());

        unsigned arg_index = 0;
        // Replace all uses of old arguments with new arguments after patch function call
        for (auto &[old_arg, new_arg] : function_args_map) {
            auto arg_spec = patch_action.action[0].arguments[arg_index];
            if (arg_spec.is_reference && arg_spec.source == ArgumentSourceType::OPERAND) {
                auto load_op = builder.create< cir::LoadOp >(
                    old_arg.getLoc(), new_arg, /*isDeref=*/true, /*isVolatile=*/false,
                    /*alignment=*/mlir::IntegerAttr{}, /*mem_order=*/cir::MemOrderAttr{},
                    /*tbaa=*/mlir::ArrayAttr{}
                );
                auto new_value =
                    create_cast_if_needed(builder, target_op, load_op, old_arg.getType());
                old_arg.replaceUsesWithIf(new_value, [&DT, &new_value](mlir::OpOperand &use) {
                    return DT.dominates(new_value, use.getOwner());
                });
            }
            arg_index++;
        }

        // Set appropriate attributes based on operation type
        set_patch_call_attributes(patch_call_op, target_op);

        if (inline_patches) {
            inline_worklists.push_back(patch_call_op);
        }
    }

    /**
     * @brief Applies a patch after the target operation.
     *
     * @param op The target operation to be instrumented
     * @param patch The patch information containing the patch function details
     * @param patch_module The module containing the patch function
     */
    void InstrumentationPass::apply_after_patch(
        mlir::Operation *target_op, const PatchInformation &patch, mlir::ModuleOp patch_module,
        bool inline_patches
    ) {
        if (target_op == nullptr) {
            LOG(ERROR) << "Patch after: Operation is null";
            return;
        }

        const auto &patch_action = patch.patch_action.value();
        const auto &patch_spec   = patch.spec.value();
        (void) patch_action; // Suppress unused warning

        mlir::OpBuilder builder(target_op);
        auto module = target_op->getParentOfType< mlir::ModuleOp >();
        builder.setInsertionPointAfter(target_op);

        std::string patch_function_name =
            namifyPatchFunction(patch_spec.implementation.function_name);
        auto input_types = llvm::to_vector(target_op->getResultTypes());
        if (!patch_module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        // check if the patch function is already in the module, if not, merge it
        if (!module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            auto result = merge_module_symbol(module, patch_module, patch_function_name);
            if (mlir::failed(result)) {
                LOG(ERROR) << "Failed to insert symbol into module\n";
                return;
            }
        } else {
            LOG(INFO) << "Patch function " << patch_function_name
                      << " already exists in module, skipping merge\n";
        }

        auto patch_func = module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!patch_func) {
            LOG(ERROR) << "Patch function " << patch_function_name
                       << " not defined. Patching failed...\n";
            return;
        }

        auto symbol_ref =
            mlir::FlatSymbolRefAttr::get(target_op->getContext(), patch_function_name);
        llvm::DenseMap< mlir::Value, mlir::Value > function_args_map;
        prepare_call_arguments(builder, target_op, patch_func, patch, function_args_map);
        llvm::SmallVector< mlir::Value > function_args;
        for (auto &[old_arg, new_arg] : function_args_map) {
            function_args.push_back(new_arg);
        }
        auto patch_call_op = builder.create< cir::CallOp >(
            target_op->getLoc(), symbol_ref,
            patch_func->getResultTypes().size() != 0 ? patch_func->getResultTypes().front()
                                                     : mlir::Type(),
            function_args
        );

        // Set appropriate attributes based on operation type
        set_patch_call_attributes(patch_call_op, target_op);

        if (inline_patches) {
            inline_worklists.push_back(patch_call_op);
        }
    }

    /**
     * @brief Replaces the function call with a patch function. This function replaces the
     * original function call with a call to the patch function.
     *
     * @param op The call operation to be instrumented.
     * @param match The match information for the function call.
     * @param patch The patch information.
     * @param patch_module The module containing the patch function.
     */

    void InstrumentationPass::replace_call(
        cir::CallOp call_op, const PatchInformation &patch, mlir::ModuleOp patch_module,
        bool inline_patches
    ) {
        mlir::OpBuilder builder(call_op);
        auto loc    = call_op.getLoc();
        auto *ctx   = call_op->getContext();
        auto module = call_op->getParentOfType< mlir::ModuleOp >();
        assert(module && "Wrap around patch: no module found");

        builder.setInsertionPoint(call_op);

        const auto &patch_spec = patch.spec.value();

        auto callee_name = call_op.getCallee()->str();
        assert(!callee_name.empty() && "Wrap around patch: callee name is empty");

        auto patch_function_name = namifyPatchFunction(patch_spec.implementation.function_name);
        auto result_types        = llvm::to_vector(call_op.getResultTypes());

        if (!patch_module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        // check if the patch function is already in the module, if not, merge it
        if (!module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            auto result = merge_module_symbol(module, patch_module, patch_function_name);
            if (mlir::failed(result)) {
                LOG(ERROR) << "Failed to insert symbol into module\n";
                return;
            }
        } else {
            LOG(INFO) << "Patch function " << patch_function_name
                      << " already exists in module, skipping merge\n";
        }

        auto wrap_func = module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!wrap_func) {
            LOG(ERROR) << "Wrap around patch: patch function "
                       << patch_spec.implementation.function_name
                       << " not defined. Patching failed...\n";
            return;
        }

        auto wrap_func_ref = mlir::FlatSymbolRefAttr::get(ctx, patch_function_name);
        auto wrap_call_op  = builder.create< cir::CallOp >(
            loc, wrap_func_ref, result_types.size() != 0 ? result_types.front() : mlir::Type(),
            call_op.getArgOperands()
        );

        // Set appropriate attributes based on operation type
        set_patch_call_attributes(wrap_call_op, call_op);

        call_op.replaceAllUsesWith(wrap_call_op);
        call_op.erase();

        if (inline_patches) {
            inline_worklists.push_back(wrap_call_op);
        }
    }

    /**
     * @brief Sets appropriate attributes for the patch call operation.
     *
     * This function handles setting attributes on the patch call based on the
     * type of the original operation being instrumented.
     *
     * @param patch_call_op The patch call operation to set attributes on
     * @param target_op The original operation being instrumented
     */
    void InstrumentationPass::set_patch_call_attributes(
        cir::CallOp patch_call_op, mlir::Operation *target_op
    ) {
        if (auto orig_call_op = mlir::dyn_cast< cir::CallOp >(target_op)) {
            // For CallOp operations, preserve the original extra attributes
            patch_call_op->setAttr("extra_attrs", orig_call_op.getExtraAttrs());
        } else {
            // For non-CallOp operations, create empty extra attributes
            mlir::NamedAttrList empty;
            patch_call_op->setAttr(
                "extra_attrs",
                cir::ExtraFuncAttributesAttr::get(
                    target_op->getContext(), empty.getDictionary(target_op->getContext())
                )
            );
        }

        // Add operation-specific attributes for debugging
        patch_call_op->setAttr(
            "patched_operation",
            mlir::StringAttr::get(target_op->getContext(), target_op->getName().getStringRef())
        );
    }

    /**
     * @brief Loads a patch module from a string representation.
     *
     * @param ctx The MLIR context.
     * @param patch_string The string representation of the patch module.
     * @return mlir::OwningOpRef< mlir::ModuleOp > The loaded patch module.
     */
    mlir::OwningOpRef< mlir::ModuleOp > InstrumentationPass::load_patch_module(
        mlir::MLIRContext &ctx, const std::string &patch_string
    ) {
        return mlir::parseSourceString< mlir::ModuleOp >(patch_string, &ctx);
    }

    /**
     * @brief Merges a symbol from a source module into a destination module.
     *
     * @param dest The destination module.
     * @param src The source module.
     * @param symbol_name The name of the symbol to be merged.
     * @return mlir::LogicalResult The result of the merge operation.
     */
    mlir::LogicalResult InstrumentationPass::merge_module_symbol(
        mlir::ModuleOp dest, mlir::ModuleOp src, const std::string &symbol_name
    ) {
        mlir::SymbolTable dest_sym_table(dest);
        mlir::SymbolTable src_sym_table(src);

        // Look up the specific symbol in the source module
        auto *src_symbol = src_sym_table.lookup(symbol_name);
        if (!src_symbol) {
            LOG(ERROR) << "Symbol " << symbol_name << " not found in source module\n";
            return mlir::failure();
        }

        // Function to check if a symbol is global (e.g., function declarations)
        auto is_global_symbol = [](mlir::Operation *op) {
            if (auto func = mlir::dyn_cast< cir::FuncOp >(op)) {
                return func.isDeclaration();
            }
            if (auto global = mlir::dyn_cast< cir::GlobalOp >(op)) {
                // Check if it's a private or dsolocal symbol
                return !(
                    global.getLinkage() == cir::GlobalLinkageKind::PrivateLinkage
                    || global.isDSOLocal()
                );
            }
            return false;
        };

        // First pass: collect all symbols that need to be copied
        std::vector< mlir::Operation * > symbols_to_copy;
        std::set< std::string > processed_symbols;

        // Create a symbol table collection for the source module
        mlir::SymbolTableCollection symbol_table_collection;
        symbol_table_collection.getSymbolTable(src);

        std::function< void(mlir::Operation *) > collect_symbols = [&](mlir::Operation *op) {
            if (auto sym_op = mlir::dyn_cast< mlir::SymbolOpInterface >(op)) {
                std::string sym_name = sym_op.getName().str();
                if (processed_symbols.count(sym_name)) {
#ifdef DEBUG
                    LOG(INFO) << "Skipping already processed symbol: " << sym_name << "\n";
#endif
                    return;
                }
                processed_symbols.insert(sym_name);

                // Check for conflicts
                if (dest_sym_table.lookup(sym_name)) {
                    if (is_global_symbol(op)) {
                        // For global symbols, keep the one in dest and warn
                        LOG(WARNING)
                            << "Global symbol " << sym_name
                            << " already exists in destination module, keeping existing\n";
                        return;
                    }
                    // For local symbols, rename
                    auto maybe_new_name = src_sym_table.renameToUnique(op, { &dest_sym_table });
                    if (mlir::failed(maybe_new_name)) {
                        LOG(ERROR) << "Failed to rename symbol " << sym_name << "\n";
                        return;
                    }
                    LOG(INFO) << "Renamed symbol: " << sym_name << " -> "
                              << maybe_new_name->getValue() << "\n";

                    // After renaming, we need to process the new symbol
                    if (auto new_sym = mlir::dyn_cast< mlir::SymbolOpInterface >(op)) {
                        std::string new_sym_name = new_sym.getName().str();
                        if (!processed_symbols.contains(new_sym_name)) {
                            processed_symbols.insert(new_sym_name);
                            symbols_to_copy.push_back(op);
                        }
                    }
                    return; // Don't process the old symbol further
                }

                symbols_to_copy.push_back(op);

                // Get all uses of this symbol in the module
                auto sym_name_attr = mlir::StringAttr::get(op->getContext(), sym_name);
#ifdef DEBUG
                LOG(INFO) << "Searching for uses of symbol: " << sym_name << "\n";
#endif
                auto uses = mlir::SymbolTable::getSymbolUses(sym_name_attr, src);
                if (uses) {
#ifdef DEBUG
                    LOG(INFO) << "Found " << std::distance(uses->begin(), uses->end())
                              << " uses\n";
#endif
                    for (const auto &use : *uses) {
#ifdef DEBUG
                        LOG(INFO)
                            << "  Use found in operation: " << use.getUser()->getName() << "\n";
                        LOG(INFO) << "    Location: " << use.getUser()->getLoc() << "\n";
                        LOG(INFO) << "    Symbol reference: " << use.getSymbolRef() << "\n";
#endif

                        // Look up the referenced symbol
                        if (auto *referenced = symbol_table_collection.lookupSymbolIn(
                                src, use.getSymbolRef().getRootReference()
                            ))
                        {
                            collect_symbols(referenced);
                        } else {
                            LOG(WARNING)
                                << "Could not find referenced symbol: "
                                << use.getSymbolRef().getRootReference().getValue() << "\n";
                        }
                    }
                }

                if (!op) {
                    LOG(WARNING) << "Operation is null, skipping nested reference walk\n";
                    return;
                }

                // Get the parent module to ensure it's still valid
                auto parent_module = op->getParentOfType< mlir::ModuleOp >();
                if (!parent_module) {
                    LOG(WARNING
                    ) << "Could not find parent module, skipping nested reference walk\n";
                    return;
                }

                op->walk([&](mlir::Operation *nested_op) {
                    if (!nested_op) {
                        LOG(WARNING) << "Found null nested operation, skipping\n";
                        return;
                    }

                    if (auto nested_sym_user =
                            mlir::dyn_cast< mlir::SymbolUserOpInterface >(nested_op))
                    {
                        // Get all symbol uses in this operation
                        auto uses = mlir::SymbolTable::getSymbolUses(nested_op);
                        if (uses) {
                            for (const auto &use : *uses) {
                                if (!use.getUser()) {
                                    LOG(WARNING) << "Found null symbol use, skipping\n";
                                    continue;
                                }
#ifdef DEBUG
                                LOG(INFO)
                                    << "  Found nested symbol reference: "
                                    << use.getSymbolRef().getRootReference().getValue() << "\n";
                                LOG(INFO)
                                    << "    In operation: " << nested_op->getName() << "\n";
                                LOG(INFO) << "    Location: " << nested_op->getLoc() << "\n";
#endif

                                if (auto *referenced = symbol_table_collection.lookupSymbolIn(
                                        src, use.getSymbolRef().getRootReference()
                                    ))
                                {
                                    collect_symbols(referenced);
                                } else {
                                    LOG(WARNING)
                                        << "Could not find referenced symbol: "
                                        << use.getSymbolRef().getRootReference().getValue()
                                        << "\n";
                                }
                            }
                        }
                    }
                });
            }
        };

        // Start with the requested symbol
        collect_symbols(src_symbol);

        // Second pass: copy all collected symbols
        for (auto *op : symbols_to_copy) {
            dest.push_back(op->clone());
        }

        return mlir::success();
    }

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
    mlir::LogicalResult
    InstrumentationPass::inline_call(mlir::ModuleOp module, cir::CallOp call_op) {
        mlir::OpBuilder builder(call_op);
        mlir::Location loc = call_op.getLoc();

        auto callee = mlir::dyn_cast< cir::FuncOp >(
            module.lookupSymbol< cir::FuncOp >(call_op.getCallee()->str())
        );
        if (!callee) {
            LOG(ERROR) << "Callee not found in module\n";
            return mlir::failure();
        }

        mlir::IRMapping mapper;
        auto callee_args   = callee.getArguments();
        auto call_operands = call_op.getArgOperands();

        // Ensure we don't have a size mismatch that could cause null dereference
        if (callee_args.size() != call_operands.size()) {
            LOG(ERROR) << "Argument count mismatch: callee expects " << callee_args.size()
                       << " arguments but call provides " << call_operands.size() << "\n";
            return mlir::failure();
        }

        for (auto [arg, operand] : llvm::zip(callee_args, call_operands)) {
            if (!arg || !operand) {
                LOG(ERROR) << "Null argument or operand encountered during inlining\n";
                return mlir::failure();
            }

            mapper.map(arg, operand);
        }

        // get caller block and split it at call site
        mlir::Block *caller_block = call_op->getBlock();
        mlir::Block *split_block  = caller_block->splitBlock(call_op->getIterator());

        // Note: Using of DenseMap is causing null-pointer dereference issue with ci.
        mlir::DenseMap< mlir::Block *, mlir::Block * > block_map;

        // First pass: clone all blocks (without operations)
        mlir::Region &callee_region = callee.getBody();
        for (mlir::Block &block : callee_region) {
            mlir::Block *cloned_block = new mlir::Block();
            if (cloned_block == nullptr) {
                LOG(ERROR) << "Failed to allocate block during inlining\n";
                return mlir::failure();
            }

            for (mlir::BlockArgument arg : block.getArguments()) {
                cloned_block->addArgument(arg.getType(), arg.getLoc());
            }

            caller_block->getParent()->getBlocks().insert(
                split_block->getIterator(), cloned_block
            );

            block_map[&block] = cloned_block;
        }

        // Second pass: clone operations and fix up block references
        for (mlir::Block &orig_block : callee_region) {
            mlir::Block *cloned_block = block_map[&orig_block];
            builder.setInsertionPointToEnd(cloned_block);

            for (mlir::Operation &op : orig_block) {
                if (op.hasTrait< mlir::OpTrait::IsTerminator >()) {
                    if (auto return_op = dyn_cast< cir::ReturnOp >(&op)) {
                        // Handle return operation - branch to continue block
                        mlir::SmallVector< mlir::Value > results;
                        for (mlir::Value result : return_op.getOperands()) {
                            auto mapped_result = mapper.lookup(result);
                            if (!mapped_result) {
                                LOG(ERROR) << "Failed to map return value during inlining\n";
                                return mlir::failure();
                            }
                            results.push_back(mapped_result);
                        }

                        // Replace call results and branch to continue block
                        auto call_results = call_op.getResults();
                        if (call_results.size() != results.size()) {
                            LOG(ERROR) << "Result count mismatch during inlining\n";
                            return mlir::failure();
                        }

                        for (auto [callResult, returnValue] : llvm::zip(call_results, results))
                        {
                            if (!callResult || !returnValue) {
                                LOG(ERROR) << "Null result encountered during inlining\n";
                                return mlir::failure();
                            }
                            callResult.replaceAllUsesWith(returnValue);
                        }

                        builder.create< cir::BrOp >(loc, split_block);
                    } else if (auto branch_op = dyn_cast< cir::BrOp >(&op)) {
                        // Fix branch destinations
                        mlir::Block *targetBlock = block_map[branch_op.getDest()];
                        if (!targetBlock) {
                            LOG(ERROR) << "Failed to find target block during inlining\n";
                            return mlir::failure();
                        }
                        mlir::SmallVector< mlir::Value > operands;
                        for (mlir::Value operand : branch_op.getDestOperands()) {
                            auto mapped_operand = mapper.lookup(operand);
                            if (!mapped_operand) {
                                LOG(ERROR) << "Failed to map branch operand during inlining\n";
                                return mlir::failure();
                            }
                            operands.push_back(mapped_operand);
                        }
                        builder.create< cir::BrOp >(loc, targetBlock, operands);
                    }
                } else {
                    // Clone regular operations
                    builder.clone(op, mapper);
                }
            }
        }

        mlir::Block *callee_entry_block = block_map[&callee_region.front()];
        builder.setInsertionPointToEnd(caller_block);

        // If entry block has arguments, pass them from the call operands
        mlir::SmallVector< mlir::Value > entry_args;
        for (mlir::Value arg : callee.getArguments()) {
            entry_args.push_back(mapper.lookup(arg));
        }
        builder.create< cir::BrOp >(loc, callee_entry_block, entry_args);

        // Remove the original call
        call_op.erase();
        callee.erase();
        return mlir::success();
    }

} // namespace patchestry::passes
