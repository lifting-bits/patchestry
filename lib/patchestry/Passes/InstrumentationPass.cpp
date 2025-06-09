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
         * @brief Parses a YAML string into a PatchConfiguration object.
         *
         * This function parses a YAML string into a PatchConfiguration object, which contains
         * the parsed patch specifications. It uses llvm::SourceMgr to manage the YAML input
         * and llvm::yaml::Input to parse the YAML content.
         *
         * @param yaml_str The YAML string to parse
         * @return llvm::Expected< patchestry::passes::PatchConfiguration > The parsed
         * configuration or an error if parsing fails
         */
        llvm::Expected< patchestry::passes::PatchConfiguration >
        parseSpecifications(llvm::StringRef yaml_str) {
            llvm::SourceMgr sm;
            PatchConfiguration config;
            llvm::yaml::Input yaml_input(yaml_str, &sm);
            yaml_input >> config;
            if (yaml_input.error()) {
                return llvm::createStringError(
                    llvm::Twine("Failed to parse YAML: ") + yaml_input.error().message()
                );
            }

            return config;
        }

#ifdef DEBUG
        /**
         * @brief Prints the parsed patch configuration to the console.
         *
         * This function prints the parsed patch configuration to the console for debugging
         * purposes. It prints the number of patches, each patch's match criteria, and the
         * patch details including code, patch file, function name, and arguments.
         *
         * @param config The PatchConfiguration object to print
         */
        void printSpecifications(const PatchConfiguration &config) {
            llvm::outs() << "Number of patches: " << config.patches.size() << "\n";
            for (const auto &spec : config.patches) {
                const auto &match = spec.match;
                llvm::outs() << "Match:\n";
                llvm::outs() << "  Symbol: " << match.symbol << "\n";
                llvm::outs() << "  Kind: " << match.kind << "\n";
                llvm::outs() << "  Operation: " << match.operation << "\n";
                llvm::outs() << "  Function Name: " << match.function_name << "\n";
                llvm::outs() << "  Argument Matches:\n";
                for (const auto &arg_match : match.argument_matches) {
                    llvm::outs() << "    - Index: " << arg_match.index << "\n";
                    llvm::outs() << "      Name: " << arg_match.name << "\n";
                    llvm::outs() << "      Type: " << arg_match.type << "\n";
                }
                llvm::outs() << "  Variable Matches:\n";
                for (const auto &var_match : match.variable_matches) {
                    llvm::outs() << "    - Name: " << var_match.name << "\n";
                    llvm::outs() << "      Type: " << var_match.type << "\n";
                }

                const auto &patch = spec.patch;
                llvm::outs() << "  Patch:\n";
                switch (patch.mode) {
                    case PatchInfoMode::APPLY_BEFORE:
                        llvm::outs() << "    mode: Apply Before\n";
                        break;
                    case PatchInfoMode::APPLY_AFTER:
                        llvm::outs() << "    mode: Apply After\n";
                        break;
                    case PatchInfoMode::REPLACE:
                        llvm::outs() << "    mode: Replace\n";
                        break;
                    default:
                        break;
                }
                llvm::outs() << "    Code:\n" << patch.code << "\n";
                llvm::outs() << "    Patch File: " << patch.patch_file << "\n";
                llvm::outs() << "    Patch Function: " << patch.patch_function << "\n";
                llvm::outs() << "    Arguments:\n";
                for (const auto &arg : patch.arguments) {
                    llvm::outs() << "      - " << arg << "\n";
                }
            }
        }
#endif

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
        std::string spec, const PatchOptions &patch_options
    )
        : spec_file(std::move(spec)), patch_options(patch_options) {
        auto buffer_or_err = llvm::MemoryBuffer::getFile(spec_file);
        if (!buffer_or_err) {
            LOG(ERROR) << "Error: Failed to read patch specification file: " << spec_file
                       << "\n";
            return;
        }

        PatchSpecContext::getInstance().set_spec_path(spec_file);
        auto config_or_err = parseSpecifications(buffer_or_err.get()->getBuffer());
        if (!config_or_err) {
            LOG(ERROR) << "Error: Failed to parse patch specification file: " << spec_file
                       << "\n";
            return;
        }

        config = std::move(config_or_err.get());
        for (auto &spec : config->patches) {
            auto &patch = spec.patch;
            if (!llvm::sys::fs::exists(patch.patch_file)) {
                LOG(ERROR) << "Patch file " << patch.patch_file << " does not exist\n";
                continue;
            }

            patch.patch_module = emitModuleAsString(patch.patch_file, config->arch);
        }

// print specifications
#ifdef DEBUG
        printSpecifications(*config);
#endif
    }

    /**
     * @brief Applies instrumentation to the MLIR module based on the patch specifications.
     *        The function iterates through all functions in the module and applies the
     *        specified patches to function calls.
     *
     * @note The function list in module can grow during instrumentation. We collect the
     * list of functions before starting the instrumentation process to avoid issues with
     *       growing functions.
     *
     * @todo Perform instrumentation based on Operations match
     */
    void InstrumentationPass::runOnOperation() {
        mlir::ModuleOp mod = getOperation();
        llvm::SmallVector< cir::FuncOp, 8 > function_worklist;
        llvm::SmallVector< mlir::Operation *, 8 > operation_worklist;

        // Second pass: gather all functions for later instrumentation
        mod.walk([&](cir::FuncOp op) { function_worklist.push_back(op); });

        // Gather operations for instrumentation
        mod.walk([&](mlir::Operation *op) {
            if (!mlir::isa< cir::FuncOp, mlir::ModuleOp, cir::GlobalOp >(op)) {
                operation_worklist.push_back(op);
            }
        });

        // Third pass: process each function to instrument function calls
        // We use indexed loop because function_worklist size could change during
        // instrumentation
        for (size_t i = 0; i < function_worklist.size(); ++i) {
            instrument_function_calls(function_worklist[i]);
        }

        // Process operations for instrumentation
        for (auto *op : operation_worklist) {
            instrument_operation(op);
        }

        // Inline inserted call operation
        if (patch_options.enable_inlining) {
            for (auto *op : inline_worklists) {
                std::ignore = inline_call(mod, mlir::cast< cir::CallOp >(op));
            }

            // clear the worklist after inlining
            inline_worklists.clear();
        }
    }

    /**
     * @brief Instruments function calls within a given function based on the patch
     * specifications. The function iterates through all function calls in the provided
     * function and applies the specified patches.
     *
     * @param func The function to instrument.
     */
    void InstrumentationPass::instrument_function_calls(cir::FuncOp func) {
        func.walk([&](cir::CallOp op) {
            if (!config || config->patches.empty()) {
                LOG(ERROR) << "No patch configuration found. Skipping...\n";
                return;
            }

            auto callee_name = op.getCallee()->str();
            assert(!callee_name.empty() && "Callee name is empty");

            auto func = op->getParentOfType< cir::FuncOp >();
            if (!func) {
                LOG(ERROR) << "Call operation is not in a function. Skipping...\n";
                return;
            }

            for (const auto &spec : config->patches) {
                if (OperationMatcher::matches(op, func, spec, OperationMatcher::Mode::FUNCTION)
                    && !exclude_from_patching(func, spec))
                {
                    const auto &patch = spec.patch;
                    const auto &match = spec.match;
                    // Create module from the patch file mlir representation
                    auto patch_module =
                        load_patch_module(*op->getContext(), *patch.patch_module);
                    if (!patch_module) {
                        LOG(ERROR)
                            << "Failed to load patch module for function: " << callee_name
                            << "\n ";
                        continue;
                    }

                    switch (patch.mode) {
                        case PatchInfoMode::APPLY_BEFORE: {
                            apply_before_patch(op, match, patch, patch_module.get());
                            break;
                        }
                        case PatchInfoMode::APPLY_AFTER: {
                            apply_after_patch(op, match, patch, patch_module.get());
                            break;
                        }
                        case PatchInfoMode::REPLACE:
                            replace_call(op, match, patch, patch_module.get());
                            break;
                        default:
                            break;
                    }
                }
            }
        });
    }

    /**
     * @brief Instruments an operation based on patch specifications.
     *
     * This method applies patches to operations that match the operation patterns
     * defined in the patch specification. It supports variable matching and applies
     * before patch modes (replace and after mode is not supported for operations yet).
     *
     * @param op The operation to be instrumented
     */
    void InstrumentationPass::instrument_operation(mlir::Operation *op) {
        if (!config || config->patches.empty()) {
            LOG(ERROR) << "No patch configuration found. Skipping...\n";
            return;
        }
        auto func = op->getParentOfType< cir::FuncOp >();
        if (!func) {
            LOG(INFO) << "Operation is not in a function. Skipping...\n";
            return;
        }

        for (const auto &spec : config->patches) {
            if (OperationMatcher::matches(op, func, spec, OperationMatcher::Mode::OPERATION)
                && !exclude_from_patching(func, spec))
            {
                const auto &patch = spec.patch;
                const auto &match = spec.match;
                auto patch_module = load_patch_module(*op->getContext(), *patch.patch_module);
                if (!patch_module) {
                    LOG(ERROR) << "Failed to load patch module for operation: "
                               << op->getName().getStringRef().str() << "\n ";
                    continue;
                }

                switch (patch.mode) {
                    case PatchInfoMode::APPLY_BEFORE: {
                        apply_before_patch(op, match, patch, patch_module.get());
                        break;
                    }
                    default:
                        LOG(ERROR)
                            << "Unsupported patch mode: " << patchInfoModeToString(patch.mode)
                            << " for operation: " << op->getName().getStringRef().str() << "\n";
                        break;
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
    void InstrumentationPass::prepare_call_arguments(
        mlir::OpBuilder &builder, mlir::Operation *call_op, cir::FuncOp patch_func,
        const PatchInfo &patch, llvm::SmallVector< mlir::Value > &args
    ) {
        auto create_cast = [&](mlir::Value value, mlir::Type type) -> mlir::Value {
            if (value.getType() == type) {
                return value;
            }

            auto cast_op = builder.create< cir::CastOp >(
                call_op->getLoc(), type, getCastKind(value.getType(), type), value
            );
            return cast_op->getResults().front();
        };

        // Handle structured argument specifications
        for (size_t i = 0;
             i < patch.argument_sources.size() && i < patch_func.getNumArguments(); ++i)
        {
            const auto &arg_spec = patch.argument_sources[i];
            auto patch_arg_type  = patch_func.getArgumentTypes()[i];
            mlir::Value arg_value;

            switch (arg_spec.source) {
                case ArgumentSourceType::OPERAND: {
                    // Get operand by index
                    if (!arg_spec.index.has_value()) {
                        LOG(ERROR) << "OPERAND source requires index field\n";
                        continue;
                    }
                    unsigned idx = arg_spec.index.value();

                    if (auto orig_call_op = mlir::dyn_cast< cir::CallOp >(call_op)) {
                        if (idx >= orig_call_op.getArgOperands().size()) {
                            LOG(ERROR) << "Operand index " << idx << " out of range\n";
                            continue;
                        }
                        arg_value = orig_call_op.getArgOperands()[idx];
                    } else {
                        if (idx >= call_op->getNumOperands()) {
                            LOG(ERROR) << "Operand index " << idx << " out of range\n";
                            continue;
                        }
                        arg_value = call_op->getOperand(idx);
                    }
                    break;
                }
                case ArgumentSourceType::VARIABLE: {
                    // Handle local variables only
                    if (!arg_spec.symbol.has_value()) {
                        LOG(ERROR) << "VARIABLE source requires symbol field\n";
                        continue;
                    }

                    const std::string &var_name = arg_spec.symbol.value();
                    mlir::Value var_value;
                    bool found = false;

                    // Look for local variables in function scope only
                    auto func = call_op->getParentOfType< cir::FuncOp >();
                    if (!func) {
                        LOG(ERROR) << "Cannot find parent function for local variable lookup\n";
                        continue;
                    }

                    // Search for local variables in function scope
                    func.walk([&](mlir::Operation *op) {
                        if (auto alloca_op = mlir::dyn_cast< cir::AllocaOp >(op)) {
                            if (auto name_attr = op->getAttrOfType< mlir::StringAttr >("name"))
                            {
                                if (name_attr.getValue() == var_name) {
                                    var_value = alloca_op.getResult();
                                    found     = true;
                                    return mlir::WalkResult::interrupt();
                                }
                            }
                        }
                        return mlir::WalkResult::advance();
                    });

                    if (!found) {
                        LOG(WARNING) << "Local variable '" << var_name << "' not found\n";
                        continue;
                    }
                    arg_value = builder.create< cir::LoadOp >(call_op->getLoc(), var_value);
                    break;
                }
                case ArgumentSourceType::SYMBOL: {
                    // Handle global variables, functions, and any symbol in symbol table
                    if (!arg_spec.symbol.has_value()) {
                        LOG(ERROR) << "SYMBOL source requires symbol field\n";
                        continue;
                    }

                    const std::string &symbol_name = arg_spec.symbol.value();
                    mlir::Value symbol_value;
                    bool found = false;

                    auto module = call_op->getParentOfType< mlir::ModuleOp >();
                    if (!module) {
                        LOG(ERROR) << "Cannot find parent module for symbol lookup\n";
                        continue;
                    }

                    // Look for global variables
                    if (auto global_op = module.lookupSymbol< cir::GlobalOp >(symbol_name)) {
                        // Create a GetGlobal operation to access the global variable
                        auto global_type = global_op.getSymType();
                        if (auto global_ptr_type =
                                mlir::dyn_cast< cir::PointerType >(global_type))
                        {
                            symbol_value = builder.create< cir::GetGlobalOp >(
                                call_op->getLoc(), global_ptr_type, symbol_name
                            );
                            found = true;
                        } else {
                            // For non-pointer globals, create a pointer type
                            auto ptr_type =
                                cir::PointerType::get(builder.getContext(), global_type);
                            symbol_value = builder.create< cir::GetGlobalOp >(
                                call_op->getLoc(), ptr_type, symbol_name
                            );
                            found = true;
                        }
                    }

                    // Look for functions
                    if (!found) {
                        if (auto func_op = module.lookupSymbol< cir::FuncOp >(symbol_name)) {
                            // Create a function reference
                            auto func_type = func_op.getFunctionType();
                            auto func_ptr_type =
                                cir::PointerType::get(builder.getContext(), func_type);
                            auto symbol_ref =
                                mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol_name);

                            // Create a constant operation for the function pointer
                            symbol_value = builder.create< cir::GetGlobalOp >(
                                call_op->getLoc(), func_ptr_type, symbol_ref
                            );
                            found = true;
                        }
                    }

                    if (!found) {
                        LOG(WARNING)
                            << "Symbol '" << symbol_name << "' not found in symbol table\n";
                        continue;
                    }
                    arg_value = builder.create< cir::LoadOp >(call_op->getLoc(), symbol_value);
                    break;
                }
                case ArgumentSourceType::RETURN_VALUE: {
                    // Handle return value of function or operation
                    if (call_op->getNumResults() == 0) {
                        LOG(ERROR) << "Operation/function does not have a return value\n";
                        continue;
                    }

                    // Get the first result (most common case)
                    arg_value = call_op->getResult(0);
                    break;
                }
                case ArgumentSourceType::CONSTANT: {
                    // Create constant value
                    if (!arg_spec.value.has_value()) {
                        LOG(ERROR) << "CONSTANT source requires value field\n";
                        continue;
                    }

                    const std::string &const_value = arg_spec.value.value();

                    // Parse constant based on patch function argument type
                    if (auto int_type = mlir::dyn_cast< cir::IntType >(patch_arg_type)) {
                        try {
                            // Parse integer constant
                            int64_t int_val =
                                std::stoll(const_value, nullptr, 0); // Support hex, oct, dec

                            auto attr = cir::IntAttr::get(
                                cir::IntType::get(
                                    builder.getContext(), int_type.getWidth(),
                                    int_type.isSigned()
                                ),
                                llvm::APSInt(int_type.getWidth(), int_val)
                            );
                            arg_value = builder.create< cir::ConstantOp >(
                                call_op->getLoc(), patch_arg_type, attr
                            );
                        } catch (const std::exception &e) {
                            LOG(ERROR) << "Failed to parse integer constant '" << const_value
                                       << "': " << e.what() << "\n";
                            continue;
                        }
                    } else if (auto ptr_type =
                                   mlir::dyn_cast< cir::PointerType >(patch_arg_type))
                    {
                        try {
                            // Parse pointer constant (usually hex address)
                            uint64_t ptr_val = std::stoull(const_value, nullptr, 0);
                            auto int_type = cir::IntType::get(builder.getContext(), 64, false);
                            auto int_attr = cir::IntAttr::get(
                                cir::IntType::get(
                                    builder.getContext(), int_type.getWidth(),
                                    int_type.isSigned()
                                ),
                                llvm::APSInt(int_type.getWidth(), ptr_val)
                            );
                            auto int_const = builder.create< cir::ConstantOp >(
                                call_op->getLoc(), int_type, int_attr
                            );
                            arg_value = builder.create< cir::CastOp >(
                                call_op->getLoc(), patch_arg_type, cir::CastKind::int_to_ptr,
                                int_const
                            );
                        } catch (const std::exception &e) {
                            LOG(ERROR) << "Failed to parse pointer constant '" << const_value
                                       << "': " << e.what() << "\n";
                            continue;
                        }
                    } else {
                        LOG(ERROR)
                            << "Unsupported constant type for value '" << const_value << "'\n";
                        continue;
                    }
                    break;
                }
            }

            if (arg_value) {
                args.push_back(create_cast(arg_value, patch_arg_type));
            }
        }
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
        mlir::Operation *call_op, const PatchMatch &match, const PatchInfo &patch,
        mlir::ModuleOp patch_module
    ) {
        if (call_op == nullptr) {
            LOG(ERROR) << "Patch before: Operation is null";
            return;
        }

        mlir::OpBuilder builder(call_op);
        builder.setInsertionPoint(call_op);
        auto module = call_op->getParentOfType< mlir::ModuleOp >();

        std::string patch_function_name = namifyPatchFunction(patch.patch_function);
        auto input_types                = llvm::to_vector(call_op->getOperandTypes());
        if (!patch_module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        auto result = merge_module_symbol(module, patch_module, patch_function_name);
        if (mlir::failed(result)) {
            LOG(ERROR) << "Failed to insert symbol into module\n";
            return;
        }

        auto patch_func = module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!patch_func) {
            LOG(ERROR) << "Patch function " << patch_function_name
                       << " not defined. Patching failed...\n";
            return;
        }

        auto symbol_ref =
            mlir::FlatSymbolRefAttr::get(call_op->getContext(), patch_function_name);
        llvm::SmallVector< mlir::Value > function_args;
        prepare_call_arguments(builder, call_op, patch_func, patch, function_args);
        auto patch_call_op = builder.create< cir::CallOp >(
            call_op->getLoc(), symbol_ref,
            patch_func->getResultTypes().size() != 0 ? patch_func->getResultTypes().front()
                                                     : mlir::Type(),
            function_args
        );

        // Add extra attributes for the patched call function
        if (auto orig_call_op = mlir::dyn_cast< cir::CallOp >(call_op)) {
            patch_call_op->setAttr("extra_attrs", orig_call_op.getExtraAttrs());
        } else {
            mlir::NamedAttrList empty;
            patch_call_op->setAttr(
                "extra_attrs",
                cir::ExtraFuncAttributesAttr::get(
                    call_op->getContext(), empty.getDictionary(call_op->getContext())
                )
            );
        }
        if (patch_options.enable_inlining) {
            inline_worklists.push_back(patch_call_op);
        }
        (void) match;
    }

    /**
     * @brief Applies a patch after the function call. This function inserts a call to the
     * patch function after the original function call.
     *
     * @param op The call operation to be instrumented.
     * @param match The match information for the function call.
     * @param patch The patch information.
     * @param patch_module The module containing the patch function.
     */
    void InstrumentationPass::apply_after_patch(
        mlir::Operation *call_op, const PatchMatch &match, const PatchInfo &patch,
        mlir::ModuleOp patch_module
    ) {
        if (call_op == nullptr) {
            LOG(ERROR) << "Patch after: Operation is null";
            return;
        }

        mlir::OpBuilder builder(call_op);
        auto module = call_op->getParentOfType< mlir::ModuleOp >();
        builder.setInsertionPointAfter(call_op);

        std::string patch_function_name = namifyPatchFunction(patch.patch_function);
        auto input_types                = llvm::to_vector(call_op->getResultTypes());
        if (!patch_module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        auto result = merge_module_symbol(module, patch_module, patch_function_name);
        if (mlir::failed(result)) {
            LOG(ERROR) << "Failed to insert symbol into module\n";
            return;
        }

        auto patch_func = module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!patch_func) {
            LOG(ERROR) << "Patch function " << patch_function_name
                       << " not defined. Patching failed...\n";
            return;
        }

        auto symbol_ref =
            mlir::FlatSymbolRefAttr::get(call_op->getContext(), patch_function_name);
        llvm::SmallVector< mlir::Value > function_args;
        prepare_call_arguments(builder, call_op, patch_func, patch, function_args);
        auto patch_call_op = builder.create< cir::CallOp >(
            call_op->getLoc(), symbol_ref,
            patch_func->getResultTypes().size() != 0 ? patch_func->getResultTypes().front()
                                                     : mlir::Type(),
            function_args
        );

        // Add extra attributes for the patched call function
        if (auto orig_call_op = mlir::dyn_cast< cir::CallOp >(call_op)) {
            patch_call_op->setAttr("extra_attrs", orig_call_op.getExtraAttrs());
        } else {
            mlir::NamedAttrList empty;
            patch_call_op->setAttr(
                "extra_attrs",
                cir::ExtraFuncAttributesAttr::get(
                    call_op->getContext(), empty.getDictionary(patch_call_op->getContext())
                )
            );
        }
        if (patch_options.enable_inlining) {
            inline_worklists.push_back(patch_call_op);
        }
        (void) match;
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
        cir::CallOp op, const PatchMatch &match, const PatchInfo &patch,
        mlir::ModuleOp patch_module
    ) {
        mlir::OpBuilder builder(op);
        auto loc    = op.getLoc();
        auto *ctx   = op->getContext();
        auto module = op->getParentOfType< mlir::ModuleOp >();
        assert(module && "Wrap around patch: no module found");

        builder.setInsertionPoint(op);

        auto callee_name = op.getCallee()->str();
        assert(!callee_name.empty() && "Wrap around patch: callee name is empty");

        auto patch_function_name = namifyPatchFunction(patch.patch_function);
        auto result_types        = llvm::to_vector(op.getResultTypes());

        if (!patch_module.lookupSymbol< cir::FuncOp >(patch_function_name)) {
            LOG(ERROR) << "Patch module not found or patch function not defined\n";
            return;
        }

        auto result = merge_module_symbol(module, patch_module, patch_function_name);
        if (mlir::failed(result)) {
            LOG(ERROR) << "Failed to insert symbol into module\n";
            return;
        }

        auto wrap_func = module.lookupSymbol< cir::FuncOp >(patch_function_name);
        if (!wrap_func) {
            LOG(ERROR) << "Wrap around patch: patch function " << patch.patch_function
                       << " not defined. Patching failed...\n";
            return;
        }

        auto wrap_func_ref = mlir::FlatSymbolRefAttr::get(ctx, patch_function_name);
        auto wrap_call_op  = builder.create< cir::CallOp >(
            loc, wrap_func_ref, result_types.size() != 0 ? result_types.front() : mlir::Type(),
            op.getArgOperands()
        );
        wrap_call_op->setAttr("extra_attrs", op.getExtraAttrs());
        op.replaceAllUsesWith(wrap_call_op);
        op.erase();
        (void) match;
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

    /**
     * @brief Determines if a function should be excluded from patching using regex checks.
     *
     * @param func The function to check for exclusion
     * @param spec The patch specification containing exclusion rules
     * @return bool True if the function should be excluded, false otherwise
     */
    bool InstrumentationPass::exclude_from_patching(cir::FuncOp func, const PatchSpec &spec) {
        // Get the function name
        auto func_name = func.getName().str();

        // If no exclusion patterns are specified, don't exclude
        if (spec.exclude.empty()) {
            return false;
        }

        // Check each exclusion pattern against the function name
        for (const auto &pattern : spec.exclude) {
            try {
                // Create regex from the pattern
                std::regex exclude_regex(pattern);

                // Check if the function name matches the exclusion pattern
                if (std::regex_match(func_name, exclude_regex)) {
                    LOG(INFO) << "Function '" << func_name
                              << "' excluded by pattern: " << pattern << "\n";
                    return true;
                }
            } catch (const std::regex_error &e) {
                LOG(ERROR) << "Invalid regex pattern in exclude list: '" << pattern
                           << "' - Error: " << e.what() << "\n";
                // Continue with other patterns even if one is invalid
                continue;
            }
        }

        // Function is not excluded by any pattern
        return false;
    }

} // namespace patchestry::passes
