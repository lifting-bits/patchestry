/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Passes/PatchSpec.hpp"
#include <memory>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <optional>
#include <set>

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
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LLVM.h>

#include <patchestry/Passes/InstrumentationPass.hpp>
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
        // Test function to print the parsed config
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

    std::unique_ptr< mlir::Pass > createInstrumentationPass(const std::string &spec_file) {
        return std::make_unique< InstrumentationPass >(spec_file);
    }

    InstrumentationPass::InstrumentationPass(std::string spec) : spec_file(std::move(spec)) {
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

        // Second pass: gather all functions for later instrumentation
        mod.walk([&](mlir::Operation *op) {
            if (auto func = mlir::dyn_cast< cir::FuncOp >(op)) {
                function_worklist.push_back(func);
            }

            llvm::outs() << "Found function: " << op->getName() << "\n";
        });

        // Third pass: process each function to instrument function calls
        // We use indexed loop because function_worklist size could change during
        // instrumentation
        for (size_t i = 0; i < function_worklist.size(); ++i) {
            instrument_function_calls(function_worklist[i]);
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
                return mlir::WalkResult::interrupt();
            }

            auto callee_name = op.getCallee()->str();
            assert(!callee_name.empty() && "Callee name is empty");

            auto patch_spec = config->matches_symbol(callee_name);
            if (patch_spec) {
                const auto &match = patch_spec->match;
                const auto &patch = patch_spec->patch;
                // Create module from the patch file mlir representation
                auto patch_module = load_patch_module(*op->getContext(), *patch.patch_module);
                if (!patch_module) {
                    LOG(ERROR) << "Failed to load patch module for function: " << callee_name
                               << "\n ";
                    return mlir::WalkResult::skip();
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

            return mlir::WalkResult::advance();
        });
    }

    /**
     * @brief Prepares the arguments for a function call based on the patch information.
     *        This function handles argument type casting and argument matching.
     *
     * @param builder The MLIR operation builder.
     * @param op The call operation to be instrumented.
     * @param patch_func The function to be called as a patch.
     * @param patch The patch information.
     * @param args The vector to store the prepared arguments.
     */
    void InstrumentationPass::prepare_call_arguments(
        mlir::OpBuilder &builder, cir::CallOp op, cir::FuncOp patch_func,
        const PatchInfo &patch, llvm::SmallVector< mlir::Value > &args
    ) {
        if (op.getNumArgOperands() == 0) {
            assert(patch.arguments.empty() && "Patch arguments are not empty");
            assert(
                patch_func.getNumArguments() == 0 && "Patch function arguments are not empty"
            );
            return;
        }

        if (patch_func.getNumArguments() > op.getNumArgOperands()) {
            LOG(ERROR) << "Number of arguments in patch is greater than the number of "
                          "arguments in the call operation\n";
            return;
        }

        // Check if patch function argument is taking return value
        if (patch.arguments.size() == 1 && patch.arguments[0] == "return_value") {
            if (op.getResultTypes().size() == 0) {
                LOG(ERROR) << "Call operation does not have a return value\n";
                return;
            }
            auto patch_argument_type = patch_func.getArguments().front().getType();
            auto call_return_type    = op->getResultTypes().front();
            if (patch_argument_type != call_return_type) {
                auto cast_op = builder.create< cir::CastOp >(
                    op->getLoc(), patch_argument_type,
                    getCastKind(call_return_type, patch_argument_type), op.getResults().front()
                );
                args.append(cast_op->getResults().begin(), cast_op->getResults().end());
                return;
            }
            args.append(op->getResults().begin(), op->getResults().end());
            return;
        }

        auto create_cast = [&](mlir::Value value, mlir::Type type) -> mlir::Value {
            if (value.getType() == type) {
                return value;
            }

            auto cast_op = builder.create< cir::CastOp >(
                op->getLoc(), type, getCastKind(value.getType(), type), value
            );
            return cast_op->getResults().front();
        };
        (void) create_cast;

        // llvm::SmallVector< mlir::Value, 4 > argument_vec;
        for (unsigned i = 0; i < patch.arguments.size(); i++) {
            auto patch_arg_type = patch_func.getArgumentTypes()[i];
            args.push_back(create_cast(op.getArgOperands()[i], patch_arg_type));
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
        cir::CallOp op, const PatchMatch &match, const PatchInfo &patch,
        mlir::ModuleOp patch_module
    ) {
        mlir::OpBuilder builder(op);
        builder.setInsertionPoint(op);
        auto module = op->getParentOfType< mlir::ModuleOp >();

        std::string patch_function_name = namifyPatchFunction(patch.patch_function);
        auto input_types                = llvm::to_vector(op->getOperandTypes());
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

        auto symbol_ref = mlir::FlatSymbolRefAttr::get(op->getContext(), patch_function_name);
        llvm::SmallVector< mlir::Value > function_args;
        prepare_call_arguments(builder, op, patch_func, patch, function_args);
        auto call_op = builder.create< cir::CallOp >(
            op->getLoc(), symbol_ref,
            patch_func->getResultTypes().size() != 0 ? patch_func->getResultTypes().front()
                                                     : mlir::Type(),
            function_args
        );
        call_op->setAttr("extra_attrs", op.getExtraAttrs());
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
        cir::CallOp op, const PatchMatch &match, const PatchInfo &patch,
        mlir::ModuleOp patch_module
    ) {
        mlir::OpBuilder builder(op);
        auto module = op->getParentOfType< mlir::ModuleOp >();
        builder.setInsertionPointAfter(op);

        std::string patch_function_name = namifyPatchFunction(patch.patch_function);
        auto input_types                = llvm::to_vector(op.getResultTypes());
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

        auto symbol_ref = mlir::FlatSymbolRefAttr::get(op->getContext(), patch_function_name);
        llvm::SmallVector< mlir::Value > function_args;
        prepare_call_arguments(builder, op, patch_func, patch, function_args);
        auto call_op = builder.create< cir::CallOp >(
            op->getLoc(), symbol_ref,
            patch_func->getResultTypes().size() != 0 ? patch_func->getResultTypes().front()
                                                     : mlir::Type(),
            function_args
        );
        call_op->setAttr("extra_attrs", op.getExtraAttrs());
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

} // namespace patchestry::passes
