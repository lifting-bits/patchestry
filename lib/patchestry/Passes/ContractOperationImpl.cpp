/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/Dialect/Contracts/ContractsDialect.hpp>
#include <patchestry/Passes/InstrumentationPass.hpp>
#include <patchestry/YAML/ContractSpec.hpp>

#include "ContractOperationImpl.hpp"

namespace patchestry::passes {

    extern std::string namifyFunction(const std::string &str);

    void ContractOperationImpl::emitRuntimeContract(
        InstrumentationPass &pass, mlir::OpBuilder &builder, mlir::Operation *targetOp,
        mlir::ModuleOp contractModule, const ContractInformation &contract, ContractMode mode,
        bool shouldInline
    ) {
        if (!targetOp) {
            LOG(ERROR
            ) << "emit_runtime_contract: the passed function to be instrumented was null";
            return;
        }

        const auto &spec = contract.spec.value();
        assert(spec.implementation && "Runtime contracts should have an implementation");

        switch (mode) {
            case ContractMode::APPLY_BEFORE:
                builder.setInsertionPoint(targetOp);
                break;
            case ContractMode::APPLY_AFTER:
                builder.setInsertionPointAfter(targetOp);
                break;
            case ContractMode::APPLY_AT_ENTRYPOINT:
                break;
            default:
                LOG(ERROR) << "Unsupported contract mode: " << contract::infoModeToString(mode)
                           << "\n";
                return;
        }

        auto module                      = targetOp->getParentOfType< mlir::ModuleOp >();
        std::string contractFunctionName = namifyFunction(spec.implementation->function_name);

        auto contractFuncFromModule =
            contractModule.lookupSymbol< cir::FuncOp >(contractFunctionName);
        if (!contractFuncFromModule) {
            LOG(ERROR) << "Contract module not found or contract function not defined: "
                       << contractFunctionName << "\n";
            return;
        }

        // Ensure function declaration exists in the module first
        if (mlir::failed(pass.insert_function_declaration(module, contractFuncFromModule))) {
            LOG(ERROR) << "Failed to ensure function declaration for " << contractFunctionName
                       << "\n";
            return;
        }

        auto contractFunc = module.lookupSymbol< cir::FuncOp >(contractFunctionName);
        if (!contractFunc) {
            if (mlir::failed(
                    pass.merge_module_symbol(module, contractModule, contractFunctionName)
                ))
            {
                LOG(ERROR) << "Failed to insert symbol into module\n";
                return;
            }
            contractFunc = module.lookupSymbol< cir::FuncOp >(contractFunctionName);
        }

        if (!contractFunc) {
            LOG(ERROR) << "Contract function " << contractFunctionName
                       << " not defined after insertion. Insertion failed...\n";
            return;
        }

        auto symbolRef =
            mlir::FlatSymbolRefAttr::get(targetOp->getContext(), contractFunctionName);
        llvm::SmallVector< mlir::Value > functionArgs;
        pass.prepare_contract_call_arguments(
            builder, targetOp, contractFunc, contract, functionArgs
        );

        auto contractCallOp = builder.create< cir::CallOp >(
            targetOp->getLoc(), symbolRef,
            contractFunc->getResultTypes().empty() ? mlir::Type()
                                                   : contractFunc->getResultTypes().front(),
            functionArgs
        );

        pass.set_instrumentation_call_attributes(contractCallOp, targetOp);

        if (shouldInline) {
            pass.inline_worklists.push_back(contractCallOp);
        }
    }

    // Helper to build PredicateAttr from parsed Predicate struct
    static std::optional< ::contracts::PredicateAttr > buildPredicateAttr(
        mlir::MLIRContext *ctx, mlir::Operation *op, const contract::Predicate &pred
    ) {
        (void) op; // Currently unused, but kept for future extensibility

        // Initialize all attributes as null
        ::contracts::TargetAttr targetAttr           = nullptr;
        mlir::Attribute valueAttr                    = nullptr;
        ::contracts::ContractAlignmentAttr alignAttr = nullptr;
        mlir::StringAttr exprAttr                    = nullptr;
        ::contracts::ContractRangeAttr rangeAttr     = nullptr;
        ::contracts::RelationKind relationKind       = pred.relation;

        // Build attributes based on predicate kind
        switch (pred.kind) {
            case ::contracts::PredicateKind::nonnull:
                // nonnull: only set target
                if (pred.target == ::contracts::TargetKind::Arg) {
                    if (!pred.arg_index) {
                        LOG(ERROR) << "Argument target requires arg_index\n";
                        return std::nullopt;
                    }
                    targetAttr = ::contracts::TargetAttr::get(
                        ctx, pred.target, *pred.arg_index, mlir::FlatSymbolRefAttr()
                    );
                    targetAttr.dump();
                } else if (pred.target == ::contracts::TargetKind::ReturnValue) {
                    targetAttr = ::contracts::TargetAttr::get(
                        ctx, pred.target, 0, mlir::FlatSymbolRefAttr()
                    );
                } else if (pred.target == ::contracts::TargetKind::Symbol) {
                    if (!pred.symbol) {
                        LOG(ERROR) << "Symbol target requires a symbol name\n";
                        return std::nullopt;
                    }
                    auto symRef = mlir::FlatSymbolRefAttr::get(ctx, *pred.symbol);
                    targetAttr  = ::contracts::TargetAttr::get(ctx, pred.target, 0, symRef);
                }
                break;

            case ::contracts::PredicateKind::relation:
                // relation: set target and value
                if (pred.target == ::contracts::TargetKind::Arg) {
                    if (!pred.arg_index) {
                        LOG(ERROR) << "Argument target requires arg_index\n";
                        return std::nullopt;
                    }
                    targetAttr = ::contracts::TargetAttr::get(
                        ctx, pred.target, *pred.arg_index, mlir::FlatSymbolRefAttr()
                    );
                } else if (pred.target == ::contracts::TargetKind::ReturnValue) {
                    targetAttr = ::contracts::TargetAttr::get(
                        ctx, pred.target, 0, mlir::FlatSymbolRefAttr()
                    );
                } else if (pred.target == ::contracts::TargetKind::Symbol) {
                    if (!pred.symbol) {
                        LOG(ERROR) << "Symbol target requires a symbol name\n";
                        return std::nullopt;
                    }
                    auto symRef = mlir::FlatSymbolRefAttr::get(ctx, *pred.symbol);
                    targetAttr  = ::contracts::TargetAttr::get(ctx, pred.target, 0, symRef);
                }

                // Parse value as integer constant
                if (pred.value) {
                    try {
                        int64_t constVal = std::stoll(*pred.value);
                        valueAttr =
                            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), constVal);
                    } catch (...) {
                        LOG(ERROR) << "Invalid constant value: " << *pred.value << "\n";
                        return std::nullopt;
                    }
                }
                break;

            case ::contracts::PredicateKind::alignment:
                // alignment: only set align
                if (pred.align) {
                    alignAttr = ::contracts::ContractAlignmentAttr::get(ctx, *pred.align);
                } else {
                    LOG(ERROR) << "Alignment predicate requires align value\n";
                    return std::nullopt;
                }
                break;

            case ::contracts::PredicateKind::expr:
                // expr: only set expr
                if (pred.expr) {
                    exprAttr = mlir::StringAttr::get(ctx, *pred.expr);
                } else {
                    LOG(ERROR) << "Expression predicate requires expr value\n";
                    return std::nullopt;
                }
                break;

            case ::contracts::PredicateKind::range:
                // range: set target and range
                if (pred.target == ::contracts::TargetKind::Arg) {
                    if (!pred.arg_index) {
                        LOG(ERROR) << "Argument target requires arg_index\n";
                        return std::nullopt;
                    }
                    targetAttr = ::contracts::TargetAttr::get(
                        ctx, pred.target, *pred.arg_index, mlir::FlatSymbolRefAttr()
                    );
                } else if (pred.target == ::contracts::TargetKind::ReturnValue) {
                    targetAttr = ::contracts::TargetAttr::get(
                        ctx, pred.target, 0, mlir::FlatSymbolRefAttr()
                    );
                } else if (pred.target == ::contracts::TargetKind::Symbol) {
                    if (!pred.symbol) {
                        LOG(ERROR) << "Symbol target requires a symbol name\n";
                        return std::nullopt;
                    }
                    auto symRef = mlir::FlatSymbolRefAttr::get(ctx, *pred.symbol);
                    targetAttr  = ::contracts::TargetAttr::get(ctx, pred.target, 0, symRef);
                }

                // Build RangeAttr
                if (pred.range) {
                    try {
                        auto minVal = std::stoll(pred.range->min);
                        auto maxVal = std::stoll(pred.range->max);
                        auto minAttr =
                            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), minVal);
                        auto maxAttr =
                            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), maxVal);
                        rangeAttr = ::contracts::ContractRangeAttr::get(ctx, minAttr, maxAttr);
                    } catch (...) {
                        LOG(ERROR) << "Invalid range values\n";
                        return std::nullopt;
                    }
                } else {
                    LOG(ERROR) << "Range predicate requires range value\n";
                    return std::nullopt;
                }
                break;
        }

        // Create PredicateAttr with only the relevant components based on kind
        return ::contracts::PredicateAttr::get(
            ctx, pred.kind, targetAttr, relationKind, valueAttr, alignAttr, exprAttr, rangeAttr
        );
    }

    // apply static contract to the target operation
    void ContractOperationImpl::emitStaticContract(
        InstrumentationPass &pass, mlir::OpBuilder &builder, mlir::Operation *targetOp,
        mlir::ModuleOp contractModule, const ContractInformation &contract, ContractMode mode,
        bool shouldInline
    ) {
        // check if the target operation is null
        if (targetOp == nullptr) {
            LOG(ERROR) << "emitStaticContract: the passed function to be instrumented was null";
            return;
        }

        const auto &spec = contract.spec.value();
        if (spec.type != ContractType::STATIC) {
            LOG(ERROR) << "Contract type is not static\n";
            return;
        }

        auto ctx = targetOp->getContext();

        // Build preconditions as PreconditionAttr
        llvm::SmallVector< mlir::Attribute > preconditionAttrs;
        if (spec.preconditions) {
            for (const auto &precondition : spec.preconditions.value()) {
                if (!precondition.pred) {
                    LOG(WARNING) << "Precondition " << precondition.id
                                 << " missing predicate, skipping\n";
                    continue;
                }

                auto predAttr = buildPredicateAttr(ctx, targetOp, *precondition.pred);
                if (predAttr) {
                    auto idAttr  = mlir::StringAttr::get(ctx, precondition.id);
                    auto preAttr = ::contracts::PreconditionAttr::get(ctx, idAttr, *predAttr);
                    preconditionAttrs.push_back(preAttr);
                }
            }
        }

        // Build postconditions as PostconditionAttr
        llvm::SmallVector< mlir::Attribute > postconditionAttrs;
        if (spec.postconditions) {
            for (const auto &postcondition : spec.postconditions.value()) {
                if (!postcondition.pred) {
                    LOG(WARNING) << "Postcondition " << postcondition.id
                                 << " missing predicate, skipping\n";
                    continue;
                }

                auto predAttr = buildPredicateAttr(ctx, targetOp, *postcondition.pred);
                if (predAttr) {
                    auto idAttr   = mlir::StringAttr::get(ctx, postcondition.id);
                    auto postAttr = ::contracts::PostconditionAttr::get(ctx, idAttr, *predAttr);
                    postconditionAttrs.push_back(postAttr);
                }
            }
        }

        // apply both preconditions and postconditions attributes without mode also handle if
        // one is empty
        if (!preconditionAttrs.empty() && !postconditionAttrs.empty()) {
            auto staticContract = ::contracts::StaticContractAttr::get(
                ctx, preconditionAttrs, postconditionAttrs
            );
            targetOp->setAttr("contract.static", staticContract);
        } else if (!preconditionAttrs.empty()) {
            auto staticContract = ::contracts::StaticContractAttr::get(
                ctx, preconditionAttrs, llvm::ArrayRef< mlir::Attribute >{}
            );
            targetOp->setAttr("contract.static", staticContract);
        } else if (!postconditionAttrs.empty()) {
            auto staticContract = ::contracts::StaticContractAttr::get(
                ctx, llvm::ArrayRef< mlir::Attribute >{}, postconditionAttrs
            );
            targetOp->setAttr("contract.static", staticContract);
        } else {
            LOG(WARNING) << "No preconditions or postconditions to apply for contract "
                         << spec.id << "\n";
            return;
        }

        (void) contractModule;
        (void) shouldInline;
        (void) pass;
        (void) builder;
        (void) mode;
    }

    void ContractOperationImpl::applyContractBefore(
        InstrumentationPass &pass, mlir::Operation *target_op,
        const ContractInformation &contract, mlir::ModuleOp contract_module, bool should_inline
    ) {
        if (target_op == nullptr) {
            LOG(ERROR
            ) << "applyContractBefore: the passed function to be instrumented was null";
            return;
        }

        const auto &spec = contract.spec.value();
        switch (spec.type) {
            case ContractType::RUNTIME: {
                mlir::OpBuilder builder(target_op);
                emitRuntimeContract(
                    pass, builder, target_op, contract_module, contract,
                    ContractMode::APPLY_BEFORE, should_inline
                );
                break;
            }
            case ContractType::STATIC: {
                mlir::OpBuilder builder(target_op);
                emitStaticContract(
                    pass, builder, target_op, contract_module, contract,
                    ContractMode::APPLY_BEFORE, should_inline
                );
                break;
            }
        }
    }

    void ContractOperationImpl::applyContractAfter(
        InstrumentationPass &pass, mlir::Operation *target_op,
        const ContractInformation &contract, mlir::ModuleOp contract_module, bool should_inline
    ) {
        if (target_op == nullptr) {
            LOG(ERROR) << "applyContractAfter: the passed function to be instrumented was null";
            return;
        }

        const auto &spec = contract.spec.value();
        switch (spec.type) {
            case ContractType::RUNTIME: {
                mlir::OpBuilder builder(target_op);
                emitRuntimeContract(
                    pass, builder, target_op, contract_module, contract,
                    ContractMode::APPLY_AFTER, should_inline
                );
                break;
            }
            case ContractType::STATIC: {
                mlir::OpBuilder builder(target_op);
                emitStaticContract(
                    pass, builder, target_op, contract_module, contract,
                    ContractMode::APPLY_AFTER, should_inline
                );
                break;
            }
        }
    }

    void ContractOperationImpl::applyContractAtEntrypoint(
        InstrumentationPass &pass, cir::CallOp call_op, const ContractInformation &contract,
        mlir::ModuleOp contract_module, bool should_inline
    ) {
        if (call_op == nullptr) {
            LOG(ERROR) << "applyContractAtEntrypoint: the passed function to be "
                          "instrumented was null";
            return;
        }

        const auto &contract_spec = contract.spec.value();
        auto contract_type        = contract_spec.type;
        if (contract_type == ContractType::STATIC) {
            LOG(ERROR) << "Static contracts are not supported in entrypoint mode\n";
            return;
        }
        assert(
            contract_spec.implementation && "Runtime contracts should have an implementation"
        );
        std::string contract_function_name =
            namifyFunction(contract_spec.implementation->function_name);

        auto contractFuncFromModule =
            contract_module.lookupSymbol< cir::FuncOp >(contract_function_name);
        if (!contractFuncFromModule) {
            LOG(ERROR) << "Contract module not found or contract function not defined\n";
            return;
        }

        auto module = call_op->getParentOfType< mlir::ModuleOp >();
        assert(module && "Wrap around patch: no module found");

        // Ensure function declaration exists in the module first
        if (mlir::failed(pass.insert_function_declaration(module, contractFuncFromModule))) {
            LOG(ERROR) << "Failed to ensure function declaration for " << contract_function_name
                       << "\n";
            return;
        }

        std::string callee_name     = call_op.getCallee()->str();
        cir::FuncOp callee_function = module.lookupSymbol< cir::FuncOp >(callee_name);
        mlir::Block &entry_block    = callee_function.getBody().front();

        auto target_op = entry_block.getParentOp();
        mlir::OpBuilder builder(target_op);
        builder.setInsertionPointToStart(&entry_block);

        if (!module.lookupSymbol< cir::FuncOp >(contract_function_name)) {
            auto result =
                pass.merge_module_symbol(module, contract_module, contract_function_name);
            if (mlir::failed(result)) {
                LOG(ERROR) << "Failed to insert symbol into module\n";
                return;
            }
        } else {
            LOG(INFO) << "Contract function " << contract_function_name
                      << " already exists in module, skipping merge\n";
        }

        auto contract_func = module.lookupSymbol< cir::FuncOp >(contract_function_name);
        if (!contract_func) {
            LOG(ERROR) << "Contract function " << contract_function_name
                       << " not defined. Patching failed...\n";
            return;
        }

        auto symbol_ref =
            mlir::FlatSymbolRefAttr::get(target_op->getContext(), contract_function_name);
        llvm::SmallVector< mlir::Value > function_args;
        pass.prepare_contract_call_arguments(
            builder, target_op, contract_func, contract, function_args
        );
        auto contract_call_op = builder.create< cir::CallOp >(
            callee_function->getLoc(), symbol_ref,
            contract_func->getResultTypes().size() != 0
                ? contract_func->getResultTypes().front()
                : mlir::Type(),
            function_args
        );

        // Set appropriate attributes based on operation type
        pass.set_instrumentation_call_attributes(contract_call_op, call_op);

        if (should_inline) {
            pass.inline_worklists.push_back(contract_call_op);
        }
    }

} // namespace patchestry::passes
