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

        if (!contractModule.lookupSymbol< cir::FuncOp >(contractFunctionName)) {
            LOG(ERROR) << "Contract module not found or contract function not defined: "
                       << contractFunctionName << "\n";
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

        // check if the contract has constraints
        assert(spec.constraints && "Static contracts should have constraints");

        auto ctx = targetOp->getContext();
        llvm::SmallVector< mlir::NamedAttribute > specAttrs;

        // add the name, id, and mode to the spec attributes
        specAttrs.push_back(mlir::NamedAttribute(
            mlir::StringAttr::get(ctx, "name"), mlir::StringAttr::get(ctx, spec.name)
        ));
        specAttrs.push_back(mlir::NamedAttribute(
            mlir::StringAttr::get(ctx, "id"), mlir::StringAttr::get(ctx, spec.id)
        ));
        specAttrs.push_back(mlir::NamedAttribute(
            mlir::StringAttr::get(ctx, "mode"),
            mlir::StringAttr::get(ctx, infoModeToString(mode))
        ));

        // add all_of attribute to the spec attributes
        llvm::SmallVector< mlir::Attribute > constraintsAttrs;
        for (const auto &constraint : spec.constraints.value()) {
            LOG(INFO) << "Processing static contract constraint: " << constraint.condition
                      << "\n";
            llvm::SmallVector< mlir::NamedAttribute > constraintAttrs;
            constraintAttrs.push_back(mlir::NamedAttribute(
                mlir::StringAttr::get(ctx, "condition"),
                mlir::StringAttr::get(ctx, constraint.condition)
            ));
            constraintAttrs.push_back(mlir::NamedAttribute(
                mlir::StringAttr::get(ctx, "description"),
                mlir::StringAttr::get(ctx, constraint.description)
            ));
            constraintsAttrs.push_back(mlir::DictionaryAttr::get(ctx, constraintAttrs));
        }

        // use all_ofAttr to create a constraint
        auto allof_attr = ::contracts::all_ofAttr::get(ctx, constraintsAttrs);
        // add the all_of attribute to the spec attributes
        specAttrs.push_back(
            mlir::NamedAttribute(mlir::StringAttr::get(ctx, "constraints"), allof_attr)
        );

        auto specDictAttr = mlir::DictionaryAttr::get(ctx, specAttrs);
        targetOp->setAttr("contract.spec", specDictAttr);
        (void) contractModule;
        (void) shouldInline;
        (void) pass;
        (void) builder;
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
        if (!contract_module.lookupSymbol< cir::FuncOp >(contract_function_name)) {
            LOG(ERROR) << "Contract module not found or contract function not defined\n";
            return;
        }

        auto module = call_op->getParentOfType< mlir::ModuleOp >();
        assert(module && "Wrap around patch: no module found");
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
