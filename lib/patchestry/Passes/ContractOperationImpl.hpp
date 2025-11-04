#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <clang/CIR/Dialect/IR/CIRDialect.h>

#include <patchestry/YAML/ContractSpec.hpp>

namespace patchestry::passes {

    class InstrumentationPass;

    using ContractMode = patchestry::passes::contract::InfoMode;

    struct ContractInformation
    { // the use of optional here takes care of some typing errors
        std::optional< contract::ContractSpec > spec;
        std::optional< contract::ContractAction > action;
    };

    class ContractOperationImpl
    {
        friend class InstrumentationPass;

      public:
        ContractOperationImpl() = default;

        static void emitRuntimeContract(
            InstrumentationPass &pass, mlir::OpBuilder &builder, mlir::Operation *targetOp,
            mlir::ModuleOp contractModule, const ContractInformation &contract,
            ContractMode mode, bool shouldInline
        );

        static void emitStaticContract(
            InstrumentationPass &pass, mlir::OpBuilder &builder, mlir::Operation *targetOp,
            mlir::ModuleOp contractModule, const ContractInformation &contract,
            ContractMode mode, bool shouldInline
        );

        static void applyContractBefore(
            InstrumentationPass &pass, mlir::Operation *targetOp,
            const ContractInformation &contract, mlir::ModuleOp contractModule,
            bool shouldInline
        );

        static void applyContractAfter(
            InstrumentationPass &pass, mlir::Operation *targetOp,
            const ContractInformation &contract, mlir::ModuleOp contractModule,
            bool shouldInline
        );

        static void applyContractAtEntrypoint(
            InstrumentationPass &pass, cir::CallOp callOp, const ContractInformation &contract,
            mlir::ModuleOp contractModule, bool shouldInline
        );
    };

} // namespace patchestry::passes
