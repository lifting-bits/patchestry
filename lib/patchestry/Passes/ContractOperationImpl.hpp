/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

 #pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

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

        // Static contracts attach a `contract.static` MLIR attribute on the
        // matched op — no call is emitted and no insertion point is needed,
        // so the signature is intentionally narrow.
        static void emitStaticContract(
            mlir::Operation *targetOp, const ContractInformation &contract
        );

        static void applyContractBefore(
            InstrumentationPass &pass, mlir::Operation *targetOp,
            const ContractInformation &contract, bool shouldInline
        );

        static void applyContractAfter(
            InstrumentationPass &pass, mlir::Operation *targetOp,
            const ContractInformation &contract, bool shouldInline
        );
    };

} // namespace patchestry::passes