/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <mlir/IR/Operation.h>

#include <patchestry/YAML/ContractSpec.hpp>

namespace patchestry::passes {

    struct ContractInformation
    {
        std::optional< contract::ContractSpec >   spec;
        std::optional< contract::ContractAction > action;
    };

    class ContractOperationImpl
    {
      public:
        ContractOperationImpl() = default;

        // Static contracts attach a `contract.static` MLIR attribute on the
        // matched op — no call is emitted and no insertion point is needed,
        // so the signature is intentionally narrow.
        static void emitStaticContract(
            mlir::Operation *target_op, const ContractInformation &contract
        );

        // apply* entry points match the dispatch in
        // `InstrumentationPass::apply_contract_action_to_targets`; both
        // currently delegate to `emitStaticContract` on the matched op.
        static void applyContractBefore(
            mlir::Operation *target_op, const ContractInformation &contract
        );

        static void applyContractAfter(
            mlir::Operation *target_op, const ContractInformation &contract
        );
    };

} // namespace patchestry::passes
