/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <regex>
#include <string>
#include <vector>

#include <patchestry/YAML/ContractSpec.hpp>

// Forward declarations
namespace mlir {
    class Operation;
    class Type;
    class Value;
} // namespace mlir

namespace cir {
    class FuncOp;
} // namespace cir

namespace patchestry::passes {
    /**
     * @brief Provides matching capabilities for MLIR operations.
     *
     * The ContractOperationMatcher class encapsulates logic for determining whether
     * an operation should be instrumented based on contract specification. It supports
     * various matching criteria, but these are more simple than for patching, since 
     * unlike patches, contracts do not replace functionality and do not change program 
     * state. Operation-level matching isn't supported for contracts.
     */
    class ContractOperationMatcher {
        public:
            enum Mode : uint8_t { FUNCTION };

            /**
             * @brief Checks if an operation matches the given patch specification.
             *
             * This is the main entry point for operation matching. It evaluates all
             * matching criteria in the patch specification against the given operation.
             *
             * @param op The operation to evaluate
             * @param func The function containing the operation
             * @param spec The patch specification to match against
             * @return true if the operation matches the specification
             */
            static bool
            matches(mlir::Operation *op, cir::FuncOp func, const contract::ContractAction &spec, Mode mode);

        private:
    }
} // namespace patchestry::passes