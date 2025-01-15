/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <mlir/IR/Operation.h>
#include <patchestry/Codegen/PassInstrumentation.hpp>

namespace patchestry::codegen {
    void PassInstrumentation::runAfterPass(mlir::Pass *pass, mlir::Operation *op) {
        llvm::outs() << "After running pipeline '" << pass->getArgument() << "\n";
        (void) op;
        (void) location_transform;
    }

    void PassInstrumentation::runBeforePass(mlir::Pass *pass, mlir::Operation *op) {
        llvm::outs() << "Before uunning pipeline '" << pass->getArgument() << "\n";
        (void) op;
        (void) location_transform;
    }
} // namespace patchestry::codegen
