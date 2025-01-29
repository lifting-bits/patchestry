/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>

namespace patchestry::codegen {

    class PassInstrumentation : public mlir::PassInstrumentation
    {
      public:
        explicit PassInstrumentation(bool enable_location_transform = false)
            : location_transform(enable_location_transform) {}

        void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;

        void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override;

      private:
        bool location_transform;
    };

} // namespace patchestry::codegen
