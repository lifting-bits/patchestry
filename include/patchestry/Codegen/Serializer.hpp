/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace patchestry::codegen {

    class Serializer
    {
      public:
        static bool serializeToFile(mlir::ModuleOp mod, const std::string &filename);

        static mlir::ModuleOp
        deserializeFromFile(mlir::MLIRContext *mctx, const std::string &filename);

        static std::string convertModuleToString(mlir::ModuleOp mod);
    };
} // namespace patchestry::codegen
