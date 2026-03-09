/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace llvm {
    class Module;
}

namespace patchestry::codegen {

    class Serializer
    {
      public:
        static bool SerializeToFile(mlir::ModuleOp mod, const std::string &filename);

        static bool SerializeToFile(llvm::Module *mod, const std::string &filename);

        static mlir::ModuleOp
        DeserializeFromFile(mlir::MLIRContext *mctx, const std::string &filename);

        static std::string ConvertModuleToString(mlir::ModuleOp mod);
    };
} // namespace patchestry::codegen
