/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <vast/Util/Common.hpp>

namespace patchestry::codegen {

    class Serializer
    {
      public:
        static bool serializeToFile(vast::mlir_module mod, const std::string &filename);

        static vast::mlir_module
        deserializeFromFile(mlir::MLIRContext *mctx, const std::string &filename);

        static std::string convertModuleToString(vast::mlir_module mod);

        static vast::mlir_module
        parseModuleFromString(mlir::MLIRContext *mctx, const std::string &module_string);
    };
} // namespace patchestry::codegen
