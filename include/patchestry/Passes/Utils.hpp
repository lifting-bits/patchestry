/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace mlir {
    class MLIRContext;
    class Type;
} // namespace mlir

namespace patchestry {
    namespace utils {

        /// Convert C-like type names to CIR types
        mlir::Type convertCTypesToCIRTypes(mlir::MLIRContext *context, std::string type_name);

        /// Convert CIR type back to C-like type name string
        std::string convertCIRTypesToCTypes(mlir::Type cir_type);

    } // namespace utils
} // namespace patchestry
