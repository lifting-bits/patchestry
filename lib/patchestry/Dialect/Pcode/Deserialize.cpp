/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/Dialect/Pcode/Deserialize.hpp>

namespace patchestry::pc {

    mlir::OwningOpRef< mlir::ModuleOp > deserialize(llvm::json::Value const &json, mcontext_t *mctx) {
        auto loc = mlir::UnknownLoc::get(mctx);
        return mlir::OwningOpRef< mlir::ModuleOp >(mlir::ModuleOp::create(loc));
    }

} // namespace patchestry::pc
