/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/Util/Common.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <llvm/Support/JSON.h>

namespace patchestry::pc
{
    using json_arr = llvm::json::Array;
    using json_obj = llvm::json::Object;
    using json_val = llvm::json::Value;

    struct deserializer {
        mlir_builder bld;

        explicit deserializer(mlir::ModuleOp mod)
            : bld(mod)
        {
            assert(mod->getNumRegions() > 0 && "Module has no regions.");
            auto &reg = mod->getRegion(0);
            assert(reg.hasOneBlock() && "Region has unexpected blocks.");
            bld.setInsertionPointToStart(&*reg.begin());
        }

        void process(const json_obj &json);
        void process_function(const json_obj &json);
        void process_block(const json_obj &json);
        void process_instruction(const json_obj &json);
    };

    mlir::OwningOpRef< mlir::ModuleOp > deserialize(const json_obj &json, mcontext_t *mctx);

} // namespace patchestry::pc
