/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/Util/Common.hpp>

#include <llvm/Support/JSON.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

namespace patchestry::pc {

    struct program;
    struct function;
    struct basic_block;
    struct instruction;
    struct pcode;

    using json_arr = llvm::json::Array;
    using json_obj = llvm::json::Object;
    using json_val = llvm::json::Value;

    struct deserializer
    {
        mlir_builder bld;

        explicit deserializer(mlir::ModuleOp mod) : bld(mod) {
            assert(mod->getNumRegions() > 0 && "Module has no regions.");
            auto &reg = mod->getRegion(0);
            assert(reg.hasOneBlock() && "Region has unexpected blocks.");
            bld.setInsertionPointToStart(&*reg.begin());
        }

        void process(const program &prog);
        void process_function(const function &func);
        void process_block(const basic_block &block);
        void process_instruction(const instruction &inst);
        void process_pcode(const pcode &code);

        mlir_operation create_int_const(uint32_t offset, uint32_t size);
        mlir_operation create_varnode(std::string type, uint32_t offset, uint32_t size);
    };

    mlir::OwningOpRef< mlir::ModuleOp > deserialize(const json_obj &json, mcontext_t *mctx);

} // namespace patchestry::pc
