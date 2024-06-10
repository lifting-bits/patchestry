/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Util/Warnings.hpp"

PATCHESTRY_RELAX_WARNINGS
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
PATCHESTRY_UNRELAX_WARNINGS

#include <cassert>
#include <string_view>

#include "patchestry/Ghidra/Deserialize.hpp"

namespace patchestry::ghidra {

    struct mlir_codegen_visitor
    {
        mlir::OpBuilder bld;
        mlir::MLIRContext *ctx;

        using operation_t = mlir::Operation;
        using type_t      = mlir::Type;
        using string_ref  = std::string_view;
        using values_ref  = llvm::ArrayRef< mlir::Value >;

        explicit mlir_codegen_visitor(mlir::ModuleOp mod) : bld(mod), ctx(bld.getContext()) {
            assert(mod->getNumRegions() > 0 && "Module has no regions.");
            auto &reg = mod->getRegion(0);
            assert(reg.hasOneBlock() && "Region has unexpected blocks.");
            bld.setInsertionPointToStart(&*reg.begin());
        }

        auto tmp() -> operation_t * {
            auto loc = bld.getUnknownLoc();
            auto val = bld.getI16IntegerAttr(42);
            return bld.create< mlir::arith::ConstantOp >(loc, val);
        }

        auto visit(const deserialized_t &ref) -> operation_t * {
            return std::visit(*this, ref);
        }

        auto mk_pcode(string_ref mnemonic, type_t result, values_ref inputs) -> operation_t *;
        auto mk_inst(string_ref mnemonic) -> operation_t *;
        auto mk_block(string_ref label) -> operation_t *;
        auto mk_func(string_ref name) -> operation_t *;

        auto operator()([[maybe_unused]] const auto &arg) -> operation_t * {
            assert(false && "Unexpected ghidra type.");
            return nullptr;
        }

        auto operator()(const varnode_t &varnode) -> operation_t *;
        auto operator()(const pcode_t &pcode) -> operation_t *;
        auto operator()(const instruction_t &inst) -> operation_t *;
        auto operator()(const code_block_t &blk) -> operation_t *;
        auto operator()(const function_t &func) -> operation_t *;
    };
} // namespace patchestry::ghidra
