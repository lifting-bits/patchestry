/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Ghidra/Codegen.hpp"
#include "patchestry/Dialect/Pcode/PcodeOps.hpp"
#include "patchestry/Dialect/Pcode/PcodeTypes.hpp"

namespace patchestry::ghidra {

    using cg = mlir_codegen_visitor;

    auto cg::mk_pcode(string_ref mnemonic, type_t result, values_ref inputs) -> operation_t {
        auto loc = bld.getUnknownLoc();

        auto mk_unary_op = [&]< typename OpTy > {
            return bld.create< OpTy >(loc, result, inputs[0]);
        };

        auto mk_bin_op = [&]< typename OpTy > {
            return bld.create< OpTy >(loc, result, inputs[0], inputs[1]);
        };

        // clang-format off
        if (mnemonic == "COPY")         { return mk_unary_op.template operator()< pc::CopyOp >(); }
        if (mnemonic == "POPCOUNT")     { return mk_unary_op.template operator()< pc::PopcountOp >(); }
        if (mnemonic == "BOOL_NEGATE")  { return mk_unary_op.template operator()< pc::BoolNegateOp >(); }
        
        if (mnemonic == "INT_LESS")     { return mk_bin_op.template operator()< pc::IntLessOp >(); }
        if (mnemonic == "INT_EQUAL")    { return mk_bin_op.template operator()< pc::IntEqualOp >(); }
        if (mnemonic == "INT_SLESS")    { return mk_bin_op.template operator()< pc::IntSLessOp >(); }
        if (mnemonic == "INT_ADD")      { return mk_bin_op.template operator()< pc::IntAddOp >(); }
        if (mnemonic == "INT_SUB")      { return mk_bin_op.template operator()< pc::IntSubOp >(); }
        if (mnemonic == "INT_SBORROW")  { return mk_bin_op.template operator()< pc::IntSBorrowOp >(); }
        if (mnemonic == "INT_AND")      { return mk_bin_op.template operator()< pc::IntAndOp >(); }
        
        if (mnemonic == "BRANCH")   { return bld.create< pc::BranchOp >(loc, inputs[0]); }
        if (mnemonic == "CBRANCH")  { return bld.create< pc::CBranchOp >(loc, inputs[0], inputs[1]); }
        if (mnemonic == "CALL")     { return bld.create< pc::CallOp >(loc, inputs[0]); }
        if (mnemonic == "RETURN")   { return bld.create< pc::ReturnOp >(loc, inputs[0]); }
        
        if (mnemonic == "STORE"){ return bld.create< pc::StoreOp >(loc, inputs[0], inputs[1], inputs[2]); }
        if (mnemonic == "LOAD") { return bld.create< pc::LoadOp >(loc, result, inputs[0], inputs[1]); }
        // clang-format on

        assert(false && "Unknown pcode operation.");

        return nullptr;
    }

    auto cg::mk_inst(string_ref mnemonic) -> operation_t {
        return bld.create< pc::InstOp >(bld.getUnknownLoc(), mnemonic);
    }

    auto cg::mk_block(string_ref label) -> operation_t {
        return bld.create< pc::BlockOp >(bld.getUnknownLoc(), label);
    }

    auto cg::mk_func(string_ref name) -> operation_t {
        return bld.create< pc::FuncOp >(bld.getUnknownLoc(), name);
    }

    auto cg::mk_varnode(const varnode_t &varnode) -> value_t {
        if (varnode.address_space == "unique") {
            return unique_as[varnode.address];
        }

        auto loc = bld.getUnknownLoc();

        if (varnode.address_space == "const") {
            auto type  = bld.getIntegerType(static_cast< unsigned >(varnode.size));
            auto value = bld.getIntegerAttr(type, varnode.address);
            return bld.create< pc::ConstOp >(loc, value);
        }

        auto addr_space = bld.getStringAttr(varnode.address_space);
        auto addr       = bld.getI64IntegerAttr(varnode.address);
        auto size       = bld.getI8IntegerAttr(static_cast< int8_t >(varnode.size));

        if (varnode.address_space == "register") {
            return bld.create< pc::RegOp >(
                loc, bld.getType< pc::RegType >(), addr_space, addr, size
            );
        }

        if (varnode.address_space == "ram") {
            return bld.create< pc::MemOp >(
                loc, bld.getType< pc::MemType >(), addr_space, addr, size
            );
        }

        return bld.create< pc::VarOp >(
            loc, bld.getType< pc::VarType >(), addr_space, addr, size
        );
    }

    auto cg::operator()(const pcode_t &pcode) -> operation_t {
        std::vector< mlir::Value > inputs;
        inputs.reserve(pcode.inputs.size());
        for (const auto &input : pcode.inputs) {
            inputs.push_back(mk_varnode(input));
        }

        if (pcode.output) {
            auto bitwidth    = static_cast< unsigned >(pcode.output->size * 8);
            auto type        = bld.getIntegerType(bitwidth);
            operation_t pcop = mk_pcode(pcode.mnemonic, type, inputs);

            unique_as[pcode.output->address] = pcop->getResult(0);

            return pcop;
        }

        return mk_pcode(pcode.mnemonic, bld.getNoneType(), inputs);
    }

    auto cg::operator()(const instruction_t &inst) -> operation_t {
        mlir::OpBuilder::InsertionGuard guard(bld);
        auto *iop = mk_inst(inst.mnemonic);

        if (inst.semantics.empty()) {
            return iop;
        }

        bld.createBlock(&iop->getRegion(0));
        for (const auto &pcode : inst.semantics) {
            visit(pcode);
        }
        return iop;
    }

    auto cg::operator()(const code_block_t &blk) -> operation_t {
        mlir::OpBuilder::InsertionGuard guard(bld);
        auto *bop = mk_block(blk.label);

        if (blk.instructions.empty()) {
            return bop;
        }

        bld.createBlock(&bop->getRegion(0));
        for (const auto &inst : blk.instructions) {
            visit(inst);
        }
        return bop;
    }

    auto cg::operator()(const function_t &func) -> operation_t {
        auto *fop = mk_func(func.name);

        if (func.basic_blocks.empty()) {
            return fop;
        }

        bld.createBlock(&fop->getRegion(0));
        for (const auto &blk : func.basic_blocks) {
            visit(blk);
        }
        return fop;
    }
} // namespace patchestry::ghidra
