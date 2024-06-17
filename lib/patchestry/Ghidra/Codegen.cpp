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

    auto cg::mk_pcode(string_view mnemonic, type_t result, values_ref inputs) -> operation_t {
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

    auto cg::mk_inst(string_view mnemonic) -> operation_t {
        return bld.create< pc::InstOp >(bld.getUnknownLoc(), mnemonic);
    }

    auto cg::mk_block(string_view label) -> operation_t {
        return bld.create< pc::BlockOp >(bld.getUnknownLoc(), label);
    }

    auto cg::mk_func(string_view name) -> operation_t {
        return bld.create< pc::FuncOp >(bld.getUnknownLoc(), name);
    }

    auto cg::get_type(const varnode_t &var) -> type_t {
        auto bitwidth = static_cast< unsigned >(var.size * 8);
        return llvm::StringSwitch< type_t >(var.address_space)
            .Cases("unique", "const", bld.getIntegerType(bitwidth))
            .Case("register", bld.getType< pc::RegType >())
            .Case("ram", bld.getType< pc::MemType >())
            .Default(bld.getType< pc::VarType >());
    }

    auto cg::mk_varnode(const varnode_t &var) -> value_t {
        if (var.address_space == "unique") {
            return unique_as[var.address];
        }

        if (var.address_space == "register" && register_as.contains(var.address)) {
            return register_as[var.address];
        }

        auto loc  = bld.getUnknownLoc();
        auto type = get_type(var);

        if (var.address_space == "const") {
            return bld.create< pc::ConstOp >(loc, bld.getIntegerAttr(type, var.address));
        }

        auto mk_var_op = [&]< typename OpTy > {
            auto adsp = bld.getStringAttr(var.address_space);
            auto addr = bld.getI64IntegerAttr(var.address);
            auto size = bld.getI8IntegerAttr(static_cast< int8_t >(var.size));
            return bld.create< OpTy >(loc, type, adsp, addr, size);
        };

        if (var.address_space == "register") {
            auto rop = mk_var_op.template operator()< pc::RegOp >();

            register_as[var.address] = rop.getResult();

            return rop;
        }

        if (var.address_space == "ram") {
            return mk_var_op.template operator()< pc::MemOp >();
        }

        return mk_var_op.template operator()< pc::VarOp >();
    }

    auto cg::operator()(const pcode_t &pcode) -> operation_t {
        std::vector< mlir::Value > inputs;
        inputs.reserve(pcode.inputs.size());
        for (const auto &input : pcode.inputs) {
            inputs.push_back(mk_varnode(input));
        }

        operation_t pcop = mk_pcode(pcode.mnemonic, get_type(pcode.output), inputs);

        if (!pcode.output) {
            return pcop;
        }

        if (pcode.output->address_space == "unique") {
            unique_as[pcode.output->address] = pcop->getResult(0);
        }

        if (pcode.output->address_space == "register") {
            register_as[pcode.output->address] = pcop->getResult(0);
        }

        return pcop;
    }

    auto cg::operator()(const instruction_t &inst) -> operation_t {
        mlir::OpBuilder::InsertionGuard guard(bld);
        operation_t iop = mk_inst(inst.mnemonic);

        if (inst.semantics.empty()) {
            return iop;
        }

        bld.createBlock(&iop->getRegion(0));
        for (const auto &pcode : inst.semantics) {
            visit(pcode);
        }

        unique_as.clear();
        register_as.clear();

        return iop;
    }

    auto cg::operator()(const code_block_t &blk) -> operation_t {
        mlir::OpBuilder::InsertionGuard guard(bld);
        operation_t bop = mk_block(blk.label);

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
        mlir::OpBuilder::InsertionGuard guard(bld);
        operation_t fop = mk_func(func.name);

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
