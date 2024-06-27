/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Ghidra/Codegen.hpp"
#include "patchestry/Dialect/Pcode/PcodeTypes.hpp"

namespace patchestry::ghidra {

    using cg = mlir_codegen_visitor;

    auto cg::create_translation_map() -> translation_map_t {

        return {
            { "COPY", op_name< pc::CopyOp >() },
            { "POPCOUNT", op_name< pc::PopcountOp >() },

            { "BOOL_NEGATE", op_name< pc::BoolNegateOp >() },

            { "INT_LESS", op_name< pc::IntLessOp >() },
            { "INT_EQUAL", op_name< pc::IntEqualOp >() },
            { "INT_SLESS", op_name< pc::IntSLessOp >() },

            { "INT_ADD", op_name< pc::IntAddOp >() },
            { "INT_SUB", op_name< pc::IntSubOp >() },

            { "INT_SBORROW", op_name< pc::IntSBorrowOp>() },
            { "INT_AND", op_name< pc::IntAndOp >() },

            { "BRANCH", op_name< pc::BranchOp >() },
            { "CBRANCH", op_name< pc::CBranchOp >() },

            { "CALL", op_name< pc::CallOp >() },
            { "RETURN", op_name< pc::ReturnOp >() },

            { "STORE", op_name< pc::StoreOp >() },
            { "LOAD", op_name< pc::LoadOp >() }
        };
    }

    auto cg::mk_pcode(string_view mnemonic, mlir_type result, values_ref inputs) -> mlir_operation {
        auto loc = bld.getUnknownLoc();

        auto it = opcode_to_op.find(mnemonic);
        assert(it != opcode_to_op.end());

        return bld.create(loc, it->second, inputs, { result });
    }

    auto cg::get_type(const varnode_t &var) -> mlir_type {
        auto adsp = var.address_space;

        if (adsp == "unique" || adsp == "const") {
            auto bitwidth = static_cast< unsigned >(var.size * 8);
            return bld.getIntegerType(bitwidth);
        }

        if (adsp == "register") {
            return bld.getType< pc::RegType >();
        }

        if (adsp == "ram") {
            return bld.getType< pc::MemType >();
        }

        return bld.getType< pc::VarType >();
    }

    auto cg::mk_varnode(const varnode_t &var) -> mlir_value {
        if (auto val = memory.lookup({ var.address_space, var.address })) {
            return val;
        }

        assert(var.address_space != "unique" && "Undefined unique varnode.");

        auto loc  = bld.getUnknownLoc();
        auto type = get_type(var);

        if (var.address_space == "const") {
            auto cop = bld.create< pc::ConstOp >(loc, bld.getIntegerAttr(type, var.address));
            memory.insert({ var.address_space, var.address }, cop);
            return cop;
        }

        auto mk_var_op = [&]< typename OpTy > {
            auto adsp = bld.getStringAttr(var.address_space);
            auto addr = bld.getI64IntegerAttr(var.address);
            auto size = bld.getI8IntegerAttr(static_cast< int8_t >(var.size));
            return bld.create< OpTy >(loc, type, adsp, addr, size);
        };

        if (var.address_space == "register") {
            auto rop = mk_var_op.template operator()< pc::RegOp >();
            memory.insert({ var.address_space, var.address }, rop);
            return rop;
        }

        // TODO(surovic): Maybe this could be treated the same as "reg"?
        if (var.address_space == "ram") {
            return mk_var_op.template operator()< pc::MemOp >();
        }

        return mk_var_op.template operator()< pc::VarOp >();
    }

    auto cg::operator()(const pcode_t &pcode) -> mlir_operation {
        std::vector< mlir::Value > inputs;
        inputs.reserve(pcode.inputs.size());
        for (const auto &input : pcode.inputs) {
            inputs.push_back(mk_varnode(input));
        }

        mlir_operation pcop = mk_pcode(pcode.mnemonic, get_type(pcode.output), inputs);

        if (!pcode.output) {
            return pcop;
        }

        const auto &adsp = pcode.output->address_space;
        const auto &addr = pcode.output->address;

        if (adsp == "unique" || adsp == "register") {
            memory.insert({ adsp, addr }, pcop->getResult(0));
        }

        return pcop;
    }
} // namespace patchestry::ghidra
