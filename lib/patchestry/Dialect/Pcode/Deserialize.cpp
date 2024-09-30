/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Util/Common.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Location.h>

#include <patchestry/Dialect/Pcode/Deserialize.hpp>
#include <patchestry/Dialect/Pcode/Json.hpp>
#include <patchestry/Dialect/Pcode/Pcode.hpp>
#include <patchestry/Dialect/Pcode/PcodeOps.hpp>
#include <patchestry/Dialect/Pcode/PcodeTypes.hpp>

namespace patchestry::pc {

    mlir_value
    create_bitcast_op(mlir_builder &bld, mlir_value &input_val, mlir_type &output_type) {
        return bld.create< mlir::arith::BitcastOp >(
            bld.getUnknownLoc(), output_type, input_val
        );
    }

    mlir_value
    create_truc_op(mlir_builder &bld, mlir_value &input_val, mlir_type &output_type) {
        return bld.create< mlir::arith::TruncIOp >(bld.getUnknownLoc(), output_type, input_val);
    }

    std::optional< program > json_parser::parse_program(const llvm::json::Object &root) {
        program program;
        program.arch = root.getString("arch").value_or("");
        program.os   = root.getString("os").value_or("");

        if (const auto *function_array = root.getArray("functions")) {
            for (const auto &function : *function_array) {
                if (const auto *func_obj = function.getAsObject()) {
                    if (auto parsed_func = parse_function(*func_obj)) {
                        program.functions.push_back(*parsed_func);
                    }
                }
            }
        }

        return program;
    }

    std::optional< pcode > json_parser::parse_pcode(const llvm::json::Object &pcode_obj) {
        pcode pcode;
        pcode.mnemonic = pcode_obj.getString("mnemonic").value_or("");

        if (const auto *output_obj = pcode_obj.getObject("output")) {
            pcode.output.type   = output_obj->getString("type").value_or("");
            pcode.output.offset = output_obj->getInteger("offset");
            pcode.output.size   = output_obj->getInteger("size");
        }

        if (const auto *inputs_array = pcode_obj.getArray("inputs")) {
            for (const auto &input : *inputs_array) {
                if (const auto *input_obj = input.getAsObject()) {
                    pcode::input input;
                    input.type   = input_obj->getString("type").value_or("");
                    input.offset = input_obj->getInteger("offset");
                    input.size   = input_obj->getInteger("size");
                    pcode.inputs.push_back(input);
                }
            }
        }

        return pcode;
    }

    std::optional< instruction >
    json_parser::parse_instruction(const llvm::json::Object &inst_obj) {
        instruction inst;
        inst.mnemonic = inst_obj.getString("mnemonic").value_or("");
        inst.address  = inst_obj.getString("address").value_or("");

        if (const auto *pcode_array = inst_obj.getArray("pcode")) {
            for (const auto &pcode : *pcode_array) {
                if (const auto *pcode_obj = pcode.getAsObject()) {
                    if (auto parsed_pcode = parse_pcode(*pcode_obj)) {
                        inst.pcodes.push_back(*parsed_pcode);
                    }
                }
            }
        }

        return inst;
    }

    std::optional< basic_block >
    json_parser::parse_basic_block(const llvm::json::Object &block_obj) {
        basic_block block;
        block.label = block_obj.getString("label").value_or("");

        if (const auto *instructions_array = block_obj.getArray("instructions")) {
            for (const auto &instruction : *instructions_array) {
                if (const auto *inst_obj = instruction.getAsObject()) {
                    if (const auto parsed_inst = parse_instruction(*inst_obj)) {
                        block.instructions.push_back(*parsed_inst);
                    }
                }
            }
        }

        return block;
    }

    std::optional< function > json_parser::parse_function(const llvm::json::Object &func_obj) {
        function func;
        func.name = func_obj.getString("name").value_or("");

        if (const auto *blocks_array = func_obj.getArray("basic_blocks")) {
            for (const auto &block : *blocks_array) {
                if (const auto *block_obj = block.getAsObject()) {
                    if (auto parsed_block = parse_basic_block(*block_obj)) {
                        func.basic_blocks.push_back(*parsed_block);
                    }
                }
            }
        }

        return func;
    }

    mlir::OwningOpRef< mlir::ModuleOp > deserialize(const json_obj &json, mcontext_t *mctx) {
        // FIXME: use implicit module creation
        auto loc = mlir::UnknownLoc::get(mctx);
        auto mod = mlir::OwningOpRef< mlir::ModuleOp >(mlir::ModuleOp::create(loc));

        deserializer des(mod.get());
        auto program = json_parser().parse_program(json);
        if (program.has_value()) {
            des.process(program.value());
        } else {
            mlir::emitError(loc, "Failed to parse JSON object.");
        }

        return mod;
    }

    mlir_operation deserializer::create_int_const(uint32_t offset, uint32_t size) {
        auto const_type = mlir::IntegerType::get(bld.getContext(), size * 8);
        auto const_attr = mlir::IntegerAttr::get(const_type, offset);
        return bld.create< ConstOp >(bld.getUnknownLoc(), const_attr);
    }

    mlir_operation
    deserializer::create_varnode(std::string type, uint32_t offset, uint32_t size) {
        auto varnode_type = varnode_from_string(type);
        switch (varnode_type) {
            case PCodeVarnodeType::unique_: {
                auto mlir_type = bld.getType< VarType >();
                return bld.create< VarOp >(bld.getUnknownLoc(), mlir_type, type, offset, size);
            }
            case PCodeVarnodeType::const_: {
                return bld.create< ConstOp >(
                    bld.getUnknownLoc(),
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(bld.getContext(), size * 8), offset
                    )
                );
            }
            case PCodeVarnodeType::register_: {
                auto mlir_type = bld.getType< RegType >();
                auto int_type  = bld.getI32Type();
                return bld.create< RegOp >(bld.getUnknownLoc(), int_type, type, offset, size);
            }
            case PCodeVarnodeType::ram_: {
                auto mlir_type = bld.getType< MemType >();
                return bld.create< RegOp >(bld.getUnknownLoc(), mlir_type, type, offset, size);
            }
            default:
                break;
        }
        return {};
    }

    void deserializer::process(const program &prog) {
        if (prog.functions.empty()) {
            mlir::emitError(bld.getUnknownLoc(), "No function to process!");
            return;
        }

        for (const auto &func : prog.functions) {
            process_function(func);
        }
    }

    void deserializer::process_function(const function &func) {
        if (func.name.empty()) {
            mlir::emitError(bld.getUnknownLoc(), "Function name is missing.");
            return;
        }

        auto _  = insertion_guard(bld);
        auto fn = bld.create< pc::FuncOp >(bld.getUnknownLoc(), func.name);

        bld.setInsertionPointToStart(bld.createBlock(&fn.getBlocks()));
        for (const auto &block : func.basic_blocks) {
            process_block(block);
        }
    }

    void deserializer::process_block(const basic_block &block) {
        if (block.label.empty()) {
            mlir::emitError(bld.getUnknownLoc(), "Basic block is missing label name.");
            return;
        }

        auto _          = insertion_guard(bld);
        auto mlir_block = bld.create< pc::BlockOp >(bld.getUnknownLoc(), block.label);

        bld.createBlock(&mlir_block.getInstructions());
        if (block.instructions.empty()) {
            mlir::emitError(bld.getUnknownLoc(), "Block does not have instruction.");
            return;
        }

        for (const auto &inst : block.instructions) {
            process_instruction(inst);
        }
    }

    void deserializer::process_instruction(const instruction &inst) {
        if (inst.mnemonic.empty()) {
            mlir::emitError(bld.getUnknownLoc(), "Instruction mnemonic is missing.");
            return;
        }

        auto _     = insertion_guard(bld);
        auto block = bld.create< pc::InstOp >(bld.getUnknownLoc(), inst.mnemonic);

        bld.createBlock(&block.getSemantics());
        if (inst.pcodes.empty()) {
            mlir::emitError(bld.getUnknownLoc(), "Instruction has no pcode");
            return;
        }

        for (const auto &pcode : inst.pcodes) {
            process_pcode(pcode);
        }
    }

    void deserializer::process_pcode(const pcode &code) {
        if (code.mnemonic.empty()) {
            mlir::emitError(bld.getUnknownLoc(), "Pcode mnemonic is missing.");
            return;
        }

        switch (from_string(code.mnemonic)) {
            case PCodeMnemonic::COPY: {
                const auto &output = code.output;
                const auto &input0 = code.inputs.front();

                auto *output_op =
                    create_varnode(output.type, output.offset.value(), output.size.value());
                auto *input_op =
                    create_varnode(input0.type, input0.offset.value(), input0.size.value());

                mlir::Type var_type = bld.getI32Type();
                mlir::Value var_result =
                    bld.create< VarOp >(bld.getUnknownLoc(), var_type, "input", 8, 8)
                        .getResult();
                bld.create< CopyOp >(bld.getUnknownLoc(), bld.getI32Type(), var_result);
                break;
            }
            case PCodeMnemonic::LOAD: {
                break;
            }
            case PCodeMnemonic::RETURN: {
                const auto &input0 = code.inputs.front();
                auto *input_op =
                    create_varnode(input0.type, input0.offset.value(), input0.size.value());
                bld.create< ReturnOp >(bld.getUnknownLoc(), input_op->getResult(0));
                break;
            }
            default:
                break;
        }
    }

} // namespace patchestry::pc
