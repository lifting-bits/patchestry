/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Ghidra/Deserialize.hpp"

namespace patchestry::ghidra {

    auto error(const std::string_view msg) -> llvm::Error {
        return llvm::createStringError(llvm::inconvertibleErrorCode(), msg);
    }

    auto varnode_t::from_json(const json_arr &varnode_arr) -> expected< varnode_t > {
        auto address_space = varnode_arr[0].getAsString();
        if (!address_space) {
            return error("invalid json value for varnode address_space");
        }

        auto address = varnode_arr[1].getAsInteger();
        if (!address) {
            return error("invalid json value for varnode address");
        }

        auto size = varnode_arr[2].getAsUINT64();
        if (!size) {
            return error("invalid json value for varnode size");
        }

        return varnode_t{ .address_space = address_space->str(),
                          .address       = *address,
                          .size          = *size };
    }

    auto pcode_t::from_json(const json_obj &pcode_obj) -> expected< pcode_t > {
        auto mnemonic = pcode_obj.getString("mnemonic");
        if (!mnemonic) {
            return error("invalid json value for pcode op mnemonic");
        };

        std::optional< varnode_t > output;

        const auto *output_val = pcode_obj.get("output");
        if (output_val == nullptr) {
            return error("missing json value for pcode op output varnode");
        }

        switch (output_val->kind()) {
            case json_val::Null:
                output = {};
                break;
            case json_val::Array: {
                auto exp = varnode_t::from_json(*output_val->getAsArray());
                if (!exp) {
                    return exp.takeError();
                }
                output = *exp;
            } break;
            default:
                return error("invalid json value for pcode op output varnode");
        }

        const auto *array = pcode_obj.getArray("inputs");
        if (array == nullptr) {
            return error("invalid json value for pcode op inputs array");
        }

        std::vector< varnode_t > inputs;

        for (const json_val &elm : *array) {
            if (const json_arr *varnode_arr = elm.getAsArray()) {
                auto exp = varnode_t::from_json(*varnode_arr);
                if (!exp) {
                    return exp.takeError();
                }
                inputs.push_back(*exp);
            }
        }

        return pcode_t{ .mnemonic = mnemonic->str(), .output = output, .inputs = inputs };
    }

    auto instruction_t::from_json(const json_obj &inst_obj) -> expected< instruction_t > {
        auto mnemonic = inst_obj.getString("mnemonic");
        if (!mnemonic) {
            return error("invalid json value for instruction mnemonic");
        };

        const auto *array = inst_obj.getArray("pcode");
        if (array == nullptr) {
            return error("invalid json value for instruction pcode op array");
        }

        std::vector< pcode_t > semantics;

        for (const json_val &elm : *array) {
            if (const json_obj *pcode_obj = elm.getAsObject()) {
                auto exp = pcode_t::from_json(*pcode_obj);
                if (!exp) {
                    return exp.takeError();
                }
                semantics.push_back(*exp);
            }
        }

        return instruction_t{ .mnemonic = mnemonic->str(), .semantics = semantics };
    }

    auto code_block_t::from_json(const json_obj &block_obj) -> expected< code_block_t > {
        auto label = block_obj.getString("label");
        if (!label) {
            return error("invalid json value for basic block label");
        }

        const auto *array = block_obj.getArray("instructions");
        if (array == nullptr) {
            return error("invalid json value for basic block instruction array");
        }

        std::vector< instruction_t > insts;

        for (const json_val &elm : *array) {
            if (const json_obj *inst_obj = elm.getAsObject()) {
                auto exp = instruction_t::from_json(*inst_obj);
                if (!exp) {
                    return exp.takeError();
                }
                insts.push_back(*exp);
            }
        }

        return code_block_t{ .label = label->str(), .instructions = insts };
    }

    auto function_t::from_json(const json_obj &func_obj) -> expected< function_t > {
        auto name = func_obj.getString("name");
        if (!name) {
            return error("invalid json value for function name");
        }

        const auto *array = func_obj.getArray("basic_blocks");
        if (array == nullptr) {
            return error("invalid json value for function basic block array");
        }

        std::vector< code_block_t > blocks;

        for (const json_val &elm : *array) {
            if (const json_obj *block_obj = elm.getAsObject()) {
                auto exp = code_block_t::from_json(*block_obj);
                if (!exp) {
                    return exp.takeError();
                }
                blocks.push_back(*exp);
            }
        }

        return function_t{ .name = name->str(), .basic_blocks = blocks };
    }

} // namespace patchestry::ghidra
