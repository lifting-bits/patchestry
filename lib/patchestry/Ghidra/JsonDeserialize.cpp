/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <optional>
#include <unordered_map>

#include <llvm/Support/raw_ostream.h>

#include <patchestry/Ghidra/JsonDeserialize.hpp>

namespace patchestry::ghidra {

    std::optional< program > json_parser::parse_program(const json_obj &root) {
        program program;
        program.arch   = root.getString("arch").value_or("");
        program.format = root.getString("format").value_or("");

        // Process types from the serialized json
        if (const auto *types_array = root.getObject("types")) {
            for (const auto &type : *types_array) {
                auto type_label = type.first;
                if (const auto *type_obj = type.second.getAsObject()) {
                    auto var_type = parse_type(*type_obj);
                    if (var_type.has_value()) {
                        program.types.emplace(std::pair(type_label.str(), var_type.value()));
                    }
                }
            }
        }

        llvm::outs() << "No of types recovered: " << program.types.size();

        if (const auto *function_array = root.getObject("functions")) {
            for (const auto &function : *function_array) {
                auto addr = function.first;
                llvm::outs() << addr.str();
                if (const auto *func_obj = function.second.getAsObject()) {
                    if (auto parsed_func = parse_function(*func_obj)) {
                        program.functions.push_back(*parsed_func);
                    }
                }
            }
        }

        return program;
    }

    void
    json_parser::process_serialized_types(std::unordered_map< std::string, varnode_type > &types
    ) {
        for (auto &[key, type] : types) {
            switch (type.type_kind) {
                case varnode_type::vt_pointer:
                    break;
                case varnode_type::vt_typedef:
                    break;
                default:
                    break;
            }
        }
    }

    std::optional< varnode > json_parser::parse_varnode(const json_obj &var_obj) {
        varnode variable;
        auto type_label = var_obj.getString("type").value_or("");
        if (!type_label.empty()) {
            // assign type to variables
        }

        auto type_size = var_obj.getInteger("size").value_or(0);
        variable.size  = static_cast< uint32_t >(type_size);

        auto var_kind     = var_obj.getString("kind").value_or("");
        variable.var_kind = varnode::convert_to_kind(var_kind.str());
        return variable;
    }

    std::optional< pcode > json_parser::parse_pcode(const json_obj &pcode_obj) {
        pcode operation;
        operation.mnemonic = pcode_obj.getString("mnemonic").value_or("");
        operation.name     = pcode_obj.getString("name").value_or("");

        if (auto maybe_output = pcode_obj.getObject("output")) {
            auto maybe_varnode = parse_varnode(*maybe_output);
            if (maybe_varnode) {
                operation.output.push_back(*maybe_varnode);
            }
        }

        if (auto input_array = pcode_obj.getArray("input")) {
            for (auto input : *input_array) {
                auto maybe_varnode = parse_varnode(*input.getAsObject());
                if (maybe_varnode) {
                    operation.inputs.emplace_back(*maybe_varnode);
                }
            }
        }

        return operation;
    }

    std::optional< basic_block > json_parser::parse_basic_block(const json_obj &block_obj) {
        basic_block block;

        if (const auto *ordered_operations = block_obj.getArray("ordered_operations")) {
            for (const auto &operation : *ordered_operations) {
                auto operation_label = operation.getAsString();
                llvm::outs() << "operations label " << operation_label->str();
                if (auto operation_obj = block_obj.getObject(*operation_label)) {
                    auto ops = parse_pcode(*operation_obj);
                    block.ops.push_back(*ops);
                }
            }
        }

        return block;
    }

    std::optional< function > json_parser::parse_function(const json_obj &func_obj) {
        function func;
        func.name = func_obj.getString("name").value_or("");
        if (const auto *proto_obj = func_obj.getObject("prototype")) {
            if (auto maybe_prototype = parse_function_prototype(*proto_obj)) {
                func.prototype = *maybe_prototype;
            }
        }

        if (const auto *blocks_array = func_obj.getObject("basic_blocks")) {
            for (const auto &block : *blocks_array) {
                auto block_label = block.first.str();
                llvm::outs() << "block label: " << block_label;
                const auto *block_obj = block.second.getAsObject();
                if (auto maybe_block = parse_basic_block(*block_obj)) {
                    func.basic_blocks.push_back(*maybe_block);
                }
            }
        }

        return func;
    }

    std::optional< function_prototype >
    json_parser::parse_function_prototype(const json_obj &proto_obj) {
        function_prototype proto;
        auto parameters = proto_obj.getArray("parameters");
        for (auto param : *parameters) {
            auto param_name = param.getAsObject()->getString("name").value_or("");
            auto param_type = param.getAsObject()->getString("type");
            (void) param_name;
            (void) param_type;
        }
        return proto;
    }

    std::optional< varnode_type > json_parser::parse_type(const json_obj &type_obj) {
        varnode_type type;
        auto type_name = type_obj.getString("name").value_or("");
        auto type_kind =
            varnode_type::convert_to_kind(type_obj.getString("kind").value_or("").str());
        auto type_size = type_obj.getInteger("size").value_or(0);

        type.name      = type_name;
        type.type_kind = type_kind;
        type.size      = static_cast< uint32_t >(type_size);
        return type;
    }

} // namespace patchestry::ghidra
