/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Ghidra/Pcode.hpp"
#include "patchestry/Ghidra/PcodeOperations.hpp"
#include <memory>
#include <optional>
#include <unordered_map>

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/Ghidra/JsonDeserialize.hpp>

namespace patchestry::ghidra {

    template< typename ObjectType >
    constexpr std::optional< std::string >
    get_string_if_valid(ObjectType &obj, const char *field) {
        if (auto value = (obj.getString)(field)) {
            if (!value->empty()) {
                return value->str();
            }
        }
        return std::nullopt;
    }

    std::optional< Program > JsonParser::deserialize_program(const JsonObject &root) {
        Program program;
        program.arch   = root.getString("arch").value_or("");
        program.format = root.getString("format").value_or("");

        // Check if root object has types array; if yes then deserialize types
        if (const auto *types_array = root.getObject("types")) {
            deserialize_types(*types_array, program.serialized_types);
        }

        llvm::outs() << "No of types recovered: " << program.serialized_types.size() << "\n";

        if (const auto *function_array = root.getObject("functions")) {
            deserialize_functions(*function_array, program.serialized_functions);
        }

        llvm::outs() << "No of functions recovered: " << program.serialized_functions.size()
                     << "\n";

        if (const auto *globals = root.getObject("globals")) {
            deserialize_globals(*globals, program.serialized_globals);
        }

        llvm::outs() << "No of globals recovered: " << program.serialized_globals.size()
                     << "\n";

        return program;
    }

    // Create varnode type from the json object
    std::shared_ptr< VarnodeType > JsonParser::create_vnode_type(const JsonObject &type_obj) {
        auto name = type_obj.getString("name").value_or("").str();
        auto size = static_cast< uint32_t >(type_obj.getInteger("size").value_or(0));
        auto kind = VarnodeType::convertToKind(type_obj.getString("kind").value_or("").str());
        switch (kind) {
            case VarnodeType::Kind::VT_INVALID: {
                // assert(false); // assert if invalid type is found
                return std::make_shared< VarnodeType >(name, kind, size);
            }
            case VarnodeType::Kind::VT_BOOLEAN:
            case VarnodeType::Kind::VT_INTEGER:
            case VarnodeType::Kind::VT_FLOAT:
            case VarnodeType::Kind::VT_CHAR:
                return std::make_shared< BuiltinType >(name, kind, size);
            case VarnodeType::Kind::VT_ARRAY:
                return std::make_shared< ArrayType >(name, kind, size);
            case VarnodeType::Kind::VT_POINTER:
                return std::make_shared< PointerType >(name, kind, size);
            case VarnodeType::Kind::VT_FUNCTION:
                return std::make_shared< FunctionType >(name, kind, size);
            case VarnodeType::Kind::VT_STRUCT:
            case VarnodeType::Kind::VT_UNION:
                return std::make_shared< CompositeType >(name, kind, size);
            case VarnodeType::Kind::VT_ENUM:
                return std::make_shared< EnumType >(name, kind, size);
            case VarnodeType::VT_TYPEDEF:
                return std::make_shared< TypedefType >(name, kind, size);
            case VarnodeType::VT_UNDEFINED:
                return std::make_shared< UndefinedType >(name, kind, size);
            case VarnodeType::Kind::VT_VOID:
                return std::make_shared< BuiltinType >(name, kind, size);
        }
    }

    // Deserialize types
    void JsonParser::deserialize_types(const JsonObject &type_obj, TypeMap &serialized_types) {
        if (type_obj.size() == 0) {
            llvm::errs() << "No type objects to deserialize\n";
            return;
        }

        std::unordered_map< std::string, const JsonValue & > types_value_map;

        for (const auto &type : type_obj) {
            auto key          = type.getFirst().str();
            const auto &value = type.getSecond();
            auto vnode_type   = create_vnode_type(*value.getAsObject());
            if (!vnode_type) {
                llvm::errs() << "Failed to create varnode type\n";
                assert(false);
                continue;
            }

            // Set type key for map lookup at later point
            vnode_type->set_key(key);
            serialized_types.emplace(key, std::move(vnode_type));
            types_value_map.emplace(key, value);
        }

        llvm::errs() << "Number of entry in serialized types: " << serialized_types.size()
                     << "\n";
        // Post process varnodes from the map and resolve recursive references of labels
        for (const auto &[key, vnode_type] : serialized_types) {
            auto iter = types_value_map.find(key);
            if (iter == types_value_map.end()) {
                assert(false);
                continue;
            }
            const auto &json_value = iter->second;
            switch (vnode_type->kind) {
                case VarnodeType::Kind::VT_BOOLEAN:
                case VarnodeType::Kind::VT_INTEGER:
                case VarnodeType::Kind::VT_FLOAT:
                case VarnodeType::Kind::VT_CHAR:
                case VarnodeType::Kind::VT_VOID:
                    deserialize_buildin(
                        *dynamic_cast< BuiltinType * >(vnode_type.get()),
                        *json_value.getAsObject(), serialized_types
                    );
                    break;
                case VarnodeType::Kind::VT_ARRAY: {
                    deserialize_array(
                        *dynamic_cast< ArrayType * >(vnode_type.get()),
                        json_value.getAsObject(), serialized_types
                    );
                    break;
                }
                case VarnodeType::Kind::VT_POINTER: {
                    deserialize_pointer(
                        *dynamic_cast< PointerType * >(vnode_type.get()),
                        *json_value.getAsObject(), serialized_types
                    );
                    break;
                }
                case VarnodeType::Kind::VT_FUNCTION: {
                    deserialize_function_type(
                        *dynamic_cast< FunctionType * >(vnode_type.get()),
                        *json_value.getAsObject(), serialized_types
                    );
                    break;
                }
                case VarnodeType::Kind::VT_STRUCT:
                case VarnodeType::Kind::VT_UNION: {
                    deserialize_composite(
                        *dynamic_cast< CompositeType * >(vnode_type.get()),
                        *json_value.getAsObject(), serialized_types
                    );
                    break;
                }
                case VarnodeType::Kind::VT_ENUM: {
                    deserialize_enum(
                        *dynamic_cast< EnumType * >(vnode_type.get()),
                        *json_value.getAsObject(), serialized_types
                    );
                    break;
                }
                case VarnodeType::Kind::VT_TYPEDEF: {
                    deserialize_typedef(
                        *dynamic_cast< TypedefType * >(vnode_type.get()),
                        *json_value.getAsObject(), serialized_types
                    );
                    break;
                }
                case VarnodeType::Kind::VT_UNDEFINED:
                    deserialize_undefined_type(
                        *dynamic_cast< UndefinedType * >(vnode_type.get()),
                        *json_value.getAsObject(), serialized_types
                    );
                    break;
                case VarnodeType::Kind::VT_INVALID:
                    break;
            }
        }
    }

    void
    JsonParser::deserialize_buildin(BuiltinType &varnode, const JsonObject &, const TypeMap &) {
        assert(
            varnode.kind == VarnodeType::Kind::VT_BOOLEAN
            || varnode.kind == VarnodeType::Kind::VT_INTEGER
            || varnode.kind == VarnodeType::Kind::VT_CHAR
            || varnode.kind == VarnodeType::Kind::VT_FLOAT
            || varnode.kind == VarnodeType::Kind::VT_VOID
        );
        (void) varnode;
    }

    void JsonParser::deserialize_array(
        ArrayType &varnode, const JsonObject *array_obj, const TypeMap &serialized_types
    ) {
        auto element_label = array_obj->getString("element_type").value_or("").str();
        if (element_label.empty()) {
            llvm::errs() << "Element type of an array is empty. key: " << varnode.key << "\n";
            assert(false);
            return;
        }

        auto iter = serialized_types.find(element_label);
        if (iter == serialized_types.end()) {
            llvm::errs() << "Element type key " << element_label
                         << " not found in serialized types."
                         << " deserializing array with key " << varnode.key << "\n";
            assert(false);
            return;
        }

        varnode.set_element_type(iter->second);
        auto num_elem =
            static_cast< uint32_t >(array_obj->getInteger("num_elements").value_or(0));
        varnode.set_element_count(num_elem);
    }

    void JsonParser::deserialize_pointer(
        PointerType &varnode, const JsonObject &pointer_obj, const TypeMap &serialized_types
    ) {
        auto pointee_key = pointer_obj.getString("element_type").value_or("").str();
        if (pointee_key.empty()) {
            llvm::errs() << "Pointer type with empty pointee key. pointer key: " << varnode.key
                         << "\n";
            assert(false);
            return;
        }

        // Check for the pointee label in serialized types
        auto iter = serialized_types.find(pointee_key);
        if (iter == serialized_types.end()) {
            llvm::errs() << "Pointee type is not availe in serialized types. Pointer key: "
                         << varnode.key << "\n";
            assert(false);
            return;
        }

        varnode.set_pointee_type(iter->second);
    }

    void JsonParser::deserialize_typedef(
        TypedefType &varnode, const JsonObject &typedef_obj, const TypeMap &serialized_types
    ) {
        auto base_key = typedef_obj.getString("base_type").value_or("").str();
        if (base_key.empty()) {
            llvm::errs() << "Base type for the tyepdef is not set. key: " << varnode.key
                         << "\n";
            assert(false);
            return;
        }

        auto iter = serialized_types.find(base_key);
        if (iter == serialized_types.end()) {
            llvm::errs() << "Base type key is not found in serialized types " << base_key
                         << " for typedef key " << varnode.key << "\n";
            assert(false);
            return;
        }

        varnode.set_base_type(iter->second);
    }

    void JsonParser::deserialize_composite(
        CompositeType &varnode, const JsonObject &composite_obj, const TypeMap &serialized_types
    ) {
        const auto *field_array = composite_obj.getArray("fields");
        for (const auto &field : *field_array) {
            const auto *field_obj = field.getAsObject();
            auto field_label      = field_obj->getString("type").value_or("").str();
            if (field_label.empty()) {
                continue;
            }

            auto iter = serialized_types.find(field_label);
            if (iter == serialized_types.end()) {
                llvm::errs() << "Field component is not found on serialized types";
                continue;
            }

            auto field_type   = iter->second;
            auto field_offset = field_obj->getInteger("offset").value_or(-1);
            if (field_offset < 0) {
                continue;
            }

            auto field_name = field_obj->getString("name").value_or("").str();
            varnode.add_components(
                field_name, *field_type, static_cast< uint32_t >(field_offset)
            );
        }
    }

    void JsonParser::deserialize_enum(
        EnumType &varnode, const JsonObject &enum_obj, const TypeMap &serialized_types
    ) {
        assert(varnode.kind == VarnodeType::Kind::VT_ENUM);
        (void) serialized_types;
        (void) varnode;
        (void) enum_obj;
    }

    void JsonParser::deserialize_function_type(
        FunctionType &varnode, const JsonObject &func_obj, const TypeMap &serialized_types
    ) {
        assert(varnode.kind == VarnodeType::Kind::VT_FUNCTION);
        (void) serialized_types;
        (void) varnode;
        (void) func_obj;
    }

    void JsonParser::deserialize_undefined_type(
        UndefinedType &varnode, const JsonObject &undef_obj, const TypeMap &serialized_types
    ) {
        assert(varnode.kind == VarnodeType::Kind::VT_UNDEFINED);
        (void) serialized_types;
        (void) varnode;
        (void) undef_obj;
    }

    // Deserialize operations
    std::optional< Varnode > JsonParser::create_varnode(const JsonObject &var_obj) {
        auto type_key = var_obj.getString("type").value_or("").str();
        auto size     = var_obj.getInteger("size").value_or(0);
        auto kind     = Varnode::convertToKind(var_obj.getString("kind").value_or("").str());

        Varnode vnode(kind, static_cast< uint32_t >(size), type_key);
        auto operation_key = var_obj.getString("operation").value_or("").str();
        if (!operation_key.empty()) {
            vnode.operation = operation_key;
        }

        auto function_key = var_obj.getString("function");
        if (function_key && !function_key->empty()) {
            vnode.function = function_key->str();
        }

        auto value = var_obj.getInteger("value");
        if (value) {
            vnode.value = static_cast< uint32_t >(*value);
        }

        auto global_key = var_obj.getString("global");
        if (global_key && !global_key->empty()) {
            vnode.global = global_key->str();
        }

        return vnode;
    }

    std::optional< Function > JsonParser::create_function(const JsonObject &func_obj) {
        Function func;
        func.name = func_obj.getString("name").value_or("");
        if (const auto *proto_obj = func_obj.getObject("type")) {
            if (auto maybe_prototype = create_function_prototype(*proto_obj)) {
                func.prototype = *maybe_prototype;
            }
        }

        auto entry_block = func_obj.getString("entry_block");
        if (entry_block && !entry_block->empty()) {
            func.entry_block = entry_block->str();
        }

        if (const auto *blocks_array = func_obj.getObject("basic_blocks")) {
            deserialize_blocks(*blocks_array, func.basic_blocks, func.entry_block);
        }

        return func;
    }

    void JsonParser::deserialize_call_operation(const JsonObject &call_obj, Operation &op) {
        if (const auto *maybe_target = call_obj.getObject("target")) {
            OperationTarget target;
            target.kind =
                Varnode::convertToKind(maybe_target->getString("kind").value_or("").str());

            auto function = maybe_target->getString("function");
            if (function.has_value() && !function->empty()) {
                target.function = function->str();
            }
            auto call_op = maybe_target->getString("operation");
            if (call_op.has_value() && !call_op->empty()) {
                target.operation = call_op->str();
            }

            target.is_noreturn = maybe_target->getBoolean("is_noreturn").value_or(false);
            op.target          = target;
        }
    }

    void JsonParser::deserialize_branch_operation(const JsonObject &branch_obj, Operation &op) {
        auto target_block = branch_obj.getString("target_block");
        if (target_block && !target_block->empty()) {
            op.target_block = target_block->str();
        }

        auto taken_block = branch_obj.getString("taken_block");
        if (taken_block && !taken_block->empty()) {
            op.taken_block = taken_block->str();
        }

        auto not_taken_block = branch_obj.getString("not_taken_block");
        if (not_taken_block && !not_taken_block->empty()) {
            op.not_taken_block = not_taken_block->str();
        }

        if (const auto *maybe_output = branch_obj.getObject("condition")) {
            if (auto maybe_varnode = create_varnode(*maybe_output)) {
                op.condition = *maybe_varnode;
            }
        }
    }

    std::optional< Operation > JsonParser::create_operation(const JsonObject &pcode_obj) {
        auto mnemonic =
            patchestry::ghidra::from_string(pcode_obj.getString("mnemonic").value_or("").str());
        if (mnemonic == Mnemonic::OP_UNKNOWN) {
            llvm::errs() << "Pcode with unknown operation\n";
            assert(false);
            return std::nullopt;
        }

        Operation operation;
        operation.mnemonic = mnemonic;
        if (const auto *maybe_output = pcode_obj.getObject("output")) {
            if (auto maybe_varnode = create_varnode(*maybe_output)) {
                operation.output = *maybe_varnode;
            }
        }

        if (const auto *input_array = pcode_obj.getArray("inputs")) {
            for (auto input : *input_array) {
                if (auto maybe_varnode = create_varnode(*input.getAsObject())) {
                    operation.inputs.emplace_back(*maybe_varnode);
                }
            }
        }

        switch (operation.mnemonic) {
            case Mnemonic::OP_CALL:
            case Mnemonic::OP_CALLIND:
                deserialize_call_operation(pcode_obj, operation);
                break;
            case Mnemonic::OP_CBRANCH:
            case Mnemonic::OP_BRANCH:
            case Mnemonic::OP_BRANCHIND:
                deserialize_branch_operation(pcode_obj, operation);
                break;
            default:
                break;
        }

        operation.name    = get_string_if_valid(pcode_obj, "name");
        operation.type    = get_string_if_valid(pcode_obj, "type");
        operation.address = get_string_if_valid(pcode_obj, "address");

        auto index = pcode_obj.getInteger("index");
        if (index) {
            operation.index = static_cast< uint32_t >(*index);
        }

        return operation;
    }

    std::optional< BasicBlock >
    JsonParser::create_basic_block(const std::string &block_key, const JsonObject &block_obj) {
        if (const auto *operations = block_obj.getObject("operations")) {
            BasicBlock block;

            for (const auto &operation : *operations) {
                auto operation_key           = operation.getFirst().str();
                const auto *operation_object = operation.getSecond().getAsObject();
                if (auto maybe_operation = create_operation(*operation_object)) {
                    maybe_operation->key              = operation_key;
                    maybe_operation->parent_block_key = block_key;
                    block.operations.emplace(operation_key, *maybe_operation);
                }
            }

            if (const auto *ordered_operations = block_obj.getArray("ordered_operations")) {
                for (const auto &operation : *ordered_operations) {
                    auto operation_label = operation.getAsString();
                    if (operation_label && !operation_label->empty()) {
                        block.ordered_operations.push_back(operation_label->str());
                    }
                }
            }

            return block;
        }
        return std::nullopt;
    }

    std::optional< FunctionPrototype >
    JsonParser::create_function_prototype(const JsonObject &proto_obj) {
        FunctionPrototype proto;
        const auto return_type = proto_obj.getString("return_type").value_or("").str();
        if (return_type.empty()) {
            llvm::errs() << "FunctionProtoType return type is empty\n";
            assert(false);
            return std::nullopt;
        }

        proto.rttype_key = return_type;

        proto.is_variadic = proto_obj.getBoolean("is_variadic").value_or(false);
        proto.is_noreturn = proto_obj.getBoolean("is_noreturn").value_or(false);

        if (const auto *parameters = proto_obj.getArray("parameter_types")) {
            for (const auto &parameter : *parameters) {
                auto parameter_key = parameter.getAsString();
                if (parameter_key && !parameter_key->empty()) {
                    proto.parameters.push_back(parameter_key->str());
                }
            }
        }

        return proto;
    }

    void JsonParser::deserialize_functions(
        const JsonObject &function_array, FunctionMap &serialized_functions
    ) {
        if (function_array.empty()) {
            llvm::errs() << "No functions to deserialize";
            return;
        }

        for (const auto &func_obj : function_array) {
            auto function_key = func_obj.getFirst().str();
            auto function     = create_function(*func_obj.getSecond().getAsObject());
            if (!function) {
                llvm::errs() << "Failed to get function for the key " << function_key << "\n";
                continue;
            }
            function->key = function_key;
            serialized_functions.emplace(function_key, *function);
        }
    }

    void JsonParser::deserialize_blocks(
        const JsonObject &blocks_array, BasicBlockMap &serialized_blocks,
        std::string &entry_block
    ) {
        if (blocks_array.empty()) {
            llvm::errs() << "No blocks in function\n";
            return;
        }

        for (const auto &block : blocks_array) {
            auto block_key        = block.getFirst().str();
            const auto *block_obj = block.getSecond().getAsObject();
            if (auto maybe_block = create_basic_block(block_key, *block_obj)) {
                if (block_key == entry_block) {
                    maybe_block->is_entry_block = true;
                }
                maybe_block->key = block_key;
                serialized_blocks.emplace(block_key, *maybe_block);
            }
        }
    }

    void JsonParser::deserialize_globals(
        const JsonObject &global_array, VariableMap &serialized_globals
    ) {
        if (global_array.empty()) {
            llvm::errs() << "No global variable to serialize\n";
            return;
        }
        for (const auto &global : global_array) {
            Variable variable;
            variable.key           = global.getFirst().str();
            const auto *global_obj = global.getSecond().getAsObject();
            if (auto maybe_name = global_obj->getString("name")) {
                variable.name = *maybe_name;
            }
            if (auto maybe_type = global_obj->getString("type")) {
                variable.type = *maybe_type;
            }
            if (auto maybe_size = global_obj->getInteger("size")) {
                variable.size = static_cast< uint32_t >(*maybe_size);
            }
            serialized_globals.emplace(variable.key, variable);
        }
    }

} // namespace patchestry::ghidra
