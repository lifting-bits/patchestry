/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <algorithm>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <optional>
#include <unordered_map>

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>
#include <vector>

namespace patchestry::ghidra {

    std::optional< std::string > stripNull(std::optional< llvm::StringRef > stref) {
        if (stref && !stref->empty()) {
            auto str = stref->str();
            str.erase(std::remove(str.begin(), str.end(), '\00'), str.end());
            str.erase(std::remove(str.begin(), str.end(), '\01'), str.end());
            return str;
        }
        return std::nullopt;
    }

    template< typename ObjectType >
    constexpr std::optional< std::string >
    get_string_if_valid(const ObjectType &obj, const char *field) {
        if (auto value = (obj.getString)(field)) {
            if (!value->empty()) {
                return value->str();
            }
        }
        return std::nullopt;
    }

    template< typename ObjectType >
    constexpr std::optional< std::string >
    get_string_with_empty(const ObjectType &obj, const char *field) {
        if (auto value = (obj.getString)(field)) {
            // Return the string even if it's empty
            return value->str();
        }
        return std::nullopt;
    }

    template< typename ObjectType >
    std::string
    get_string(const ObjectType &obj, const char *field, std::string default_value = "") {
        if (auto value = (obj.getString)(field)) {
            if (!value->empty()) {
                return value->str();
            }
        }
        return default_value;
    }

    [[maybe_unused]] static std::string
    dump_json_object(const JsonObject &object) {
        std::string result;
        llvm::raw_string_ostream os(result);

        llvm::json::Object copy = object;  // Make a copy
        os << llvm::json::Value(std::move(copy));

        return result;
    }

    std::optional< Program > JsonParser::deserialize_program(const JsonObject &root) {
        Program program;
        program.arch   = get_string_if_valid(root, "architecture");
        program.lang   = get_string_if_valid(root, "id");
        program.format = get_string_if_valid(root, "format");

        if (!program.arch.has_value()) {
            LOG(ERROR) << "Required field 'architecture' is absent from JSON\n";
            return std::nullopt;
        }
        if (!program.lang.has_value()) {
            LOG(ERROR) << "Required field 'id' is absent from JSON\n";
            return std::nullopt;
        }

        // Deserialize types recovered
        if (const auto *types_array = root.getObject("types")) {
            deserialize_types(*types_array, program.serialized_types);
        }

        // Deserialize functions recovered
        if (const auto *function_array = root.getObject("functions")) {
            deserialize_functions(*function_array, program.serialized_functions);
        }

        // Deserialize global variables recovered
        if (const auto *globals = root.getObject("globals")) {
            deserialize_globals(*globals, program.serialized_globals);
        }

        LOG(INFO) << "No of types: " << program.serialized_types.size() << "\n"
                  << "No of functions: " << program.serialized_functions.size() << "\n"
                  << "No of globals: " << program.serialized_globals.size() << "\n";

        return program;
    }

    // Deserialize types from the json object
    void JsonParser::deserialize_types(const JsonObject &type_obj, TypeMap &serialized_types) {
        if (type_obj.empty()) {
            LOG(INFO) << "No type objects to deserialize\n";
            return;
        }

        std::unordered_map< std::string, const JsonValue & > types_value_map;

        for (const auto &type : type_obj) {
            const auto &type_value = type.getSecond();
            auto vnode_type        = create_vnode_type(*type_value.getAsObject());
            if (!vnode_type) {
                LOG(ERROR) << "Failed to create varnode type\n";
                continue;
            }

            const auto type_key = type.getFirst().str();
            vnode_type->set_key(type_key);
            serialized_types.emplace(type_key, std::move(vnode_type));
            types_value_map.emplace(type_key, type_value);
        }

        LOG(INFO) << "Number of entry in serialized types: " << serialized_types.size() << "\n";

        // Post process varnodes from the map and resolve recursive references of labels
        for (const auto &[type_key, vnode_type] : serialized_types) {
            auto iter = types_value_map.find(type_key);
            if (iter == types_value_map.end()) {
                assert(false && "type_key is missing from value map");
                continue;
            }

            const auto &json_value = iter->second;
            switch (vnode_type->kind) {
                case VarnodeType::Kind::VT_BOOLEAN:
                case VarnodeType::Kind::VT_INTEGER:
                case VarnodeType::Kind::VT_FLOAT:
                case VarnodeType::Kind::VT_CHAR:
                case VarnodeType::Kind::VT_WIDECHAR:
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

    // Create varnode type from the json object
    std::shared_ptr< VarnodeType > JsonParser::create_vnode_type(const JsonObject &type_obj) {
        auto name     = get_string(type_obj, "name");
        auto size     = static_cast< uint32_t >(type_obj.getInteger("size").value_or(0));
        auto kind_str = get_string(type_obj, "kind");
        LOG(INFO) << "Attempting to convert kind string: [" << kind_str << "]" << "\n";
        auto kind = VarnodeType::convertToKind(kind_str);
        switch (kind) {
            case VarnodeType::Kind::VT_INVALID: {
                LOG(ERROR) << "Invalid varnode type: " << name << "\n";
                return std::make_shared< VarnodeType >(name, kind, size);
            }
            case VarnodeType::Kind::VT_BOOLEAN:
            case VarnodeType::Kind::VT_INTEGER:
            case VarnodeType::Kind::VT_FLOAT:
            case VarnodeType::Kind::VT_CHAR:
            case VarnodeType::Kind::VT_WIDECHAR:
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

        return nullptr;
    }

    void JsonParser::deserialize_buildin(
        BuiltinType &varnode, const JsonObject & /*unused*/, const TypeMap & /*unused*/
    ) {
        assert(
            varnode.kind == VarnodeType::Kind::VT_BOOLEAN
            || varnode.kind == VarnodeType::Kind::VT_INTEGER
            || varnode.kind == VarnodeType::Kind::VT_CHAR
            || varnode.kind == VarnodeType::Kind::VT_WIDECHAR
            || varnode.kind == VarnodeType::Kind::VT_FLOAT
            || varnode.kind == VarnodeType::Kind::VT_VOID
        );
        (void) varnode;
    }

    // Deserialize array types
    void JsonParser::deserialize_array(
        ArrayType &varnode, const JsonObject *array_obj, const TypeMap &serialized_types
    ) {
        auto element_key = get_string_if_valid(*array_obj, "element_type");
        if (!element_key) {
            LOG(ERROR) << "Element type of an array is empty. key: " << varnode.key << "\n";
            return;
        }

        auto iter = serialized_types.find(*element_key);
        if (iter == serialized_types.end()) {
            LOG(ERROR) << "Element type key " << *element_key
                       << " not found in serialized types."
                       << " deserializing array with key " << varnode.key << "\n";
            return;
        }

        varnode.set_element_type(iter->second);
        auto num_elem =
            static_cast< uint32_t >(array_obj->getInteger("num_elements").value_or(0));
        varnode.set_element_count(num_elem);
    }

    // Deserialize pointer types
    void JsonParser::deserialize_pointer(
        PointerType &varnode, const JsonObject &pointer_obj, const TypeMap &serialized_types
    ) {
        auto pointee_key = pointer_obj.getString("element_type").value_or("").str();
        if (pointee_key.empty()) {
            LOG(ERROR) << "Pointer type with empty pointee key. pointer key: " << varnode.key
                       << "\n";
            return;
        }

        // Check for the pointee label in serialized types
        auto iter = serialized_types.find(pointee_key);
        if (iter == serialized_types.end()) {
            LOG(ERROR) << "Pointee type is not availe in serialized types. Pointer key: "
                       << varnode.key << "\n";
            return;
        }

        varnode.set_pointee_type(iter->second);
    }

    // Deserialize typedef types
    void JsonParser::deserialize_typedef(
        TypedefType &varnode, const JsonObject &typedef_obj, const TypeMap &serialized_types
    ) {
        auto base_key = typedef_obj.getString("base_type").value_or("").str();
        if (base_key.empty()) {
            LOG(ERROR) << "Missing base type for typedef : " << varnode.key << "\n";
            return;
        }

        auto iter = serialized_types.find(base_key);
        if (iter == serialized_types.end()) {
            LOG(ERROR) << "Base type is not found in serialized types " << base_key
                       << " for typedef : " << varnode.key << "\n";
            return;
        }

        varnode.set_base_type(iter->second);
    }

    // Deserialize composite types
    void JsonParser::deserialize_composite(
        CompositeType &varnode, const JsonObject &composite_obj, const TypeMap &serialized_types
    ) {
        const auto *field_array = composite_obj.getArray("fields");
        if (field_array == nullptr || field_array->empty()) {
            LOG(ERROR) << "No fields found in composite type object\n";
            return;
        }

        // Iterate through the fields and initialize them
        unsigned field_index = 0;
        for (const auto &field : *field_array) {
            const auto *field_obj = field.getAsObject();
            if (field_obj == nullptr) {
                LOG(ERROR) << "Field #" << field_index
                           << ": Invalid field object format, skipping\n";
                ++field_index;
                continue;
            }

            auto field_type_key = get_string_if_valid(*field_obj, "type");
            if (!field_type_key) {
                LOG(ERROR) << "Field #" << field_index
                           << ": Missing required 'type' attribute, skipping\n";
                ++field_index;
                continue;
            }

            auto iter = serialized_types.find(*field_type_key);
            if (iter == serialized_types.end()) {
                LOG(ERROR) << "Field #" << field_index << ": Type '" << *field_type_key
                           << "' not found in serialized types registry, skipping\n";
                ++field_index;
                continue;
            }

            auto maybe_offset = field_obj->getInteger("offset");
            if (!maybe_offset) {
                LOG(ERROR) << "Field #" << field_index
                           << ": Missing or invalid 'offset' value, skipping\n";
                ++field_index;
                continue;
            }

            std::string field_name;
            auto maybe_name = get_string_if_valid(*field_obj, "name");
            if (maybe_name) {
                field_name = *maybe_name;
            } else {
                field_name = "field_" + std::to_string(field_index);
                LOG(WARNING) << "Field #" << field_index
                             << " Missing or invalid name, using default: '" << field_name
                             << "'\n";
            }

            varnode.add_components(
                field_name, *iter->second, static_cast< uint32_t >(*maybe_offset)
            );

            ++field_index;
        }
    }

    void JsonParser::deserialize_enum(
        EnumType &varnode, const JsonObject &enum_obj, const TypeMap & /*unused*/
    ) {
        assert(varnode.kind == VarnodeType::Kind::VT_ENUM);

        const auto *entries = enum_obj.getArray("entries");
        if (entries == nullptr || entries->empty()) {
            LOG(WARNING) << "Enum type '" << varnode.name << "' has no entries\n";
            return;
        }

        unsigned entry_index = 0;
        for (const auto &entry : *entries) {
            const auto *entry_obj = entry.getAsObject();
            if (entry_obj == nullptr) {
                LOG(ERROR) << "Enum '" << varnode.name << "' entry #" << entry_index
                           << ": Invalid format, skipping\n";
                ++entry_index;
                continue;
            }

            auto maybe_name = get_string_if_valid(*entry_obj, "name");
            if (!maybe_name) {
                LOG(ERROR) << "Enum '" << varnode.name << "' entry #" << entry_index
                           << ": Missing 'name', skipping\n";
                ++entry_index;
                continue;
            }

            auto maybe_value = entry_obj->getInteger("value");
            if (!maybe_value) {
                LOG(ERROR) << "Enum '" << varnode.name << "' entry #" << entry_index
                           << ": Missing 'value', skipping\n";
                ++entry_index;
                continue;
            }

            varnode.add_constant(*maybe_name, *maybe_value);
            ++entry_index;
        }
    }

    void JsonParser::deserialize_function_type(
        FunctionType &varnode, const JsonObject &func_obj, const TypeMap & /*unused*/
    ) {
        assert(varnode.kind == VarnodeType::Kind::VT_FUNCTION);

        auto maybe_return_type = get_string_if_valid(func_obj, "return_type");
        if (maybe_return_type) {
            varnode.return_type_key = *maybe_return_type;
        } else {
            LOG(WARNING) << "Function type '" << varnode.name << "': Missing 'return_type'\n";
        }

        varnode.is_variadic = func_obj.getBoolean("is_variadic").value_or(false);
        varnode.is_noreturn = func_obj.getBoolean("is_noreturn").value_or(false);

        const auto *params = func_obj.getArray("parameter_types");
        if (params != nullptr) {
            for (const auto &param : *params) {
                auto maybe_key = param.getAsString();
                if (maybe_key) {
                    varnode.param_type_keys.emplace_back(maybe_key->str());
                } else {
                    LOG(WARNING) << "Function type '" << varnode.name
                                 << "': Invalid parameter type entry, skipping\n";
                }
            }
        }
    }

    void JsonParser::deserialize_undefined_type(
        UndefinedType &varnode, const JsonObject & /*unused*/, const TypeMap & /*unused*/
    ) {
        assert(varnode.kind == VarnodeType::Kind::VT_UNDEFINED);
        (void) varnode;
    }

    // Deserialize operation varnode
    std::optional< Varnode > JsonParser::create_varnode(const JsonObject &var_obj) {
        auto type_key = get_string(var_obj, "type");
        auto size     = var_obj.getInteger("size").value_or(0);
        auto kind     = Varnode::convertToKind(var_obj.getString("kind").value_or("").str());
        if (kind == Varnode::VARNODE_UNKNOWN) {
            LOG(ERROR) << "Operation with unknown varnode.\n";
            return std::nullopt;
        }

        auto set_field_if_valid = [](const std::optional< std::string > &field,
                                     std::optional< std::string > &vnode_field) {
            if (field) {
                vnode_field = *field;
            }
        };

        Varnode vnode(kind, static_cast< uint32_t >(size), type_key);

        set_field_if_valid(get_string_if_valid(var_obj, "operation"), vnode.operation);
        set_field_if_valid(get_string_if_valid(var_obj, "function"), vnode.function);
        set_field_if_valid(get_string_if_valid(var_obj, "global"), vnode.global);
        set_field_if_valid(get_string_with_empty(var_obj, "string_value"), vnode.string_value);
        vnode.value = var_obj.getInteger("value");
        return vnode;
    }

    std::optional< Function > JsonParser::create_function(const JsonObject &func_obj) {
        const auto function_name = stripNull(func_obj.getString("name"));
        if (!function_name || function_name->empty()) {
            LOG(ERROR) << "Missing function name from the object.\n";
            return std::nullopt;
        }

        Function function;

        function.name = *function_name;
        if (const auto *proto_obj = func_obj.getObject("type")) {
            if (auto maybe_prototype = create_function_prototype(*proto_obj)) {
                function.prototype = *maybe_prototype;
            }
        }

        auto entry_block = func_obj.getString("entry_block");
        if (entry_block && !entry_block->empty()) {
            function.entry_block = entry_block->str();
        }

        if (const auto *blocks_array = func_obj.getObject("basic_blocks")) {
            deserialize_blocks(*blocks_array, function.basic_blocks, function.entry_block);
        }

        return function;
    }

    void JsonParser::deserialize_call_operation(const JsonObject &call_obj, Operation &op) {
        const auto *maybe_target = call_obj.getObject("target");
        if (maybe_target == nullptr) {
            LOG(ERROR) << "No target for the call operation.\n";
            return;
        }

        OperationTarget target;
        target.kind =
            Varnode::convertToKind(maybe_target->getString("kind").value_or("").str());

        auto function = maybe_target->getString("function");
        if (function.has_value() && !function->empty()) {
            target.function = stripNull(function);
        }

        auto call_op = maybe_target->getString("operation");
        if (call_op.has_value() && !call_op->empty()) {
            target.operation = call_op->str();
        }

        // For CALLIND: parse global variable target
        auto global_key = maybe_target->getString("global");
        if (global_key.has_value() && !global_key->empty()) {
            target.global = global_key->str();
        }

        // For CALLIND: parse type of the target
        auto type_key = maybe_target->getString("type");
        if (type_key.has_value() && !type_key->empty()) {
            target.type_key = type_key->str();
        }

        target.is_noreturn  = maybe_target->getBoolean("is_noreturn").value_or(false);
        op.target           = std::move(target);
        op.has_return_value = call_obj.getBoolean("has_return_value").value_or(false);
    }

    void JsonParser::deserialize_branch_operation(const JsonObject &branch_obj, Operation &op) {
        auto set_target_field_if_valid = [](const std::optional< std::string > &block_key,
                                            std::optional< std::string > &block_target) {
            if (block_key && !block_key->empty()) {
                block_target = *block_key;
            }
        };

        set_target_field_if_valid(
            get_string_if_valid(branch_obj, "target_block"), op.target_block
        );
        set_target_field_if_valid(
            get_string_if_valid(branch_obj, "taken_block"), op.taken_block
        );
        set_target_field_if_valid(
            get_string_if_valid(branch_obj, "not_taken_block"), op.not_taken_block
        );

        if (const auto *maybe_output = branch_obj.getObject("condition")) {
            if (auto maybe_varnode = create_varnode(*maybe_output)) {
                op.condition = std::move(*maybe_varnode);
            }
        }

        if (const auto *succ_array = branch_obj.getArray("successor_blocks")) {
            for (auto item : *succ_array) {
                if (auto s = item.getAsString()) {
                    if (!s->empty()) {
                        op.successor_blocks.emplace_back(*s);
                    }
                }
            }
        }

        // fallback_block
        set_target_field_if_valid(
            get_string_if_valid(branch_obj, "fallback_block"), op.fallback_block
        );

        // switch_input varnode
        if (const auto *sw_in = branch_obj.getObject("switch_input")) {
            if (auto vn = create_varnode(*sw_in)) {
                op.switch_input = std::move(*vn);
            }
        }

        // switch_cases array
        if (const auto *cases = branch_obj.getArray("switch_cases")) {
            for (auto item : *cases) {
                const auto *obj = item.getAsObject();
                if (obj == nullptr) {
                    continue;
                }
                auto val   = obj->getInteger("value");
                auto block = get_string_if_valid(*obj, "target_block");
                if (val && block && !block->empty()) {
                    op.switch_cases.push_back({ *val, *block });
                }
            }
        }
    }

    std::optional< Operation > JsonParser::create_operation(const JsonObject &pcode_obj) {
        auto mnemonic_str = get_string_if_valid(pcode_obj, "mnemonic");
        if (!mnemonic_str) {
            LOG(ERROR) << "Missing mnemonic from pcode operation.\n";
            return std::nullopt;
        }

        auto mnemonic = patchestry::ghidra::from_string(*mnemonic_str);
        if (mnemonic == Mnemonic::OP_UNKNOWN) {
            LOG(ERROR) << "Pcode with unknown operation mnemonic: " << mnemonic_str << "\n";
            return std::nullopt;
        }

        Operation operation;
        operation.mnemonic = mnemonic;
        if (const auto *maybe_output = pcode_obj.getObject("output")) {
            if (auto maybe_varnode = create_varnode(*maybe_output)) {
                operation.output = std::move(*maybe_varnode);
            }
        }

        if (const auto *input_array = pcode_obj.getArray("inputs")) {
            for (auto input : *input_array) {
                if (auto maybe_varnode = create_varnode(*input.getAsObject())) {
                    operation.inputs.emplace_back(std::move(*maybe_varnode));
                }
            }
        }

        // Deserialize additional operation fields
        operation.name    = get_string_if_valid(pcode_obj, "name");
        operation.type    = get_string_if_valid(pcode_obj, "type");
        operation.address = get_string_if_valid(pcode_obj, "address");

        if (auto index = pcode_obj.getInteger("index")) {
            operation.index = static_cast< uint32_t >(*index);
        }

        // Deserialize operation specific fields
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

        return operation;
    }

    std::optional< BasicBlock >
    JsonParser::create_basic_block(const std::string &block_key, const JsonObject &block_obj) {
        const auto *maybe_operations = block_obj.getObject("operations");
        if (maybe_operations == nullptr) {
            LOG(ERROR) << "No operations in the basic block.\n";
            return std::nullopt;
        }

        BasicBlock block;
        for (const auto &operation : *maybe_operations) {
            auto operation_key           = operation.getFirst().str();
            const auto *operation_object = operation.getSecond().getAsObject();
            if (operation_object == nullptr) {
                LOG(WARNING) << "Basic Block \"" << block_key
                             << "\": Skipping invalid operation: " << operation_key << "\n";
                continue;
            }

            if (auto maybe_operation = create_operation(*operation_object)) {
                maybe_operation->key              = operation_key;
                maybe_operation->parent_block_key = block_key;
                block.operations.emplace(operation_key, std::move(*maybe_operation));
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

    std::optional< FunctionPrototype >
    JsonParser::create_function_prototype(const JsonObject &proto_obj) {
        // Validate the return type
        const auto return_type_key = get_string_if_valid(proto_obj, "return_type");
        if (!return_type_key) {
            LOG(ERROR) << "FunctionPrototype: Return type is empty.\n";
            return std::nullopt;
        }

        FunctionPrototype proto;
        proto.rttype_key  = *return_type_key;
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
            LOG(INFO) << "No functions to deserialize.";
            return;
        }

        for (const auto &func_obj : function_array) {
            const auto &function_key  = func_obj.getFirst();
            const auto *function_data = func_obj.getSecond().getAsObject();
            if (function_data == nullptr) {
                LOG(ERROR) << "Skipping invalid function object for key: " << function_key
                           << "\n";
                continue;
            }

            auto function = create_function(*func_obj.getSecond().getAsObject());
            if (!function) {
                LOG(ERROR) << "Failed to create function for the key " << function_key << "\n";
                continue;
            }

            function->key = function_key.str();
            serialized_functions.emplace(function_key.str(), std::move(*function));
        }
    }

    void JsonParser::deserialize_blocks(
        const JsonObject &blocks_array, BasicBlockMap &serialized_blocks,
        std::string &entry_block
    ) {
        if (blocks_array.empty()) {
            LOG(INFO) << "No blocks in function to deserialize.";
            return;
        }

        for (const auto &block : blocks_array) {
            const auto &block_key = block.getFirst();
            const auto *block_obj = block.getSecond().getAsObject();

            if (block_obj == nullptr) {
                LOG(ERROR) << "Skipping invalid block object for key: " << block_key;
                continue;
            }

            auto maybe_block = create_basic_block(block_key.str(), *block_obj);
            if (!maybe_block) {
                LOG(ERROR) << "Failed to create basic block for key: " << block_key << "\n";
                continue;
            }

            maybe_block->is_entry_block = (block_key.str() == entry_block);
            maybe_block->key            = block_key.str();

            // Emplace into the serialized_blocks map
            serialized_blocks.emplace(block_key.str(), std::move(*maybe_block));
        }
    }

    void JsonParser::deserialize_globals(
        const JsonObject &global_array, VariableMap &serialized_globals
    ) {
        if (global_array.empty()) {
            LOG(INFO) << "No global variables found to deserialize.";
            return;
        }

        for (const auto &global : global_array) {
            const auto &variable_key = global.getFirst();
            const auto *global_obj   = global.getSecond().getAsObject();
            if (global_obj == nullptr) {
                LOG(ERROR) << "Skipping invalid global object for key: " << variable_key;
                continue;
            }

            Variable variable;
            variable.key = variable_key.str();
            if (auto maybe_name = get_string_if_valid(*global_obj, "name")) {
                variable.name = *maybe_name;
            }
            if (auto maybe_type = get_string_if_valid(*global_obj, "type")) {
                variable.type = *maybe_type;
            }
            if (auto maybe_size = global_obj->getInteger("size")) {
                variable.size = static_cast< uint32_t >(*maybe_size);
            }

            serialized_globals.emplace(variable.key, std::move(variable));
        }
    }

} // namespace patchestry::ghidra
