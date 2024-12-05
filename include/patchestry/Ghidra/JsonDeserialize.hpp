/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>

#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Ghidra/PcodeTypes.hpp>

namespace patchestry::ghidra {

    class JsonParser
    {
      public:
        std::optional< Program > deserialize_program(const JsonObject &root);

      private:
        // Process types from Json object
        void deserialize_types(const JsonObject &type_obj, TypeMap &serialized_types);

        // Create varnode type for each type object
        std::shared_ptr< VarnodeType > create_vnode_type(const JsonObject &type_obj);

        void deserialize_buildin(
            BuiltinType &varnode, const JsonObject &builtin_obj, const TypeMap &serialized_types
        );

        void deserialize_array(
            ArrayType &varnode, const JsonObject *array_obj, const TypeMap &serialized_types
        );

        void deserialize_pointer(
            PointerType &varnode, const JsonObject &pointer_obj, const TypeMap &serialized_types
        );

        void deserialize_typedef(
            TypedefType &varnode, const JsonObject &typedef_obj, const TypeMap &serialized_types
        );

        void deserialize_composite(
            CompositeType &varnode, const JsonObject &composite_obj,
            const TypeMap &serialized_types
        );

        void deserialize_enum(
            EnumType &varnode, const JsonObject &enum_obj, const TypeMap &serialized_types
        );

        void deserialize_function_type(
            FunctionType &varnode, const JsonObject &func_obj, const TypeMap &serialized_types
        );

        void deserialize_undefined_type(
            UndefinedType &varnode, const JsonObject &undef_obj, const TypeMap &serialized_types
        );

        // Handle pcode operations
        std::optional< Varnode > create_varnode(const JsonObject &var_obj);

        std::optional< Function > create_function(const JsonObject &func_obj);

        std::optional< FunctionPrototype > create_function_prototype(const JsonObject &proto_obj
        );

        // Create basic block object from deserialized json object
        std::optional< BasicBlock >
        create_basic_block(const std::string &block_key, const JsonObject &block_obj);

        // Create operation from the deserialized pcode json object
        std::optional< Operation > create_operation(const JsonObject &pcode_obj);

        void deserialize_call_operation(const JsonObject &call_obj, Operation &op);

        void deserialize_branch_operation(const JsonObject &branch_obj, Operation &op);

        // Deserialize functions from serialized json
        void deserialize_functions(
            const JsonObject &function_array, FunctionMap &serialized_functions
        );

        // Deserialize basic blocks for functions from serialized json
        void deserialize_blocks(
            const JsonObject &blocks_array, BasicBlockMap &serialized_blocks,
            std::string &entry_block
        );

        // Deserialize global variables from serialized json
        void
        deserialize_globals(const JsonObject &global_array, VariableMap &serialized_globals);
    };

} // namespace patchestry::ghidra
