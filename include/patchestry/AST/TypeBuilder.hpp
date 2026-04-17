/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <string>

#include <clang/AST/ASTContext.h>

#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {
    using namespace patchestry::ghidra;

    using SerializedTypeMap = std::unordered_map< std::string, clang::QualType >;

    class TypeBuilder
    {
      public:
        /* consts */
        static constexpr uint32_t kNumBitsInByte = 8U;
        static constexpr uint32_t kNumBitsUint   = 32U;

        explicit TypeBuilder(clang::ASTContext &ctx) : context(ctx), serialized_types({}) {}

        TypeBuilder &operator=(const TypeBuilder &)  = delete;
        TypeBuilder(const TypeBuilder &)             = delete;
        TypeBuilder &operator=(const TypeBuilder &&) = delete;
        TypeBuilder(const TypeBuilder &&)            = delete;

        virtual ~TypeBuilder() = default;

        /**
         * @brief Provides access to the serialized type map.
         *
         * This method returns a reference to the internal `serialized_types` map, which
         * contains the mappings between type keys and their corresponding `clang::QualType`
         * instances. The map represents all the types that have been created and cached by the
         * `TypeBuilder`.
         *
         * @return A reference to the `SerializedTypeMap` (alias for `std::unordered_map`) that
         * maps type keys (`std::string`) to their serialized `clang::QualType` objects.
         *
         * @details
         * - The `serialized_types` map serves as a cache to avoid redundant type creation and
         *  reuse of already generated `QualType` instances.
         */

        SerializedTypeMap &GetSerializedTypes(void) { return serialized_types; }

        /**
         * @brief Safely retrieves a serialized `clang::QualType` for the given key.
         *
         * Performs a lookup in the internal `serialized_types` map without throwing
         * `std::out_of_range` when the key is not present. When the key is missing,
         * a warning is logged and an empty `clang::QualType` is returned so that
         * callers can propagate the failure via `QualType::isNull()`.
         *
         * @param key The type key to look up in the serialized type map.
         *
         * @return The `clang::QualType` associated with `key`, or an empty
         *         `clang::QualType{}` if the key is not present in the map.
         */
        clang::QualType GetSerializedType(const std::string &key) const {
            auto it = serialized_types.find(key);
            if (it == serialized_types.end()) {
                LOG(WARNING) << "Type key not found in serialized types: " << key;
                return clang::QualType{};
            }
            return it->second;
        }

        /**
         * @brief Creates and serializes all types defined in the `lifted_types`.
         *
         * @param ctx The `clang::ASTContext` used for type creation and AST generation.
         * @param lifted_types A map of type keys to `VarnodeType` objects representing all the
         * types that need to be processed and serialized.
         */
        void create_types(clang::ASTContext &ctx, TypeMap &lifted_types);

      private:
        clang::QualType
        fix_type_for_undefined_array(clang::ASTContext &ctx, const ArrayType &array_type);

        /**
         * @brief Generates the `clang::QualType` corresponding to a given `VarnodeType`.
         *
         * @param ctx The `clang::ASTContext` used for type creation.
         * @param vnode_type A shared pointer to the `VarnodeType` object representing the type
         * to be created.
         *
         * @return A `clang::QualType` corresponding to the provided `VarnodeType`.
         *
         */
        clang::QualType
        create_type(clang::ASTContext &ctx, const std::shared_ptr< VarnodeType > &vnode_type);

        /**
         * @brief Creates a "placeholder" type for invalid Varnode types.
         *
         * @param ctx The Clang AST context used to manage types and declarations.
         * @param vnode_type A shared pointer to the `VarnodeType` representing the invalid
         * type. This parameter is included for potential future use or extensions.
         *
         * @return A `clang::QualType` representing the invalid type.
         */
        clang::QualType create_invalid(
            clang::ASTContext &ctx, const std::shared_ptr< VarnodeType > &vnode_type
        );

        /**
         * @brief Creates a `clang::QualType` for a ghidra typedef declaration.
         *
         * @param ctx Reference to the `clang::ASTContext`.
         * @param typedef_type Reference to the `TypedefType` containing metadata about
         *        the typedef to be created.
         *
         * @return A `clang::QualType` representing the newly created typedef type.
         *         Returns an empty `QualType` if error occurs.
         */
        clang::QualType create_typedef(clang::ASTContext &ctx, const TypedefType &typedef_type);

        /**
         * @brief Creates a `clang::QualType` for a ghidra pointer type.
         *
         * @param ctx Reference to the `clang::ASTContext`.
         * @param pointer_type Reference to the `PointerType` containing metadata about
         *        the pointer to be created.
         *
         * @return A `clang::QualType` representing the newly created pointer type.
         *         Returns `clang::ASTContext::VoidPtrTy` for invalid input.
         */
        clang::QualType create_pointer(clang::ASTContext &ctx, const PointerType &pointer_type);

        /**
         * @brief Creates a `clang::QualType` for an array type.
         *
         * @param ctx Reference to the `clang::ASTContext`.
         * @param array_type The metadata describing the array type, including its key,
         *                   element type, and size.
         *
         * @return A `clang::QualType` representing the newly created array type.
         *         Returns an empty type (`clang::QualType{}`) if errors occur.
         */
        clang::QualType create_array(clang::ASTContext &ctx, const ArrayType &array_type);

        /**
         * @brief Creates a `clang::QualType` for a composite type (struct or union).
         *
         * @param ctx The `clang::ASTContext` used for type creation.
         * @param composite_type Metadata representing the composite type, including
         *                       its kind (struct or union), name, and type key.
         *
         * @return A `clang::QualType` representing the composite type.
         */
        clang::QualType
        create_composite(clang::ASTContext &ctx, const VarnodeType &composite_type);

        /**
         * @brief Creates a `clang::QualType` for an undefined type.
         *
         * @param ctx Reference to the `clang::ASTContext`.
         * @param undefined_type The metadata representing the undefined type, including
         *                       its size (in bytes) and name.
         *
         * @return A `clang::QualType` representing the newly created undefined type.
         *         Falls back to an integer type (`ctx.IntTy`) if no valid base type is found.
         */
        clang::QualType
        create_undefined(clang::ASTContext &ctx, const UndefinedType &undefined_type);

        /**
         * @brief Creates a `clang::QualType` for undefined array.
         *
         * @param ctx Reference to the `clang::ASTContext`.
         * @param undefined_type The metadata representing the undefined type, including
         *                       its size (in bytes) and name.
         *
         * @return A `clang::QualType` representing the newly created undefined array
         */
        clang::QualType create_type_for_undefined_array(
            clang::ASTContext &ctx, const ArrayType &undefined_array
        );

        /**
         * @brief Completes the definition of a composite type (e.g., struct) in the AST.
         *
         * @param ctx The `clang::ASTContext` used for AST node creation.
         * @param varnode Metadata representing the composite type, including its name
         *                and components.
         * @param prev_decl Optional previous declaration of the composite type, used
         *                  to link to an existing incomplete type if provided.
         * @param clang_types A map from type keys to `clang::QualType` objects, used
         *                    to resolve types of composite components.
         */
        void complete_definition(
            clang::ASTContext &ctx, const CompositeType &varnode, clang::Decl *prev_decl,
            const SerializedTypeMap &clang_types
        );

        /**
         * @brief Creates a `clang::QualType` for an enumeration type.
         *
         * @param ctx The `clang::ASTContext` used for type creation.
         * @param enum_type Metadata representing the enumeration, including its
         *                  name and unique key.
         *
         * @return A `clang::QualType` representing the enumeration type.
         */
        clang::QualType create_enum(clang::ASTContext &ctx, const EnumType &enum_type);

        /**
         * @brief get reference to the `clang::ASTContext` used for AST nodes
         */
        clang::ASTContext &GetASTContext() { return context.get(); }

        std::unordered_map< std::string, clang::Decl * > missing_type_definition;
        std::reference_wrapper< clang::ASTContext > context;
        SerializedTypeMap serialized_types;
        const TypeMap *lifted_types_ = nullptr;
    };
} // namespace patchestry::ast
