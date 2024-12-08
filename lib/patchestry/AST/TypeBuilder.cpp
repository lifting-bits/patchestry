/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <clang/AST/Type.h>

#include <patchestry/AST/TypeBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/PcodeTypes.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {

    /**
     * @brief Creates and serializes all types defined in the given `TypeMap`.
     *
     * This function generates the `clang::QualType` representations for all types in the
     * provided `lifted_types` map, adding them to the `serialized_types` map.
     *
     * @param ctx The `clang::ASTContext` used for type creation and AST node management.
     * @param lifted_types A map of type keys to `VarnodeType` objects representing all the
     * types that need to be processed and serialized.
     *
     * @details
     * - **Serialized Types**: The function iterates through `lifted_types` and generates the
     *   corresponding `clang::QualType` for each type, storing the result in the
     * `serialized_types` map.
     */

    void TypeBuilder::create_types(clang::ASTContext &ctx, TypeMap &lifted_types) {
        for (auto &[key, vnode_type] : lifted_types) {
            serialized_types.emplace(key, create_type(ctx, vnode_type));
        }

        // Traverse through missing type definition for a composite type and create the complete
        // definition for them
        for (auto &[key, decl] : missing_type_definition) {
            if (const auto *record_decl = llvm::dyn_cast< clang::RecordDecl >(decl)) {
                auto iter = lifted_types.find(key);
                // if type key does not exist in the lifted types, it could be a bug in create
                // types.
                if (iter == lifted_types.end()) {
                    LOG(ERROR) << "Composite type with key : " << key
                               << " not found in lifted types.\n";
                    continue;
                }

                complete_definition(
                    ctx, dynamic_cast< CompositeType & >(*iter->second), decl, serialized_types
                );
            }
        }
    }

    /**
     * @brief Generates the `clang::QualType` corresponding to a given `VarnodeType`.
     *
     * This method processes a high-level representation of a type (`VarnodeType`) and
     * translates it into a `clang::QualType` for use within the Clang AST.
     *
     * @param ctx The `clang::ASTContext` used for type creation and management.
     * @param vnode_type A shared pointer to the `VarnodeType` object representing the type to
     * be created.
     *
     * @return A `clang::QualType` corresponding to the provided `VarnodeType`.
     *
     */
    clang::QualType TypeBuilder::create_type(
        clang::ASTContext &ctx, const std::shared_ptr< VarnodeType > &vnode_type
    ) {
        if (serialized_types.contains(vnode_type->key)) {
            return serialized_types[vnode_type->key];
        }

        switch (vnode_type->kind) {
            case VarnodeType::VT_INVALID:
                return create_invalid(ctx, vnode_type);

            case VarnodeType::VT_BOOLEAN:
                return ctx.BoolTy;

            case VarnodeType::VT_INTEGER:
                return get_type_for_size(
                    ctx, vnode_type->size * TypeBuilder::num_bits_in_byte, /*is_signed=*/false,
                    /*is_integer=*/true
                );

            case VarnodeType::VT_CHAR:
                return ctx.CharTy;

            case VarnodeType::VT_FLOAT:
                return get_type_for_size(
                    ctx, vnode_type->size * TypeBuilder::num_bits_in_byte, /*is_signed=*/false,
                    /*is_integer=*/false
                );

            case VarnodeType::VT_ARRAY:
                return create_array(ctx, dynamic_cast< const ArrayType & >(*vnode_type));

            case VarnodeType::VT_POINTER:
                return create_pointer(ctx, dynamic_cast< const PointerType & >(*vnode_type));

            case VarnodeType::Kind::VT_FUNCTION:
                return ctx.VoidPtrTy;

            case VarnodeType::VT_STRUCT:
            case VarnodeType::VT_UNION:
                return create_composite(ctx, *vnode_type);

            case VarnodeType::VT_ENUM:
                return create_enum(ctx, dynamic_cast< const EnumType & >(*vnode_type));

            case VarnodeType::VT_TYPEDEF:
                return create_typedef(ctx, dynamic_cast< const TypedefType & >(*vnode_type));

            case VarnodeType::VT_UNDEFINED:
                return create_undefined(
                    ctx, dynamic_cast< const UndefinedType & >(*vnode_type)
                );

            case VarnodeType::VT_VOID: {
                return ctx.VoidTy;
            }
        }
    }

    /**
     * @brief Creates a "placeholder" type for invalid Varnode types.
     *
     * @param ctx The Clang AST context used to manage types and declarations.
     * @param vnode_type A shared pointer to the `VarnodeType` representing the invalid type.
     *                   This parameter is included for potential future use or extensions.
     *
     * @return A `clang::QualType` representing the invalid type.
     */

    clang::QualType TypeBuilder::
        create_invalid(clang::ASTContext &ctx, const std::shared_ptr< VarnodeType > &) {
        // An invalid type is defined as typedef of a void type
        auto underlying_type = ctx.VoidTy;

        auto *typedef_decl = clang::TypedefDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), clang::SourceLocation(), clang::SourceLocation(),
            &ctx.Idents.get("invalid_type"), ctx.getTrivialTypeSourceInfo(underlying_type)
        );

        typedef_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(typedef_decl);

        return ctx.getTypedefType(typedef_decl);
    }

    /**
     * @brief Creates a `clang::QualType` for a ghidra typedef declaration.
     *
     * This method generates a new `clang::TypedefType` in the provided AST context
     * based on the given `typedef_type`. It ensures that the base type of the typedef
     * is valid, does not reference itself, and is properly serialized.
     *
     * @param ctx Reference to the `clang::ASTContext` used for creating the typedef.
     * @param typedef_type A reference to the `TypedefType` containing metadata about
     *        the typedef to be created, including its name and base type.
     *
     * @return A `clang::QualType` representing the newly created typedef type.
     *         Returns an empty `QualType` if error occurs.
     */

    clang::QualType
    TypeBuilder::create_typedef(clang::ASTContext &ctx, const TypedefType &typedef_type) {
        if (!typedef_type.get_base_type()) {
            LOG(ERROR) << "Base Type of a typedef shouldn't be empty. key: " << typedef_type.key
                       << "\n";
            return {};
        }

        const auto &base_type = typedef_type.get_base_type();
        if (base_type->key == typedef_type.key) {
            LOG(ERROR) << "Base Type of typedef is pointing to itself. key: "
                       << typedef_type.key << "\n";
            return {};
        }

        auto underlying_type = create_type(ctx, base_type);
        serialized_types.emplace(base_type->key, underlying_type);

        auto *tinfo        = ctx.getTrivialTypeSourceInfo(underlying_type);
        auto *typedef_decl = clang::TypedefDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), clang::SourceLocation(), clang::SourceLocation(),
            &ctx.Idents.get(typedef_type.name), tinfo
        );

        typedef_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(typedef_decl);

        return ctx.getTypedefType(typedef_decl);
    }

    /**
     * @brief Creates a `clang::QualType` for a ghidra pointer type.
     *
     * This method generates a pointer type (`clang::PointerType`) in the given AST context
     * based on the provided `pointer_type`. It ensures the pointee type is valid,
     * does not reference itself.
     *
     * @param ctx Reference to the `clang::ASTContext` used for creating the pointer type.
     * @param pointer_type A reference to the `PointerType` containing metadata about
     *        the pointer to be created, including its key and pointee type.
     *
     * @return A `clang::QualType` representing the newly created pointer type.
     *         Returns `clang::ASTContext::VoidPtrTy` for invalid input.
     *
     * @note Updates the `serialized_types` map with the pointee type's key and
     *       the corresponding `clang::QualType`.
     */

    clang::QualType
    TypeBuilder::create_pointer(clang::ASTContext &ctx, const PointerType &pointer_type) {
        if (!pointer_type.get_pointee_type()) {
            LOG(ERROR) << "Pointer type must have a valid pointee. Key: " << pointer_type.key
                       << "\n";
            return ctx.VoidPtrTy;
        }

        const auto &pointee = pointer_type.get_pointee_type();
        if (pointee->key == pointer_type.key) {
            LOG(ERROR) << "Pointer type cannot reference itself. Key: " << pointer_type.key
                       << "\n";
            return ctx.VoidPtrTy;
        }

        auto pointee_type = create_type(ctx, pointee);
        serialized_types.emplace(pointee->key, pointee_type);
        return ctx.getPointerType(pointee_type);
    }

    clang::QualType TypeBuilder::fix_type_for_undefined_array(
        clang::ASTContext &ctx, const ArrayType &array_type
    ) {
        auto element = array_type.get_element_type();
        if (element->kind != VarnodeType::VT_UNDEFINED) {
            return {};
        }

        auto pointee_type = create_type(ctx, element);
        serialized_types.emplace(element->key, pointee_type);
        return ctx.getPointerType(pointee_type);
    }

    /**
     * @brief Creates a `clang::QualType` for an array type.
     *
     * This function constructs a constant array type in the provided `ASTContext`
     * based on the given `ArrayType`. It validates the array's element type, ensures
     * there are no cyclic dependencies.
     *
     * @param ctx Reference to the `clang::ASTContext` used for creating the array type.
     * @param array_type The metadata describing the array type, including its key,
     *                   element type, and size.
     *
     * @return A `clang::QualType` representing the newly created array type.
     *         Returns an empty type (`clang::QualType{}`) if errors occur.
     *
     * @note Updates the `serialized_types` map with the element type's key and
     *       the corresponding `clang::QualType`.
     */
    clang::QualType
    TypeBuilder::create_array(clang::ASTContext &ctx, const ArrayType &array_type) {
        if (!array_type.get_element_type()) {
            LOG(ERROR) << "No element types for array\n";
            return {};
        }

        const auto &element = array_type.get_element_type();
        if (element->kind == VarnodeType::VT_UNDEFINED) {
            return fix_type_for_undefined_array(ctx, array_type);
        }

        if (element->key == array_type.key) {
            LOG(ERROR) << "Element key is same as array key. Key " << element->key << "\n";
            return {};
        }

        // If element key is same as array_type key, it will lead to infinite recursive. If it
        // happens something is wrong and need to check ghidra scripts
        auto element_type = create_type(ctx, element);
        serialized_types.emplace(element->key, element_type);

        auto size = array_type.get_element_count();
        return ctx.getConstantArrayType(
            element_type, llvm::APInt(num_bits_uint, size), nullptr,
            clang::ArraySizeModifier::Normal, 0
        );
    }

    /**
     * @brief Completes the definition of a composite type (e.g., struct) in the AST.
     *
     * This function creates and finalizes the definition of a `clang::RecordDecl`
     * for a composite type. It iterates through the components of the composite type,
     * adding fields to the record declaration using the corresponding types from
     * the provided `SerializedTypeMap`.
     *
     * @param ctx The `clang::ASTContext` used for AST node creation.
     * @param varnode Metadata representing the composite type, including its name
     *                and components.
     * @param prev_decl Optional previous declaration of the composite type, used
     *                  to link to an existing incomplete type if provided.
     * @param clang_types A map from type keys to `clang::QualType` objects, used
     *                    to resolve types of composite components.
     */
    void TypeBuilder::complete_definition(
        clang::ASTContext &ctx, const CompositeType &varnode, clang::Decl *prev_decl,
        const SerializedTypeMap &clang_types
    ) {
        auto *record_decl = clang::RecordDecl::Create(
            ctx, clang::TagDecl::TagKind::Struct, ctx.getTranslationUnitDecl(),
            source_location_from_key(ctx, varnode.key),
            source_location_from_key(ctx, varnode.key), &ctx.Idents.get(varnode.name),
            llvm::dyn_cast< clang::RecordDecl >(prev_decl)
        );

        record_decl->completeDefinition();

        auto components = varnode.get_components();
        for (auto &component : components) {
            // Resolve the type of the current component.
            if (!clang_types.contains(component.type->key)) {
                LOG(ERROR) << "Record component does not have type key. Key: " << varnode.key
                           << "\n";
                continue;
            }

            const auto &iter = clang_types.find(component.type->key);
            auto *field_decl = clang::FieldDecl::Create(
                ctx, record_decl, clang::SourceLocation(), clang::SourceLocation(),
                &ctx.Idents.get(component.name), iter->second, nullptr, nullptr, false,
                clang::ICIS_NoInit
            );

            record_decl->addDecl(field_decl);
        }

        record_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(record_decl);
    }

    /**
     * @brief Creates a `clang::QualType` for a composite type (struct or union).
     *
     * This function generates a `clang::RecordDecl` to represent a composite type
     * (struct or union) and associates it with the given `VarnodeType`. The
     * composite type is defined based on its kind (struct or union) and is added
     * to the translation unit's declaration context.
     *
     * @param ctx The `clang::ASTContext` used for type creation.
     * @param composite_type Metadata representing the composite type, including
     *                       its kind (struct or union), name, and unique key.
     *
     * @return A `clang::QualType` representing the composite type.
     */

    clang::QualType
    TypeBuilder::create_composite(clang::ASTContext &ctx, const VarnodeType &composite_type) {
        auto tag_kind = [&]() -> clang::TagDecl::TagKind {
            switch (composite_type.kind) {
                case VarnodeType::Kind::VT_STRUCT:
                    return clang::TagDecl::TagKind::Struct;
                case VarnodeType::Kind::VT_UNION:
                    return clang::TagDecl::TagKind::Union;
                default:
                    LOG(ERROR) << "Unsupported composite type kind: " << composite_type.kind;
                    return {};
            }
        }();

        // Create a RecordDecl for the composite type.
        auto *decl = clang::RecordDecl::Create(
            ctx, tag_kind, ctx.getTranslationUnitDecl(),
            source_location_from_key(ctx, composite_type.key),
            source_location_from_key(ctx, composite_type.key),
            &ctx.Idents.get(composite_type.name)
        );

        decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(decl);

        // Track the missing type definition for completing at later stage
        missing_type_definition.emplace(composite_type.key, decl);

        return ctx.getRecordType(decl);
    }

    /**
     * @brief Creates a `clang::QualType` for an enumeration type.
     *
     * This function generates a `clang::EnumDecl` to represent an enumeration
     * and associates it with the given `EnumType`. The enumeration is added
     * to the translation unit's declaration context.
     *
     * @param ctx The `clang::ASTContext` used for type creation.
     * @param enum_type Metadata representing the enumeration, including its
     *                  name and unique key.
     *
     * @return A `clang::QualType` representing the enumeration type.
     */

    clang::QualType
    TypeBuilder::create_enum(clang::ASTContext &ctx, const EnumType &enum_type) {
        auto *enum_decl = clang::EnumDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), source_location_from_key(ctx, enum_type.key),
            source_location_from_key(ctx, enum_type.key), &ctx.Idents.get(enum_type.name),
            nullptr, true, false, false
        );

        enum_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(enum_decl);

        return ctx.getEnumType(enum_decl);
    }

    /**
     * @brief Creates a `clang::QualType` for an undefined type.
     *
     * This function generates a typedef for an undefined type based on its size
     * and default attributes. It first attempts to derive an appropriate base type
     * using `get_type_for_size`. If unsuccessful, it defaults to the `int` type.
     *
     * @param ctx Reference to the `clang::ASTContext` for type creation.
     * @param undefined_type The metadata representing the undefined type, including
     *                       its size (in bytes) and name.
     *
     * @return A `clang::QualType` representing the newly created undefined type.
     *         Falls back to an integer type (`ctx.IntTy`) if no valid base type is found.
     *
     * @note Updates the `ASTContext` with a new `TypedefDecl` for the undefined type.
     */

    clang::QualType
    TypeBuilder::create_undefined(clang::ASTContext &ctx, const UndefinedType &undefined_type) {
        auto base_type = get_type_for_size(
            ctx, undefined_type.size * num_bits_in_byte, /*is_signed=*/false,
            /*is_integer=*/true
        );

        if (base_type.isNull()) {
            LOG(ERROR) << "Unable to determine base type for undefined type.\n";
            base_type = ctx.IntTy;
        }

        auto *typedef_decl = clang::TypedefDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), clang::SourceLocation(), clang::SourceLocation(),
            &ctx.Idents.get(undefined_type.name), ctx.getTrivialTypeSourceInfo(base_type)
        );

        typedef_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(typedef_decl);

        return ctx.getTypedefType(typedef_decl);
    }

} // namespace patchestry::ast
