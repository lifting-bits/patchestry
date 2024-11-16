/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/TypeBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/PcodeTypes.hpp>

namespace patchestry::ast {

    void TypeBuilder::create_types(clang::ASTContext &ctx, TypeMap &lifted_types) {
        for (auto &[key, vnode_type] : lifted_types) {
            serialized_types.emplace(key, create_type(ctx, vnode_type));
        }

        // Traverse through missing_type_definition list and complete definitions
        for (auto &[key, decl] : missing_type_definition) {
            if (const auto *record_decl = llvm::dyn_cast< clang::RecordDecl >(decl)) {
                auto iter = lifted_types.find(key);
                if (iter == lifted_types.end()) {
                    llvm::errs() << "Key not found in type map\n";
                    assert(false);
                    continue;
                }
                auto vnode_type = iter->second;
                create_record_definition(
                    ctx, dynamic_cast< CompositeType & >(*vnode_type), decl, serialized_types
                );
            }
        }
    }

    clang::QualType TypeBuilder::create_type(
        clang::ASTContext &ctx, const std::shared_ptr< VarnodeType > &vnode_type
    ) {
        auto type_iter = serialized_types.find(vnode_type->key);
        if (type_iter != serialized_types.end()) {
            return type_iter->second;
        }

        switch (vnode_type->kind) {
            case VarnodeType::VT_INVALID:
                return ctx.CharTy;
            case VarnodeType::VT_BOOLEAN:
                return ctx.BoolTy;
            case VarnodeType::VT_INTEGER:
                return ctx.IntTy;
            case VarnodeType::VT_CHAR:
                return ctx.CharTy;
            case VarnodeType::VT_FLOAT:
                return ctx.FloatTy;
            case VarnodeType::VT_ARRAY:
                return create_array_type(ctx, dynamic_cast< const ArrayType & >(*vnode_type));
            case VarnodeType::VT_POINTER:
                return create_pointer_type(
                    ctx, dynamic_cast< const PointerType & >(*vnode_type)
                );
            case VarnodeType::Kind::VT_FUNCTION:
                return ctx.VoidPtrTy;
            case VarnodeType::VT_STRUCT:
            case VarnodeType::VT_UNION:
                return create_composite_type(ctx, *vnode_type);
            case VarnodeType::VT_ENUM:
                return create_enum_type(ctx, dynamic_cast< const EnumType & >(*vnode_type));
            case VarnodeType::VT_TYPEDEF:
                return create_typedef_type(
                    ctx, dynamic_cast< const TypedefType & >(*vnode_type)
                );
            case VarnodeType::VT_UNDEFINED:
                return create_undefined_type(
                    ctx, dynamic_cast< const UndefinedType & >(*vnode_type)
                );
            case VarnodeType::VT_VOID: {
                return ctx.VoidTy;
            }
        }
    }

    clang::QualType
    TypeBuilder::create_typedef_type(clang::ASTContext &ctx, const TypedefType &typedef_type) {
        auto &identifier = ctx.Idents.get(typedef_type.name);
        auto base_type   = typedef_type.get_base_type();
        if (!base_type) {
            llvm::errs() << "Base Type of a typedef shouldn't be empty. key: "
                         << typedef_type.key << "\n";
            assert(false);
            return clang::QualType();
        }

        if (base_type->key == typedef_type.key) {
            llvm::errs() << "Base Type of typedef is pointing to itself. key: "
                         << typedef_type.key << "\n";
            assert(false);
            return clang::QualType();
        }

        auto underlying_type = create_type(ctx, base_type);
        serialized_types.emplace(base_type->key, underlying_type);
        auto *tinfo        = ctx.getTrivialTypeSourceInfo(underlying_type);
        auto *typedef_decl = clang::TypedefDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), clang::SourceLocation(), clang::SourceLocation(),
            &identifier, tinfo
        );

        typedef_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(typedef_decl);

        return ctx.getTypedefType(typedef_decl);
    }

    clang::QualType
    TypeBuilder::create_pointer_type(clang::ASTContext &ctx, const PointerType &pointer_type) {
        auto pointee = pointer_type.get_pointee_type();
        if (!pointee) {
            llvm::errs() << "No pointee type in pointer with key " << pointer_type.key << "\n";
            assert(false);
            return ctx.VoidPtrTy;
        }
        if (pointee->key == pointer_type.key) {
            llvm::errs() << "Pointer type shouldn't have itself as pointee. key: "
                         << pointer_type.key << "\n";
            assert(false);
            return clang::QualType();
        }

        auto pointee_type = create_type(ctx, pointee);
        serialized_types.emplace(pointee->key, pointee_type);
        return ctx.getPointerType(pointee_type);
    }

    clang::QualType
    TypeBuilder::create_array_type(clang::ASTContext &ctx, const ArrayType &array_type) {
        auto element = array_type.get_element_type();
        if (!element) {
            llvm::errs() << "No element types for array\n";
            assert(false);
            return clang::QualType();
        }

        // If element key is same as array_type key, it will lead to infinite recursive. If it
        // happens something is wrong and need to check ghidra scripts
        if (element->key != array_type.key) {
            auto element_type = create_type(ctx, element);
            serialized_types.emplace(element->key, element_type);
            auto size     = array_type.get_element_count();
            auto num_bits = 32u;
            return ctx.getConstantArrayType(
                element_type, llvm::APInt(num_bits, size), nullptr,
                clang::ArraySizeModifier::Normal, 0
            );
        }
        assert(false);
        return clang::QualType();
    }

    void TypeBuilder::create_record_definition(
        clang::ASTContext &ctx, const CompositeType &varnode, clang::Decl *prev_decl,
        const ASTTypeMap &clang_types
    ) {
        auto &identifier  = ctx.Idents.get(varnode.name);
        auto *record_decl = clang::RecordDecl::Create(
            ctx, clang::TagDecl::TagKind::Struct, ctx.getTranslationUnitDecl(),
            source_location_from_key(ctx, varnode.key),
            source_location_from_key(ctx, varnode.key), &identifier,
            llvm::dyn_cast< clang::RecordDecl >(prev_decl)
        );

        record_decl->completeDefinition();
        auto components = varnode.get_components();
        for (auto &comp : components) {
            auto type_key = comp.type->key;
            auto iter     = clang_types.find(type_key);
            if (iter == clang_types.end()) {
                assert(false);
                continue;
            }

            auto field_type  = iter->second;
            auto *field_decl = clang::FieldDecl::Create(
                ctx, record_decl, clang::SourceLocation(), clang::SourceLocation(),
                &ctx.Idents.get(comp.name), field_type, nullptr, nullptr, false,
                clang::ICIS_NoInit
            );
            record_decl->addDecl(field_decl);
        }

        record_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(record_decl);
    }

    clang::QualType TypeBuilder::create_composite_type(
        clang::ASTContext &ctx, const VarnodeType &composite_type
    ) {
        auto tag_kind = [&]() -> clang::TagDecl::TagKind {
            switch (composite_type.kind) {
                case VarnodeType::Kind::VT_STRUCT:
                    return clang::TagDecl::TagKind::Struct;
                case VarnodeType::Kind::VT_UNION:
                    return clang::TagDecl::TagKind::Union;
                default:
                    assert(false);
                    return clang::TagDecl::TagKind::Struct;
            }
        }();
        auto *decl = clang::RecordDecl::Create(
            ctx, tag_kind, ctx.getTranslationUnitDecl(),
            source_location_from_key(ctx, composite_type.key),
            source_location_from_key(ctx, composite_type.key),
            &ctx.Idents.get(composite_type.name)
        );

        decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(decl);
        missing_type_definition.emplace(composite_type.key, decl);
        return ctx.getRecordType(decl);
    }

    clang::QualType
    TypeBuilder::create_enum_type(clang::ASTContext &ctx, const EnumType &enum_type) {
        auto &identifier = ctx.Idents.get(enum_type.name);
        auto *enum_decl  = clang::EnumDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), source_location_from_key(ctx, enum_type.key),
            source_location_from_key(ctx, enum_type.key), &identifier, nullptr, true, false,
            false
        );

        enum_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(enum_decl);
        return ctx.getEnumType(enum_decl);
    }

    clang::QualType TypeBuilder::create_undefined_type(
        clang::ASTContext &ctx, const UndefinedType &undefined_type
    ) {
        if (undefined_type.kind != VarnodeType::Kind::VT_UNDEFINED) {
            assert(false);
            return clang::QualType();
        }

        auto base_type = get_type_for_size(
            ctx, undefined_type.size * 8, /*is_signed=*/false, /*is_integer=*/true
        );

        if (base_type.isNull()) {
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
