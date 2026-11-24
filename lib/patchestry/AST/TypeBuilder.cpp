/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <algorithm>
#include <string>
#include <vector>

#include <clang/AST/Attr.h>
#include <clang/AST/Expr.h>
#include <clang/AST/RecordLayout.h>
#include <clang/AST/Type.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/TargetInfo.h>

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

    namespace {

        // `natural` when its storage matches the recovered byte size, otherwise
        // an integer of the recovered width with `natural`'s signedness.  A
        // recovered `bool`/`char`/`wchar_t` names how the value is read, not how
        // many bytes the object occupies: lifting a four-byte boolean as C
        // `bool` would shrink the object and normalize its values.  Size 0
        // means the width is unknown, so `natural` stands.
        clang::QualType storage_faithful(
            clang::ASTContext &ctx, clang::QualType natural, uint32_t size_in_bytes
        ) {
            const auto bit_size = size_in_bytes * TypeBuilder::kNumBitsInByte;
            if (bit_size == 0 || ctx.getTypeSize(natural) == bit_size) { return natural; }

            const bool is_signed = !natural->isBooleanType() && natural->isSignedIntegerType();
            auto sized = ctx.getIntTypeForBitwidth(
                bit_size, static_cast< unsigned int >(is_signed)
            );
            // Null when the target has no integer of that width; the caller
            // reports it and keeps `natural`.
            return sized;
        }

    } // namespace

    void TypeBuilder::create_types(clang::ASTContext &ctx, TypeMap &lifted_types) {
        lifted_types_ = &lifted_types;
        for (auto &[key, vnode_type] : lifted_types) {
            serialized_types.emplace(key, create_type(ctx, vnode_type));
        }

        // Traverse through missing type definition for a composite type and create the complete
        // definition for them
        for (auto &[key, decl] : missing_type_definition) {
            if (llvm::isa< clang::RecordDecl >(decl)) {
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

        // A recovered bool/char/wchar_t keeps its recovered byte size.
        auto with_storage_width = [&](clang::QualType natural) {
            auto sized = storage_faithful(ctx, natural, vnode_type->size);
            if (sized.isNull()) {
                report_fallback(
                    ctx, vnode_type->key,
                    "recovered '" + natural.getAsString() + "' of "
                        + std::to_string(vnode_type->size)
                        + " byte(s) has no matching integer type and keeps its natural width"
                );
                return natural;
            }
            return sized;
        };

        switch (vnode_type->kind) {
            case VarnodeType::VT_INVALID:
                return create_invalid(ctx, vnode_type);

            case VarnodeType::VT_BOOLEAN:
                return with_storage_width(ctx.BoolTy);

            case VarnodeType::VT_INTEGER: {
                auto &builtin = dynamic_cast< const BuiltinType & >(*vnode_type);
                return GetTypeFromSize(
                    ctx, vnode_type->size * TypeBuilder::kNumBitsInByte,
                    /*is_signed=*/builtin.is_signed, /*is_integer=*/true
                );
            }

            case VarnodeType::VT_CHAR:
                return with_storage_width(ctx.CharTy);

            case VarnodeType::VT_WIDECHAR:
                return with_storage_width(ctx.WideCharTy);

            case VarnodeType::VT_FLOAT:
                return GetTypeFromSize(
                    ctx, vnode_type->size * TypeBuilder::kNumBitsInByte, /*is_signed=*/false,
                    /*is_integer=*/false
                );

            case VarnodeType::VT_ARRAY:
                // Undefined byte-blobs (`undefinedN[...]`) lift as plain
                // `undefined1[N]` arrays, not a `struct struct_undefinedN` wrapper.
                return create_array(ctx, dynamic_cast< const ArrayType & >(*vnode_type));
            case VarnodeType::VT_POINTER:
                return create_pointer(ctx, dynamic_cast< const PointerType & >(*vnode_type));

            case VarnodeType::Kind::VT_FUNCTION: {
                const auto &fn_type = dynamic_cast< const FunctionType & >(*vnode_type);

                // Resolve return type (fall back to void if key is absent or unknown)
                clang::QualType return_type = ctx.VoidTy;
                if (lifted_types_ != nullptr && !fn_type.return_type_key.empty()) {
                    auto it = lifted_types_->find(fn_type.return_type_key);
                    if (it != lifted_types_->end()) {
                        return_type = create_type(ctx, it->second);
                    } else {
                        report_fallback(
                            ctx, fn_type.key,
                            "unknown return type '" + fn_type.return_type_key
                                + "' of a function type is replaced with void"
                        );
                    }
                }

                // Resolve parameter types
                llvm::SmallVector< clang::QualType, 8 > param_types;
                if (lifted_types_ != nullptr) {
                    for (const auto &pk : fn_type.param_type_keys) {
                        auto it = lifted_types_->find(pk);
                        if (it != lifted_types_->end()) {
                            param_types.push_back(create_type(ctx, it->second));
                        } else {
                            report_fallback(
                                ctx, fn_type.key,
                                "unknown parameter type '" + pk
                                    + "' of a function type is replaced with void *"
                            );
                            param_types.push_back(ctx.getPointerType(ctx.VoidTy));
                        }
                    }
                }

                clang::FunctionProtoType::ExtProtoInfo epi;
                epi.Variadic = fn_type.is_variadic;
                clang::QualType fn_qual =
                    ctx.getFunctionType(return_type, param_types, epi);
                return ctx.getPointerType(fn_qual);
            }

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

            case VarnodeType::VT_BITFIELD: {
                // BitField: return the base integer type.  The bit width is
                // applied later when creating FieldDecls in composites.
                auto &bf = dynamic_cast< const BitFieldType & >(*vnode_type);
                if (bf.GetBaseType()) {
                    return create_type(ctx, bf.GetBaseType());
                }
                return GetTypeFromSize(
                    ctx, vnode_type->size * TypeBuilder::kNumBitsInByte,
                    /*is_signed=*/false, /*is_integer=*/true
                );
            }

            case VarnodeType::VT_STRING: {
                // String: char[N] if size is known, else char*.
                if (vnode_type->size > 0) {
                    return ctx.getConstantArrayType(
                        ctx.CharTy,
                        llvm::APInt(TypeBuilder::kNumBitsUint, vnode_type->size), nullptr,
                        clang::ArraySizeModifier::Normal, 0
                    );
                }
                return ctx.getPointerType(ctx.CharTy);
            }

            case VarnodeType::VT_VOID: {
                return ctx.VoidTy;
            }
        }

        return clang::QualType();
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

    clang::QualType TypeBuilder::create_invalid(
        clang::ASTContext &ctx, const std::shared_ptr< VarnodeType > &vnode_type
    ) {
        // An invalid type is defined as typedef of a void type
        auto underlying_type = ctx.VoidTy;
        auto ty_loc          = SourceLocation(ctx.getSourceManager(), vnode_type->key);
        auto *typedef_decl   = clang::TypedefDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), ty_loc, ty_loc, &ctx.Idents.get("invalid_type"),
            ctx.getTrivialTypeSourceInfo(underlying_type)
        );

        typedef_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(typedef_decl);

        return ctx.getTypedefType(clang::ElaboratedTypeKeyword::None, std::nullopt, typedef_decl);
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
        if (!typedef_type.GetBaseType()) {
            LOG(ERROR) << "Base Type of a typedef shouldn't be empty. key: " << typedef_type.key
                       << "\n";
            return {};
        }

        const auto &base_type = typedef_type.GetBaseType();
        if (base_type->key == typedef_type.key) {
            LOG(ERROR) << "Base Type of typedef is pointing to itself. key: "
                       << typedef_type.key << "\n";
            return {};
        }

        auto underlying_type = create_type(ctx, base_type);
        serialized_types.emplace(base_type->key, underlying_type);

        auto *tinfo        = ctx.getTrivialTypeSourceInfo(underlying_type);
        auto tydef_loc     = SourceLocation(ctx.getSourceManager(), typedef_type.key);
        auto *typedef_decl = clang::TypedefDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), tydef_loc, tydef_loc,
            &ctx.Idents.get(typedef_type.name), tinfo
        );

        typedef_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(typedef_decl);
        return ctx.getTypedefType(clang::ElaboratedTypeKeyword::None, std::nullopt, typedef_decl);
    }

    void TypeBuilder::report_fallback(
        clang::ASTContext &ctx, const std::string &type_key, const std::string &what
    ) const {
        if (loss_reporter == nullptr) {
            LOG(ERROR) << "type '" << type_key << "': " << what << "\n";
            return;
        }
        loss_reporter->report(
            SourceLocation(ctx.getSourceManager(), type_key), "type-fallback",
            "type '" + type_key + "': " + what
        );
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
        if (!pointer_type.GetPointeeType()) {
            report_fallback(
                ctx, pointer_type.key, "a pointer with no pointee type is replaced with void *"
            );
            return ctx.VoidPtrTy;
        }

        const auto &pointee = pointer_type.GetPointeeType();
        if (pointee->key == pointer_type.key) {
            report_fallback(
                ctx, pointer_type.key, "a pointer to itself is replaced with void *"
            );
            return ctx.VoidPtrTy;
        }

        auto pointee_type = create_type(ctx, pointee);
        serialized_types.emplace(pointee->key, pointee_type);
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
        if (!array_type.GetElementType()) {
            LOG(ERROR) << "No element types for array\n";
            return {};
        }

        const auto &element = array_type.GetElementType();
        if (element->key == array_type.key) {
            LOG(ERROR) << "Element key is same as array key. Key " << element->key << "\n";
            return {};
        }

        // If element key is same as array_type key, it will lead to infinite recursive. If it
        // happens something is wrong and need to check ghidra scripts
        auto element_type = create_type(ctx, element);
        serialized_types.emplace(element->key, element_type);

        auto size = array_type.GetElementCount();
        return ctx.getConstantArrayType(
            element_type, llvm::APInt(kNumBitsUint, size), nullptr,
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
        if (!completed_definitions.insert(varnode.key).second) {
            return;
        }

        const bool is_union = varnode.kind == VarnodeType::VT_UNION;
        // A record with no recovered members is opaque (libc `FILE`, a type
        // only used through pointers); its size says nothing about a layout.
        const bool is_opaque = varnode.GetComponents().empty();
        const auto kind_name = std::string(is_union ? "union" : "struct");

        // Struct members are laid out in offset order; the serializer already
        // writes them that way, the sort only guards hand-written input.
        auto components = varnode.GetComponents();
        if (!is_union) {
            std::stable_sort(
                components.begin(), components.end(),
                [](const auto &lhs, const auto &rhs) { return lhs.offset < rhs.offset; }
            );
        }

        // A member whose type is another recovered record needs that record's
        // layout, so complete it first.  A record cannot contain itself by
        // value; `completed_definitions` stops a malformed cycle.
        for (const auto &component : components) {
            auto iter = clang_types.find(component.type->key);
            if (iter == clang_types.end() || iter->second.isNull()) {
                continue;
            }
            const auto *record_type =
                ctx.getBaseElementType(iter->second)->getAs< clang::RecordType >();
            if (record_type == nullptr || lifted_types_ == nullptr) {
                continue;
            }
            auto nested = definition_keys.find(record_type->getDecl());
            if (nested == definition_keys.end()
                || completed_definitions.contains(nested->second))
            {
                continue;
            }
            auto lifted  = lifted_types_->find(nested->second);
            auto forward = missing_type_definition.find(nested->second);
            if (lifted != lifted_types_->end() && forward != missing_type_definition.end()) {
                complete_definition(
                    ctx, dynamic_cast< CompositeType & >(*lifted->second), forward->second,
                    clang_types
                );
            }
        }

        // Plan the members before creating declarations: every recovered
        // component keeps its offset, gaps become explicit padding, and the
        // record is packed when natural alignment would move a member.
        struct Member
        {
            std::string name;
            clang::QualType type;
            const CompositeType::Component *component = nullptr; // null for padding
        };

        std::vector< Member > members;
        std::vector< std::string > problems;
        uint64_t cursor     = 0; // first byte not yet covered by a member
        uint64_t max_align  = 1;
        uint64_t union_size = 0;
        bool needs_packed   = false;
        bool verifiable     = true;
        // Members already reported as overlapping; their offset is not
        // reported a second time.
        std::vector< const CompositeType::Component * > overlapping;

        auto add_padding = [&](uint64_t offset, uint64_t size) {
            members.push_back({ "__pad_" + std::to_string(offset),
                                ctx.getConstantArrayType(
                                    ctx.UnsignedCharTy, llvm::APInt(kNumBitsUint, size), nullptr,
                                    clang::ArraySizeModifier::Normal, 0
                                ),
                                nullptr });
        };

        for (const auto &component : components) {
            auto iter = clang_types.find(component.type->key);
            if (iter == clang_types.end() || iter->second.isNull()) {
                // No usable type: the bytes stay in the record as padding (the
                // next member's gap, or the tail) rather than disappearing.
                // The layout is kept but the member itself is gone, so it is
                // still a loss.
                report_fallback(
                    ctx, varnode.key,
                    "a record field '" + component.name + "' of unknown type '"
                        + component.type->key + "' is kept as unnamed padding"
                );
                continue;
            }

            auto field_type = iter->second;
            // A string member's type has no size of its own (it lifts as
            // `char *`); inside a record it is the `char[N]` the member
            // occupies.
            // JSON without member sizes: the string runs to the next member,
            // or to the end of the record.
            uint64_t string_size = component.size;
            if (component.type->kind == VarnodeType::VT_STRING && string_size == 0
                && !is_union)
            {
                uint64_t end = varnode.size;
                for (const auto &other : components) {
                    if (other.offset > component.offset) {
                        end = other.offset;
                        break;
                    }
                }
                string_size = end > component.offset ? end - component.offset : 0;
            }
            if (component.type->kind == VarnodeType::VT_STRING && string_size > 0
                && !field_type->isIncompleteType()
                && static_cast< uint64_t >(ctx.getTypeSizeInChars(field_type).getQuantity())
                    != string_size)
            {
                field_type = ctx.getConstantArrayType(
                    ctx.CharTy, llvm::APInt(kNumBitsUint, string_size), nullptr,
                    clang::ArraySizeModifier::Normal, 0
                );
            }
            if (field_type->isIncompleteType() || field_type->isFunctionType()) {
                // Size unknown: keep the member as before, but the layout
                // cannot be planned or checked past it.
                members.push_back({ component.name, field_type, &component });
                verifiable = false;
                continue;
            }

            const auto size  = static_cast< uint64_t >(ctx.getTypeSizeInChars(field_type).getQuantity());
            const auto align = static_cast< uint64_t >(ctx.getTypeAlignInChars(field_type).getQuantity());

            if (is_union) {
                if (component.offset != 0) {
                    problems.push_back(
                        "member '" + component.name + "' has recovered offset "
                        + std::to_string(component.offset) + " in a union"
                    );
                }
                members.push_back({ component.name, field_type, &component });
                union_size = std::max(union_size, size);
                max_align  = std::max(max_align, align);
                continue;
            }

            if (component.type->kind == VarnodeType::VT_BITFIELD) {
                // Bit-fields that share a storage unit have overlapping byte
                // ranges; Clang places them, the check below verifies where.
                if (component.offset > cursor) {
                    add_padding(cursor, component.offset - cursor);
                }
                members.push_back({ component.name, field_type, &component });
                cursor    = std::max< uint64_t >(cursor, component.offset + component.type->size);
                max_align = std::max(max_align, align);
                continue;
            }

            if (component.offset < cursor) {
                problems.push_back(
                    "member '" + component.name + "' at recovered offset "
                    + std::to_string(component.offset) + " overlaps the member before it"
                );
                overlapping.push_back(&component);
            } else if (component.offset > cursor) {
                add_padding(cursor, component.offset - cursor);
            }
            if (component.offset % align != 0) {
                needs_packed = true;
            }
            members.push_back({ component.name, field_type, &component });
            cursor    = std::max< uint64_t >(cursor, component.offset + size);
            max_align = std::max(max_align, align);
        }

        if (verifiable && !is_opaque && varnode.size > 0) {
            if (is_union) {
                if (varnode.size > union_size) {
                    add_padding(0, varnode.size);
                }
            } else if (varnode.size > cursor) {
                add_padding(cursor, varnode.size - cursor);
            }
            if (varnode.size % max_align != 0) {
                needs_packed = true;
            }
        }

        auto *record_decl = clang::RecordDecl::Create(
            ctx, is_union ? clang::TagDecl::TagKind::Union : clang::TagDecl::TagKind::Struct,
            ctx.getTranslationUnitDecl(), prev_decl->getBeginLoc(), prev_decl->getLocation(),
            &ctx.Idents.get(varnode.name), llvm::dyn_cast< clang::RecordDecl >(prev_decl)
        );
        record_decl->startDefinition();

        std::vector< std::pair< clang::FieldDecl *, const CompositeType::Component * > > fields;
        for (const auto &member : members) {
            auto location = SourceLocation(
                ctx.getSourceManager(),
                member.component != nullptr ? member.component->type->key : varnode.key
            );

            // For bitfield components, create a ConstantExpr wrapping the bit width.
            clang::Expr *bit_width = nullptr;
            if (member.component != nullptr
                && member.component->type->kind == VarnodeType::VT_BITFIELD)
            {
                auto &bf = dynamic_cast< const BitFieldType & >(*member.component->type);
                auto *int_lit = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(TypeBuilder::kNumBitsUint, bf.bit_size),
                    ctx.UnsignedIntTy, location
                );
                bit_width = clang::ConstantExpr::Create(
                    ctx, int_lit, clang::APValue(llvm::APSInt(
                        llvm::APInt(TypeBuilder::kNumBitsUint, bf.bit_size), /*isUnsigned=*/true
                    ))
                );
            }

            auto *field_decl = clang::FieldDecl::Create(
                ctx, record_decl, location, location, &ctx.Idents.get(member.name),
                member.type, nullptr, bit_width, false, clang::ICIS_NoInit
            );
            field_decl->setAccess(clang::AS_public);
            record_decl->addDecl(field_decl);
            fields.emplace_back(field_decl, member.component);
        }

        if (needs_packed) {
            record_decl->addAttr(clang::PackedAttr::Create(ctx));
        }
        record_decl->completeDefinition();

        record_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(record_decl);

        // Compare what Clang laid out with what was recovered.  An unchanged
        // callee reads the record at the recovered offsets, so any difference
        // changes what it sees.
        if (verifiable && !is_opaque && varnode.size > 0) {
            const auto &layout      = ctx.getASTRecordLayout(record_decl);
            const bool little_endian = ctx.getTargetInfo().isLittleEndian();
            for (const auto &[field, component] : fields) {
                if (component == nullptr
                    || std::find(overlapping.begin(), overlapping.end(), component)
                        != overlapping.end())
                {
                    continue;
                }
                uint64_t expected = static_cast< uint64_t >(component->offset) * kNumBitsInByte;
                if (component->type->kind == VarnodeType::VT_BITFIELD) {
                    // Ghidra's bit offset is the right shift within the
                    // big-endian view of the member's minimal storage.
                    const auto &bf = dynamic_cast< const BitFieldType & >(*component->type);
                    const uint64_t storage_bits =
                        static_cast< uint64_t >(component->type->size) * kNumBitsInByte;
                    expected += little_endian
                        ? bf.bit_offset
                        : storage_bits - std::min< uint64_t >(
                              storage_bits, bf.bit_offset + bf.bit_size
                          );
                }
                const uint64_t actual = layout.getFieldOffset(field->getFieldIndex());
                if (actual != expected) {
                    problems.push_back(
                        "member '" + component->name + "' is at bit " + std::to_string(actual)
                        + " in the lifted record, recovered at bit " + std::to_string(expected)
                    );
                }
            }
            const auto lifted_size = static_cast< uint64_t >(layout.getSize().getQuantity());
            if (lifted_size != varnode.size) {
                problems.push_back(
                    "lifted size is " + std::to_string(lifted_size) + " bytes, recovered size is "
                    + std::to_string(varnode.size)
                );
            }
        }

        if (loss_reporter != nullptr) {
            for (const auto &problem : problems) {
                loss_reporter->report(
                    SourceLocation(ctx.getSourceManager(), varnode.key), "record-layout-mismatch",
                    kind_name + " '" + varnode.name + "': " + problem
                );
            }
        }
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
            SourceLocation(ctx.getSourceManager(), composite_type.key),
            SourceLocation(ctx.getSourceManager(), composite_type.key),
            &ctx.Idents.get(composite_type.name)
        );

        decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(decl);

        // Track the missing type definition for completing at later stage
        missing_type_definition.emplace(composite_type.key, decl);
        definition_keys.emplace(decl, composite_type.key);

        return ctx.getCanonicalTagType(decl);
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
        auto loc = SourceLocation(ctx.getSourceManager(), enum_type.key);
        auto *enum_decl = clang::EnumDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), loc, loc,
            &ctx.Idents.get(enum_type.name), nullptr, false, false, false
        );

        enum_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(enum_decl);

        // Derive the underlying integer type from the serialized enum size.
        unsigned bit_width = enum_type.size * TypeBuilder::kNumBitsInByte;
        if (bit_width == 0) {
            bit_width = TypeBuilder::kNumBitsUint;
        }

        auto underlying_type =
            GetTypeFromSize(ctx, bit_width, /*is_signed=*/true, /*is_integer=*/true);
        if (underlying_type.isNull()) {
            report_fallback(
                ctx, enum_type.key,
                "an enum of " + std::to_string(enum_type.size)
                    + " byte(s) has no matching integer type and is given int"
            );
            underlying_type = ctx.IntTy;
            bit_width       = TypeBuilder::kNumBitsUint;
        }

        // Populate enum with actual serialized constants.  Build each value
        // with implicitTrunc=true: unsigned flag enumerators (e.g. Qt's
        // InputMethodQuery::ImQueryAll = 0xFFFFFFFF in a 4-byte enum) are not
        // valid *signed* bit_width values, so the APInt ctor would otherwise
        // assert (isIntN).  Truncation keeps the low bit_width bits, which
        // already hold the full value here, preserving the bit pattern.
        const auto &constants = enum_type.GetConstants();
        for (const auto &c : constants) {
            auto make_apint = [&] {
                return llvm::APInt(
                    bit_width, static_cast< uint64_t >(c.value),
                    /*isSigned=*/true, /*implicitTrunc=*/true
                );
            };
            auto *val = clang::IntegerLiteral::Create(
                ctx, make_apint(), underlying_type, loc
            );
            auto *ecd = clang::EnumConstantDecl::Create(
                ctx, enum_decl, loc, &ctx.Idents.get(c.name), underlying_type, val,
                llvm::APSInt(make_apint(), /*isUnsigned=*/false)
            );
            enum_decl->addDecl(ecd);
        }

        enum_decl->completeDefinition(underlying_type, underlying_type, bit_width, 0U);

        return ctx.getCanonicalTagType(enum_decl);
    }

    /**
     * @brief Creates a `clang::QualType` for an undefined type.
     *
     * This function generates a typedef for an undefined type based on its size
     * and default attributes. It first attempts to derive an appropriate base type
     * using `GetTypeFromSize`. If unsuccessful, it defaults to the `int` type.
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
        auto base_type = GetTypeFromSize(
            ctx, undefined_type.size * TypeBuilder::kNumBitsInByte, /*is_signed=*/false,
            /*is_integer=*/true
        );

        if (base_type.isNull()) {
            report_fallback(
                ctx, undefined_type.key,
                "an undefined type of " + std::to_string(undefined_type.size)
                    + " byte(s) has no matching integer type and is given int"
            );
            base_type = ctx.IntTy;
        }

        auto *typedef_decl = clang::TypedefDecl::Create(
            ctx, ctx.getTranslationUnitDecl(),
            SourceLocation(ctx.getSourceManager(), undefined_type.key),
            SourceLocation(ctx.getSourceManager(), undefined_type.key),
            &ctx.Idents.get(undefined_type.name), ctx.getTrivialTypeSourceInfo(base_type)
        );

        typedef_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(typedef_decl);

        return ctx.getTypedefType(clang::ElaboratedTypeKeyword::None, std::nullopt, typedef_decl);
    }

} // namespace patchestry::ast
