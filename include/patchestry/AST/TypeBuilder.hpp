/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <clang/AST/ASTContext.h>
#include <functional>
#include <patchestry/Ghidra/JsonDeserialize.hpp>

namespace patchestry::ast {
    using namespace patchestry::ghidra;

    using ASTTypeMap = std::unordered_map< std::string, clang::QualType >;

    class TypeBuilder
    {
      public:
        explicit TypeBuilder(clang::ASTContext &ctx) : context(ctx), serialized_types({}) {}

        TypeBuilder &operator=(const TypeBuilder &)  = delete;
        TypeBuilder(const TypeBuilder &)             = delete;
        TypeBuilder &operator=(const TypeBuilder &&) = delete;
        TypeBuilder(const TypeBuilder &&)            = delete;

        virtual ~TypeBuilder() = default;

        ASTTypeMap &get_serialized_types(void) { return serialized_types; }

        void create_types(clang::ASTContext &ctx, TypeMap &lifted_types);

      private:
        clang::QualType
        create_type(clang::ASTContext &ctx, const std::shared_ptr< VarnodeType > &vnode_type);

        clang::QualType
        create_typedef_type(clang::ASTContext &ctx, const TypedefType &typedef_type);

        clang::QualType
        create_pointer_type(clang::ASTContext &ctx, const PointerType &pointer_type);

        clang::QualType create_array_type(clang::ASTContext &ctx, const ArrayType &array_type);

        clang::QualType
        create_composite_type(clang::ASTContext &ctx, const VarnodeType &composite_type);

        clang::QualType
        create_undefined_type(clang::ASTContext &ctx, const UndefinedType &undefined_type);

        void create_record_definition(
            clang::ASTContext &ctx, const CompositeType &varnode, clang::Decl *prev_decl,
            const ASTTypeMap &clang_types
        );

        clang::QualType create_enum_type(clang::ASTContext &ctx, const EnumType &enum_type);

        clang::ASTContext &get_context(void) { return context.get(); }

        std::unordered_map< std::string, clang::Decl * > missing_type_definition;

        std::reference_wrapper< clang::ASTContext > context;
        ASTTypeMap serialized_types;
    };
} // namespace patchestry::ast
