/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Ghidra/Pcode.hpp"
#include <cassert>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclBase.h>
#include <clang/AST/Expr.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <clang/Basic/ExceptionSpecificationType.h>
#include <clang/Basic/LLVM.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/Specifiers.h>
#include <llvm/Support/Casting.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>

namespace patchestry::ast {

    void PcodeASTConsumer::HandleTranslationUnit(clang::ASTContext &ctx) {
        if (get_program().serialized_types.size() > 0) {
            create_types(ctx, get_program().serialized_types);
        }

        if (get_program().serialized_functions.size() > 0) {
            create_functions(
                ctx, get_program().serialized_functions, get_program().serialized_types
            );
        }

        // TODO: Create Operation node for ASTs
    }

    void PcodeASTConsumer::create_types(clang::ASTContext &ctx, TypeMap &type_map) {
        for (auto &[key, vnode_type] : type_map) {
            serialized_types_clang.emplace(key, create_type(ctx, vnode_type));
        }

        // Traverse through serialized_types_clang and complete definitions
        for (auto &[key, decl] : incomplete_definition) {
            if (const auto *record_decl = llvm::dyn_cast< clang::RecordDecl >(decl)) {
                auto iter = type_map.find(key);
                if (iter == type_map.end()) {
                    llvm::errs() << "Key not found in type map\n";
                    assert(false);
                    continue;
                }
                auto vnode_type = iter->second;
                create_record_definition(
                    ctx, dynamic_cast< CompositeType & >(*vnode_type), decl,
                    serialized_types_clang
                );
            }
        }
    }

    clang::QualType PcodeASTConsumer::create_type(
        clang::ASTContext &ctx, const std::shared_ptr< VarnodeType > &vnode_type
    ) {
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
                return ctx.VoidTy;
            case VarnodeType::VT_VOID: {
                return ctx.VoidTy;
            }
        }
    }

    clang::QualType PcodeASTConsumer::create_typedef_type(
        clang::ASTContext &ctx, const TypedefType &typedef_type
    ) {
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
        auto *tinfo          = ctx.getTrivialTypeSourceInfo(underlying_type);
        auto *typedef_decl   = clang::TypedefDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), clang::SourceLocation(), clang::SourceLocation(),
            &identifier, tinfo
        );

        return ctx.getTypedefType(typedef_decl);
    }

    clang::QualType PcodeASTConsumer::create_pointer_type(
        clang::ASTContext &ctx, const PointerType &pointer_type
    ) {
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
        return ctx.getPointerType(pointee_type);
    }

    clang::QualType
    PcodeASTConsumer::create_array_type(clang::ASTContext &ctx, const ArrayType &array_type) {
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
            auto size         = array_type.get_element_count();
            auto num_bits     = 32u;
            return ctx.getConstantArrayType(
                element_type, llvm::APInt(num_bits, size), nullptr,
                clang::ArraySizeModifier::Normal, 0
            );
        }
        assert(false);
        return clang::QualType();
    }

    clang::QualType PcodeASTConsumer::create_composite_type(
        clang::ASTContext &ctx, const VarnodeType &composite_type
    ) {
        auto &identifier = ctx.Idents.get(composite_type.name);
        auto tag_kind    = [&] -> clang::TagDecl::TagKind {
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
            ctx, tag_kind, ctx.getTranslationUnitDecl(), clang::SourceLocation(),
            clang::SourceLocation(), &identifier
        );
        incomplete_definition.emplace(composite_type.key, decl);
        return ctx.getRecordType(decl);
    }

    clang::QualType
    PcodeASTConsumer::create_enum_type(clang::ASTContext &ctx, const EnumType &enum_type) {
        auto &identifier = ctx.Idents.get(enum_type.name);
        auto *enum_decl  = clang::EnumDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), clang::SourceLocation(), clang::SourceLocation(),
            &identifier, nullptr, true, false, false
        );
        return ctx.getEnumType(enum_decl);
    }

    void PcodeASTConsumer::create_record_definition(
        clang::ASTContext &ctx, const CompositeType &varnode, clang::Decl *prev_decl,
        const ASTTypeMap &clang_types
    ) {
        auto &identifier  = ctx.Idents.get(varnode.name);
        auto *record_decl = clang::RecordDecl::Create(
            ctx, clang::TagDecl::TagKind::Struct, ctx.getTranslationUnitDecl(),
            clang::SourceLocation(), clang::SourceLocation(), &identifier,
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
        record_decl->dump();
    }

    clang::QualType PcodeASTConsumer::create_function_prototype(
        clang::ASTContext &ctx, FunctionPrototype &proto
    ) {
        auto return_key = proto.rttype_key;
        auto iter       = serialized_types_clang.find(return_key);
        if (iter == serialized_types_clang.end()) {
            llvm::errs() << "Function return type is not found\n";
            assert(false);
            return clang::QualType();
        }
        auto rttype = iter->second;

        std::vector< clang::QualType > args_vec;
        for (const auto &param : proto.parameters) {
            auto param_iter = serialized_types_clang.find(param);
            if (param_iter == serialized_types_clang.end()) {
                assert(false);
            }
            args_vec.push_back(param_iter->second);
        }
        clang::FunctionProtoType::ExtProtoInfo proto_info;
        proto_info.Variadic = proto.is_variadic;
        if (proto.is_noreturn) {
            proto_info.ExceptionSpec.Type = clang::EST_DependentNoexcept;
        }

        return ctx.getFunctionType(rttype, args_vec, proto_info);
    }

    void PcodeASTConsumer::create_functions(
        clang::ASTContext &ctx, FunctionMap &serialized_functions, TypeMap &serialized_types
    ) {
        for (auto &[key, function] : serialized_functions) {
            auto function_name = function.name;
            auto proto_type    = create_function_prototype(ctx, function.prototype);
            auto *func_decl    = clang::FunctionDecl::Create(
                ctx, ctx.getTranslationUnitDecl(), clang::SourceLocation(),
                clang::SourceLocation(), &ctx.Idents.get(function_name), proto_type, nullptr,
                clang::SC_None
            );

            auto entry_block_key = function.entry_block;
            auto iter            = function.basic_blocks.find(entry_block_key);
            if (iter == function.basic_blocks.end()) {
                assert(false);
            }
            auto entry_block = iter->second;
            create_function_parameters(ctx, func_decl, entry_block);

            for (auto &op_key : entry_block.ordered_operations) {
                auto op_iter = entry_block.operations.find(op_key);
                if (op_iter == entry_block.operations.end()) {
                    llvm::errs() << "Operation with key " << op_key
                                 << " is not found in entry block.\n";
                    // assert(false); // disable assert because it is getting hit because of
                    // missing operations from entry block.
                    continue;
                }
                auto operation = op_iter->second;
                switch (operation.mnemonic) {
                    case patchestry::ghidra::Mnemonic::OP_DECLARE_LOCAL_VAR: {
                        auto variable_name = operation.name;
                        auto variable_type =
                            serialized_types_clang.find(operation.type)->second;
                        clang::VarDecl *var_decl = clang::VarDecl::Create(
                            ctx, func_decl, clang::SourceLocation(), clang::SourceLocation(),
                            &ctx.Idents.get(variable_name), variable_type, nullptr,
                            clang::SC_None
                        );

                        func_decl->setBody(clang::CompoundStmt::Create(
                            ctx, { create_decl_stmt(ctx, var_decl) },
                            clang::FPOptionsOverride(), clang::SourceLocation(),
                            clang::SourceLocation()
                        ));
                        break;
                    }
                    case patchestry::ghidra::Mnemonic::OP_DECLARE_PARAM_VAR: {
                        break;
                    }
                    case patchestry::ghidra::Mnemonic::OP_COPY: {
                        auto output_vnode = operation.output.front();

                        break;
                    }
                    default:
                        break;
                }
            }
            func_decl->dump();
        }
        (void) serialized_types;
    }

    void PcodeASTConsumer::create_function_parameters(
        clang::ASTContext &ctx, clang::FunctionDecl *func_decl, const BasicBlock &entry
    ) {
        std::vector< clang::ParmVarDecl * > params;
        for (const auto &order_key : entry.ordered_operations) {
            auto iter = entry.operations.find(order_key);
            if (iter == entry.operations.end()) {
                continue;
            }
            auto operation = iter->second;
            if (operation.mnemonic == Mnemonic::OP_DECLARE_PARAM_VAR) {
                auto &identifier = ctx.Idents.get(operation.name);
                auto type_iter   = serialized_types_clang.find(operation.type);

                auto *param_decl = clang::ParmVarDecl::Create(
                    ctx, func_decl, clang::SourceLocation(), clang::SourceLocation(),
                    &identifier, type_iter->second, nullptr, clang::SC_None, nullptr
                );
                params.push_back(param_decl);
            }
        }

        if (func_decl->getNumParams() != params.size()) {
            llvm::errs() << "Number of params decl does not match with function prototype\n";
            return;
        }

        func_decl->setParams(params);
    }

    clang::VarDecl *PcodeASTConsumer::create_variable_decl(
        clang::ASTContext &ctx, clang::DeclContext &dc, const std::string &var_name,
        clang::QualType var_type
    ) {
        auto &identifier = ctx.Idents.get(var_name);
        return clang::VarDecl::Create(
            ctx, &dc, clang::SourceLocation(), clang::SourceLocation(), &identifier, var_type,
            nullptr, clang::SC_None
        );
    }

    clang::BinaryOperator *PcodeASTConsumer::create_assignment_stmt(
        clang::ASTContext &ctx, clang::Expr *lhs, clang::Expr *rhs
    ) {
        return clang::BinaryOperator::Create(
            ctx, lhs, rhs, clang::BO_Assign, lhs->getType(), clang::VK_PRValue,
            clang::OK_Ordinary, clang::SourceLocation(), clang::FPOptions()
        );
    }

    clang::DeclStmt *
    PcodeASTConsumer::create_decl_stmt(clang::ASTContext &ctx, clang::Decl *decl) {
        auto decl_group = clang::DeclGroupRef(decl);
        return new (ctx)
            clang::DeclStmt(decl_group, clang::SourceLocation(), clang::SourceLocation());
    }

} // namespace patchestry::ast
