/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <clang/AST/Type.h>
#include <optional>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/NestedNameSpecifier.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/LLVM.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/Specifiers.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/Support/Casting.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>

namespace patchestry::ast {

    extern clang::QualType get_type_for_size(
        clang::ASTContext &ctx, unsigned bit_size, bool is_signed, bool is_integer
    );

    std::optional< Operation >
    operation_from_key(const Function &function, const std::string &lookup_key) {
        if (function.basic_blocks.empty()) {
            return std::nullopt;
        }

        for (const auto &[_, block] : function.basic_blocks) {
            for (const auto &[operation_key, operation] : block.operations) {
                if (operation_key == lookup_key) {
                    return operation;
                }
            }
        }

        assert(false); // assert if failed to find operation
        return std::nullopt;
    }

    clang::CallExpr *create_function_call(clang::ASTContext &ctx, clang::FunctionDecl *decl) {
        auto *ref_expr = clang::DeclRefExpr::Create(
            ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), decl, false,
            clang::SourceLocation(), decl->getType(), clang::VK_LValue
        );

        return clang::CallExpr::Create(
            ctx, ref_expr, {}, decl->getReturnType(), clang::VK_PRValue,
            clang::SourceLocation(), clang::FPOptionsOverride()
        );
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_operation(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic == Mnemonic::OP_UNKNOWN) {
            llvm::errs() << "Operation with unknown mnemonic. operation key ( " << op.key
                         << " )\n";
            return std::make_pair(nullptr, true);
        }

        switch (op.mnemonic) {
            case Mnemonic::OP_COPY:
                return create_copy(ctx, function, op);
            case Mnemonic::OP_LOAD:
                return create_load(ctx, function, op);
            case Mnemonic::OP_STORE:
                return create_store(ctx, function, op);
            case Mnemonic::OP_BRANCH:
                return create_branch(ctx, function, op);
            case Mnemonic::OP_CBRANCH:
                return create_cbranch(ctx, function, op);
            case Mnemonic::OP_BRANCHIND:
                return create_branchind(ctx, function, op);
            case Mnemonic::OP_CALL:
                return create_call(ctx, function, op);
            case Mnemonic::OP_CALLIND:
                return create_callind(ctx, function, op);
            case Mnemonic::OP_USERDEFINED:
                return create_userdefined(ctx, function, op);
            case Mnemonic::OP_RETURN:
                return create_return(ctx, function, op);
            case Mnemonic::OP_PIECE:
                return create_piece(ctx, function, op);
            case Mnemonic::OP_SUBPIECE:
                return create_subpiece(ctx, function, op);
            case Mnemonic::OP_INT_EQUAL:
                return create_binary_operation< clang::BO_EQ >(ctx, function, op);
            case Mnemonic::OP_INT_NOTEQUAL:
                return create_binary_operation< clang::BO_NE >(ctx, function, op);
            case Mnemonic::OP_INT_LESS:
            case Mnemonic::OP_INT_SLESS:
                return create_binary_operation< clang::BO_LT >(ctx, function, op);
            case Mnemonic::OP_INT_LESSEQUAL:
            case Mnemonic::OP_INT_SLESSEQUAL:
                return create_binary_operation< clang::BO_LE >(ctx, function, op);
            case Mnemonic::OP_INT_ZEXT:
                return create_int_zext(ctx, function, op);
            case Mnemonic::OP_INT_SEXT:
                return create_int_sext(ctx, function, op);
            case Mnemonic::OP_INT_ADD:
                return create_binary_operation< clang::BO_Add >(ctx, function, op);
            case Mnemonic::OP_INT_SUB:
                return create_int_sub(ctx, function, op);
            case Mnemonic::OP_INT_CARRY:
                return create_int_carry(ctx, function, op);
            case Mnemonic::OP_INT_SCARRY:
                return create_int_scarry(ctx, function, op);
            case Mnemonic::OP_INT_SBORROW:
                return create_int_sborrow(ctx, function, op);
            case Mnemonic::OP_INT_2COMP:
                return create_int_2comp(ctx, function, op);
            case Mnemonic::OP_INT_NEGATE:
                return create_unary_operation< clang::UO_LNot >(ctx, function, op);
            case Mnemonic::OP_INT_XOR:
                return create_binary_operation< clang::BO_Xor >(ctx, function, op);
            case Mnemonic::OP_INT_AND:
                return create_binary_operation< clang::BO_And >(ctx, function, op);
            case Mnemonic::OP_INT_OR:
                return create_binary_operation< clang::BO_Or >(ctx, function, op);
            case Mnemonic::OP_INT_LEFT:
                return create_binary_operation< clang::BO_Shl >(ctx, function, op);
            case Mnemonic::OP_INT_RIGHT:
            case Mnemonic::OP_INT_SRIGHT:
                return create_binary_operation< clang::BO_Shr >(ctx, function, op);
            case Mnemonic::OP_INT_MULT:
                return create_binary_operation< clang::BO_Mul >(ctx, function, op);
            case Mnemonic::OP_INT_DIV:
                return create_binary_operation< clang::BO_Div >(ctx, function, op);
            case Mnemonic::OP_INT_REM:
                return create_binary_operation< clang::BO_Rem >(ctx, function, op);
            case Mnemonic::OP_INT_SDIV:
                return create_binary_operation< clang::BO_Div >(ctx, function, op);
            case Mnemonic::OP_INT_SREM:
                return create_binary_operation< clang::BO_Rem >(ctx, function, op);
            case Mnemonic::OP_BOOL_NEGATE:
                return create_unary_operation< clang::UO_LNot >(ctx, function, op);
            case Mnemonic::OP_BOOL_OR:
                return create_binary_operation< clang::BO_Or >(ctx, function, op);
            case Mnemonic::OP_BOOL_AND:
            case Mnemonic::OP_FLOAT_EQUAL:
                return create_float_equal(ctx, function, op);
            case Mnemonic::OP_FLOAT_NOTEQUAL:
                return create_float_notequal(ctx, function, op);
            case Mnemonic::OP_FLOAT_LESS:
                return create_float_less(ctx, function, op);
            case Mnemonic::OP_FLOAT_LESSEQUAL:
                return create_float_lessequal(ctx, function, op);
            case Mnemonic::OP_FLOAT_ADD:
                return create_float_add(ctx, function, op);
            case Mnemonic::OP_FLOAT_SUB:
                return create_float_sub(ctx, function, op);
            case Mnemonic::OP_FLOAT_MULT:
                return create_float_mult(ctx, function, op);
            case Mnemonic::OP_FLOAT_DIV:
                return create_float_div(ctx, function, op);
            case Mnemonic::OP_FLOAT_NEG:
                return create_float_neg(ctx, function, op);
            case Mnemonic::OP_FLOAT_ABS:
                return create_float_abs(ctx, function, op);
            case Mnemonic::OP_FLOAT_SQRT:
                return create_float_sqrt(ctx, function, op);
            case Mnemonic::OP_FLOAT_CEIL:
                return create_float_ceil(ctx, function, op);
            case Mnemonic::OP_FLOAT_FLOOR:
                return create_float_floor(ctx, function, op);
            case Mnemonic::OP_FLOAT_ROUND:
                return create_float_round(ctx, function, op);
            case Mnemonic::OP_FLOAT_NAN:
                return create_float_nan(ctx, function, op);
            case Mnemonic::OP_INT2FLOAT:
                return create_int2float(ctx, function, op);
            case Mnemonic::OP_FLOAT2FLOAT:
                return create_float2float(ctx, function, op);
            case Mnemonic::OP_TRUNC:
                return create_trunc(ctx, function, op);
            case Mnemonic::OP_PTRSUB:
                return create_ptrsub(ctx, function, op);
            case Mnemonic::OP_PTRADD:
                return create_ptradd(ctx, function, op);
            case Mnemonic::OP_CAST:
                return create_cast(ctx, function, op);
            case Mnemonic::OP_DECLARE_LOCAL:
                return create_declare_local(ctx, function, op);
            case Mnemonic::OP_DECLARE_PARAMETER:
                return create_declare_parameter(ctx, function, op);
            case Mnemonic::OP_DECLARE_TEMPORARY:
                return create_declare_temporary(ctx, function, op);
            case Mnemonic::OP_ADDRESS_OF:
                return create_address_of(ctx, function, op);
            case Mnemonic::OP_UNKNOWN:
                assert(false);
                break;
        }

        // Fallback to returning the stmt;
        return std::make_pair(nullptr, true);
    }

    clang::Stmt *
    PcodeASTConsumer::create_call_stmt(clang::ASTContext &ctx, const Operation &op) {
        if (op.mnemonic != Mnemonic::OP_CALL) {
            assert(false);
            return nullptr;
        }

        auto call_target = op.target;
        if (!call_target.has_value()) {
            return nullptr;
        }
        auto function_key = call_target->function;
        auto iter         = function_declarations.find(function_key.value());
        if (iter == function_declarations.end()) {
            return nullptr;
        }
        auto *func_decl = iter->second;
        return create_function_call(ctx, func_decl);
    }

    clang::QualType
    PcodeASTConsumer::get_varnode_type(clang::ASTContext &ctx, const Varnode &vnode) {
        if (!vnode.type_key.empty()) {
            auto iter = type_builder->get_serialized_types().find(vnode.type_key);
            assert(iter != type_builder->get_serialized_types().end());
            return iter->second;
        }

        if (vnode.size != 0U) {
            return get_type_for_size(ctx, vnode.size, /*is_signed=*/false, /*is_integer=*/true);
        }

        return clang::QualType();
    }

    clang::Stmt *PcodeASTConsumer::create_varnode(
        clang::ASTContext &ctx, const Function &function, const Varnode &vnode, bool is_input
    ) {
        switch (vnode.kind) {
            case Varnode::VARNODE_UNKNOWN:
                break;
            case Varnode::VARNODE_GLOBAL:
                return create_global(ctx, function, vnode, is_input);
            case Varnode::VARNODE_PARAM:
                return create_parameter(ctx, function, vnode, is_input);
            case Varnode::VARNODE_FUNCTION:
                return create_function(ctx, function, vnode);
            case Varnode::VARNODE_LOCAL:
                return create_local(ctx, function, vnode, is_input);
            case Varnode::VARNODE_TEMPORARY:
                return create_temporary(ctx, function, vnode, is_input);
            case Varnode::VARNODE_CONSTANT:
                return create_constant(ctx, vnode);
        }

        return nullptr;
    }

    clang::Stmt *PcodeASTConsumer::create_parameter(
        clang::ASTContext &ctx, const Function &function, const Varnode &vnode, bool is_input
    ) {
        if (!vnode.operation || vnode.kind != Varnode::VARNODE_PARAM) {
            assert(false && "Invalid parameter varnode");
            return nullptr;
        }

        auto iter = local_variable_declarations.find(vnode.operation.value());
        assert(
            iter != local_variable_declarations.end()
            && "Failed to find parameter variable declaration"
        );
        auto *param_decl = clang::dyn_cast< clang::ParmVarDecl >(iter->second);
        return clang::DeclRefExpr::Create(
            ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), param_decl, false,
            clang::SourceLocation(), param_decl->getType(),
            is_input ? clang::VK_PRValue : clang::VK_LValue
        );
        (void) function;
    }

    clang::Stmt *PcodeASTConsumer::create_global(
        clang::ASTContext &ctx, const Function &function, const Varnode &vnode, bool is_input
    ) {
        if (!vnode.global || vnode.kind != Varnode::VARNODE_GLOBAL) {
            assert(false && "Invalid global varnode");
            return nullptr;
        }

        auto iter = global_variable_declarations.find(vnode.global.value());
        assert(
            iter != global_variable_declarations.end()
            && "Failed to find global variable declaration"
        );

        auto *var_decl = clang::dyn_cast< clang::VarDecl >(iter->second);
        return clang::DeclRefExpr::Create(
            ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), var_decl, false,
            clang::SourceLocation(), var_decl->getType(),
            is_input ? clang::VK_PRValue : clang::VK_LValue
        );
        (void) function;
    }

    clang::Stmt *PcodeASTConsumer::create_temporary(
        clang::ASTContext &ctx, const Function &function, const Varnode &vnode, bool is_input
    ) {
        if (vnode.kind != Varnode::VARNODE_TEMPORARY) {
            assert(false && "Invalid temporary varnode");
            return nullptr;
        }

        if (!vnode.operation) {
            return nullptr;
        }

        auto var_iter = local_variable_declarations.find(vnode.operation.value());
        if (var_iter != local_variable_declarations.end()) {
            assert(var_iter->second != nullptr);
            return clang::DeclRefExpr::Create(
                ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), var_iter->second,
                false, clang::SourceLocation(), var_iter->second->getType(),
                is_input ? clang::VK_PRValue : clang::VK_LValue
            );
        }

        auto stmt_iter = function_operation_stmts.find(vnode.operation.value());
        if (stmt_iter != function_operation_stmts.end()) {
            assert(stmt_iter->second != nullptr);
            return stmt_iter->second;
        }

        if (auto maybe_operation = operation_from_key(function, vnode.operation.value())) {
            auto [stmt, _] = create_operation(ctx, function, *maybe_operation);
            return stmt;
        }

        assert(false && "Failed to get operation for key");
        return nullptr;
    }

    clang::Stmt *PcodeASTConsumer::create_function(
        clang::ASTContext &ctx, const Function &function, const Varnode &vnode, bool is_input
    ) {
        if (!vnode.function || vnode.kind != Varnode::VARNODE_FUNCTION) {
            assert(false && "Invalid function varnode");
            return nullptr;
        }

        auto iter = function_declarations.find(vnode.function.value());
        assert(iter != function_declarations.end() && "Failed to find function declaration");
        auto *func_decl = clang::dyn_cast< clang::FunctionDecl >(iter->second);
        return clang::DeclRefExpr::Create(
            ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), func_decl, false,
            clang::SourceLocation(), func_decl->getType(), clang::VK_PRValue
        );
        (void) function, is_input;
    }

    clang::Stmt *PcodeASTConsumer::create_local(
        clang::ASTContext &ctx, const Function &function, const Varnode &vnode, bool is_input
    ) {
        if (!vnode.operation || vnode.kind != Varnode::VARNODE_LOCAL) {
            assert(false && "Invalid local varnode");
            return nullptr;
        }

        auto iter = local_variable_declarations.find(vnode.operation.value());
        if (iter != local_variable_declarations.end()) {
            assert(iter != local_variable_declarations.end());
            auto *var_decl = clang::dyn_cast< clang::VarDecl >(iter->second);
            return clang::DeclRefExpr::Create(
                ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), var_decl, false,
                clang::SourceLocation(), var_decl->getType(),
                is_input ? clang::VK_PRValue : clang::VK_LValue
            );
        }

        auto maybe_operation = operation_from_key(function, vnode.operation.value());
        if (maybe_operation) {
            auto [stmt, _] = create_operation(ctx, function, *maybe_operation);
            return stmt;
        }

        return nullptr;
    }

    clang::Stmt *
    PcodeASTConsumer::create_constant(clang::ASTContext &ctx, const Varnode &vnode) {
        if (vnode.kind != Varnode::VARNODE_CONSTANT) {
            assert(false && "Invalid constant varnode");
            return nullptr;
        }

        clang::QualType type = get_varnode_type(ctx, vnode);

        if (type->isIntegerType()) {
            auto value = vnode.value;
            return new (ctx) clang::IntegerLiteral(
                ctx, llvm::APInt(static_cast< uint32_t >(ctx.getTypeSize(type)), *value), type,
                clang::SourceLocation()
            );
        }

        if (type->isVoidType()) {
            auto value    = vnode.value;
            auto *literal = new (ctx) clang::IntegerLiteral(
                ctx, llvm::APInt(32U, *value), ctx.IntTy, clang::SourceLocation()
            );
            return clang::CStyleCastExpr::Create(
                ctx, type, clang::VK_PRValue, clang::CK_ToVoid, literal, nullptr,
                clang::FPOptionsOverride(), ctx.getTrivialTypeSourceInfo(type),
                clang::SourceLocation(), clang::SourceLocation()
            );
        }

        if (type->isPointerType()) {
            auto value    = vnode.value;
            auto *literal = new (ctx) clang::IntegerLiteral(
                ctx, llvm::APInt(32U, *value), ctx.IntTy, clang::SourceLocation()
            );
            return clang::CStyleCastExpr::Create(
                ctx, type, clang::VK_PRValue, clang::CK_IntegralToPointer, literal, nullptr,
                clang::FPOptionsOverride(), ctx.getTrivialTypeSourceInfo(type),
                clang::SourceLocation(), clang::SourceLocation()
            );
        }

        if (type->isFloatingType()) {
            auto value = static_cast< double >(*vnode.value);
            return clang::FloatingLiteral::Create(
                ctx, llvm::APFloat(value), true, type, clang::SourceLocation()
            );
        }

        return nullptr;
    }

} // namespace patchestry::ast
