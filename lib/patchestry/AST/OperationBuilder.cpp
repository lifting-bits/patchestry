/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <optional>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/NestedNameSpecifier.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <clang/Basic/LLVM.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/Specifiers.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/Support/Casting.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/AST/OperationBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {

    extern clang::QualType
    getTypeFromSize(clang::ASTContext &ctx, unsigned bit_size, bool is_signed, bool is_integer);

    std::optional< Operation >
    operationFromKey(const Function &function, const std::string &lookup_key) {
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

    clang::Stmt *OpBuilder::create_varnode(
        clang::ASTContext &ctx, const Function &function, const Varnode &vnode,
        const std::string &op_key
    ) {
        auto varnode_operation = [&](clang::ASTContext &ctx, const Function &function,
                                     const Varnode &vnode) -> clang::Stmt * {
            switch (vnode.kind) {
                case Varnode::VARNODE_UNKNOWN:
                    break;
                case Varnode::VARNODE_GLOBAL:
                    return create_global(ctx, vnode);
                case Varnode::VARNODE_PARAM:
                    return create_parameter(ctx, vnode);
                case Varnode::VARNODE_FUNCTION:
                    return create_function(ctx, vnode);
                case Varnode::VARNODE_LOCAL:
                    return create_local(ctx, function, vnode);
                case Varnode::VARNODE_TEMPORARY:
                    return create_temporary(ctx, function, vnode);
                case Varnode::VARNODE_CONSTANT:
                    return create_constant(ctx, vnode);
            }

            return nullptr;
        };

        if (auto *expr = varnode_operation(ctx, function, vnode)) {
            return expr;
        }

        (void) op_key;

        return {};
    }

    clang::Stmt *OpBuilder::create_parameter(clang::ASTContext &ctx, const Varnode &vnode) {
        if (!vnode.operation || vnode.kind != Varnode::VARNODE_PARAM) {
            assert(false && "Invalid parameter varnode");
            return nullptr;
        }

        if (!function_builder().local_variables.contains(*vnode.operation)) {
            return {};
        }

        auto *param_decl = function_builder().local_variables.at(*vnode.operation);
        return clang::DeclRefExpr::Create(
            ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), param_decl, false,
            clang::SourceLocation(), param_decl->getType(), clang::VK_LValue
        );
    }

    clang::Stmt *OpBuilder::create_global(clang::ASTContext &ctx, const Varnode &vnode) {
        if (!vnode.global || vnode.kind != Varnode::VARNODE_GLOBAL) {
            assert(false && "Invalid global varnode");
            return {};
        }

        if (!function_builder().global_var_list.get().contains(*vnode.global)) {
            return {};
        }

        auto *var_decl = function_builder().global_var_list.get().at(*vnode.global);
        return clang::DeclRefExpr::Create(
            ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), var_decl, false,
            clang::SourceLocation(), var_decl->getType(), clang::VK_LValue
        );
    }

    clang::Stmt *OpBuilder::create_temporary(
        clang::ASTContext &ctx, const Function &function, const Varnode &vnode
    ) {
        if (!vnode.operation || vnode.kind != Varnode::VARNODE_TEMPORARY) {
            LOG(ERROR) << "Invalid temporary varnode or operation.\n";
            return {};
        }

        if (function_builder().local_variables.contains(*vnode.operation)) {
            auto *var_decl = function_builder().local_variables.at(*vnode.operation);
            return clang::DeclRefExpr::Create(
                ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), var_decl, false,
                clang::SourceLocation(), var_decl->getType(), clang::VK_LValue
            );
        }

        if (function_builder().operation_stmts.contains(*vnode.operation)) {
            return function_builder().operation_stmts.at(*vnode.operation);
        }

        if (auto maybe_operation = operationFromKey(function, vnode.operation.value())) {
            auto [stmt, _] = function_builder().create_operation(ctx, *maybe_operation);
            return stmt;
        }

        assert(false && "Failed to get operation for key");
        return {};
    }

    clang::Stmt *OpBuilder::create_function(clang::ASTContext &ctx, const Varnode &vnode) {
        if (!vnode.function || vnode.kind != Varnode::VARNODE_FUNCTION) {
            LOG(ERROR) << "Invalid varnode or local operation.\n";
            return {};
        }

        if (function_builder().function_list.get().contains(*vnode.function)) {
            auto *function_decl = function_builder().function_list.get().at(*vnode.function);
            auto location       = clang::SourceLocation();
            auto *function_ref  = clang::DeclRefExpr::Create(
                ctx, clang::NestedNameSpecifierLoc(), location, function_decl, false, location,
                function_decl->getType(), clang::VK_PRValue
            );
            return function_ref;
        }

        return {};
    }

    clang::Stmt *OpBuilder::create_local(
        clang::ASTContext &ctx, const Function &function, const Varnode &vnode
    ) {
        if (!vnode.operation || vnode.kind != Varnode::VARNODE_LOCAL) {
            LOG(ERROR) << "Invalid varnode or local operation.\n";
            return {};
        }

        if (function_builder().local_variables.contains(*vnode.operation)) {
            auto *var_decl = function_builder().local_variables.at(*vnode.operation);
            return clang::DeclRefExpr::Create(
                ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), var_decl, false,
                clang::SourceLocation(), var_decl->getType(), clang::VK_LValue
            );
        }

        if (auto maybe_operation = operationFromKey(function, vnode.operation.value())) {
            auto [stmt, _] = function_builder().create_operation(ctx, *maybe_operation);
            return stmt;
        }

        return {};
    }

    clang::Stmt *OpBuilder::create_constant(clang::ASTContext &ctx, const Varnode &vnode) {
        if (vnode.kind != Varnode::VARNODE_CONSTANT) {
            LOG(ERROR) << "Varnode is not constant, invalid varnode.\n";
            return {};
        }

        clang::QualType vnode_type = get_varnode_type(ctx, vnode);
        auto location              = sourceLocation(ctx.getSourceManager(), vnode.type_key);

        // Note: EnumDecl has promotional type as int and an enum type is also identified
        // as integer.
        if (vnode_type->isIntegralOrUnscopedEnumerationType()) {
            auto *literal = new (ctx)
                clang::IntegerLiteral(ctx, llvm::APInt(32U, *vnode.value), ctx.IntTy, location);

            auto result = sema().BuildCStyleCastExpr(
                location, ctx.getTrivialTypeSourceInfo(vnode_type), location, literal
            );

            assert(!result.isInvalid());
            return result.getAs< clang::Expr >();
        }

        if (vnode_type->isVoidType()) {
            auto *literal = new (ctx)
                clang::IntegerLiteral(ctx, llvm::APInt(32U, *vnode.value), ctx.IntTy, location);

            auto result = sema().BuildCStyleCastExpr(
                location, ctx.getTrivialTypeSourceInfo(vnode_type), location, literal
            );

            assert(!result.isInvalid());
            return result.getAs< clang::Expr >();
        }

        if (vnode_type->isPointerType()) {
            auto *literal = new (ctx)
                clang::IntegerLiteral(ctx, llvm::APInt(32U, *vnode.value), ctx.IntTy, location);

            auto result = sema().BuildCStyleCastExpr(
                location, ctx.getTrivialTypeSourceInfo(vnode_type), location, literal
            );

            assert(!result.isInvalid());
            return result.getAs< clang::Expr >();
        }

        if (vnode_type->isFloatingType()) {
            return clang::FloatingLiteral::Create(
                ctx, llvm::APFloat(static_cast< double >(*vnode.value)), true, vnode_type,
                location
            );
        }

        return {};
    }

    /**
     * @brief Retrieves the Clang `QualType` for a `Varnode` based on its `type_key` or
     * size.
     *
     * If the `Varnode` has a valid `type_key`, the type is fetched from `type_builder`.
     * Otherwise, a fallback type is derived using the `size` of the `Varnode` as an
     * unsigned integer.
     *
     * @param ctx Reference to the `clang::ASTContext`.
     * @param vnode The `Varnode` containing `type_key` and `size` information.
     * @return The resolved `clang::QualType`, or an empty type if `type_key` and `size` are
     * invalid.
     *
     * @note Logs an error if both `type_key` is empty and `size` is 0.
     */
    clang::QualType OpBuilder::get_varnode_type(clang::ASTContext &ctx, const Varnode &vnode) {
        if (vnode.type_key.empty() && vnode.size == 0U) {
            LOG(ERROR) << "Varnode with empty or invalid type key.\n";
            return {};
        }

        if (type_builder().get_serialized_types().contains(vnode.type_key)) {
            return type_builder().get_serialized_types().at(vnode.type_key);
        }

        return getTypeFromSize(ctx, vnode.size, /*is_signed=*/false, /*is_integer=*/true);
    }

} // namespace patchestry::ast
