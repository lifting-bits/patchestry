/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <functional>

#include <clang/AST/ASTContext.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Type.h>
#include <clang/Basic/SourceLocation.h>

#include <patchestry/AST/FunctionBuilder.hpp>
#include <patchestry/AST/TypeBuilder.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>

namespace patchestry::ast {
    using namespace patchestry::ghidra;

    class OpBuilder
    {
      public:
        OpBuilder(
            clang::ASTContext &ctx, const std::shared_ptr< FunctionBuilder > &func_builder
        )
            : context(ctx), builder(func_builder) {}

        OpBuilder(const OpBuilder &)            = default;
        OpBuilder &operator=(const OpBuilder &) = default;

        OpBuilder(const OpBuilder &&)            = delete;
        OpBuilder &operator=(const OpBuilder &&) = delete;

        virtual ~OpBuilder() = default;

        std::pair< clang::Stmt *, bool >
        create_copy(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_load(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_store(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_branch(clang::ASTContext &ctx, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_cbranch(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_branchind(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_call(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_callind(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool > create_userdefined(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool >
        create_return(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_piece(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_subpiece(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_int_zext(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_int_carry(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool > create_int_scarry(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool > create_int_sborrow(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool >
        create_int_2comp(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_int_sext(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool > create_unary_operation(
            clang::ASTContext &ctx, const Function &function, const Operation &op,
            clang::UnaryOperatorKind kind
        );

        std::pair< clang::Stmt *, bool > create_binary_operation(
            clang::ASTContext &ctx, const Function &function, const Operation &op,
            clang::BinaryOperatorKind kind
        );

        std::pair< clang::Stmt *, bool >
        create_float_abs(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool > create_float_sqrt(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool > create_float_floor(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool > create_float_ceil(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool > create_float_round(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool >
        create_int2float(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_float_nan(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool > create_float2float(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool >
        create_trunc(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_ptrsub(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_ptradd(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_cast(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_declare_local(clang::ASTContext &ctx, const Operation &op);

        std::pair< clang::Stmt *, bool > create_declare_parameter(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

      private:
        clang::Expr *build_callexpr_from_function(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        clang::Stmt *create_assign_operation(
            clang::ASTContext &ctx, clang::Expr *input_expr, clang::Expr *output_expr,
            clang::SourceLocation loc = clang::SourceLocation()
        );

        /**
         * @brief Performs an implicit and explicit cast of an expression to a specified type,
         * falling back to a manual pointer-based cast if necessary.
         *
         * @param ctx Reference to clang ASTContext.
         * @param expr The input expression to be cast.
         * @param to_type The target type to which the expression should be cast.
         *
         * @return Pointer to casted `Expr`. null if `to_type` is null, or an invalid cast
         * occurs.
         */
        clang::Expr *make_cast(
            clang::ASTContext &ctx, clang::Expr *expr, clang::QualType to_type,
            clang::SourceLocation loc
        );

        clang::Expr *make_explicit_cast(
            clang::ASTContext &ctx, clang::Expr *expr, clang::QualType to_type,
            clang::SourceLocation loc
        );

        clang::Expr *make_implicit_cast(
            clang::ASTContext &ctx, clang::Expr *expr, clang::QualType to_type,
            clang::CastKind kind
        );

        clang::Stmt *create_varnode(
            clang::ASTContext &ctx, const Function &function, const Varnode &vnode,
            clang::SourceLocation = clang::SourceLocation()
        );

        void extend_callexpr_agruments(
            clang::ASTContext &ctx, clang::FunctionDecl *fndecl,
            std::vector< clang::Expr * > &arguments
        );

        clang::Stmt *create_parameter(clang::ASTContext &ctx, const Varnode &vnode);

        clang::Stmt *create_global(clang::ASTContext &ctx, const Varnode &vnode);

        clang::Stmt *create_temporary(
            clang::ASTContext &ctx, const Function &function, const Varnode &vnode
        );

        clang::Stmt *create_function(clang::ASTContext &ctx, const Varnode &vnode);

        clang::Stmt *
        create_local(clang::ASTContext &ctx, const Function &function, const Varnode &vnode);

        clang::Stmt *create_constant(clang::ASTContext &ctx, const Varnode &vnode);

        clang::QualType get_varnode_type(clang::ASTContext &ctx, const Varnode &vnode);

        TypeBuilder &type_builder(void) { return builder->type_builder.get(); }

        FunctionBuilder &function_builder(void) { return *builder; }

        clang::Sema &sema(void) { return function_builder().sema(); }

        std::reference_wrapper< const clang::ASTContext > context;
        std::shared_ptr< FunctionBuilder > builder;
    };

} // namespace patchestry::ast
