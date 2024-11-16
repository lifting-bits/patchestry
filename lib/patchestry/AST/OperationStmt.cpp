/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <optional>
#include <utility>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/NestedNameSpecifier.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <clang/Basic/LLVM.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/Specifiers.h>
#include <clang/Sema/Sema.h>
#include <llvm/ADT/APInt.h>
#include <llvm/Support/Casting.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>

namespace patchestry::ast {

    extern std::optional< Operation >
    operation_from_key(const Function &function, const std::string &lookup_key);

    namespace {
        clang::CallExpr *create_function_call(
            clang::ASTContext &ctx, clang::FunctionDecl *decl,
            std::vector< clang::Expr * > &args
        ) {
            auto *ref_expr = clang::DeclRefExpr::Create(
                ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), decl, false,
                clang::SourceLocation(), decl->getType(), clang::VK_LValue
            );

            return clang::CallExpr::Create(
                ctx, ref_expr, args, decl->getReturnType(), clang::VK_PRValue,
                clang::SourceLocation(), clang::FPOptionsOverride()
            );
        }

        clang::VarDecl *create_variable_decl(
            clang::ASTContext &ctx, clang::DeclContext *dc, const std::string &name,
            clang::QualType type, clang::SourceLocation loc
        ) {
            return clang::VarDecl::Create(
                ctx, dc, loc, loc, &ctx.Idents.get(name), type,
                ctx.getTrivialTypeSourceInfo(type), clang::SC_None
            );
        }
    } // namespace

    clang::DeclStmt *
    PcodeASTConsumer::create_decl_stmt(clang::ASTContext &ctx, clang::Decl *decl) {
        auto decl_group = clang::DeclGroupRef(decl);
        return new (ctx)
            clang::DeclStmt(decl_group, clang::SourceLocation(), clang::SourceLocation());
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_declare_local(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_DECLARE_LOCAL) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        // Get type of the declared variable
        auto type_iter = type_builder->get_serialized_types().find(*op.type);
        assert(
            (type_iter != type_builder->get_serialized_types().end())
            && "Failed to find type for declared variable."
        );

        auto *var_decl = create_variable_decl(
            ctx, get_sema().CurContext, *op.name, type_iter->second,
            source_location_from_key(ctx, op.key)
        );

        // add variable declaration to list for future references
        local_variable_declarations.emplace(op.key, var_decl);
        (void) function;
        return std::make_pair(create_decl_stmt(ctx, var_decl), false);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_declare_temporary(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_DECLARE_TEMPORARY) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        // Get type of the declared variable
        auto type_iter = type_builder->get_serialized_types().find(*op.type);
        assert(
            (type_iter != type_builder->get_serialized_types().end())
            && "Failed to find type for declared variable."
        );

        auto *var_decl = create_variable_decl(
            ctx, get_sema().CurContext, *op.name, type_iter->second,
            source_location_from_key(ctx, op.key)
        );

        // add variable declaration to list for future references
        local_variable_declarations.emplace(op.key, var_decl);
        return std::make_pair(create_decl_stmt(ctx, var_decl), false);
        (void) function;
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_declare_parameter(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_copy(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_COPY) {
            assert(false && "Invalid copy operation");
            return std::make_pair(nullptr, false);
        }

        // Copy operation does not have output varnode. Create stmt that will be merged to next
        // operation
        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs.front()));
        if (!op.output) {
            assert((op.inputs.size() == 1) && "Invalid input for copy operation");
            return std::make_pair(input_expr, true);
        }

        auto *output_expr = clang::dyn_cast< clang::Expr >(
            create_varnode(ctx, function, *op.output, /*is_input=*/false)
        );

        if (clang::dyn_cast< clang::Expr >(input_expr)->getType() != output_expr->getType()) {
            auto cast_result = get_sema().BuildCStyleCastExpr(
                clang::SourceLocation(), ctx.getTrivialTypeSourceInfo(output_expr->getType()),
                clang::SourceLocation(), clang::dyn_cast< clang::Expr >(input_expr)
            );

            assert(!cast_result.isInvalid() && "Invalid cstyle cast to output expr");
            input_expr = cast_result.getAs< clang::Expr >();
        }

        auto assign_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign, output_expr, input_expr
        );
        assert(!assign_result.isInvalid());

        return std::make_pair(assign_result.getAs< clang::Stmt >(), false);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_load(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_LOAD) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        auto merge_to_next = !op.output.has_value();

        auto *input0_expr = create_varnode(ctx, function, op.inputs[0]);
        assert(input0_expr != nullptr);

        auto deref_result = get_sema().CreateBuiltinUnaryOp(
            clang::SourceLocation(), clang::UO_Deref,
            clang::dyn_cast< clang::Expr >(input0_expr)
        );

        if (merge_to_next) {
            return std::make_pair(deref_result.getAs< clang::Expr >(), merge_to_next);
        }

        auto *result_expr = deref_result.getAs< clang::Expr >();
        // auto is_lvalue    = result_expr->isLValue();

        auto *output_expr = clang::dyn_cast< clang::Expr >(
            create_varnode(ctx, function, *op.output, /*is_input=*/false)
        );

        // auto *result_expr = deref_result.getAs< clang::Expr >();

        if (result_expr->getType() != output_expr->getType()) {
            auto cast_result = get_sema().BuildCStyleCastExpr(
                clang::SourceLocation(), ctx.getTrivialTypeSourceInfo(output_expr->getType()),
                clang::SourceLocation(), result_expr
            );

            assert(!cast_result.isInvalid() && "Invalid cstyle cast to output expr");
            result_expr = cast_result.getAs< clang::Expr >();
        }

        auto assign_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign, output_expr, result_expr
        );
        assert(!assign_result.isInvalid());

        return std::make_pair(assign_result.getAs< clang::Stmt >(), false);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_store(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_STORE) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        assert(op.inputs.size() >= 2);

        auto *input0_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        auto *input1_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));

        if (op.inputs.size() == 2) {
            auto deref_result = get_sema().CreateBuiltinUnaryOp(
                clang::SourceLocation(), clang::UO_Deref,
                clang::dyn_cast< clang::Expr >(input0_expr)
            );

            auto *result_expr = deref_result.getAs< clang::Expr >();

            if (result_expr->getType() != input1_expr->getType()) {
                auto cast_result = get_sema().BuildCStyleCastExpr(
                    clang::SourceLocation(),
                    ctx.getTrivialTypeSourceInfo(result_expr->getType()),
                    clang::SourceLocation(), input1_expr
                );

                assert(!cast_result.isInvalid() && "Invalid cstyle cast to output expr");
                input1_expr = cast_result.getAs< clang::Expr >();
            }

            auto store_result = get_sema().CreateBuiltinBinOp(
                source_location_from_key(ctx, op.key), clang::BO_Assign, result_expr,
                input1_expr
            );

            return std::make_pair(store_result.getAs< clang::Expr >(), false);
        }

        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_branch(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_BRANCH) {
            assert(false && "Invalid branch operation.");
            return std::make_pair(nullptr, false);
        }

        assert(op.target_block);
        auto iter = basic_block_labels.find(*op.target_block);
        assert(iter != basic_block_labels.end());

        (void) function;
        // Create GotoStmt for branch operation
        return std::make_pair(
            new (ctx) clang::GotoStmt(
                iter->second, source_location_from_key(ctx, op.key),
                source_location_from_key(ctx, *op.target_block)
            ),
            false
        );
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_cbranch(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_CBRANCH) {
            assert(false && "Invalid cbranch operation");
            return std::make_pair(nullptr, false);
        }

        // TODO(kumarak): Could there be case where conditional statement is missing?? In
        // such case treat it as branch instruction.
        auto *condition_expr        = create_varnode(ctx, function, *op.condition);
        clang::Stmt *taken_stmt     = nullptr;
        clang::Stmt *not_taken_stmt = nullptr;

        if (op.taken_block && !op.taken_block->empty()) {
            auto taken_block_key = *op.taken_block;
            auto label_iter      = basic_block_labels.find(taken_block_key);
            assert(label_iter != basic_block_labels.end());

            taken_stmt = new (ctx) clang::GotoStmt(
                label_iter->second, source_location_from_key(ctx, op.key),
                source_location_from_key(ctx, *op.target_block)
            );
        } else {
            taken_stmt = new (ctx) clang::NullStmt(clang::SourceLocation(), false);
        }

        if (op.not_taken_block && !op.not_taken_block->empty()) {
            auto not_taken_block_key = *op.not_taken_block;
            auto label_iter          = basic_block_labels.find(not_taken_block_key);
            assert(label_iter != basic_block_labels.end());

            not_taken_stmt = new (ctx) clang::GotoStmt(
                label_iter->second, source_location_from_key(ctx, op.key),
                source_location_from_key(ctx, *op.target_block)
            );
        } else {
            not_taken_stmt = new (ctx) clang::NullStmt(clang::SourceLocation(), false);
        }

        return std::make_pair(
            clang::IfStmt::Create(
                ctx, clang::SourceLocation(), clang::IfStatementKind::Ordinary, nullptr,
                nullptr, clang::dyn_cast< clang::Expr >(condition_expr),
                clang::SourceLocation(), clang::SourceLocation(), taken_stmt,
                clang::SourceLocation(), not_taken_stmt
            ),
            false
        );
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_branchind(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_call(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_CALL) {
            assert(false && "Invalid call operation.");
            return std::make_pair(nullptr, false);
        }

        auto call_target = op.target;
        if (!call_target.has_value()) {
            return std::make_pair(nullptr, false);
        }

        if (!call_target->function && !call_target->operation) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        clang::Expr *call_expr = nullptr;

        std::vector< clang::Expr * > arguments;
        for (const auto &input : op.inputs) {
            auto *arg_expr = create_varnode(ctx, function, input);
            assert(arg_expr != nullptr);
            arguments.push_back(clang::dyn_cast< clang::Expr >(arg_expr));
        }

        if (call_target->function) {
            auto iter = function_declarations.find(call_target->function.value());
            if (iter == function_declarations.end()) {
                return std::make_pair(nullptr, false);
            }
            call_expr = create_function_call(ctx, iter->second, arguments);
            if (!op.output || iter->second->getReturnType()->isVoidType()) {
                return std::make_pair(clang::dyn_cast< clang::Expr >(call_expr), false);
            }

        } else if (call_target->operation) {
            auto op        = operation_from_key(function, call_target->operation.value());
            auto [stmt, _] = create_operation(ctx, function, op.value());
            auto result    = get_sema().ActOnCallExpr(
                nullptr, clang::dyn_cast< clang::Expr >(stmt), clang::SourceLocation(),
                arguments, clang::SourceLocation()
            );
            call_expr = result.getAs< clang::Expr >();
            if (!op->output) {
                return std::make_pair(clang::dyn_cast< clang::Expr >(call_expr), false);
            }
        }

        auto *out_expr = create_varnode(ctx, function, *op.output, false);
        set_sema_context(ctx.getTranslationUnitDecl());

        auto rty_type = type_builder->get_serialized_types().at(*op.type);

        auto cast_result =
            get_sema().ImpCastExprToType(call_expr, rty_type, clang::CastKind::CK_BitCast);

        auto out_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign,
            clang::dyn_cast< clang::Expr >(out_expr), cast_result.getAs< clang::Expr >()
        );

        return std::make_pair(out_result.getAs< clang::Expr >(), false);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_callind(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_CALLIND) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        auto call_target = op.target;
        if (!call_target.has_value()) {
            return std::make_pair(nullptr, false);
        }
        clang::Expr *call_expr = nullptr;

        std::vector< clang::Expr * > arguments;
        for (const auto &input : op.inputs) {
            auto *arg_expr = create_varnode(ctx, function, input);
            assert(arg_expr != nullptr);
            arguments.push_back(clang::dyn_cast< clang::Expr >(arg_expr));
        }

        if (call_target->function) {
            auto iter = function_declarations.find(call_target->function.value());
            if (iter == function_declarations.end()) {
                return std::make_pair(nullptr, false);
            }
            call_expr = create_function_call(ctx, iter->second, arguments);
            if (!op.output || iter->second->getReturnType()->isVoidType()) {
                return std::make_pair(clang::dyn_cast< clang::Expr >(call_expr), false);
            }
        } else if (call_target->operation) {
            auto op        = operation_from_key(function, call_target->operation.value());
            auto [stmt, _] = create_operation(ctx, function, op.value());
            auto result    = get_sema().ActOnCallExpr(
                nullptr, clang::dyn_cast< clang::Expr >(stmt), clang::SourceLocation(),
                arguments, clang::SourceLocation()
            );
            call_expr = result.getAs< clang::Expr >();
            if (!op->output) {
                return std::make_pair(clang::dyn_cast< clang::Expr >(call_expr), false);
            }
        }

        auto *out_expr = create_varnode(ctx, function, *op.output, /*is_input=*/false);
        auto rty_type  = type_builder->get_serialized_types().at(*op.type);

        auto cast_result =
            get_sema().ImpCastExprToType(call_expr, rty_type, clang::CastKind::CK_BitCast);

        auto out_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign,
            clang::dyn_cast< clang::Expr >(out_expr), cast_result.getAs< clang::Expr >()
        );

        return std::make_pair(out_result.getAs< clang::Expr >(), false);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_userdefined(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_return(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_RETURN) {
            assert(false && "Invalid return operation");
            return std::make_pair(nullptr, false);
        }

        // Assert if number
        // assert(ret_op.inputs.size() < 2);
        if (!op.inputs.empty()) {
            auto varnode   = op.inputs.size() == 1 ? op.inputs.front() : op.inputs.at(1);
            auto *ret_expr = create_varnode(ctx, function, varnode);
            return std::make_pair(
                clang::ReturnStmt::Create(
                    ctx, clang::SourceLocation(), llvm::dyn_cast< clang::Expr >(ret_expr),
                    nullptr
                ),
                false
            );
        }
        return std::make_pair(
            clang::ReturnStmt::Create(ctx, clang::SourceLocation(), nullptr, nullptr), false
        );
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_piece(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_PIECE) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        auto merge_to_next = !op.output.has_value();

        unsigned low_width = 8U;

        auto *shift_value = clang::IntegerLiteral::Create(
            ctx, llvm::APInt(32, low_width), ctx.IntTy, clang::SourceLocation()
        );

        auto *input0_expr = create_varnode(ctx, function, op.inputs[0]);
        assert(input0_expr != nullptr);

        auto *input1_expr = create_varnode(ctx, function, op.inputs[1]);
        assert(input1_expr != nullptr);

        auto shifted_high_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Shl,
            clang::dyn_cast< clang::Expr >(input0_expr),
            clang::dyn_cast< clang::Expr >(shift_value)
        );

        if (shifted_high_result.isInvalid()) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        auto or_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Or,
            shifted_high_result.getAs< clang::Expr >(),
            clang::dyn_cast< clang::Expr >(input1_expr)
        );

        if (or_result.isInvalid()) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        if (merge_to_next) {
            return std::make_pair(or_result.getAs< clang::Expr >(), merge_to_next);
        }

        auto *output_expr = create_varnode(ctx, function, *op.output, /*is_input=*/false);
        auto out_result   = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign,
            clang::dyn_cast< clang::Expr >(output_expr), or_result.getAs< clang::Expr >()
        );

        if (out_result.isInvalid()) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        return std::make_pair(out_result.getAs< clang::Stmt >(), merge_to_next);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_subpiece(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_SUBPIECE) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        auto merge_to_next = !op.output.has_value();
        assert(op.inputs.size() == 2);

        auto *shift_value = create_varnode(ctx, function, op.inputs[1]);
        assert(shift_value != nullptr);

        auto *expr = create_varnode(ctx, function, op.inputs[0]);
        assert(expr != nullptr);

        auto *expr_with_paren = new (ctx) clang::ParenExpr(
            clang::SourceLocation(), clang::SourceLocation(),
            clang::dyn_cast< clang::Expr >(expr)
        );

        auto shifted_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Shr,
            clang::dyn_cast< clang::Expr >(expr_with_paren),
            clang::dyn_cast< clang::Expr >(shift_value)
        );

        if (shifted_result.isInvalid()) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        auto *shifted_expr = new (ctx) clang::ParenExpr(
            clang::SourceLocation(), clang::SourceLocation(),
            shifted_result.getAs< clang::Expr >()
        );

        auto mask_value = llvm::APInt::getAllOnes(32);
        auto *mask =
            clang::IntegerLiteral::Create(ctx, mask_value, ctx.IntTy, clang::SourceLocation());

        auto result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_And, shifted_expr,
            clang::dyn_cast< clang::Expr >(mask)
        );

        if (result.isInvalid()) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        auto *result_expr = new (ctx) clang::ParenExpr(
            clang::SourceLocation(), clang::SourceLocation(), result.getAs< clang::Expr >()
        );

        if (merge_to_next) {
            return std::make_pair(result_expr, merge_to_next);
        }

        auto *out_expr  = create_varnode(ctx, function, *op.output, /*is_input=*/false);
        auto out_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign,
            clang::dyn_cast< clang::Expr >(out_expr), result_expr
        );

        if (out_result.isInvalid()) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        return std::make_pair(out_result.getAs< clang::Stmt >(), merge_to_next);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_equal(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_EQUAL) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        return create_binary_operation< clang::BinaryOperatorKind::BO_EQ >(ctx, function, op);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_notequal(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_NOTEQUAL) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        return create_binary_operation< clang::BinaryOperatorKind::BO_NE >(ctx, function, op);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_less(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_LESS) {
            assert(false && "Invalid int_less operation");
            return std::make_pair(nullptr, false);
        }

        return create_binary_operation< clang::BO_LT >(ctx, function, op);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_sless(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_SLESS) {
            assert(false && "Invalid int_sless operation");
            return std::make_pair(nullptr, false);
        }

        return create_binary_operation< clang::BO_LT >(ctx, function, op);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_lessequal(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_LESSEQUAL) {
            assert(false && "Invalid int_lessequal operation");
            return std::make_pair(nullptr, false);
        }

        return create_binary_operation< clang::BO_LE >(ctx, function, op);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_slessequal(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_SLESSEQUAL) {
            assert(false && "Invalid int_slessequal operation");
            return std::make_pair(nullptr, false);
        }

        return create_binary_operation< clang::BO_LE >(ctx, function, op);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_zext(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_ZEXT) {
            assert(false);
            return std::make_pair(nullptr, true);
        }

        auto merge_to_next = !op.output.has_value();
        auto *input_expr   = create_varnode(ctx, function, op.inputs[0]);
        assert(input_expr != nullptr);

        auto target_type = type_builder->get_serialized_types().at(*op.type);

        auto result = get_sema().BuildCStyleCastExpr(
            clang::SourceLocation(), ctx.getTrivialTypeSourceInfo(target_type),
            clang::SourceLocation(), clang::dyn_cast< clang::Expr >(input_expr)
        );

        if (result.isInvalid()) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        if (merge_to_next) {
            return std::make_pair(result.getAs< clang::Stmt >(), merge_to_next);
        }

        auto *out_expr  = create_varnode(ctx, function, *op.output, /*is_input=*/false);
        auto out_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign,
            clang::dyn_cast< clang::Expr >(out_expr), result.getAs< clang::Expr >()
        );

        return std::make_pair(out_result.getAs< clang::Stmt >(), merge_to_next);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_sext(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_SEXT) {
            assert(false);
            return std::make_pair(nullptr, true);
        }

        auto merge_to_next = !op.output.has_value();
        auto *input_expr   = create_varnode(ctx, function, op.inputs[0]);
        assert(input_expr != nullptr);

        auto target_type = type_builder->get_serialized_types().at(*op.type);

        auto result = get_sema().BuildCStyleCastExpr(
            clang::SourceLocation(), ctx.getTrivialTypeSourceInfo(target_type),
            clang::SourceLocation(), clang::dyn_cast< clang::Expr >(input_expr)
        );

        if (result.isInvalid()) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        if (merge_to_next) {
            return std::make_pair(result.getAs< clang::Stmt >(), merge_to_next);
        }

        auto *out_expr  = create_varnode(ctx, function, *op.output, /*is_input=*/false);
        auto out_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign,
            clang::dyn_cast< clang::Expr >(out_expr), result.getAs< clang::Expr >()
        );

        return std::make_pair(out_result.getAs< clang::Stmt >(), merge_to_next);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_add(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_ADD) {
            assert(false && "Invalid int_add operation");
            return std::make_pair(nullptr, false);
        }

        return create_binary_operation< clang::BO_Add >(ctx, function, op);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_sub(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_SUB) {
            assert(false && "Invalid int_add operation");
            return std::make_pair(nullptr, false);
        }

        return create_binary_operation< clang::BO_Sub >(ctx, function, op);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_carry(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_scarry(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_sborrow(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_2comp(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    template std::pair< clang::Stmt *, bool >
    PcodeASTConsumer::create_unary_operation< clang::UO_LNot >(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    );

    template std::pair< clang::Stmt *, bool >
    PcodeASTConsumer::create_binary_operation< clang::BO_Xor >(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    );

    template std::pair< clang::Stmt *, bool >
    PcodeASTConsumer::create_binary_operation< clang::BO_And >(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    );

    template std::pair< clang::Stmt *, bool >
    PcodeASTConsumer::create_binary_operation< clang::BO_Or >(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    );

    template std::pair< clang::Stmt *, bool >
    PcodeASTConsumer::create_binary_operation< clang::BO_Shl >(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    );

    template std::pair< clang::Stmt *, bool >
    PcodeASTConsumer::create_binary_operation< clang::BO_Shr >(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    );

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_mult(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_MULT) {
            assert(false && "Invalid int_add operation");
            return std::make_pair(nullptr, false);
        }

        return create_binary_operation< clang::BO_Mul >(ctx, function, op);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_div(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_DIV) {
            assert(false && "Invalid int_add operation");
            return std::make_pair(nullptr, false);
        }

        return create_binary_operation< clang::BO_Div >(ctx, function, op);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_rem(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_REM) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        return create_binary_operation< clang::BO_Rem >(ctx, function, op);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int_sdiv(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT_SDIV) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        return create_binary_operation< clang::BO_Div >(ctx, function, op);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_equal(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_notequal(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_less(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_lessequal(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_add(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_sub(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_mult(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_div(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_FLOAT_DIV) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        return create_binary_operation< clang::BO_Div >(ctx, function, op);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_neg(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_abs(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_sqrt(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_ceil(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_floor(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_round(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float_nan(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_int2float(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_INT2FLOAT) {
            assert(false && "Invalid int2float operation");
            return std::make_pair(nullptr, false);
        }

        auto type_iter = type_builder->get_serialized_types().find(*op.type);
        assert(type_iter != type_builder->get_serialized_types().end());

        auto *lhs = clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));
        auto result = get_sema().BuildCStyleCastExpr(
            clang::SourceLocation(), ctx.getTrivialTypeSourceInfo(type_iter->second),
            clang::SourceLocation(), lhs
        );
        assert(!result.isInvalid() && "Invalid cast expr result");

        if (!op.output) {
            return std::make_pair(result.getAs< clang::Stmt >(), true);
        }

        auto *output = clang::dyn_cast< clang::Expr >(
            create_varnode(ctx, function, *op.output, /*is_input=*/false)
        );

        auto output_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign, output,
            result.getAs< clang::Expr >()
        );
        assert(!output_result.isInvalid() && "Invalid assignment result");

        return std::make_pair(output_result.getAs< clang::Expr >(), false);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_float2float(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_trunc(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_TRUNC) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        auto merge_to_next = !op.output.has_value();
        assert(op.inputs.size() == 1u);

        auto type_iter = type_builder->get_serialized_types().find(*op.type);
        assert(type_iter != type_builder->get_serialized_types().end());

        auto *src_expr = create_varnode(ctx, function, op.inputs[0]);

        set_sema_context(ctx.getTranslationUnitDecl());
        auto result = get_sema().ImpCastExprToType(
            clang::dyn_cast< clang::Expr >(src_expr), type_iter->second, clang::CK_IntegralCast,
            clang::VK_PRValue, nullptr
        );

        if (result.isInvalid()) {
            llvm::errs() << "Failed to create operation for trunc\n";
            return std::make_pair(nullptr, true);
        }

        if (merge_to_next) {
            return std::make_pair(result.getAs< clang::Stmt >(), true);
        }

        // If output varnode is avaiable
        auto *dest_expr = create_varnode(ctx, function, *op.output, /*is_input=*/false);
        auto out_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign,
            clang::dyn_cast< clang::Expr >(dest_expr), result.getAs< clang::Expr >()
        );

        return std::make_pair(out_result.getAs< clang::Stmt >(), merge_to_next);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_ptrsub(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_PTRSUB) {
            assert(false && "Invalid PTRSUB operation.");
            return std::make_pair(nullptr, false);
        }

        auto merge_to_next = !op.output.has_value();
        auto *input0_expr  = create_varnode(ctx, function, op.inputs[0]);
        auto type_iter     = type_builder->get_serialized_types().find(*op.type);
        assert(type_iter != type_builder->get_serialized_types().end());
        auto ptr_type = type_iter->second;

        auto *ptr_expr = clang::ImplicitCastExpr::Create(
            ctx, ptr_type, clang::CK_BitCast, clang::dyn_cast< clang::Expr >(input0_expr),
            nullptr, clang::VK_PRValue, clang::FPOptionsOverride()
        );

        auto *byte_offset =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));

        auto *ptr_add_expr = clang::BinaryOperator::Create(
            ctx, ptr_expr, byte_offset, clang::BO_Add, ptr_type, clang::VK_PRValue,
            clang::OK_Ordinary, clang::SourceLocation(), clang::FPOptionsOverride()
        );

        auto *result_expr = clang::ImplicitCastExpr::Create(
            ctx, ptr_type, clang::CK_BitCast, ptr_add_expr, nullptr, clang::VK_PRValue,
            clang::FPOptionsOverride()
        );

        return std::make_pair(result_expr, merge_to_next);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_ptradd(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_PTRADD) {
            assert(false);
            return std::make_pair(nullptr, true);
        }

        auto merge_to_next = !op.output.has_value();
        assert(op.inputs.size() == 3U);

        auto *base =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));
        auto *index =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));
        auto *scale =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[2]));

        auto mult_result =
            get_sema().CreateBuiltinBinOp(clang::SourceLocation(), clang::BO_Mul, index, scale);
        assert(!mult_result.isInvalid());

        auto result = get_sema().CreateBuiltinBinOp(
            clang::SourceLocation(), clang::BO_Add, base, mult_result.getAs< clang::Expr >()
        );
        assert(!result.isInvalid());

        if (merge_to_next) {
            return std::make_pair(result.getAs< clang::Stmt >(), merge_to_next);
        }

        auto *output_stmt = create_varnode(ctx, function, *op.output, /*is_input=*/false);
        if (output_stmt->getStmtClass() == clang::Stmt::DeclStmtClass) {
            auto *decl     = clang::dyn_cast< clang::DeclStmt >(output_stmt)->getSingleDecl();
            auto *ref_expr = clang::DeclRefExpr::Create(
                ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(),
                clang::dyn_cast< clang::VarDecl >(decl), false, clang::SourceLocation(),
                clang::dyn_cast< clang::VarDecl >(decl)->getType(), clang::VK_LValue
            );

            auto ref_result = get_sema().CreateBuiltinBinOp(
                source_location_from_key(ctx, op.key), clang::BO_Assign,
                clang::dyn_cast< clang::Expr >(ref_expr), result.getAs< clang::Expr >()
            );

            assert(!ref_result.isInvalid());
            return std::make_pair(ref_result.getAs< clang::Stmt >(), false);
        }

        auto output_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign,
            clang::dyn_cast< clang::Expr >(output_stmt), result.getAs< clang::Expr >()
        );

        if (output_result.isInvalid()) {
            assert(false && "Invalid result from assignment operation");
            return std::make_pair(nullptr, false);
        }

        return std::make_pair(output_result.getAs< clang::Stmt >(), false);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_cast(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_CAST) {
            assert(false);
            return std::make_pair(nullptr, true);
        }

        auto merge_to_next = !op.output.has_value();
        assert(op.inputs.size() == 1U);

        auto type_iter = type_builder->get_serialized_types().find(*op.type);
        assert(type_iter != type_builder->get_serialized_types().end());

        auto *input_expr = create_varnode(ctx, function, op.inputs[0]);
        auto result      = get_sema().BuildCStyleCastExpr(
            clang::SourceLocation(), ctx.getTrivialTypeSourceInfo(type_iter->second),
            clang::SourceLocation(), clang::dyn_cast< clang::Expr >(input_expr)
        );
        assert(!result.isInvalid());

        return std::make_pair(result.getAs< clang::Stmt >(), merge_to_next);
    }

    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_address_of(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.mnemonic != Mnemonic::OP_ADDRESS_OF) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        auto merge_to_next = !op.output.has_value();
        auto *input_expr   = create_varnode(ctx, function, op.inputs[0]);
        assert(input_expr != nullptr);

        auto result = get_sema().CreateBuiltinUnaryOp(
            clang::SourceLocation(), clang::UO_AddrOf,
            clang::dyn_cast< clang::Expr >(input_expr)
        );
        assert(!result.isInvalid());

        if (merge_to_next) {
            return std::make_pair(result.getAs< clang::Expr >(), merge_to_next);
        }

        auto *output_expr = create_varnode(ctx, function, *op.output, /*is_input=*/false);

        auto *result_expr = result.getAs< clang::Expr >();
        auto output_type  = clang::dyn_cast< clang::Expr >(output_expr)->getType();

        if (result_expr->getType() != output_type) {
            auto cast_result = get_sema().BuildCStyleCastExpr(
                clang::SourceLocation(), ctx.getTrivialTypeSourceInfo(output_type),
                clang::SourceLocation(), result_expr
            );

            assert(!cast_result.isInvalid() && "Invalid cstyle cast to output expr");
            result_expr = cast_result.getAs< clang::Expr >();
        }

        auto output_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign,
            clang::dyn_cast< clang::Expr >(output_expr), result_expr
        );
        assert(!output_result.isInvalid());

        return std::make_pair(output_result.getAs< clang::Expr >(), merge_to_next);
    }

    template< clang::BinaryOperatorKind Kind >
    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_binary_operation(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        auto merge_to_next = !op.output.has_value();
        assert(op.inputs.size() == 2 && "Insufficient input operators");

        auto *lhs = clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        auto *rhs = clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));

        auto result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), Kind, clang::dyn_cast< clang::Expr >(lhs),
            clang::dyn_cast< clang::Expr >(rhs)
        );

        assert(!result.isInvalid() && "Invalid result from binary operation");

        if (merge_to_next) {
            return std::make_pair(result.getAs< clang::Stmt >(), merge_to_next);
        }

        auto *output_expr = clang::dyn_cast< clang::Expr >(
            create_varnode(ctx, function, *op.output, /*is_input=*/false)
        );

        auto output_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign,
            clang::dyn_cast< clang::Expr >(output_expr), result.getAs< clang::Expr >()
        );
        assert(
            !output_result.isInvalid() && "Invalid assignment operation after binary operator"
        );

        return std::make_pair(output_result.getAs< clang::Expr >(), merge_to_next);
    }

    template< clang::UnaryOperatorKind Kind >
    std::pair< clang::Stmt *, bool > PcodeASTConsumer::create_unary_operation(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        // If output varnode is emptry, the stmt will get merged to next operation.
        auto merge_to_next = !op.output.has_value();
        auto *input =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        auto result = get_sema().CreateBuiltinUnaryOp(clang::SourceLocation(), Kind, input);
        assert(!result.isInvalid() && "Invalid unary operation");

        if (merge_to_next) {
            // merge to next operation
            return std::make_pair(result.getAs< clang::Stmt >(), merge_to_next);
        }

        auto *output = clang::dyn_cast< clang::Expr >(
            create_varnode(ctx, function, *op.output, /*is_input=*/false)
        );

        auto *result_expr = result.getAs< clang::Expr >();
        if (result_expr->getType() != output->getType()) {
            auto cast_result = get_sema().BuildCStyleCastExpr(
                clang::SourceLocation(), ctx.getTrivialTypeSourceInfo(output->getType()),
                clang::SourceLocation(), result_expr
            );

            assert(!cast_result.isInvalid() && "Invalid cstyle cast to output expr");
            result_expr = cast_result.getAs< clang::Expr >();
        }

        auto output_result = get_sema().CreateBuiltinBinOp(
            source_location_from_key(ctx, op.key), clang::BO_Assign, output, result_expr
        );
        assert(!output_result.isInvalid());

        return std::make_pair(output_result.getAs< clang::Expr >(), false);
    }

} // namespace patchestry::ast
