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
#include <patchestry/AST/FunctionBuilder.hpp>
#include <patchestry/AST/OperationBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {

    extern std::optional< Operation >
    operationFromKey(const Function &function, const std::string &lookup_key); // NOLINT

    namespace {

        clang::VarDecl *create_variable_decl( // NOLINT
            clang::ASTContext &ctx, clang::DeclContext *dc, const std::string &name,
            clang::QualType type, clang::SourceLocation loc
        ) {
            return clang::VarDecl::Create(
                ctx, dc, loc, loc, &ctx.Idents.get(name), type,
                ctx.getTrivialTypeSourceInfo(type), clang::SC_None
            );
        }

        clang::DeclStmt *create_decl_stmt(
            clang::ASTContext &ctx, clang::Decl *decl, clang::SourceLocation loc
        ) { // NOLINT
            auto decl_group = clang::DeclGroupRef(decl);
            return new (ctx) clang::DeclStmt(decl_group, loc, loc);
        }

        clang::Expr *createDefaultArgument(clang::ASTContext &ctx, clang::ParmVarDecl *param) {
            if (param == nullptr) {
                return nullptr;
            }

            auto param_type = param->getType();
            if (param_type->isIntegerType()) {
                return new (ctx) clang::IntegerLiteral(
                    ctx, llvm::APInt(32U, 0), param_type, clang::SourceLocation()
                );
            } else if (param_type->isFloatingType()) {
                llvm::APFloat value(llvm::APFloat::IEEEsingle(), "0.0");
                return clang::FloatingLiteral::Create(
                    ctx, value, false, param_type, clang::SourceLocation()
                );
            } else if (param_type->isPointerType()) {
                return new (ctx) clang::IntegerLiteral(
                    ctx, llvm::APInt(32U, 0), param_type, clang::SourceLocation()
                );
            } else if (param_type->isBooleanType()) {
                return new (ctx)
                    clang::CXXBoolLiteralExpr(true, param_type, clang::SourceLocation());
            } else {
                LOG(ERROR) << "Failed to create default value for paramer\n";
                return nullptr;
            }
        }

    } // namespace

    /**
     * Performs an implicit and explicit cast of an expression to a specified type,
     * falling back to a manual pointer-based cast if necessary.
     */
    clang::Expr *OpBuilder::make_cast(
        clang::ASTContext &ctx, clang::Expr *expr, clang::QualType to_type,
        clang::SourceLocation loc
    ) {
        if (expr == nullptr || to_type.isNull()) {
            LOG(ERROR) << "Invalid expr of type to perform explicit cast";
            return {};
        }

        auto from_type = expr->getType();
        if (ctx.hasSameUnqualifiedType(from_type, to_type)) {
            return expr;
        }

        if (expr->isPRValue()) {
            auto *cast_expr =
                make_implicit_cast(ctx, expr, to_type, getCastKind(ctx, from_type, to_type));
            if (cast_expr != nullptr) {
                return cast_expr;
            }
        } else {
            auto *cast_expr = make_explicit_cast(ctx, expr, to_type, loc);
            if (cast_expr != nullptr) {
                return cast_expr;
            }
        }

        // Fallbak to reinterpret cast like expression
        auto addr_of_expr = sema().CreateBuiltinUnaryOp(loc, clang::UO_AddrOf, expr);
        assert(!addr_of_expr.isInvalid());

        auto to_pointer_type = ctx.getPointerType(to_type);
        auto casted_expr     = sema().BuildCStyleCastExpr(
            loc, ctx.getTrivialTypeSourceInfo(to_pointer_type), loc,
            addr_of_expr.getAs< clang::Expr >()
        );
        assert(!casted_expr.isInvalid());

        auto derefed_expr = sema().CreateBuiltinUnaryOp(
            loc, clang::UO_Deref, casted_expr.getAs< clang::Expr >()
        );
        assert(!derefed_expr.isInvalid());

        return derefed_expr.getAs< clang::Expr >();
    }

    clang::Expr *OpBuilder::make_explicit_cast(
        clang::ASTContext &ctx, clang::Expr *expr, clang::QualType to_type,
        clang::SourceLocation loc
    ) {
        auto casted_expr =
            sema().BuildCStyleCastExpr(loc, ctx.getTrivialTypeSourceInfo(to_type), loc, expr);
        assert(!casted_expr.isInvalid() && "Invalid casted result");
        return casted_expr.getAs< clang::Expr >();
    }

    clang::Expr *OpBuilder::make_implicit_cast(
        clang::ASTContext &ctx, clang::Expr *expr, clang::QualType to_type, clang::CastKind kind
    ) {
        if (expr == nullptr || to_type.isNull()) {
            LOG(ERROR) << "Invalid arguments: "
                       << (!expr ? "null expression" : "null target type");
            return nullptr;
        }

        if (!expr->isPRValue()) {
            switch (kind) {
                default:
                    UNREACHABLE(("can't implicitly cast glvalue to prvalue with this cast "
                                 "kind: "
                                 + std::string(clang::CastExpr::getCastKindName(kind)))
                                    .c_str());
                case clang::CastKind::CK_Dependent:
                case clang::CastKind::CK_LValueToRValue:
                case clang::CastKind::CK_ArrayToPointerDecay:
                case clang::CastKind::CK_FunctionToPointerDecay:
                case clang::CastKind::CK_ToVoid:
                case clang::CastKind::CK_NonAtomicToAtomic:
                    break;
            }
        }

        // Handle CK_LValueToRValue differently because sema function does not do anything
        // if SrcTy and DestTy is same.
        if (kind == clang::CastKind::CK_LValueToRValue) {
            return clang::ImplicitCastExpr::Create(
                ctx, expr->getType(), clang::CastKind::CK_LValueToRValue, expr, nullptr,
                clang::VK_PRValue, sema().CurFPFeatureOverrides()
            );
        }

        auto result = sema().ImpCastExprToType(expr, to_type, kind);
        assert(!result.isInvalid() && "Failed to make implicit cast expr");
        return result.getAs< clang::Expr >();
    }

    clang::Stmt *OpBuilder::create_assign_operation(
        clang::ASTContext &ctx, clang::Expr *input_expr, clang::Expr *output_expr,
        clang::SourceLocation loc
    ) {
        if ((input_expr == nullptr) || (output_expr == nullptr)) {
            return {};
        }

        clang::QualType input_type  = input_expr->getType();
        clang::QualType output_type = output_expr->getType();

        // Handle exact type match: no cast required
        if (ctx.hasSameUnqualifiedType(input_type, output_type)) {
            auto assign_operation =
                sema().CreateBuiltinBinOp(loc, clang::BO_Assign, output_expr, input_expr);
            assert(!assign_operation.isInvalid());
            return assign_operation.getAs< clang::Stmt >();
        }

        if ((input_type->isArithmeticType() && output_type->isArithmeticType())) {
            auto casted_expr = sema().PerformImplicitConversion(
                input_expr, output_type, clang::AssignmentAction::Converting
            );
            assert(!casted_expr.isInvalid());

            auto assign_operation = sema().CreateBuiltinBinOp(
                loc, clang::BO_Assign, output_expr, casted_expr.getAs< clang::Expr >()
            );
            assert(!assign_operation.isInvalid());
            return assign_operation.getAs< clang::Stmt >();
        }

        // Same size record types - direct conversion
        if (output_type->isRecordType() && input_type->isRecordType()
            && ctx.getTypeSize(output_type) == ctx.getTypeSize(input_type))
        {
            auto implicit_cast = sema().PerformImplicitConversion(
                input_expr, output_expr->getType(), clang::AssignmentAction::Converting
            );

            if (!implicit_cast.isInvalid()) {
                auto assign_operation = sema().CreateBuiltinBinOp(
                    loc, clang::BO_Assign, output_expr, implicit_cast.getAs< clang::Expr >()
                );
                assert(!assign_operation.isInvalid());
                return assign_operation.getAs< clang::Stmt >();
            }
        }

        // Handle pointer and array conversions
        if ((input_type->isPointerType() || input_type->isArrayType())
            && (output_type->isPointerType() || output_type->isArrayType()))
        {
            auto implicit_cast = sema().PerformImplicitConversion(
                input_expr, output_type, clang::AssignmentAction::Converting
            );
            if (!implicit_cast.isInvalid()) {
                auto assign_operation = sema().CreateBuiltinBinOp(
                    loc, clang::BO_Assign, output_expr, implicit_cast.getAs< clang::Expr >()
                );
                assert(!assign_operation.isInvalid());
                return assign_operation.getAs< clang::Stmt >();
            }
        }

        if (input_type->isPointerType() && output_type->isArithmeticType()) {
            auto cast_result = sema().ImpCastExprToType(
                input_expr, output_type, clang::CastKind::CK_PointerToIntegral
            );
            assert(!cast_result.isInvalid());
            auto assign_operation = sema().CreateBuiltinBinOp(
                loc, clang::BO_Assign, output_expr, cast_result.getAs< clang::Expr >()
            );
            assert(!assign_operation.isInvalid());
            return assign_operation.getAs< clang::Stmt >();
        }

        auto bitcast_result =
            sema().ImpCastExprToType(input_expr, output_type, clang::CastKind::CK_BitCast);
        assert(!bitcast_result.isInvalid());

        auto assign_operation = sema().CreateBuiltinBinOp(
            loc, clang::BO_Assign, output_expr, bitcast_result.getAs< clang::Expr >()
        );
        assert(!assign_operation.isInvalid());
        return assign_operation.getAs< clang::Stmt >();
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_copy(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.empty()) {
            LOG(ERROR) << "Copy operation does not have input operand. key: " << op.key << "\n";
            return { nullptr, false };
        }

        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs.front()));
        if (input_expr == nullptr) {
            LOG(ERROR) << "Failed to create input expression for copy operaion. key: " << op.key
                       << "\n";
            return { nullptr, false };
        }

        if (!op.output) {
            // if copy operation has no output, return input expression that will get merged
            // to next operation.
            return { input_expr, true };
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));
        if (output_expr == nullptr) {
            LOG(ERROR) << "Failed to create output expression for copy operaion. key: "
                       << op.key << "\n";
            return { nullptr, false };
        }

        return { create_assign_operation(
                     ctx, input_expr, output_expr,
                     sourceLocation(ctx.getSourceManager(), op.key)
                 ),
                 false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_load(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.empty()) {
            LOG(ERROR) << "Load operation with no input operand. key: " << op.key << "\n";
            return { nullptr, false };
        }

        auto merge_to_next = !op.output.has_value();
        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));
        if (input_expr == nullptr) {
            LOG(ERROR) << "Skipping, load operation with invalid expression. key: " << op.key
                       << "\n";
            return { nullptr, false };
        }

        auto derefed_expr = sema().CreateBuiltinUnaryOp(
            sourceLocation(ctx.getSourceManager(), op.key), clang::UO_Deref,
            clang::dyn_cast< clang::Expr >(input_expr)
        );

        if (merge_to_next) {
            return std::make_pair(derefed_expr.getAs< clang::Expr >(), merge_to_next);
        }

        auto *result_expr = derefed_expr.getAs< clang::Expr >();

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(
                     ctx, result_expr, output_expr,
                     sourceLocation(ctx.getSourceManager(), op.key)
                 ),
                 false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_store(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.empty()) {
            LOG(ERROR) << "Store operation with zero input operand. key: " << op.key << "\n";
            return {};
        }

        if (op.inputs.size() == 2) {
            auto *lhs_expr =
                clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

            auto *rhs_expr =
                clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));

            auto deref_result = sema().CreateBuiltinUnaryOp(
                sourceLocation(ctx.getSourceManager(), op.key), clang::UO_Deref,
                clang::dyn_cast< clang::Expr >(lhs_expr)
            );
            assert(!deref_result.isInvalid());

            return { create_assign_operation(
                         ctx, rhs_expr, deref_result.getAs< clang::Expr >(),
                         sourceLocation(ctx.getSourceManager(), op.key)
                     ),
                     false };
        }

        // TODO(kumarak): A pointer can come with space id and offset part. We don't handle
        // such cases yet. Parameters           Description
        //   input0(special)    Constant ID of space to storeinto.
        //   input1             Varnode containing pointer offset of destination.
        //   input2             Varnode containing data to be stored.
        // Semantic statement
        //   *input1 = input2;
        //   *[input0] input1 = input2;

        return {};
    }

    std::pair< clang::Stmt *, bool >
    OpBuilder::create_branch(clang::ASTContext &ctx, const Operation &op) {
        if (!op.target_block) {
            LOG(ERROR) << "Branch instruction with no target block. key: " << op.key << "\n";
            return {};
        }

        if (!function_builder().labels_declaration.contains(*op.target_block)) {
            LOG(ERROR) << "Target block does not have a label declaration. key: " << op.key
                       << "\n";
            return {};
        }

        auto op_loc     = sourceLocation(ctx.getSourceManager(), op.key);
        auto target_loc = sourceLocation(ctx.getSourceManager(), *op.target_block);
        auto *expr      = new (ctx) clang::GotoStmt(
            function_builder().labels_declaration.at(*op.target_block), op_loc, target_loc
        );
        if (expr == nullptr) {
            LOG(ERROR) << "Failed to create goto statement. key " << op.key << "\n";
            return {};
        }

        return { expr, false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_cbranch(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (!op.condition) {
            LOG(ERROR) << "Conditional branch with no condition operator. key: " << op.key
                       << "\n";
            return {};
        }

        // TODO(kumarak): Could there be case where conditional statement is missing?? In
        // such case treat it as branch instruction.
        auto loc = sourceLocation(ctx.getSourceManager(), op.key);
        auto *condition_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.condition));

        clang::Stmt *taken_stmt     = nullptr;
        clang::Stmt *not_taken_stmt = nullptr;

        if (op.taken_block && !op.taken_block->empty()
            && function_builder().labels_declaration.contains(*op.taken_block))
        {
            auto ll    = sourceLocation(ctx.getSourceManager(), *op.taken_block);
            taken_stmt = new (ctx) clang::GotoStmt(
                function_builder().labels_declaration.at(*op.taken_block), loc, ll
            );
        } else {
            taken_stmt = new (ctx) clang::NullStmt(loc, false);
        }

        if (op.not_taken_block && !op.not_taken_block->empty()
            && function_builder().labels_declaration.contains(*op.not_taken_block))
        {
            auto ll        = sourceLocation(ctx.getSourceManager(), *op.not_taken_block);
            not_taken_stmt = new (ctx) clang::GotoStmt(
                function_builder().labels_declaration.at(*op.not_taken_block), loc, ll
            );
        } else {
            not_taken_stmt = new (ctx) clang::NullStmt(loc, false);
        }

        return std::make_pair(
            clang::IfStmt::Create(
                ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                clang::dyn_cast< clang::Expr >(condition_expr), condition_expr->getBeginLoc(),
                taken_stmt->getBeginLoc(), taken_stmt, not_taken_stmt->getBeginLoc(),
                not_taken_stmt
            ),
            false
        );
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_branchind(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return {};
    }

    void OpBuilder::extend_callexpr_agruments(
        clang::ASTContext &ctx, clang::FunctionDecl *fndecl,
        std::vector< clang::Expr * > &arguments
    ) {
        auto minargs = fndecl->getMinRequiredArguments();
        for (unsigned i = static_cast< unsigned >(arguments.size()); i < minargs; i++) {
            auto *param = fndecl->getParamDecl(i);
            arguments.emplace_back(createDefaultArgument(ctx, param));
        }
    }

    clang::Expr *OpBuilder::build_callexpr_from_function(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (!op.target || !op.target->function) {
            LOG(ERROR) << "Missing target function, can't build callexpr\n";
            return nullptr;
        }

        if (!function_builder().function_list.get().contains(*op.target->function)) {
            LOG(ERROR) << "Missing target function from list, can't build callexpr\n";
            return nullptr;
        }

        auto op_loc = sourceLocation(ctx.getSourceManager(), op.key);

        // get the callee and its prototy for creating function argument list
        const auto &callee = function_builder().function_list.get().at(*op.target->function);
        auto *proto_type   = callee->getType()->getAs< clang::FunctionProtoType >();
        auto num_params    = proto_type->getNumParams();

        // The callee return type may be missing or incorrect during representing high pcode
        // into JSON format. Double check the function return type with operation type and fix
        // if there is any mismatch.
        // TODO(kumarak): Switch to delay the creation of AST node for function declaration and
        // fix return type during creating the node.
        if (op.type) {
            const auto &op_type = type_builder().get_serialized_types().at(*op.type);
            if (callee->getReturnType() != op_type && callee->getReturnType()->isVoidType()) {
                auto param_types    = proto_type->getParamTypes();
                auto epi            = proto_type->getExtProtoInfo();
                auto new_proto_type = ctx.getFunctionType(op_type, param_types, epi);
                callee->setType(new_proto_type);
            }
        }

        unsigned index = 0;
        std::vector< clang::Expr * > arguments;
        for (const auto &input : op.inputs) {
            if (index >= num_params) {
                // The recovered operation inputs does not match with number of function
                // arguments. Drop extra inputs and don't add them to the callee arguments.
                continue;
            }
            auto *vnode_expr =
                clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, input));
            auto arg_type = proto_type->getParamType(index++);
            if (!vnode_expr->isPRValue()) {
                vnode_expr = make_implicit_cast(
                    ctx, vnode_expr, vnode_expr->getType(), clang::CastKind::CK_LValueToRValue
                );
                assert(vnode_expr != nullptr && "Failed to convert to rvalue");
            }
            auto *arg = make_implicit_cast(
                ctx, vnode_expr, arg_type, getCastKind(ctx, vnode_expr->getType(), arg_type)
            );
            assert(arg != nullptr && "Function argument is null");
            arguments.push_back(arg);
        }

        // Check if the number of arguments is less then minimum number of require arguments;
        // extend it with the default value.
        unsigned min_args = callee->getMinRequiredArguments();
        if (arguments.size() < min_args) {
            extend_callexpr_agruments(ctx, callee, arguments);
        }

        auto *refexpr = clang::DeclRefExpr::Create(
            ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), callee, false,
            op_loc, ctx.getPointerType(callee->getType()), clang::VK_PRValue
        );

        auto result = sema().BuildCallExpr(
            nullptr, clang::dyn_cast< clang::Expr >(refexpr), op_loc, arguments, op_loc
        );
        assert(!result.isInvalid() && "Failed to build call expr");
        return result.getAs< clang::Expr >();
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_call(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (!op.target || op.mnemonic != Mnemonic::OP_CALL) {
            LOG(ERROR) << "Call operation or call target is invalid. key: " << op.key << "\n";
            return {};
        }

        if (!op.target->function && !op.target->operation) {
            LOG(ERROR) << "Call target does not have function or operation associated. key: "
                       << op.key << "\n";
            return {};
        }

        auto op_loc            = sourceLocation(ctx.getSourceManager(), op.key);
        clang::Expr *call_expr = nullptr;

        if (op.target->function) {
            if (!function_builder().function_list.get().contains(*op.target->function)) {
                return {};
            }

            const auto &callee =
                function_builder().function_list.get().at(*op.target->function);

            call_expr = build_callexpr_from_function(ctx, function, op);
            if (!op.output || callee->getReturnType()->isVoidType()) {
                return std::make_pair(clang::dyn_cast< clang::Expr >(call_expr), false);
            }

        } else if (op.target->operation) {
            // Get list of arguments for function call
            std::vector< clang::Expr * > arguments;
            for (const auto &input : op.inputs) {
                auto *arg_expr =
                    clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, input));
                arguments.push_back(clang::dyn_cast< clang::Expr >(arg_expr));
            }

            auto operation = operationFromKey(function, *op.target->operation);
            auto [stmt, _] = function_builder().create_operation(ctx, *operation);
            auto result    = sema().BuildCallExpr(
                nullptr, clang::dyn_cast< clang::Expr >(stmt), op_loc, arguments, op_loc
            );
            assert(!result.isInvalid());
            call_expr = result.getAs< clang::Expr >();
            if (!operation->output || call_expr->getType()->isVoidType()) {
                return std::make_pair(clang::dyn_cast< clang::Expr >(call_expr), false);
            }
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(
                     ctx, call_expr, clang::dyn_cast< clang::Expr >(output_expr), op_loc
                 ),
                 false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_callind(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_userdefined(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_return(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        auto location = sourceLocation(ctx.getSourceManager(), op.key);
        if (!op.inputs.empty()) {
            auto varnode   = op.inputs.size() == 1 ? op.inputs.front() : op.inputs.at(1);
            auto *ret_expr = create_varnode(ctx, function, varnode);
            return std::make_pair(
                clang::ReturnStmt::Create(
                    ctx, location, llvm::dyn_cast< clang::Expr >(ret_expr), nullptr
                ),
                false
            );
        }
        return std::make_pair(
            clang::ReturnStmt::Create(ctx, location, nullptr, nullptr), false
        );
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_piece(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.size() != 2U || op.mnemonic != Mnemonic::OP_PIECE) {
            LOG(ERROR) << "PIECE Operation with invalid input operands or invalid "
                          "operation. key: "
                       << op.key;
            return {};
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "PIECE Operation type is not serialized. key: " << op.key;
            return {};
        }

        auto *input0_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));
        auto *input1_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));
        auto location = sourceLocation(ctx.getSourceManager(), op.key);

        // TODO(kumarak): It should be the size of input1 field in bits; At the moment
        // consider it as 4 bytes but should be fixed.
        unsigned low_width = 32U;

        auto merge_to_next = !op.output.has_value();
        auto *shift_value =
            clang::IntegerLiteral::Create(ctx, llvm::APInt(32, low_width), ctx.IntTy, location);

        auto shifted_high_result = sema().CreateBuiltinBinOp(
            location, clang::BO_Shl, input0_expr, clang::dyn_cast< clang::Expr >(shift_value)
        );

        if (shifted_high_result.isInvalid()) {
            LOG(ERROR) << "PIECE Operation invalid shifted high result.\n";
            return {};
        }

        auto or_result = sema().CreateBuiltinBinOp(
            location, clang::BO_Or, shifted_high_result.getAs< clang::Expr >(),
            clang::dyn_cast< clang::Expr >(input1_expr)
        );
        assert(!or_result.isInvalid());

        if (merge_to_next) {
            return std::make_pair(or_result.getAs< clang::Expr >(), merge_to_next);
        }

        auto *output_expr = create_varnode(ctx, function, *op.output);
        return { create_assign_operation(
                     ctx, or_result.getAs< clang::Expr >(),
                     clang::dyn_cast< clang::Expr >(output_expr),
                     sourceLocation(ctx.getSourceManager(), op.key)
                 ),
                 false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_subpiece(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.size() != 2U || op.mnemonic != Mnemonic::OP_SUBPIECE) {
            LOG(ERROR) << "SUBPIECE Operation with invalid input operands or invalid "
                          "operation. key: "
                       << op.key;
            return {};
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "SUBPIECE Operation type is not serialized. key: " << op.key;
            return {};
        }

        auto merge_to_next  = !op.output.has_value();
        const auto &op_type = type_builder().get_serialized_types().at(*op.type);
        auto op_location    = sourceLocation(ctx.getSourceManager(), op.key);

        auto *shift_value =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));

        auto *expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        if (!ctx.hasSameUnqualifiedType(expr->getType(), op_type)) {
            if (auto *casted_expr = make_cast(ctx, expr, op_type, op_location)) {
                expr = casted_expr;
            }
        }

        auto *expr_with_paren = new (ctx)
            clang::ParenExpr(op_location, op_location, clang::dyn_cast< clang::Expr >(expr));

        auto shifted_result = sema().CreateBuiltinBinOp(
            op_location, clang::BO_Shr, clang::dyn_cast< clang::Expr >(expr_with_paren),
            clang::dyn_cast< clang::Expr >(shift_value)
        );

        if (shifted_result.isInvalid()) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        auto *shifted_expr = new (ctx)
            clang::ParenExpr(op_location, op_location, shifted_result.getAs< clang::Expr >());

        auto mask_value = llvm::APInt::getAllOnes(32U);
        auto *mask = clang::IntegerLiteral::Create(ctx, mask_value, ctx.IntTy, op_location);

        auto result = sema().CreateBuiltinBinOp(
            sourceLocation(ctx.getSourceManager(), op.key), clang::BO_And, shifted_expr,
            clang::dyn_cast< clang::Expr >(mask)
        );

        if (result.isInvalid()) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        auto *result_expr =
            new (ctx) clang::ParenExpr(op_location, op_location, result.getAs< clang::Expr >());

        if (merge_to_next) {
            return std::make_pair(result_expr, merge_to_next);
        }

        auto *out_expr  = create_varnode(ctx, function, *op.output);
        auto out_result = sema().CreateBuiltinBinOp(
            op_location, clang::BO_Assign, clang::dyn_cast< clang::Expr >(out_expr), result_expr
        );

        if (out_result.isInvalid()) {
            assert(false);
            return std::make_pair(nullptr, false);
        }

        return std::make_pair(out_result.getAs< clang::Stmt >(), merge_to_next);
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_int_zext(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.size() != 1U && op.mnemonic != Mnemonic::OP_INT_ZEXT) {
            LOG(ERROR) << "INT_ZEXT operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return {};
        }

        auto merge_to_next = !op.output.has_value();
        auto op_loc        = sourceLocation(ctx.getSourceManager(), op.key);
        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        auto target_type = type_builder().get_serialized_types().at(*op.type);

        auto implicit_result = sema().PerformImplicitConversion(
            input_expr, target_type, clang::AssignmentAction::Converting, true
        );

        if (implicit_result.isInvalid()) {
            // If fail to perform implcit cast perform explicit cast
            auto result = sema().BuildCStyleCastExpr(
                op_loc, ctx.getTrivialTypeSourceInfo(target_type), op_loc,
                clang::dyn_cast< clang::Expr >(input_expr)
            );

            input_expr = result.getAs< clang::Expr >();
        } else {
            input_expr = implicit_result.getAs< clang::Expr >();
        }

        if (merge_to_next) {
            return { input_expr, merge_to_next };
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(ctx, input_expr, output_expr, op_loc), false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_int_sext(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.size() != 1U && op.mnemonic != Mnemonic::OP_INT_SEXT) {
            LOG(ERROR) << "INT_SEXT operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return {};
        }

        auto merge_to_next = !op.output.has_value();
        auto op_loc        = sourceLocation(ctx.getSourceManager(), op.key);

        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        auto target_type = type_builder().get_serialized_types().at(*op.type);

        auto implicit_result = sema().PerformImplicitConversion(
            input_expr, target_type, clang::AssignmentAction::Converting, true
        );

        if (implicit_result.isInvalid()) {
            // If fail to perform implcit cast perform explicit cast
            auto result = sema().BuildCStyleCastExpr(
                sourceLocation(ctx.getSourceManager(), op.key),
                ctx.getTrivialTypeSourceInfo(target_type),
                sourceLocation(ctx.getSourceManager(), op.key),
                clang::dyn_cast< clang::Expr >(input_expr)
            );

            input_expr = result.getAs< clang::Expr >();
        } else {
            input_expr = implicit_result.getAs< clang::Expr >();
        }

        if (merge_to_next) {
            return { input_expr, merge_to_next };
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(ctx, input_expr, output_expr, op_loc), false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_int_carry(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        UNIMPLEMENTED(" %s not implemented", __FUNCTION__); // NOLINT
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_int_scarry(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        UNIMPLEMENTED(" %s not implemented", __FUNCTION__); // NOLINT
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_int_sborrow(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        UNIMPLEMENTED(" %s not implemented", __FUNCTION__); // NOLINT
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_int_2comp(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        UNIMPLEMENTED(" %s not implemented", __FUNCTION__); // NOLINT
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_unary_operation(
        clang::ASTContext &ctx, const Function &function, const Operation &op,
        clang::UnaryOperatorKind kind
    ) {
        if (op.inputs.empty() || op.inputs.size() > 1U) {
            LOG(ERROR
            ) << "Skipping, unary operation with more than one or zero input operand. key: "
              << op.key << "\n";
            return {};
        }

        auto op_loc = sourceLocation(ctx.getSourceManager(), op.key);
        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        // TODO(kumarak): Should check the operation type before creating unary operation???
        auto unary_operation = sema().CreateBuiltinUnaryOp(op_loc, kind, input_expr);
        assert(!unary_operation.isInvalid());

        if (!op.output.has_value()) {
            return { unary_operation.getAs< clang::Stmt >(), true };
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(
                     ctx, unary_operation.getAs< clang::Expr >(), output_expr, op_loc
                 ),
                 false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_binary_operation(
        clang::ASTContext &ctx, const Function &function, const Operation &op,
        clang::BinaryOperatorKind kind
    ) {
        if (op.inputs.size() != 2U) {
            LOG(ERROR) << "Skipping, binary operation with insufficient input operand. key: "
                       << op.key << "\n";
            return {};
        }

        auto op_loc = sourceLocation(ctx.getSourceManager(), op.key);

        auto *lhs = clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        auto *rhs = clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));

        auto result = sema().CreateBuiltinBinOp(
            op_loc, kind, clang::dyn_cast< clang::Expr >(lhs),
            clang::dyn_cast< clang::Expr >(rhs)
        );
        assert(!result.isInvalid() && "Invalid result from binary operation");

        if (!op.output) {
            return std::make_pair(result.getAs< clang::Stmt >(), true);
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return {
            create_assign_operation(ctx, result.getAs< clang::Expr >(), output_expr, op_loc),
            false
        };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_float_abs(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        UNIMPLEMENTED(" %s not implemented", __FUNCTION__); // NOLINT
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_float_sqrt(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        UNIMPLEMENTED(" %s not implemented", __FUNCTION__); // NOLINT
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_float_floor(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        UNIMPLEMENTED(" %s not implemented", __FUNCTION__); // NOLINT
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_float_ceil(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        UNIMPLEMENTED(" %s not implemented", __FUNCTION__); // NOLINT
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_float_round(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        UNIMPLEMENTED(" %s not implemented", __FUNCTION__); // NOLINT
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_int2float(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        UNIMPLEMENTED(" %s not implemented", __FUNCTION__); // NOLINT
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_float_nan(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        UNIMPLEMENTED(" %s not implemented", __FUNCTION__); // NOLINT
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_float2float(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        UNIMPLEMENTED(" %s not implemented", __FUNCTION__); // NOLINT
        return {};
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_trunc(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.size() != 1U || op.mnemonic != Mnemonic::OP_TRUNC) {
            LOG(ERROR) << "TRUNC operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return { nullptr, false };
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "TRUNC operation type is not serialized. key: " << op.key << "\n";
            return { nullptr, false };
        }

        auto merge_to_next = !op.output.has_value();

        const auto &op_type = type_builder().get_serialized_types().at(*op.type);
        auto op_loc         = sourceLocation(ctx.getSourceManager(), op.key);

        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        auto implicit_cast_result = sema().PerformImplicitConversion(
            input_expr, op_type, clang::AssignmentAction::Converting
        );
        assert(!implicit_cast_result.isInvalid());

        auto *implicit_cast = implicit_cast_result.getAs< clang::Expr >();

        if (merge_to_next) {
            return std::make_pair(implicit_cast, true);
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(ctx, implicit_cast, output_expr, op_loc), false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_ptrsub(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.size() != 2U && op.mnemonic != Mnemonic::OP_PTRSUB) {
            LOG(ERROR) << "PTRSUB operation is invalid or has invalid input operands. key: "
                       << op.key << "\n";
            return { nullptr, false };
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "PTRSUB operation type is not serialized. key: " << op.key << "\n";
            return { nullptr, false };
        }

        auto merge_to_next = !op.output.has_value();

        const auto &op_type = type_builder().get_serialized_types().at(*op.type);
        auto op_loc         = sourceLocation(ctx.getSourceManager(), op.key);

        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        auto cast_result =
            sema().ImpCastExprToType(input_expr, op_type, clang::CastKind::CK_BitCast);
        assert(!cast_result.isInvalid());

        auto *ptr_expr = cast_result.getAs< clang::Expr >();

        auto *byte_offset =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));

        auto add_result =
            sema().CreateBuiltinBinOp(op_loc, clang::BO_Add, ptr_expr, byte_offset);
        assert(!add_result.isInvalid());

        auto *ptr_add_expr = add_result.getAs< clang::Expr >();
        if (merge_to_next) {
            return { ptr_add_expr, true };
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(ctx, ptr_add_expr, output_expr, op_loc), false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_ptradd(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.size() != 3U && op.mnemonic != Mnemonic::OP_PTRADD) {
            LOG(ERROR) << "Ptradd operation is invalid or has invalid input operands. key: "
                       << op.key << "\n";
            return { nullptr, false };
        }

        auto merge_to_next = !op.output.has_value();
        auto op_loc        = sourceLocation(ctx.getSourceManager(), op.key);

        auto *base =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));
        auto *index =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));
        auto *scale =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[2]));

        auto mult_result = sema().CreateBuiltinBinOp(op_loc, clang::BO_Mul, index, scale);
        assert(!mult_result.isInvalid());

        auto add_result = sema().CreateBuiltinBinOp(
            op_loc, clang::BO_Add, base, mult_result.getAs< clang::Expr >()
        );
        assert(!add_result.isInvalid());

        if (merge_to_next) {
            return std::make_pair(add_result.getAs< clang::Stmt >(), merge_to_next);
        }

        auto *output_stmt = create_varnode(ctx, function, *op.output);
        if (output_stmt->getStmtClass() == clang::Stmt::DeclStmtClass) {
            auto *decl     = clang::dyn_cast< clang::DeclStmt >(output_stmt)->getSingleDecl();
            auto *ref_expr = clang::DeclRefExpr::Create(
                ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(),
                clang::dyn_cast< clang::VarDecl >(decl), false, op_loc,
                clang::dyn_cast< clang::VarDecl >(decl)->getType(), clang::VK_LValue
            );

            return { create_assign_operation(
                         ctx, add_result.getAs< clang::Expr >(), ref_expr, op_loc
                     ),
                     false };
        }

        auto *output_expr = clang::dyn_cast< clang::Expr >(output_stmt);
        return { create_assign_operation(
                     ctx, add_result.getAs< clang::Expr >(), output_expr, op_loc
                 ),
                 false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_cast(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.size() != 1U || op.mnemonic != Mnemonic::OP_CAST) {
            LOG(ERROR) << "Cast operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return {};
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "Operation type does not exist in serialized list. key: " << op.key
                       << "\n";
            return {};
        }

        const auto &op_type = type_builder().get_serialized_types().at(*op.type);
        auto op_loc         = sourceLocation(ctx.getSourceManager(), op.key);
        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        if (!op.output && ctx.hasSameUnqualifiedType(op_type, input_expr->getType())) {
            return { input_expr, true };
        }

        // Create cast operation to op_type
        auto casted_result = sema().BuildCStyleCastExpr(
            op_loc, ctx.getTrivialTypeSourceInfo(op_type), op_loc, input_expr
        );
        assert(!casted_result.isInvalid());

        if (!op.output) {
            return { casted_result.getAs< clang::Stmt >(), true };
        }

        auto *casted_expr = casted_result.getAs< clang::Expr >();
        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(ctx, casted_expr, output_expr, op_loc), false };
    }

    std::pair< clang::Stmt *, bool >
    OpBuilder::create_declare_local(clang::ASTContext &ctx, const Operation &op) {
        if (!op.name || !op.type) {
            LOG(ERROR) << "Local/temporary variable declaration has invalid name or types.\n";
            return {};
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "Skipping, local/temporary variable type is not serialized. key: "
                       << op.key << "\n";
            return {};
        }

        const auto &var_type = type_builder().get_serialized_types()[*op.type];
        auto op_loc          = sourceLocation(ctx.getSourceManager(), op.key);
        auto *var_decl =
            create_variable_decl(ctx, sema().CurContext, *op.name, var_type, op_loc);
        // Mark all local variable used to avoid warning about unused variable
        var_decl->setIsUsed();

        // add variable declaration to list for future references
        function_builder().local_variables.emplace(op.key, var_decl);
        return std::make_pair(create_decl_stmt(ctx, var_decl, op_loc), false);
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_declare_parameter(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        (void) ctx, (void) function, (void) op;
        return std::make_pair(nullptr, true);
    }

} // namespace patchestry::ast
