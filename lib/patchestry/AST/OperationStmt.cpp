/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <optional>
#include <utility>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclAccessPair.h>
#include <clang/AST/Expr.h>
#include <clang/AST/NestedNameSpecifier.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/RecordLayout.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <clang/Basic/Builtins.h>
#include <clang/Basic/LLVM.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/Specifiers.h>
#include <clang/Sema/Lookup.h>
#include <clang/Sema/Sema.h>
#include <llvm/ADT/APInt.h>
#include <llvm/Support/Casting.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/AST/FunctionBuilder.hpp>
#include <patchestry/AST/IntrinsicHandlers.hpp>
#include <patchestry/AST/OperationBuilder.hpp>
#include <patchestry/AST/TypeBuilder.hpp>
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

        clang::DeclStmt *create_decl_stmt( // NOLINT
            clang::ASTContext &ctx, clang::Decl *decl, clang::SourceLocation loc
        ) {
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
                    ctx, llvm::APInt(ctx.getIntWidth(param_type), 0), param_type,
                    clang::SourceLocation()
                );
            } else if (param_type->isFloatingType()) {
                llvm::APFloat value(llvm::APFloat::IEEEsingle(), "0.0");
                return clang::FloatingLiteral::Create(
                    ctx, value, false, param_type, clang::SourceLocation()
                );
            } else if (param_type->isPointerType()) {
                return new (ctx) clang::IntegerLiteral(
                    ctx, llvm::APInt(ctx.getIntWidth(param_type), 0), param_type,
                    clang::SourceLocation()
                );
            } else if (param_type->isBooleanType()) {
                return new (ctx)
                    clang::CXXBoolLiteralExpr(true, param_type, clang::SourceLocation());
            } else {
                LOG(ERROR) << "Failed to create default value for paramer\n";
                return nullptr;
            }
        }

        /**
         * Helper function to create a builtin call expression.
         */
        clang::Expr *create_builtin_call(
            const clang::ASTContext &ctx, clang::Sema &sema,
            const clang::Builtin::ID builtin_id, std::vector< clang::Expr * > &args,
            const clang::SourceLocation loc
        ) {
            // Get the builtin name and create a lookup result
            const auto name = ctx.BuiltinInfo.getName(builtin_id);
            const clang::LookupResult r(
                sema, &ctx.Idents.get(name), loc, clang::Sema::LookupOrdinaryName
            );
            auto *II = r.getLookupName().getAsIdentifierInfo();

            // Get the builtin type
            clang::ASTContext::GetBuiltinTypeError error{};
            const auto ty = ctx.GetBuiltinType(builtin_id, error);
            assert(!error && "Failed to get builtin type");

            // Create the builtin declaration
            auto *builtin_decl = sema.CreateBuiltin(II, ty, builtin_id, loc);

            // Create a DeclRefExpr for the builtin
            auto *decl_ref = clang::DeclRefExpr::Create(
                ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), builtin_decl,
                false, loc, builtin_decl->getType(), clang::VK_PRValue
            );

            // Build the call expression
            auto result = sema.BuildCallExpr(nullptr, decl_ref, loc, args, loc);
            assert(!result.isInvalid() && "Failed to build builtin call expr");

            return result.getAs< clang::Expr >();
        }

    } // namespace

    /**
     * Performs an implicit and explicit cast of an expression to a specified type,
     * falling back to a manual pointer-based cast if necessary.
     */
    clang::Expr *OpBuilder::make_cast(
        clang::ASTContext &ctx, clang::Expr *expr, const clang::QualType &to_type,
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

        // Note: The exported high-pcode from ghidra may have types that can't be casted causing
        // diagnostics errors. For example:
        //   unsigned int -> struct
        //   void* -> struct
        //   struct -> unsigned int
        //   struct -> void*
        // Identify such cases early and make expression doing reinterpret cast.
        if (shouldReinterpretCast(from_type, to_type)) {
            auto *cast_expr = make_reinterpret_cast(ctx, expr, to_type, loc);
            assert(cast_expr != nullptr && "Failed to create cast expression");
            return cast_expr;
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
        return make_reinterpret_cast(ctx, expr, to_type, loc);
    }

    clang::Expr *OpBuilder::make_explicit_cast(
        clang::ASTContext &ctx, clang::Expr *expr, clang::QualType to_type,
        clang::SourceLocation loc
    ) {
        auto cast =
            sema().BuildCStyleCastExpr(loc, ctx.getTrivialTypeSourceInfo(to_type), loc, expr);
        assert(!cast.isInvalid() && "Invalid casted result");
        return cast.getAs< clang::Expr >();
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

    clang::Expr *OpBuilder::make_reinterpret_cast(
        clang::ASTContext &ctx, clang::Expr *expr, clang::QualType to_type,
        clang::SourceLocation loc
    ) {
        if (expr == nullptr || to_type.isNull()) {
            LOG(ERROR) << "Invalid expr of type to perform reinterpret cast\n";
            return nullptr;
        }

        auto create_temporary_expr = [&](clang::ASTContext &ctx,
                                         clang::Expr *expr) -> clang::Expr * {
            if (expr->isPRValue()) {
                return sema().CreateMaterializeTemporaryExpr(expr->getType(), expr, true);
            }
            return expr;
            (void) ctx;
        };

        auto *temp_expr  = create_temporary_expr(ctx, expr);
        auto addrof_expr = sema().CreateBuiltinUnaryOp(loc, clang::UO_AddrOf, temp_expr);
        assert(!addrof_expr.isInvalid() && "Invalid AddressOf expression");

        auto to_pointer_type = ctx.getPointerType(to_type);
        auto casted_expr     = sema().BuildCStyleCastExpr(
            loc, ctx.getTrivialTypeSourceInfo(to_pointer_type), loc,
            addrof_expr.getAs< clang::Expr >()
        );
        assert(!casted_expr.isInvalid());

        auto deref_expr = sema().CreateBuiltinUnaryOp(
            loc, clang::UO_Deref, casted_expr.getAs< clang::Expr >()
        );
        assert(!deref_expr.isInvalid());

        return deref_expr.getAs< clang::Expr >();
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
            // Array types are not directly assignable in C; use element-wise copy
            if (output_type->isArrayType() || output_type->isConstantArrayType()) {
                return create_array_assignment_operation(ctx, input_expr, output_expr, loc);
            }
            auto assign_operation =
                sema().CreateBuiltinBinOp(loc, clang::BO_Assign, output_expr, input_expr);
            assert(!assign_operation.isInvalid());
            return assign_operation.getAs< clang::Stmt >();
        }

        auto *cast_expr = make_cast(ctx, input_expr, output_type, loc);
        assert(cast_expr != nullptr && "Failed to create cast expressions!");

        if (output_type->isArrayType() || output_type->isConstantArrayType()) {
            return create_array_assignment_operation(ctx, cast_expr, output_expr, loc);
        }

        auto assign_operation =
            sema().CreateBuiltinBinOp(loc, clang::BO_Assign, output_expr, cast_expr);

        assert(!assign_operation.isInvalid());
        return assign_operation.getAs< clang::Stmt >();
    }

    clang::Stmt *OpBuilder::create_array_assignment_operation(
        clang::ASTContext &ctx, clang::Expr *input_expr, clang::Expr *output_expr,
        clang::SourceLocation loc
    ) {
        if ((input_expr == nullptr) || (output_expr == nullptr)) {
            return nullptr;
        }

        auto to_type      = output_expr->getType();
        auto *elem_type   = to_type->getPointeeOrArrayElementType();
        auto num_elements = ctx.getTypeSize(to_type) / ctx.getTypeSize(elem_type);
        clang::SmallVector< clang::Stmt *, 4 > body;

        auto *index = clang::VarDecl::Create(
            ctx, sema().CurContext, loc, loc, &ctx.Idents.get("i"), ctx.IntTy, nullptr,
            clang::SC_None
        );

        auto *index_init = clang::IntegerLiteral::Create(
            ctx, llvm::APInt(static_cast< unsigned >(ctx.getTypeSize(ctx.IntTy)), 0), ctx.IntTy,
            loc
        );
        index->setInit(index_init);
        auto *index_decl = new (ctx) clang::DeclStmt(clang::DeclGroupRef(index), loc, loc);

        auto *index_ref = clang::DeclRefExpr::Create(
            ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), index, false,
            clang::SourceLocation(), index->getType(), clang::VK_LValue
        );

        auto *cond_expr = sema()
                              .BuildBinOp(
                                  sema().getCurScope(), loc, clang::BO_LT, index_ref,
                                  clang::IntegerLiteral::Create(
                                      ctx, llvm::APInt(32, num_elements), ctx.IntTy, loc
                                  )
                              )
                              .get();

        auto *inc_expr =
            sema().BuildUnaryOp(sema().getCurScope(), loc, clang::UO_PreInc, index_ref).get();

        // Array assignment: LHS[i] = RHS[i];
        auto *lhs_expr =
            sema().CreateBuiltinArraySubscriptExpr(output_expr, loc, index_ref, loc).get();

        auto *rhs_expr =
            sema().CreateBuiltinArraySubscriptExpr(input_expr, loc, index_ref, loc).get();

        auto *assign_expr =
            sema()
                .BuildBinOp(sema().getCurScope(), loc, clang::BO_Assign, lhs_expr, rhs_expr)
                .get();

        // Wrap assignment in a CompoundStmt (loop body)
        body.push_back(assign_expr);
        auto *loop_body =
            clang::CompoundStmt::Create(ctx, body, sema().CurFPFeatureOverrides(), loc, loc);

        // Set the condVar to nullptr to avoid redeclaration. The ForStmt redeclare the
        // conditional stmt causing conflit with init stmt.
        return new (ctx) clang::ForStmt(
            ctx, index_decl, cond_expr, nullptr, inc_expr, loop_body, loc, loc, loc
        );
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

        auto op_loc = sourceLocation(ctx.getSourceManager(), op.key);

        if (op.type) {
            const auto &op_type = type_builder().get_serialized_types().at(*op.type);
            auto pointer_type   = ctx.getPointerType(op_type);
            auto *cast_expr     = make_cast(ctx, input_expr, pointer_type, op_loc);
            assert(cast_expr != nullptr && "Failed to make cast expression");
            input_expr = cast_expr;
        }

        // TODO(kumarak): If input expr is of type void*, derefencing it will lead to type void.
        // Should it be typecasted to int*??

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
            auto op_loc = sourceLocation(ctx.getSourceManager(), op.key);
            auto *lhs_expr =
                clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

            auto *rhs_expr =
                clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));

            // If not member expression, fallback to derefencing and assigning the value to lhs
            // expression.
            auto deref_result = sema().CreateBuiltinUnaryOp(
                op_loc, clang::UO_Deref, clang::dyn_cast< clang::Expr >(lhs_expr)
            );
            assert(!deref_result.isInvalid());

            return { create_assign_operation(
                         ctx, rhs_expr, deref_result.getAs< clang::Expr >(), op_loc
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
        if (op.inputs.size() != 1U) {
            LOG(ERROR) << "BRANCHIND operation with invalid input operand. key: " << op.key
                       << "\n";
            return {};
        }

        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));
        auto loc       = sourceLocation(ctx.getSourceManager(), op.key);
        auto expr_type = input_expr->getType();
        if (!expr_type->isPointerType()) {
            auto op_type    = ctx.getPointerType(expr_type);
            auto *cast_expr = make_cast(ctx, input_expr, op_type, loc);
            assert(cast_expr != nullptr && "Failed to create cast expression");
            input_expr = cast_expr;
        }

        // Create indirect goto statement for branchind
        auto *result_stmt = new (ctx) clang::IndirectGotoStmt(loc, loc, input_expr);
        assert(result_stmt != nullptr && "Failed to create indirect goto statement");

        return { result_stmt, false };
    }

    void OpBuilder::extend_callexpr_agruments(
        clang::ASTContext &ctx, clang::FunctionDecl *fndecl,
        std::vector< clang::Expr * > &arguments
    ) {
        auto minargs = fndecl->getMinRequiredArguments();
        for (auto i = static_cast< unsigned >(arguments.size()); i < minargs; i++) {
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
        const auto *proto_type = callee->getType()->getAs< clang::FunctionProtoType >();
        auto num_params        = proto_type->getNumParams();

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
                for (auto *decl : callee->redecls()) {
                    decl->setType(new_proto_type);
                }
            }
        }

        auto get_argument_type = [&](const clang::FunctionDecl *callee, clang::Expr *arg,
                                     unsigned index) -> clang::QualType {
            const auto *proto = callee->getType()->getAs< clang::FunctionProtoType >();
            if (proto->isVariadic() && (index >= proto->getNumParams())) {
                return arg->getType();
            }
            return proto->getParamType(index);
        };

        unsigned index = 0;
        std::vector< clang::Expr * > arguments;
        for (const auto &input : op.inputs) {
            if (index >= num_params && !proto_type->isVariadic()) {
                // The recovered operation inputs does not match with number of function
                // arguments. Drop extra inputs and don't add them to the callee arguments.
                continue;
            }
            auto *vnode_expr =
                clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, input));
            auto arg_type = get_argument_type(callee, vnode_expr, index++);
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
        // 1. Validate operation
        if (!op.target || op.mnemonic != Mnemonic::OP_CALLIND) {
            LOG(ERROR) << "CALLIND operation or target is invalid. key: " << op.key << "\n";
            return {};
        }

        auto op_loc            = sourceLocation(ctx.getSourceManager(), op.key);
        clang::Expr *fn_ptr_expr = nullptr;

        // 2. Get function pointer expression from target
        if (op.target->global) {
            // Target is a global variable containing the function pointer
            if (!function_builder().global_var_list.get().contains(*op.target->global)) {
                LOG(ERROR) << "CALLIND target global not found: " << *op.target->global
                           << ". key: " << op.key << "\n";
                return {};
            }
            auto *var_decl = function_builder().global_var_list.get().at(*op.target->global);
            fn_ptr_expr    = clang::DeclRefExpr::Create(
                ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), var_decl, false,
                clang::SourceLocation(), var_decl->getType(), clang::VK_LValue
            );
        } else if (op.target->operation) {
            // Target is a computed expression (temporary)
            auto maybe_operation = operationFromKey(function, *op.target->operation);
            if (!maybe_operation) {
                LOG(ERROR) << "CALLIND target operation not found. key: " << op.key << "\n";
                return {};
            }
            auto [stmt, _] = function_builder().create_operation(ctx, *maybe_operation);
            fn_ptr_expr    = clang::dyn_cast< clang::Expr >(stmt);
        } else {
            LOG(ERROR) << "CALLIND target missing global or operation. key: " << op.key << "\n";
            return {};
        }

        if (fn_ptr_expr == nullptr) {
            LOG(ERROR) << "Failed to create function pointer expression. key: " << op.key << "\n";
            return {};
        }

        // 3. Infer function pointer type and cast if needed
        auto expr_type = fn_ptr_expr->getType();

        // Check if we need to build an explicit function pointer type
        const bool needs_type_inference =
            !expr_type->isPointerType() || !expr_type->getPointeeType()->isFunctionType();

        if (needs_type_inference) {
            // Infer parameter types from inputs
            std::vector< clang::QualType > param_types;
            for (const auto &input : op.inputs) {
                if (type_builder().get_serialized_types().contains(input.type_key)) {
                    param_types.push_back(type_builder().get_serialized_types().at(input.type_key));
                } else {
                    param_types.push_back(ctx.IntTy); // Fallback
                }
            }

            // Infer return type from op.type
            clang::QualType return_type = ctx.VoidTy;
            if (op.type && type_builder().get_serialized_types().contains(*op.type)) {
                return_type = type_builder().get_serialized_types().at(*op.type);
            }

            // Build function type and pointer type
            const clang::FunctionProtoType::ExtProtoInfo epi;
            auto func_type     = ctx.getFunctionType(return_type, param_types, epi);
            auto func_ptr_type = ctx.getPointerType(func_type);

            // Cast callee to inferred function pointer type
            fn_ptr_expr = make_cast(ctx, fn_ptr_expr, func_ptr_type, op_loc);
            if (fn_ptr_expr == nullptr) {
                LOG(ERROR) << "Failed to cast to function pointer type. key: " << op.key << "\n";
                return {};
            }
        }

        // 4. Build argument list from inputs
        std::vector< clang::Expr * > arguments;
        for (const auto &input : op.inputs) {
            auto *arg_expr =
                clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, input));
            if (arg_expr != nullptr) {
                arguments.push_back(arg_expr);
            }
        }

        // 5. Build the indirect call expression
        auto result = sema().BuildCallExpr(nullptr, fn_ptr_expr, op_loc, arguments, op_loc);
        if (result.isInvalid()) {
            LOG(ERROR) << "Failed to build CALLIND expression. key: " << op.key << "\n";
            return {};
        }
        auto *call_expr = result.getAs< clang::Expr >();

        // 6. Handle return value assignment if present
        if (!op.output || !op.has_return_value.value_or(false)) {
            return std::make_pair(call_expr, false);
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));
        return { create_assign_operation(ctx, call_expr, output_expr, op_loc), false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_callother(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (!op.target || op.target->kind != Varnode::VARNODE_INTRINSIC || !op.target->function) {
            LOG(ERROR) << "Invalid CALLOTHER intrinsic target. key: " << op.key << "\n";
            return {};
        }

        const auto &label = *op.target->function;
        auto name         = parse_intrinsic_name(label);

        // Look up specific handler
        const auto &handlers = get_intrinsic_handlers();
        auto it              = handlers.find(name);
        if (it != handlers.end()) {
            return it->second(*this, ctx, function, op);
        }

        // Fallback: use __patchestry_missing_<name> for unrecognized intrinsics
        // This includes cases where Ghidra couldn't determine the actual intrinsic name
        // (e.g., "stringdata" is a placeholder name)
        // The function declaration includes an AnnotateAttr with metadata for debugging
        return create_missing_intrinsic_call(ctx, function, op, name, label);
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
        unsigned low_width = ctx.getIntWidth(ctx.IntTy);
        auto merge_to_next = !op.output.has_value();
        auto *shift_value  = clang::IntegerLiteral::Create(
            ctx, llvm::APInt(ctx.getIntWidth(ctx.IntTy), low_width), ctx.IntTy, location
        );

        // If Operation has type, convert expression to operation type and perform bit-shift and
        // or operation.
        if (op.type) {
            auto op_type           = type_builder().get_serialized_types().at(*op.type);
            auto *cast_expr_input0 = make_cast(ctx, input0_expr, op_type, location);
            assert(cast_expr_input0 != nullptr && "Failed to create cast expression");
            input0_expr            = cast_expr_input0;
            auto *cast_expr_input1 = make_cast(ctx, input1_expr, op_type, location);
            assert(cast_expr_input1 != nullptr && "Failed to create cast expression");
            input1_expr = cast_expr_input1;
        }

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
                op_loc, ctx.getTrivialTypeSourceInfo(target_type), op_loc, input_expr
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
        if (op.inputs.empty() || op.inputs.size() != 2U) {
            LOG(ERROR) << "Skipping, carry operation due to input operands. key: " << op.key
                       << "\n";
            return {};
        }

        auto op_loc = sourceLocation(ctx.getSourceManager(), op.key);
        auto *input0 =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));
        auto *input1 =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));

        auto sum =
            sema().BuildBinOp(sema().getCurScope(), op_loc, clang::BO_Add, input0, input1);
        assert(!sum.isInvalid() && "Failed to create add operation");

        auto carry = sema().BuildBinOp(
            sema().getCurScope(), op_loc, clang::BO_LT, sum.getAs< clang::Expr >(), input0
        );
        assert(!carry.isInvalid() && "Failed to create carry operation");

        if (!op.output.has_value()) {
            return { carry.getAs< clang::Stmt >(), true };
        }

        auto *output =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(ctx, carry.getAs< clang::Expr >(), output, op_loc),
                 false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_int_scarry(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.empty() || op.inputs.size() != 2U) {
            LOG(ERROR) << "Skipping, scarry operation due to input operands. key: " << op.key
                       << "\n";
            return {};
        }

        auto op_loc  = sourceLocation(ctx.getSourceManager(), op.key);
        auto *input0 =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));
        auto *input1 =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));

        // Compute sum = input0 + input1
        auto sum =
            sema().BuildBinOp(sema().getCurScope(), op_loc, clang::BO_Add, input0, input1);
        assert(!sum.isInvalid() && "Failed to create add operation");

        // Signed overflow on addition occurs when both inputs have the same sign AND
        // the result has a different sign from the inputs.
        // Formula: (~(input0 ^ input1)) & (input0 ^ sum) < 0

        // xor_inputs = input0 ^ input1
        auto xor_inputs =
            sema().BuildBinOp(sema().getCurScope(), op_loc, clang::BO_Xor, input0, input1);
        assert(!xor_inputs.isInvalid() && "Failed to create xor operation");

        // not_xor_inputs = ~(input0 ^ input1)
        // MSB is 1 when both inputs have the same sign
        auto not_xor_inputs = sema().BuildUnaryOp(
            sema().getCurScope(), op_loc, clang::UO_Not, xor_inputs.getAs< clang::Expr >()
        );
        assert(!not_xor_inputs.isInvalid() && "Failed to create bitwise not operation");

        // xor_result = input0 ^ sum
        // MSB is 1 when the sum's sign differs from input0
        auto xor_result = sema().BuildBinOp(
            sema().getCurScope(), op_loc, clang::BO_Xor, input0,
            sum.getAs< clang::Expr >()
        );
        assert(!xor_result.isInvalid() && "Failed to create xor operation");

        // and_result = not_xor_inputs & xor_result
        auto and_result = sema().BuildBinOp(
            sema().getCurScope(), op_loc, clang::BO_And, not_xor_inputs.getAs< clang::Expr >(),
            xor_result.getAs< clang::Expr >()
        );
        assert(!and_result.isInvalid() && "Failed to create and operation");

        // scarry = and_result < 0
        auto *zero = clang::IntegerLiteral::Create(
            ctx, llvm::APInt(ctx.getIntWidth(input0->getType()), 0, true), input0->getType(),
            op_loc
        );
        auto scarry = sema().BuildBinOp(
            sema().getCurScope(), op_loc, clang::BO_LT, and_result.getAs< clang::Expr >(),
            zero
        );
        assert(!scarry.isInvalid() && "Failed to create scarry comparison");

        if (!op.output.has_value()) {
            return { scarry.getAs< clang::Stmt >(), true };
        }

        auto *output =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(ctx, scarry.getAs< clang::Expr >(), output, op_loc),
                 false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_int_sborrow(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.empty() || op.inputs.size() != 2U) {
            LOG(ERROR) << "Skipping, sborrow operation due to input operands. key: " << op.key
                       << "\n";
            return {};
        }

        auto op_loc  = sourceLocation(ctx.getSourceManager(), op.key);
        auto *input0 =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));
        auto *input1 =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[1]));

        // Compute diff = input0 - input1
        auto diff =
            sema().BuildBinOp(sema().getCurScope(), op_loc, clang::BO_Sub, input0, input1);
        assert(!diff.isInvalid() && "Failed to create sub operation");

        // Signed overflow occurs when input0 and input1 have different signs AND
        // input0 and diff have different signs.
        // Formula: ((input0 ^ input1) & (input0 ^ diff)) < 0

        // xor_inputs = input0 ^ input1
        auto xor_inputs =
            sema().BuildBinOp(sema().getCurScope(), op_loc, clang::BO_Xor, input0, input1);
        assert(!xor_inputs.isInvalid() && "Failed to create xor operation");

        // xor_result = input0 ^ diff
        auto xor_result = sema().BuildBinOp(
            sema().getCurScope(), op_loc, clang::BO_Xor, input0,
            diff.getAs< clang::Expr >()
        );
        assert(!xor_result.isInvalid() && "Failed to create xor operation");

        // and_result = xor_inputs & xor_result
        auto and_result = sema().BuildBinOp(
            sema().getCurScope(), op_loc, clang::BO_And, xor_inputs.getAs< clang::Expr >(),
            xor_result.getAs< clang::Expr >()
        );
        assert(!and_result.isInvalid() && "Failed to create and operation");

        // sborrow = and_result < 0
        auto *zero = clang::IntegerLiteral::Create(
            ctx, llvm::APInt(ctx.getIntWidth(input0->getType()), 0, true), input0->getType(),
            op_loc
        );
        auto sborrow = sema().BuildBinOp(
            sema().getCurScope(), op_loc, clang::BO_LT, and_result.getAs< clang::Expr >(),
            zero
        );
        assert(!sborrow.isInvalid() && "Failed to create sborrow comparison");

        if (!op.output.has_value()) {
            return { sborrow.getAs< clang::Stmt >(), true };
        }

        auto *output =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(ctx, sborrow.getAs< clang::Expr >(), output, op_loc),
                 false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_int_2comp(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        return create_unary_operation(ctx, function, op, clang::UnaryOperatorKind::UO_Minus);
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

        auto make_paren_expr = [&](clang::ASTContext &ctx, clang::Expr *expr,
                                   clang::SourceLocation loc) -> clang::Expr * {
            if (auto *uo = clang::dyn_cast< clang::UnaryOperator >(expr); uo) {
                auto *paren = new (ctx) clang::ParenExpr(loc, loc, expr);
                assert(paren != nullptr && "Failed to create paren expression.");
                return paren;
            }
            return expr;
        };

        auto result = sema().CreateBuiltinBinOp(
            op_loc, kind, make_paren_expr(ctx, lhs, op_loc), make_paren_expr(ctx, rhs, op_loc)
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
        if ((op.mnemonic != Mnemonic::OP_FLOAT_ABS) || (op.inputs.size() != 1U)) {
            LOG(ERROR) << "FLOAT_FLOOR operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return {};
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "FLOAT_FLOOR operation type is not serialized. key: " << op.key
                       << "\n";
            return {};
        }

        return create_builtin_call_expr(ctx, function, op, clang::Builtin::BIabs);
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_float_sqrt(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if ((op.mnemonic != Mnemonic::OP_FLOAT_SQRT) || (op.inputs.size() != 1U)) {
            LOG(ERROR) << "FLOAT_FLOOR operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return {};
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "FLOAT_FLOOR operation type is not serialized. key: " << op.key
                       << "\n";
            return {};
        }

        return create_builtin_call_expr(ctx, function, op, clang::Builtin::BIsqrt);
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_float_floor(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if ((op.mnemonic != Mnemonic::OP_FLOAT_FLOOR) || (op.inputs.size() != 1U)) {
            LOG(ERROR) << "FLOAT_FLOOR operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return {};
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "FLOAT_FLOOR operation type is not serialized. key: " << op.key
                       << "\n";
            return {};
        }

        return create_builtin_call_expr(ctx, function, op, clang::Builtin::BIfloorf);
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_float_ceil(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if ((op.mnemonic != Mnemonic::OP_FLOAT_CEIL) || (op.inputs.size() != 1U)) {
            LOG(ERROR) << "FLOAT_CEIL operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return {};
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "FLOAT_CEIL operation type is not serialized. key: " << op.key
                       << "\n";
            return {};
        }

        return create_builtin_call_expr(ctx, function, op, clang::Builtin::BIceilf);
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_float_round(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if ((op.mnemonic != Mnemonic::OP_FLOAT_ROUND) || (op.inputs.size() != 1U)) {
            LOG(ERROR) << "FLOAT_ROUND operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return {};
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "FLOAT_ROUND operation type is not serialized. key: " << op.key
                       << "\n";
            return {};
        }

        return create_builtin_call_expr(ctx, function, op, clang::Builtin::BIroundf);
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_popcount(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.size() != 1U || op.mnemonic != Mnemonic::OP_POPCOUNT) {
            LOG(ERROR) << "POPCOUNT operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return {};
        }

        // Use __builtin_popcount for population count
        return create_builtin_call_expr(
            ctx, function, op, clang::Builtin::BI__builtin_popcount
        );
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_lzcount(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if (op.inputs.size() != 1U || op.mnemonic != Mnemonic::OP_LZCOUNT) {
            LOG(ERROR) << "LZCOUNT operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return {};
        }

        // Use __builtin_clz for count leading zeros
        return create_builtin_call_expr(ctx, function, op, clang::Builtin::BI__builtin_clz);
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_builtin_call_expr(
        clang::ASTContext &ctx, const Function &function, const Operation &op,
        clang::Builtin::ID id
    ) {
        auto op_loc = sourceLocation(ctx.getSourceManager(), op.key);

        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        std::vector args = { input_expr };
        auto *call_expr = create_builtin_call(ctx, sema(), id, args, op_loc);

        if (!op.output.has_value()) {
            return { call_expr, true };
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));
        return { create_assign_operation(ctx, call_expr, output_expr), false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_int2float(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if ((op.mnemonic != Mnemonic::OP_INT2FLOAT) || (op.inputs.size() != 1U)) {
            LOG(ERROR) << "INT2FLOAT operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return {};
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "INT2FLOAT operation type is not serialized. key: " << op.key << "\n";
            return {};
        }

        const auto &op_type = type_builder().get_serialized_types().at(*op.type);
        auto op_loc         = sourceLocation(ctx.getSourceManager(), op.key);

        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        if (!input_expr->isPRValue()) {
            input_expr = make_implicit_cast(
                ctx, input_expr, input_expr->getType(), clang::CastKind::CK_LValueToRValue
            );
        }

        auto *cast_expr =
            make_implicit_cast(ctx, input_expr, op_type, clang::CK_IntegralToFloating);
        if (!op.output.has_value()) {
            return { cast_expr, true };
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(ctx, cast_expr, output_expr, op_loc), false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_float_nan(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if ((op.inputs.size() != 1U) || (op.mnemonic != Mnemonic::OP_FLOAT_NAN)) {
            LOG(ERROR) << "FLOAT_NAN operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return {};
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "FLOAT_NAN operation type is not serialized. key: " << op.key << "\n";
            return {};
        }

        auto nan_value = llvm::APFloat::getQNaN(llvm::APFloat::IEEEsingle());
        auto *nan_expr = clang::FloatingLiteral::Create(
            ctx, nan_value, true, ctx.FloatTy, clang::SourceLocation()
        );

        auto op_loc = sourceLocation(ctx.getSourceManager(), op.key);

        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        auto nan_check =
            sema().BuildBinOp(sema().getCurScope(), op_loc, clang::BO_NE, input_expr, nan_expr);
        assert(!nan_check.isInvalid() && "Invalid nan check");
        if (!op.output.has_value()) {
            return { nan_check.getAs< clang::Stmt >(), true };
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(ctx, nan_check.get(), output_expr, op_loc), false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_float2float(
        clang::ASTContext &ctx, const Function &function, const Operation &op
    ) {
        if ((op.mnemonic != Mnemonic::OP_FLOAT2FLOAT) || (op.inputs.size() != 1U)) {
            LOG(ERROR) << "FLOAT2FLOAT operation is invalid or has invalid input operand. key: "
                       << op.key << "\n";
            return {};
        }

        if (!type_builder().get_serialized_types().contains(*op.type)) {
            LOG(ERROR) << "FLOAT2FLOAT operation type is not serialized. key: " << op.key
                       << "\n";
            return {};
        }

        const auto &op_type = type_builder().get_serialized_types().at(*op.type);
        auto op_loc         = sourceLocation(ctx.getSourceManager(), op.key);

        auto *input_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0]));

        if (!input_expr->isPRValue()) {
            input_expr = make_implicit_cast(
                ctx, input_expr, input_expr->getType(), clang::CastKind::CK_LValueToRValue
            );
        }

        auto *cast_expr = make_implicit_cast(ctx, input_expr, op_type, clang::CK_FloatingCast);
        if (!op.output.has_value()) {
            return { cast_expr, true };
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(ctx, cast_expr, output_expr, op_loc), false };
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

    clang::Expr *OpBuilder::make_member_expr(
        clang::ASTContext &ctx, clang::Expr *base, unsigned offset, clang::SourceLocation loc
    ) {
        auto base_type = base->getType();
        if (!base_type->isPointerType() || !base_type->getPointeeType()->isRecordType()) {
            LOG(ERROR) << "Can't make member expre, base type is not pointer of record decl!\n";
            return nullptr;
        }

        auto find_field_decl = [&](clang::ASTContext &ctx, clang::RecordDecl *decl,
                                   unsigned int target_offset) -> clang::FieldDecl * {
            if (decl == nullptr || !decl->isCompleteDefinition()) {
                return nullptr;
            }

            const auto &layout = ctx.getASTRecordLayout(decl);
            for (auto *field : decl->fields()) {
                auto offset =
                    static_cast< unsigned int >(layout.getFieldOffset(field->getFieldIndex()));
                if (offset >= target_offset * 8U) {
                    return field;
                }
            }

            assert(false && "Failed to find field decl at offset, check!");
            return nullptr;
        };

        // Convert to rvalue if not there
        auto convert_rvalue = [&](clang::ASTContext &ctx, clang::Expr *expr) -> clang::Expr * {
            if (!expr->isPRValue()) {
                return make_implicit_cast(
                    ctx, expr, expr->getType(), clang::CastKind::CK_LValueToRValue
                );
            }
            return expr;
        };

        auto *pointee    = base_type->getPointeeType()->getAs< clang::RecordType >();
        auto *decl       = pointee->getDecl();
        // get the definition of record decl
        auto *definition = decl->getDefinition();

        auto *field = find_field_decl(ctx, definition, offset);
        assert(field != nullptr && "failed to find record decl field at offset");

        clang::DeclarationNameInfo member_name_info(field->getDeclName(), loc);
        return sema().BuildMemberExpr(
            convert_rvalue(ctx, base), true, loc, clang::NestedNameSpecifierLoc(),
            clang::SourceLocation(), field,
            clang::DeclAccessPair::make(field, clang::AS_public), false, member_name_info,
            field->getType(), clang::VK_LValue, clang::OK_Ordinary
        );
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
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, op.inputs[0], op_loc));

        clang::Expr *ptr_expr = nullptr;

        auto input_type = input_expr->getType();
        if (input_type->isPointerType() && input_type->getPointeeType()->isRecordType()) {
            auto *mem_expr = make_member_expr(ctx, input_expr, *op.inputs[1].value);
            auto *addrof_expr =
                sema()
                    .BuildUnaryOp(sema().getCurScope(), op_loc, clang::UO_AddrOf, mem_expr)
                    .get();
            ptr_expr = addrof_expr;
        } else {
            auto *byte_offset = clang::dyn_cast< clang::Expr >(
                create_varnode(ctx, function, op.inputs[1], op_loc)
            );

            auto add_result = sema().CreateBuiltinBinOp(
                op_loc, clang::BO_Add, make_cast(ctx, input_expr, op_type, op_loc), byte_offset
            );
            assert(!add_result.isInvalid() && "Invalid ptr_sub expression");
            ptr_expr = add_result.getAs< clang::Expr >();
        }

        if (merge_to_next) {
            return { ptr_expr, true };
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));
        return { create_assign_operation(ctx, ptr_expr, output_expr, op_loc), false };
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

        auto make_cast_expr = [&](clang::ASTContext &ctx, clang::Expr *expr,
                                  clang::QualType to_type,
                                  clang::SourceLocation loc) -> clang::Expr * {
            auto input_type = expr->getType();
            if (shouldReinterpretCast(input_type, to_type)) {
                return make_reinterpret_cast(ctx, expr, to_type, loc);
            }

            return make_explicit_cast(ctx, expr, to_type, loc);
        };

        auto *cast = make_cast_expr(ctx, input_expr, op_type, op_loc);
        assert(cast != nullptr && "failed to create vast expression");

        if (!op.output) {
            return { cast, true };
        }

        auto *output_expr =
            clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));

        return { create_assign_operation(ctx, cast, output_expr, op_loc), false };
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

    clang::FunctionDecl *OpBuilder::get_or_create_intrinsic_decl(
        clang::ASTContext &ctx, const std::string &name, clang::QualType return_type,
        bool is_variadic
    ) {
        // Check cache first
        auto it = intrinsic_decls.find(name);
        if (it != intrinsic_decls.end()) {
            return it->second;
        }

        // Create a new intrinsic function declaration
        auto loc = clang::SourceLocation();

        // Create function type with variadic signature if requested
        clang::FunctionProtoType::ExtProtoInfo epi;
        epi.Variadic = is_variadic;

        auto func_type = ctx.getFunctionType(return_type, {}, epi);

        auto *func_decl = clang::FunctionDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), loc, loc, &ctx.Idents.get(name), func_type,
            ctx.getTrivialTypeSourceInfo(func_type), clang::SC_Extern
        );

        if (func_decl == nullptr) {
            LOG(ERROR) << "Failed to create intrinsic function declaration for: " << name << "\n";
            return nullptr;
        }

        // Add to translation unit
        ctx.getTranslationUnitDecl()->addDecl(func_decl);

        // Cache it
        intrinsic_decls[name] = func_decl;
        return func_decl;
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_intrinsic_call(
        clang::ASTContext &ctx, const Function &function, const Operation &op,
        const std::string &name
    ) {
        auto op_loc = sourceLocation(ctx.getSourceManager(), op.key);

        // Determine return type
        clang::QualType ret_type = ctx.VoidTy;
        if (op.output) {
            ret_type = get_varnode_type(ctx, *op.output);
        }

        // Get or create function declaration
        auto *fn_decl = get_or_create_intrinsic_decl(ctx, name, ret_type, true);
        if (fn_decl == nullptr) {
            LOG(ERROR) << "Failed to get intrinsic declaration. key: " << op.key << "\n";
            return {};
        }

        // Build arguments from inputs
        std::vector< clang::Expr * > args;
        for (const auto &input : op.inputs) {
            auto *e = clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, input));
            if (e != nullptr) {
                args.push_back(e);
            }
        }

        // Build call expression
        auto *fn_ref = clang::DeclRefExpr::Create(
            ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), fn_decl, false, op_loc,
            fn_decl->getType(), clang::VK_LValue
        );

        auto result = sema().BuildCallExpr(nullptr, fn_ref, op_loc, args, op_loc);
        if (result.isInvalid()) {
            LOG(ERROR) << "Failed to build intrinsic call expression. key: " << op.key << "\n";
            return {};
        }

        auto *call_expr = result.getAs< clang::Expr >();

        // If no output, just return the call
        if (!op.output) {
            return { call_expr, false };
        }

        // Assign result to output
        auto *out = clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));
        return { create_assign_operation(ctx, call_expr, out, op_loc), false };
    }

    std::pair< clang::Stmt *, bool > OpBuilder::create_missing_intrinsic_call(
        clang::ASTContext &ctx, const Function &function, const Operation &op,
        const std::string &original_name, const std::string &original_label
    ) {
        auto op_loc = sourceLocation(ctx.getSourceManager(), op.key);

        // Build descriptive function name: __patchestry_missing_<original_name>
        std::string func_name = "__patchestry_missing_" + original_name;

        // Determine return type
        clang::QualType ret_type = ctx.VoidTy;
        if (op.output) {
            ret_type = get_varnode_type(ctx, *op.output);
        }

        // Build metadata annotation string with useful debugging info
        // Format: "intrinsic:<label> addr:<key> ret:<type> args:<count>"
        std::string annotation = "intrinsic:" + original_label + " addr:" + op.key;
        if (op.type) {
            annotation += " ret:" + *op.type;
        }
        annotation += " args:" + std::to_string(op.inputs.size());

        // Check cache for existing declaration
        auto cache_it = intrinsic_decls.find(func_name);
        clang::FunctionDecl *fn_decl = nullptr;

        if (cache_it != intrinsic_decls.end()) {
            fn_decl = cache_it->second;
        } else {
            // Create function type with variadic signature
            clang::FunctionProtoType::ExtProtoInfo epi;
            epi.Variadic = true;

            auto func_type = ctx.getFunctionType(ret_type, {}, epi);

            fn_decl = clang::FunctionDecl::Create(
                ctx, ctx.getTranslationUnitDecl(), clang::SourceLocation(),
                clang::SourceLocation(), &ctx.Idents.get(func_name), func_type,
                ctx.getTrivialTypeSourceInfo(func_type), clang::SC_Extern
            );

            if (fn_decl == nullptr) {
                LOG(ERROR) << "Failed to create missing intrinsic declaration for: " << func_name
                           << "\n";
                return {};
            }

            // Add AnnotateAttr with metadata for debugging
            fn_decl->addAttr(clang::AnnotateAttr::Create(
                ctx, annotation, nullptr, 0, clang::SourceRange()
            ));

            // Add to translation unit and cache
            ctx.getTranslationUnitDecl()->addDecl(fn_decl);
            intrinsic_decls[func_name] = fn_decl;
        }

        // Build arguments from inputs
        std::vector< clang::Expr * > args;
        for (const auto &input : op.inputs) {
            auto *e = clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, input));
            if (e != nullptr) {
                args.push_back(e);
            }
        }

        // Build call expression
        auto *fn_ref = clang::DeclRefExpr::Create(
            ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), fn_decl, false, op_loc,
            fn_decl->getType(), clang::VK_LValue
        );

        auto result = sema().BuildCallExpr(nullptr, fn_ref, op_loc, args, op_loc);
        if (result.isInvalid()) {
            LOG(ERROR) << "Failed to build missing intrinsic call expression. key: " << op.key
                       << "\n";
            return {};
        }

        auto *call_expr = result.getAs< clang::Expr >();

        // If no output, just return the call
        if (!op.output) {
            return { call_expr, false };
        }

        // Assign result to output
        auto *out = clang::dyn_cast< clang::Expr >(create_varnode(ctx, function, *op.output));
        return { create_assign_operation(ctx, call_expr, out, op_loc), false };
    }

} // namespace patchestry::ast
