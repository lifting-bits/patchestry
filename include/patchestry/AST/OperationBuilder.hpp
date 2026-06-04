/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <string_view>
#include <unordered_set>

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

        // Coerce a STORE address operand into a dereferenceable pointer (see
        // definition).  Returns the operand unchanged when it is already a
        // non-void pointer, or nullptr if the cast cannot be built.
        clang::Expr *coerce_store_address(
            clang::ASTContext &ctx, clang::Expr *addr, clang::Expr *value,
            const Operation &op, clang::SourceLocation op_loc
        );

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

        std::pair< clang::Stmt *, bool >
        create_callother(clang::ASTContext &ctx, const Function &function, const Operation &op);

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

        std::pair< clang::Stmt *, bool >
        create_popcount(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_lzcount(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool > create_builtin_call_expr(
            clang::ASTContext &ctx, const Function &function, const Operation &op,
            clang::Builtin::ID id
        );

        // Methods exposed for CALLOTHER intrinsic handlers.
        //
        // narrow_to_size_hint: when false, suppresses the
        // resolution-time aggregate-to-integer narrowing (#225).  Pass
        // false from callers whose op needs the raw storage lvalue
        // (e.g. ADDRESS_OF).
        clang::Stmt *create_varnode(
            clang::ASTContext &ctx, const Function &function, const Varnode &vnode,
            clang::SourceLocation loc      = clang::SourceLocation(),
            bool narrow_to_size_hint       = true
        );

        clang::QualType get_varnode_type(clang::ASTContext &ctx, const Varnode &vnode);

        clang::Expr *make_cast(
            clang::ASTContext &ctx, clang::Expr *expr, const clang::QualType &to_type,
            clang::SourceLocation loc
        );

        clang::Stmt *create_assign_operation(
            clang::ASTContext &ctx, clang::Expr *input_expr, clang::Expr *output_expr,
            clang::SourceLocation loc = clang::SourceLocation()
        );

        clang::Sema &sema(void) { return function_builder().sema(); }

        // CALLOTHER intrinsic helper - exposed for custom handlers
        std::pair< clang::Stmt *, bool > create_intrinsic_call(
            clang::ASTContext &ctx, const Function &function, const Operation &op,
            const std::string &name
        );

        // Build a call against the FunctionDecl the Ghidra serialiser
        // registered for `op.target->function` in the JSON `functions`
        // table.  Preferred over create_missing_intrinsic_call when the
        // intrinsic is known: variadic signature absorbs any arg-type
        // mismatches without synthesising a parallel placeholder.
        std::pair< clang::Stmt *, bool > build_intrinsic_call_against_registered(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool > create_tail_call(
            clang::ASTContext &ctx, const Function &function, const Operation &op,
            clang::FunctionDecl *enclosing_decl
        );

        // Create a call to __patchestry_missing_<name> with metadata annotation
        std::pair< clang::Stmt *, bool > create_missing_intrinsic_call(
            clang::ASTContext &ctx, const Function &function, const Operation &op,
            const std::string &original_name, const std::string &original_label
        );

        // Build `__patchestry_error("reason")`. Return type is `long long`
        // so the caller can cast to any scalar/pointer at the use site.
        clang::Expr *emit_patchestry_error(
            clang::ASTContext &ctx, const std::string &reason,
            clang::SourceLocation loc
        );

      private:
        clang::FunctionDecl *get_or_create_intrinsic_decl(
            clang::ASTContext &ctx, const std::string &name, clang::QualType return_type,
            llvm::ArrayRef< clang::QualType > param_types = {}
        );

        clang::Expr *build_callexpr_from_function(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        clang::Stmt *create_array_assignment_operation(
            clang::ASTContext &ctx, clang::Expr *input_expr, clang::Expr *output_expr,
            clang::SourceLocation loc = clang::SourceLocation()
        );

        clang::Expr *make_explicit_cast(
            clang::ASTContext &ctx, clang::Expr *expr, clang::QualType to_type,
            clang::SourceLocation loc
        );

        clang::Expr *make_implicit_cast(
            clang::ASTContext &ctx, clang::Expr *expr, clang::QualType to_type,
            clang::CastKind kind
        );

        clang::Expr *make_reinterpret_cast(
            clang::ASTContext &ctx, clang::Expr *expr, clang::QualType to_type,
            clang::SourceLocation loc
        );

        // Pointer→integer cast with explicit extension semantics (#224
        // follow-up).  A single (target)ptr lowers as ptrtoint+zext in
        // CIRGen; for sign-extension we route through intptr_t so the
        // widening step sees a signed source and emits sext.
        enum class PtrToIntExtension { kZero, kSign };
        clang::Expr *cast_pointer_to_int(
            clang::ASTContext &ctx, clang::Expr *ptr, clang::QualType target,
            clang::SourceLocation loc, PtrToIntExtension kind,
            std::string_view op_key
        );

        // Reinterpret a record operand as a same-width integer for C scalar
        // operators. Records wider than 128 bits are returned unchanged; pass
        // quiet_oversized (e.g. from ADDRESS_OF) to mute the "too large for
        // integer coercion" warning on that no-op path.
        clang::Expr *coerce_record_to_integer(
            clang::ASTContext &ctx, clang::Expr *expr, clang::SourceLocation loc,
            bool quiet_oversized = false
        );

        // Narrow an aggregate lvalue to an integer access (#225).  Tries
        // `expr[0]` when target_bytes matches the array's element width,
        // else `*(uintN_t*)&expr` at target_bytes (or the aggregate's
        // full size when target_bytes == 0).  Returns nullptr when
        // neither path is sound — caller should refuse loudly.
        clang::Expr *narrow_aggregate_to_integer(
            clang::ASTContext &ctx, clang::Expr *expr, clang::SourceLocation loc,
            unsigned target_bytes = 0
        );

        /// Materialize a non-void call result into a temporary variable.
        /// Creates VarDecl + DeclStmt (pushed to pending for hoisting),
        /// assignment (call stays in-place), and returns the appropriate
        /// pair for the caller to emit.
        std::pair< clang::Stmt *, bool > materialize_call_return(
            clang::ASTContext &ctx, clang::Expr *call_expr,
            clang::QualType ret_type, const Operation &op,
            clang::SourceLocation loc
        );

        clang::Expr *make_member_expr(
            clang::ASTContext &ctx, clang::Expr *base, unsigned offset,
            clang::SourceLocation loc = clang::SourceLocation()
        );

        // Pad `arguments` up to fndecl's minimum required count with synthesized
        // defaults.  Returns false if a required parameter has no representable
        // default (so the caller can drop the call instead of building one that
        // would fail Sema with "too few arguments").
        [[nodiscard]] bool extend_callexpr_agruments(
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

        clang::Stmt *create_string(clang::ASTContext &ctx, const Varnode &vnode);

        std::optional< clang::QualType >
        lookup_op_type(const Operation &op);

        TypeBuilder &type_builder(void) { return builder->type_builder.get(); }

        FunctionBuilder &function_builder(void) { return *builder; }

        std::unordered_map< std::string, clang::FunctionDecl * > &
        intrinsic_decls(void) { return builder->intrinsic_list.get(); }

        std::reference_wrapper< const clang::ASTContext > context;
        std::shared_ptr< FunctionBuilder > builder;

        // Cycle detection for create_temporary forward-reference resolution
        std::unordered_set< std::string > resolving_temporaries;
    };

} // namespace patchestry::ast
