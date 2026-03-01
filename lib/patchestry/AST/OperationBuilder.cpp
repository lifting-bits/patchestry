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
#include <llvm/Support/ErrorHandling.h>

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

        llvm_unreachable("Failed to find operation for varnode lookup key");
    }

    clang::Stmt *OpBuilder::create_varnode(
        clang::ASTContext &ctx, const Function &function, const Varnode &vnode,
        clang::SourceLocation loc
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
                case Varnode::VARNODE_STRING:
                    return create_string(ctx, vnode);
            }

            return nullptr;
        };

        if (auto *expr = varnode_operation(ctx, function, vnode)) {
            return expr;
        }

        (void) loc;

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

        // Already materialized as a named VarDecl — return a fresh DeclRefExpr.
        if (function_builder().local_variables.contains(*vnode.operation)) {
            auto *var_decl = function_builder().local_variables.at(*vnode.operation);
            return clang::DeclRefExpr::Create(
                ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), var_decl, false,
                clang::SourceLocation(), var_decl->getType(), clang::VK_LValue
            );
        }

        // Case 2: result cached in operation_stmts (merge_to_next op whose expression has
        // not yet been given a name).  Materialize it as a VarDecl so that every use site
        // gets a fresh DeclRefExpr instead of sharing the same Stmt* node across multiple
        // parent expressions (which violates the AST tree property).
        if (function_builder().operation_stmts.contains(*vnode.operation)) {
            return function_builder().operation_stmts.at(*vnode.operation);
        }

        // Forward reference — the defining op lives in a block not yet processed.
        // Build the operation once, cache it, then immediately materialize it as a VarDecl
        // via a recursive call (which will fall into Case 2).  This prevents re-execution
        // if create_temporary is called again for the same key before create_basic_block
        // reaches the defining block.
        if (auto maybe_operation = operationFromKey(function, vnode.operation.value())) {
            auto [stmt, _] = function_builder().create_operation(ctx, *maybe_operation);
            if (stmt) {
                function_builder().operation_stmts.emplace(*vnode.operation, stmt);
                // Recurse: will hit Case 1 (if already local) or Case 2.
                return create_temporary(ctx, function, vnode);
            }
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
            auto location       = sourceLocation(ctx.getSourceManager(), *vnode.function);
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
            auto op_loc    = sourceLocation(ctx.getSourceManager(), *vnode.operation);
            return clang::DeclRefExpr::Create(
                ctx, clang::NestedNameSpecifierLoc(), clang::SourceLocation(), var_decl, false,
                op_loc, var_decl->getType(), clang::VK_LValue
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

        if (vnode_type->isIntegralOrUnscopedEnumerationType()) {
            // ctx.getIntWidth() returns the value-bit width for the type, matching
            // what IntegerLiteral's internal assertion requires.  It resolves enum
            // types to their underlying integer type automatically.
            unsigned bit_width = ctx.getIntWidth(vnode_type);

            if (vnode_type->isEnumeralType()) {
                // For enum constants we keep a CStyleCastExpr to the enum type
                // because assigning a plain integer to an enum requires an explicit
                // cast in C.  The literal itself is typed as the enum's underlying
                // integer type so downstream pattern-matching (e.g. isAlwaysFalse)
                // reaches the IntegerLiteral without any IgnoreParenCasts workaround.
                auto underlying =
                    vnode_type->castAs< clang::EnumType >()->getDecl()->getIntegerType();
                auto *literal = new (ctx)
                    clang::IntegerLiteral(ctx, llvm::APInt(bit_width, *vnode.value), underlying, location);
                auto result = sema().BuildCStyleCastExpr(
                    location, ctx.getTrivialTypeSourceInfo(vnode_type), location, literal
                );
                assert(!result.isInvalid());
                return result.getAs< clang::Expr >();
            }

            // Plain integer: create the literal at the exact target width and type.
            // This eliminates the CStyleCastExpr wrapper that previously obscured
            // constant values from normalization-pass pattern matchers and silently
            // truncated constants narrower than 32 bits (e.g. uint8_t) or zero-
            // extended constants wider than 32 bits (e.g. uint64_t on 64-bit targets).
            auto apint = llvm::APInt(bit_width, *vnode.value);

            // Represent unsigned all-ones constants with the signed equivalent type
            // so they print as -1 rather than the large unsigned decimal (e.g.
            // 4294967295U).  The all-ones bit pattern is the canonical binary encoding
            // of -1 and almost always denotes an error sentinel in firmware code.
            // Signed/unsigned comparison semantics are preserved because C implicitly
            // converts -1 to UINT_MAX when comparing with an unsigned operand.
            clang::QualType literal_type = vnode_type;
            if (vnode_type->isUnsignedIntegerType() && apint.isAllOnes()) {
                auto signed_type = ctx.getIntTypeForBitwidth(bit_width, /*isSigned=*/1);
                if (!signed_type.isNull()) {
                    literal_type = signed_type;
                }
            }

            // For types narrower than int (e.g. unsigned char, short), Clang's
            // printer emits MSVC-specific suffixes like Ui8 / Ui16 which are not
            // valid standard C.  Promote the literal to int / unsigned int width
            // and wrap in an invisible ImplicitCastExpr back to the narrow type.
            unsigned int_width = ctx.getIntWidth(ctx.IntTy);
            if (bit_width < int_width) {
                bool lit_unsigned = literal_type->isUnsignedIntegerType();
                auto wide_type   = lit_unsigned ? ctx.UnsignedIntTy : ctx.IntTy;
                auto wide_val    = lit_unsigned ? apint.zext(int_width)
                                               : apint.sext(int_width);
                auto *literal =
                    new (ctx) clang::IntegerLiteral(ctx, wide_val, wide_type, location);
                return make_implicit_cast(
                    ctx, literal, vnode_type, clang::CK_IntegralCast
                );
            }

            return new (ctx) clang::IntegerLiteral(ctx, apint, literal_type, location);
        }

        if (vnode_type->isVoidType()) {
            // Void-typed constants are unusual; keep them as a cast from int so the
            // resulting expression is at least well-formed.
            auto *literal = new (ctx)
                clang::IntegerLiteral(ctx, llvm::APInt(32U, *vnode.value), ctx.IntTy, location);
            auto result = sema().BuildCStyleCastExpr(
                location, ctx.getTrivialTypeSourceInfo(vnode_type), location, literal
            );
            assert(!result.isInvalid());
            return result.getAs< clang::Expr >();
        }

        if (vnode_type->isPointerType()) {
            // Use the target's pointer-integer width so that pointer constants are
            // not truncated on 64-bit targets.
            unsigned ptr_bits = ctx.getIntWidth(ctx.getUIntPtrType());
            auto *literal     = new (ctx) clang::IntegerLiteral(
                ctx, llvm::APInt(ptr_bits, *vnode.value), ctx.getUIntPtrType(), location
            );
            auto result = sema().BuildCStyleCastExpr(
                location, ctx.getTrivialTypeSourceInfo(vnode_type), location, literal
            );
            assert(!result.isInvalid());
            return result.getAs< clang::Expr >();
        }

        if (vnode_type->isFloatingType()) {
            // Select the APFloat semantics that match the actual target float type
            // so that the bit pattern stored in vnode.value is interpreted correctly.
            const llvm::fltSemantics &sem = ctx.getFloatTypeSemantics(vnode_type);
            unsigned float_bits           = static_cast<unsigned>(ctx.getTypeSize(vnode_type));
            llvm::APFloat float_value(sem, llvm::APInt(float_bits, *vnode.value));
            return clang::FloatingLiteral::Create(ctx, float_value, true, vnode_type, location);
        }

        return {};
    }

    clang::Stmt *OpBuilder::create_string(clang::ASTContext &ctx, const Varnode &vnode) {
        if (vnode.kind != Varnode::VARNODE_STRING) {
            LOG(ERROR) << "Varnode is not string, invalid varnode.\n";
            return {};
        }

        if (!vnode.string_value) {
            LOG(ERROR) << "No string value found, invalid varnode.\n";
#ifdef ENABLE_DEBUG
            LOG(ERROR) << vnode.dump() << "\n";
#endif
            return {};
        }

        // Determine if this is a wide string based on the type
        bool is_wide = false;
        if (!vnode.type_key.empty()) {
            if (type_builder().get_serialized_types().contains(vnode.type_key)) {
                auto type = type_builder().get_serialized_types().at(vnode.type_key);
                is_wide   = type->isWideCharType();
            }
        }

        // For empty string, we still need an array of size 1 for the null terminator
        const size_t string_length = vnode.string_value->length();
        const size_t array_size    = string_length + 1; // +1 for null terminator

        // Create the appropriate array type
        auto char_type    = is_wide ? ctx.WideCharTy : ctx.CharTy;
        auto string_array = ctx.getConstantArrayType(
            char_type.withConst(), llvm::APInt(32, array_size), nullptr,
            clang::ArraySizeModifier::Normal, 0
        );

        return clang::StringLiteral::Create(
            ctx,
            *vnode.string_value, // Use the string value directly
            is_wide ? clang::StringLiteralKind::Wide : clang::StringLiteralKind::Ordinary,
            false, string_array, clang::SourceLocation()
        );
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
