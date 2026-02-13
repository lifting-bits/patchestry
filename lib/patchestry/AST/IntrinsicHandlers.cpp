/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/IntrinsicHandlers.hpp>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/OperationKinds.h>
#include <clang/Sema/Sema.h>
#include <llvm/Support/Casting.h>

#include <patchestry/AST/OperationBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {

    std::string parse_intrinsic_name(std::string_view label) {
        auto pos = label.rfind('_');
        if (pos != std::string_view::npos) {
            auto suffix = label.substr(pos + 1);
            // Check if suffix looks like a type
            if (suffix == "void" || suffix.find("int") != std::string_view::npos
                || suffix.find("ptr") != std::string_view::npos)
            {
                return std::string(label.substr(0, pos));
            }
        }
        return std::string(label);
    }

    namespace {

        std::pair< clang::Stmt *, bool > handle_volatile_read(
            OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
            const ghidra::Operation &op
        ) {
            if (op.inputs.empty()) {
                LOG(ERROR) << "volatile_read requires at least one input. key: " << op.key << "\n";
                return {};
            }

            auto op_loc = sourceLocation(ctx.getSourceManager(), op.key);

            // Get address expression
            auto *addr = clang::dyn_cast< clang::Expr >(b.create_varnode(ctx, fn, op.inputs[0]));
            if (addr == nullptr) {
                LOG(ERROR) << "Failed to create address expr for volatile_read. key: " << op.key
                           << "\n";
                return {};
            }

            // Determine result type from output
            clang::QualType result_type = ctx.IntTy;
            if (op.output) {
                result_type = b.get_varnode_type(ctx, *op.output);
            }

            // Cast to volatile pointer and dereference: *(volatile T*)addr
            auto vol_ptr = ctx.getPointerType(ctx.getVolatileType(result_type));
            auto *cast   = b.make_cast(ctx, addr, vol_ptr, op_loc);
            auto deref   = b.sema().CreateBuiltinUnaryOp(op_loc, clang::UO_Deref, cast);

            if (deref.isInvalid()) {
                LOG(ERROR) << "Failed to create dereference for volatile_read. key: " << op.key
                           << "\n";
                return {};
            }

            // If no output, return the deref expression directly
            if (!op.output) {
                return { deref.getAs< clang::Stmt >(), true };
            }

            // Assign result to output
            auto *out = clang::dyn_cast< clang::Expr >(b.create_varnode(ctx, fn, *op.output));
            return { b.create_assign_operation(ctx, deref.getAs< clang::Expr >(), out, op_loc),
                     false };
        }

        std::pair< clang::Stmt *, bool > handle_volatile_write(
            OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
            const ghidra::Operation &op
        ) {
            if (op.inputs.size() < 2) {
                LOG(ERROR) << "volatile_write requires at least two inputs. key: " << op.key
                           << "\n";
                return {};
            }

            auto op_loc = sourceLocation(ctx.getSourceManager(), op.key);

            // Get address and value expressions
            auto *addr = clang::dyn_cast< clang::Expr >(b.create_varnode(ctx, fn, op.inputs[0]));
            auto *val  = clang::dyn_cast< clang::Expr >(b.create_varnode(ctx, fn, op.inputs[1]));

            if (addr == nullptr || val == nullptr) {
                LOG(ERROR) << "Failed to create expressions for volatile_write. key: " << op.key
                           << "\n";
                return {};
            }

            // Cast to volatile pointer: (volatile T*)addr
            auto vol_ptr = ctx.getPointerType(ctx.getVolatileType(val->getType()));
            auto *cast   = b.make_cast(ctx, addr, vol_ptr, op_loc);

            // Dereference and assign: *(volatile T*)addr = val
            auto deref = b.sema().CreateBuiltinUnaryOp(op_loc, clang::UO_Deref, cast);
            if (deref.isInvalid()) {
                LOG(ERROR) << "Failed to create dereference for volatile_write. key: " << op.key
                           << "\n";
                return {};
            }

            auto assign = b.sema().CreateBuiltinBinOp(
                op_loc, clang::BO_Assign, deref.getAs< clang::Expr >(), val
            );

            if (assign.isInvalid()) {
                LOG(ERROR) << "Failed to create assign for volatile_write. key: " << op.key
                           << "\n";
                return {};
            }

            return { assign.getAs< clang::Stmt >(), false };
        }

        std::pair< clang::Stmt *, bool > handle_builtin_memcpy(
            OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
            const ghidra::Operation &op
        ) {
            // For memcpy/strncpy/wcsncpy, just use the generic fallback
            // which creates a function call with all inputs as arguments
            const auto &label = *op.target->function;
            auto name         = parse_intrinsic_name(label);
            return b.create_intrinsic_call(ctx, fn, op, name);
        }

    } // anonymous namespace

    const std::unordered_map< std::string, IntrinsicHandler > &get_intrinsic_handlers() {
        static const std::unordered_map< std::string, IntrinsicHandler > handlers = {
            { "volatile_read", handle_volatile_read },
            { "volatile_write", handle_volatile_write },
            { "builtin_memcpy", handle_builtin_memcpy },
            { "builtin_strncpy", handle_builtin_memcpy }, // Same impl as memcpy
            { "builtin_wcsncpy", handle_builtin_memcpy },
            // Add new handlers here
        };
        return handlers;
    }

} // namespace patchestry::ast
