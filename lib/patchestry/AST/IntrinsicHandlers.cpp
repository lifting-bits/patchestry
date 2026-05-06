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

#include <array>
#include <cctype>
#include <optional>

#include <patchestry/AST/OperationBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {

    namespace {

        bool starts_with(std::string_view value, std::string_view prefix) {
            return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
        }

        bool ends_with(std::string_view value, std::string_view suffix) {
            return value.size() >= suffix.size()
                && value.substr(value.size() - suffix.size()) == suffix;
        }

        std::string to_lower_ascii(std::string_view value) {
            std::string result;
            result.reserve(value.size());
            for (auto c : value) {
                result.push_back(
                    static_cast< char >(std::tolower(static_cast< unsigned char >(c)))
                );
            }
            return result;
        }

        bool is_hex_digit(char c) {
            return std::isxdigit(static_cast< unsigned char >(c)) != 0;
        }

        bool looks_like_serializer_type_id(std::string_view suffix) {
            if (suffix.size() < 2 || suffix[0] != 't') { return false; }

            for (auto c : suffix.substr(1)) {
                if (!is_hex_digit(c)) { return false; }
            }

            return true;
        }

        bool looks_like_type_suffix(std::string_view suffix) {
            if (suffix.empty()) { return false; }

            if (looks_like_serializer_type_id(suffix)) { return true; }

            auto lower_suffix = to_lower_ascii(suffix);
            return lower_suffix == "void" || lower_suffix == "bool"
                || starts_with(lower_suffix, "type")
                || lower_suffix.find("int") != std::string::npos
                || lower_suffix.find("ptr") != std::string::npos;
        }

        std::string strip_return_type_suffix(std::string_view label) {
            auto colon_pos = label.rfind(':');
            if (colon_pos != std::string_view::npos) {
                auto suffix = label.substr(colon_pos + 1);
                if (looks_like_type_suffix(suffix)) {
                    return std::string(label.substr(0, colon_pos));
                }
            }

            auto underscore_pos = label.rfind('_');
            if (underscore_pos != std::string_view::npos) {
                auto suffix = label.substr(underscore_pos + 1);
                if (looks_like_type_suffix(suffix)) {
                    return std::string(label.substr(0, underscore_pos));
                }
            }

            return std::string(label);
        }

        std::string_view strip_trailing_size(std::string_view value) {
            while (!value.empty()
                   && std::isdigit(static_cast< unsigned char >(value.back())) != 0)
            {
                value.remove_suffix(1);
            }
            return value;
        }

        std::optional< std::string > normalize_aarch64_atomic_intrinsic(std::string_view name) {
            constexpr std::string_view prefix = "__aarch64_";
            if (!starts_with(name, prefix)) { return std::nullopt; }

            auto op    = name.substr(prefix.size());
            auto order = std::string_view("relaxed");

            if (ends_with(op, "_acq_rel")) {
                order = "acq_rel";
                op.remove_suffix(std::string_view("_acq_rel").size());
            } else if (ends_with(op, "_acq")) {
                order = "acquire";
                op.remove_suffix(std::string_view("_acq").size());
            } else if (ends_with(op, "_rel")) {
                order = "release";
                op.remove_suffix(std::string_view("_rel").size());
            }

            op = strip_trailing_size(op);

            std::string_view canonical_op;
            if (op == "ldadd") {
                canonical_op = "fetch_add";
            } else if (op == "ldclr") {
                canonical_op = "fetch_clear";
            } else if (op == "ldeor") {
                canonical_op = "fetch_xor";
            } else if (op == "ldset") {
                canonical_op = "fetch_or";
            } else if (op == "ldsmax") {
                canonical_op = "fetch_max";
            } else if (op == "ldsmin") {
                canonical_op = "fetch_min";
            } else if (op == "ldumax") {
                canonical_op = "fetch_umax";
            } else if (op == "ldumin") {
                canonical_op = "fetch_umin";
            } else if (op == "swp") {
                canonical_op = "exchange";
            } else if (op == "cas") {
                canonical_op = "compare_exchange";
            } else if (op == "casp") {
                canonical_op = "compare_exchange_pair";
            } else {
                return std::nullopt;
            }

            return "atomic_" + std::string(canonical_op) + "_" + std::string(order);
        }

        std::optional< std::string >
        normalize_x86_lock_atomic_intrinsic(std::string_view name) {
            auto lower_name = to_lower_ascii(name);
            std::string_view op(lower_name);

            if (starts_with(op, "__x86_lock_")) {
                op.remove_prefix(std::string_view("__x86_lock_").size());
            } else if (starts_with(op, "x86_lock_")) {
                op.remove_prefix(std::string_view("x86_lock_").size());
            } else if (starts_with(op, "lock_")) {
                op.remove_prefix(std::string_view("lock_").size());
            } else {
                return std::nullopt;
            }

            op = strip_trailing_size(op);

            std::string_view canonical_op;
            if (op == "add" || op == "xadd" || op == "inc") {
                canonical_op = "fetch_add";
            } else if (op == "sub" || op == "dec") {
                canonical_op = "fetch_sub";
            } else if (op == "and") {
                canonical_op = "fetch_and";
            } else if (op == "or") {
                canonical_op = "fetch_or";
            } else if (op == "xor") {
                canonical_op = "fetch_xor";
            } else if (op == "xchg") {
                canonical_op = "exchange";
            } else if (op == "cmpxchg") {
                canonical_op = "compare_exchange";
            } else {
                return std::nullopt;
            }

            return "atomic_" + std::string(canonical_op) + "_seq_cst";
        }

        std::string normalize_intrinsic_name(std::string_view name) {
            if (auto normalized = normalize_aarch64_atomic_intrinsic(name)) {
                return *normalized;
            }

            if (auto normalized = normalize_x86_lock_atomic_intrinsic(name)) {
                return *normalized;
            }

            return std::string(name);
        }

        std::pair< clang::Stmt *, bool > handle_generic_intrinsic_call(
            OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
            const ghidra::Operation &op
        ) {
            const auto &label = *op.target->function;
            auto name         = parse_intrinsic_name(label);
            return b.create_intrinsic_call(ctx, fn, op, name);
        }

        std::pair< clang::Stmt *, bool > handle_volatile_read(
            OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
            const ghidra::Operation &op
        ) {
            if (op.inputs.empty()) {
                LOG(ERROR) << "volatile_read requires at least one input. key: " << op.key
                           << "\n";
                return {};
            }

            auto op_loc = SourceLocation(ctx.getSourceManager(), op.key);

            // Get address expression
            auto *addr =
                clang::dyn_cast< clang::Expr >(b.create_varnode(ctx, fn, op.inputs[0]));
            if (addr == nullptr) {
                LOG(ERROR) << "Failed to create address expr for volatile_read. key: " << op.key
                           << "\n";
                return {};
            }

            // Determine result type from output
            clang::QualType result_type = ctx.IntTy;
            if (op.output) { result_type = b.get_varnode_type(ctx, *op.output); }

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
            if (!op.output) { return { deref.getAs< clang::Stmt >(), true }; }

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

            auto op_loc = SourceLocation(ctx.getSourceManager(), op.key);

            // Get address and value expressions
            auto *addr =
                clang::dyn_cast< clang::Expr >(b.create_varnode(ctx, fn, op.inputs[0]));
            auto *val = clang::dyn_cast< clang::Expr >(b.create_varnode(ctx, fn, op.inputs[1]));

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
            return handle_generic_intrinsic_call(b, ctx, fn, op);
        }

        void add_atomic_intrinsic_handlers(
            std::unordered_map< std::string, IntrinsicHandler > &handlers,
            std::string_view operation
        ) {
            static constexpr std::array< std::string_view, 5 > orders = { "relaxed", "acquire",
                                                                          "release", "acq_rel",
                                                                          "seq_cst" };

            for (auto order : orders) {
                handlers.emplace(
                    "atomic_" + std::string(operation) + "_" + std::string(order),
                    handle_generic_intrinsic_call
                );
            }
        }

    } // anonymous namespace

    std::string parse_intrinsic_name(std::string_view label) {
        return normalize_intrinsic_name(strip_return_type_suffix(label));
    }

    const std::unordered_map< std::string, IntrinsicHandler > &get_intrinsic_handlers() {
        static const auto handlers = [] {
            std::unordered_map< std::string, IntrinsicHandler > result = {
                {   "volatile_read",  handle_volatile_read },
                {  "volatile_write", handle_volatile_write },
                {  "builtin_memcpy", handle_builtin_memcpy },
                { "builtin_strncpy", handle_builtin_memcpy }, // Same impl as memcpy
                { "builtin_wcsncpy", handle_builtin_memcpy },
            };

            add_atomic_intrinsic_handlers(result, "fetch_add");
            add_atomic_intrinsic_handlers(result, "fetch_sub");
            add_atomic_intrinsic_handlers(result, "fetch_and");
            add_atomic_intrinsic_handlers(result, "fetch_clear");
            add_atomic_intrinsic_handlers(result, "fetch_or");
            add_atomic_intrinsic_handlers(result, "fetch_xor");
            add_atomic_intrinsic_handlers(result, "fetch_max");
            add_atomic_intrinsic_handlers(result, "fetch_min");
            add_atomic_intrinsic_handlers(result, "fetch_umax");
            add_atomic_intrinsic_handlers(result, "fetch_umin");
            add_atomic_intrinsic_handlers(result, "exchange");
            add_atomic_intrinsic_handlers(result, "compare_exchange");
            add_atomic_intrinsic_handlers(result, "compare_exchange_pair");
            return result;
        }();
        return handlers;
    }

} // namespace patchestry::ast
