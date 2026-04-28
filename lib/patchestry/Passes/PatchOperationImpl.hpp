/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <string>

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>

namespace mlir {
    class Operation;
} // namespace mlir

namespace cir {
    class CallOp;
    class FuncOp;
} // namespace cir

namespace patchestry::passes {

    // Forward declarations
    class InstrumentationPass;
    struct PatchInformation;

    /**
     * @brief Implementation class for patch operations.
     *
     * This class provides static methods for applying patches to operations
     * in five modes: apply_before, apply_after, apply_at_entrypoint,
     * replace (for both CallOp and any op with results), and erase.
     */
    class PatchOperationImpl
    {
      public:
        /**
         * @brief Applies a patch before the target operation.
         *
         * This method inserts a call to the patch function immediately before the target
         * operation. It handles module symbol merging, argument preparation, and call creation.
         * The inserted call is added to the inline worklist if inlining is enabled.
         *
         * @param pass The instrumentation pass instance
         * @param op The target operation to be instrumented
         * @param patch The patch information containing the patch function details
         * @param patch_module The module containing the patch function
         * @param should_inline Whether or not to inline at application
         */
        static void applyBeforePatch(
            InstrumentationPass &pass, mlir::Operation *op, const PatchInformation &patch,
            mlir::ModuleOp patch_module, bool should_inline
        );

        /**
         * @brief Applies a patch after the target operation.
         *
         * This method inserts a call to the patch function immediately after the target
         * operation. It handles module symbol merging, argument preparation, and call creation.
         * The inserted call is added to the inline worklist if inlining is enabled.
         *
         * @param pass The instrumentation pass instance
         * @param op The target operation to be instrumented
         * @param patch The patch information containing the patch function details
         * @param patch_module The module containing the patch function
         * @param should_inline Whether or not to inline at application
         */
        static void applyAfterPatch(
            InstrumentationPass &pass, mlir::Operation *op, const PatchInformation &patch,
            mlir::ModuleOp patch_module, bool should_inline
        );

        /**
         * @brief Replaces a function call with a patch function call.
         *
         * This method completely replaces the original function call with a call to the
         * patch function. It preserves the original call's arguments and return types
         * while redirecting the call to the patch function.
         *
         * @param pass The instrumentation pass instance
         * @param call_op The original call operation to be replaced
         * @param patch The patch information containing the replacement function details
         * @param patch_module The module containing the patch function
         * @param should_inline Whether or not to inline at application
         */
        static void replaceCallWithPatch(
            InstrumentationPass &pass, cir::CallOp call_op, const PatchInformation &patch,
            mlir::ModuleOp patch_module, bool should_inline
        );

        /**
         * @brief Applies a patch at the entry block of the enclosing (caller) function.
         *
         * The matched call identifies which enclosing function to instrument; the patch
         * call is inserted right after the caller's alloca prologue so that all parameter
         * allocas and their initialization stores are in scope. OPERAND argument sources
         * are remapped to the enclosing function's block arguments; RETURN_VALUE and
         * CAPTURE sources are rejected because they are only defined at the call site.
         *
         * @param pass The instrumentation pass instance
         * @param call_op The matched call operation
         * @param patch The patch information containing the patch function details
         * @param patch_module The module containing the patch function
         * @param should_inline Whether or not to inline at application
         */
        static void applyPatchAtEntrypoint(
            InstrumentationPass &pass, cir::CallOp call_op,
            const PatchInformation &patch, mlir::ModuleOp patch_module, bool should_inline
        );

        /**
         * @brief Replaces any non-CallOp operation that has at least one result
         *        (e.g. cir.binop, cir.cmp, cir.load, cir.cast, cir.get_member,
         *        cir.unary, cir.ptr_stride) with a call to the patch function.
         *
         * The operation's operands are passed as arguments to the patch function,
         * and the original result is replaced by the call's return value.
         *
         * @param pass The instrumentation pass instance
         * @param op The operation to replace (must have at least one result)
         * @param patch The patch information containing the replacement function
         * @param patch_module The module containing the patch function
         * @param should_inline Whether or not to inline at application
         */
        static void replaceOperationWithPatch(
            InstrumentationPass &pass, mlir::Operation *op,
            const PatchInformation &patch, mlir::ModuleOp patch_module,
            bool should_inline
        );

        /**
         * @brief Erases the target operation without inserting any patch.
         *
         * Used by ERASE mode. If the op has live result uses, each used
         * result is first replaced with a typed default value (zero / null)
         * so dependent ops remain well-formed; if any result type has no
         * supported default, the op is left in place with an error.
         *
         * @param pass The InstrumentationPass driving the erase; needed so
         *             `inline_worklists` stays in sync when the op being
         *             erased was a previously-emitted patch call that a
         *             later action chained against.
         * @param op The operation to erase
         */
        static void eraseOperation(InstrumentationPass &pass, mlir::Operation *op);

        /**
         * @brief REWRITE mode: substitute the matched op with the inline
         * C fragment `expr`. Compiled via FragmentCompiler and
         * inlined at the matched op's site; `$IDENT` metavars bind to
         * `patch.captures`. `arch` is the Ghidra `lang` string
         * ("ARM:LE:32").
         */
        static void rewriteWithExpression(
            InstrumentationPass &pass, mlir::Operation *op,
            const PatchInformation &patch, const std::string &expr,
            const std::string &arch
        );

        /**
         * @brief REWRITE mode (stmt form): splice a C statement body
         * at the matched op's site (which must be 0-result).
         * Captures, storage-write validation, and arg_ref temp
         * materialisation match `rewriteWithExpression`'s void path.
         * MVP rejects `return X;` / `return;` bodies — marker → CFG
         * rewrite is a follow-up.
         */
        static void rewriteWithStatements(
            InstrumentationPass &pass, mlir::Operation *op,
            const PatchInformation &patch, const std::string &stmt,
            const std::string &arch
        );

      private:
        /**
         * @brief Ensures that the patch function is available in the target module.
         *
         * This method checks if the patch function is available in the target module. If not,
         * it loads the patch module and merges the patch function into the target module.
         *
         * @param pass The instrumentation pass instance
         * @param target_module The target module to check
         * @param patch_module The patch module to load
         * @param patch_function_name The name of the patch function to check
         * @param context_label The context label to use for logging
         * @return The patch function if it is available, otherwise an empty cir::FuncOp
         */
        static cir::FuncOp ensurePatchFunctionAvailable(
            InstrumentationPass &pass, mlir::ModuleOp target_module,
            mlir::ModuleOp patch_module, const std::string &patch_function_name,
            llvm::StringRef context_label = {}
        );
    };

} // namespace patchestry::passes
