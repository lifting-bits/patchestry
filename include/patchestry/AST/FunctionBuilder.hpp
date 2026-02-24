/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclBase.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Sema/Sema.h>

#include <patchestry/AST/TypeBuilder.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>

namespace patchestry::ast {
    class OpBuilder;
}

namespace patchestry::ast {
    using namespace patchestry::ghidra;

    class [[nodiscard]] FunctionBuilder : public std::enable_shared_from_this< FunctionBuilder >
    {
        friend OpBuilder;

      public:
        FunctionBuilder(
            clang::CompilerInstance &ci, const Function &function, TypeBuilder &type_builder,
            std::unordered_map< std::string, clang::FunctionDecl * > &functions,
            std::unordered_map< std::string, clang::VarDecl * > &globals
        );

        // copy operations
        FunctionBuilder(const FunctionBuilder &)            = default;
        FunctionBuilder &operator=(const FunctionBuilder &) = delete;

        // move operations
        FunctionBuilder(FunctionBuilder &&) noexcept            = default;
        FunctionBuilder &operator=(FunctionBuilder &&) noexcept = delete;

        // Virtual destructor
        virtual ~FunctionBuilder() = default;

        void initialize_op_builder(void);

        /**
         * @brief Creates a `FunctionDecl` object for the given function, including its type,
         * parameters, and adds it to the translation unit declaration context.
         *
         * @param ctx Reference to the `clang::ASTContext`.
         * @param is_definition Boolean flag indicating whether the function is a definition
         * (`true`) or just a declaration (`false`).
         *
         * @return A pointer to the created `clang::FunctionDecl` object. Returns `nullptr` if
         * the creation fails due to errors such as an empty function name or an invalid type.
         */
        clang::FunctionDecl *create_declaration(
            clang::ASTContext &ctx, const clang::QualType &function_type,
            bool is_definition = false
        );

        /**
         * @brief Creates a function type based on the provided function prototype.
         *
         * @param ctx A reference to the Clang AST context.
         * @param proto The function prototype, containing:
         *              - `rttype_key`: The key representing the return type.
         *              - `parameters`: A list of keys for the parameter types.
         *              - `is_variadic`: Boolean indicating if the function is variadic.
         *              - `is_noreturn`: Boolean indicating if the function has a `noreturn`
         * specifier.
         *
         * @return A `clang::QualType` representing the function type.
         */
        clang::QualType
        create_function_type(clang::ASTContext &ctx, const FunctionPrototype &proto);

        std::vector< clang::ParmVarDecl * > create_default_paramaters(
            clang::ASTContext &ctx, clang::FunctionDecl *func_decl,
            const FunctionPrototype &proto
        );

        clang::FunctionDecl *create_definition(clang::ASTContext &ctx);

      private:
        void create_labels(clang::ASTContext &ctx, clang::FunctionDecl *func_decl);

        std::vector< clang::Stmt * >
        create_function_body(clang::ASTContext &ctx, clang::FunctionDecl *func_decl);

        std::vector< clang::Stmt * >
        create_basic_block(clang::ASTContext &ctx, const BasicBlock &block);

        std::pair< clang::Stmt *, bool >
        create_operation(clang::ASTContext &ctx, const Operation &op);

        void set_sema_context(clang::DeclContext *dc) { sema().CurContext = dc; }

        clang::DeclContext *get_sema_context(void) { return sema().CurContext; }

        clang::Sema &sema(void) const { return cii.get().getSema(); }

        clang::FunctionDecl *prev_decl;
        std::reference_wrapper< const clang::CompilerInstance > cii;
        std::reference_wrapper< const Function > function;
        std::reference_wrapper< TypeBuilder > type_builder;
        std::shared_ptr< OpBuilder > op_builder;

        std::reference_wrapper< std::unordered_map< std::string, clang::FunctionDecl * > >
            function_list;
        std::reference_wrapper< std::unordered_map< std::string, clang::VarDecl * > >
            global_var_list;

        std::unordered_map< std::string, clang::VarDecl * > local_variables;
        std::unordered_map< std::string, clang::LabelDecl * > labels_declaration;
        std::unordered_map< std::string, clang::Stmt * > operation_stmts;

        // Statements queued by create_temporary during an operation build that must be
        // emitted immediately before the operation that triggered them.  Drained by
        // create_basic_block after each create_operation call.
        std::vector< clang::Stmt * > pending_materialized;
    };
} // namespace patchestry::ast
