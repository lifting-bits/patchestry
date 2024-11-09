/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include "patchestry/Ghidra/PcodeOperations.hpp"
#include "patchestry/Ghidra/PcodeTypes.hpp"
#include <clang/Basic/SourceLocation.h>
#include <clang/Sema/Sema.h>
#include <functional>
#include <llvm-18/llvm/Support/raw_ostream.h>
#include <memory>
#include <unordered_map>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>

#include <patchestry/AST/Codegen.hpp>
#include <patchestry/AST/TypeBuilder.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>

namespace patchestry::ast {
    using namespace patchestry::ghidra;

    using ASTTypeMap = std::unordered_map< std::string, clang::QualType >;
    using ASTDeclMap = std::unordered_map< std::string, clang::Decl * >;

    class PcodeASTConsumer : public clang::ASTConsumer
    {
      public:
        explicit PcodeASTConsumer(
            clang::CompilerInstance &ci, Program &prog, llvm::raw_ostream &out,
            llvm::raw_ostream &ast_out
        )
            : program(prog)
            , ci(ci)
            , out(out)
            , ast_out(ast_out)
            , codegen(std::make_unique< CodeGenerator >())
            , type_builder(std::make_unique< TypeBuilder >(ci.getASTContext())) {}

        void HandleTranslationUnit(clang::ASTContext &ctx) override;

      private:
        void set_sema_context(clang::DeclContext *dc);

        void create_globals(clang::ASTContext &ctx, VariableMap &serialized_variables);

        void create_functions(
            clang::ASTContext &ctx, FunctionMap &serialized_functions, TypeMap &serialized_types
        );

        clang::QualType
        create_function_prototype(clang::ASTContext &ctx, const FunctionPrototype &proto);

        std::vector< clang::ParmVarDecl * > create_default_paramaters(
            clang::ASTContext &ctx, clang::FunctionDecl *func_decl,
            const FunctionPrototype &proto
        );

        clang::FunctionDecl *create_function_declaration(
            clang::ASTContext &ctx, const Function &function, bool is_definition = false
        );

        clang::FunctionDecl *
        create_function_definition(clang::ASTContext &ctx, const Function &function);

        std::vector< clang::Stmt * >
        create_function_body(clang::ASTContext &ctx, const Function &function);

        void create_label_for_basic_blocks(clang::ASTContext &ctx, const Function &function);

        std::vector< clang::Stmt * > create_basic_block(
            clang::ASTContext &ctx, const Function &function, const BasicBlock &block
        );

        std::pair< clang::Stmt *, bool >
        create_operation(clang::ASTContext &ctx, const Function &function, const Operation &op);

        clang::DeclStmt *create_decl_stmt(clang::ASTContext &ctx, clang::Decl *decl);

        clang::Stmt *create_call_stmt(clang::ASTContext &ctx, const Operation &op);

        clang::Stmt *create_branch_stmt(clang::ASTContext &ctx, const Operation &branch);

        clang::Stmt *create_return_stmt(
            clang::ASTContext &ctx, const Function &function, const Operation &ret_op
        );

        clang::Stmt *create_varnode(
            clang::ASTContext &ctx, const Function &function, const Varnode &vnode,
            bool is_input = true
        );

        clang::Stmt *create_temporary(
            clang::ASTContext &ctx, const Function &function, const Varnode &vnode,
            bool is_input = true
        );

        // List of functions to generate AST node for Pcode operations

        // OP_DECLARE_LOCAL
        std::pair< clang::Stmt *, bool > create_declare_local(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        // OP_DECLARE_PARAMETER
        std::pair< clang::Stmt *, bool > create_declare_parameter(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool > create_declare_temporary(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool >
        create_copy(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_load(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_store(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_branch(clang::ASTContext &ctx, const Function &function, const Operation &op);
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
        create_int_equal(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool > create_int_notequal(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool >
        create_int_less(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_int_sless(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool > create_int_lessequal(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool > create_int_slessequal(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool >
        create_int_zext(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_int_sext(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_int_add(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_int_sub(clang::ASTContext &ctx, const Function &function, const Operation &op);

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

        std::pair< clang::Stmt *, bool > create_int_negative(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool >
        create_int_xor(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool >
        create_int_and(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_int_or(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_int_left(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_int_right(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool > create_int_sright(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );
        std::pair< clang::Stmt *, bool >
        create_int_mult(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_int_div(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_int_rem(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_int_sdiv(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_int_srem(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool > create_bool_negate(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );
        std::pair< clang::Stmt *, bool >
        create_bool_or(clang::ASTContext &ctx, const Function &function, const Operation &op);

        std::pair< clang::Stmt *, bool > create_float_equal(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );
        std::pair< clang::Stmt *, bool > create_float_notequal(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );
        std::pair< clang::Stmt *, bool > create_float_less(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );
        std::pair< clang::Stmt *, bool > create_float_lessequal(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        std::pair< clang::Stmt *, bool >
        create_float_add(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_float_sub(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool > create_float_mult(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );
        std::pair< clang::Stmt *, bool >
        create_float_div(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_float_neg(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_float_abs(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool > create_float_sqrt(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );
        std::pair< clang::Stmt *, bool > create_float_ceil(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );
        std::pair< clang::Stmt *, bool > create_float_floor(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );
        std::pair< clang::Stmt *, bool > create_float_round(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );
        std::pair< clang::Stmt *, bool >
        create_float_nan(clang::ASTContext &ctx, const Function &function, const Operation &op);
        std::pair< clang::Stmt *, bool >
        create_int2float(clang::ASTContext &ctx, const Function &function, const Operation &op);
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

        std::pair< clang::Stmt *, bool > create_address_of(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        template< clang::BinaryOperatorKind Kind >
        std::pair< clang::Stmt *, bool > create_binary_operation(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        template< clang::UnaryOperatorKind Kind >
        std::pair< clang::Stmt *, bool > create_unary_operation(
            clang::ASTContext &ctx, const Function &function, const Operation &op
        );

        Program &get_program(void) const { return program.get(); }

        clang::Sema &get_sema(void) const { return ci.get().getSema(); }

        std::reference_wrapper< Program > program;
        std::reference_wrapper< clang::CompilerInstance > ci;
        std::reference_wrapper< llvm::raw_ostream > out;
        std::reference_wrapper< llvm::raw_ostream > ast_out;

        std::unique_ptr< CodeGenerator > codegen;

        std::unique_ptr< TypeBuilder > type_builder;

        std::unordered_map< std::string, clang::Decl * > incomplete_definition;
        std::unordered_map< std::string, clang::FunctionDecl * > function_declarations;

        /* Map of basic block label decls and stmt for creating branch instructions */
        std::unordered_map< std::string, clang::LabelDecl * > basic_block_labels;

        std::unordered_map< std::string, clang::Stmt * > function_operation_stmts;
        std::unordered_map< std::string, clang::VarDecl * > local_variable_declarations;
        std::unordered_map< std::string, clang::VarDecl * > global_variable_declarations;

        std::unordered_map< std::string, std::vector< clang::Stmt * > > basic_block_stmts;
    };

} // namespace patchestry::ast
