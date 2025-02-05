/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <cassert>
#include <memory>
#include <unordered_map>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Attrs.inc>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclBase.h>
#include <clang/AST/Expr.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <clang/Basic/AttrKinds.h>
#include <clang/Basic/ExceptionSpecificationType.h>
#include <clang/Basic/LLVM.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/Specifiers.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/AST/FunctionBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>

namespace patchestry::ast {

    void PcodeASTConsumer::HandleTranslationUnit(clang::ASTContext &ctx) {
        if (!get_program().serialized_types.empty()) {
            type_builder->create_types(ctx, get_program().serialized_types);
        }

        if (!get_program().serialized_globals.empty()) {
            create_globals(ctx, get_program().serialized_globals);
        }

        if (!get_program().serialized_functions.empty()) {
            create_functions(
                ctx, get_program().serialized_functions, get_program().serialized_types
            );
        }

        std::error_code ec;
        auto out =
            std::make_unique< llvm::raw_fd_ostream >(outfile, ec, llvm::sys::fs::OF_Text);

        llvm::errs() << "Print AST dump\n";
        ctx.getTranslationUnitDecl()->dumpColor();

        ctx.getTranslationUnitDecl()->print(
            *llvm::dyn_cast< llvm::raw_ostream >(out), ctx.getPrintingPolicy(), 0
        );

        llvm::errs() << "Generate mlir\n";
        llvm::raw_fd_ostream file_os(outfile + ".mlir", ec);
        codegen->generate_source_ir(ctx, location_map, file_os);
    }

    void PcodeASTConsumer::set_sema_context(clang::DeclContext *dc) { sema().CurContext = dc; }

    void PcodeASTConsumer::write_to_file(void) {}

    void PcodeASTConsumer::create_functions(
        clang::ASTContext &ctx, FunctionMap &serialized_functions, TypeMap &serialized_types
    ) {
        std::vector< std::shared_ptr< FunctionBuilder > > func_builders;
        for (const auto &[key, function] : serialized_functions) {
            auto builder = std::make_shared< FunctionBuilder >(
                ci.get(), function, *type_builder, function_declarations,
                global_variable_declarations, location_map
            );

            builder->initialize_op_builder();
            func_builders.emplace_back(std::move(builder));
        }

        for (auto &builder : func_builders) {
            auto *func_decl = builder->create_definition(ctx);
            (void) func_decl;
        }
        (void) serialized_types;
    }

    void PcodeASTConsumer::create_globals(
        clang::ASTContext &ctx, VariableMap &serialized_variables
    ) {
        for (auto &[key, variable] : serialized_variables) {
            if (variable.name.empty() || variable.type.empty()) {
                continue;
            }

            auto var_type = type_builder->get_serialized_types().at(variable.type);

            auto *var_decl = clang::VarDecl::Create(
                ctx, ctx.getTranslationUnitDecl(), clang::SourceLocation(),
                clang::SourceLocation(), &ctx.Idents.get(variable.name), var_type,
                ctx.getTrivialTypeSourceInfo(var_type), clang::SC_Static
            );

            var_decl->setDeclContext(ctx.getTranslationUnitDecl());
            ctx.getTranslationUnitDecl()->addDecl(var_decl);
            global_variable_declarations.emplace(variable.key, var_decl);
            location_map.emplace(var_decl, variable.key);
        }
    }

} // namespace patchestry::ast
