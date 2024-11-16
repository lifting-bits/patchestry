/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <cassert>
#include <memory>
#include <sstream>
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
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>

namespace patchestry::ast {

    namespace {

        std::vector< std::string > __attribute__((unused))
        get_keys(const std::unordered_map< std::string, BasicBlock > &map) {
            std::vector< std::string > keys;
            keys.reserve(map.size());

            for (const auto &[key, _] : map) {
                keys.push_back(key);
            }

            std::sort(keys.begin(), keys.end());
            return keys;
        }

        std::vector< std::shared_ptr< Operation > > __attribute__((unused))
        get_parameter_operations(const Function &function) {
            auto entry_block_key = function.entry_block;
            if (entry_block_key.empty() && function.basic_blocks.empty()) {
                return {};
            }

            auto iter = function.basic_blocks.find(entry_block_key);
            if (iter == function.basic_blocks.end()) {
                llvm::errs() << "Function entry block " << entry_block_key
                             << " not present in basic block list\n";
                assert(false);
                return {};
            }

            std::vector< std::shared_ptr< Operation > > ops_vec;
            auto entry_block = iter->second;
            for (const auto &operation_key : entry_block.ordered_operations) {
                auto iter = entry_block.operations.find(operation_key);
                if (iter != entry_block.operations.end()) {
                    auto operation = iter->second;
                    if (operation.mnemonic == Mnemonic::OP_DECLARE_PARAMETER) {
                        ops_vec.push_back(std::make_shared< Operation >(operation));
                    }
                }
            }
            return ops_vec;
        }
    } // namespace

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
        codegen->generate_source_ir(ctx, file_os);
    }

    void PcodeASTConsumer::set_sema_context(clang::DeclContext *dc) {
        get_sema().CurContext = dc;
    }

    void PcodeASTConsumer::write_to_file(void) {}

    clang::QualType PcodeASTConsumer::create_function_prototype(
        clang::ASTContext &ctx, const FunctionPrototype &proto
    ) {
        auto return_key = proto.rttype_key;
        auto iter       = type_builder->get_serialized_types().find(return_key);
        if (iter == type_builder->get_serialized_types().end()) {
            llvm::errs() << "Function return type is not found\n";
            assert(false);
            return clang::QualType();
        }
        auto rttype = iter->second;

        std::vector< clang::QualType > args_vec;
        for (const auto &param : proto.parameters) {
            auto param_iter = type_builder->get_serialized_types().find(param);
            if (param_iter == type_builder->get_serialized_types().end()) {
                assert(false);
            }
            args_vec.push_back(param_iter->second);
        }
        clang::FunctionProtoType::ExtProtoInfo proto_info;
        proto_info.Variadic = proto.is_variadic;
        if (proto.is_noreturn) {
            proto_info.ExceptionSpec.Type = clang::EST_DependentNoexcept;
        }

        return ctx.getFunctionType(rttype, args_vec, proto_info);
    }

    std::vector< clang::ParmVarDecl * > PcodeASTConsumer::create_default_paramaters(
        clang::ASTContext &ctx, clang::FunctionDecl *func_decl, const FunctionPrototype &proto
    ) {
        if (proto.parameters.empty()) {
            return {};
        }

        std::vector< clang::ParmVarDecl * > params;
        int index = 0;
        for (const auto &param_key : proto.parameters) {
            auto param_type = type_builder->get_serialized_types().at(param_key);
            std::stringstream ss;
            ss << "param_" << index++;
            auto param_name  = ss.str();
            auto *param_decl = clang::ParmVarDecl::Create(
                ctx, func_decl, clang::SourceLocation(), clang::SourceLocation(),
                &ctx.Idents.get(param_name), param_type,
                ctx.getTrivialTypeSourceInfo(param_type, clang::SourceLocation()),
                clang::SC_None, nullptr
            );
            params.emplace_back(param_decl);
        }

        return params;
    }

    void PcodeASTConsumer::create_functions(
        clang::ASTContext &ctx, FunctionMap &serialized_functions, TypeMap &serialized_types
    ) {
        for (const auto &[key, function] : serialized_functions) {
            auto *function_decl = create_function_declaration(ctx, function);
            if (function_decl != nullptr) {
                function_declarations.emplace(key, function_decl);
            }

            // TODO: Create global variables
        }

        // Create definition for declared functions
        for (const auto &[key, decl] : function_declarations) {
            auto iter = serialized_functions.find(key);
            assert(iter != serialized_functions.end());
            const auto &parsed_function = iter->second;
            auto *func_def              = create_function_definition(ctx, parsed_function);
            if (func_def != nullptr) {
                func_def->setPreviousDecl(decl);
            }
        }
        (void) serialized_types;
    }

    clang::FunctionDecl *PcodeASTConsumer::create_function_declaration(
        clang::ASTContext &ctx, const Function &function, bool is_definition
    ) {
        if (function.name.empty()) {
            llvm::errs() << "Function name is empty. function key " << function.key << "\n";
            return nullptr;
        }

        auto function_type = create_function_prototype(ctx, function.prototype);
        auto *func_decl    = clang::FunctionDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), source_location_from_key(ctx, function.key),
            source_location_from_key(ctx, function.key), &ctx.Idents.get(function.name),
            function_type, nullptr, clang::SC_None
        );

        // Add function declaration to tralsation unit
        func_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(func_decl);

        // Set asm label attribute to symbol name
        if (!is_definition) {
            auto *asm_attr = clang::AsmLabelAttr::Create(
                ctx, function.name, true, func_decl->getSourceRange()
            );
            if (asm_attr != nullptr) {
                func_decl->addAttr(asm_attr);
            }
        }

        // Create parameters for function declarations;
        auto num_params           = function.prototype.parameters.size();
        auto parameter_operations = get_parameter_operations(function);
        if (parameter_operations.size() == num_params) {
            std::vector< clang::ParmVarDecl * > params;
            for (const auto &param_op : parameter_operations) {
                auto type_iter = type_builder->get_serialized_types().find(*param_op->type);
                assert(type_iter != type_builder->get_serialized_types().end());

                auto *param_decl = clang::ParmVarDecl::Create(
                    ctx, func_decl, source_location_from_key(ctx, param_op->key),
                    source_location_from_key(ctx, param_op->key),
                    &ctx.Idents.get(*param_op->name), type_iter->second, nullptr,
                    clang::SC_None, nullptr
                );
                params.push_back(param_decl);
                local_variable_declarations.emplace(param_op->key, param_decl);
            }

            func_decl->setParams(params);
            return func_decl;
        }

        func_decl->setParams(create_default_paramaters(ctx, func_decl, function.prototype));
        return func_decl;
    }

    clang::FunctionDecl *PcodeASTConsumer::create_function_definition(
        clang::ASTContext &ctx, const Function &function
    ) {
        if (function.name.empty() || function.basic_blocks.empty()) {
            return nullptr;
        }

        function_operation_stmts.clear();
        local_variable_declarations.clear();
        basic_block_stmts.clear();

        auto *func_def = create_function_declaration(ctx, function, true);
        if (func_def != nullptr) {
            set_sema_context(func_def);
            auto body_vec = create_function_body(ctx, func_def, function);
            func_def->setBody(clang::CompoundStmt::Create(
                ctx, body_vec, clang::FPOptionsOverride(), clang::SourceLocation(),
                clang::SourceLocation()
            ));
        }

        return func_def;
    }

    std::vector< clang::Stmt * > PcodeASTConsumer::create_function_body(
        clang::ASTContext &ctx, clang::FunctionDecl *func_decl, const Function &function
    ) {
        if (function.basic_blocks.empty()) {
            llvm::errs() << "Function " << function.name << " doesn't have body\n";
            return {};
        }

        // Create label decl for all basic blocks
        create_label_for_basic_blocks(ctx, func_decl, function);

        std::vector< clang::Stmt * > stmts;

        // If function has entry block, create it first to ensure we have local variables and
        // parameter variables declared
        if (!function.entry_block.empty()) {
            auto iter = function.basic_blocks.find(function.entry_block);
            assert(iter != function.basic_blocks.end());
            auto entry_stmts = create_basic_block(ctx, function, iter->second);
            stmts.insert(stmts.end(), entry_stmts.begin(), entry_stmts.end());
        }

        // get lexicographically sorted keys for basic blocks
        auto block_keys = get_keys(function.basic_blocks);
        for (const auto &block_key : block_keys) {
            llvm::errs() << "Processing basic block with key " << block_key << "\n";
            const auto &bb = function.basic_blocks.at(block_key);
            if (bb.is_entry_block) {
                continue;
            }

            auto block_stmts = create_basic_block(ctx, function, bb);
            basic_block_stmts.emplace(block_key, block_stmts);
        }

        for (auto &[key, block_stmts] : basic_block_stmts) {
            if (!block_stmts.empty()) {
                auto *label_stmt = new (ctx) clang::LabelStmt(
                    clang::SourceLocation(), basic_block_labels.at(key), block_stmts[0]
                );
                // replace first stmt of block with label stmts
                block_stmts[0] = label_stmt;
                stmts.insert(stmts.end(), block_stmts.begin(), block_stmts.end());
            }
        }

        return stmts;
    }

    void PcodeASTConsumer::create_label_for_basic_blocks(
        clang::ASTContext &ctx, clang::FunctionDecl *func_decl, const Function &function
    ) {
        if (function.basic_blocks.empty()) {
            llvm::errs() << "Function " << function.name << " does not have any basic block\n";
            return;
        }

        for (const auto &[key, block] : function.basic_blocks) {
            // entry block is custom added to each function; we don't need to make labels for
            // entry block;
            if (block.is_entry_block) {
                continue;
            }

            auto *label_decl = clang::LabelDecl::Create(
                ctx, func_decl, clang::SourceLocation(),
                &ctx.Idents.get(label_name_from_key(key))
            );

            label_decl->setDeclContext(func_decl);
            if (clang::DeclContext *dc = label_decl->getLexicalDeclContext()) {
                dc->addDecl(label_decl);
            }

            basic_block_labels.emplace(key, label_decl);
        }
    }

    std::vector< clang::Stmt * > PcodeASTConsumer::create_basic_block(
        clang::ASTContext &ctx, const Function &function, const BasicBlock &block
    ) {
        std::vector< clang::Stmt * > stmt_vec;
        for (const auto &operation_key : block.ordered_operations) {
            auto iter = block.operations.find(operation_key);
            if (iter == block.operations.end()) {
                assert(false);
                continue;
            }
            auto operation                    = iter->second;
            auto [stmt, should_merge_to_next] = create_operation(ctx, function, operation);
            if (stmt != nullptr) {
                function_operation_stmts.emplace(operation.key, stmt);
                if (!should_merge_to_next) {
                    stmt_vec.push_back(stmt);
                }
            }
        }

        return stmt_vec;
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
        }
    }

} // namespace patchestry::ast
