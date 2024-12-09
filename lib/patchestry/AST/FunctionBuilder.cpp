/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <memory>
#include <sstream>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Attrs.inc>
#include <clang/AST/OperationKinds.h>
#include <clang/Frontend/CompilerInstance.h>

#include <patchestry/AST/FunctionBuilder.hpp>
#include <patchestry/AST/OperationBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeTypes.hpp>
#include <patchestry/Util/Log.hpp>
#include <unordered_map>

namespace patchestry::ast {

    namespace {

        /**
         * @brief Retrieves a list of `Operation` objects declared as parameters within the
         * entry block of a given function.
         *
         * This function examines the `entry_block` of the provided `Function` object,
         * identifies operations marked with the mnemonic `OP_DECLARE_PARAMETER`, and collects
         * them into a vector.
         *
         * @param function A reference to a `Function` object containing an `entry_block` and
         * `basic_blocks`.
         *
         * @return A `std::vector` objects representing declared parameters. If the function has
         * no valid entry block or no parameter declarations, the vector is empty.
         *
         */

        std::vector< std::shared_ptr< Operation > > get_parameters(const Function &function) {
            if (function.entry_block.empty() && function.basic_blocks.empty()) {
                return {};
            }

            if (!function.basic_blocks.contains(function.entry_block)) {
                LOG(ERROR) << "Function basic blocks doen't have entry block into it. key "
                           << function.key << "\n";
                return {};
            }

            std::vector< std::shared_ptr< Operation > > operation_vec;
            const auto &entry_block = function.basic_blocks.at(function.entry_block);
            for (const auto &operation_key : entry_block.ordered_operations) {
                if (!entry_block.operations.contains(operation_key)) {
                    LOG(ERROR) << "Skipping, invalid operation key in entry block. key "
                               << function.key << "\n";
                    continue;
                }

                const auto &operation = entry_block.operations.at(operation_key);
                if (operation.mnemonic == Mnemonic::OP_DECLARE_PARAMETER) {
                    operation_vec.push_back(std::make_shared< Operation >(operation));
                }
            }

            return operation_vec;
        }
    } // namespace

    FunctionBuilder::FunctionBuilder(
        clang::CompilerInstance &ci, const Function &function, TypeBuilder &type_builder,
        std::unordered_map< std::string, clang::FunctionDecl * > &functions,
        std::unordered_map< std::string, clang::VarDecl * > &globals
    )
        : prev_decl(nullptr)
        , cii(ci)
        , function(function)
        , type_builder(type_builder)
        , op_builder(nullptr)
        , function_list(functions)
        , global_var_list(globals)
        , local_variables({}) {
        if (!function.key.empty()) {
            if (auto *function_decl = create_declaration(ci.getASTContext())) {
                prev_decl = function_decl;
                function_list.get().emplace(function.key, prev_decl);
            }
        }

        // Initialize `op_builder` after the `FunctionBuilder` is fully constructed
        // op_builder = std::make_shared< OpBuilder >(ci.getASTContext(), shared_from_this());
    }

    void FunctionBuilder::initialize_op_builder(void) {
        // If `op_builder` is initialized don't initailize them
        if (!op_builder) {
            op_builder =
                std::make_shared< OpBuilder >(cii.get().getASTContext(), shared_from_this());
        }
    }

    /**
     * @brief Creates a `FunctionDecl` object for the given function, including its type,
     * parameters, and optional attributes, and adds it to the translation unit declaration
     * context. This method builds a function declaration (or definition) based on the provided
     * function metadata.
     *
     * @param ctx Reference to the `clang::ASTContext`.
     * @param is_definition Boolean flag indicating whether the function is a definition
     * (`true`) or just a declaration (`false`).
     *
     * @return A pointer to the created `clang::FunctionDecl` object. Returns `nullptr` if the
     * creation fails due to errors such as an empty function name or an invalid type.
     */
    clang::FunctionDecl *
    FunctionBuilder::create_declaration(clang::ASTContext &ctx, bool is_definition) {
        if (function.get().name.empty()) {
            LOG(ERROR) << "Function name is empty. function key " << function.get().key << "\n";
            return {};
        }

        auto function_type = create_function_type(ctx, function.get().prototype);
        auto *func_decl    = clang::FunctionDecl::Create(
            ctx, ctx.getTranslationUnitDecl(),
            source_location_from_key(ctx, function.get().key),
            source_location_from_key(ctx, function.get().key),
            &ctx.Idents.get(function.get().name), function_type, nullptr, clang::SC_None
        );

        if (func_decl == nullptr) {
            LOG(ERROR) << "Failed to create declaration for the function. Key: "
                       << function.get().key << "\n";
            return {};
        }

        func_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(func_decl);

        // if function is a declaration, add asm attribute with symbol name
        if (!is_definition) {
            if (auto *asm_attr = clang::AsmLabelAttr::Create(
                    ctx, function.get().name, true, func_decl->getSourceRange()
                ))
            {
                func_decl->addAttr(asm_attr);
            }
        }

        auto parameters     = get_parameters(function);
        auto num_parameters = function.get().prototype.parameters.size();
        if (parameters.size() != num_parameters) {
            // If there is mismatch between number of parameters in function prototype and
            // paramater declaration object, create default parameter considering there is an
            // issue recovering the paramter operation from ghidra and function prototype is
            // correct.
            auto default_params =
                create_default_paramaters(ctx, func_decl, function.get().prototype);
            func_decl->setParams(default_params);
            return func_decl;
        }

        std::vector< clang::ParmVarDecl * > parameter_vec;
        for (const auto &param_op : parameters) {
            const auto &param_type =
                type_builder.get().get_serialized_types().at(*param_op->type);

            auto *param_decl = clang::ParmVarDecl::Create(
                ctx, func_decl, source_location_from_key(ctx, param_op->key),
                source_location_from_key(ctx, param_op->key), &ctx.Idents.get(*param_op->name),
                param_type, nullptr, clang::SC_None, nullptr
            );
            parameter_vec.push_back(param_decl);
            local_variables.emplace(param_op->key, param_decl);
        }

        func_decl->setParams(parameter_vec);
        return func_decl;
    }

    /**
     * @brief Creates a function type based on the provided function prototype.
     *
     * This function constructs a `clang::QualType` representing a function type. The function
     * type is determined by its return type, parameter types, and additional properties such as
     * whether it is variadic or marked as `noreturn`. The method utilizes serialized type
     * from the `TypeBuilder` to resolve return and parameter types.
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
     *
     * @example
     * ```cpp
     * FunctionPrototype proto = {
     *     .rttype_key = "int",
     *     .parameters = {"float", "double"},
     *     .is_variadic = false,
     *     .is_noreturn = false
     * };
     * clang::QualType funcType = create_function_type(ctx, proto);
     * ```
     */
    clang::QualType FunctionBuilder::create_function_type(
        clang::ASTContext &ctx, const FunctionPrototype &proto
    ) {
        if (proto.rttype_key.empty()) {
            LOG(ERROR) << "Function type with invalid return type key.\n";
            return {};
        }

        if (!type_builder.get().get_serialized_types().contains(proto.rttype_key)) {
            LOG(ERROR) << "Function return type is not serialized.\n";
            return {};
        }

        std::vector< clang::QualType > args_vector;
        const auto &rttype = type_builder.get().get_serialized_types()[proto.rttype_key];
        for (const auto &param : proto.parameters) {
            if (!type_builder.get().get_serialized_types().contains(param)) {
                LOG(ERROR) << "Skipping, invalid paramater key in function.\n";
                continue;
            }

            args_vector.emplace_back(type_builder.get().get_serialized_types()[param]);
        }

        clang::FunctionProtoType::ExtProtoInfo ext_proto_info;
        ext_proto_info.Variadic = proto.is_variadic;
        ext_proto_info.ExceptionSpec.Type =
            proto.is_noreturn ? clang::EST_DependentNoexcept : clang::EST_None;

        return ctx.getFunctionType(rttype, args_vector, ext_proto_info);
    }

    /**
     * @brief Creates default parameter declarations for a function based on a given prototype.
     *
     * This function generates a list of parameter declarations (`clang::ParmVarDecl`)
     * for a function declaration (`clang::FunctionDecl`) based on the provided function
     * prototype (`FunctionPrototype`).
     *
     * @param ctx The `clang::ASTContext` used to create AST nodes.
     * @param func_decl The `clang::FunctionDecl` representing the function to which
     *                  the parameters will be added.
     * @param proto The `FunctionPrototype` containing the parameter type keys to
     *              determine the parameter types.
     *
     * @return A vector of pointers to `clang::ParmVarDecl` representing the
     *         parameters for the function.
     */
    std::vector< clang::ParmVarDecl * > FunctionBuilder::create_default_paramaters(
        clang::ASTContext &ctx, clang::FunctionDecl *func_decl, const FunctionPrototype &proto
    ) {
        if (proto.parameters.empty()) {
            return {};
        }

        auto parameter_name = [](uint index) -> std::string {
            std::stringstream ss;
            ss << "param_" << index;
            return ss.str();
        };

        uint index = 0;
        std::vector< clang::ParmVarDecl * > parameter_vec;
        for (const auto &param_key : proto.parameters) {
            if (!type_builder.get().get_serialized_types().contains(param_key)) {
                LOG(ERROR) << "Skipping, invalid paramater type key in function prototype.\n";
                continue;
            }

            auto param_type  = type_builder.get().get_serialized_types().at(param_key);
            auto *param_decl = clang::ParmVarDecl::Create(
                ctx, func_decl, clang::SourceLocation(), clang::SourceLocation(),
                &ctx.Idents.get(parameter_name(index++)), param_type,
                ctx.getTrivialTypeSourceInfo(param_type, clang::SourceLocation()),
                clang::SC_None, nullptr
            );
            assert(param_decl != nullptr);

            parameter_vec.emplace_back(param_decl);
        }

        return parameter_vec;
    }

    clang::FunctionDecl *FunctionBuilder::create_definition(clang::ASTContext &ctx) {
        if (function.get().name.empty() || function.get().basic_blocks.empty()) {
            LOG(ERROR) << "Can't create function definition. Missing function name or has no "
                          "basic blocks.\n";
            return {};
        }

        // basic_block_stmts.clear();

        auto *function_def = create_declaration(ctx, /*is_definition=*/true);
        if (function_def == nullptr) {
            LOG(ERROR) << "Failed to create function definition. key: " << function.get().key
                       << "\n";
            return {};
        }

        // Set previous declaration if exist
        function_def->setPreviousDecl(prev_decl);
        prev_decl = function_def;

        // Before creating function body, set sema context to the current function. It gets used
        // to set lexical and sema decl context for ast nodes.
        auto *prev_context = get_sema_context();
        set_sema_context(function_def);
        auto body_vec = create_function_body(ctx, function_def);
        function_def->setBody(clang::CompoundStmt::Create(
            ctx, body_vec, clang::FPOptionsOverride(), clang::SourceLocation(),
            clang::SourceLocation()
        ));
        set_sema_context(prev_context);

        return function_def;
    }

    void
    FunctionBuilder::create_labels(clang::ASTContext &ctx, clang::FunctionDecl *func_decl) {
        if (function.get().basic_blocks.empty()) {
            return;
        }

        for (const auto &[key, block] : function.get().basic_blocks) {
            // entry block is custom added during recovery and branch instruction will not have
            // entry block as target.
            if (block.is_entry_block) {
                continue;
            }

            auto *label_decl = clang::LabelDecl::Create(
                ctx, func_decl, clang::SourceLocation(),
                &ctx.Idents.get(label_name_from_key(key))
            );
            if (label_decl == nullptr) {
                LOG(ERROR) << "Skipping, fail to create label for basic block with key: " << key
                           << "\n";
                continue;
            }

            label_decl->setDeclContext(func_decl);
            if (clang::DeclContext *dc = label_decl->getLexicalDeclContext()) {
                dc->addDecl(label_decl);
            }

            labels_declaration.emplace(key, label_decl);
        }
    }

    std::vector< clang::Stmt * > FunctionBuilder::create_function_body(
        clang::ASTContext &ctx, clang::FunctionDecl *func_decl
    ) {
        if (function.get().basic_blocks.empty()) {
            LOG(ERROR) << "Function " << function.get().name << " doesn't have body\n";
            return {};
        }

        std::vector< clang::Stmt * > stmt_vec;
        create_labels(ctx, func_decl);

        // Create entry block first
        if (function.get().basic_blocks.contains(function.get().entry_block)) {
            const auto &entry_block =
                function.get().basic_blocks.at(function.get().entry_block);
            auto entry_stmts = create_basic_block(ctx, entry_block);
            stmt_vec.insert(stmt_vec.end(), entry_stmts.begin(), entry_stmts.end());
        }

        std::unordered_map< std::string, std::vector< clang::Stmt * > > bb_stmts;
        for (const auto &[block_key, block] : function.get().basic_blocks) {
            LOG(INFO) << "Processing basic block with key " << block_key << "\n";
            const auto &bb = function.get().basic_blocks.at(block_key);
            if (bb.is_entry_block) {
                continue;
            }

            auto block_stmts = create_basic_block(ctx, bb);
            bb_stmts.emplace(block_key, block_stmts);
        }

        for (auto &[key, block_stmts] : bb_stmts) {
            if (!block_stmts.empty()) {
                auto *label_stmt = new (ctx) clang::LabelStmt(
                    clang::SourceLocation(), labels_declaration.at(key), block_stmts[0]
                );

                // replace first stmt of block with label stmts
                block_stmts[0] = label_stmt;
                stmt_vec.insert(stmt_vec.end(), block_stmts.begin(), block_stmts.end());
            }
        }

        return stmt_vec;
    }

    std::vector< clang::Stmt * >
    FunctionBuilder::create_basic_block(clang::ASTContext &ctx, const BasicBlock &block) {
        if (block.ordered_operations.empty()) {
            LOG(ERROR) << "Basic block with no ordered operations. key: " << block.key << "\n";
            return {};
        }

        std::vector< clang::Stmt * > stmt_vec;
        for (const auto &operation_key : block.ordered_operations) {
            if (!block.operations.contains(operation_key)) {
                LOG(ERROR) << "Skipping, invalid operations key in the block. key "
                           << operation_key << "\n";
                continue;
            }

            const auto &operation = block.operations.at(operation_key);
            if (auto [stmt, should_merge_to_next] = create_operation(ctx, operation); stmt) {
                operation_stmts.emplace(operation.key, stmt);
                if (!should_merge_to_next) {
                    stmt_vec.push_back(stmt);
                }
            }
        }

        return stmt_vec;
    }

    std::pair< clang::Stmt *, bool >
    FunctionBuilder::create_operation(clang::ASTContext &ctx, const Operation &op) {
        if (op.mnemonic == Mnemonic::OP_UNKNOWN) {
            llvm::errs() << "Operation with unknown mnemonic. operation key ( " << op.key
                         << " )\n";
            return {};
        }

        switch (op.mnemonic) {
            case Mnemonic::OP_COPY:
                return op_builder->create_copy(ctx, function, op);
            case Mnemonic::OP_LOAD:
                return op_builder->create_load(ctx, function, op);
            case Mnemonic::OP_STORE:
                return op_builder->create_store(ctx, function, op);
            case Mnemonic::OP_BRANCH:
                return op_builder->create_branch(ctx, op);
            case Mnemonic::OP_CBRANCH:
                return op_builder->create_cbranch(ctx, function, op);
            case Mnemonic::OP_BRANCHIND:
                return op_builder->create_branchind(ctx, function, op);
            case Mnemonic::OP_CALL:
                return op_builder->create_call(ctx, function, op);
            case Mnemonic::OP_CALLIND:
                return op_builder->create_callind(ctx, function, op);
            case Mnemonic::OP_USERDEFINED:
                return op_builder->create_userdefined(ctx, function, op);
            case Mnemonic::OP_RETURN:
                return op_builder->create_return(ctx, function, op);
            case Mnemonic::OP_PIECE:
                return op_builder->create_piece(ctx, function, op);
            case Mnemonic::OP_SUBPIECE:
                return op_builder->create_subpiece(ctx, function, op);
            case Mnemonic::OP_INT_EQUAL:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_EQ);
            case Mnemonic::OP_INT_NOTEQUAL:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_NE);
            case Mnemonic::OP_INT_LESS:
            case Mnemonic::OP_INT_SLESS:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_LT);
            case Mnemonic::OP_INT_LESSEQUAL:
            case Mnemonic::OP_INT_SLESSEQUAL:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_LE);
            case Mnemonic::OP_INT_ZEXT:
                return op_builder->create_int_zext(ctx, function, op);
            case Mnemonic::OP_INT_SEXT:
                return op_builder->create_int_sext(ctx, function, op);
            case Mnemonic::OP_INT_ADD:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Add);
            case Mnemonic::OP_INT_SUB:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Sub);
            case Mnemonic::OP_INT_CARRY:
                return op_builder->create_int_carry(ctx, function, op);
            case Mnemonic::OP_INT_SCARRY:
                return op_builder->create_int_scarry(ctx, function, op);
            case Mnemonic::OP_INT_SBORROW:
                return op_builder->create_int_sborrow(ctx, function, op);
            case Mnemonic::OP_INT_2COMP:
                return op_builder->create_int_2comp(ctx, function, op);
            case Mnemonic::OP_INT_NEGATE:
                return op_builder->create_unary_operation(ctx, function, op, clang::UO_LNot);
            case Mnemonic::OP_INT_XOR:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Xor);
            case Mnemonic::OP_INT_AND:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_And);
            case Mnemonic::OP_INT_OR:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Or);
            case Mnemonic::OP_INT_LEFT:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Shl);
            case Mnemonic::OP_INT_RIGHT:
            case Mnemonic::OP_INT_SRIGHT:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Shr);
            case Mnemonic::OP_INT_MULT:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Mul);
            case Mnemonic::OP_INT_DIV:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Div);
            case Mnemonic::OP_INT_REM:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Rem);
            case Mnemonic::OP_INT_SDIV:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Div);
            case Mnemonic::OP_INT_SREM:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Rem);
            case Mnemonic::OP_BOOL_NEGATE:
                return op_builder->create_unary_operation(ctx, function, op, clang::UO_LNot);
            case Mnemonic::OP_BOOL_OR:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_LOr);
            case Mnemonic::OP_BOOL_AND:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_LAnd);
            case Mnemonic::OP_FLOAT_EQUAL:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_EQ);
            case Mnemonic::OP_FLOAT_NOTEQUAL:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_NE);
            case Mnemonic::OP_FLOAT_LESS:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_LT);
            case Mnemonic::OP_FLOAT_LESSEQUAL:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_LE);
            case Mnemonic::OP_FLOAT_ADD:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Add);
            case Mnemonic::OP_FLOAT_SUB:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Sub);
            case Mnemonic::OP_FLOAT_MULT:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Mul);
            case Mnemonic::OP_FLOAT_DIV:
                return op_builder->create_binary_operation(ctx, function, op, clang::BO_Div);
            case Mnemonic::OP_FLOAT_NEG:
                return op_builder->create_unary_operation(ctx, function, op, clang::UO_LNot);
            case Mnemonic::OP_FLOAT_ABS:
                return op_builder->create_float_abs(ctx, function, op);
            case Mnemonic::OP_FLOAT_SQRT:
                return op_builder->create_float_sqrt(ctx, function, op);
            case Mnemonic::OP_FLOAT_CEIL:
                return op_builder->create_float_ceil(ctx, function, op);
            case Mnemonic::OP_FLOAT_FLOOR:
                return op_builder->create_float_floor(ctx, function, op);
            case Mnemonic::OP_FLOAT_ROUND:
                return op_builder->create_float_round(ctx, function, op);
            case Mnemonic::OP_FLOAT_NAN:
                return op_builder->create_float_nan(ctx, function, op);
            case Mnemonic::OP_INT2FLOAT:
                return op_builder->create_int2float(ctx, function, op);
            case Mnemonic::OP_FLOAT2FLOAT:
                return op_builder->create_float2float(ctx, function, op);
            case Mnemonic::OP_TRUNC:
                return op_builder->create_trunc(ctx, function, op);
            case Mnemonic::OP_PTRSUB:
                return op_builder->create_ptrsub(ctx, function, op);
            case Mnemonic::OP_PTRADD:
                return op_builder->create_ptradd(ctx, function, op);
            case Mnemonic::OP_CAST:
                return op_builder->create_cast(ctx, function, op);
            case Mnemonic::OP_DECLARE_LOCAL:
                return op_builder->create_declare_local(ctx, op);
            case Mnemonic::OP_DECLARE_PARAMETER:
                return op_builder->create_declare_parameter(ctx, function, op);
            case Mnemonic::OP_DECLARE_TEMPORARY:
                return op_builder->create_declare_local(ctx, op);
            case Mnemonic::OP_ADDRESS_OF:
                return op_builder->create_unary_operation(ctx, function, op, clang::UO_AddrOf);
            case Mnemonic::OP_UNKNOWN:
                assert(false);
                break;
        }

        return {};
    }

} // namespace patchestry::ast
