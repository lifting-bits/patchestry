/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Attrs.inc>
#include <clang/AST/OperationKinds.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Frontend/CompilerInstance.h>
#include <llvm/Support/ErrorHandling.h>

#include <patchestry/AST/FunctionBuilder.hpp>
#include <patchestry/AST/OperationBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeTypes.hpp>
#include <patchestry/Util/Log.hpp>

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

        std::vector< std::shared_ptr< Operation > > getParameters(const Function &function) {
            if (function.entry_block.empty() && function.basic_blocks.empty()) { return {}; }

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
        std::unordered_map< std::string, clang::VarDecl * > &globals,
        std::unordered_map< std::string, clang::FunctionDecl * > &intrinsics,
        std::string program_arch
    )
        : prev_decl(nullptr)
        , cii(ci)
        , function(function)
        , type_builder(type_builder)
        , op_builder(nullptr)
        , arch(std::move(program_arch))
        , function_list(functions)
        , global_var_list(globals)
        , intrinsic_list(intrinsics)
        , local_variables({}) {
        if (!function.key.empty()) {
            if (auto *function_decl = create_declaration(
                    ci.getASTContext(),
                    create_function_type(ci.getASTContext(), function.prototype)
                ))
            {
                prev_decl = function_decl;
                function_list.get().emplace(function.key, prev_decl);
            }
        }
    }

    void FunctionBuilder::InitializeOpBuilder(void) {
        // Create op_builder on first call; later calls are no-ops.
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
    clang::FunctionDecl *FunctionBuilder::create_declaration(
        clang::ASTContext &ctx, const clang::QualType &function_type, bool is_definition
    ) {
        const auto &c_name = GetCName();

        if (c_name.empty()) {
            LOG(ERROR) << "Function name is empty. function key " << function.get().key << "\n";
            return {};
        }

        auto location   = SourceLocation(ctx.getSourceManager(), function.get().key);
        auto *func_decl = clang::FunctionDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), location, location, &ctx.Idents.get(c_name),
            function_type, ctx.getTrivialTypeSourceInfo(function_type), clang::SC_None
        );

        if (func_decl == nullptr) {
            LOG(ERROR) << "Failed to create declaration for the function. Key: "
                       << function.get().key << "\n";
            return {};
        }

        func_decl->setDeclContext(ctx.getTranslationUnitDecl());
        ctx.getTranslationUnitDecl()->addDecl(func_decl);

        // Add asm label with the binary linker symbol when:
        //   (1) the original name differs from the C identifier, AND
        //   (2) the original name is a genuine mangled symbol (_Z for
        //       Itanium ABI, ? for MSVC).
        // Short demangled names like "append" or "operator=" are NOT valid
        // linker symbols and must not appear in asm labels — they would
        // cause link failures when recompiling for binary patching.
        const auto &original_name = function.get().name;
        bool is_mangled           = original_name.size() >= 2
            && ((original_name[0] == '_' && original_name[1] == 'Z')
                || original_name[0] == '?');
        if (is_mangled && original_name != c_name) {
            if (auto *asm_attr = clang::AsmLabelAttr::Create(
                    ctx, original_name, func_decl->getSourceRange()
                ))
            {
                func_decl->addAttr(asm_attr);
            }
        }

        // Propagate prototype noreturn so isNoReturn() agrees with Ghidra.
        if (function.get().prototype.is_noreturn) {
            if (auto *nr_attr = clang::NoReturnAttr::Create(ctx, func_decl->getSourceRange())) {
                func_decl->addAttr(nr_attr);
            }
        }

        auto parameters     = getParameters(function);
        auto num_parameters = function.get().prototype.parameters.size();
        if (parameters.size() != num_parameters) {
            // If there is mismatch between number of parameters in function prototype and
            // parameter declaration object, create default parameter considering there is an
            // issue recovering the parameter operation from ghidra and function prototype is
            // correct.
            auto default_params =
                create_default_paramaters(ctx, func_decl, function.get().prototype);
            func_decl->setParams(default_params);
            return func_decl;
        }

        std::vector< clang::ParmVarDecl * > parameter_vec;
        for (const auto &param_op : parameters) {
            if (!param_op->type || !param_op->name) {
                LOG(ERROR) << "Parameter operation missing type or name, key: "
                           << param_op->key;
                continue;
            }
            auto type_iter = type_builder.get().GetSerializedTypes().find(*param_op->type);
            if (type_iter == type_builder.get().GetSerializedTypes().end()) {
                LOG(ERROR) << "Parameter type not found in serialized types: "
                           << *param_op->type << ", key: " << param_op->key;
                continue;
            }
            const auto &param_type = type_iter->second;
            auto location          = SourceLocation(ctx.getSourceManager(), param_op->key);

            auto *param_decl = clang::ParmVarDecl::Create(
                ctx, func_decl, location, location, &ctx.Idents.get(*param_op->name),
                param_type, nullptr, clang::SC_None, nullptr
            );
            param_decl->setIsUsed();
            parameter_vec.push_back(param_decl);

            // If this is for definition
            if (is_definition) { local_variables.emplace(param_op->key, param_decl); }
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

        auto rttype = type_builder.get().GetSerializedType(proto.rttype_key);
        if (rttype.isNull()) {
            LOG(ERROR) << "Function return type is not serialized.\n";
            return {};
        }

        std::vector< clang::QualType > args_vector;
        for (const auto &param : proto.parameters) {
            auto param_type = type_builder.get().GetSerializedType(param);
            if (param_type.isNull()) {
                LOG(ERROR) << "Skipping, invalid parameter key in function.\n";
                continue;
            }

            args_vector.emplace_back(param_type);
        }

        clang::FunctionProtoType::ExtProtoInfo ext_proto_info;
        ext_proto_info.Variadic           = proto.is_variadic;
        ext_proto_info.ExceptionSpec.Type = clang::EST_None;

        // TODO: Map non-default calling conventions (e.g. __stdcall,
        // __fastcall, __thiscall, __vectorcall) to clang::CallingConv once
        // ClangIR supports them. Currently ClangIR only handles the default
        // CC_C convention; all others hit UNREACHABLE in CIRGenTypes.cpp.
        // The calling convention string is available in proto.calling_convention.

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
        if (proto.parameters.empty()) { return {}; }

        auto parameter_name = [](uint index) -> std::string {
            std::stringstream ss;
            ss << "param_" << index;
            return ss.str();
        };

        uint index = 0;
        std::vector< clang::ParmVarDecl * > parameter_vec;
        for (const auto &param_key : proto.parameters) {
            auto param_type = type_builder.get().GetSerializedType(param_key);
            if (param_type.isNull()) {
                LOG(ERROR) << "Skipping, invalid paramater type key in function prototype.\n";
                continue;
            }

            auto param_loc   = SourceLocation(ctx.getSourceManager(), param_key);
            auto *param_decl = clang::ParmVarDecl::Create(
                ctx, func_decl, param_loc, param_loc, &ctx.Idents.get(parameter_name(index++)),
                param_type, ctx.getTrivialTypeSourceInfo(param_type, param_loc), clang::SC_None,
                nullptr
            );
            assert(param_decl != nullptr);
            param_decl->setIsUsed();

            parameter_vec.emplace_back(param_decl);
        }

        return parameter_vec;
    }

    /**
     * @brief Creates a `clang::FunctionDecl` representing the definition of a function,
     * including its body.
     *
     * @param ctx The `clang::ASTContext` used to create AST nodes.
     * @return A pointer to the created `clang::FunctionDecl` representing the function
     * definition. Returns `nullptr` if the function name is empty, the function has no basic
     * blocks, or if the function definition cannot be created.
     */
    clang::FunctionDecl *FunctionBuilder::create_definition(clang::ASTContext &ctx) {
        const auto &c_name = GetCName();
        if (c_name.empty()) {
            LOG(ERROR) << "Can't create function shell. Missing function name.\n";
            return {};
        }

        if (function.get().basic_blocks.empty()) {
            // The canonical forward decl for a no-basic-blocks function (e.g.
            // a Ghidra-exported callee that the decompile request did not
            // descend into, like emit_before_base in the serial-qemu fixture)
            // is created by the FunctionBuilder constructor and registered in
            // function_list. ASTConsumer skips these via has_basic_blocks(),
            // so reaching here means a caller violated that contract.
            LOG(ERROR) << "create_definition called for '" << c_name
                       << "' which has no basic blocks; caller should skip "
                          "via has_basic_blocks(). The forward decl from the "
                          "FunctionBuilder constructor is canonical.\n";
            return {};
        }

        auto function_type = prev_decl != nullptr
            ? prev_decl->getType()
            : create_function_type(ctx, function.get().prototype);

        auto *function_def = create_declaration(ctx, function_type, /*is_definition=*/true);
        if (function_def == nullptr) {
            LOG(ERROR) << "Failed to create function shell. key: " << function.get().key
                       << "\n";
            return {};
        }

        function_def->setPreviousDecl(prev_decl);
        function_def->setWillHaveBody(true);

        auto *prev_context = get_sema_context();
        set_sema_context(function_def);
        create_labels(ctx, function_def);
        set_sema_context(prev_context);

        return function_def;
    }

    std::vector< clang::Stmt * >
    FunctionBuilder::create_block_stmts(clang::ASTContext &ctx, const BasicBlock &block) {
        RegisterSourceBlock(block.key);

        if (block.ordered_operations.empty()) { return {}; }

        std::vector< clang::Stmt * > stmt_vec;
        for (const auto &operation_key : block.ordered_operations) {
            if (!block.operations.contains(operation_key)) {
                LOG(ERROR) << "Skipping, invalid operations key in the block. key "
                           << operation_key << "\n";
                continue;
            }

            const auto &operation = block.operations.at(operation_key);
            RegisterSourceOperation(
                block.key, operation,
                /*has_payload_carrier=*/false
            );

            // Skip branch terminals — these become edges in the CGraph.
            // RETURN is NOT skipped: it produces a ReturnStmt that must
            // appear in the block's stmts (RETURN has no outgoing edges,
            // and its inputs may chain from preceding operations via the
            // merge mechanism).
            if (operation.mnemonic == Mnemonic::OP_BRANCH
                || operation.mnemonic == Mnemonic::OP_CBRANCH
                || operation.mnemonic == Mnemonic::OP_BRANCHIND)
            {
                continue;
            }

            auto saved_pending = std::move(pending_materialized);
            pending_materialized.clear();

            if (auto [stmt, should_merge_to_next] = create_operation(ctx, operation); stmt) {
                for (auto *pending : pending_materialized) {
                    TrackOperationStmt(
                        block.key, operation, pending, ClassifyOperationStmt(operation, pending)
                    );
                    stmt_vec.push_back(pending);
                }
                pending_materialized.clear();
                operation_stmts.emplace(operation.key, stmt);
                if (!should_merge_to_next) {
                    TrackOperationStmt(
                        block.key, operation, stmt, ClassifyOperationStmt(operation, stmt)
                    );
                    stmt_vec.push_back(stmt);
                }
            } else {
                pending_materialized.clear();
            }

            for (auto *s : saved_pending) { pending_materialized.push_back(s); }
        }

        InlineSingleUseTemps(ctx, stmt_vec);
        return stmt_vec;
    }

    void FunctionBuilder::RegisterSourceBlock(const std::string &block_key) {
        if (block_key.empty()) { return; }

        std::string stable_id  = function.get().key;
        stable_id             += "::block::";
        stable_id             += block_key;
        source_block_ids.try_emplace(block_key, std::move(stable_id));
    }

    void FunctionBuilder::RegisterSourceOperation(
        const std::string &block_key, const Operation &op, bool has_payload_carrier
    ) {
        if (op.key.empty()) { return; }

        RegisterSourceBlock(block_key);

        std::string stable_id  = function.get().key;
        stable_id             += "::op::";
        stable_id             += block_key;
        stable_id             += "::";
        stable_id             += op.key;

        auto [it, inserted] = source_ops.try_emplace(
            op.key, SourceId{ std::move(stable_id), block_key, op.key, has_payload_carrier }
        );
        if (!inserted && has_payload_carrier) { it->second.has_payload_carrier = true; }
    }

    void FunctionBuilder::TrackOperationStmt(
        const std::string &block_key, const Operation &op, clang::Stmt *stmt, PayloadKind kind,
        bool primary
    ) {
        if (!stmt) { return; }

        RegisterSourceOperation(
            block_key, op,
            /*has_payload_carrier=*/true
        );

        auto build_origin = [&](PayloadKind origin_kind) {
            StmtOrigin origin;
            origin.block_key     = block_key;
            origin.operation_key = op.key;
            origin.kind          = origin_kind;
            origin.primary       = primary;

            std::function< void(const clang::Stmt *) > scan = [&](const clang::Stmt *cur) {
                if (!cur) { return; }
                if (llvm::isa< clang::CallExpr >(cur)) { origin.may_call = true; }
                if (auto *expr = llvm::dyn_cast< clang::Expr >(cur)) {
                    if (!expr->getType().isNull() && expr->getType().isVolatileQualified()) {
                        origin.may_volatile = true;
                    }
                }
                if (auto *bin = llvm::dyn_cast< clang::BinaryOperator >(cur)) {
                    if (bin->isAssignmentOp()) { origin.may_store = true; }
                }
                if (auto *unary = llvm::dyn_cast< clang::UnaryOperator >(cur)) {
                    if (unary->isIncrementDecrementOp()) { origin.may_store = true; }
                }
                if (llvm::isa< clang::IfStmt >(cur) || llvm::isa< clang::ForStmt >(cur)
                    || llvm::isa< clang::WhileStmt >(cur) || llvm::isa< clang::DoStmt >(cur)
                    || llvm::isa< clang::SwitchStmt >(cur) || llvm::isa< clang::GotoStmt >(cur)
                    || llvm::isa< clang::IndirectGotoStmt >(cur)
                    || llvm::isa< clang::LabelStmt >(cur) || llvm::isa< clang::CaseStmt >(cur)
                    || llvm::isa< clang::DefaultStmt >(cur)
                    || llvm::isa< clang::BreakStmt >(cur)
                    || llvm::isa< clang::ContinueStmt >(cur)
                    || llvm::isa< clang::ReturnStmt >(cur))
                {
                    origin.contains_internal_control = true;
                }
                for (const clang::Stmt *child : cur->children()) { scan(child); }
            };
            scan(stmt);

            origin.movable = origin_kind != PayloadKind::PayloadSyntheticDecl
                && origin_kind != PayloadKind::TerminalSynthetic
                && origin_kind != PayloadKind::RegionSynthetic;
            origin.cloneable = origin.movable && !origin.may_call && !origin.may_store
                && !origin.may_volatile && !origin.contains_internal_control;
            return origin;
        };

        stmt_origins[stmt] = build_origin(kind);
        source_coverage_nodes[stmt].insert(op.key);
        if (primary) { source_primary_nodes[stmt].insert(op.key); }

        // InlineSingleUseTemps may remove a DeclStmt and splice its
        // initializer into the following statement.  Treat the initializer
        // as a secondary carrier for the declaration operation so coverage
        // survives that local AST rewrite without reporting a false loss.
        if (auto *decl_stmt = llvm::dyn_cast< clang::DeclStmt >(stmt)) {
            if (decl_stmt->isSingleDecl()) {
                if (auto *var = llvm::dyn_cast< clang::VarDecl >(decl_stmt->getSingleDecl())) {
                    if (auto *init = var->getInit()) {
                        TrackOperationStmt(
                            block_key, op, init, PayloadKind::PayloadExpression,
                            /*primary=*/false
                        );
                    }
                }
            }
        }
    }

    PayloadKind
    FunctionBuilder::ClassifyOperationStmt(const Operation &op, const clang::Stmt *stmt) const {
        using M = Mnemonic;
        if (op.mnemonic == M::OP_BRANCH || op.mnemonic == M::OP_CBRANCH
            || op.mnemonic == M::OP_BRANCHIND)
        {
            return PayloadKind::TerminalSynthetic;
        }

        if (llvm::isa< clang::DeclStmt >(stmt)) { return PayloadKind::PayloadSyntheticDecl; }

        if (llvm::isa< clang::Expr >(stmt)) { return PayloadKind::PayloadExpression; }

        return PayloadKind::AtomicPayload;
    }

    void FunctionBuilder::TrackTerminalStmt(
        const std::string &block_key, const Operation &op, clang::Stmt *stmt
    ) {
        if (!stmt) { return; }
        RegisterSourceOperation(
            block_key, op,
            /*has_payload_carrier=*/false
        );

        StmtOrigin origin;
        origin.block_key                 = block_key;
        origin.operation_key             = op.key;
        origin.kind                      = PayloadKind::TerminalSynthetic;
        origin.movable                   = false;
        origin.cloneable                 = false;
        origin.contains_internal_control = true;
        stmt_origins[stmt]               = std::move(origin);
    }

    const StmtOrigin *FunctionBuilder::GetStmtOrigin(const clang::Stmt *stmt) const {
        auto it = stmt_origins.find(stmt);
        if (it == stmt_origins.end()) { return nullptr; }
        return &it->second;
    }

    void FunctionBuilder::CollectStmtOrigins(
        const clang::Stmt *stmt, std::vector< StmtOrigin > &origins
    ) const {
        if (!stmt) { return; }

        if (const auto *origin = GetStmtOrigin(stmt)) { origins.push_back(*origin); }

        if (const auto *decl_stmt = llvm::dyn_cast< clang::DeclStmt >(stmt)) {
            for (const clang::Decl *decl : decl_stmt->decls()) {
                if (const auto *var = llvm::dyn_cast< clang::VarDecl >(decl)) {
                    CollectStmtOrigins(var->getInit(), origins);
                }
            }
        }

        for (const clang::Stmt *child : stmt->children()) {
            CollectStmtOrigins(child, origins);
        }
    }

    namespace {

        void CollectSourceCoverage(
            const clang::Stmt *stmt,
            const std::unordered_map< const clang::Stmt *, std::unordered_set< std::string > >
                &coverage_nodes,
            const std::unordered_map< const clang::Stmt *, std::unordered_set< std::string > >
                &primary_nodes,
            std::unordered_set< std::string > &covered_ops,
            std::unordered_map< std::string, unsigned > &primary_visits
        ) {
            if (!stmt) { return; }

            if (auto it = coverage_nodes.find(stmt); it != coverage_nodes.end()) {
                for (const auto &op_key : it->second) { covered_ops.insert(op_key); }
            }
            if (auto it = primary_nodes.find(stmt); it != primary_nodes.end()) {
                for (const auto &op_key : it->second) { ++primary_visits[op_key]; }
            }

            if (auto *decl_stmt = llvm::dyn_cast< clang::DeclStmt >(stmt)) {
                for (const clang::Decl *decl : decl_stmt->decls()) {
                    if (auto *var = llvm::dyn_cast< clang::VarDecl >(decl)) {
                        CollectSourceCoverage(
                            var->getInit(), coverage_nodes, primary_nodes, covered_ops,
                            primary_visits
                        );
                    }
                }
            }

            for (const clang::Stmt *child : stmt->children()) {
                CollectSourceCoverage(
                    child, coverage_nodes, primary_nodes, covered_ops, primary_visits
                );
            }
        }

    } // anonymous namespace

    FunctionBuilder::SourceVerificationReport
    FunctionBuilder::BuildSourceVerificationReport(const clang::FunctionDecl *fn) const {
        SourceVerificationReport report;
        for (const auto &[op_key, id] : source_ops) {
            if (id.has_payload_carrier) { report.input_ops.push_back(op_key); }
        }
        std::sort(report.input_ops.begin(), report.input_ops.end());

        for (const auto &[_, origin] : stmt_origins) {
            ++report.origin_kind_counts[PayloadKindName(origin.kind)];
        }

        if (!fn || !fn->hasBody()) {
            report.missing_ops = report.input_ops;
            return report;
        }

        std::unordered_set< std::string > covered_ops;
        std::unordered_map< std::string, unsigned > primary_visits;
        CollectSourceCoverage(
            fn->getBody(), source_coverage_nodes, source_primary_nodes, covered_ops,
            primary_visits
        );

        for (const auto &op_key : report.input_ops) {
            if (covered_ops.contains(op_key)) {
                report.emitted_ops.push_back(op_key);
            } else {
                report.missing_ops.push_back(op_key);
            }
        }
        std::sort(report.emitted_ops.begin(), report.emitted_ops.end());
        std::sort(report.missing_ops.begin(), report.missing_ops.end());

        for (const auto &[op_key, count] : primary_visits) {
            if (!source_ops.contains(op_key) || !source_ops.at(op_key).has_payload_carrier) {
                continue;
            }

            report.emitted_counts[op_key] = count;
            if (count > 1) {
                report.duplicated_ops.push_back(op_key);
                report.cloned_counts[op_key] = count - 1;
            }
        }
        std::sort(report.duplicated_ops.begin(), report.duplicated_ops.end());
        return report;
    }

    bool FunctionBuilder::VerifyNoNodeLoss(const clang::FunctionDecl *fn) const {
        auto report = BuildSourceVerificationReport(fn);

        if (report.missing_ops.empty()) {
            if (!report.duplicated_ops.empty()) {
                LOG(INFO) << "VerifyNoNodeLoss: function " << GetCName()
                          << " input_ops=" << report.input_ops.size()
                          << " emitted_ops=" << report.emitted_ops.size() << " missing_ops=0"
                          << " duplicated_ops=" << report.duplicated_ops.size()
                          << " cloned_carrier_occurrences=" <<
                    [&]() {
                        unsigned total = 0;
                        for (const auto &[_, count] : report.cloned_counts) { total += count; }
                        return total;
                    }() << "\n";
            }
            return true;
        }

        LOG(ERROR) << "VerifyNoNodeLoss: function " << GetCName()
                   << " input_ops=" << report.input_ops.size()
                   << " emitted_ops=" << report.emitted_ops.size()
                   << " missing_ops=" << report.missing_ops.size()
                   << " duplicated_ops=" << report.duplicated_ops.size() << "\n";
        constexpr size_t kMaxReport = 20;
        for (size_t i = 0; i < std::min(report.missing_ops.size(), kMaxReport); ++i) {
            const auto &op_key = report.missing_ops[i];
            const auto &info   = source_ops.at(op_key);
            LOG(ERROR) << "  missing op " << info.operation_key << " from block "
                       << info.block_key << " source_id=" << info.stable_id << "\n";
        }
        if (report.missing_ops.size() > kMaxReport) {
            LOG(ERROR) << "  ... " << (report.missing_ops.size() - kMaxReport)
                       << " more missing operation node(s)\n";
        }
        return false;
    }

    clang::Expr *
    FunctionBuilder::create_branch_condition(clang::ASTContext &ctx, const Operation &op) {
        if (!op.condition) {
            LOG(ERROR) << "CBRANCH with no condition. key: " << op.key << "\n";
            return nullptr;
        }

        auto *cond_stmt = op_builder->create_varnode(ctx, function.get(), *op.condition);
        auto *cond_expr = clang::dyn_cast_or_null< clang::Expr >(cond_stmt);
        if (!cond_expr) {
            LOG(ERROR) << "Failed to create condition for CBRANCH. key: " << op.key << "\n";
        }
        return cond_expr;
    }

    clang::Expr *
    FunctionBuilder::create_switch_discriminant(clang::ASTContext &ctx, const Operation &op) {
        if (op.inputs.empty()) {
            LOG(ERROR) << "BRANCHIND with no inputs. key: " << op.key << "\n";
            return nullptr;
        }

        clang::Expr *disc = nullptr;
        // Prefer named local/param as discriminant, then switch_input,
        // then fall back to inputs[0] regardless of kind (handles constants
        // for address-based jump tables).
        if (op.inputs[0].kind == Varnode::VARNODE_LOCAL
            || op.inputs[0].kind == Varnode::VARNODE_PARAM)
        {
            disc = clang::dyn_cast_or_null< clang::Expr >(
                op_builder->create_varnode(ctx, function.get(), op.inputs[0])
            );
        } else if (op.switch_input.has_value()) {
            disc = clang::dyn_cast_or_null< clang::Expr >(
                op_builder->create_varnode(ctx, function.get(), *op.switch_input)
            );
        } else {
            disc = clang::dyn_cast_or_null< clang::Expr >(
                op_builder->create_varnode(ctx, function.get(), op.inputs[0])
            );
        }

        if (disc) {
            // Promote narrow discriminant to int width
            auto disc_type  = disc->getType();
            bool needs_cast = !disc_type->isIntegerType()
                || ctx.getIntWidth(disc_type) < ctx.getIntWidth(ctx.IntTy);
            if (needs_cast) {
                auto loc = SourceLocation(ctx.getSourceManager(), op.key);
                disc     = op_builder->make_cast(ctx, disc, ctx.IntTy, loc);
            }
        }

        if (!disc) {
            LOG(ERROR) << "Failed to create switch discriminant. key: " << op.key << "\n";
        }
        return disc;
    }

    clang::Expr *FunctionBuilder::create_cast(
        clang::ASTContext &ctx, clang::Expr *expr, const clang::QualType &to_type,
        clang::SourceLocation loc
    ) {
        return op_builder->make_cast(ctx, expr, to_type, loc);
    }

    /**
     * @brief Creates and registers label declarations for basic blocks in a function.
     *
     * @param ctx The Clang ASTContext used to create label declarations.
     * @param func_decl The Clang function declaration to which the labels belong.
     */
    void
    FunctionBuilder::create_labels(clang::ASTContext &ctx, clang::FunctionDecl *func_decl) {
        if (function.get().basic_blocks.empty()) { return; }

        for (const auto &[key, block] : function.get().basic_blocks) {
            // entry block is custom added during recovery and branch instruction will not have
            // entry block as target.
            if (block.is_entry_block) { continue; }

            auto *label_decl = clang::LabelDecl::Create(
                ctx, func_decl, SourceLocation(ctx.getSourceManager(), key),
                &ctx.Idents.get(LabelNameFromKey(key))
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

    namespace {

        // Count references to a VarDecl within a Clang Stmt tree.
        unsigned CountVarRefs(const clang::Stmt *s, const clang::VarDecl *vd) {
            if (!s) { return 0; }
            if (auto *dre = llvm::dyn_cast< clang::DeclRefExpr >(s)) {
                if (dre->getDecl() == vd) { return 1; }
            }
            unsigned count = 0;
            for (auto *child : s->children()) { count += CountVarRefs(child, vd); }
            return count;
        }

        // Replace all DeclRefExpr(vd) with replacement_expr in the stmt tree.
        // Returns true if a replacement was made.
        bool ReplaceVarRef(
            clang::Stmt *s, clang::VarDecl *vd, clang::Expr *replacement, clang::ASTContext &ctx
        ) {
            if (!s) { return false; }

            // Check each child. If a child is a DeclRefExpr to vd, replace it
            // by mutating the parent's child pointer.
            for (auto it = s->child_begin(); it != s->child_end(); ++it) {
                if (!*it) { continue; }
                if (auto *dre = llvm::dyn_cast< clang::DeclRefExpr >(*it)) {
                    if (dre->getDecl() == vd) {
                        // Clang Stmt children are mutable via the iterator
                        *it = replacement;
                        return true;
                    }
                }
                if (ReplaceVarRef(*it, vd, replacement, ctx)) { return true; }
            }
            return false;
        }

    } // anonymous namespace

    void FunctionBuilder::InlineSingleUseTemps(
        clang::ASTContext &ctx, std::vector< clang::Stmt * > &stmts
    ) {
        for (size_t i = 0; i + 1 < stmts.size();) {
            // Look for: DeclStmt { VarDecl var = init_expr; }
            auto *ds = llvm::dyn_cast< clang::DeclStmt >(stmts[i]);
            if (!ds || !ds->isSingleDecl()) {
                ++i;
                continue;
            }
            auto *vd = llvm::dyn_cast< clang::VarDecl >(ds->getSingleDecl());
            if (!vd || !vd->hasInit()) {
                ++i;
                continue;
            }

            // Don't inline if the init has side effects that must execute
            // at this point (calls, increments, volatile loads).
            auto *init = vd->getInit();
            if (init->HasSideEffects(ctx)) {
                ++i;
                continue;
            }

            // Count references in the NEXT statement only.
            // If exactly 1 reference, inline; otherwise skip.
            unsigned refs_in_next = CountVarRefs(stmts[i + 1], vd);
            if (refs_in_next != 1) {
                ++i;
                continue;
            }

            // Check no references in any later statements
            bool used_later = false;
            for (size_t j = i + 2; j < stmts.size(); ++j) {
                if (CountVarRefs(stmts[j], vd) > 0) {
                    used_later = true;
                    break;
                }
            }
            if (used_later) {
                ++i;
                continue;
            }

            // Inline: replace the DeclRefExpr in stmts[i+1] with init
            if (ReplaceVarRef(stmts[i + 1], vd, init, ctx)) {
                // Remove the DeclStmt
                stmts.erase(stmts.begin() + static_cast< ptrdiff_t >(i));
                // Don't increment — re-check same index (the next stmt shifted down)
            } else {
                ++i;
            }
        }
    }

    /**
     * @brief Creates a Clang AST representation of a given operation based on its mnemonic.
     *
     * This function takes an operation and generates the corresponding Clang AST node,
     * which represents the operation in the target AST.
     *
     * @param ctx The Clang ASTContext for creating AST nodes.
     * @param op The operation to be translated into a Clang AST node.
     * @return A pair consisting of:
     *         - A pointer to the generated Clang Stmt representing the operation.
     *         - A boolean indicating if stmt should be merged with previous one.
     */
    std::pair< clang::Stmt *, bool >
    FunctionBuilder::create_operation(clang::ASTContext &ctx, const Operation &op) {
        if (op.mnemonic == Mnemonic::OP_UNKNOWN) {
            LOG(ERROR) << "Operation with unknown mnemonic. operation key ( " << op.key
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
            case Mnemonic::OP_TAIL_CALL: {
                auto *enclosing =
                    clang::dyn_cast_or_null< clang::FunctionDecl >(get_sema_context());
                return op_builder->create_tail_call(ctx, function, op, enclosing);
            }
            case Mnemonic::OP_CALLOTHER:
                return op_builder->create_callother(ctx, function, op);
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
                return op_builder->create_unary_operation(ctx, function, op, clang::UO_Not);
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
                return op_builder->create_unary_operation(ctx, function, op, clang::UO_Minus);
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
            case Mnemonic::OP_POPCOUNT:
                return op_builder->create_popcount(ctx, function, op);
            case Mnemonic::OP_LZCOUNT:
                return op_builder->create_lzcount(ctx, function, op);
            case Mnemonic::OP_UNKNOWN:
                llvm_unreachable("Encountered OP_UNKNOWN P-Code mnemonic");
        }

        return {};
    }

} // namespace patchestry::ast
