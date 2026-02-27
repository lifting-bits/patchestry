/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stack>
#include <unordered_map>
#include <unordered_set>

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

        /**
         * @brief Comparator for basic-block keys that sorts numerically rather than
         * lexicographically.
         *
         * Block keys have the form "ram:HEXADDR:DECIMALIDX:basic" (or ":entry" for the entry
         * block).  A plain std::map<string> would sort "10" before "2", producing wrong block
         * ordering for functions with >= 10 basic blocks.  This comparator parses the address
         * and index fields as integers so the ordering is always correct.
         */
        struct BlockKeyComparator {
            static std::pair< uint64_t, int64_t > parse_key(const std::string &key) {
                // "ram:HEXADDR:DECIMALIDX:basic"  or  "ram:HEXADDR:entry"
                constexpr std::string_view kExpectedPrefix = "ram:";
                if (key.substr(0, kExpectedPrefix.size()) != kExpectedPrefix) {
                    LOG(WARNING) << "BlockKeyComparator: unexpected key format (no 'ram:' prefix): "
                                 << key;
                    return {std::numeric_limits< uint64_t >::max(),
                            std::numeric_limits< int64_t >::max()};
                }

                auto first_colon = key.find(':');
                auto second_colon = key.find(':', first_colon + 1);
                if (second_colon == std::string::npos) {
                    LOG(WARNING) << "BlockKeyComparator: malformed key (missing second ':'): " << key;
                    return {std::numeric_limits< uint64_t >::max(),
                            std::numeric_limits< int64_t >::max()};
                }

                uint64_t addr = 0;
                try {
                    addr = std::stoull(
                        key.substr(first_colon + 1, second_colon - first_colon - 1), nullptr, 16
                    );
                } catch (...) {
                    LOG(WARNING) << "BlockKeyComparator: failed to parse hex address in key: " << key;
                    return {std::numeric_limits< uint64_t >::max(),
                            std::numeric_limits< int64_t >::max()};
                }

                auto third_colon = key.find(':', second_colon + 1);
                std::string idx_str = key.substr(
                    second_colon + 1,
                    third_colon == std::string::npos ? std::string::npos
                                                     : third_colon - second_colon - 1
                );

                if (idx_str == "entry") {
                    return {addr, -1};
                }

                int64_t idx = 0;
                try {
                    idx = std::stoll(idx_str);
                } catch (...) {
                    LOG(WARNING) << "BlockKeyComparator: failed to parse block index in key: " << key;
                    return {addr, std::numeric_limits< int64_t >::max()};
                }

                return {addr, idx};
            }

            bool operator()(const std::string &a, const std::string &b) const {
                auto [addr_a, idx_a] = parse_key(a);
                auto [addr_b, idx_b] = parse_key(b);
                if (addr_a != addr_b) {
                    return addr_a < addr_b;
                }
                return idx_a < idx_b;
            }
        };

        /**
         * @brief Compute RPO block order from the P-Code CFG embedded in the function.
         *
         * Performs an iterative post-order DFS from the entry block, then reverses to
         * obtain RPO.  Successor order (taken before not_taken) mirrors the ordering
         * used by BasicBlockReorderPass so that the two orderings are identical.
         * Unreachable blocks are appended after the RPO sequence in deterministic
         * (address-sorted) order.
         */
        std::vector< std::string > compute_rpo(const Function &function) {
            // Build per-block successor lists from branch operations.
            std::unordered_map< std::string, std::vector< std::string > > succs;
            for (const auto &[key, blk] : function.basic_blocks) {
                for (const auto &op_key : blk.ordered_operations) {
                    if (!blk.operations.contains(op_key)) {
                        continue;
                    }
                    const auto &op = blk.operations.at(op_key);
                    // taken_block before not_taken_block — mirrors BasicBlockReorderPass DFS
                    // order so the emitted block sequence is identical to what that pass
                    // would produce.
                    if (op.taken_block) {
                        succs[key].push_back(*op.taken_block);
                    }
                    if (op.not_taken_block) {
                        succs[key].push_back(*op.not_taken_block);
                    }
                    if (op.target_block) {
                        succs[key].push_back(*op.target_block);
                    }
                    for (const auto &s : op.successor_blocks) {
                        succs[key].push_back(s);
                    }
                    for (const auto &sc : op.switch_cases) {
                        succs[key].push_back(sc.target_block);
                    }
                    if (op.fallback_block.has_value()) {
                        succs[key].push_back(*op.fallback_block);
                    }
                }
            }

            // Iterative post-order DFS, then reverse to get RPO.
            std::vector< std::string > post_order;
            std::unordered_set< std::string > visited;
            std::stack< std::pair< std::string, bool > > stk;
            if (!function.entry_block.empty()
                && function.basic_blocks.contains(function.entry_block))
            {
                stk.push({function.entry_block, false});
            }
            while (!stk.empty()) {
                auto [key, expanded] = stk.top();
                stk.pop();
                if (expanded) {
                    post_order.push_back(key);
                    continue;
                }
                if (visited.count(key)) {
                    continue;
                }
                visited.insert(key);
                stk.push({key, true});
                for (const auto &s : succs[key]) {
                    if (!visited.count(s)) {
                        stk.push({s, false});
                    }
                }
            }
            std::reverse(post_order.begin(), post_order.end());

            // Append unreachable blocks in deterministic (address-sorted) order.
            std::vector< std::string > unreachable;
            for (const auto &[key, block] : function.basic_blocks) {
                if (!visited.count(key)) {
                    unreachable.push_back(key);
                }
            }
            std::sort(unreachable.begin(), unreachable.end(), BlockKeyComparator{});
            for (auto &key : unreachable) {
                post_order.push_back(std::move(key));
            }

            return post_order;
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
    clang::FunctionDecl *FunctionBuilder::create_declaration(
        clang::ASTContext &ctx, const clang::QualType &function_type, bool is_definition
    ) {
        if (function.get().name.empty()) {
            LOG(ERROR) << "Function name is empty. function key " << function.get().key << "\n";
            return {};
        }

        auto location   = clang::SourceLocation();
        auto *func_decl = clang::FunctionDecl::Create(
            ctx, ctx.getTranslationUnitDecl(), location, location,
            &ctx.Idents.get(function.get().name), function_type,
            ctx.getTrivialTypeSourceInfo(function_type), clang::SC_None
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

        auto parameters     = getParameters(function);
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
            auto location = sourceLocation(ctx.getSourceManager(), param_op->key);

            auto *param_decl = clang::ParmVarDecl::Create(
                ctx, func_decl, location, location, &ctx.Idents.get(*param_op->name),
                param_type, nullptr, clang::SC_None, nullptr
            );
            parameter_vec.push_back(param_decl);

            // If this is for definition
            if (is_definition) {
                local_variables.emplace(param_op->key, param_decl);
            }
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
        ext_proto_info.Variadic           = proto.is_variadic;
        ext_proto_info.ExceptionSpec.Type = clang::EST_None;

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
                ctx, func_decl, sourceLocation(ctx.getSourceManager(), param_key),
                sourceLocation(ctx.getSourceManager(), param_key),
                &ctx.Idents.get(parameter_name(index++)), param_type,
                ctx.getTrivialTypeSourceInfo(param_type, clang::SourceLocation()),
                clang::SC_None, nullptr
            );
            assert(param_decl != nullptr);

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
        if (function.get().name.empty()) {
            LOG(ERROR) << "Can't create function definition. Missing function name.\n";
            return {};
        }

        if (function.get().basic_blocks.empty()) {
            LOG(ERROR) << "Can't create function definition for '" << function.get().name << "'. Function has no basic blocks.\n";
            return {};
        }

        // if previous declaration exist use it's type
        auto function_type = prev_decl != nullptr
            ? prev_decl->getType()
            : create_function_type(ctx, function.get().prototype);

        auto *function_def = create_declaration(ctx, function_type, /*is_definition=*/true);
        if (function_def == nullptr) {
            LOG(ERROR) << "Failed to create function definition. key: " << function.get().key
                       << "\n";
            return {};
        }

        // Set previous declaration if exist
        function_def->setPreviousDecl(prev_decl);

        // Before creating function body, set sema context to the current function. It gets used
        // to set lexical and sema decl context for ast nodes.
        auto *prev_context = get_sema_context();
        set_sema_context(function_def);
        auto body_vec = create_function_body(ctx, function_def);
        function_def->setBody(clang::CompoundStmt::Create(
            ctx, body_vec, clang::FPOptionsOverride(),
            sourceLocation(ctx.getSourceManager(), function.get().key),
            sourceLocation(ctx.getSourceManager(), function.get().key)
        ));
        function_def->setWillHaveBody(true);
        set_sema_context(prev_context);

        return function_def;
    }

    /**
     * @brief Creates and registers label declarations for basic blocks in a function.
     *
     * @param ctx The Clang ASTContext used to create label declarations.
     * @param func_decl The Clang function declaration to which the labels belong.
     */
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
                ctx, func_decl, sourceLocation(ctx.getSourceManager(), key),
                &ctx.Idents.get(labelNameFromKey(key))
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

    /**
     * @brief Creates the body of a function in the form of a vector of Clang statements.
     *
     * This method constructs the Abstract Syntax Tree (AST) for a function body based on
     * its basic blocks. It processes the function's entry block first, then processes
     * remaining basic blocks, assigning labels to each block and appending the generated
     * statements in sequence.
     *
     * @param ctx The ASTContext object used for managing AST nodes.
     * @param func_decl The function declaration associated with the function body.
     *
     * @return A vector of Clang statements representing the function body.
     *         Returns an empty vector if the function has no basic blocks.
     */
    std::vector< clang::Stmt * > FunctionBuilder::create_function_body(
        clang::ASTContext &ctx, clang::FunctionDecl *func_decl
    ) {
        if (function.get().basic_blocks.empty()) {
            LOG(ERROR) << "Function " << function.get().name << " doesn't have body\n";
            return {};
        }

        std::vector< clang::Stmt * > stmt_vec;
        create_labels(ctx, func_decl);

        // Compute RPO block order from the P-Code CFG.  Emitting in RPO order
        // matches what BasicBlockReorderPass would produce, making that pass a
        // near-no-op and letting GotoCanonicalizePass see clean input earlier.
        const auto rpo = compute_rpo(function.get());

        // Emit the entry block first (no label).  Set current_next_block_key so
        // that create_branch can skip the goto when it targets the immediately
        // following block (RPO[1]).
        if (function.get().basic_blocks.contains(function.get().entry_block)) {
            current_next_block_key = (rpo.size() > 1) ? rpo[1] : std::string{};
            const auto &entry_block =
                function.get().basic_blocks.at(function.get().entry_block);
            auto entry_stmts = create_basic_block(ctx, entry_block);
            stmt_vec.insert(stmt_vec.end(), entry_stmts.begin(), entry_stmts.end());
        }

        // Emit non-entry blocks in RPO order with labels.  RPO[0] is the entry
        // block so the loop starts at index 1.
        for (size_t i = 1; i < rpo.size(); ++i) {
            const auto &key = rpo[i];
            if (!function.get().basic_blocks.contains(key)) {
                continue;
            }
            const auto &bb = function.get().basic_blocks.at(key);
            if (bb.is_entry_block) {
                continue;
            }

            LOG(INFO) << "Processing basic block with key " << key << "\n";

            // Expose the next block in RPO order so create_branch can elide
            // gotos to the immediately following block.
            current_next_block_key = (i + 1 < rpo.size()) ? rpo[i + 1] : std::string{};

            auto block_stmts = create_basic_block(ctx, bb);
            auto loc         = sourceLocation(ctx.getSourceManager(), key);
            clang::Stmt *first = block_stmts.empty()
                ? static_cast< clang::Stmt * >(new (ctx) clang::NullStmt(loc, false))
                : block_stmts[0];
            auto *label_stmt = new (ctx)
                clang::LabelStmt(loc, labels_declaration.at(key), first);
            if (block_stmts.empty()) {
                stmt_vec.push_back(label_stmt);
            } else {
                block_stmts[0] = label_stmt;
                stmt_vec.insert(stmt_vec.end(), block_stmts.begin(), block_stmts.end());
            }
        }

        return stmt_vec;
    }

    /**
     * @brief Generates a vector of `clang::Stmt*` representing the operations in a basic block.
     *
     * This function iterates over the `ordered_operations` of a `BasicBlock` object to create
     * corresponding `clang::Stmt` objects. Each statement is created using the
     * `create_operation` method and added to the vector `stmt_vec` unless the operation is
     * flagged to merge with the next.
     *
     * @param ctx The Clang ASTContext used for creating statements.
     * @param block The BasicBlock containing the operations to be processed.
     *
     * @return A vector of `clang::Stmt*` representing the operations in the basic block.
     *
     */
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
                // Drain any VarDecl materializations queued during create_operation
                // (e.g., from create_temporary promoting a cached expr into a VarDecl).
                // These must appear before the consuming statement in the output.
                for (auto *pending : pending_materialized) {
                    stmt_vec.push_back(pending);
                }
                pending_materialized.clear();

                operation_stmts.emplace(operation.key, stmt);
                if (!should_merge_to_next) {
                    stmt_vec.push_back(stmt);
                }
            }
        }

        return stmt_vec;
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
