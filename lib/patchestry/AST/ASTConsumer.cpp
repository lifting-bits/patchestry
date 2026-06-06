/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInvocation.h>
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
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/ExceptionSpecificationType.h>
#include <clang/Basic/LLVM.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/Specifiers.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/AST/BuildSNodeFromRegion.hpp>
#include <patchestry/AST/CfgDotEmitter.hpp>
#include <patchestry/AST/ClangEmitter.hpp>
#include <patchestry/AST/SNodePostPasses.hpp>
#include <patchestry/AST/FunctionBuilder.hpp>
#include <patchestry/AST/IntrinsicHandlers.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {

    namespace {
        // Emit CGraph blocks sequentially with goto-based terminals.
        // Used by --emit-flat-baseline and as the no-region fallback;
        // the post-pass chain drops residual gotos in the latter case.
        void EmitFlatCFG(
            CGraph &flow_graph, SNodeFactory &factory,
            clang::ASTContext &ctx, std::vector< SNode * > &root_body)
        {
            for (auto &node : flow_graph.nodes) {
                if (node.IsCollapsed()) continue;
                std::vector< SNode * > blk_nodes;
                for (auto *s : node.stmts)
                    if (s) blk_nodes.push_back(factory.Make< SStmt >(s));

                if (!node.switch_cases.empty() && node.branch_cond) {
                    auto *sw = factory.Make< SSwitch >(node.branch_cond);
                    // Use discriminant type to avoid truncation when
                    // uintptr_t > int; enums need the underlying integer
                    // type because getIntWidth requires a BuiltinType.
                    auto case_type = node.branch_cond->getType();
                    if (case_type->isEnumeralType()) {
                        case_type = case_type->castAs< clang::EnumType >()
                            ->getDecl()->getIntegerType();
                    }
                    unsigned case_width = ctx.getIntWidth(case_type);
                    for (const auto &sc : node.switch_cases) {
                        if (sc.is_default) {
                            if (sc.succ_index < node.succs.size()) {
                                auto &tn = flow_graph.Node(
                                    node.succs[sc.succ_index]);
                                if (tn.original_label.empty()) {
                                    LOG(FATAL) << "switch default target node "
                                               << node.succs[sc.succ_index]
                                               << " has no original_label — "
                                                  "CGraph builder bug.\n";
                                }
                                sw->SetDefaultBody(factory.Make< SGoto >(
                                    factory.Intern(tn.original_label)));
                            } else {
                                LOG(WARNING)
                                    << "switch default succ_index "
                                    << sc.succ_index
                                    << " out of range (succs="
                                    << node.succs.size() << ")\n";
                            }
                        } else {
                            auto *val = clang::IntegerLiteral::Create(
                                ctx,
                                llvm::APInt(case_width,
                                            static_cast< uint64_t >(sc.value),
                                            true),
                                case_type, VirtualLoc(ctx));
                            SNode *body = nullptr;
                            if (sc.succ_index < node.succs.size()) {
                                auto &tn = flow_graph.Node(
                                    node.succs[sc.succ_index]);
                                if (tn.original_label.empty()) {
                                    LOG(FATAL) << "switch case " << sc.value
                                               << " target node "
                                               << node.succs[sc.succ_index]
                                               << " has no original_label — "
                                                  "CGraph builder bug.\n";
                                }
                                body = factory.Make< SGoto >(
                                    factory.Intern(tn.original_label));
                            } else {
                                LOG(WARNING)
                                    << "switch case " << sc.value
                                    << " succ_index " << sc.succ_index
                                    << " out of range (succs="
                                    << node.succs.size() << ")\n";
                            }
                            sw->AddCase(val, body);
                        }
                    }
                    std::vector< SNode * > sw_seq = std::move(blk_nodes);
                    sw_seq.push_back(sw);
                    if (!node.label.empty()) {
                        root_body.push_back(factory.Make< SLabel >(
                            factory.Intern(node.label),
                            std::move(sw_seq)));
                    } else {
                        for (SNode *s : sw_seq)
                            root_body.push_back(s);
                    }
                    continue;
                }

                if (node.terminal)
                    blk_nodes.push_back(
                        factory.Make< SStmt >(node.terminal));
                if (!node.label.empty()) {
                    root_body.push_back(factory.Make< SLabel >(
                        factory.Intern(node.label),
                        std::move(blk_nodes)));
                } else {
                    for (SNode *s : blk_nodes)
                        root_body.push_back(s);
                }
            }
        }

    } // namespace

    void PcodeASTConsumer::HandleTranslationUnit(clang::ASTContext &ctx) {
        type_builder = std::make_unique< TypeBuilder >(ci.getASTContext());
        if (!get_program().serialized_types.empty()) {
            type_builder->create_types(ctx, get_program().serialized_types);
        }

        if (!get_program().serialized_globals.empty()) {
            create_globals(ctx, get_program().serialized_globals);
        }

        if (!get_program().serialized_functions.empty()) {
            // Pipeline: JSON -> CGraph -> SNode -> Clang AST. Functions
            // without basic blocks get forward declarations only.
            std::vector<std::shared_ptr<FunctionBuilder>> func_builders;
            const auto &program_arch = get_program().arch.value_or(std::string{});

            // Disambiguate duplicate C names across distinct function
            // *definitions*.  A binary can contain several file-local (`static`)
            // copies of one source function -- distinct code at distinct
            // addresses, all recovered with the same name (e.g. cyaml__log,
            // bv_value_unsigned).  Emitting each as a top-level definition
            // collides on a single C symbol, and the CIR backend rejects the
            // second ("Duplicate function definition").  Give every member of a
            // definition-bearing name collision a unique display_name suffixed
            // with its address.  GetCName() then yields distinct symbols; the
            // linker asm label (built from the raw `name`) and call edges
            // (resolved by key, not name) are unaffected, so this is lossless.
            {
                auto cname_of = [](const Function &f) -> const std::string & {
                    return f.display_name.empty() ? f.name : f.display_name;
                };
                std::unordered_map<std::string, int> def_name_count;
                for (auto &[key, function] : get_program().serialized_functions) {
                    if (function.basic_blocks.empty()) { continue; }
                    const auto &cname = cname_of(function);
                    if (!cname.empty()) { def_name_count[cname]++; }
                }
                for (auto &[key, function] : get_program().serialized_functions) {
                    if (function.basic_blocks.empty()) { continue; }
                    const auto &cname = cname_of(function);
                    if (cname.empty() || def_name_count[cname] < 2) { continue; }
                    std::string suffix = key;
                    if (auto colon = suffix.find(':'); colon != std::string::npos) {
                        suffix = suffix.substr(colon + 1);
                    }
                    for (char &c : suffix) {
                        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z')
                            || (c >= 'A' && c <= 'Z') || c == '_';
                        if (!ok) { c = '_'; }
                    }
                    std::string renamed = cname + "_" + suffix;
                    LOG(WARNING) << "Duplicate definition name '" << cname << "' at "
                                 << key << "; renaming to '" << renamed
                                 << "' to avoid a colliding C symbol.\n";
                    function.display_name = renamed;
                }
            }

            // For C names shared by >1 function with distinct return types
            // (e.g. intrinsic variants VectorSignedToFloat:tb / :t12), record a
            // canonical (widest) return type that FunctionBuilder normalizes
            // every variant to, so they emit one CIR symbol instead of colliding
            // on a NYI return-value bitcast. int->float userops additionally get
            // their undefined<N> result resolved to float/double here.
            std::unordered_map<std::string, std::unordered_set<std::string>>
                name_return_types;
            std::unordered_map<std::string, std::unordered_set<uint64_t>>
                name_return_sizes;
            std::unordered_map<std::string, clang::QualType> widest_return;
            // int->float intrinsic userops (gated on is_intrinsic, not name
            // alone). float_userop_unresolved = those with an unmappable width,
            // which fall back to the generic size-aware path.
            std::unordered_set<std::string> float_userop_names;
            std::unordered_set<std::string> float_userop_unresolved;
            const auto &serialized_types = type_builder->GetSerializedTypes();
            for (const auto &[key, function] : get_program().serialized_functions) {
                const auto &cname = function.display_name.empty()
                    ? function.name : function.display_name;
                if (cname.empty()) continue;
                auto it = serialized_types.find(function.prototype.rttype_key);
                if (it == serialized_types.end() || it->second.isNull()) continue;

                // Resolve an int->float userop's undefined<N> result to
                // float/double; an unmappable width keeps the raw type.
                clang::QualType ret_type = it->second;
                if (function.is_intrinsic && IsFloatReturningUserop(cname)) {
                    float_userop_names.insert(cname);
                    auto resolved = ResolveUseropFloatReturn(
                        ctx, cname, ctx.getTypeSize(ret_type));
                    if (!resolved.isNull()) {
                        ret_type = resolved;
                    } else {
                        float_userop_unresolved.insert(cname);
                    }
                }

                name_return_types[cname].insert(ret_type.getAsString());
                name_return_sizes[cname].insert(ctx.getTypeSize(ret_type));
                auto cur = widest_return.find(cname);
                if (cur == widest_return.end()
                    || ctx.getTypeSize(ret_type) > ctx.getTypeSize(cur->second)) {
                    widest_return[cname] = ret_type;
                }
            }
            // Per shared name: int->float userops and same-size variants
            // collapse to one canonical decl; different-size variants stay
            // distinct via return-type-suffixed symbols (no truncation).
            std::unordered_map<std::string, clang::QualType> canonical_returns;
            std::unordered_set<std::string> suffix_names;
            for (const auto &[cname, rets] : name_return_types) {
                if (float_userop_names.find(cname) != float_userop_names.end()
                    && float_userop_unresolved.find(cname)
                           == float_userop_unresolved.end())
                {
                    canonical_returns[cname] = widest_return[cname];
                    continue;
                }
                if (rets.size() <= 1) continue;
                if (name_return_sizes[cname].size() == 1) {
                    canonical_returns[cname] = widest_return[cname];
                } else {
                    suffix_names.insert(cname);
                }
            }

            for (const auto &[key, function] : get_program().serialized_functions) {
                auto builder = std::make_shared<FunctionBuilder>(
                    ci, function, *type_builder, function_declarations,
                    global_variable_declarations, intrinsic_declarations,
                    canonical_returns, suffix_names, program_arch
                );
                builder->InitializeOpBuilder();
                func_builders.emplace_back(std::move(builder));
            }

            for (auto &builder : func_builders) {
                if (!builder->has_basic_blocks()) continue;

                auto *fn = builder->create_definition(ctx);
                if (!fn) continue;

                const auto &func = builder->get_function();
                std::string fn_name = func.display_name.empty()
                    ? func.name : func.display_name;
                bool seeded_from_ghidra                = false;
                bool snode_cleanup_reached_fixed_point = true;
                // Set when the verifier finds a dangling goto.
                bool unrepaired_region_flow            = false;

                auto *prev_ctx = builder->enter_function_context(fn);
                CGraph flow_graph = BuildCGraph(*builder, ctx);
                builder->leave_function_context(prev_ctx);
                if (flow_graph.nodes.empty()) {
                    LOG(WARNING) << "BuildCGraph produced empty graph for "
                                 << fn_name << "\n";
                    continue;
                }

                if (options.emit_dot_cfg) {
                    CGraphDotTracer tracer;
                    tracer.fn_name = fn_name;
                    tracer.enabled = true;
                    tracer.Dump(flow_graph, "BuildCGraph", true);
                }

                SNodeFactory factory;
                std::vector<SNode *> root_body;
                // Pristine graph for the flat fallback, snapshotted
                // before seeding collapses nodes / clears labels.  Shallow copy
                // is safe: it shares leaf clang::Stmt*, but only one SNode tree
                // is ever emitted and passes never mutate leaf Stmts in place.
                CGraph flat_snapshot;

                if (options.emit_flat_baseline) {
                    EmitFlatCFG(flow_graph, factory, ctx, root_body);
                } else {
                    // Snapshot before seeding mutates flow_graph.
                    flat_snapshot = flow_graph;
                    // Seed from Function.region when present; fall back
                    // to flat-CFG. Post-pass chain runs on either shape.
                    if (func.region.has_value()) {
                        BuildSNodeFromRegion translator(func, flow_graph, factory, ctx);
                        seeded_from_ghidra = translator.TryBuild();
                    }
                    if (seeded_from_ghidra) {
                        // Region translator collapsed every node into entry.
                        for (auto &node : flow_graph.nodes) {
                            if (node.IsCollapsed()) continue;
                            for (SNode *s : node.structured)
                                root_body.push_back(s);
                        }
                    } else {
                        EmitFlatCFG(flow_graph, factory, ctx, root_body);
                    }

                    if (!root_body.empty()) {
                        // Lift raw clang control flow inside SStmt leaves
                        // into first-class SNodes so cleanup passes see it.
                        NormalizeRawControlFlow(root_body, factory, ctx);

                        ConvertGotoToBreakContinue(root_body, factory);
                        ConvertGotoToReturn(root_body, factory, ctx);
                        DuplicateSwitchCaseTargets(root_body, factory);

                        // Cross-region goto resolution: ordering is
                        // load-bearing — see each pass's hpp doc.
                        FoldSwitchLocalCaseTargets(root_body, factory, ctx);
                        DuplicateSmallTerminatingTargets(root_body, factory);
                        SplitAndCloneCrossScopeEntries(root_body, factory, ctx);
                        DuplicateSmallEpilogueTargets(root_body, factory, ctx);
                        DuplicateSwitchFallthroughTargets(root_body, factory, ctx);
                        DuplicateLoopContinueTargets(root_body, factory, ctx);
                        FoldGuardedFallthroughTargets(root_body, factory, ctx);
                        RepairCrossScopeLabelEntries(root_body, factory, ctx);
                        FoldSiblingArmLabelEntries(root_body, factory, ctx);
                        DuplicateStackGuardReturnTargets(root_body, factory, ctx);
                        DuplicateCleanupReturnTargets(root_body, factory, ctx);

                        // Run structural cleanup before goto cleanup so alias
                        // chains and empty shells are gone before InlineResidualGotos
                        // / EliminateGotoToNextLabel observe the tree.
                        SimplifyEmptyControlFlow(root_body, ctx);
                        CollapsePassThroughLabels(root_body, factory);
                        MergeRedundantGotoGuards(root_body, factory, ctx);

                        InlineResidualGotos(root_body, factory);
                        InlineCrossScopeSingleRef(root_body, factory);

                        // Fixed-point loop bounded by kMaxGotoEliminationPasses.
                        // Re-running cross-region resolvers is load-bearing:
                        // cloning one target can expose a fresh cross-region goto.
                        snode_cleanup_reached_fixed_point = false;
                        for (int pass = 0;
                             pass < kMaxGotoEliminationPasses; ++pass) {
                            bool did_absorb = AbsorbFallthroughIntoElse(
                                root_body, factory);
                            bool did_scope = ScopeifyIfGotos(
                                root_body, factory, ctx);
                            bool did_xscope = AbsorbCrossScopeIfGoto(
                                root_body, factory, ctx);
                            bool did_elim = EliminateGotoToNextLabel(
                                root_body, factory, ctx);
                            bool did_fold_sw = FoldSwitchLocalCaseTargets(
                                root_body, factory, ctx);
                            bool did_dup_term =
                                DuplicateSmallTerminatingTargets(
                                    root_body, factory);
                            bool did_split_clone =
                                SplitAndCloneCrossScopeEntries(
                                    root_body, factory, ctx);
                            bool did_dup_epi = DuplicateSmallEpilogueTargets(
                                root_body, factory, ctx);
                            bool did_dup_swf =
                                DuplicateSwitchFallthroughTargets(
                                    root_body, factory, ctx);
                            bool did_dup_lc = DuplicateLoopContinueTargets(
                                root_body, factory, ctx);
                            bool did_fold_gf = FoldGuardedFallthroughTargets(
                                root_body, factory, ctx);
                            bool did_repair = RepairCrossScopeLabelEntries(
                                root_body, factory, ctx);
                            bool did_sibling = FoldSiblingArmLabelEntries(
                                root_body, factory, ctx);
                            bool did_dup_sg =
                                DuplicateStackGuardReturnTargets(
                                    root_body, factory, ctx);
                            bool did_dup_cl = DuplicateCleanupReturnTargets(
                                root_body, factory, ctx);
                            bool did_inline = InlineResidualGotos(
                                root_body, factory);
                            bool did_cross = InlineCrossScopeSingleRef(
                                root_body, factory);
                            if (!did_absorb && !did_scope && !did_xscope
                                && !did_elim && !did_fold_sw && !did_dup_term
                                && !did_split_clone && !did_dup_epi
                                && !did_dup_swf && !did_dup_lc && !did_fold_gf
                                && !did_repair && !did_sibling
                                && !did_dup_sg && !did_dup_cl && !did_inline
                                && !did_cross) {
                                snode_cleanup_reached_fixed_point = true;
                                break;
                            }
                        }
                        if (!snode_cleanup_reached_fixed_point) {
                            // Surfaces pass oscillation; convergence is <=3 iters on corpus.
                            LOG(WARNING)
                                << "goto-elimination cleanup hit cap of "
                                << kMaxGotoEliminationPasses
                                << " passes for " << fn_name
                                << " — possible pass oscillation\n";
                        }

                        RemoveDeadSSeqChildren(root_body);

                        // Final bounded fixed point so any late-exposed region
                        // entry/exit goto patterns still get repaired.
                        FinalizeRegionRepairs(root_body, factory, ctx);

                        // Gate: a dangling goto is a recoverable Diag Error +
                        // flat fallback below, not an abort.  All output modes,
                        // so --print-tu is guarded too.
                        auto verify =
                            VerifyRegionRepairedBeforeLowering(root_body);
                        if (verify.hasFatalErrors()) {
                            unrepaired_region_flow = true;
                            auto diag_id = ci.getDiagnostics().getCustomDiagID(
                                clang::DiagnosticsEngine::Error,
                                "invalid structured-region control flow "
                                "before lowering in function %0: %1");
                            constexpr size_t kMaxReportedVerifierErrors = 5;
                            for (size_t i = 0;
                                 i < verify.diagnostics.size()
                                 && i < kMaxReportedVerifierErrors;
                                 ++i)
                            {
                                ci.getDiagnostics().Report(diag_id)
                                    << fn_name << verify.diagnostics[i];
                            }
                            if (verify.diagnostics.size()
                                > kMaxReportedVerifierErrors)
                            {
                                ci.getDiagnostics().Report(diag_id)
                                    << fn_name
                                    << (std::to_string(
                                            verify.diagnostics.size()
                                            - kMaxReportedVerifierErrors)
                                        + " additional unrepaired "
                                          "region-control-flow issues");
                            }
                        }
                    }
                }

                auto emit_body = [&](std::vector< SNode * > &body) {
                    EmitClangAST(body, fn, ctx);
                    if (options.clang_ast_cleanup) {
                        CleanupPrettyPrint(fn, ctx);
                    }
                };

                if (unrepaired_region_flow) {
                    // Salvage the function as goto-based flat C instead of
                    // dropping it.  The Diag Error above already made it loud.
                    LOG(ERROR)
                        << "Falling back to flat goto-based AST for " << fn_name
                        << " (dangling goto, issue #265); see prior diagnostics.\n";

                    std::vector< SNode * > flat_body;
                    EmitFlatCFG(flat_snapshot, factory, ctx, flat_body);

                    // Flat CFG should never dangle; guard anyway.  Report a
                    // Clang Error too so tools that surface only diagnostics see
                    // that the body was omitted, not just the stderr log.
                    auto flat_verify =
                        VerifyRegionRepairedBeforeLowering(flat_body);
                    if (flat_verify.hasFatalErrors()) {
                        auto diag_id = ci.getDiagnostics().getCustomDiagID(
                            clang::DiagnosticsEngine::Error,
                            "flat fallback for function %0 still has unresolved "
                            "gotos; emitting no body");
                        ci.getDiagnostics().Report(diag_id) << fn_name;
                        LOG(ERROR)
                            << "Flat fallback for " << fn_name
                            << " still has unresolved gotos; skipping "
                               "Clang-AST emission.\n";
                    } else {
                        emit_body(flat_body);
                    }
                } else {
                    emit_body(root_body);
                }
            }
        }

        // A FunctionDecl and a VarDecl sharing a C identifier collide as one
        // MLIR symbol and trip an isa<GlobalOp>/isa<FuncOp> assertion in vendored
        // CIRGen.  The #226 check above covers same-address clashes; this catches
        // same-name ones (e.g. "__errno" func vs "errno" global).  Fail loudly here.
        {
            std::unordered_set< std::string > function_names;
            std::unordered_set< std::string > variable_names;
            for (const auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                const auto *named = llvm::dyn_cast< clang::NamedDecl >(decl);
                if (!named || !named->getIdentifier()) {
                    continue;
                }
                auto name = named->getName().str();
                if (llvm::isa< clang::FunctionDecl >(decl)) {
                    function_names.insert(name);
                } else if (llvm::isa< clang::VarDecl >(decl)) {
                    variable_names.insert(name);
                }
            }
            for (const auto &name : function_names) {
                if (variable_names.count(name)) {
                    LOG_FATAL("symbol name collision: '{0}' is declared as both a "
                              "function and a global variable.  These map to a single "
                              "MLIR module symbol and would trip a CIRGen assertion.  "
                              "This usually means name sanitization collapsed two "
                              "distinct binary symbols onto one identifier.",
                              name);
                }
            }
        }

        if (options.print_tu) {
            // Pretty-print before codegen so the C file emits even when
            // CIR lowering later fails.
            if (!options.output_file.empty()) {
                std::error_code ec;
                llvm::raw_fd_ostream out(options.output_file + ".c", ec,
                                         llvm::sys::fs::OF_Text);
                if (!ec) {
                    ctx.getTranslationUnitDecl()->print(
                        out, ctx.getPrintingPolicy(), /*Indentation=*/0);
                } else {
                    LOG(ERROR) << "Failed to write C output: " << ec.message() << "\n";
                }
            } else {
                ctx.getTranslationUnitDecl()->print(
                    llvm::outs(), ctx.getPrintingPolicy(), /*Indentation=*/0);
            }
#ifdef ENABLE_DEBUG
            ctx.getTranslationUnitDecl()->dumpColor();
#endif
        }
    }

    void PcodeASTConsumer::set_sema_context(clang::DeclContext *dc) { sema().CurContext = dc; }

    void PcodeASTConsumer::write_to_file(void) {}

    void PcodeASTConsumer::create_globals(
        clang::ASTContext &ctx, VariableMap &serialized_variables
    ) {
        const auto &fns = get_program().serialized_functions;
        for (auto &[key, variable] : serialized_variables) {
            if (variable.name.empty() || variable.type.empty()) {
                continue;
            }

            // Stale JSON without the #226 producer fix would trip CIRGen's
            // FuncOp assertion deep in vendor code — refuse loudly here.
            if (fns.find(key) != fns.end()) {
                LOG_FATAL("create_globals: global '{0}' at {1} collides with a "
                          "function entry of the same address — pcode.json is "
                          "malformed.  Regenerate it with a PcodeSerializer "
                          "that includes the #226 fix.",
                          variable.name, key);
            }

            auto var_type       = type_builder->GetSerializedType(variable.type);
            if (var_type.isNull()) {
                continue;
            }
            auto location       = SourceLocation(ctx.getSourceManager(), key);
            auto sanitized_name = SanitizeKeyToIdent(variable.name);
            auto *var_decl      = clang::VarDecl::Create(
                ctx, ctx.getTranslationUnitDecl(), location, location,
                &ctx.Idents.get(sanitized_name), var_type,
                ctx.getTrivialTypeSourceInfo(var_type), clang::SC_Extern
            );
            var_decl->setIsUsed();
            var_decl->setDeclContext(ctx.getTranslationUnitDecl());
            ctx.getTranslationUnitDecl()->addDecl(var_decl);
            global_variable_declarations.emplace(variable.key, var_decl);
        }
    }

} // namespace patchestry::ast
