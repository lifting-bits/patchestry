/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/PcodeValidator.hpp>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclarationName.h>
#include <clang/AST/Expr.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <clang/Analysis/CFG.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeTypes.hpp>
#include <patchestry/Util/Log.hpp>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace patchestry::ast {

    bool FunctionValidation::HasCritical() const {
        return llvm::any_of(flags, [](const ValidationFlag &flag) {
            return flag.severity == ValidationFlag::kCritical;
        });
    }

    bool FunctionValidation::HasWarning() const {
        return llvm::any_of(flags, [](const ValidationFlag &flag) {
            return flag.severity == ValidationFlag::kWarning;
        });
    }

    llvm::StringRef FunctionValidation::Verdict() const {
        if (HasCritical()) { return "fail"; }
        return HasWarning() ? "warn" : "pass";
    }

    const clang::FunctionDecl *
    FindFunctionDefinition(clang::ASTContext &ctx, llvm::StringRef name) {
        auto *tu = ctx.getTranslationUnitDecl();
        for (auto *decl : tu->lookup(clang::DeclarationName(&ctx.Idents.get(name)))) {
            if (const auto *function = llvm::dyn_cast< clang::FunctionDecl >(decl)) {
                if (const auto *definition = function->getDefinition()) { return definition; }
            }
        }
        return nullptr;
    }

    namespace {

        using ghidra::Function;
        using ghidra::Mnemonic;
        using ghidra::Operation;
        using ghidra::Program;
        using ghidra::Varnode;
        using ghidra::VarnodeType;

        std::string CNameOf(const Function &function) {
            return function.display_name.empty() ? function.name : function.display_name;
        }

        // Names the lifter itself introduces; never expected, never hallucinated.
        bool IsLifterHelper(llvm::StringRef name) {
            return name.starts_with("__builtin_") || name.starts_with("__patchestry_");
        }

        // ------------------------------------------------------------------
        // Facts from the P-Code model
        // ------------------------------------------------------------------

        struct Expected
        {
            std::map< std::string, unsigned > callees; // C name -> call sites
            std::map< std::string, std::set< unsigned > > callee_argc;
            // CALLOTHER userops: allowed, but the lifter may lower them to
            // builtins, so their absence is not an error.
            std::set< std::string > optional_callees;
            // CALLOTHER ops the lifter lowers to helper functions of its own
            // naming (atomic_*, __arm_*): that many calls to names the model
            // does not know are legitimate.
            unsigned callother_ops = 0;
            // Every C name the model declares; a call to any other name is a
            // lowered CALLOTHER or a hallucination.
            std::set< std::string > program_functions;
            unsigned indirect_calls = 0;
            std::set< std::string > strings;
            std::set< std::string > globals;
            unsigned returns    = 0;
            bool returns_value  = false;
            unsigned stores     = 0;
            unsigned conditions = 0;
            std::set< int64_t > switch_values;
        };

        Expected CollectExpected(const Function &function, const Program &program) {
            Expected expected;
            for (const auto &[key, known] : program.serialized_functions) {
                (void) key;
                expected.program_functions.insert(CNameOf(known));
                if (known.is_intrinsic) {
                    // Userop helper records: the lifter may call any of them.
                    expected.optional_callees.insert(CNameOf(known));
                }
            }
            auto note_varnode = [&](const Varnode &varnode) {
                if (varnode.string_value) { expected.strings.insert(*varnode.string_value); }
                if (varnode.kind == Varnode::VARNODE_GLOBAL && varnode.global) {
                    auto global = program.serialized_globals.find(*varnode.global);
                    // #226: a "global" at a function entry resolves as a function.
                    if (global != program.serialized_globals.end()
                        && !global->second.name.empty()
                        && !program.serialized_functions.contains(*varnode.global))
                    {
                        expected.globals.insert(SanitizeKeyToIdent(global->second.name));
                    }
                }
            };
            auto resolve_callee = [&](const Operation &op) -> const Function * {
                if (!op.target || !op.target->function) { return nullptr; }
                auto callee = program.serialized_functions.find(*op.target->function);
                return callee == program.serialized_functions.end() ? nullptr : &callee->second;
            };

            for (const auto &[block_key, block] : function.basic_blocks) {
                (void) block_key;
                for (const auto &op_key : block.ordered_operations) {
                    auto op_iter = block.operations.find(op_key);
                    if (op_iter == block.operations.end()) { continue; }
                    const Operation &op = op_iter->second;
                    for (const auto &input : op.inputs) { note_varnode(input); }
                    if (op.output) { note_varnode(*op.output); }
                    if (op.condition) { note_varnode(*op.condition); }
                    if (op.string_value) { expected.strings.insert(*op.string_value); }

                    switch (op.mnemonic) {
                        case Mnemonic::OP_CALL: {
                            if (const auto *callee = resolve_callee(op)) {
                                const auto name = CNameOf(*callee);
                                if (callee->is_intrinsic) {
                                    // Lowered like a CALLOTHER: the helper may be renamed.
                                    expected.optional_callees.insert(name);
                                    ++expected.callother_ops;
                                } else {
                                    ++expected.callees[name];
                                    expected.callee_argc[name].insert(
                                        static_cast< unsigned >(op.inputs.size())
                                    );
                                }
                            } else {
                                // No function record: the lifter emits a placeholder or
                                // a helper of its own naming.
                                ++expected.callother_ops;
                            }
                            break;
                        }
                        case Mnemonic::OP_CALLIND:
                            ++expected.indirect_calls;
                            break;
                        case Mnemonic::OP_CALLOTHER: {
                            ++expected.callother_ops;
                            if (const auto *callee = resolve_callee(op)) {
                                expected.optional_callees.insert(CNameOf(*callee));
                            }
                            break;
                        }
                        case Mnemonic::OP_STORE:
                            ++expected.stores;
                            break;
                        case Mnemonic::OP_RETURN:
                            ++expected.returns;
                            if (!op.inputs.empty()) { expected.returns_value = true; }
                            break;
                        case Mnemonic::OP_CBRANCH:
                            if (!op.condition
                                || op.condition->kind != Varnode::VARNODE_CONSTANT)
                            {
                                ++expected.conditions;
                            }
                            break;
                        case Mnemonic::OP_BRANCHIND:
                            for (const auto &switch_case : op.switch_cases) {
                                // The lifter drops cases whose target block is missing.
                                if (!switch_case.is_default
                                    && function.basic_blocks.contains(switch_case.target_block))
                                {
                                    expected.switch_values.insert(switch_case.value);
                                }
                            }
                            break;
                        default:
                            break;
                    }
                }
            }
            return expected;
        }

        // ------------------------------------------------------------------
        // Facts from the parsed C body
        // ------------------------------------------------------------------

        struct Actual
        {
            std::map< std::string, unsigned > callees;
            std::map< std::string, std::set< unsigned > > callee_argc;
            std::set< std::string > lib_builtins; // callees that are C library builtins
            unsigned indirect_calls = 0;
            std::set< std::string > strings;
            std::set< std::string > globals;
            unsigned returns            = 0;
            unsigned returns_with_value = 0;
            unsigned bare_returns       = 0;
            unsigned stores             = 0;
            unsigned conditions         = 0;
            std::set< int64_t > case_values;
            std::set< int64_t > int_literals;
            unsigned dup_assigns = 0;
        };

        // A write through memory, as opposed to a plain local assignment.
        bool IsMemoryLValue(const clang::Expr *expr) {
            expr = expr->IgnoreParenImpCasts();
            if (const auto *unary = llvm::dyn_cast< clang::UnaryOperator >(expr)) {
                return unary->getOpcode() == clang::UO_Deref;
            }
            if (llvm::isa< clang::ArraySubscriptExpr >(expr)) { return true; }
            if (const auto *member = llvm::dyn_cast< clang::MemberExpr >(expr)) {
                return member->isArrow() || IsMemoryLValue(member->getBase());
            }
            return false;
        }

        class BodyVisitor : public clang::RecursiveASTVisitor< BodyVisitor >
        {
          public:
            BodyVisitor(clang::ASTContext &ctx, Actual &actual) : ctx_(ctx), actual_(actual) {}

            bool VisitCallExpr(clang::CallExpr *call) {
                if (const auto *callee = call->getDirectCallee()) {
                    const auto name = callee->getNameAsString();
                    if (const unsigned id = callee->getBuiltinID(); id != 0) {
                        if (!ctx_.BuiltinInfo.isPredefinedLibFunction(id)) {
                            return true; // __builtin_* or a target intrinsic, not a P-Code CALL
                        }
                        actual_.lib_builtins.insert(name);
                    }
                    ++actual_.callees[name];
                    actual_.callee_argc[name].insert(call->getNumArgs());
                } else {
                    ++actual_.indirect_calls;
                }
                return true;
            }

            bool VisitStringLiteral(clang::StringLiteral *literal) {
                actual_.strings.insert(literal->getBytes().str());
                return true;
            }

            bool VisitDeclRefExpr(clang::DeclRefExpr *ref) {
                if (const auto *var = llvm::dyn_cast< clang::VarDecl >(ref->getDecl())) {
                    if (var->hasGlobalStorage() && !var->isStaticLocal()
                        && var->getDeclContext()->isTranslationUnit())
                    {
                        actual_.globals.insert(var->getNameAsString());
                    }
                }
                return true;
            }

            bool VisitReturnStmt(clang::ReturnStmt *ret) {
                ++actual_.returns;
                if (ret->getRetValue() != nullptr) {
                    ++actual_.returns_with_value;
                } else {
                    ++actual_.bare_returns;
                }
                return true;
            }

            bool VisitBinaryOperator(clang::BinaryOperator *binary) {
                if (binary->isAssignmentOp() && IsMemoryLValue(binary->getLHS())) {
                    ++actual_.stores;
                }
                if (binary->getOpcode() == clang::BO_LAnd
                    || binary->getOpcode() == clang::BO_LOr)
                {
                    ++actual_.conditions;
                }
                return true;
            }

            bool VisitIfStmt(clang::IfStmt *) {
                ++actual_.conditions;
                return true;
            }

            bool VisitWhileStmt(clang::WhileStmt *) {
                ++actual_.conditions;
                return true;
            }

            bool VisitDoStmt(clang::DoStmt *) {
                ++actual_.conditions;
                return true;
            }

            bool VisitForStmt(clang::ForStmt *loop) {
                if (loop->getCond() != nullptr) { ++actual_.conditions; }
                return true;
            }

            bool VisitConditionalOperator(clang::ConditionalOperator *) {
                ++actual_.conditions;
                return true;
            }

            bool VisitCaseStmt(clang::CaseStmt *case_stmt) {
                ++actual_.conditions;
                if (auto value = case_stmt->getLHS()->getIntegerConstantExpr(ctx_)) {
                    actual_.case_values.insert(value->getSExtValue());
                    actual_.case_values.insert(static_cast< int64_t >(value->getZExtValue()));
                }
                return true;
            }

            bool VisitIntegerLiteral(clang::IntegerLiteral *literal) {
                const auto &value = literal->getValue();
                if (value.getBitWidth() <= 64) {
                    actual_.int_literals.insert(value.getSExtValue());
                    actual_.int_literals.insert(static_cast< int64_t >(value.getZExtValue()));
                }
                return true;
            }

            // Two consecutive writes to the same lvalue with nothing in
            // between usually mean a lost guard or a lost use.
            bool VisitCompoundStmt(clang::CompoundStmt *compound) {
                std::string previous_lhs;
                for (auto *stmt : compound->body()) {
                    std::string lhs;
                    if (auto *binary = llvm::dyn_cast< clang::BinaryOperator >(stmt);
                        binary != nullptr && binary->getOpcode() == clang::BO_Assign)
                    {
                        lhs = Print(binary->getLHS());
                        if (!lhs.empty() && lhs == previous_lhs
                            && Print(binary->getRHS()).find(lhs) == std::string::npos)
                        {
                            ++actual_.dup_assigns;
                        }
                    }
                    previous_lhs = lhs;
                }
                return true;
            }

          private:
            std::string Print(const clang::Expr *expr) const {
                std::string text;
                llvm::raw_string_ostream os(text);
                expr->printPretty(os, nullptr, ctx_.getPrintingPolicy());
                return text;
            }

            clang::ASTContext &ctx_;
            Actual &actual_;
        };

        // Statements in CFG blocks that no path from the entry reaches.
        unsigned CountUnreachable(clang::ASTContext &ctx, const clang::FunctionDecl *function) {
            auto cfg = clang::CFG::buildCFG(
                function, function->getBody(), &ctx, clang::CFG::BuildOptions()
            );
            if (!cfg) { return 0; }
            std::unordered_set< const clang::CFGBlock * > reachable;
            llvm::SmallVector< const clang::CFGBlock *, 32 > worklist{ &cfg->getEntry() };
            while (!worklist.empty()) {
                const auto *block = worklist.pop_back_val();
                if (!reachable.insert(block).second) { continue; }
                for (const auto &adjacent : block->succs()) {
                    if (const auto *successor = adjacent.getReachableBlock()) {
                        worklist.push_back(successor);
                    }
                }
            }
            unsigned unreachable = 0;
            for (const auto *block : *cfg) {
                if (!reachable.contains(block) && !block->empty()) {
                    unreachable += static_cast< unsigned >(block->size());
                }
            }
            return unreachable;
        }

        // ------------------------------------------------------------------
        // Comparison
        // ------------------------------------------------------------------

        void AddFlag(
            FunctionValidation &out, llvm::StringRef code, ValidationFlag::Severity severity,
            std::string detail
        ) {
            if (severity == ValidationFlag::kCritical) {
                LOG(ERROR) << "validate: " << out.name << ": " << code << ": " << detail
                           << "\n";
            } else {
                LOG(WARNING) << "validate: " << out.name << ": " << code << ": " << detail
                             << "\n";
            }
            out.flags.push_back({ code.str(), severity, std::move(detail) });
        }

        void CheckSignature(
            clang::ASTContext &ctx, const clang::FunctionDecl *function, const Function &lifted,
            const Program &program, FunctionValidation &out
        ) {
            const auto &prototype     = lifted.prototype;
            const size_t expected_num = lifted.is_interrupt ? 0 : prototype.parameters.size();
            const bool expected_variadic = !lifted.is_interrupt && prototype.is_variadic;
            if (function->getNumParams() != expected_num) {
                AddFlag(
                    out, "SIGNATURE_MISMATCH", ValidationFlag::kCritical,
                    llvm::formatv(
                        "expected {0} parameter(s), found {1}", expected_num,
                        function->getNumParams()
                    )
                );
            }
            if (function->isVariadic() != expected_variadic) {
                AddFlag(
                    out, "SIGNATURE_MISMATCH", ValidationFlag::kCritical,
                    expected_variadic ? "prototype is variadic, C is not"
                                      : "C is variadic, prototype is not"
                );
            }

            auto type_of = [&](const std::string &key) -> const VarnodeType * {
                auto iter = program.serialized_types.find(key);
                return iter == program.serialized_types.end() ? nullptr : iter->second.get();
            };

            if (!lifted.is_interrupt) {
                if (const auto *ret = type_of(prototype.rttype_key)) {
                    const bool expected_void = ret->kind == VarnodeType::Kind::VT_VOID;
                    if (function->getReturnType()->isVoidType() != expected_void) {
                        AddFlag(
                            out, "SIGNATURE_MISMATCH", ValidationFlag::kCritical,
                            expected_void ? "prototype returns void, C returns a value"
                                          : "prototype returns a value, C returns void"
                        );
                    } else if (!expected_void && ret->size != 0) {
                        const auto c_size = ctx.getTypeSize(function->getReturnType()) / 8;
                        if (c_size != ret->size) {
                            AddFlag(
                                out, "SIGNATURE_TYPE_MISMATCH", ValidationFlag::kWarning,
                                llvm::formatv(
                                    "return type is {0} byte(s) in C, {1} in the prototype",
                                    c_size, ret->size
                                )
                            );
                        }
                    }
                }
                const auto num = std::min< size_t >(expected_num, function->getNumParams());
                for (size_t index = 0; index < num; ++index) {
                    const auto *expected_type = type_of(prototype.parameters[index]);
                    if (expected_type == nullptr || expected_type->size == 0) { continue; }
                    const auto param_type =
                        function->getParamDecl(static_cast< unsigned >(index))->getType();
                    const auto c_size    = ctx.getTypeSize(param_type) / 8;
                    const bool c_pointer = param_type->isPointerType();
                    const bool expected_pointer =
                        expected_type->kind == VarnodeType::Kind::VT_POINTER;
                    if (c_size != expected_type->size || c_pointer != expected_pointer) {
                        AddFlag(
                            out, "SIGNATURE_TYPE_MISMATCH", ValidationFlag::kWarning,
                            llvm::formatv(
                                "parameter {0} is {1} byte(s){2} in C, {3} byte(s){4} in the "
                                "prototype",
                                index, c_size, c_pointer ? " (pointer)" : "",
                                expected_type->size, expected_pointer ? " (pointer)" : ""
                            )
                        );
                    }
                }
            }
        }

        std::string Truncate(const std::string &text) {
            constexpr size_t kMax = 48;
            return text.size() <= kMax ? text : text.substr(0, kMax) + "...";
        }

        void CompareBodies(
            const Expected &expected, const Actual &actual, const clang::FunctionDecl *function,
            const ValidationOptions &options, FunctionValidation &out
        ) {
            // Calls.
            std::vector< std::string > lost;
            for (const auto &[name, count] : expected.callees) {
                auto found = actual.callees.find(name);
                if (found == actual.callees.end()) {
                    lost.push_back(name);
                    continue;
                }
                if (found->second != count) {
                    AddFlag(
                        out, "CALL_COUNT_DELTA", ValidationFlag::kWarning,
                        llvm::formatv(
                            "{0}: {1} call site(s) in P-Code, {2} in C", name, count,
                            found->second
                        )
                    );
                }
                auto expected_argc = expected.callee_argc.find(name);
                auto actual_argc   = actual.callee_argc.find(name);
                if (expected_argc != expected.callee_argc.end()
                    && actual_argc != actual.callee_argc.end()
                    && expected_argc->second.size() == 1)
                {
                    for (auto argc : actual_argc->second) {
                        if (!expected_argc->second.contains(argc)) {
                            AddFlag(
                                out, "CALL_ARGC_MISMATCH", ValidationFlag::kWarning,
                                llvm::formatv(
                                    "{0}: {1} argument(s) in C, {2} in P-Code", name, argc,
                                    *expected_argc->second.begin()
                                )
                            );
                        }
                    }
                }
            }
            // Calls to names the model does not declare are either lowered
            // CALLOTHER helpers or the lifter's rewrite of a callee (AArch64
            // outline atomics become atomic_* helpers); anything beyond that
            // budget is invented.
            size_t renamed = 0;
            for (const auto &[name, count] : actual.callees) {
                if (expected.callees.contains(name) || expected.optional_callees.contains(name)
                    || IsLifterHelper(name))
                {
                    continue;
                }
                if (!expected.program_functions.contains(name)
                    && (actual.lib_builtins.contains(name)
                        || llvm::StringRef(name).starts_with("__")))
                {
                    // The lifter's lowering of an opcode (FLOAT_CEIL -> ceilf) or a
                    // reserved-name target intrinsic (__wfi); never model symbols.
                    continue;
                }
                if (expected.callother_ops > 0) {
                    continue; // intrinsic lowering present: helper names are the lifter's
                }
                if (renamed < lost.size()) {
                    ++renamed; // the lifter's rename of a callee
                    continue;
                }
                AddFlag(
                    out, "CALL_HALLUCINATED", ValidationFlag::kCritical,
                    "call to " + name + " has no CALL in the P-Code"
                );
            }
            for (size_t index = 0; index < lost.size(); ++index) {
                if (index < renamed) {
                    AddFlag(
                        out, "CALL_REWRITTEN", ValidationFlag::kWarning,
                        "no call to " + lost[index]
                            + "; an unknown helper call may be its lowering"
                    );
                } else {
                    AddFlag(
                        out, "CALL_LOST", ValidationFlag::kCritical, "no call to " + lost[index]
                    );
                }
            }
            if (actual.indirect_calls < expected.indirect_calls) {
                AddFlag(
                    out, "CALLIND_DEFICIT", ValidationFlag::kWarning,
                    llvm::formatv(
                        "{0} indirect call(s) in P-Code, {1} in C", expected.indirect_calls,
                        actual.indirect_calls
                    )
                );
            }

            // Strings and globals.
            for (const auto &text : expected.strings) {
                if (!actual.strings.contains(text)) {
                    AddFlag(
                        out, "STRING_LOST", ValidationFlag::kCritical,
                        "string literal \"" + Truncate(text) + "\" is missing"
                    );
                }
            }
            for (const auto &name : expected.globals) {
                if (!actual.globals.contains(name)) {
                    AddFlag(
                        out, "GLOBAL_LOST", ValidationFlag::kCritical,
                        "global " + name + " is never referenced"
                    );
                }
            }
            for (const auto &name : actual.globals) {
                if (!expected.globals.contains(name)) {
                    AddFlag(
                        out, "GLOBAL_HALLUCINATED", ValidationFlag::kCritical,
                        "global " + name + " is not referenced by the P-Code"
                    );
                }
            }

            // Returns.
            const bool c_void = function->getReturnType()->isVoidType();
            if (!c_void && expected.returns_value) {
                if (actual.returns_with_value == 0) {
                    AddFlag(
                        out, "RET_MISMATCH", ValidationFlag::kCritical,
                        "non-void function never returns a value"
                    );
                }
                if (actual.bare_returns > 0) {
                    AddFlag(
                        out, "RET_MISMATCH", ValidationFlag::kCritical,
                        "bare return in a function that returns a value"
                    );
                }
            }
            if (c_void && actual.returns_with_value > 0) {
                AddFlag(
                    out, "RET_MISMATCH", ValidationFlag::kCritical,
                    "void function returns a value"
                );
            }
            if (actual.returns < expected.returns) {
                AddFlag(
                    out, "RET_COUNT_DELTA", ValidationFlag::kWarning,
                    llvm::formatv(
                        "{0} RETURN op(s) in P-Code, {1} return statement(s) in C",
                        expected.returns, actual.returns
                    )
                );
            }

            // Stores and conditions: deficits only, since the lifter may add
            // helper writes and the structurer may duplicate blocks.
            if (expected.stores > 0 && actual.stores < expected.stores) {
                const double deficit = static_cast< double >(expected.stores - actual.stores)
                    / static_cast< double >(expected.stores);
                if (deficit > options.store_warn_ratio) {
                    AddFlag(
                        out, "STORE_DEFICIT",
                        deficit > options.store_crit_ratio ? ValidationFlag::kCritical
                                                           : ValidationFlag::kWarning,
                        llvm::formatv(
                            "{0} STORE op(s) in P-Code, {1} memory write(s) in C",
                            expected.stores, actual.stores
                        )
                    );
                }
            }
            if (expected.conditions > 0 && actual.conditions < expected.conditions) {
                const double deficit =
                    static_cast< double >(expected.conditions - actual.conditions)
                    / static_cast< double >(expected.conditions);
                if (deficit > options.cond_warn_ratio) {
                    AddFlag(
                        out, "COND_DEFICIT", ValidationFlag::kWarning,
                        llvm::formatv(
                            "{0} CBRANCH op(s) in P-Code, {1} condition(s) in C",
                            expected.conditions, actual.conditions
                        )
                    );
                }
            }

            // Switch cases: as `case V:` or as a compared constant.
            for (auto value : expected.switch_values) {
                if (!actual.case_values.contains(value) && !actual.int_literals.contains(value))
                {
                    AddFlag(
                        out, "SWITCH_CASE_LOST", ValidationFlag::kCritical,
                        llvm::formatv("switch case {0} is missing", value)
                    );
                }
            }

            if (actual.dup_assigns > 0) {
                AddFlag(
                    out, "DUP_ASSIGN", ValidationFlag::kWarning,
                    llvm::formatv(
                        "{0} consecutive write(s) to the same lvalue", actual.dup_assigns
                    )
                );
            }
        }

    } // namespace

    ValidationReport ValidateAgainstPcode(
        clang::ASTContext &ctx, const Program &program,
        const std::vector< MarkerEntry > &markers, const ValidationOptions &options
    ) {
        ValidationReport report;
        for (const auto &marker : markers) {
            FunctionValidation result;
            result.key  = marker.key;
            result.name = marker.name;

            const Function *lifted = nullptr;
            if (auto iter = program.serialized_functions.find(marker.key);
                iter != program.serialized_functions.end())
            {
                lifted = &iter->second;
            } else {
                for (const auto &[key, function] : program.serialized_functions) {
                    if (CNameOf(function) == marker.name) {
                        lifted     = &function;
                        result.key = key;
                        break;
                    }
                }
            }

            if (lifted == nullptr) {
                AddFlag(
                    result, "UNKNOWN_FUNCTION", ValidationFlag::kWarning,
                    "no function with this key or name in the P-Code model"
                );
            } else if (
                const auto *function = FindFunctionDefinition(ctx, marker.name);
                function == nullptr
            )
            {
                AddFlag(
                    result, "FUNCTION_MISSING", ValidationFlag::kCritical,
                    "no definition in the C translation unit"
                );
            } else {
                CheckSignature(ctx, function, *lifted, program, result);
                const auto expected = CollectExpected(*lifted, program);
                Actual actual;
                BodyVisitor(ctx, actual).TraverseStmt(function->getBody());
                CompareBodies(expected, actual, function, options, result);
                if (const auto unreachable = CountUnreachable(ctx, function); unreachable > 0) {
                    AddFlag(
                        result, "UNREACHABLE_CODE", ValidationFlag::kWarning,
                        llvm::formatv("{0} statement(s) are unreachable", unreachable)
                    );
                }
            }

            if (result.HasCritical()) {
                ++report.failed;
            } else if (result.HasWarning()) {
                ++report.warned;
            }
            report.functions.push_back(std::move(result));
        }
        return report;
    }

    void WriteValidationReport(llvm::raw_ostream &os, const ValidationReport &report) {
        llvm::json::Object functions;
        for (const auto &function : report.functions) {
            llvm::json::Array flags;
            for (const auto &flag : function.flags) {
                flags.push_back(
                    llvm::json::Object{
                        {     "code",         flag.code                                     },
                        { "severity",
                         flag.severity == ValidationFlag::kCritical ? "critical" : "warning" },
                        {   "detail",                                            flag.detail },
                }
                );
            }
            functions[function.key] = llvm::json::Object{
                {    "name",      function.name },
                { "verdict", function.Verdict() },
                {   "flags",   std::move(flags) },
            };
        }
        llvm::json::Object root{
            {    "format",1                          },
            {   "summary",
             llvm::json::Object{
             { "functions", static_cast< int64_t >(report.functions.size()) },
             { "failed", static_cast< int64_t >(report.failed) },
             { "warned", static_cast< int64_t >(report.warned) },
             }                                 },
            { "functions", std::move(functions) },
        };
        os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
    }

} // namespace patchestry::ast
