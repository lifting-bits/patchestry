/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Linker/Linker.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>

#include <patchestry/Util/Log.hpp>

#include <cctype>
#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
    const llvm::cl::opt< std::string > input_filename(
        llvm::cl::Positional, llvm::cl::desc("<input LLVM IR file>"), llvm::cl::init("-")
    );

    const llvm::cl::opt< std::string > output_filename(
        "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
        llvm::cl::init("-")
    );

    const llvm::cl::opt< std::string > target_function(
        "target-function", llvm::cl::desc("Function to verify (required)"),
        llvm::cl::value_desc("name"), llvm::cl::Required
    );

    const llvm::cl::opt< bool >
        emit_ll("S", llvm::cl::desc("Emit LLVM IR instead of bitcode"), llvm::cl::init(false));

    const llvm::cl::opt< std::string > model_library(
        "model-library",
        llvm::cl::desc("LLVM IR/bitcode file with external function models (linked before stubbing)"),
        llvm::cl::value_desc("file"), llvm::cl::init("")
    );

    const llvm::cl::opt< bool >
        verbose("v", llvm::cl::desc("Enable verbose output"), llvm::cl::init(false));

    const llvm::cl::opt< unsigned > symbolic_ptr_size(
        "symbolic-ptr-size",
        llvm::cl::desc("Default symbolic buffer size (bytes) for pointer arguments"),
        llvm::cl::value_desc("bytes"), llvm::cl::init(256)
    );

    // Runtime depth cap for the recursive globals initializer. Self-
    // referential and mutually-recursive types emit a call to the
    // already-being-built per-type init function at codegen time (the
    // TypeInitCache is the cycle breaker), so a runtime guard is needed
    // to stop KLEE from evaluating malloc -> init -> malloc -> init …
    // forever. At depth >= max the pointer field is set to null and
    // the recursive call is skipped. 2 is enough for list.next.next;
    // raise for deeper structures (trees, chains) at the cost of larger
    // symbolic states.
    const llvm::cl::opt< unsigned > klee_init_max_depth(
        "klee-init-max-depth",
        llvm::cl::desc("Runtime depth cap for recursive pointee initialization"),
        llvm::cl::value_desc("depth"), llvm::cl::init(2)
    );

    // Arrays longer than this collapse to a single flat klee_make_symbolic
    // call over the whole storage (matching the pre-change behavior for
    // the whole module). Keeps generated IR bounded on modules with huge
    // static tables. Pointer fields inside the fall-back array are then
    // left as symbolic bytes — a known limitation documented in the plan.
    const llvm::cl::opt< unsigned > klee_init_array_expand_limit(
        "klee-init-array-expand-limit",
        llvm::cl::desc("Above this length, arrays of aggregates fall back to flat byte symbolization"),
        llvm::cl::value_desc("elements"), llvm::cl::init(64)
    );

    // When any contract predicate fails to parse, a silently-passing
    // verifier run is worse than one that refuses to run: the resulting
    // harness is under-constrained and can "verify" against a weaker
    // condition than the author intended. Default-on so typos in contract
    // YAML always surface; set to false to fall back to the old
    // warn-and-continue behavior (e.g. for CI jobs that parse partial
    // contracts intentionally).
    const llvm::cl::opt< bool > strict_contracts(
        "strict-contracts",
        llvm::cl::desc("Exit non-zero when any contract predicate fails to parse"),
        llvm::cl::init(true)
    );
} // namespace

// ============================================================================
// Predicate parsing — reused from patchir-seahorn-verifier
// ============================================================================
namespace {
    enum PredicateKind {
        PK_Unknown,
        PK_Nonnull,
        PK_RelNeqArgConst,
        PK_RelEqArgConst,
        PK_RelLtArgConst,
        PK_RelLeArgConst,
        PK_RelGtArgConst,
        PK_RelGeArgConst,
        PK_RangeRet,
        PK_RangeArg,
        PK_Alignment
    };

    struct ParsedPredicate {
        PredicateKind kind = PK_Unknown;
        std::string target;
        unsigned arg_index = 0;
        int64_t constant   = 0;
        int64_t min_val    = 0;
        int64_t max_val    = 0;
        uint64_t alignment = 0;
        bool is_precondition = true;
    };

    // Parse a contract-target string of the form `Arg(N)` or `ReturnValue`.
    //
    // On success, `target` is set to the canonical string form and `index`
    // to the parsed argument index (0 for `ReturnValue`). On any failure,
    // *both* out-parameters are reset to an unambiguously-invalid state
    // (`target` empty, `index = 0`) so a caller that accidentally ignores
    // the return value cannot silently reinterpret a malformed target as
    // `Arg(0)`. Downstream code then sees an empty target and its
    // PK_Range/PK_Nonnull/etc. dispatch naturally leaves `pred.kind`
    // at `PK_Unknown`, which `parseContractSection` drops.
    static bool parseTarget(const std::string &target_str, std::string &target, unsigned &index) {
        target.clear();
        index = 0;

        if (target_str.substr(0, 4) == "Arg(") {
            auto end_pos = target_str.find(')');
            if (end_pos == std::string::npos) {
                LOG(WARNING) << "malformed target '" << target_str
                             << "': missing ')'\n";
                return false;
            }
            std::string index_str = target_str.substr(4, end_pos - 4);
            try {
                index = static_cast< unsigned >(std::stoul(index_str));
            } catch (const std::exception &e) {
                LOG(WARNING) << "failed to parse argument index '"
                             << index_str << "': " << e.what() << "\n";
                return false;
            }
            target = target_str;
            return true;
        }

        if (target_str == "ReturnValue") {
            target = target_str;
            return true;
        }

        return false;
    }

    static size_t findMatchingBracket(const std::string &str, size_t start);

    static std::map< std::string, std::string >
    parseKeyValues(const std::string &pred_str) {
        std::map< std::string, std::string > kv;
        size_t pos = 0;

        while (pos < pred_str.length()) {
            size_t eq_pos = pred_str.find('=', pos);
            if (eq_pos == std::string::npos)
                break;

            std::string key = pred_str.substr(pos, eq_pos - pos);
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);

            pos            = eq_pos + 1;
            std::string value;

            if (pos < pred_str.length() && pred_str[pos] == '"') {
                pos++;
                size_t end_quote = pred_str.find('"', pos);
                if (end_quote != std::string::npos) {
                    value = pred_str.substr(pos, end_quote - pos);
                    pos   = end_quote + 1;
                }
            } else if (pos < pred_str.length() && pred_str[pos] == '[') {
                // Bracketed value — find matching ']'
                size_t bracket_end = findMatchingBracket(pred_str, pos + 1);
                if (bracket_end != std::string::npos) {
                    // Include content between brackets (exclusive)
                    value = pred_str.substr(pos + 1, bracket_end - pos - 1);
                    pos   = bracket_end + 1;
                } else {
                    value = pred_str.substr(pos);
                    pos   = pred_str.length();
                }
            } else {
                size_t next_delim = pred_str.find_first_of(",;]}", pos);
                if (next_delim != std::string::npos) {
                    value = pred_str.substr(pos, next_delim - pos);
                    pos   = next_delim;
                } else {
                    value = pred_str.substr(pos);
                    pos   = pred_str.length();
                }
            }

            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            kv[key] = value;

            if (pos < pred_str.length() && (pred_str[pos] == ',' || pred_str[pos] == ';'))
                pos++;
            while (pos < pred_str.length() && (pred_str[pos] == ' ' || pred_str[pos] == '\t'))
                pos++;
        }

        return kv;
    }

    static ParsedPredicate kvToPredicate(const std::map< std::string, std::string > &kv) {
        ParsedPredicate pred;

        auto kind_it = kv.find("kind");
        if (kind_it == kv.end()) {
            LOG(WARNING) << "predicate missing 'kind' key\n";
            return pred;
        }

        std::string kind_str = kind_it->second;

        auto target_it = kv.find("target");
        if (target_it != kv.end()) {
            if (!parseTarget(target_it->second, pred.target, pred.arg_index)) {
                LOG(WARNING) << "malformed target '" << target_it->second
                             << "' — skipping predicate\n";
                return pred; // kind remains PK_Unknown → predicate is skipped
            }
        }

        if (kind_str == "nonnull") {
            pred.kind = PK_Nonnull;
        } else if (kind_str == "relation") {
            auto rel_it = kv.find("relation");
            auto val_it = kv.find("value");

            if (rel_it != kv.end() && val_it != kv.end()) {
                std::string rel = rel_it->second;
                try {
                    pred.constant = std::stoll(val_it->second);

                    if (rel == "neq")
                        pred.kind = PK_RelNeqArgConst;
                    else if (rel == "eq")
                        pred.kind = PK_RelEqArgConst;
                    else if (rel == "lt")
                        pred.kind = PK_RelLtArgConst;
                    else if (rel == "lte")
                        pred.kind = PK_RelLeArgConst;
                    else if (rel == "gt")
                        pred.kind = PK_RelGtArgConst;
                    else if (rel == "gte")
                        pred.kind = PK_RelGeArgConst;
                } catch (const std::exception &e) {
                    LOG(WARNING) << "failed to parse relation value '"
                                 << val_it->second << "': " << e.what() << "\n";
                }
            }
        } else if (kind_str == "range") {
            // Tentatively classify by target — we'll revert to PK_Unknown
            // below if min/max parsing fails or the `range` field is
            // missing/empty, so a typo like `range=[min=oops,max=10]`
            // cannot ship as a [0, parsed-max] silently-narrower bound.
            PredicateKind tentative = PK_Unknown;
            if (pred.target == "ReturnValue") {
                tentative = PK_RangeRet;
            } else if (pred.target.substr(0, 3) == "Arg") {
                tentative = PK_RangeArg;
            }

            auto range_it = kv.find("range");
            if (tentative != PK_Unknown && range_it != kv.end()) {
                std::string range_str = range_it->second;
                size_t min_pos = range_str.find("min=");
                size_t max_pos = range_str.find("max=");

                bool min_ok = false;
                bool max_ok = false;

                if (min_pos != std::string::npos) {
                    min_pos += 4;
                    size_t min_end = range_str.find_first_of(",]", min_pos);
                    try {
                        pred.min_val = std::stoll(range_str.substr(min_pos, min_end - min_pos));
                        min_ok = true;
                    } catch (const std::exception &e) {
                        LOG(WARNING) << "failed to parse range min: " << e.what() << "\n";
                    }
                }

                if (max_pos != std::string::npos) {
                    max_pos += 4;
                    size_t max_end = range_str.find_first_of(",]", max_pos);
                    try {
                        pred.max_val = std::stoll(range_str.substr(max_pos, max_end - max_pos));
                        max_ok = true;
                    } catch (const std::exception &e) {
                        LOG(WARNING) << "failed to parse range max: " << e.what() << "\n";
                    }
                }

                // Require BOTH bounds: a one-sided range that silently
                // defaults the other side to 0 is almost certainly a
                // typo, and "Arg(0) ∈ [0, parsed-max]" is strictly
                // weaker than any intended full bound.
                if (min_ok && max_ok && pred.min_val <= pred.max_val) {
                    pred.kind = tentative;
                } else if (min_ok != max_ok) {
                    LOG(WARNING) << "range predicate requires both 'min' and "
                                    "'max'; got only one — dropping\n";
                } else if (min_ok && max_ok) {
                    LOG(WARNING) << "range predicate has min > max ("
                                 << pred.min_val << " > " << pred.max_val
                                 << ") — dropping\n";
                }
            }
        } else if (kind_str == "alignment") {
            auto align_it = kv.find("align");
            if (align_it == kv.end()) {
                LOG(WARNING) << "alignment predicate missing 'align' key\n";
            } else {
                try {
                    pred.alignment = std::stoull(align_it->second);
                    // Alignment 0 is meaningless (it would emit `x & -1 == 0`
                    // which is trivially true and wastes a constraint slot);
                    // treat it the same as a parse failure.
                    if (pred.alignment != 0) {
                        pred.kind = PK_Alignment;
                    } else {
                        LOG(WARNING) << "alignment predicate has align=0 — dropping\n";
                    }
                } catch (const std::exception &e) {
                    LOG(WARNING) << "failed to parse alignment '"
                                 << align_it->second << "': " << e.what() << "\n";
                }
            }
        } else {
            LOG(WARNING) << "unknown predicate kind '" << kind_str << "'\n";
        }

        return pred;
    }

    static size_t findMatchingBracket(const std::string &str, size_t start) {
        int depth = 1;
        for (size_t i = start; i < str.length(); ++i) {
            if (str[i] == '[')
                depth++;
            else if (str[i] == ']') {
                depth--;
                if (depth == 0)
                    return i;
            }
        }
        return std::string::npos;
    }

    // Parses one section (either "preconditions" or "postconditions") of a
    // contract string. Appends successfully parsed predicates to `preds` and
    // increments `dropped` for every `{...}` block whose contents fail to
    // produce a valid predicate, so the caller can surface silent drops.
    static void parseContractSection(
        const std::string &contract_str, llvm::StringRef section_key,
        bool is_precondition, std::vector< ParsedPredicate > &preds,
        unsigned &dropped
    ) {
        std::string key_eq = (section_key + "=[").str();
        size_t start_key = contract_str.find(key_eq);
        if (start_key == std::string::npos)
            return;
        size_t section_start = start_key + key_eq.length();
        size_t section_end   = findMatchingBracket(contract_str, section_start);
        if (section_end == std::string::npos)
            return;

        std::string section = contract_str.substr(section_start, section_end - section_start);
        size_t pos = 0;
        while (pos < section.length()) {
            size_t start = section.find('{', pos);
            if (start == std::string::npos)
                break;
            start++;
            size_t end = section.find('}', start);
            if (end == std::string::npos)
                break;
            std::string pred_str = section.substr(start, end - start);
            auto kv              = parseKeyValues(pred_str);
            auto pred            = kvToPredicate(kv);
            if (pred.kind != PK_Unknown) {
                pred.is_precondition = is_precondition;
                preds.push_back(pred);
            } else {
                dropped++;
                LOG(WARNING) << section_key << " predicate dropped: {"
                             << pred_str << "}\n";
            }
            pos = end + 1;
        }
    }

    static std::vector< ParsedPredicate >
    parseStaticContractText(const std::string &contract_str, unsigned &dropped) {
        std::vector< ParsedPredicate > preds;
        parseContractSection(contract_str, "preconditions", true, preds, dropped);
        parseContractSection(contract_str, "postconditions", false, preds, dropped);
        return preds;
    }
} // namespace

// ============================================================================
// KLEE harness generation
// ============================================================================
namespace {

    // Get or declare klee_make_symbolic: void klee_make_symbolic(void*, size_t, const char*)
    static llvm::FunctionCallee getKleeMakeSymbolic(llvm::Module &M) {
        auto &Ctx    = M.getContext();
        auto *voidTy = llvm::Type::getVoidTy(Ctx);
        auto *ptrTy  = llvm::PointerType::getUnqual(Ctx);
        auto *sizeTy = M.getDataLayout().getIntPtrType(Ctx);
        auto *FT     = llvm::FunctionType::get(voidTy, { ptrTy, sizeTy, ptrTy }, false);
        return M.getOrInsertFunction("klee_make_symbolic", FT);
    }

    // Get or declare klee_assume: void klee_assume(uintptr_t)
    // KLEE's runtime uses uintptr_t, so match the target's pointer width.
    static llvm::FunctionCallee getKleeAssume(llvm::Module &M) {
        auto &Ctx      = M.getContext();
        auto *voidTy   = llvm::Type::getVoidTy(Ctx);
        auto *intptrTy = M.getDataLayout().getIntPtrType(Ctx);
        auto *FT       = llvm::FunctionType::get(voidTy, { intptrTy }, false);
        return M.getOrInsertFunction("klee_assume", FT);
    }

    // Get or declare klee_abort: void klee_abort(void)
    // KLEE's klee_assert is a macro (not a function), so we use
    // if (!cond) klee_abort() to implement postcondition assertions.
    static llvm::FunctionCallee getKleeAbort(llvm::Module &M) {
        auto &Ctx    = M.getContext();
        auto *voidTy = llvm::Type::getVoidTy(Ctx);
        auto *FT     = llvm::FunctionType::get(voidTy, {}, false);
        return M.getOrInsertFunction("klee_abort", FT);
    }

    // Collect all GlobalVariable references from a function (transitively
    // through internal calls).
    //
    // Limitation: only direct calls are followed. Indirect calls (function
    // pointers, HAL dispatch tables, calls through aliases or bitcast
    // constant-exprs) are not traversed, so globals that are only reached
    // via an indirect dispatch will not be symbolized. KLEE will then see
    // their initializer values as concrete constants and may over-constrain
    // the search.
    //
    // Note: no longer called from the harness path as of the globals-init
    // rewrite (generateHarness now uses collectModuleGlobals for a
    // module-wide walk). Retained — marked maybe_unused — because the
    // transitive-reachability walk is the right building block for a
    // future reachability-based skip hook on the runtime dispatcher.
    [[maybe_unused]] static void collectGlobals(
        llvm::Function *F,
        std::set< llvm::GlobalVariable * > &out_globals,
        std::set< llvm::Function * > &visited_fns
    ) {
        if (!F || !visited_fns.insert(F).second)
            return;

        for (auto &BB : *F) {
            for (auto &I : BB) {
                for (unsigned i = 0; i < I.getNumOperands(); ++i) {
                    auto *op = I.getOperand(i);
                    if (auto *GV = llvm::dyn_cast< llvm::GlobalVariable >(op)) {
                        if (!GV->isConstant()) {
                            out_globals.insert(GV);
                        }
                    } else {
                        // Walk through ConstantExpr wrappers (bitcast, GEP)
                        // to find the underlying GlobalVariable.
                        auto *val = op->stripPointerCasts();
                        if (auto *GV = llvm::dyn_cast< llvm::GlobalVariable >(val)) {
                            if (!GV->isConstant()) {
                                out_globals.insert(GV);
                            }
                        }
                    }
                }
                if (auto *CB = llvm::dyn_cast< llvm::CallBase >(&I)) {
                    if (auto *callee = CB->getCalledFunction()) {
                        if (!callee->isDeclaration()) {
                            collectGlobals(callee, out_globals, visited_fns);
                        }
                    }
                }
            }
        }
    }

    // Collect static_contract metadata pertaining to the target function.
    //
    // The `static_contract` MDTuple schema is:
    //   operand(0) -> MDString: function name the contract applies to
    //   operand(1) -> MDString: serialised contract body
    //
    // We match on the *name* in operand(0) rather than deriving the target
    // from `CallBase::getCalledFunction()`. Pointer equality on the called
    // function is brittle: it returns null for indirect calls through a
    // function pointer (decompiled dispatch tables, callbacks), for calls
    // that go through a bitcast `ConstantExpr` (common when the Ghidra
    // decompiler emits a call-site signature that doesn't exactly match
    // the callee's definition), and for calls through a `GlobalAlias`.
    // Every one of those cases silently dropped contract predicates with
    // no diagnostic before. Since the decompiler stamps operand(0) at
    // attachment time, it's the authoritative identifier and covers all
    // call-site shapes uniformly.
    //
    // We walk the whole module once and collect any instruction whose
    // metadata names the target, regardless of whether it's a call site
    // or an instruction inside the target function's body.
    static std::vector< std::string >
    collectStaticContracts(llvm::Module &M, llvm::Function *target_fn) {
        std::vector< std::string > contracts;
        if (!target_fn)
            return contracts;

        llvm::StringRef target_name = target_fn->getName();

        for (auto &F : M) {
            for (auto &BB : F) {
                for (auto &I : BB) {
                    auto *contract_md = I.getMetadata("static_contract");
                    if (!contract_md)
                        continue;

                    auto *tuple = llvm::dyn_cast< llvm::MDTuple >(contract_md);
                    if (!tuple || tuple->getNumOperands() < 2)
                        continue;

                    auto *fn_name = llvm::dyn_cast< llvm::MDString >(tuple->getOperand(0));
                    if (!fn_name || fn_name->getString() != target_name)
                        continue;

                    auto *md_str = llvm::dyn_cast< llvm::MDString >(tuple->getOperand(1));
                    if (!md_str)
                        continue;

                    contracts.push_back(md_str->getString().str());
                }
            }
        }

        return contracts;
    }

    // Helper to extend integer value to i64 for range comparisons.
    // `is_signed` selects sign- vs zero-extension so that an operand's full
    // value range is preserved under the signed i64 comparison used by
    // PK_Range predicates. A range whose min is < 0 is treated as signed;
    // otherwise the operand is zero-extended so the full unsigned range
    // (e.g. a size_t) maps cleanly into non-negative i64.
    //
    // Operands wider than 64 bits are rejected rather than truncated —
    // silently dropping high bits can hide a real out-of-range value.
    static llvm::Value *
    toI64(llvm::IRBuilder<> &B, llvm::Value *V, bool is_signed) {
        if (!V || !V->getType()->isIntegerTy())
            return nullptr;

        auto *i64 = llvm::Type::getInt64Ty(B.getContext());
        if (V->getType() == i64)
            return V;

        unsigned width = V->getType()->getIntegerBitWidth();
        if (width > 64) {
            LOG(WARNING) << "range predicate on " << width
                         << "-bit operand exceeds i64; skipping\n";
            return nullptr;
        }

        return is_signed ? B.CreateSExt(V, i64) : B.CreateZExt(V, i64);
    }

    // Build a KLEE condition from a predicate and emit klee_assume or klee_assert
    static void emitKleePredicate(
        llvm::IRBuilder<> &B, llvm::Module &M,
        const ParsedPredicate &P, llvm::Value *arg_val, llvm::Value *ret_val
    ) {
        auto &Ctx    = M.getContext();
        auto *i64Ty  = llvm::Type::getInt64Ty(Ctx);

        llvm::Value *V = nullptr;
        if (P.target == "ReturnValue") {
            V = ret_val;
        } else if (P.target.substr(0, 3) == "Arg") {
            V = arg_val;
        }

        if (!V)
            return;

        llvm::Value *cond = nullptr;

        switch (P.kind) {
        case PK_Nonnull: {
            if (!V->getType()->isPointerTy())
                break;
            llvm::Value *null_val = llvm::ConstantPointerNull::get(
                llvm::cast< llvm::PointerType >(V->getType())
            );
            cond = B.CreateICmpNE(V, null_val);
            if (verbose)
                llvm::outs() << "  " << (P.is_precondition ? "Precondition" : "Postcondition")
                             << ": nonnull\n";
            break;
        }
        case PK_RelNeqArgConst: {
            if (V->getType()->isPointerTy()) {
                // ptr != 0 is equivalent to nonnull
                if (P.constant == 0) {
                    auto *null_ptr = llvm::ConstantPointerNull::get(
                        llvm::cast< llvm::PointerType >(V->getType()));
                    cond = B.CreateICmpNE(V, null_ptr);
                } else {
                    auto *intptr_ty = B.getIntPtrTy(M.getDataLayout());
                    auto *ptr_int   = B.CreatePtrToInt(V, intptr_ty);
                    cond = B.CreateICmpNE(
                        ptr_int, llvm::ConstantInt::get(intptr_ty,
                                     static_cast< uint64_t >(P.constant)));
                }
            } else if (V->getType()->isIntegerTy()) {
                cond = B.CreateICmpNE(V, llvm::ConstantInt::getSigned(V->getType(), P.constant));
            }
            break;
        }
        case PK_RelEqArgConst: {
            if (V->getType()->isPointerTy()) {
                if (P.constant == 0) {
                    auto *null_ptr = llvm::ConstantPointerNull::get(
                        llvm::cast< llvm::PointerType >(V->getType()));
                    cond = B.CreateICmpEQ(V, null_ptr);
                } else {
                    auto *intptr_ty = B.getIntPtrTy(M.getDataLayout());
                    auto *ptr_int   = B.CreatePtrToInt(V, intptr_ty);
                    cond = B.CreateICmpEQ(
                        ptr_int, llvm::ConstantInt::get(intptr_ty,
                                     static_cast< uint64_t >(P.constant)));
                }
            } else if (V->getType()->isIntegerTy()) {
                cond = B.CreateICmpEQ(V, llvm::ConstantInt::getSigned(V->getType(), P.constant));
            }
            break;
        }
        case PK_RelLtArgConst: {
            if (!V->getType()->isIntegerTy())
                break;
            cond = B.CreateICmpSLT(V, llvm::ConstantInt::getSigned(V->getType(), P.constant));
            break;
        }
        case PK_RelLeArgConst: {
            if (!V->getType()->isIntegerTy())
                break;
            cond = B.CreateICmpSLE(V, llvm::ConstantInt::getSigned(V->getType(), P.constant));
            break;
        }
        case PK_RelGtArgConst: {
            if (!V->getType()->isIntegerTy())
                break;
            cond = B.CreateICmpSGT(V, llvm::ConstantInt::getSigned(V->getType(), P.constant));
            break;
        }
        case PK_RelGeArgConst: {
            if (!V->getType()->isIntegerTy())
                break;
            cond = B.CreateICmpSGE(V, llvm::ConstantInt::getSigned(V->getType(), P.constant));
            break;
        }
        case PK_RangeRet:
        case PK_RangeArg: {
            if (!V->getType()->isIntegerTy())
                break;
            // Treat the operand as signed only if the contract's min is
            // negative; otherwise zero-extend so unsigned ranges (e.g.
            // [0, UINT32_MAX]) are preserved under the signed compares below.
            bool is_signed = P.min_val < 0;
            llvm::Value *v64 = toI64(B, V, is_signed);
            if (!v64)
                break;
            llvm::Value *lo = B.CreateICmpSGE(v64, llvm::ConstantInt::getSigned(i64Ty, P.min_val));
            llvm::Value *hi = B.CreateICmpSLE(v64, llvm::ConstantInt::getSigned(i64Ty, P.max_val));
            cond = B.CreateAnd(lo, hi);
            if (verbose)
                llvm::outs() << "  " << (P.is_precondition ? "Precondition" : "Postcondition")
                             << ": " << P.min_val << " <= val <= " << P.max_val << "\n";
            break;
        }
        case PK_Alignment: {
            if (!V->getType()->isPointerTy() || P.alignment == 0)
                break;
            llvm::Type *intptr_ty = B.getIntPtrTy(M.getDataLayout());
            llvm::Value *ptr_int  = B.CreatePtrToInt(V, intptr_ty);
            llvm::Value *align_val = llvm::ConstantInt::get(intptr_ty, P.alignment);
            llvm::Value *mod       = B.CreateURem(ptr_int, align_val);
            llvm::Value *zero      = llvm::ConstantInt::get(intptr_ty, 0);
            cond = B.CreateICmpEQ(mod, zero);
            break;
        }
        default:
            break;
        }

        if (!cond)
            return;

        if (P.is_precondition) {
            // klee_assume takes uintptr_t — match target pointer width
            auto *intptr_ty = M.getDataLayout().getIntPtrType(Ctx);
            llvm::Value *cond_intptr = B.CreateZExt(cond, intptr_ty);
            B.CreateCall(getKleeAssume(M), { cond_intptr });
        } else {
            // Postcondition: if (!cond) klee_abort()
            // KLEE's klee_assert is a macro, not a function, so we
            // emit a conditional branch to klee_abort() instead.
            auto *parent_fn = B.GetInsertBlock()->getParent();
            auto *abort_bb  = llvm::BasicBlock::Create(Ctx, "assert.fail", parent_fn);
            auto *cont_bb   = llvm::BasicBlock::Create(Ctx, "assert.cont", parent_fn);
            B.CreateCondBr(cond, cont_bb, abort_bb);

            B.SetInsertPoint(abort_bb);
            B.CreateCall(getKleeAbort(M), {});
            B.CreateUnreachable();

            B.SetInsertPoint(cont_bb);
        }
    }

    // Get or declare malloc: i8* malloc(size_t). KLEE intercepts this at
    // runtime and returns a tracked heap allocation that survives the
    // caller's stack frame — the pointer-returning stub path below relies
    // on this to avoid returning a dangling alloca. We declare it unconditionally
    // so every pointer stub lowers to the same call site.
    static llvm::FunctionCallee getMalloc(llvm::Module &M) {
        auto &Ctx    = M.getContext();
        auto *ptrTy  = llvm::PointerType::getUnqual(Ctx);
        auto *sizeTy = M.getDataLayout().getIntPtrType(Ctx);
        auto *FT     = llvm::FunctionType::get(ptrTy, { sizeTy }, /*isVarArg=*/false);
        return M.getOrInsertFunction("malloc", FT);
    }

    // Stub undefined external functions with symbolic return values
    static unsigned stubExternalFunctions(llvm::Module &M, llvm::Function *target_fn) {
        unsigned count = 0;
        auto make_sym   = getKleeMakeSymbolic(M);
        auto malloc_fn  = getMalloc(M);

        // Names we must never stub: the allocator family is intercepted
        // natively by KLEE, so the declarations have to reach KLEE intact.
        // If we stubbed e.g. `malloc`, the pointer-returning stub path below
        // would synthesize a `malloc` body that in turn calls `malloc` on
        // itself — infinite recursion, plus the original critical bug
        // (returning a pointer to the stub's own stack frame) is back.
        static const llvm::StringRef kKleeAllocators[] = {
            "malloc", "calloc", "realloc", "free",
            "posix_memalign", "valloc", "memalign",
        };

        // Collect declarations to stub (avoid modifying while iterating)
        std::vector< llvm::Function * > to_stub;
        for (auto &F : M) {
            if (!F.isDeclaration())
                continue;
            if (&F == target_fn)
                continue;

            llvm::StringRef name = F.getName();
            // Skip KLEE intrinsics
            if (name.starts_with("klee_"))
                continue;
            // Skip LLVM intrinsics
            if (name.starts_with("llvm."))
                continue;
            // Skip allocator family — KLEE intercepts these.
            bool is_allocator = false;
            for (auto allocator : kKleeAllocators) {
                if (name == allocator) {
                    is_allocator = true;
                    break;
                }
            }
            if (is_allocator)
                continue;

            to_stub.push_back(&F);
        }

        auto &Ctx    = M.getContext();
        auto *ptrTy  = llvm::PointerType::getUnqual(Ctx);
        auto *sizeTy = M.getDataLayout().getIntPtrType(Ctx);

        for (auto *F : to_stub) {
            llvm::Type *ret_ty = F->getReturnType();

            auto *BB = llvm::BasicBlock::Create(Ctx, "entry", F);
            llvm::IRBuilder<> B(BB);

            if (ret_ty->isVoidTy()) {
                B.CreateRetVoid();
            } else if (!ret_ty->isSized()) {
                // Unsized return type (e.g. opaque struct) — return undef
                B.CreateRet(llvm::UndefValue::get(ret_ty));
            } else if (ret_ty->isPointerTy()) {
                // For pointer-returning externals, hand the caller a fresh
                // *heap* allocation filled with symbolic bytes. We MUST NOT
                // use `alloca` here: an alloca lives in the stub's own stack
                // frame, so returning it would hand the caller a dangling
                // pointer — the caller's first load/store would hit freed
                // stack memory (KLEE reports this as an out-of-bound pointer
                // error, and the semantics are undefined in LLVM). `malloc`
                // is intercepted by KLEE and produces a tracked allocation
                // that outlives this call, so the returned pointer remains
                // valid for the duration of the caller's use.
                uint64_t buf_size     = symbolic_ptr_size;
                llvm::Value *size_val = llvm::ConstantInt::get(sizeTy, buf_size);
                llvm::Value *buf      = B.CreateCall(malloc_fn, { size_val });

                std::string sym_name  = (F->getName() + "_ret_buf").str();
                llvm::Value *name_str = B.CreateGlobalString(sym_name);
                B.CreateCall(make_sym, { buf, size_val, name_str });

                // `malloc` already returns an opaque pointer; with typed
                // pointers we'd need a bitcast, with opaque pointers it's
                // a no-op, so emit the bitcast unconditionally and let
                // LLVM fold it.
                llvm::Value *ret_ptr = B.CreateBitCast(buf, ret_ty);
                B.CreateRet(ret_ptr);
            } else {
                uint64_t ret_size = M.getDataLayout().getTypeAllocSize(ret_ty);
                if (ret_size == 0) {
                    // Zero-sized type — return undef
                    B.CreateRet(llvm::UndefValue::get(ret_ty));
                } else {
                    // Allocate space for return value, make it symbolic, return it
                    llvm::AllocaInst *ret_alloca = B.CreateAlloca(ret_ty);

                    std::string sym_name = (F->getName() + "_ret").str();
                    llvm::Value *cast_ptr = B.CreateBitCast(ret_alloca, ptrTy);
                    llvm::Value *size_val = llvm::ConstantInt::get(sizeTy, ret_size);
                    llvm::Value *name_str = B.CreateGlobalString(sym_name);
                    B.CreateCall(make_sym, { cast_ptr, size_val, name_str });

                    llvm::Value *ret_val = B.CreateLoad(ret_ty, ret_alloca);
                    B.CreateRet(ret_val);
                }
            }

            count++;
            if (verbose) {
                llvm::outs() << "  Stubbed external: " << F->getName() << "\n";
            }
        }

        return count;
    }

    // ========================================================================
    // Symbolic global-variable initialization
    //
    // Replaces the old inline-per-global klee_make_symbolic loop in
    // generateHarness with a four-stage design:
    //
    //   1. inferPointerFieldTypes — one linear pass over the module's
    //      instructions to recover pointee types for every struct field of
    //      type `ptr`. Under LLVM opaque pointers the `ptr` field in a
    //      struct carries no pointee info, so we reconstruct it from the
    //      types used at load/store/GEP/call sites that consume the field.
    //
    //   2. getOrCreatePerTypeInit — synthesizes one
    //      @__klee_init_type_<T>(ptr p, i32 depth) per struct type reachable
    //      from a symbolizable global. Critically, the Function* is inserted
    //      into a cache BEFORE its body is built. When the recursive body
    //      walker encounters a pointer field whose inferred pointee is the
    //      same (or an ancestor on the construction stack), the cache hit
    //      returns the existing in-progress function and the walker emits a
    //      plain call. That closes self-reference and longer type cycles at
    //      codegen time without any explicit on-stack bookkeeping.
    //
    //   3. getOrCreatePerGlobalInit + emitGlobalsDescriptorTable — per
    //      symbolizable global, a trivial wrapper
    //      `@__klee_init_g_<name>() { call @__klee_init_type_<T>(@g, 0); ret }`.
    //      All wrappers are enumerated in an internal `{name, init_fn}`
    //      descriptor table that future runtime hooks (skip lists, logging,
    //      constraint injection) can iterate.
    //
    //   4. getOrCreateInitGlobalsDispatcher — synthesizes @__klee_init_globals
    //      as a counted loop over the descriptor table. main()'s first
    //      instruction is a single call to this dispatcher.
    //
    // A runtime depth cap (`--klee-init-max-depth`, default 2) bounds the
    // live recursion of cyclic type initializers: at `depth >= max` the
    // pointer field is set to null and the recursive call is skipped,
    // so KLEE doesn't explode on self-referential lists/trees.
    // ========================================================================

    // Forward declaration — definition lives further down the file
    // (near the retargeting helpers). We need the predicate here to
    // short-circuit the fast-path in buildTypeInitBody.
    static bool typeContainsPointer(
        llvm::Type *T, llvm::SmallPtrSetImpl< llvm::Type * > &seen
    );

    // Inference result for a single `(StructType *, fieldIdx)` pointer field.
    //
    // The kinds are mutually exclusive. The resolution rule when multiple
    // use sites across the module produce different candidates is documented
    // in inferPointerFieldTypes and boils down to: prefer PointeeType (most
    // specific), then ScalarBytes, then FunctionPointer; on unresolvable
    // struct-vs-struct conflicts, degrade to Unknown.
    struct PointerFieldInference {
        enum Kind {
            // No useful signal from any use site. The field is either dead,
            // only ever stored to, or only accessed through operations that
            // do not carry a type. At codegen time this falls back to a
            // flat `symbolic_ptr_size`-byte malloc'd buffer — same behavior
            // the old code produced for every pointer field.
            Unknown,
            // The field is called indirectly (the loaded pointer is the
            // callee of a call site). At codegen time the field is set to
            // null and no buffer is allocated: KLEE will report a clean
            // null-function-pointer error if the target actually makes the
            // call, rather than jumping to arbitrary bits.
            FunctionPointer,
            // The field is used for byte-level pointer arithmetic (a GEP
            // whose source element type is i8, or a plain byte load). The
            // `size` is the largest observed offset + 1, used as a lower
            // bound on how many bytes the allocation must cover. Falls
            // back to `symbolic_ptr_size` if the observed offset is 0.
            ScalarBytes,
            // The field points at a concrete LLVM type (usually a struct)
            // discovered from a typed load/store/GEP of the loaded pointer.
            // This is the path that enables recursive typed initialization
            // and lets cycles close via the TypeInitCache.
            PointeeType,
        };

        Kind kind             = Unknown;
        llvm::Type *pointee   = nullptr; // Set for PointeeType
        uint64_t scalar_bytes = 0;       // Set for ScalarBytes
        llvm::FunctionType *fn_type = nullptr; // Set for FunctionPointer (may be null if mixed)
    };

    using PointerFieldInferenceMap =
        llvm::DenseMap< std::pair< llvm::StructType *, unsigned >,
                        PointerFieldInference >;

    // Merge a new candidate into the existing inference slot for a field.
    //
    // Resolution rules (documented in the plan):
    //   * PointeeType beats ScalarBytes beats FunctionPointer beats Unknown
    //     (more-specific wins).
    //   * Two PointeeType candidates with the *same* concrete type reinforce
    //     each other (no-op). Two PointeeType candidates with *different*
    //     concrete types degrade to Unknown — decompiled IR sometimes
    //     reuses a pointer slot across incompatible struct views (e.g.
    //     union-through-ptr) and we cannot safely pick one.
    //   * Two FunctionPointer candidates with different function types
    //     degrade to FunctionPointer with a null function type (still
    //     stored as null at runtime; the caller handles nullptr fn_type).
    //   * ScalarBytes candidates take the MAX of observed sizes (lower
    //     bound on the buffer we must allocate).
    static void mergeInference(
        PointerFieldInference &slot, const PointerFieldInference &cand
    ) {
        if (cand.kind == PointerFieldInference::Unknown)
            return;

        if (slot.kind == PointerFieldInference::Unknown) {
            slot = cand;
            return;
        }

        // Both non-Unknown. Handle same-kind merges, then mixed-kind.
        if (slot.kind == cand.kind) {
            switch (slot.kind) {
            case PointerFieldInference::PointeeType:
                if (slot.pointee != cand.pointee) {
                    // Irreconcilable struct-vs-struct conflict.
                    slot.kind    = PointerFieldInference::Unknown;
                    slot.pointee = nullptr;
                }
                break;
            case PointerFieldInference::FunctionPointer:
                if (slot.fn_type != cand.fn_type) {
                    slot.fn_type = nullptr; // Still FunctionPointer, but type-erased
                }
                break;
            case PointerFieldInference::ScalarBytes:
                if (cand.scalar_bytes > slot.scalar_bytes)
                    slot.scalar_bytes = cand.scalar_bytes;
                break;
            default:
                break;
            }
            return;
        }

        // Mixed kinds: prefer more specific.
        //   PointeeType > ScalarBytes > FunctionPointer > Unknown
        auto rank = [](PointerFieldInference::Kind k) {
            switch (k) {
            case PointerFieldInference::PointeeType:     return 3;
            case PointerFieldInference::ScalarBytes:     return 2;
            case PointerFieldInference::FunctionPointer: return 1;
            case PointerFieldInference::Unknown:         return 0;
            }
            return 0;
        };

        if (rank(cand.kind) > rank(slot.kind))
            slot = cand;
    }

    // Walk every instruction in the module and infer pointer-field pointee
    // types from the typed operations that consume the loaded field value.
    //
    // Why this works under opaque pointers: while `%struct.Foo` has no
    // pointee type on any of its `ptr` fields, the *operations* that
    // consume the loaded pointer value do carry types — every load/store/
    // GEP has an explicit source-element type, and every call site has a
    // function type. By collecting the typed operations that use the
    // loaded value for each `(ST, fieldIdx)` we can reconstruct the
    // high-level intent ("this is a Buffer*", "this is an fn pointer",
    // "this is an i8 array").
    //
    // The walk is a single linear pass (~O(module instructions)). It is
    // consulted but not mutated by stage 2, so no synchronization is
    // needed and no re-inference happens during harness generation.
    static PointerFieldInferenceMap
    inferPointerFieldTypes(llvm::Module &M) {
        PointerFieldInferenceMap result;

        // For each GEP of form `gep StructType, ptr base, i32 0, i32 fieldIdx`
        // where the field is a `ptr`, we trace users to recover the pointee
        // type. We do this in two phases to keep the code readable:
        //
        //   Phase A — find GEPs into pointer fields and their loaded values.
        //   Phase B — classify each loaded value's uses into a candidate.
        //
        // Store sites directly into the field also contribute a
        // FunctionPointer hint when the stored value is a global function.

        for (auto &F : M) {
            for (auto &BB : F) {
                for (auto &I : BB) {
                    auto *gep = llvm::dyn_cast< llvm::GetElementPtrInst >(&I);
                    if (!gep)
                        continue;

                    auto *ST = llvm::dyn_cast< llvm::StructType >(
                        gep->getSourceElementType());
                    if (!ST)
                        continue;

                    // We only care about `gep %ST, ptr, 0, fieldIdx` — a
                    // direct field access. Anything fancier (non-zero
                    // leading index, nested multi-index GEP) is harder to
                    // interpret and we skip it; the field still gets
                    // inferred via other sites that use the simple form.
                    if (gep->getNumIndices() != 2)
                        continue;

                    auto *zero_idx = llvm::dyn_cast< llvm::ConstantInt >(
                        gep->getOperand(1));
                    if (!zero_idx || !zero_idx->isZero())
                        continue;

                    auto *field_idx_val = llvm::dyn_cast< llvm::ConstantInt >(
                        gep->getOperand(2));
                    if (!field_idx_val)
                        continue;

                    unsigned fieldIdx = static_cast< unsigned >(
                        field_idx_val->getZExtValue());
                    if (fieldIdx >= ST->getNumElements())
                        continue;

                    llvm::Type *field_ty = ST->getElementType(fieldIdx);
                    if (!field_ty->isPointerTy())
                        continue;

                    auto key = std::make_pair(ST, fieldIdx);

                    // Classify uses of the GEP itself.
                    for (auto *U : gep->users()) {
                        // Store *into* the field: stored value may be a
                        // global function (strong FunctionPointer hint)
                        // or a typed pointer value (weak type hint).
                        if (auto *store = llvm::dyn_cast< llvm::StoreInst >(U)) {
                            if (store->getPointerOperand() != gep)
                                continue;
                            llvm::Value *stored = store->getValueOperand();
                            if (llvm::isa< llvm::Function >(stored)) {
                                PointerFieldInference cand;
                                cand.kind    = PointerFieldInference::FunctionPointer;
                                cand.fn_type = llvm::cast< llvm::Function >(stored)
                                                   ->getFunctionType();
                                mergeInference(result[key], cand);
                            }
                            continue;
                        }

                        // Load from the field: the loaded value is a
                        // `ptr` whose uses tell us the pointee type.
                        auto *load = llvm::dyn_cast< llvm::LoadInst >(U);
                        if (!load || !load->getType()->isPointerTy())
                            continue;

                        for (auto *LU : load->users()) {
                            // Typed load: `load Ty, ptr %loaded` ->
                            // pointee is Ty.
                            if (auto *inner_load =
                                    llvm::dyn_cast< llvm::LoadInst >(LU)) {
                                if (inner_load->getPointerOperand() != load)
                                    continue;
                                PointerFieldInference cand;
                                if (inner_load->getType()->isIntegerTy(8)) {
                                    cand.kind = PointerFieldInference::ScalarBytes;
                                    cand.scalar_bytes = 1;
                                } else {
                                    cand.kind    = PointerFieldInference::PointeeType;
                                    cand.pointee = inner_load->getType();
                                }
                                mergeInference(result[key], cand);
                                continue;
                            }
                            // Typed store: `store Ty %v, ptr %loaded`.
                            if (auto *inner_store =
                                    llvm::dyn_cast< llvm::StoreInst >(LU)) {
                                if (inner_store->getPointerOperand() != load)
                                    continue;
                                llvm::Type *VT = inner_store->getValueOperand()->getType();
                                PointerFieldInference cand;
                                if (VT->isIntegerTy(8)) {
                                    cand.kind = PointerFieldInference::ScalarBytes;
                                    cand.scalar_bytes = 1;
                                } else {
                                    cand.kind    = PointerFieldInference::PointeeType;
                                    cand.pointee = VT;
                                }
                                mergeInference(result[key], cand);
                                continue;
                            }
                            // Typed GEP through the loaded pointer:
                            // `gep Ty, ptr %loaded, ...` -> pointee is Ty.
                            if (auto *inner_gep =
                                    llvm::dyn_cast< llvm::GetElementPtrInst >(LU)) {
                                if (inner_gep->getPointerOperand() != load)
                                    continue;
                                llvm::Type *SET = inner_gep->getSourceElementType();
                                PointerFieldInference cand;
                                if (SET->isIntegerTy(8)) {
                                    // Byte-level pointer arithmetic. Use
                                    // the constant offset (if any) as a
                                    // lower bound on the needed buffer.
                                    uint64_t lower = 1;
                                    if (inner_gep->getNumIndices() == 1) {
                                        if (auto *CI = llvm::dyn_cast< llvm::ConstantInt >(
                                                inner_gep->getOperand(1))) {
                                            lower = CI->getZExtValue() + 1;
                                        }
                                    }
                                    cand.kind         = PointerFieldInference::ScalarBytes;
                                    cand.scalar_bytes = lower;
                                } else {
                                    cand.kind    = PointerFieldInference::PointeeType;
                                    cand.pointee = SET;
                                }
                                mergeInference(result[key], cand);
                                continue;
                            }
                            // Indirect call: the loaded pointer is the callee.
                            if (auto *call = llvm::dyn_cast< llvm::CallBase >(LU)) {
                                if (call->getCalledOperand() != load)
                                    continue;
                                PointerFieldInference cand;
                                cand.kind    = PointerFieldInference::FunctionPointer;
                                cand.fn_type = call->getFunctionType();
                                mergeInference(result[key], cand);
                                continue;
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    // Stage-1.5: collect the set of module-wide symbolizable globals.
    //
    // Widened from the old target-reachable walk (`collectGlobals` higher
    // up in this file) to every non-constant, defined global in the module.
    // Sound over-approximation: symbolic is a superset of concrete, so
    // initializing more globals never introduces false verdicts (it can
    // cost solver time, but not correctness). The dispatcher is the
    // future hook for narrowing this at runtime.
    //
    // Filters:
    //   * skip external declarations (no storage to symbolize)
    //   * skip constants (their value is authoritative)
    //   * skip zero-sized / unsized globals
    //   * skip LLVM-internal globals (@llvm.global_ctors etc.)
    //   * skip the tool's own synthesized descriptor/name storage, so we
    //     don't accidentally symbolize our own runtime data on re-runs
    static bool isToolSynthesizedGlobal(const llvm::GlobalVariable *GV) {
        llvm::StringRef name = GV->getName();
        return name.starts_with("__klee_") || name.starts_with(".str.klee_");
    }

    // Decompiled firmware exports frequently emit `@foo = external global
    // T` for module-scope storage whose name and size are known but whose
    // initializer was not in the captured translation unit. Left as a
    // plain declaration, the global has no storage, so the harness cannot
    // make it symbolic and any reference from the target function (e.g.
    // passing `@usb_g` as a call argument) produces an unlinked symbol at
    // KLEE time.
    //
    // Materialize every such external by attaching a zero initializer and
    // switching to internal linkage, so the subsequent collectModuleGlobals
    // walk treats it as first-class storage and the per-global init
    // dispatcher can hand it to klee_make_symbolic. Safe: the value about
    // to be injected is symbolic, so any concrete seed (including zero) is
    // immediately overwritten before the target runs.
    //
    // Only touch externals that appear sized and non-constant; skip any
    // LLVM-internal or tool-synthesized names to keep the pass idempotent
    // across re-runs on an already-processed module.
    static unsigned materializeExternalGlobals(llvm::Module &M) {
        unsigned materialized = 0;
        for (auto &GV : M.globals()) {
            if (!GV.isDeclaration())
                continue;
            if (GV.isConstant())
                continue;
            if (GV.getName().starts_with("llvm."))
                continue;
            if (isToolSynthesizedGlobal(&GV))
                continue;
            llvm::Type *VT = GV.getValueType();
            if (!VT->isSized())
                continue;
            GV.setInitializer(llvm::Constant::getNullValue(VT));
            GV.setLinkage(llvm::GlobalValue::InternalLinkage);
            ++materialized;
            if (verbose) {
                llvm::outs() << "[globals] materialized external "
                             << GV.getName().str() << " : " << *VT << "\n";
            }
        }
        return materialized;
    }

    static void collectModuleGlobals(
        llvm::Module &M, std::vector< llvm::GlobalVariable * > &out
    ) {
        auto &DL = M.getDataLayout();
        for (auto &GV : M.globals()) {
            if (GV.isDeclaration())
                continue;
            if (GV.isConstant())
                continue;
            if (GV.getName().starts_with("llvm."))
                continue;
            if (isToolSynthesizedGlobal(&GV))
                continue;
            llvm::Type *VT = GV.getValueType();
            if (!VT->isSized())
                continue;
            if (DL.getTypeAllocSize(VT) == 0)
                continue;
            out.push_back(&GV);
        }
    }

    // Emit a single-line, stable-ordered inventory of the globals the
    // harness has picked up. Sorted by name so diffs across runs are
    // readable; each line shows the mangled name, the allocation size
    // in bytes, and the original linkage/declaration flavour.
    //
    // Callers should gate on `verbose` before calling; the log is intended
    // for humans (patch authors, CI log readers) rather than a consumer.
    static void logCollectedGlobals(
        const std::vector< llvm::GlobalVariable * > &module_globals,
        const llvm::Module &M
    ) {
        auto &DL = M.getDataLayout();
        std::vector< llvm::GlobalVariable * > sorted(
            module_globals.begin(), module_globals.end());
        std::sort(sorted.begin(), sorted.end(),
            [](llvm::GlobalVariable *a, llvm::GlobalVariable *b) {
                return a->getName() < b->getName();
            });
        llvm::outs() << "[globals] " << sorted.size()
                     << " collected for initialization:\n";
        for (auto *GV : sorted) {
            uint64_t size = DL.getTypeAllocSize(GV->getValueType());
            llvm::outs() << "[globals]   " << GV->getName().str()
                         << "  size=" << size << "B  type=" << *GV->getValueType()
                         << "\n";
        }
    }

    // Cache mapping LLVM types to their synthesized per-type init function.
    // Critically, entries are inserted BEFORE the function body is built,
    // so a recursive call during body construction observes an existing
    // Function* for the in-progress type and short-circuits with a plain
    // call rather than infinitely re-entering the builder. This is the
    // codegen-time cycle breaker for self-referential and mutually-
    // recursive types.
    using TypeInitCache = llvm::DenseMap< llvm::Type *, llvm::Function * >;

    // Forward declarations — these four functions form a tight recursive
    // cluster (per-type init bodies invoke emitPointerField, which calls
    // getOrCreatePerTypeInit, which calls back into buildTypeInitBody).
    static llvm::Function *getOrCreatePerTypeInit(
        llvm::Module &M, llvm::Type *T,
        const PointerFieldInferenceMap &im, TypeInitCache &cache
    );
    static void buildTypeInitBody(
        llvm::IRBuilder<> &B, llvm::Value *addr, llvm::Value *depth,
        llvm::Type *T, const PointerFieldInferenceMap &im,
        TypeInitCache &cache, const llvm::Twine &name, llvm::Module &M
    );

    // Sanitize an LLVM type name so it can be embedded in a function
    // symbol. Non-identifier characters are replaced with '_'. Anonymous
    // and literal struct types get a "anon_<N>" suffix based on a running
    // counter so the generated symbols stay unique per type instance.
    static std::string mangleTypeName(llvm::Type *T) {
        std::string result;
        llvm::raw_string_ostream os(result);
        if (auto *ST = llvm::dyn_cast< llvm::StructType >(T)) {
            if (ST->hasName()) {
                for (char c : ST->getName()) {
                    if (std::isalnum(static_cast< unsigned char >(c)) || c == '_')
                        os << c;
                    else
                        os << '_';
                }
                return os.str();
            }
            // Literal / anonymous: use the type's address as a disambiguator.
            os << "anon_" << reinterpret_cast< uintptr_t >(ST);
            return os.str();
        }
        if (auto *AT = llvm::dyn_cast< llvm::ArrayType >(T)) {
            os << "arr" << AT->getNumElements() << "_"
               << mangleTypeName(AT->getElementType());
            return os.str();
        }
        if (T->isIntegerTy()) {
            os << "i" << T->getIntegerBitWidth();
            return os.str();
        }
        if (T->isFloatTy())   return "f32";
        if (T->isDoubleTy())  return "f64";
        if (T->isPointerTy()) return "ptr";
        os << "ty" << reinterpret_cast< uintptr_t >(T);
        return os.str();
    }

    // Emit the pointer-field initialization logic into the current insertion
    // point of B. Three branches, selected by the inference result:
    //
    //   * FunctionPointer  — store null. An indirect call through a null
    //                        function pointer surfaces as a clean KLEE
    //                        error instead of today's arbitrary-bits crash.
    //   * Unknown/ScalarBytes — malloc a flat symbolic byte buffer of
    //                           `symbolic_ptr_size` (or the inferred
    //                           ScalarBytes size), symbolize, store.
    //                           Matches the one-level-of-indirection
    //                           behavior the old harness had for pointer
    //                           arguments.
    //   * PointeeType     — emit a runtime depth check. At depth >= max
    //                       store null (bounds cyclic recursion). Otherwise
    //                       malloc(sizeof(pointee)), store, and call the
    //                       per-type init function for the pointee with
    //                       depth+1. The called function may be the one
    //                       we are currently building (self-reference) —
    //                       getOrCreatePerTypeInit returns the cached
    //                       in-progress Function*.
    static void emitPointerField(
        llvm::IRBuilder<> &B, llvm::Value *field_ptr, llvm::Value *depth,
        std::pair< llvm::StructType *, unsigned > key,
        const PointerFieldInferenceMap &im, TypeInitCache &cache,
        const llvm::Twine &name, llvm::Module &M
    ) {
        auto &Ctx    = M.getContext();
        auto *ptrTy  = llvm::PointerType::getUnqual(Ctx);
        auto *sizeTy = M.getDataLayout().getIntPtrType(Ctx);
        auto *i32Ty  = llvm::Type::getInt32Ty(Ctx);

        PointerFieldInference inf;
        auto it = im.find(key);
        if (it != im.end())
            inf = it->second;

        if (inf.kind == PointerFieldInference::FunctionPointer) {
            B.CreateStore(llvm::ConstantPointerNull::get(ptrTy), field_ptr);
            return;
        }

        auto make_sym = getKleeMakeSymbolic(M);
        auto malloc_fn = getMalloc(M);

        if (inf.kind == PointerFieldInference::Unknown ||
            inf.kind == PointerFieldInference::ScalarBytes)
        {
            uint64_t buf_size = symbolic_ptr_size;
            if (inf.kind == PointerFieldInference::ScalarBytes &&
                inf.scalar_bytes > buf_size)
            {
                buf_size = inf.scalar_bytes;
            }
            llvm::Value *size_val = llvm::ConstantInt::get(sizeTy, buf_size);
            llvm::Value *buf      = B.CreateCall(malloc_fn, { size_val });
            llvm::Value *name_str = B.CreateGlobalString(name.str());
            B.CreateCall(make_sym, { buf, size_val, name_str });
            B.CreateStore(buf, field_ptr);
            return;
        }

        // PointeeType — emit the runtime depth gate and recursive call.
        llvm::Type *pointee = inf.pointee;
        uint64_t pointee_size = M.getDataLayout().getTypeAllocSize(pointee);
        if (pointee_size == 0) {
            // Nothing to allocate; treat as Unknown flat buffer.
            llvm::Value *size_val = llvm::ConstantInt::get(sizeTy, symbolic_ptr_size);
            llvm::Value *buf      = B.CreateCall(malloc_fn, { size_val });
            llvm::Value *name_str = B.CreateGlobalString(name.str());
            B.CreateCall(make_sym, { buf, size_val, name_str });
            B.CreateStore(buf, field_ptr);
            return;
        }

        llvm::Function *parent_fn = B.GetInsertBlock()->getParent();
        auto *recurse_bb = llvm::BasicBlock::Create(Ctx, "init.recurse", parent_fn);
        auto *null_bb    = llvm::BasicBlock::Create(Ctx, "init.null", parent_fn);
        auto *cont_bb    = llvm::BasicBlock::Create(Ctx, "init.cont", parent_fn);

        llvm::Value *max_depth_val =
            llvm::ConstantInt::get(i32Ty, klee_init_max_depth);
        llvm::Value *too_deep = B.CreateICmpSGE(depth, max_depth_val);
        B.CreateCondBr(too_deep, null_bb, recurse_bb);

        // Null branch — bounds the cyclic recursion at runtime.
        B.SetInsertPoint(null_bb);
        B.CreateStore(llvm::ConstantPointerNull::get(ptrTy), field_ptr);
        B.CreateBr(cont_bb);

        // Recurse branch — allocate a typed pointee and call its init.
        B.SetInsertPoint(recurse_bb);
        llvm::Value *pointee_size_val =
            llvm::ConstantInt::get(sizeTy, pointee_size);
        llvm::Value *buf = B.CreateCall(malloc_fn, { pointee_size_val });
        B.CreateStore(buf, field_ptr);
        // Cache hit on cycle — returns the in-progress Function*.
        llvm::Function *init_fn = getOrCreatePerTypeInit(M, pointee, im, cache);
        llvm::Value *next_depth =
            B.CreateAdd(depth, llvm::ConstantInt::get(i32Ty, 1));
        B.CreateCall(init_fn, { buf, next_depth });
        B.CreateBr(cont_bb);

        B.SetInsertPoint(cont_bb);
    }

    // Recursive structural walker: drives the per-type init function body.
    // Terminates on aggregate leaves (no cycle detection needed inside
    // aggregates — LLVM rejects infinite-size types) and at pointer fields
    // (handled by emitPointerField, which uses the TypeInitCache to close
    // type-graph cycles).
    static void buildTypeInitBody(
        llvm::IRBuilder<> &B, llvm::Value *addr, llvm::Value *depth,
        llvm::Type *T, const PointerFieldInferenceMap &im,
        TypeInitCache &cache, const llvm::Twine &name, llvm::Module &M
    ) {
        auto &Ctx    = M.getContext();
        auto *ptrTy  = llvm::PointerType::getUnqual(Ctx);
        auto *sizeTy = M.getDataLayout().getIntPtrType(Ctx);
        auto &DL     = M.getDataLayout();

        // Fast path: no pointers transitively reachable from this type —
        // one flat klee_make_symbolic over the whole storage. Matches the
        // old per-global shape and avoids structural recursion when it
        // isn't needed.
        {
            llvm::SmallPtrSet< llvm::Type *, 8 > seen;
            if (!typeContainsPointer(T, seen)) {
                uint64_t size = DL.getTypeAllocSize(T);
                if (size == 0)
                    return;
                auto make_sym = getKleeMakeSymbolic(M);
                llvm::Value *size_val = llvm::ConstantInt::get(sizeTy, size);
                llvm::Value *name_str = B.CreateGlobalString(name.str());
                (void) ptrTy; // addr is already a ptr in opaque-pointer world
                B.CreateCall(make_sym, { addr, size_val, name_str });
                return;
            }
        }

        if (auto *ST = llvm::dyn_cast< llvm::StructType >(T)) {
            for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
                llvm::Type *field_ty = ST->getElementType(i);
                llvm::Value *field_ptr = B.CreateStructGEP(ST, addr, i);
                llvm::Twine child_name = name + ".f" + llvm::Twine(i);
                if (field_ty->isPointerTy()) {
                    emitPointerField(
                        B, field_ptr, depth, std::make_pair(ST, i),
                        im, cache, child_name, M
                    );
                } else {
                    buildTypeInitBody(
                        B, field_ptr, depth, field_ty, im, cache, child_name, M
                    );
                }
            }
            return;
        }

        if (auto *AT = llvm::dyn_cast< llvm::ArrayType >(T)) {
            uint64_t N = AT->getNumElements();
            llvm::Type *ET = AT->getElementType();

            // Fast-path scalar array — one flat call.
            {
                llvm::SmallPtrSet< llvm::Type *, 8 > seen;
                if (!typeContainsPointer(ET, seen)) {
                    uint64_t size = DL.getTypeAllocSize(T);
                    if (size == 0)
                        return;
                    auto make_sym = getKleeMakeSymbolic(M);
                    llvm::Value *size_val = llvm::ConstantInt::get(sizeTy, size);
                    llvm::Value *name_str = B.CreateGlobalString(name.str());
                    B.CreateCall(make_sym, { addr, size_val, name_str });
                    return;
                }
            }

            if (N <= klee_init_array_expand_limit) {
                for (uint64_t j = 0; j < N; ++j) {
                    llvm::Value *idxs[] = {
                        llvm::ConstantInt::get(
                            llvm::Type::getInt64Ty(Ctx), 0),
                        llvm::ConstantInt::get(
                            llvm::Type::getInt64Ty(Ctx), j),
                    };
                    llvm::Value *elt_ptr = B.CreateInBoundsGEP(AT, addr, idxs);
                    buildTypeInitBody(
                        B, elt_ptr, depth, ET, im, cache,
                        name + "[" + llvm::Twine(j) + "]", M
                    );
                }
                return;
            }

            // Large array fallback — flat symbolic bytes. Any pointer
            // fields inside the element type become unconstrained bytes
            // for elements in this array; documented limitation.
            if (verbose) {
                llvm::outs() << "  Large array fallback: " << name.str()
                             << " (" << N << " elems of aggregate type)\n";
            }
            uint64_t size = DL.getTypeAllocSize(T);
            auto make_sym = getKleeMakeSymbolic(M);
            llvm::Value *size_val = llvm::ConstantInt::get(sizeTy, size);
            llvm::Value *name_str = B.CreateGlobalString(name.str());
            B.CreateCall(make_sym, { addr, size_val, name_str });
            return;
        }

        // Top-level pointer (rare for globals): there is no enclosing
        // StructType to key into the inference map, so we fall back to
        // the flat malloc'd buffer unconditionally. A future patch can
        // extend inference to cover top-level-pointer globals directly.
        if (T->isPointerTy()) {
            auto make_sym = getKleeMakeSymbolic(M);
            auto malloc_fn = getMalloc(M);
            llvm::Value *size_val =
                llvm::ConstantInt::get(sizeTy, symbolic_ptr_size);
            llvm::Value *buf      = B.CreateCall(malloc_fn, { size_val });
            llvm::Value *name_str = B.CreateGlobalString(name.str());
            B.CreateCall(make_sym, { buf, size_val, name_str });
            B.CreateStore(buf, addr);
            return;
        }

        // Any other kind (vector, scalable vector, etc.): treat as a flat
        // leaf, matching the fast path.
        uint64_t size = DL.getTypeAllocSize(T);
        if (size == 0)
            return;
        auto make_sym = getKleeMakeSymbolic(M);
        llvm::Value *size_val = llvm::ConstantInt::get(sizeTy, size);
        llvm::Value *name_str = B.CreateGlobalString(name.str());
        B.CreateCall(make_sym, { addr, size_val, name_str });
    }

    // Entry point for the per-type init machinery. Creates (or returns a
    // cached) internal `void __klee_init_type_<T>(ptr p, i32 depth)`
    // function and, on first creation, populates its body with the
    // structural walk rooted at `T`.
    //
    // The cache insert happens BEFORE body construction so any recursive
    // lookup for the same type during body construction returns the
    // in-progress Function* — closing self-reference and mutual recursion
    // at codegen time.
    static llvm::Function *getOrCreatePerTypeInit(
        llvm::Module &M, llvm::Type *T,
        const PointerFieldInferenceMap &im, TypeInitCache &cache
    ) {
        if (auto it = cache.find(T); it != cache.end())
            return it->second;

        auto &Ctx    = M.getContext();
        auto *voidTy = llvm::Type::getVoidTy(Ctx);
        auto *ptrTy  = llvm::PointerType::getUnqual(Ctx);
        auto *i32Ty  = llvm::Type::getInt32Ty(Ctx);

        auto *FT = llvm::FunctionType::get(voidTy, { ptrTy, i32Ty }, false);
        std::string fn_name = "__klee_init_type_" + mangleTypeName(T);

        auto *F = llvm::Function::Create(
            FT, llvm::GlobalValue::InternalLinkage, fn_name, &M
        );
        F->arg_begin()->setName("p");
        std::next(F->arg_begin())->setName("depth");

        // Cache first — body construction may recursively ask for F.
        cache[T] = F;

        auto *entry_bb = llvm::BasicBlock::Create(Ctx, "entry", F);
        llvm::IRBuilder<> B(entry_bb);

        llvm::Value *arg_p     = F->getArg(0);
        llvm::Value *arg_depth = F->getArg(1);

        buildTypeInitBody(
            B, arg_p, arg_depth, T, im, cache, mangleTypeName(T), M
        );
        B.CreateRetVoid();

        return F;
    }

    // Create (or return a cached) internal `void @__klee_init_g_<name>()`
    // wrapper that calls `@__klee_init_type_<T>(@g, 0)`. Wrappers keep
    // the descriptor table's signature uniform regardless of the global's
    // LLVM type.
    static llvm::Function *getOrCreatePerGlobalInit(
        llvm::Module &M, llvm::GlobalVariable *GV,
        const PointerFieldInferenceMap &im, TypeInitCache &cache
    ) {
        auto &Ctx    = M.getContext();
        auto *voidTy = llvm::Type::getVoidTy(Ctx);
        auto *i32Ty  = llvm::Type::getInt32Ty(Ctx);

        std::string wrapper_name = ("__klee_init_g_" + GV->getName()).str();
        // Sanitize the global's name the same way as types.
        for (char &c : wrapper_name) {
            if (!std::isalnum(static_cast< unsigned char >(c)) && c != '_')
                c = '_';
        }

        if (auto *existing = M.getFunction(wrapper_name))
            return existing;

        auto *FT = llvm::FunctionType::get(voidTy, {}, false);
        auto *F  = llvm::Function::Create(
            FT, llvm::GlobalValue::InternalLinkage, wrapper_name, &M
        );

        auto *entry_bb = llvm::BasicBlock::Create(Ctx, "entry", F);
        llvm::IRBuilder<> B(entry_bb);

        llvm::Function *type_init =
            getOrCreatePerTypeInit(M, GV->getValueType(), im, cache);
        B.CreateCall(type_init, { GV, llvm::ConstantInt::get(i32Ty, 0) });
        B.CreateRetVoid();

        return F;
    }

    // Build the internal-constant descriptor table used by the dispatcher.
    //
    // Layout: `[N x { ptr, ptr }]` where each entry is `{ name_str, init_fn }`.
    // The `name` field is unused in the initial implementation but exists
    // so future runtime hooks (skip lists, logging, constraint injection)
    // can filter by string match without touching the generator. We use
    // an anonymous literal struct type rather than adding a named type to
    // the module, to avoid clashing with user types on repeat runs.
    static std::pair< llvm::GlobalVariable *, uint64_t >
    emitGlobalsDescriptorTable(
        llvm::Module &M,
        const std::vector< llvm::GlobalVariable * > &gvs,
        const std::unordered_map< llvm::GlobalVariable *, llvm::Function * >
            &wrappers
    ) {
        auto &Ctx   = M.getContext();
        auto *ptrTy = llvm::PointerType::getUnqual(Ctx);

        auto *entry_ty = llvm::StructType::get(Ctx, { ptrTy, ptrTy });

        std::vector< llvm::Constant * > entries;
        entries.reserve(gvs.size());

        for (auto *GV : gvs) {
            auto wit = wrappers.find(GV);
            if (wit == wrappers.end())
                continue;

            // Use a dedicated .str.klee_ prefix so collectModuleGlobals
            // re-runs will filter these back out (isToolSynthesizedGlobal).
            std::string name_str = GV->getName().str();
            auto *name_const = llvm::ConstantDataArray::getString(
                Ctx, name_str, /*AddNull=*/true
            );
            auto *name_gv = new llvm::GlobalVariable(
                M, name_const->getType(), /*isConstant=*/true,
                llvm::GlobalValue::PrivateLinkage, name_const,
                ".str.klee_globals_desc_name"
            );
            name_gv->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

            auto *entry = llvm::ConstantStruct::get(
                entry_ty, { name_gv, wit->second }
            );
            entries.push_back(entry);
        }

        if (entries.empty())
            return { nullptr, 0 };

        auto *arr_ty = llvm::ArrayType::get(entry_ty, entries.size());
        auto *arr    = llvm::ConstantArray::get(arr_ty, entries);

        auto *desc = new llvm::GlobalVariable(
            M, arr_ty, /*isConstant=*/true,
            llvm::GlobalValue::InternalLinkage, arr,
            "__klee_globals_desc"
        );
        desc->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

        return { desc, entries.size() };
    }

    // Synthesize the dispatcher: a counted loop over the descriptor table
    // that loads each entry's init_fn pointer and calls it with no args.
    // This is the single designated hook point — future features (skip
    // lists, logging, constraint injection) slot into the loop body.
    static llvm::Function *getOrCreateInitGlobalsDispatcher(
        llvm::Module &M, llvm::GlobalVariable *desc_table, uint64_t count
    ) {
        auto &Ctx    = M.getContext();
        auto *voidTy = llvm::Type::getVoidTy(Ctx);
        auto *ptrTy  = llvm::PointerType::getUnqual(Ctx);
        auto *i64Ty  = llvm::Type::getInt64Ty(Ctx);

        if (auto *existing = M.getFunction("__klee_init_globals"))
            return existing;

        auto *FT = llvm::FunctionType::get(voidTy, {}, false);
        auto *F  = llvm::Function::Create(
            FT, llvm::GlobalValue::InternalLinkage,
            "__klee_init_globals", &M
        );

        auto *entry_bb  = llvm::BasicBlock::Create(Ctx, "entry", F);
        llvm::IRBuilder<> B(entry_bb);

        // Nothing to initialize — single ret.
        if (!desc_table || count == 0) {
            B.CreateRetVoid();
            return F;
        }

        auto *header_bb = llvm::BasicBlock::Create(Ctx, "header", F);
        auto *body_bb   = llvm::BasicBlock::Create(Ctx, "body", F);
        auto *latch_bb  = llvm::BasicBlock::Create(Ctx, "latch", F);
        auto *exit_bb   = llvm::BasicBlock::Create(Ctx, "exit", F);

        B.CreateBr(header_bb);

        B.SetInsertPoint(header_bb);
        auto *phi = B.CreatePHI(i64Ty, 2, "i");
        phi->addIncoming(llvm::ConstantInt::get(i64Ty, 0), entry_bb);
        llvm::Value *done = B.CreateICmpUGE(
            phi, llvm::ConstantInt::get(i64Ty, count)
        );
        B.CreateCondBr(done, exit_bb, body_bb);

        B.SetInsertPoint(body_bb);
        // Entry type matches emitGlobalsDescriptorTable above.
        auto *entry_ty = llvm::StructType::get(Ctx, { ptrTy, ptrTy });
        llvm::Value *ep = B.CreateInBoundsGEP(
            llvm::ArrayType::get(entry_ty, count), desc_table,
            { llvm::ConstantInt::get(i64Ty, 0), phi }
        );
        // Load the init_fn field (index 1).
        llvm::Value *fnp = B.CreateStructGEP(entry_ty, ep, 1);
        llvm::Value *fn  = B.CreateLoad(ptrTy, fnp);
        auto *wrapper_fty = llvm::FunctionType::get(voidTy, {}, false);
        B.CreateCall(wrapper_fty, fn, {});
        B.CreateBr(latch_bb);

        B.SetInsertPoint(latch_bb);
        llvm::Value *next = B.CreateAdd(
            phi, llvm::ConstantInt::get(i64Ty, 1)
        );
        phi->addIncoming(next, latch_bb);
        B.CreateBr(header_bb);

        B.SetInsertPoint(exit_bb);
        B.CreateRetVoid();

        return F;
    }

    // Generate the main() harness function
    static bool generateHarness(llvm::Module &M, llvm::Function *target_fn) {
        auto &Ctx   = M.getContext();
        auto &DL    = M.getDataLayout();
        auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
        auto *ptrTy = llvm::PointerType::getUnqual(Ctx);
        auto *sizeTy = DL.getIntPtrType(Ctx);

        // Remove existing main if present — but not if it's the target function.
        //
        // A decompiled firmware module often contains a `main` that is still
        // called from an (also-decompiled) startup stub or referenced from a
        // constant initializer (e.g. a vtable / interrupt table / `@llvm.*.ctors`
        // entry). `Function::eraseFromParent()` asserts when the value still
        // has users, so we must break those uses explicitly before the erase:
        // rewrite every remaining use to a poison value of the correct function
        // pointer type. The old callers become unreachable dead code which the
        // harness never enters — KLEE only executes the freshly-synthesized
        // `main` we create below.
        if (auto *old_main = M.getFunction("main")) {
            if (old_main == target_fn) {
                // Rename the target out of the way so we can create a new main().
                // Renames don't touch uses, so no RAUW is needed here.
                old_main->setName("__klee_orig_main");
            } else {
                if (!old_main->use_empty()) {
                    LOG(WARNING)
                        << "existing main() has " << old_main->getNumUses()
                        << " remaining use(s); replacing with poison before "
                           "erase (dead startup-path callers become "
                           "unreachable)\n";
                    old_main->replaceAllUsesWith(
                        llvm::PoisonValue::get(old_main->getType())
                    );
                }
                old_main->eraseFromParent();
            }
        }

        // Create main() -> i32
        auto *main_ft = llvm::FunctionType::get(i32Ty, false);
        auto *main_fn = llvm::Function::Create(
            main_ft, llvm::GlobalValue::ExternalLinkage, "main", M
        );

        auto *entry_bb = llvm::BasicBlock::Create(Ctx, "entry", main_fn);
        llvm::IRBuilder<> B(entry_bb);

        auto make_sym = getKleeMakeSymbolic(M);

        // 1. Collect static contract predicates
        auto contracts = collectStaticContracts(M, target_fn);
        std::vector< ParsedPredicate > all_preds;
        unsigned dropped_preds = 0;
        for (auto &c : contracts) {
            auto preds = parseStaticContractText(c, dropped_preds);
            all_preds.insert(all_preds.end(), preds.begin(), preds.end());
        }

        if (dropped_preds > 0) {
            if (strict_contracts) {
                LOG(ERROR)
                    << dropped_preds
                    << " predicate(s) failed to parse for target '"
                    << target_fn->getName()
                    << "' — refusing to emit an under-constrained harness "
                       "(pass --strict-contracts=false to override)\n";
                return false;
            }
            LOG(WARNING) << dropped_preds
                         << " predicate(s) dropped during parsing for target '"
                         << target_fn->getName()
                         << "' — harness may be under-constrained\n";
        }

        if (verbose) {
            llvm::outs() << "Found " << contracts.size() << " static contract(s) with "
                         << all_preds.size() << " predicate(s)\n";
        }

        // 2. Symbolically initialize module-wide globals via the per-type
        //    init machinery. Flow:
        //
        //      a. One linear inference pass over the module recovers
        //         pointee types for every struct field of type `ptr`.
        //      b. Collect the set of module-wide symbolizable globals.
        //      c. For each global, build (or look up) its per-global
        //         wrapper, which transitively builds per-type init
        //         functions with codegen-time cycle closure.
        //      d. Build the descriptor table and the dispatcher.
        //      e. Emit a single call to the dispatcher at the top of
        //         main(), before argument symbolization so the target
        //         sees initialized globals when it runs.
        //
        //    The old per-global inline `klee_make_symbolic` emission at
        //    this site has been deleted — everything now lives in
        //    internally-linked helper functions that can be augmented
        //    with runtime hooks (skip lists, logging, constraints)
        //    without touching the harness generator.
        auto inference_map = inferPointerFieldTypes(M);

        // Pre-pass: promote `external global` declarations referenced by
        // the code under analysis into internally-linked definitions with
        // zero initializers. Without this, decompiled-firmware globals
        // that the JSON exporter left as externs (common when the source
        // binary had no initializer in its own translation unit) are
        // skipped by collectModuleGlobals and the target function then
        // references an unlinked symbol at KLEE time — the exact failure
        // shape seen on `@usb_g` in bl_usb__send_message.
        unsigned materialized = materializeExternalGlobals(M);

        std::vector< llvm::GlobalVariable * > module_globals;
        collectModuleGlobals(M, module_globals);

        if (verbose) {
            logCollectedGlobals(module_globals, M);
        }

        TypeInitCache type_init_cache;
        std::unordered_map< llvm::GlobalVariable *, llvm::Function * > wrappers;
        for (auto *GV : module_globals) {
            llvm::Function *wrapper = getOrCreatePerGlobalInit(
                M, GV, inference_map, type_init_cache
            );
            wrappers[GV] = wrapper;
        }

        auto [desc_table, desc_count] =
            emitGlobalsDescriptorTable(M, module_globals, wrappers);
        llvm::Function *dispatcher =
            getOrCreateInitGlobalsDispatcher(M, desc_table, desc_count);

        B.CreateCall(dispatcher, {});

        if (verbose) {
            llvm::outs() << "Globals init: " << module_globals.size()
                         << " global(s) ("
                         << materialized << " materialized from external), "
                         << type_init_cache.size()
                         << " per-type init function(s), "
                         << inference_map.size()
                         << " pointer field(s) inferred\n";
        }

        // 3. Create symbolic arguments for the target function
        llvm::FunctionType *target_ft = target_fn->getFunctionType();
        std::vector< llvm::Value * > args;
        // Map from arg index to the loaded argument value (for predicates)
        std::map< unsigned, llvm::Value * > arg_values;

        for (unsigned i = 0; i < target_ft->getNumParams(); ++i) {
            llvm::Type *param_ty = target_ft->getParamType(i);
            uint64_t param_size  = DL.getTypeAllocSize(param_ty);

            std::string arg_name = "arg" + std::to_string(i);

            if (param_ty->isPointerTy()) {
                // For pointer args: allocate a buffer, make it symbolic, pass pointer.
                // Try to infer the pointee size from parameter attributes (byval, sret,
                // dereferenceable); fall back to --symbolic-ptr-size.
                uint64_t bufSize = symbolic_ptr_size;

                if (auto byval_ty = target_fn->getParamByValType(i)) {
                    uint64_t sz = DL.getTypeAllocSize(byval_ty);
                    if (sz > 0)
                        bufSize = sz;
                } else if (auto sret_ty = target_fn->getParamStructRetType(i)) {
                    uint64_t sz = DL.getTypeAllocSize(sret_ty);
                    if (sz > 0)
                        bufSize = sz;
                } else if (uint64_t deref = target_fn->getParamDereferenceableBytes(i)) {
                    if (deref > 0)
                        bufSize = deref;
                }

                auto *bufTy = llvm::ArrayType::get(llvm::Type::getInt8Ty(Ctx), bufSize);
                auto *Buf   = B.CreateAlloca(bufTy);

                llvm::Value *cast_ptr = B.CreateBitCast(Buf, ptrTy);
                llvm::Value *size_val = llvm::ConstantInt::get(sizeTy, bufSize);
                llvm::Value *name_str = B.CreateGlobalString(arg_name);
                B.CreateCall(make_sym, { cast_ptr, size_val, name_str });

                llvm::Value *arg_ptr = B.CreateBitCast(Buf, param_ty);
                args.push_back(arg_ptr);
                arg_values[i] = arg_ptr;
            } else {
                // For non-pointer args: alloca, make symbolic, load
                auto *Alloca = B.CreateAlloca(param_ty);

                llvm::Value *cast_ptr = B.CreateBitCast(Alloca, ptrTy);
                llvm::Value *size_val = llvm::ConstantInt::get(sizeTy, param_size);
                llvm::Value *name_str = B.CreateGlobalString(arg_name);
                B.CreateCall(make_sym, { cast_ptr, size_val, name_str });

                llvm::Value *Val = B.CreateLoad(param_ty, Alloca);
                args.push_back(Val);
                arg_values[i] = Val;
            }

            if (verbose) {
                llvm::outs() << "  Symbolic arg" << i << ": ";
                param_ty->print(llvm::outs());
                llvm::outs() << "\n";
            }
        }

        // 4. Emit precondition assumes
        for (auto &P : all_preds) {
            if (!P.is_precondition)
                continue;

            llvm::Value *arg_val = nullptr;
            if (P.target.substr(0, 3) == "Arg") {
                auto it = arg_values.find(P.arg_index);
                if (it != arg_values.end())
                    arg_val = it->second;
            }

            emitKleePredicate(B, M, P, arg_val, nullptr);
        }

        // 5. Call the target function
        llvm::Value *ret_val = nullptr;
        if (target_ft->getReturnType()->isVoidTy()) {
            B.CreateCall(target_fn, args);
        } else {
            ret_val = B.CreateCall(target_fn, args);
        }

        // 6. Emit postcondition asserts
        for (auto &P : all_preds) {
            if (P.is_precondition)
                continue;

            llvm::Value *arg_val = nullptr;
            if (P.target.substr(0, 3) == "Arg") {
                auto it = arg_values.find(P.arg_index);
                if (it != arg_values.end())
                    arg_val = it->second;
            }

            emitKleePredicate(B, M, P, arg_val, ret_val);
        }

        // 7. Return 0
        B.CreateRet(llvm::ConstantInt::get(i32Ty, 0));

        if (verbose) {
            llvm::outs() << "Generated main() harness for " << target_fn->getName() << "\n";
        }

        return true;
    }

    // Write module to output file.
    //
    // Error handling contract: we must catch stream errors at every stage
    // of the write pipeline. `Module::print` and `WriteBitcodeToFile` swallow
    // write failures into the stream's sticky error flag, so we have to
    // probe that flag ourselves. For on-disk output, failures can surface
    // *after* the last bitcode byte is handed to the kernel — flush errors,
    // and in particular close() errors from delayed allocation on NFS/SMB
    // and ENOSPC/EDQUOT flushed on close — so we explicitly `close()` the
    // fd stream and re-check `has_error()` before letting its destructor
    // run. `raw_fd_ostream`'s destructor calls `report_fatal_error` (which
    // aborts) if the error flag is still set at teardown, so every failure
    // path clears the flag before returning.
    static bool writeModuleToFile(llvm::Module &module, llvm::StringRef out) {
        auto emitTo = [&](llvm::raw_ostream &os) {
            if (emit_ll) {
                module.print(os, nullptr);
            } else {
                llvm::WriteBitcodeToFile(module, os);
            }
        };

        if (out == "-") {
            emitTo(llvm::outs());
            llvm::outs().flush();
            if (llvm::outs().has_error()) {
                LOG(ERROR) << "writing module to stdout: stream error\n";
                llvm::outs().clear_error();
                return false;
            }
            return true;
        }

        std::error_code ec;
        llvm::raw_fd_ostream os(out, ec, llvm::sys::fs::OF_None);
        if (ec) {
            LOG(ERROR) << "opening " << out << ": " << ec.message() << "\n";
            return false;
        }

        emitTo(os);

        // Surface write/flush errors before close so the diagnostic can
        // attribute them to the write phase rather than the close phase.
        os.flush();
        if (os.has_error()) {
            LOG(ERROR) << "writing module to " << out << ": stream error\n";
            os.clear_error();
            return false;
        }

        // Explicitly close to capture late errors (delayed allocation on
        // network filesystems, disk-quota/ENOSPC flushed on close). If we
        // let the destructor do this the error surfaces via
        // report_fatal_error and aborts the process.
        os.close();
        if (os.has_error()) {
            LOG(ERROR) << "closing " << out << ": stream error\n";
            os.clear_error();
            return false;
        }

        return true;
    }

    // Recursively test whether a type (transitively) contains a pointer.
    // Used by the retargeting check to count named struct types whose
    // layout changes when pointer width changes (32-bit -> 64-bit).
    static bool typeContainsPointer(
        llvm::Type *T, llvm::SmallPtrSetImpl< llvm::Type * > &seen
    ) {
        if (!T || !seen.insert(T).second)
            return false;
        if (T->isPointerTy())
            return true;
        if (auto *ST = llvm::dyn_cast< llvm::StructType >(T)) {
            if (ST->isOpaque())
                return false;
            for (llvm::Type *elt : ST->elements()) {
                if (typeContainsPointer(elt, seen))
                    return true;
            }
            return false;
        }
        if (auto *AT = llvm::dyn_cast< llvm::ArrayType >(T))
            return typeContainsPointer(AT->getElementType(), seen);
        if (auto *VT = llvm::dyn_cast< llvm::VectorType >(T))
            return typeContainsPointer(VT->getElementType(), seen);
        return false;
    }

    // Retarget `module` to x86_64 in-place for KLEE compatibility.
    //
    // KLEE's memory allocator uses host (64-bit) addresses, so 32-bit
    // target modules (e.g. ARM32 from Ghidra decompilation) cannot run
    // directly. Since KLEE interprets IR semantically and the harness
    // only exercises value-level contract predicates (nonnull, range,
    // relation), retargeting is safe for *those* predicates.
    //
    // **Layout reshaping caveat.** When the original pointer width differs
    // from 64, every struct that (transitively) contains a pointer has
    // its size, field offsets, and alignment silently recomputed under
    // the new datalayout. Typed GEPs stay correct because LLVM recomputes
    // them from the (new) struct layout, but any of the following patterns
    // will be miscompiled after retargeting:
    //
    //   - raw memcpy/memset/memmove sized against the original struct
    //     (e.g. `memcpy(dst, src, 24)` where the original 32-bit layout
    //     was 24 bytes but the 64-bit layout is 32);
    //   - constant byte-array globals whose bytes encode a pointer-
    //     containing struct instance (initializer bytes no longer align
    //     to the new field offsets);
    //   - inttoptr / ptrtoint round-trips through a 32-bit integer
    //     (upper 32 bits of the 64-bit pointer are lost).
    //
    // Detect the case and emit a prominent warning quantifying how many
    // named struct types are affected. See
    // docs/klee-integration.md#architecture-retargeting for the full
    // limitation write-up.
    static void retargetModuleToX86_64(llvm::Module &M, bool verbose) {
        static constexpr const char *kX86_64_Triple = "x86_64-unknown-linux-gnu";
        static constexpr const char *kX86_64_DataLayout =
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128"
            "-f80:128-n8:16:32:64-S128";

        std::string currentTriple = M.getTargetTriple();
        llvm::Triple parsed(currentTriple);
        bool is_x86_64 = parsed.getArch() == llvm::Triple::x86_64;
        if (!currentTriple.empty() && is_x86_64)
            return;

        // Snapshot the original pointer width (address space 0) before
        // we stamp in the x86_64 datalayout. An empty datalayout yields 0,
        // which we treat as "unknown, do not warn".
        unsigned old_ptr_bits = 0;
        if (!M.getDataLayoutStr().empty()) {
            llvm::DataLayout old_dl(M.getDataLayoutStr());
            old_ptr_bits = old_dl.getPointerSizeInBits(/*AS=*/0);
        }

        if (verbose) {
            llvm::outs() << "Retargeting module from '"
                         << (currentTriple.empty() ? "<none>" : currentTriple)
                         << "' to x86_64 for KLEE compatibility\n";
        }

        M.setTargetTriple(kX86_64_Triple);
        M.setDataLayout(kX86_64_DataLayout);

        if (old_ptr_bits == 0 || old_ptr_bits == 64)
            return;

        unsigned reshaped_structs = 0;
        for (llvm::StructType *ST : M.getIdentifiedStructTypes()) {
            if (ST->isOpaque())
                continue;
            llvm::SmallPtrSet< llvm::Type *, 8 > seen;
            if (typeContainsPointer(ST, seen))
                ++reshaped_structs;
        }

        LOG(WARNING)
            << "retargeted module pointer width " << old_ptr_bits
            << " -> 64 bits: " << reshaped_structs
            << " named struct type(s) transitively contain pointer "
               "fields and were silently reshaped (sizes, offsets, and "
               "alignment recomputed under the x86_64 datalayout). "
               "Value-level contract predicates (nonnull/range/relation) "
               "remain valid, but raw memcpy/memset sized against the "
               "original layout, byte-initialized globals encoding "
               "pointer-bearing structs, and inttoptr round-trips "
               "through 32-bit integers will be miscompiled. See "
               "docs/klee-integration.md#architecture-retargeting.\n";
    }
} // namespace

int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);
    llvm::cl::ParseCommandLineOptions(argc, argv, "patchir-klee-verifier\n");

    llvm::LLVMContext context;
    llvm::SMDiagnostic err;

    std::unique_ptr< llvm::Module > module = llvm::parseIRFile(input_filename, err, context);
    if (!module) {
        err.print(argv[0], llvm::errs());
        return EXIT_FAILURE;
    }

    // Retarget module to x86_64 for KLEE compatibility. See the comment on
    // retargetModuleToX86_64 for the layout-reshaping caveat — this call
    // will emit a prominent warning if the original pointer width differed
    // from 64 and any struct types had to be silently reshaped.
    retargetModuleToX86_64(*module, verbose);

    if (verbose) {
        llvm::outs() << "Loaded module: " << module->getName() << "\n";
        llvm::outs() << "Target function: " << target_function << "\n";
    }

    // Find the target function
    llvm::Function *target_fn = module->getFunction(target_function);
    if (!target_fn) {
        LOG(ERROR) << "target function '" << target_function << "' not found in module\n";
        return EXIT_FAILURE;
    }

    if (target_fn->isDeclaration()) {
        LOG(ERROR) << "target function '" << target_function
                   << "' is a declaration (no body)\n";
        return EXIT_FAILURE;
    }

    // 1. Link model library if provided (replaces declarations with definitions)
    if (!model_library.empty()) {
        auto model_mod = llvm::parseIRFile(model_library, err, context);
        if (!model_mod) {
            LOG(ERROR) << "failed to load model library '" << model_library << "'\n";
            err.print(argv[0], llvm::errs());
            return EXIT_FAILURE;
        }

        // The model must have been compiled against the same (retargeted)
        // datalayout as the input module — mismatched pointer widths or
        // struct alignments would produce miscompiled code after linking.
        // Accept an empty triple/DL (older bitcode) and stamp it with the
        // module's values; otherwise require an exact match and reject
        // anything else loudly rather than silently rewriting it.
        const std::string &model_triple = model_mod->getTargetTriple();
        const std::string &model_dl     = model_mod->getDataLayoutStr();
        const std::string &host_triple  = module->getTargetTriple();
        const std::string &host_dl      = module->getDataLayoutStr();

        if (model_triple.empty() && model_dl.empty()) {
            model_mod->setTargetTriple(host_triple);
            model_mod->setDataLayout(module->getDataLayout());
        } else if (model_triple != host_triple || model_dl != host_dl) {
            LOG(ERROR) << "model library '" << model_library
                       << "' targets '" << model_triple << "' / '" << model_dl
                       << "', expected '" << host_triple << "' / '"
                       << host_dl << "'; rebuild the model for the host layout\n";
            return EXIT_FAILURE;
        }

        if (llvm::Linker::linkModules(*module, std::move(model_mod),
                                      llvm::Linker::Flags::LinkOnlyNeeded)) {
            LOG(ERROR) << "failed to link model library '" << model_library << "'\n";
            return EXIT_FAILURE;
        }

        if (verbose) {
            llvm::outs() << "Linked model library: " << model_library << "\n";
        }
    }

    // 2. Stub external functions (must do before harness so stubs exist)
    unsigned stub_count = stubExternalFunctions(*module, target_fn);
    if (verbose) {
        llvm::outs() << "Stubbed " << stub_count << " external function(s)\n";
    }

    // 3. Generate main() harness
    if (!generateHarness(*module, target_fn)) {
        LOG(ERROR) << "failed to generate harness\n";
        return EXIT_FAILURE;
    }

    // 4. Write output
    if (!writeModuleToFile(*module, output_filename)) {
        LOG(ERROR) << "failed to write output\n";
        return EXIT_FAILURE;
    }

    if (verbose) {
        llvm::outs() << "Successfully wrote output to: " << output_filename << "\n";
    }

    return EXIT_SUCCESS;
}
