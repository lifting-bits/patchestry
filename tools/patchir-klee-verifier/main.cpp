/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <llvm/Bitcode/BitcodeWriter.h>
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

#include <patchestry/Util/Log.hpp>

#include <map>
#include <set>
#include <string>
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

    const llvm::cl::opt< bool >
        verbose("v", llvm::cl::desc("Enable verbose output"), llvm::cl::init(false));

    const llvm::cl::opt< unsigned > symbolic_ptr_size(
        "symbolic-ptr-size",
        llvm::cl::desc("Default symbolic buffer size (bytes) for pointer arguments"),
        llvm::cl::value_desc("bytes"), llvm::cl::init(256)
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

    static bool parseTarget(const std::string &target_str, std::string &target, unsigned &index) {
        target = target_str;
        if (target_str.substr(0, 4) == "Arg(") {
            auto end_pos = target_str.find(')');
            if (end_pos != std::string::npos) {
                std::string index_str = target_str.substr(4, end_pos - 4);
                try {
                    index = static_cast< unsigned >(std::stoul(index_str));
                    return true;
                } catch (const std::exception &e) {
                    LOG(WARNING) << "failed to parse argument index '"
                                 << index_str << "': " << e.what() << "\n";
                    return false;
                }
            }
        }
        return target_str == "ReturnValue";
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
            parseTarget(target_it->second, pred.target, pred.arg_index);
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
            if (pred.target == "ReturnValue") {
                pred.kind = PK_RangeRet;
            } else if (pred.target.substr(0, 3) == "Arg") {
                pred.kind = PK_RangeArg;
            }

            auto range_it = kv.find("range");
            if (range_it != kv.end()) {
                std::string range_str = range_it->second;
                size_t min_pos = range_str.find("min=");
                size_t max_pos = range_str.find("max=");

                if (min_pos != std::string::npos) {
                    min_pos += 4;
                    size_t min_end = range_str.find_first_of(",]", min_pos);
                    try {
                        pred.min_val = std::stoll(range_str.substr(min_pos, min_end - min_pos));
                    } catch (const std::exception &e) {
                        LOG(WARNING) << "failed to parse range min: " << e.what() << "\n";
                    }
                }

                if (max_pos != std::string::npos) {
                    max_pos += 4;
                    size_t max_end = range_str.find_first_of(",]", max_pos);
                    try {
                        pred.max_val = std::stoll(range_str.substr(max_pos, max_end - max_pos));
                    } catch (const std::exception &e) {
                        LOG(WARNING) << "failed to parse range max: " << e.what() << "\n";
                    }
                }
            }
        } else if (kind_str == "alignment") {
            pred.kind = PK_Alignment;
            auto align_it = kv.find("align");
            if (align_it != kv.end()) {
                try {
                    pred.alignment = std::stoull(align_it->second);
                } catch (const std::exception &e) {
                    LOG(WARNING) << "failed to parse alignment: " << e.what() << "\n";
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

    static std::vector< ParsedPredicate >
    parseStaticContractText(const std::string &contract_str) {
        std::vector< ParsedPredicate > preds;

        size_t pre_start = contract_str.find("preconditions=[");
        if (pre_start != std::string::npos) {
            pre_start += 15;
            size_t pre_end = findMatchingBracket(contract_str, pre_start);
            if (pre_end != std::string::npos) {
                std::string pre_section = contract_str.substr(pre_start, pre_end - pre_start);
                size_t pos = 0;
                while (pos < pre_section.length()) {
                    size_t start = pre_section.find('{', pos);
                    if (start == std::string::npos)
                        break;
                    start++;
                    size_t end = pre_section.find('}', start);
                    if (end == std::string::npos)
                        break;
                    std::string pred_str = pre_section.substr(start, end - start);
                    auto kv              = parseKeyValues(pred_str);
                    auto pred            = kvToPredicate(kv);
                    if (pred.kind != PK_Unknown) {
                        pred.is_precondition = true;
                        preds.push_back(pred);
                    }
                    pos = end + 1;
                }
            }
        }

        size_t post_start = contract_str.find("postconditions=[");
        if (post_start != std::string::npos) {
            post_start += 16;
            size_t post_end = findMatchingBracket(contract_str, post_start);
            if (post_end != std::string::npos) {
                std::string post_section = contract_str.substr(post_start, post_end - post_start);
                size_t pos = 0;
                while (pos < post_section.length()) {
                    size_t start = post_section.find('{', pos);
                    if (start == std::string::npos)
                        break;
                    start++;
                    size_t end = post_section.find('}', start);
                    if (end == std::string::npos)
                        break;
                    std::string pred_str = post_section.substr(start, end - start);
                    auto kv              = parseKeyValues(pred_str);
                    auto pred            = kvToPredicate(kv);
                    if (pred.kind != PK_Unknown) {
                        pred.is_precondition = false;
                        preds.push_back(pred);
                    }
                    pos = end + 1;
                }
            }
        }

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

    // Get or declare klee_assert: void klee_assert(int)
    static llvm::FunctionCallee getKleeAssert(llvm::Module &M) {
        auto &Ctx   = M.getContext();
        auto *voidTy = llvm::Type::getVoidTy(Ctx);
        auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
        auto *FT    = llvm::FunctionType::get(voidTy, { i32Ty }, false);
        return M.getOrInsertFunction("klee_assert", FT);
    }

    // Collect all GlobalVariable references from a function (transitively through internal calls)
    static void collectGlobals(
        llvm::Function *F,
        std::set< llvm::GlobalVariable * > &out_globals,
        std::set< llvm::Function * > &visited_fns
    ) {
        if (!F || !visited_fns.insert(F).second)
            return;

        for (auto &BB : *F) {
            for (auto &I : BB) {
                for (unsigned i = 0; i < I.getNumOperands(); ++i) {
                    if (auto *GV = llvm::dyn_cast< llvm::GlobalVariable >(I.getOperand(i))) {
                        if (!GV->isConstant()) {
                            out_globals.insert(GV);
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

    // Collect static_contract metadata from all call instructions to the target function
    static std::vector< std::string >
    collectStaticContracts(llvm::Module &M, llvm::Function *target_fn) {
        std::vector< std::string > contracts;

        for (auto &F : M) {
            for (auto &BB : F) {
                for (auto &I : BB) {
                    auto *contract_md = I.getMetadata("static_contract");
                    if (!contract_md)
                        continue;

                    auto *tuple = llvm::dyn_cast< llvm::MDTuple >(contract_md);
                    if (!tuple || tuple->getNumOperands() < 2)
                        continue;

                    auto *md_str = llvm::dyn_cast< llvm::MDString >(tuple->getOperand(1));
                    if (!md_str)
                        continue;

                    // Check if this contract is for the target function
                    if (auto *CB = llvm::dyn_cast< llvm::CallBase >(&I)) {
                        if (CB->getCalledFunction() == target_fn) {
                            contracts.push_back(md_str->getString().str());
                        }
                    }
                }
            }
        }

        // Also check if the target function itself has contract metadata
        // (on non-call instructions or calls to other functions)
        if (target_fn && !target_fn->isDeclaration()) {
            for (auto &BB : *target_fn) {
                for (auto &I : BB) {
                    // Skip recursive self-calls — already collected by the first loop
                    if (auto *CB = llvm::dyn_cast< llvm::CallBase >(&I)) {
                        if (CB->getCalledFunction() == target_fn)
                            continue;
                    }

                    auto *contract_md = I.getMetadata("static_contract");
                    if (!contract_md)
                        continue;
                    auto *tuple = llvm::dyn_cast< llvm::MDTuple >(contract_md);
                    if (!tuple || tuple->getNumOperands() < 2)
                        continue;
                    auto *md_str = llvm::dyn_cast< llvm::MDString >(tuple->getOperand(1));
                    if (md_str)
                        contracts.push_back(md_str->getString().str());
                }
            }
        }

        return contracts;
    }

    // Helper to extend/truncate integer value to i64
    static llvm::Value *toI64(llvm::IRBuilder<> &B, llvm::Value *V) {
        if (!V || !V->getType()->isIntegerTy())
            return nullptr;

        auto *i64 = llvm::Type::getInt64Ty(B.getContext());
        if (V->getType() == i64)
            return V;

        unsigned width = V->getType()->getIntegerBitWidth();
        if (width < 64)
            return B.CreateSExt(V, i64);
        else if (width > 64)
            return B.CreateTrunc(V, i64);

        return V;
    }

    // Build a KLEE condition from a predicate and emit klee_assume or klee_assert
    static void emitKleePredicate(
        llvm::IRBuilder<> &B, llvm::Module &M,
        const ParsedPredicate &P, llvm::Value *arg_val, llvm::Value *ret_val
    ) {
        auto &Ctx    = M.getContext();
        auto *i32Ty  = llvm::Type::getInt32Ty(Ctx);
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
            if (!V->getType()->isIntegerTy())
                break;
            cond = B.CreateICmpNE(V, llvm::ConstantInt::getSigned(V->getType(), P.constant));
            break;
        }
        case PK_RelEqArgConst: {
            if (!V->getType()->isIntegerTy())
                break;
            cond = B.CreateICmpEQ(V, llvm::ConstantInt::getSigned(V->getType(), P.constant));
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
            llvm::Value *v64 = toI64(B, V);
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
            // klee_assert takes int (i32)
            llvm::Value *cond_i32 = B.CreateZExt(cond, i32Ty);
            B.CreateCall(getKleeAssert(M), { cond_i32 });
        }
    }

    // Stub undefined external functions with symbolic return values
    static unsigned stubExternalFunctions(llvm::Module &M, llvm::Function *target_fn) {
        unsigned count = 0;
        auto make_sym   = getKleeMakeSymbolic(M);

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

    // Generate the main() harness function
    static bool generateHarness(llvm::Module &M, llvm::Function *target_fn) {
        auto &Ctx   = M.getContext();
        auto &DL    = M.getDataLayout();
        auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
        auto *ptrTy = llvm::PointerType::getUnqual(Ctx);
        auto *sizeTy = DL.getIntPtrType(Ctx);

        // Remove existing main if present — but not if it's the target function
        if (auto *old_main = M.getFunction("main")) {
            if (old_main == target_fn) {
                // Rename the target out of the way so we can create a new main()
                old_main->setName("__klee_orig_main");
            } else {
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
        for (auto &c : contracts) {
            auto preds = parseStaticContractText(c);
            all_preds.insert(all_preds.end(), preds.begin(), preds.end());
        }

        if (verbose) {
            llvm::outs() << "Found " << contracts.size() << " static contract(s) with "
                         << all_preds.size() << " predicate(s)\n";
        }

        // 2. Make referenced globals symbolic
        std::set< llvm::GlobalVariable * > globals;
        std::set< llvm::Function * > visited;
        collectGlobals(target_fn, globals, visited);

        for (auto *GV : globals) {
            uint64_t size = DL.getTypeAllocSize(GV->getValueType());
            if (size == 0)
                continue;

            std::string sym_name = GV->getName().str();
            llvm::Value *cast_ptr = B.CreateBitCast(GV, ptrTy);
            llvm::Value *size_val = llvm::ConstantInt::get(sizeTy, size);
            llvm::Value *name_str = B.CreateGlobalString(sym_name);
            B.CreateCall(make_sym, { cast_ptr, size_val, name_str });

            if (verbose) {
                llvm::outs() << "  Global symbolic: " << GV->getName()
                             << " (" << size << " bytes)\n";
            }
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

    // Write module to output file
    static bool writeModuleToFile(llvm::Module &module, llvm::StringRef out) {
        if (out == "-") {
            if (emit_ll) {
                module.print(llvm::outs(), nullptr);
            } else {
                llvm::WriteBitcodeToFile(module, llvm::outs());
            }
            return true;
        }

        std::error_code ec;
        llvm::raw_fd_ostream os(out, ec, llvm::sys::fs::OF_None);
        if (ec) {
            LOG(ERROR) << "opening " << out << ": " << ec.message() << "\n";
            return false;
        }

        if (emit_ll) {
            module.print(os, nullptr);
        } else {
            llvm::WriteBitcodeToFile(module, os);
        }

        return true;
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

    // Retarget module to x86_64 for KLEE compatibility.
    // KLEE's memory allocator uses host (64-bit) addresses, so 32-bit target
    // modules (e.g. ARM32 from Ghidra decompilation) cannot run directly.
    // Since KLEE interprets IR semantically and the harness only exercises
    // contract predicates (nonnull, range, relation), retargeting is safe.
    {
        static constexpr const char *kX86_64_Triple = "x86_64-unknown-linux-gnu";
        static constexpr const char *kX86_64_DataLayout =
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128"
            "-f80:128-n8:16:32:64-S128";
        std::string currentTriple = module->getTargetTriple();
        if (currentTriple.empty() || currentTriple.find("x86_64") == std::string::npos) {
            if (verbose) {
                llvm::outs() << "Retargeting module from '"
                             << (currentTriple.empty() ? "<none>" : currentTriple)
                             << "' to x86_64 for KLEE compatibility\n";
            }
            module->setTargetTriple(kX86_64_Triple);
            module->setDataLayout(kX86_64_DataLayout);
        }
    }

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

    // 1. Stub external functions (must do before harness so stubs exist)
    unsigned stub_count = stubExternalFunctions(*module, target_fn);
    if (verbose) {
        llvm::outs() << "Stubbed " << stub_count << " external function(s)\n";
    }

    // 2. Generate main() harness
    if (!generateHarness(*module, target_fn)) {
        LOG(ERROR) << "failed to generate harness\n";
        return EXIT_FAILURE;
    }

    // 3. Write output
    if (!writeModuleToFile(*module, output_filename)) {
        LOG(ERROR) << "failed to write output\n";
        return EXIT_FAILURE;
    }

    if (verbose) {
        llvm::outs() << "Successfully wrote output to: " << output_filename << "\n";
    }

    return EXIT_SUCCESS;
}
