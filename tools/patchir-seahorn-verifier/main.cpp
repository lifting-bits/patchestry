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

#include <map>
#include <string>
#include <vector>

namespace {
    // Command line option for input file. Defaults to "-" (stdin)
    const llvm::cl::opt< std::string > input_filename(
        llvm::cl::Positional, llvm::cl::desc("<input LLVM IR file>"), llvm::cl::init("-")
    );

    // Command line option for output file. Defaults to "-" (stdout)
    const llvm::cl::opt< std::string > output_filename(
        "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
        llvm::cl::init("-")
    );

    // Command line flag to emit LLVM IR instead of bitcode
    const llvm::cl::opt< bool >
        emit_ll("S", llvm::cl::desc("Emit LLVM IR Representation"), llvm::cl::init(false));

    // Command line flag to enable verbose output
    const llvm::cl::opt< bool >
        verbose("v", llvm::cl::desc("Enable verbose output"), llvm::cl::init(false));
} // namespace

namespace {
    // Predicate kinds
    enum PredicateKind {
        PK_Unknown,
        PK_Nonnull,         // target != null
        PK_RelNeqArgConst,  // arg[i] != constant
        PK_RelEqArgConst,   // arg[i] == constant
        PK_RelLtArgConst,   // arg[i] < constant
        PK_RelLeArgConst,   // arg[i] <= constant
        PK_RelGtArgConst,   // arg[i] > constant
        PK_RelGeArgConst,   // arg[i] >= constant
        PK_RangeRet,        // min <= return_value <= max
        PK_RangeArg,        // min <= arg[i] <= max
        PK_Alignment        // (ptrtoint(ptr) % align) == 0
    };

    // Structure to hold parsed predicate information
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

    // Parse target string like "Arg(0)", "ReturnValue"
    static bool parseTarget(const std::string &target_str, std::string &target, unsigned &index) {
        target = target_str;

        // Parse Arg(N)
        if (target_str.substr(0, 4) == "Arg(") {
            auto end_pos = target_str.find(')');
            if (end_pos != std::string::npos) {
                std::string index_str = target_str.substr(4, end_pos - 4);
                try {
                    index = static_cast< unsigned >(std::stoul(index_str));
                    return true;
                } catch (const std::exception &e) {
                    llvm::errs() << "Warning: failed to parse argument index '"
                                 << index_str << "': " << e.what() << "\n";
                    return false;
                }
            }
        }
        return target_str == "ReturnValue";
    }

    // Parse key-value pairs from a predicate string fragment
    static std::map< std::string, std::string >
    parseKeyValues(const std::string &pred_str) {
        std::map< std::string, std::string > kv;
        size_t pos = 0;

        while (pos < pred_str.length()) {
            // Find next key
            size_t eq_pos = pred_str.find('=', pos);
            if (eq_pos == std::string::npos)
                break;

            std::string key = pred_str.substr(pos, eq_pos - pos);
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);

            // Find value (could be quoted or not)
            pos            = eq_pos + 1;
            std::string value;

            if (pos < pred_str.length() && pred_str[pos] == '"') {
                // Quoted value
                pos++;
                size_t end_quote = pred_str.find('"', pos);
                if (end_quote != std::string::npos) {
                    value = pred_str.substr(pos, end_quote - pos);
                    pos   = end_quote + 1;
                }
            } else {
                // Unquoted value - find next comma, semicolon, bracket, or end
                size_t next_delim = pred_str.find_first_of(",;]}", pos);
                if (next_delim != std::string::npos) {
                    value = pred_str.substr(pos, next_delim - pos);
                    pos   = next_delim;
                } else {
                    value = pred_str.substr(pos);
                    pos   = pred_str.length();
                }
            }

            // Trim whitespace from value
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            kv[key] = value;

            // Skip delimiter
            if (pos < pred_str.length() && (pred_str[pos] == ',' || pred_str[pos] == ';'))
                pos++;
            while (pos < pred_str.length() && (pred_str[pos] == ' ' || pred_str[pos] == '\t'))
                pos++;
        }

        return kv;
    }

    // Convert parsed key-values to a structured predicate
    static ParsedPredicate kvToPredicate(const std::map< std::string, std::string > &kv) {
        ParsedPredicate pred;

        auto kind_it = kv.find("kind");
        if (kind_it == kv.end()) {
            llvm::errs() << "Warning: predicate missing 'kind' key\n";
            return pred;
        }

        std::string kind_str = kind_it->second;

        // Parse target
        auto target_it = kv.find("target");
        if (target_it != kv.end()) {
            parseTarget(target_it->second, pred.target, pred.arg_index);
        }

        // Determine predicate kind
        if (kind_str == "nonnull") {
            pred.kind = PK_Nonnull;
        } else if (kind_str == "relation") {
            // Need to look at relation field
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
                    llvm::errs() << "Warning: failed to parse relation value '"
                                 << val_it->second << "': " << e.what() << "\n";
                }
            }
        } else if (kind_str == "range") {
            // Check if target is ReturnValue or Arg
            if (pred.target == "ReturnValue") {
                pred.kind = PK_RangeRet;
            } else if (pred.target.substr(0, 3) == "Arg") {
                pred.kind = PK_RangeArg;
            }

            // Parse range values from the string
            // Look for "range=[min=X, max=Y]" pattern
            auto range_it = kv.find("range");
            if (range_it != kv.end()) {
                std::string range_str = range_it->second;
                // Try to extract min= and max=
                size_t min_pos = range_str.find("min=");
                size_t max_pos = range_str.find("max=");

                if (min_pos != std::string::npos) {
                    min_pos += 4;
                    size_t min_end = range_str.find_first_of(",]", min_pos);
                    try {
                        pred.min_val = std::stoll(range_str.substr(min_pos, min_end - min_pos));
                    } catch (const std::exception &e) {
                        llvm::errs() << "Warning: failed to parse range min value '"
                                     << range_str.substr(min_pos, min_end - min_pos)
                                     << "': " << e.what() << "\n";
                    }
                }

                if (max_pos != std::string::npos) {
                    max_pos += 4;
                    size_t max_end = range_str.find_first_of(",]", max_pos);
                    try {
                        pred.max_val = std::stoll(range_str.substr(max_pos, max_end - max_pos));
                    } catch (const std::exception &e) {
                        llvm::errs() << "Warning: failed to parse range max value '"
                                     << range_str.substr(max_pos, max_end - max_pos)
                                     << "': " << e.what() << "\n";
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
                    llvm::errs() << "Warning: failed to parse alignment value '"
                                 << align_it->second << "': " << e.what() << "\n";
                }
            }
        } else {
            llvm::errs() << "Warning: unknown predicate kind '" << kind_str << "'\n";
        }

        return pred;
    }

    // Find the position of the closing ']' that matches the opening '[' at
    // the given start position, handling nested brackets (e.g. range=[min=0, max=255]).
    // Returns std::string::npos if no matching bracket is found.
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

    // Parse the static contract metadata string
    static std::vector< ParsedPredicate >
    parseStaticContractText(const std::string &contract_str) {
        std::vector< ParsedPredicate > preds;

        // Find preconditions section
        size_t pre_start = contract_str.find("preconditions=[");
        if (pre_start != std::string::npos) {
            pre_start += 15; // Skip "preconditions=["
            size_t pre_end = findMatchingBracket(contract_str, pre_start);
            if (pre_end != std::string::npos) {
                std::string pre_section = contract_str.substr(pre_start, pre_end - pre_start);

                // Parse individual preconditions
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

        // Find postconditions section
        size_t post_start = contract_str.find("postconditions=[");
        if (post_start != std::string::npos) {
            post_start += 16; // Skip "postconditions=["
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

    // Inject verification calls from static contract
    static void injectFromStaticContract(
        llvm::Module &M, llvm::Instruction &Inst, const std::string &Txt
    ) {
        auto preds = parseStaticContractText(Txt);
        if (preds.empty())
            return;

        // Get or insert __VERIFIER_assume and __VERIFIER_assert.
        // SeaHorn's C runtime declares these with int (i32) parameters.
        auto *i32Ty = llvm::Type::getInt32Ty(M.getContext());

        llvm::FunctionCallee Assume = M.getOrInsertFunction(
            "__VERIFIER_assume",
            llvm::FunctionType::get(
                llvm::Type::getVoidTy(M.getContext()), { i32Ty }, false
            )
        );

        llvm::FunctionCallee Assert = M.getOrInsertFunction(
            "__VERIFIER_assert",
            llvm::FunctionType::get(
                llvm::Type::getVoidTy(M.getContext()), { i32Ty }, false
            )
        );

        // Determine if this is a call instruction
        llvm::CallBase *CB = llvm::dyn_cast< llvm::CallBase >(&Inst);

        // 1) Preconditions: insert right before instruction
        llvm::IRBuilder<> Bpre(&Inst);

        for (auto &P : preds) {
            if (!P.is_precondition)
                continue;

            llvm::Value *A = nullptr;

            // Get the argument value — requires a CallBase instruction
            if (!CB) {
                if (verbose)
                    llvm::errs() << "  Warning: precondition on arg" << P.arg_index
                                 << " skipped — instruction is not a call in "
                                 << Inst.getFunction()->getName() << "\n";
                continue;
            }
            if (P.arg_index < CB->arg_size()) {
                A = CB->getArgOperand(P.arg_index);
            }

            if (!A) {
                if (verbose)
                    llvm::errs() << "  Warning: precondition on arg" << P.arg_index
                                 << " skipped — arg index out of range (function has "
                                 << CB->arg_size() << " args) in "
                                 << Inst.getFunction()->getName() << "\n";
                continue;
            }

            llvm::Value *Cond = nullptr;

            switch (P.kind) {
            case PK_Nonnull: {
                if (!A->getType()->isPointerTy())
                    break;
                llvm::Value *Null = llvm::ConstantPointerNull::get(
                    llvm::cast< llvm::PointerType >(A->getType())
                );
                Cond = Bpre.CreateICmpNE(A, Null);
                if (verbose)
                    llvm::outs() << "  Precondition: nonnull on arg " << P.arg_index << "\n";
                break;
            }
            case PK_RelNeqArgConst: {
                if (!A->getType()->isIntegerTy())
                    break;
                Cond = Bpre.CreateICmpNE(
                    A, llvm::ConstantInt::getSigned(A->getType(), P.constant)
                );
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index << " != " << P.constant
                                 << "\n";
                break;
            }
            case PK_RelEqArgConst: {
                if (!A->getType()->isIntegerTy())
                    break;
                Cond = Bpre.CreateICmpEQ(
                    A, llvm::ConstantInt::getSigned(A->getType(), P.constant)
                );
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index << " == " << P.constant
                                 << "\n";
                break;
            }
            case PK_RelLtArgConst: {
                if (!A->getType()->isIntegerTy())
                    break;
                Cond = Bpre.CreateICmpSLT(
                    A, llvm::ConstantInt::getSigned(A->getType(), P.constant)
                );
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index << " < " << P.constant
                                 << "\n";
                break;
            }
            case PK_RelLeArgConst: {
                if (!A->getType()->isIntegerTy())
                    break;
                Cond = Bpre.CreateICmpSLE(
                    A, llvm::ConstantInt::getSigned(A->getType(), P.constant)
                );
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index << " <= " << P.constant
                                 << "\n";
                break;
            }
            case PK_RelGtArgConst: {
                if (!A->getType()->isIntegerTy())
                    break;
                Cond = Bpre.CreateICmpSGT(
                    A, llvm::ConstantInt::getSigned(A->getType(), P.constant)
                );
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index << " > " << P.constant
                                 << "\n";
                break;
            }
            case PK_RelGeArgConst: {
                if (!A->getType()->isIntegerTy())
                    break;
                Cond = Bpre.CreateICmpSGE(
                    A, llvm::ConstantInt::getSigned(A->getType(), P.constant)
                );
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index << " >= " << P.constant
                                 << "\n";
                break;
            }
            case PK_RangeArg: {
                if (!A->getType()->isIntegerTy())
                    break;
                llvm::Value *A64 = toI64(Bpre, A);
                if (!A64)
                    break;
                auto *i64     = llvm::Type::getInt64Ty(M.getContext());
                llvm::Value *lo =
                    Bpre.CreateICmpSGE(A64, llvm::ConstantInt::getSigned(i64, P.min_val));
                llvm::Value *hi =
                    Bpre.CreateICmpSLE(A64, llvm::ConstantInt::getSigned(i64, P.max_val));
                Cond            = Bpre.CreateAnd(lo, hi);
                if (verbose)
                    llvm::outs() << "  Precondition: " << P.min_val << " <= arg" << P.arg_index
                                 << " <= " << P.max_val << "\n";
                break;
            }
            case PK_Alignment: {
                if (!A->getType()->isPointerTy() || P.alignment == 0)
                    break;
                llvm::Type *intptr_ty = Bpre.getIntPtrTy(M.getDataLayout());
                llvm::Value *ptr_int  = Bpre.CreatePtrToInt(A, intptr_ty);
                llvm::Value *align_val = llvm::ConstantInt::get(intptr_ty, P.alignment);
                llvm::Value *mod       = Bpre.CreateURem(ptr_int, align_val);
                llvm::Value *zero      = llvm::ConstantInt::get(intptr_ty, 0);
                Cond                   = Bpre.CreateICmpEQ(mod, zero);
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index << " aligned to "
                                 << P.alignment << "\n";
                break;
            }
            default:
                if (verbose)
                    llvm::errs() << "  Warning: unhandled precondition kind "
                                 << static_cast< int >(P.kind) << " in "
                                 << Inst.getFunction()->getName() << "\n";
                break;
            }

            if (Cond) {
                llvm::Value *CondI32 = Bpre.CreateZExt(Cond, i32Ty);
                Bpre.CreateCall(Assume, { CondI32 });
            }
        }

        // 2) Postconditions: insert immediately after instruction.
        // Check whether there are any postconditions before doing work.
        bool has_postconditions = false;
        for (auto &P : preds) {
            if (!P.is_precondition) {
                has_postconditions = true;
                break;
            }
        }
        if (!has_postconditions)
            return;

        // If the instruction is at the end of its basic block (e.g. an invoke
        // terminator), split the block to create a valid insertion point.
        llvm::Instruction *After = Inst.getNextNode();
        if (!After) {
            if (Inst.isTerminator()) {
                if (verbose)
                    llvm::errs() << "  Warning: cannot inject postconditions after "
                                    "terminator instruction in "
                                 << Inst.getFunction()->getName() << "\n";
                return;
            }
            // Non-terminator at block end — split to create a successor block.
            llvm::BasicBlock *BB = Inst.getParent();
            llvm::BasicBlock *NewBB =
                BB->splitBasicBlock(BB->getTerminator(), "post_contract");
            After = &*NewBB->begin();
        }

        llvm::IRBuilder<> Bpost(After);

        // For call instructions, the return value is the call itself
        llvm::Value *RetV = nullptr;
        if (CB && !CB->getType()->isVoidTy()) {
            RetV = CB;
        }

        for (auto &P : preds) {
            if (P.is_precondition)
                continue;

            llvm::Value *Cond = nullptr;

            switch (P.kind) {
            case PK_RangeRet: {
                if (!RetV || !RetV->getType()->isIntegerTy())
                    break;

                llvm::Value *Ri64 = toI64(Bpost, RetV);
                if (!Ri64)
                    break;

                auto *i64 = llvm::Type::getInt64Ty(M.getContext());
                llvm::Value *lo =
                    Bpost.CreateICmpSGE(Ri64, llvm::ConstantInt::getSigned(i64, P.min_val));
                llvm::Value *hi =
                    Bpost.CreateICmpSLE(Ri64, llvm::ConstantInt::getSigned(i64, P.max_val));
                Cond            = Bpost.CreateAnd(lo, hi);

                if (verbose)
                    llvm::outs() << "  Postcondition: " << P.min_val << " <= return <= "
                                 << P.max_val << "\n";
                break;
            }
            case PK_Nonnull: {
                // Postcondition on return value
                if (P.target == "ReturnValue" && RetV && RetV->getType()->isPointerTy()) {
                    llvm::Value *Null = llvm::ConstantPointerNull::get(
                        llvm::cast< llvm::PointerType >(RetV->getType())
                    );
                    Cond = Bpost.CreateICmpNE(RetV, Null);
                    if (verbose)
                        llvm::outs() << "  Postcondition: return nonnull\n";
                }
                break;
            }
            default:
                if (verbose)
                    llvm::errs() << "  Warning: unhandled postcondition kind "
                                 << static_cast< int >(P.kind) << " in "
                                 << Inst.getFunction()->getName() << "\n";
                break;
            }

            if (Cond) {
                // Postconditions use assert
                llvm::Value *CondI32 = Bpost.CreateZExt(Cond, i32Ty);
                Bpost.CreateCall(Assert, { CondI32 });
            }
        }
    }

    // Process all instructions with static contract metadata
    static unsigned processModule(llvm::Module &M) {
        unsigned count = 0;

        for (auto &F : M) {
            for (auto &BB : F) {
                // Collect instructions with metadata first (to avoid iterator invalidation)
                std::vector< std::pair< llvm::Instruction *, std::string > > worklist;

                for (auto &I : BB) {
                    auto *contract_md = I.getMetadata("static_contract");
                    if (!contract_md)
                        continue;

                    auto *tuple = llvm::dyn_cast< llvm::MDTuple >(contract_md);
                    if (!tuple) {
                        llvm::errs() << "Warning: static_contract metadata is not an MDTuple"
                                     << " in " << F.getName() << "\n";
                        continue;
                    }
                    if (tuple->getNumOperands() < 2) {
                        llvm::errs() << "Warning: static_contract metadata has "
                                     << tuple->getNumOperands() << " operand(s), expected >= 2"
                                     << " in " << F.getName() << "\n";
                        continue;
                    }
                    auto *md_str =
                        llvm::dyn_cast< llvm::MDString >(tuple->getOperand(1));
                    if (!md_str) {
                        llvm::errs() << "Warning: static_contract metadata operand(1) is not"
                                     << " an MDString in " << F.getName() << "\n";
                        continue;
                    }
                    worklist.push_back({ &I, md_str->getString().str() });
                }

                // Process collected instructions
                for (auto &[inst, contract_str] : worklist) {
                    if (verbose) {
                        llvm::outs() << "\nFound static contract in function: " << F.getName()
                                     << "\n";
                        llvm::outs() << "Contract: " << contract_str << "\n";
                    }

                    injectFromStaticContract(M, *inst, contract_str);
                    count++;
                }
            }
        }

        return count;
    }

    // Write module to output file or stdout when output_filename is "-".
    static bool writeModuleToFile(llvm::Module &module, llvm::StringRef output_filename) {
        if (output_filename == "-") {
            if (emit_ll) {
                module.print(llvm::outs(), nullptr);
            } else {
                llvm::WriteBitcodeToFile(module, llvm::outs());
            }
            return true;
        }

        std::error_code ec;
        llvm::raw_fd_ostream os(output_filename, ec, llvm::sys::fs::OF_None);
        if (ec) {
            llvm::errs() << "Error opening " << output_filename << ": " << ec.message() << "\n";
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

// Main function
int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);
    llvm::cl::ParseCommandLineOptions(argc, argv, "patchir-seahorn-verifier\n");

    llvm::LLVMContext context;
    llvm::SMDiagnostic err;

    // Load the LLVM IR module
    std::unique_ptr< llvm::Module > module = llvm::parseIRFile(input_filename, err, context);
    if (!module) {
        err.print(argv[0], llvm::errs());
        return EXIT_FAILURE;
    }

    if (verbose) {
        llvm::outs() << "Loaded module: " << module->getName() << "\n";
        llvm::outs() << "Processing static contracts and generating SeaHorn invariants...\n\n";
    }

    // Process the module
    unsigned count = processModule(*module);

    if (count == 0) {
        llvm::errs() << "Warning: no static_contract metadata found in module — "
                        "output will contain no verification predicates\n";
    }

    if (verbose) {
        llvm::outs() << "\nProcessed " << count << " instruction(s) with static contracts\n";
    }

    // Write output
    if (!writeModuleToFile(*module, output_filename)) {
        llvm::errs() << "Failed to write output\n";
        return EXIT_FAILURE;
    }

    if (verbose) {
        llvm::outs() << "Successfully wrote output to: " << output_filename << "\n";
    }

    return EXIT_SUCCESS;
}
