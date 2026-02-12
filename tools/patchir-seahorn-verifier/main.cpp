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

#include "contract_parser.h"
#include "utils.h"

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
    // ============================================================================
    // EXTENSIBILITY GUIDE: Adding New Predicate Types
    // ============================================================================
    //
    // This tool converts static contract metadata into SeaHorn verification calls.
    // To add support for a new predicate type (e.g., buffer bounds checking),
    // follow these steps:
    //
    // 1. ADD ENUM VALUE
    //    Add new value to PredicateKind enum (e.g., PK_BufferBounds)
    //    Update getPredicateKindName() to return name for the new kind
    //    Update getSupportedPredicateKinds() to include the new kind
    //
    // 2. EXTEND PARSED PREDICATE STRUCTURE
    //    Add fields to ParsedPredicate for new predicate's parameters
    //    Example: unsigned buffer_size_arg = 0;
    //
    // 3. ADD PARSING LOGIC
    //    In kvToPredicate(), add new else-if branch for the predicate kind
    //    Parse required fields from key-value map
    //    Validate inputs and return descriptive errors
    //    Example:
    //      else if (kind_str == "buffer_bounds") {
    //          pred.kind = PK_BufferBounds;
    //          auto size_it = kv.find("size_arg");
    //          if (size_it == kv.end()) {
    //              return ParseResult<ParsedPredicate>::error(
    //                  "Missing 'size_arg' field for buffer_bounds predicate");
    //          }
    //          // ... parse and validate size_arg
    //      }
    //
    // 4. ADD CODE GENERATION
    //    In injectPredicates(), add new case to the switch statement
    //    Generate appropriate LLVM IR comparison/check
    //    Call __VERIFIER_assume for preconditions or __VERIFIER_assert for postconditions
    //    Example:
    //      case PK_BufferBounds: {
    //          llvm::Value *Buf = CB->getArgOperand(P.buffer_arg_index);
    //          llvm::Value *Size = CB->getArgOperand(P.size_arg_index);
    //          // Generate: size >= min_access
    //          Cond = Bpre.CreateICmpUGE(Size, ConstantInt::get(..., P.min_access));
    //          break;
    //      }
    //
    // 5. UPDATE DOCUMENTATION
    //    Add example to tools/patchir-seahorn-verifier/README.md
    //    Document contract metadata format
    //    Add test case to test/fixtures/
    //
    // 6. ADD TESTS
    //    Create test .ll file with new predicate type
    //    Verify parser correctly handles the new predicate
    //    Verify code generation produces expected LLVM IR
    //
    // EXAMPLE: Buffer bounds checking predicate
    // ------------------------------------------
    // Contract metadata format:
    //   {kind=buffer_bounds, buffer=Arg(0), size=Arg(1), min_access=100}
    //
    // Meaning: Verify that Arg(1) >= 100 (buffer has at least 100 bytes)
    //
    // Generated code:
    //   %cmp = icmp uge i64 %size, 100
    //   call void @__VERIFIER_assume(i1 %cmp)
    //
    // ============================================================================

    // Forward declaration
    static void injectPredicates(
        llvm::Module &M,
        llvm::Instruction &Inst,
        const std::vector< ParsedPredicate > &preds,
        llvm::FunctionCallee Assume,
        llvm::FunctionCallee Assert
    );

    // Process all instructions with static contract metadata
    static unsigned processModule(llvm::Module &M) {
        unsigned count = 0;
        unsigned errors = 0;
        unsigned warnings = 0;

        // Create parser instance
        ContractParser parser(verbose);

        for (auto &F : M) {
            unsigned instruction_index = 0;

            for (auto &BB : F) {
                // Collect instructions with metadata first (to avoid iterator invalidation)
                std::vector< std::tuple< llvm::Instruction *, std::string, unsigned > >
                    worklist;

                for (auto &I : BB) {
                    if (auto *contract_md = I.getMetadata("static_contract")) {
                        if (auto *tuple = llvm::dyn_cast< llvm::MDTuple >(contract_md)) {
                            if (tuple->getNumOperands() >= 2) {
                                if (auto *md_str = llvm::dyn_cast< llvm::MDString >(
                                        tuple->getOperand(1)
                                    ))
                                {
                                    worklist.push_back(
                                        { &I, md_str->getString().str(), instruction_index }
                                    );
                                }
                            }
                        }
                    }
                    instruction_index++;
                }

                // Process collected instructions
                for (auto &[inst, contract_str, inst_idx] : worklist) {
                    if (verbose) {
                        llvm::outs() << "\n=== Processing contract in function: "
                                     << F.getName() << " ===\n";
                        llvm::outs() << "Instruction #" << inst_idx << "\n";
                        llvm::outs() << "Contract: " << contract_str << "\n";
                    }

                    // Parse contract with error collection using ContractParser
                    std::vector< std::string > parse_errors;
                    auto preds = parser.parseStaticContractText(contract_str, &parse_errors);

                    // Report any parsing errors with context
                    if (!parse_errors.empty()) {
                        ContractLocation loc{
                            F.getName().str(), inst_idx, contract_str
                        };

                        for (const auto &err_msg : parse_errors) {
                            ERROR(loc, err_msg);
                            errors++;
                        }

                        // Skip this contract if there were errors
                        if (preds.empty()) {
                            continue;
                        }
                    }

                    // If we got at least some valid predicates, inject them
                    if (!preds.empty()) {
                        // Get __VERIFIER_assume and __VERIFIER_assert using helper functions
                        llvm::FunctionCallee Assume = getAssumeFn(M);
                        llvm::FunctionCallee Assert = getAssertFn(M);

                        // Process the predicates
                        injectPredicates(M, *inst, preds, Assume, Assert);
                        count++;
                    }
                }
            }
        }

        // Summary
        if (errors > 0 || warnings > 0) {
            llvm::errs() << "\n";
            llvm::errs() << "===========================================\n";
            llvm::errs() << "Contract Processing Summary\n";
            llvm::errs() << "===========================================\n";
            llvm::errs() << "Successfully processed: " << count << " contract(s)\n";
            if (errors > 0) {
                llvm::errs() << "Errors encountered:     " << errors << "\n";
            }
            if (warnings > 0) {
                llvm::errs() << "Warnings:               " << warnings << "\n";
            }
            llvm::errs() << "===========================================\n\n";
        }

        return count;
    }

    // Helper to inject predicates (extracted from injectFromStaticContract)
    static void injectPredicates(
        llvm::Module &module,
        llvm::Instruction &inst,
        const std::vector< ParsedPredicate > &preds,
        llvm::FunctionCallee assume_fn,
        llvm::FunctionCallee assert_fn
    ) {
        // Determine if this is a call instruction
        llvm::CallBase *call_base = llvm::dyn_cast< llvm::CallBase >(&inst);

        // 1) Preconditions: insert right before instruction
        llvm::IRBuilder<> pre_builder(&inst);

        for (auto &P : preds) {
            if (!P.is_precondition)
                continue;

            llvm::Value *arg_value = nullptr;

            // Get the argument value
            if (call_base && P.arg_index < call_base->arg_size()) {
                arg_value = call_base->getArgOperand(P.arg_index);
            }

            if (!arg_value)
                continue;

            llvm::Value *cond = nullptr;

            switch (P.kind) {
            case PK_Nonnull: {
                if (!arg_value->getType()->isPointerTy())
                    break;
                llvm::Value *null_ptr = llvm::ConstantPointerNull::get(
                    llvm::cast< llvm::PointerType >(arg_value->getType())
                );
                cond = pre_builder.CreateICmpNE(arg_value, null_ptr);
                if (verbose)
                    llvm::outs() << "  Precondition: nonnull on arg " << P.arg_index
                                 << "\n";
                break;
            }
            case PK_RelNeqArgConst: {
                if (!arg_value->getType()->isIntegerTy())
                    break;
                cond = pre_builder.CreateICmpNE(
                    arg_value,
                    llvm::ConstantInt::get(
                        arg_value->getType(), static_cast< uint64_t >(P.constant)
                    )
                );
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index << " != "
                                 << P.constant << "\n";
                break;
            }
            case PK_RelEqArgConst: {
                if (!arg_value->getType()->isIntegerTy())
                    break;
                cond = pre_builder.CreateICmpEQ(
                    arg_value,
                    llvm::ConstantInt::get(
                        arg_value->getType(), static_cast< uint64_t >(P.constant)
                    )
                );
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index << " == "
                                 << P.constant << "\n";
                break;
            }
            case PK_RelLtArgConst: {
                if (!arg_value->getType()->isIntegerTy())
                    break;
                cond = pre_builder.CreateICmpSLT(
                    arg_value,
                    llvm::ConstantInt::get(
                        arg_value->getType(), static_cast< uint64_t >(P.constant)
                    )
                );
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index << " < "
                                 << P.constant << "\n";
                break;
            }
            case PK_RelLeArgConst: {
                if (!arg_value->getType()->isIntegerTy())
                    break;
                cond = pre_builder.CreateICmpSLE(
                    arg_value,
                    llvm::ConstantInt::get(
                        arg_value->getType(), static_cast< uint64_t >(P.constant)
                    )
                );
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index << " <= "
                                 << P.constant << "\n";
                break;
            }
            case PK_RelGtArgConst: {
                if (!arg_value->getType()->isIntegerTy())
                    break;
                cond = pre_builder.CreateICmpSGT(
                    arg_value,
                    llvm::ConstantInt::get(
                        arg_value->getType(), static_cast< uint64_t >(P.constant)
                    )
                );
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index << " > "
                                 << P.constant << "\n";
                break;
            }
            case PK_RelGeArgConst: {
                if (!arg_value->getType()->isIntegerTy())
                    break;
                cond = pre_builder.CreateICmpSGE(
                    arg_value,
                    llvm::ConstantInt::get(
                        arg_value->getType(), static_cast< uint64_t >(P.constant)
                    )
                );
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index << " >= "
                                 << P.constant << "\n";
                break;
            }
            case PK_RangeArg: {
                if (!arg_value->getType()->isIntegerTy())
                    break;
                llvm::Value *arg_i64 = toI64(pre_builder, arg_value);
                if (!arg_i64)
                    break;
                auto *i64 = llvm::Type::getInt64Ty(module.getContext());
                llvm::Value *lo = pre_builder.CreateICmpSGE(
                    arg_i64,
                    llvm::ConstantInt::get(i64, static_cast< uint64_t >(P.min_val))
                );
                llvm::Value *hi = pre_builder.CreateICmpSLE(
                    arg_i64,
                    llvm::ConstantInt::get(i64, static_cast< uint64_t >(P.max_val))
                );
                cond = pre_builder.CreateAnd(lo, hi);
                if (verbose)
                    llvm::outs() << "  Precondition: " << P.min_val << " <= arg"
                                 << P.arg_index << " <= " << P.max_val << "\n";
                break;
            }
            case PK_Alignment: {
                if (!arg_value->getType()->isPointerTy())
                    break;
                llvm::Type *intptr_ty = pre_builder.getIntPtrTy(module.getDataLayout());
                llvm::Value *ptr_int = pre_builder.CreatePtrToInt(arg_value, intptr_ty);
                llvm::Value *align_val =
                    llvm::ConstantInt::get(intptr_ty, P.alignment);
                llvm::Value *mod = pre_builder.CreateURem(ptr_int, align_val);
                llvm::Value *zero = llvm::ConstantInt::get(intptr_ty, 0);
                cond = pre_builder.CreateICmpEQ(mod, zero);
                if (verbose)
                    llvm::outs() << "  Precondition: arg" << P.arg_index
                                 << " aligned to " << P.alignment << "\n";
                break;
            }
            default:
                // Unknown or unimplemented predicate kind
                if (verbose || P.kind != PK_Unknown) {
                    llvm::errs() << "Warning: No code generation for precondition kind: "
                                 << ContractParser::getPredicateKindName(P.kind)
                                 << " (predicate will be ignored)\n";
                }
                break;
            }

            if (cond) {
                pre_builder.CreateCall(assume_fn, { cond });
            }
        }

        // 2) Postconditions: insert immediately after instruction
        llvm::Instruction *after_inst = inst.getNextNode();
        if (!after_inst)
            return;

        llvm::IRBuilder<> post_builder(after_inst);

        // For call instructions, the return value is the call itself
        llvm::Value *ret_value = nullptr;
        if (call_base && !call_base->getType()->isVoidTy()) {
            ret_value = call_base;
        }

        for (auto &P : preds) {
            if (P.is_precondition)
                continue;

            llvm::Value *cond = nullptr;

            switch (P.kind) {
            case PK_RangeRet: {
                if (!ret_value || !ret_value->getType()->isIntegerTy())
                    break;

                llvm::Value *ret_i64 = toI64(post_builder, ret_value);
                if (!ret_i64)
                    break;

                auto *i64 = llvm::Type::getInt64Ty(module.getContext());
                llvm::Value *lo = post_builder.CreateICmpSGE(
                    ret_i64,
                    llvm::ConstantInt::get(i64, static_cast< uint64_t >(P.min_val))
                );
                llvm::Value *hi = post_builder.CreateICmpSLE(
                    ret_i64,
                    llvm::ConstantInt::get(i64, static_cast< uint64_t >(P.max_val))
                );
                cond = post_builder.CreateAnd(lo, hi);

                if (verbose)
                    llvm::outs() << "  Postcondition: " << P.min_val
                                 << " <= return <= " << P.max_val << "\n";
                break;
            }
            case PK_Nonnull: {
                // Postcondition on return value
                if (P.target == "ReturnValue" && ret_value &&
                    ret_value->getType()->isPointerTy()) {
                    llvm::Value *null_ptr = llvm::ConstantPointerNull::get(
                        llvm::cast< llvm::PointerType >(ret_value->getType())
                    );
                    cond = post_builder.CreateICmpNE(ret_value, null_ptr);
                    if (verbose)
                        llvm::outs() << "  Postcondition: return nonnull\n";
                }
                break;
            }
            default:
                // Unknown or unimplemented predicate kind
                if (verbose || P.kind != PK_Unknown) {
                    llvm::errs() << "Warning: No code generation for postcondition kind: "
                                 << ContractParser::getPredicateKindName(P.kind)
                                 << " (predicate will be ignored)\n";
                }
                break;
            }

            if (cond) {
                // Postconditions use assert
                post_builder.CreateCall(assert_fn, { cond });
            }
        }
    }

    // Write module to output file
    static bool writeModuleToFile(llvm::Module &module, llvm::StringRef output_filename) {
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
