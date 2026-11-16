/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// Product harness: one module that holds the lifted original and the patched
// version of a function, and a main() that runs both on the same input and
// classifies the outcome.
//
// Each side keeps its own copy of every global (suffix `.orig`, `.patched`),
// so neither run sees what the other wrote.  Writable globals that hold no
// pointer start symbolic and equal on both sides; the others keep their
// initializers.  Pointer arguments, and integer arguments declared as
// buffers, point at one buffer both runs use: its contents are symbolic,
// saved before the first run and restored before the second, so a returned
// pointer into it compares equal.
//
// The patched version runs first.  A path ends with a call to
// `__patchestry_path_end(code)`:
//
//    0  both versions accepted the input and returned the same value
//    1  the patched version rejected the input
//   10  P1: the patched version accepted an input the original rejects
//   11  P3: both accepted and the return values differ
//   20  inconclusive: a version called an external function the harness
//       cannot model (one that returns a pointer or a non-integer)
//   30  the original, run alone, accepted the input
//   31  the original, run alone, rejected the input
//
// The harness runs the original alone when `__patchestry_only_original` is
// set.  A runner on concrete inputs uses that after the patched version
// crashed: under symbolic execution the flag stays 0.
//
// A version rejects an input when it calls a declared reject handler or
// returns a declared reject value.  An external function is replaced by a
// stub that returns the same value at the same call index on both sides.

#include "Compare.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/Util/Log.hpp>

#include "HarnessGen.hpp"
#include "Runtime.hpp"

namespace patchestry::klee_verifier {

    extern llvm::FunctionCallee getKleeMakeSymbolic(llvm::Module &M);
    extern llvm::FunctionCallee getMalloc(llvm::Module &M);

    namespace {

        constexpr unsigned kAccepted        = 0;
        constexpr unsigned kPatchedRejected = 1;
        constexpr unsigned kP1Violation     = 10;
        constexpr unsigned kP3Violation     = 11;
        constexpr unsigned kUnmodelled      = 20;
        constexpr unsigned kOriginalAloneAccepted = 30;
        constexpr unsigned kOriginalAloneRejected = 31;
        constexpr unsigned kPhaseOriginal   = 1;
        constexpr unsigned kPhasePatched    = 2;

        // Provided by whatever runs the harness; never given a stub.
        bool isRuntimeName(llvm::StringRef name) {
            static const char *const names[] = {
                "malloc", "free", "memcpy", "memmove", "memset", "klee_make_symbolic",
                "klee_assume", "klee_abort", "__patchestry_path_end",
            };
            for (const char *runtime_name : names) {
                if (name == runtime_name) {
                    return true;
                }
            }
            return name.starts_with("llvm.") || name.starts_with("klee_");
        }

        // Give every definition and every global of `M` a per-side name, and
        // storage to globals the module only declares.
        void suffixSide(llvm::Module &M, llvm::StringRef suffix) {
            for (auto &global : M.globals()) {
                if (global.getName().starts_with("llvm.")) {
                    continue;
                }
                if (global.isDeclaration() && global.getValueType()->isSized()) {
                    global.setInitializer(llvm::Constant::getNullValue(global.getValueType()));
                    global.setLinkage(llvm::GlobalValue::InternalLinkage);
                }
                global.setName(global.getName() + suffix);
            }
            for (auto &function : M) {
                if (!function.isDeclaration()) {
                    function.setName(function.getName() + suffix);
                }
            }
        }

        llvm::GlobalVariable *
        newGlobal(llvm::Module &M, llvm::Type *type, const llvm::Twine &name) {
            return new llvm::GlobalVariable(
                M, type, false, llvm::GlobalValue::InternalLinkage,
                llvm::Constant::getNullValue(type), name
            );
        }

        llvm::Constant *nameString(llvm::IRBuilder<> &builder, llvm::StringRef text) {
            return builder.CreateGlobalString(text, "__patchestry_name");
        }

        struct Harness
        {
            llvm::Module &module;
            const CompareOptions &options;
            llvm::LLVMContext &context;
            llvm::Type *i32;
            llvm::GlobalVariable *phase;
            llvm::GlobalVariable *rejected;
            llvm::FunctionCallee path_end;
            std::vector< llvm::GlobalVariable * > call_counters;
            std::vector< llvm::GlobalVariable * > value_tables;
            llvm::json::Array stubs;

            Harness(llvm::Module &M, const CompareOptions &opts)
                : module(M)
                , options(opts)
                , context(M.getContext())
                , i32(llvm::Type::getInt32Ty(M.getContext()))
                , phase(new llvm::GlobalVariable(
                      // External: a runner reads which side was running.
                      M, i32, false, llvm::GlobalValue::ExternalLinkage,
                      llvm::ConstantInt::get(i32, 0), "__patchestry_phase"
                  ))
                , rejected(newGlobal(M, i32, "__patchestry_rejected")) {
                auto *type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(context), { i32 }, false
                );
                path_end = M.getOrInsertFunction("__patchestry_path_end", type);
                llvm::cast< llvm::Function >(path_end.getCallee())->setDoesNotReturn();
            }

            void endPath(llvm::IRBuilder<> &builder, unsigned code) {
                builder.CreateCall(path_end, { llvm::ConstantInt::get(i32, code) });
                builder.CreateUnreachable();
            }

            // A reject handler records the rejection.  One that does not
            // return ends the path where it is called.
            void stubRejectHandler(llvm::Function &function) {
                auto *entry = llvm::BasicBlock::Create(context, "entry", &function);
                llvm::IRBuilder<> builder(entry);
                if (!function.doesNotReturn()) {
                    builder.CreateStore(llvm::ConstantInt::get(i32, 1), rejected);
                    if (function.getReturnType()->isVoidTy()) {
                        builder.CreateRetVoid();
                    } else {
                        builder.CreateRet(llvm::Constant::getNullValue(function.getReturnType()));
                    }
                    return;
                }
                auto *original = llvm::BasicBlock::Create(context, "original", &function);
                auto *patched  = llvm::BasicBlock::Create(context, "patched", &function);
                auto *running  = builder.CreateLoad(i32, phase);
                builder.CreateCondBr(
                    builder.CreateICmpEQ(running, llvm::ConstantInt::get(i32, kPhaseOriginal)),
                    original, patched
                );
                builder.SetInsertPoint(original);
                endPath(builder, kP1Violation);
                builder.SetInsertPoint(patched);
                endPath(builder, kPatchedRejected);
            }

            // The n-th call returns the n-th entry of a symbolic table, the
            // same table for both sides.
            void stubValues(llvm::Function &function) {
                auto *entry = llvm::BasicBlock::Create(context, "entry", &function);
                llvm::IRBuilder<> builder(entry);
                auto *result = function.getReturnType();
                if (result->isVoidTy()) {
                    builder.CreateRetVoid();
                    return;
                }
                auto *table_type = llvm::ArrayType::get(result, options.stub_values);
                auto *table = newGlobal(
                    module, table_type, "__patchestry_values." + function.getName()
                );
                auto *counter =
                    newGlobal(module, i32, "__patchestry_calls." + function.getName());
                value_tables.push_back(table);
                call_counters.push_back(counter);
                auto *index = builder.CreateLoad(i32, counter);
                builder.CreateStore(
                    builder.CreateAdd(index, llvm::ConstantInt::get(i32, 1)), counter
                );
                auto *last    = llvm::ConstantInt::get(i32, options.stub_values - 1);
                auto *clamped = builder.CreateSelect(builder.CreateICmpULT(index, last), index, last);
                auto *slot    = builder.CreateInBoundsGEP(
                    table_type, table, { llvm::ConstantInt::get(i32, 0), clamped }
                );
                builder.CreateRet(builder.CreateLoad(result, slot));
            }

            // False when a declared reject handler is never called.
            bool stubExternals() {
                std::set< std::string > handlers(
                    options.reject_handlers.begin(), options.reject_handlers.end()
                );
                std::set< std::string > seen;
                for (auto &function : module) {
                    if (!function.isDeclaration() || function.use_empty()
                        || isRuntimeName(function.getName()))
                    {
                        continue;
                    }
                    const std::string name = function.getName().str();
                    llvm::json::Object row{ { "name", name } };
                    if (handlers.count(name) != 0U) {
                        seen.insert(name);
                        row["kind"] = function.doesNotReturn() ? "reject-noreturn" : "reject";
                        stubRejectHandler(function);
                    } else if (function.getReturnType()->isVoidTy()) {
                        row["kind"] = "void";
                        stubValues(function);
                    } else if (function.getReturnType()->isIntegerTy()) {
                        row["kind"] = "values";
                        stubValues(function);
                    } else {
                        // A pointer or a float from nowhere cannot be made
                        // up; a path that needs one is inconclusive.
                        row["kind"] = "unmodelled";
                        llvm::IRBuilder<> builder(
                            llvm::BasicBlock::Create(context, "entry", &function)
                        );
                        endPath(builder, kUnmodelled);
                    }
                    function.setLinkage(llvm::GlobalValue::InternalLinkage);
                    stubs.push_back(std::move(row));
                }
                for (const auto &handler : options.reject_handlers) {
                    if (seen.count(handler) == 0U) {
                        LOG(ERROR) << "reject handler '" << handler
                                   << "' is not called by either version\n";
                        return false;
                    }
                }
                return true;
            }
        };

    } // namespace

    std::unique_ptr< llvm::Module > buildProductHarness(
        std::unique_ptr< llvm::Module > original, std::unique_ptr< llvm::Module > patched,
        const std::string &target, const CompareOptions &options, llvm::json::Object &info
    ) {
        auto *original_fn = original->getFunction(target);
        auto *patched_fn  = patched->getFunction(target);
        if (original_fn == nullptr || original_fn->isDeclaration() || patched_fn == nullptr
            || patched_fn->isDeclaration())
        {
            LOG(ERROR) << "target function '" << target
                       << "' is not defined in both the original and the patched module\n";
            return nullptr;
        }
        bool layout_adopted = false;
        if (original->getDataLayoutStr() != patched->getDataLayoutStr()) {
            // Two front ends spell the layout of one ABI differently.  The
            // patched module takes the original's when the two agree on
            // what decides the size and place of data.
            const auto &ours   = original->getDataLayout();
            const auto &theirs = patched->getDataLayout();
            auto &context      = original->getContext();
            auto *i64          = llvm::Type::getInt64Ty(context);
            auto *f64          = llvm::Type::getDoubleTy(context);
            if (ours.isLittleEndian() != theirs.isLittleEndian()
                || ours.getPointerSizeInBits() != theirs.getPointerSizeInBits()
                || ours.getPointerABIAlignment(0) != theirs.getPointerABIAlignment(0)
                || ours.getABITypeAlign(i64) != theirs.getABITypeAlign(i64)
                || ours.getABITypeAlign(f64) != theirs.getABITypeAlign(f64))
            {
                LOG(ERROR) << "the original and the patched module have different data layouts\n";
                return nullptr;
            }
            patched->setDataLayout(ours);
            patched->setTargetTriple(original->getTargetTriple());
            layout_adopted = true;
        }
        auto *signature = original_fn->getFunctionType();
        if (signature != patched_fn->getFunctionType() || signature->isVarArg()) {
            LOG(ERROR) << "target function '" << target
                       << "' must have the same, non-variadic type in both modules\n";
            return nullptr;
        }
        for (const auto &[index, size] : options.arg_buffers) {
            if (index >= signature->getNumParams()
                || !(signature->getParamType(index)->isPointerTy()
                     || signature->getParamType(index)->isIntegerTy())
                || size == 0)
            {
                LOG(ERROR) << "--arg-buffer names argument " << index << " of '" << target
                           << "', which is not a pointer or integer argument, or gives no size\n";
                return nullptr;
            }
        }

        for (const auto &[index, range] : options.arg_ranges) {
            if (index >= signature->getNumParams()
                || !signature->getParamType(index)->isIntegerTy()
                || options.arg_buffers.count(index) != 0U || range.first > range.second)
            {
                LOG(ERROR) << "--arg-range names argument " << index << " of '" << target
                           << "', which is not a plain integer argument, or gives an empty range\n";
                return nullptr;
            }
        }

        // Names of the original's globals, to pair them after the link.
        std::vector< std::string > global_names;
        for (auto &global : original->globals()) {
            if (!global.getName().starts_with("llvm.")) {
                global_names.push_back(global.getName().str());
            }
        }
        suffixSide(*original, ".orig");
        suffixSide(*patched, ".patched");
        if (llvm::Linker::linkModules(*original, std::move(patched))) {
            LOG(ERROR) << "failed to link the patched module with the original\n";
            return nullptr;
        }
        llvm::Module &module = *original;
        rewriteAbortCalls(module);

        Harness harness(module, options);
        if (!harness.stubExternals()) {
            return nullptr;
        }

        auto &context  = module.getContext();
        const auto &dl = module.getDataLayout();
        auto *i32      = harness.i32;
        auto *size_ty  = dl.getIntPtrType(context);
        auto *main_fn  = llvm::Function::Create(
            llvm::FunctionType::get(i32, false), llvm::GlobalValue::ExternalLinkage, "main",
            module
        );
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", main_fn));
        auto make_symbolic = getKleeMakeSymbolic(module);
        auto allocate      = getMalloc(module);
        auto symbolic = [&](llvm::Value *address, uint64_t size, llvm::StringRef name) {
            builder.CreateCall(
                make_symbolic,
                { address, llvm::ConstantInt::get(size_ty, size), nameString(builder, name) }
            );
        };

        // Globals: symbolic and equal where that is sound, otherwise as lifted.
        llvm::json::Array symbolic_globals;
        llvm::json::Array concrete_globals;
        for (const auto &name : global_names) {
            auto *ours   = module.getNamedGlobal(name + ".orig");
            auto *theirs = module.getNamedGlobal(name + ".patched");
            if (ours == nullptr || ours->isConstant() || !ours->getValueType()->isSized()) {
                continue;
            }
            llvm::SmallPtrSet< llvm::Type *, 8 > seen;
            const uint64_t size = dl.getTypeAllocSize(ours->getValueType()).getFixedValue();
            if (theirs == nullptr || theirs->isConstant()
                || theirs->getValueType() != ours->getValueType()
                || typeContainsPointer(ours->getValueType(), seen) || size == 0)
            {
                concrete_globals.push_back(name);
                continue;
            }
            symbolic(ours, size, name);
            builder.CreateMemCpy(
                theirs, llvm::MaybeAlign(1), ours, llvm::MaybeAlign(1),
                llvm::ConstantInt::get(size_ty, size)
            );
            symbolic_globals.push_back(name);
        }
        for (auto *table : harness.value_tables) {
            symbolic(
                table, dl.getTypeAllocSize(table->getValueType()).getFixedValue(),
                table->getName()
            );
        }

        // Arguments.
        struct Buffer
        {
            llvm::Value *live;
            llvm::Value *initial;
            uint64_t size;
        };
        std::vector< Buffer > buffers;
        std::vector< llvm::Value * > arguments;
        llvm::json::Array argument_rows;
        for (unsigned index = 0; index < signature->getNumParams(); ++index) {
            auto *type      = signature->getParamType(index);
            auto declared   = options.arg_buffers.find(index);
            const auto name = "arg" + std::to_string(index);
            if (type->isPointerTy() || declared != options.arg_buffers.end()) {
                const uint64_t size = declared != options.arg_buffers.end()
                    ? declared->second
                    : options.pointer_size;
                auto *bytes   = llvm::ConstantInt::get(size_ty, size);
                auto *live    = builder.CreateCall(allocate, { bytes }, name);
                auto *initial = builder.CreateCall(allocate, { bytes }, name + ".initial");
                symbolic(live, size, name);
                builder.CreateMemCpy(
                    initial, llvm::MaybeAlign(1), live, llvm::MaybeAlign(1), bytes
                );
                buffers.push_back({ live, initial, size });
                arguments.push_back(
                    type->isPointerTy() ? static_cast< llvm::Value * >(live)
                                        : builder.CreatePtrToInt(live, type)
                );
                argument_rows.push_back(llvm::json::Object{
                    { "index", index }, { "kind", "buffer" }, { "size", size } });
            } else if (type->isIntegerTy()) {
                auto *slot = builder.CreateAlloca(type, nullptr, name);
                symbolic(slot, dl.getTypeAllocSize(type).getFixedValue(), name);
                llvm::Value *value = builder.CreateLoad(type, slot);
                llvm::json::Object row{ { "index", index },
                                        { "kind", "integer" },
                                        { "size", dl.getTypeAllocSize(type).getFixedValue() } };
                auto range = options.arg_ranges.find(index);
                if (range != options.arg_ranges.end()) {
                    // low + (value mod span): every value of the range, and
                    // no other, whatever the symbolic value is.
                    const auto [low, high] = range->second;
                    const uint64_t span =
                        static_cast< uint64_t >(high) - static_cast< uint64_t >(low) + 1;
                    if (span != 0) {
                        value = builder.CreateURem(value, llvm::ConstantInt::get(type, span));
                    }
                    value = builder.CreateAdd(
                        value, llvm::ConstantInt::get(type, static_cast< uint64_t >(low), true),
                        name + ".bounded"
                    );
                    row["low"]  = low;
                    row["high"] = high;
                }
                arguments.push_back(value);
                argument_rows.push_back(std::move(row));
            } else {
                LOG(ERROR) << "argument " << index << " of '" << target
                           << "' is neither an integer nor a pointer\n";
                return nullptr;
            }
        }

        auto *result_type = signature->getReturnType();
        if (!(result_type->isVoidTy() || result_type->isIntegerTy()
              || result_type->isPointerTy()))
        {
            LOG(ERROR) << "'" << target << "' returns neither an integer nor a pointer\n";
            return nullptr;
        }
        auto begin_side = [&](unsigned which) {
            builder.CreateStore(llvm::ConstantInt::get(i32, which), harness.phase);
            builder.CreateStore(llvm::ConstantInt::get(i32, 0), harness.rejected);
            for (auto *counter : harness.call_counters) {
                builder.CreateStore(llvm::ConstantInt::get(i32, 0), counter);
            }
        };
        // Whether the side that just returned `result` rejected the input.
        auto did_reject = [&](llvm::Value *result) -> llvm::Value * {
            llvm::Value *flag = builder.CreateICmpNE(
                builder.CreateLoad(i32, harness.rejected), llvm::ConstantInt::get(i32, 0)
            );
            if (result != nullptr && result->getType()->isPointerTy()) {
                // A reject value of a pointer is its address (0: null).
                result = builder.CreatePtrToInt(result, size_ty);
            }
            if (result != nullptr) {
                for (int64_t value : options.reject_returns) {
                    flag = builder.CreateOr(
                        flag,
                        builder.CreateICmpEQ(
                            result, llvm::ConstantInt::get(
                                result->getType(), static_cast< uint64_t >(value), true
                            )
                        )
                    );
                }
            }
            return flag;
        };
        auto branch_to_end = [&](llvm::Value *condition, unsigned code, const char *name) {
            auto *stop = llvm::BasicBlock::Create(context, name, main_fn);
            auto *next = llvm::BasicBlock::Create(context, "continue", main_fn);
            builder.CreateCondBr(condition, stop, next);
            builder.SetInsertPoint(stop);
            harness.endPath(builder, code);
            builder.SetInsertPoint(next);
        };

        // A runner sets this to run the original alone, to tell whether an
        // input the patched version crashed on is one the original handles.
        auto *only_flag = new llvm::GlobalVariable(
            module, i32, false, llvm::GlobalValue::ExternalLinkage,
            llvm::ConstantInt::get(i32, 0), "__patchestry_only_original"
        );
        auto *only_original = builder.CreateICmpNE(
            builder.CreateLoad(i32, only_flag), llvm::ConstantInt::get(i32, 0), "only.original"
        );
        llvm::Value *patched_slot = result_type->isVoidTy()
            ? nullptr
            : builder.CreateAlloca(result_type, nullptr, "patched.result");
        auto *run_patched  = llvm::BasicBlock::Create(context, "run.patched", main_fn);
        auto *run_original = llvm::BasicBlock::Create(context, "run.original", main_fn);
        builder.CreateCondBr(only_original, run_original, run_patched);

        builder.SetInsertPoint(run_patched);
        begin_side(kPhasePatched);
        llvm::Value *patched_result =
            builder.CreateCall(module.getFunction(target + ".patched"), arguments);
        if (patched_slot != nullptr) {
            builder.CreateStore(patched_result, patched_slot);
        }
        branch_to_end(
            did_reject(result_type->isVoidTy() ? nullptr : patched_result), kPatchedRejected,
            "patched.rejected"
        );
        for (const auto &buffer : buffers) {
            builder.CreateMemCpy(
                buffer.live, llvm::MaybeAlign(1), buffer.initial, llvm::MaybeAlign(1),
                llvm::ConstantInt::get(size_ty, buffer.size)
            );
        }
        builder.CreateBr(run_original);

        run_original->moveAfter(builder.GetInsertBlock());
        builder.SetInsertPoint(run_original);
        begin_side(kPhaseOriginal);
        llvm::Value *original_result =
            builder.CreateCall(module.getFunction(target + ".orig"), arguments);
        auto *original_rejected =
            did_reject(result_type->isVoidTy() ? nullptr : original_result);
        {
            auto *stop = llvm::BasicBlock::Create(context, "original.rejected", main_fn);
            auto *next = llvm::BasicBlock::Create(context, "continue", main_fn);
            builder.CreateCondBr(original_rejected, stop, next);
            builder.SetInsertPoint(stop);
            builder.CreateCall(
                harness.path_end,
                { builder.CreateSelect(
                    only_original, llvm::ConstantInt::get(i32, kOriginalAloneRejected),
                    llvm::ConstantInt::get(i32, kP1Violation)
                ) }
            );
            builder.CreateUnreachable();
            builder.SetInsertPoint(next);
        }
        branch_to_end(only_original, kOriginalAloneAccepted, "original.alone");
        if (patched_slot != nullptr) {
            branch_to_end(
                builder.CreateICmpNE(
                    original_result, builder.CreateLoad(result_type, patched_slot)
                ),
                kP3Violation, "p3.violation"
            );
        }
        harness.endPath(builder, kAccepted);

        llvm::json::Array handlers;
        for (const auto &handler : options.reject_handlers) {
            handlers.push_back(handler);
        }
        llvm::json::Array returns;
        for (int64_t value : options.reject_returns) {
            returns.push_back(value);
        }
        info["target"]           = target;
        info["triple"]           = module.getTargetTriple().str();
        info["reject_handlers"]  = std::move(handlers);
        info["reject_returns"]   = std::move(returns);
        info["arguments"]        = std::move(argument_rows);
        info["pointer_size"]     = options.pointer_size;
        info["symbolic_globals"] = std::move(symbolic_globals);
        info["concrete_globals"] = std::move(concrete_globals);
        info["stubs"]            = std::move(harness.stubs);
        info["stub_values"]      = options.stub_values;
        info["compares_return"]  = !result_type->isVoidTy();
        info["layout_adopted"]   = layout_adopted;
        return original;
    }

    bool writeHarnessInfo(const llvm::json::Object &info, llvm::StringRef path) {
        std::error_code error;
        llvm::raw_fd_ostream out(path, error, llvm::sys::fs::OF_Text);
        if (error) {
            LOG(ERROR) << "cannot write '" << path.str() << "': " << error.message() << "\n";
            return false;
        }
        out << llvm::formatv("{0:2}", llvm::json::Value(llvm::json::Object(info))) << "\n";
        return true;
    }

} // namespace patchestry::klee_verifier
