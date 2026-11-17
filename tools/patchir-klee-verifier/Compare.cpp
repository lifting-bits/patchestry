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
//    0  both versions accepted the input and agree on everything compared
//    1  the patched version rejected the input
//   10  P1: the patched version accepted an input the original rejects
//   20  inconclusive: a version called an external function the harness
//       cannot model (one that returns a pointer or a non-integer and has no
//       model), or made more calls than the trace holds
//   30  the original, run alone, accepted the input
//   31  the original, run alone, rejected the input
//   64 + bits  both accepted and differ: 1 the return value (P3), 2 the
//       final memory (P2), 4 the external calls (P4)
//
// P2 compares, after both runs, each side's copy of every global that started
// symbolic, and the contents of every argument buffer.  P4 compares the calls
// to external functions, in order, with their arguments: a stub records the
// call (`__patchestry_trace_call`, `__patchestry_trace_arg`) before it
// returns.  An argument that is the address of a global is compared as
// (global, offset), since each side has its own copy; `__patchestry_pairs`
// lists the copies.  The address of constant data is compared by the bytes
// it points at (`__patchestry_constants`), so a string literal equals the
// same literal in the other module.  Calls of a pure callee are not recorded.  A declared
// substitution records the patched version's call of one function as a call
// of the function it replaces, with the leading arguments the two share.
//
// The harness runs the original alone when `__patchestry_only_original` is
// set.  A runner on concrete inputs uses that after the patched version
// crashed: under symbolic execution the flag stays 0.
//
// A version rejects an input when it calls a declared reject handler or
// returns a declared reject value.  An external function is replaced by a
// stub that returns the same value at the same call index on both sides, or,
// when the runner has a model of it, by a call of that model.  A pure callee
// without a model returns a value chosen by its integer arguments.

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
        constexpr unsigned kViolation       = 64;
        constexpr unsigned kReturnDiffers   = 1;
        constexpr unsigned kMemoryDiffers   = 2;
        constexpr unsigned kCallsDiffer     = 4;
        constexpr unsigned kTraceOverflow   = 2;
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
            return name.starts_with("llvm.") || name.starts_with("klee_")
                || name.starts_with("__patchestry_");
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
            llvm::IntegerType *size_ty;
            llvm::GlobalVariable *phase;
            llvm::GlobalVariable *rejected;
            llvm::FunctionCallee path_end;
            llvm::FunctionCallee trace_call;
            llvm::FunctionCallee trace_arg;
            std::map< std::string, unsigned > callee_ids;
            std::map< std::string, std::pair< llvm::GlobalVariable *, llvm::GlobalVariable * > >
                tables;
            std::vector< llvm::GlobalVariable * > call_counters;
            std::vector< llvm::GlobalVariable * > value_tables;
            llvm::json::Array stubs;

            Harness(llvm::Module &M, const CompareOptions &opts)
                : module(M)
                , options(opts)
                , context(M.getContext())
                , i32(llvm::Type::getInt32Ty(M.getContext()))
                , size_ty(M.getDataLayout().getIntPtrType(M.getContext()))
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
                auto *nothing = llvm::Type::getVoidTy(context);
                trace_call    = M.getOrInsertFunction(
                    "__patchestry_trace_call", llvm::FunctionType::get(nothing, { i32, i32 }, false)
                );
                trace_arg = M.getOrInsertFunction(
                    "__patchestry_trace_arg", llvm::FunctionType::get(nothing, { size_ty }, false)
                );
            }

            unsigned calleeId(const std::string &name) {
                auto found = callee_ids.find(name);
                if (found != callee_ids.end()) {
                    return found->second;
                }
                const auto id     = static_cast< unsigned >(callee_ids.size()) + 1;
                callee_ids[name] = id;
                return id;
            }

            // Record a call of callee `id` with the first `limit` arguments
            // of `function`.  A value wider than a pointer takes two entries.
            void logCall(
                llvm::IRBuilder<> &builder, llvm::Function &function, unsigned id, unsigned limit
            ) {
                std::vector< llvm::Value * > values;
                unsigned index = 0;
                for (auto &argument : function.args()) {
                    if (index++ >= limit) {
                        break;
                    }
                    auto *type = argument.getType();
                    if (type->isPointerTy()) {
                        values.push_back(builder.CreatePtrToInt(&argument, size_ty));
                    } else if (type->isIntegerTy()
                               && type->getIntegerBitWidth() <= size_ty->getBitWidth()) {
                        values.push_back(builder.CreateZExtOrTrunc(&argument, size_ty));
                    } else if (type->isIntegerTy()) {
                        values.push_back(builder.CreateTrunc(&argument, size_ty));
                        values.push_back(builder.CreateTrunc(
                            builder.CreateLShr(&argument, size_ty->getBitWidth()), size_ty
                        ));
                    } else {
                        values.push_back(llvm::ConstantInt::get(size_ty, 0));
                    }
                }
                builder.CreateCall(
                    trace_call,
                    { llvm::ConstantInt::get(i32, id),
                      llvm::ConstantInt::get(i32, static_cast< unsigned >(values.size())) }
                );
                for (auto *value : values) {
                    builder.CreateCall(trace_arg, { value });
                }
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

            // The table of values a stub returns, and its call counter.  Null
            // when `key` already has a table of another type.
            std::pair< llvm::GlobalVariable *, llvm::GlobalVariable * > *
            tableFor(const std::string &key, llvm::Type *result) {
                auto *table_type = llvm::ArrayType::get(result, options.stub_values);
                auto found       = tables.find(key);
                if (found != tables.end()) {
                    return found->second.first->getValueType() == table_type ? &found->second
                                                                             : nullptr;
                }
                auto *table   = newGlobal(module, table_type, "__patchestry_values." + key);
                auto *counter = newGlobal(module, i32, "__patchestry_calls." + key);
                value_tables.push_back(table);
                call_counters.push_back(counter);
                return &(tables[key] = { table, counter });
            }

            // The body of an external function: record the call, then return
            // what the model returns, or the next value of a symbolic table
            // both sides share.  False when the function cannot be stubbed as
            // declared.
            bool stubExternal(
                llvm::Function &function, llvm::json::Object &row, bool pure,
                const Substitution *substitution
            ) {
                const std::string name = function.getName().str();
                llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", &function));
                if (substitution != nullptr) {
                    // In the patched version this call stands for a call of
                    // the function it replaces.
                    auto *as_patched  = llvm::BasicBlock::Create(context, "as.patched", &function);
                    auto *as_original = llvm::BasicBlock::Create(context, "as.original", &function);
                    auto *body        = llvm::BasicBlock::Create(context, "body", &function);
                    builder.CreateCondBr(
                        builder.CreateICmpEQ(
                            builder.CreateLoad(i32, phase),
                            llvm::ConstantInt::get(i32, kPhasePatched)
                        ),
                        as_patched, as_original
                    );
                    builder.SetInsertPoint(as_patched);
                    logCall(
                        builder, function, calleeId(substitution->original),
                        substitution->compare_args
                    );
                    builder.CreateBr(body);
                    builder.SetInsertPoint(as_original);
                    logCall(builder, function, calleeId(name), ~0U);
                    builder.CreateBr(body);
                    builder.SetInsertPoint(body);
                    row["substitutes"] = substitution->original;
                } else if (!pure) {
                    logCall(builder, function, calleeId(name), ~0U);
                }
                if (pure) {
                    row["pure"] = true;
                }
                if (function.isVarArg()) {
                    row["variadic"] = true;
                }

                auto *result = function.getReturnType();
                if (options.models.count(name) != 0U && !function.isVarArg()) {
                    row["kind"] = "model";
                    auto model  = module.getOrInsertFunction(
                        "__patchestry_model_" + name, function.getFunctionType()
                    );
                    std::vector< llvm::Value * > arguments;
                    for (auto &argument : function.args()) {
                        arguments.push_back(&argument);
                    }
                    auto *value = builder.CreateCall(model, arguments);
                    if (result->isVoidTy()) {
                        builder.CreateRetVoid();
                    } else {
                        builder.CreateRet(value);
                    }
                    return true;
                }
                if (result->isVoidTy()) {
                    row["kind"] = "void";
                    builder.CreateRetVoid();
                    return true;
                }
                if (!result->isIntegerTy()) {
                    // A pointer or a float from nowhere cannot be made up; a
                    // path that needs one is inconclusive.
                    row["kind"] = "unmodelled";
                    endPath(builder, kUnmodelled);
                    return true;
                }
                row["kind"] = "values";
                auto *table =
                    tableFor(substitution != nullptr ? substitution->original : name, result);
                if (table == nullptr) {
                    LOG(ERROR) << "'" << name
                               << "' and the function it substitutes return different types\n";
                    return false;
                }
                llvm::Value *index = nullptr;
                if (pure) {
                    // The same arguments give the same value, at any call index.
                    index = llvm::ConstantInt::get(i32, 0);
                    for (auto &argument : function.args()) {
                        if (argument.getType()->isIntegerTy()) {
                            index = builder.CreateAdd(
                                index, builder.CreateZExtOrTrunc(&argument, i32)
                            );
                        }
                    }
                    index = builder.CreateURem(
                        index, llvm::ConstantInt::get(i32, options.stub_values)
                    );
                } else {
                    auto *count = builder.CreateLoad(i32, table->second);
                    builder.CreateStore(
                        builder.CreateAdd(count, llvm::ConstantInt::get(i32, 1)), table->second
                    );
                    auto *last = llvm::ConstantInt::get(i32, options.stub_values - 1);
                    index = builder.CreateSelect(builder.CreateICmpULT(count, last), count, last);
                }
                auto *slot = builder.CreateInBoundsGEP(
                    table->first->getValueType(), table->first,
                    { llvm::ConstantInt::get(i32, 0), index }
                );
                builder.CreateRet(builder.CreateLoad(result, slot));
                return true;
            }

            // False when a declaration names a function neither version
            // calls, or cannot be honoured.
            bool stubExternals() {
                std::set< std::string > handlers(
                    options.reject_handlers.begin(), options.reject_handlers.end()
                );
                std::set< std::string > pures(
                    options.pure_callees.begin(), options.pure_callees.end()
                );
                std::map< std::string, const Substitution * > swaps;
                for (const auto &swap : options.substitutions) {
                    if (pures.count(swap.original) != 0U || pures.count(swap.patched) != 0U
                        || swap.original == swap.patched)
                    {
                        LOG(ERROR) << "substitution '" << swap.original << "' to '" << swap.patched
                                   << "' names a pure callee, or one function twice\n";
                        return false;
                    }
                    swaps[swap.patched] = &swap;
                }
                std::vector< llvm::Function * > externals;
                for (auto &function : module) {
                    if (function.isDeclaration() && !function.use_empty()
                        && !isRuntimeName(function.getName()))
                    {
                        externals.push_back(&function);
                    }
                }
                std::set< std::string > seen;
                for (auto *function : externals) {
                    const std::string name = function->getName().str();
                    seen.insert(name);
                    llvm::json::Object row{ { "name", name } };
                    if (handlers.count(name) != 0U) {
                        row["kind"] = function->doesNotReturn() ? "reject-noreturn" : "reject";
                        stubRejectHandler(*function);
                    } else {
                        auto swap = swaps.find(name);
                        if (!stubExternal(
                                *function, row, pures.count(name) != 0U,
                                swap == swaps.end() ? nullptr : swap->second
                            ))
                        {
                            return false;
                        }
                    }
                    function->setLinkage(llvm::GlobalValue::InternalLinkage);
                    stubs.push_back(std::move(row));
                }
                for (const auto &handler : options.reject_handlers) {
                    if (seen.count(handler) == 0U) {
                        LOG(ERROR) << "reject handler '" << handler
                                   << "' is not called by either version\n";
                        return false;
                    }
                }
                for (const auto &pure : options.pure_callees) {
                    if (seen.count(pure) == 0U) {
                        LOG(ERROR) << "pure callee '" << pure
                                   << "' is not called by either version\n";
                        return false;
                    }
                }
                for (const auto &swap : options.substitutions) {
                    if (seen.count(swap.original) == 0U || seen.count(swap.patched) == 0U) {
                        LOG(ERROR) << "substitution '" << swap.original << "' to '" << swap.patched
                                   << "' names a function neither version calls\n";
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
        auto *size_ty  = harness.size_ty;
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
        struct Region
        {
            llvm::Value *original;
            llvm::Value *patched;
            uint64_t size;
        };
        std::vector< Region > regions;
        llvm::json::Array region_names;
        auto *pointer  = llvm::PointerType::getUnqual(context);
        auto *pair_ty  = llvm::StructType::get(context, { pointer, pointer, size_ty });
        std::vector< llvm::Constant * > pairs;
        llvm::json::Array symbolic_globals;
        llvm::json::Array concrete_globals;
        for (const auto &name : global_names) {
            auto *ours   = module.getNamedGlobal(name + ".orig");
            auto *theirs = module.getNamedGlobal(name + ".patched");
            if (ours != nullptr && theirs != nullptr && ours->getValueType()->isSized()
                && !ours->isConstant() && !theirs->isConstant())
            {
                // An address inside either copy names the same place.
                pairs.push_back(llvm::ConstantStruct::get(
                    pair_ty,
                    { ours, theirs,
                      llvm::ConstantInt::get(
                          size_ty, dl.getTypeAllocSize(ours->getValueType()).getFixedValue()
                      ) }
                ));
            }
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
            regions.push_back({ ours, theirs, size });
            region_names.push_back(name);
        }
        {
            const auto count = static_cast< unsigned >(pairs.size());
            if (pairs.empty()) {
                pairs.push_back(llvm::Constant::getNullValue(pair_ty));
            }
            auto *table_ty = llvm::ArrayType::get(pair_ty, pairs.size());
            new llvm::GlobalVariable(
                module, table_ty, true, llvm::GlobalValue::ExternalLinkage,
                llvm::ConstantArray::get(table_ty, pairs), "__patchestry_pairs"
            );
            new llvm::GlobalVariable(
                module, i32, true, llvm::GlobalValue::ExternalLinkage,
                llvm::ConstantInt::get(i32, count), "__patchestry_pair_count"
            );

            // Constant data of either side, for comparison by content.
            auto *constant_ty = llvm::StructType::get(context, { pointer, size_ty });
            std::vector< llvm::Constant * > constants;
            for (auto &global : module.globals()) {
                if (global.isConstant() && !global.isDeclaration()
                    && global.getValueType()->isSized()
                    && !global.getName().starts_with("llvm.")
                    && !global.getName().starts_with("__patchestry_"))
                {
                    constants.push_back(llvm::ConstantStruct::get(
                        constant_ty,
                        { &global,
                          llvm::ConstantInt::get(
                              size_ty, dl.getTypeAllocSize(global.getValueType()).getFixedValue()
                          ) }
                    ));
                }
            }
            const auto constant_count = static_cast< unsigned >(constants.size());
            if (constants.empty()) {
                constants.push_back(llvm::Constant::getNullValue(constant_ty));
            }
            auto *constants_ty = llvm::ArrayType::get(constant_ty, constants.size());
            new llvm::GlobalVariable(
                module, constants_ty, true, llvm::GlobalValue::ExternalLinkage,
                llvm::ConstantArray::get(constants_ty, constants), "__patchestry_constants"
            );
            new llvm::GlobalVariable(
                module, i32, true, llvm::GlobalValue::ExternalLinkage,
                llvm::ConstantInt::get(i32, constant_count), "__patchestry_constant_count"
            );
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
            llvm::Value *patched; // contents after the patched version ran
            uint64_t size;
        };
        std::vector< Buffer > buffers;
        std::vector< llvm::Value * > arguments;
        llvm::json::Array argument_rows;
        auto buffer_size = [&](unsigned index) -> uint64_t {
            auto declared = options.arg_buffers.find(index);
            if (declared != options.arg_buffers.end()) {
                return declared->second;
            }
            return signature->getParamType(index)->isPointerTy() ? options.pointer_size : 0;
        };
        // The harness's own copies are allocated before any buffer a version
        // is given.  A version that reads past the end of its buffer then
        // finds the same bytes in both runs, not a copy that changed between
        // them.
        std::map< unsigned, llvm::Value * > copies_after;
        std::map< unsigned, llvm::Value * > copies_initial;
        for (unsigned index = 0; index < signature->getNumParams(); ++index) {
            if (const uint64_t size = buffer_size(index)) {
                const auto name     = "arg" + std::to_string(index);
                auto *bytes         = llvm::ConstantInt::get(size_ty, size);
                copies_after[index] = builder.CreateCall(allocate, { bytes }, name + ".patched");
                copies_initial[index] =
                    builder.CreateCall(allocate, { bytes }, name + ".initial");
            }
        }
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
                auto *initial = copies_initial[index];
                symbolic(live, size, name);
                builder.CreateMemCpy(
                    initial, llvm::MaybeAlign(1), live, llvm::MaybeAlign(1), bytes
                );
                auto *after = copies_after[index];
                buffers.push_back({ live, initial, after, size });
                regions.push_back({ live, after, size });
                region_names.push_back(name);
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
        auto *nothing    = llvm::Type::getVoidTy(context);
        auto side_begin  = module.getOrInsertFunction(
            "__patchestry_side_begin", llvm::FunctionType::get(nothing, false)
        );
        auto trace_differs = module.getOrInsertFunction(
            "__patchestry_trace_differs", llvm::FunctionType::get(i32, false)
        );
        auto memory_differs = module.getOrInsertFunction(
            "__patchestry_memory_differs",
            llvm::FunctionType::get(i32, { pointer, pointer, size_ty, i32 }, false)
        );
        auto begin_side = [&](unsigned which) {
            builder.CreateStore(llvm::ConstantInt::get(i32, which), harness.phase);
            // The runner gives both sides the same heap.
            builder.CreateCall(side_begin);
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
            auto *bytes = llvm::ConstantInt::get(size_ty, buffer.size);
            builder.CreateMemCpy(
                buffer.patched, llvm::MaybeAlign(1), buffer.live, llvm::MaybeAlign(1), bytes
            );
            builder.CreateMemCpy(
                buffer.live, llvm::MaybeAlign(1), buffer.initial, llvm::MaybeAlign(1), bytes
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
        // Both accepted: what differs, as bits.
        auto *zero        = llvm::ConstantInt::get(i32, 0);
        llvm::Value *bits = zero;
        if (patched_slot != nullptr) {
            bits = builder.CreateSelect(
                builder.CreateICmpNE(
                    original_result, builder.CreateLoad(result_type, patched_slot)
                ),
                llvm::ConstantInt::get(i32, kReturnDiffers), zero, "return.differs"
            );
        }
        auto *traced = builder.CreateCall(trace_differs, {}, "trace");
        branch_to_end(
            builder.CreateICmpEQ(traced, llvm::ConstantInt::get(i32, kTraceOverflow)),
            kUnmodelled, "trace.overflow"
        );
        bits = builder.CreateOr(
            bits, builder.CreateSelect(
                      builder.CreateICmpNE(traced, zero),
                      llvm::ConstantInt::get(i32, kCallsDiffer), zero, "calls.differ"
                  )
        );
        unsigned region_index = 0;
        for (const auto &region : regions) {
            auto *differs = builder.CreateCall(
                memory_differs,
                { region.original, region.patched, llvm::ConstantInt::get(size_ty, region.size),
                  llvm::ConstantInt::get(i32, region_index++) }
            );
            bits = builder.CreateOr(
                bits, builder.CreateSelect(
                          builder.CreateICmpNE(differs, zero),
                          llvm::ConstantInt::get(i32, kMemoryDiffers), zero
                      )
            );
        }
        {
            auto *stop = llvm::BasicBlock::Create(context, "violation", main_fn);
            auto *next = llvm::BasicBlock::Create(context, "agree", main_fn);
            builder.CreateCondBr(builder.CreateICmpNE(bits, zero), stop, next);
            builder.SetInsertPoint(stop);
            builder.CreateCall(
                harness.path_end,
                { builder.CreateOr(bits, llvm::ConstantInt::get(i32, kViolation)) }
            );
            builder.CreateUnreachable();
            builder.SetInsertPoint(next);
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
        llvm::json::Array pure_rows;
        for (const auto &pure : options.pure_callees) {
            pure_rows.push_back(pure);
        }
        llvm::json::Array swap_rows;
        for (const auto &swap : options.substitutions) {
            swap_rows.push_back(llvm::json::Object{ { "original", swap.original },
                                                    { "patched", swap.patched },
                                                    { "compare_args", swap.compare_args } });
        }
        info["memory_regions"]   = std::move(region_names);
        info["pure_callees"]     = std::move(pure_rows);
        info["substitutions"]    = std::move(swap_rows);
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
