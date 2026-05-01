/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "HarnessGen.hpp"

#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>

#include <patchestry/Util/Log.hpp>

#include "GlobalsInit.hpp"
#include "Options.hpp"
#include "PredicateParser.hpp"
#include "Runtime.hpp"

namespace patchestry::klee_verifier {

    // Get or declare klee_make_symbolic: void klee_make_symbolic(void*, size_t, const char*)
    extern llvm::FunctionCallee getKleeMakeSymbolic(llvm::Module &M);

    // Get or declare klee_assume: void klee_assume(uintptr_t)
    // KLEE's runtime uses uintptr_t, so match the target's pointer width.
    extern llvm::FunctionCallee getKleeAssume(llvm::Module &M);

    // Get or declare klee_abort: void klee_abort(void)
    // KLEE's klee_assert is a macro (not a function), so we use
    // if (!cond) klee_abort() to implement postcondition assertions.
    extern llvm::FunctionCallee getKleeAbort(llvm::Module &M);

    // Get or declare malloc: i8* malloc(size_t). KLEE intercepts this at
    // runtime and returns a tracked heap allocation that survives the
    // caller's stack frame.
    extern llvm::FunctionCallee getMalloc(llvm::Module &M);

    namespace {

        // Collect every CallBase in `M` carrying a `!static_contract`
        // MDTuple, alongside its serialised body. The schema is:
        //   operand(0) -> MDString: tag (informational, not used to scope)
        //   operand(1) -> MDString: serialised contract body
        //
        // Earlier revisions filtered by operand(0) against the harness's
        // target function name, but contracts are an *operation-level*
        // property: they're stamped on the matched call op in CIR (see
        // ContractOperationImpl::applyContractBefore/After), so the right
        // place to instrument them is at the call site, regardless of which
        // function contains it. Multiple contracts at multiple call sites
        // are independent and each get their own assume/assert pair.
        struct ContractSite {
            llvm::CallBase *call;
            std::string     body;
        };
        std::vector< ContractSite > collectContractSites(llvm::Module &M) {
            std::vector< ContractSite > sites;
            for (auto &F : M) {
                for (auto &BB : F) {
                    for (auto &I : BB) {
                        auto *cb = llvm::dyn_cast< llvm::CallBase >(&I);
                        if (!cb)
                            continue;
                        auto *contract_md = cb->getMetadata("static_contract");
                        if (!contract_md)
                            continue;

                        auto *tuple = llvm::dyn_cast< llvm::MDTuple >(contract_md);
                        if (!tuple || tuple->getNumOperands() < 2)
                            continue;
                        auto *body = llvm::dyn_cast< llvm::MDString >(tuple->getOperand(1));
                        if (!body)
                            continue;
                        sites.push_back({ cb, body->getString().str() });
                    }
                }
            }
            return sites;
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
        llvm::Value *toI64(llvm::IRBuilder<> &B, llvm::Value *V, bool is_signed) {
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
        void emitKleePredicate(
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
                        // Two's-complement cast: a negative int64_t (e.g. -1
                        // for the (void*)-1 error sentinel) wraps to the
                        // correct unsigned bit pattern for the comparison.
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
                        // See PK_RelNeqArgConst: two's-complement wrap is
                        // intentional for negative sentinel values.
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
                auto *insert_bb = B.GetInsertBlock();
                assert(insert_bb && "IRBuilder must have an active insertion point");
                auto *parent_fn = insert_bb->getParent();
                auto *abort_bb  = llvm::BasicBlock::Create(Ctx, "assert.fail", parent_fn);
                auto *cont_bb   = llvm::BasicBlock::Create(Ctx, "assert.cont", parent_fn);
                B.CreateCondBr(cond, cont_bb, abort_bb);

                B.SetInsertPoint(abort_bb);
                B.CreateCall(getKleeAbort(M), {});
                B.CreateUnreachable();

                B.SetInsertPoint(cont_bb);
            }
        }

    } // namespace

    unsigned rewriteAbortCalls(llvm::Module &M) {
        auto klee_abort = getKleeAbort(M);
        unsigned count = 0;

        // Explicit list of names that semantically represent "property
        // failure" — calling any of them under KLEE should surface as an
        // error path, not as a successful termination. Libc abort family
        // and patchestry's assert intrinsic.
        //
        // We do NOT match on the generic `noreturn` attribute: longjmp,
        // __cxa_throw, __cxa_rethrow, pthread_exit, thrd_exit, and
        // siglongjmp are all noreturn but represent legitimate non-local
        // control flow (stack unwinding, thread-scoped exit). Rewriting
        // them to klee_abort would convert those flows into false bug
        // reports. Add new abort-like functions to this list explicitly.
        static const llvm::StringRef kAbortLikeNames[] = {
            "abort", "_abort", "__abort",
            "exit", "_exit", "_Exit",
            "__assert_fail", "__assert_rtn",
            "__patchestry_assert_fail",
        };

        // Names we must never redirect — KLEE intercepts these natively,
        // or they are our own synthesized hooks.
        auto isExcluded = [](llvm::StringRef name) {
            return name.starts_with("klee_")
                || name.starts_with("llvm.")
                || name.starts_with("__klee_");
        };

        // Give each abort-like declaration a body that calls klee_abort.
        // All existing call sites (direct and indirect through function
        // pointers) route through the new body automatically — no per-
        // call-site IR surgery needed.
        for (auto &F : M) {
            if (!F.isDeclaration())
                continue;
            if (isExcluded(F.getName()))
                continue;

            bool is_abort_like = false;
            for (auto name : kAbortLikeNames) {
                if (F.getName() == name) {
                    is_abort_like = true;
                    break;
                }
            }
            if (!is_abort_like)
                continue;

            auto &Ctx = M.getContext();
            auto *BB = llvm::BasicBlock::Create(Ctx, "entry", &F);
            llvm::IRBuilder<> B(BB);
            B.CreateCall(klee_abort, {});
            B.CreateUnreachable();

            ++count;
            if (verbose) {
                llvm::outs() << "  Redirected " << F.getName()
                             << " to klee_abort\n";
            }
        }

        return count;
    }

    // TODO (kumarak): target-specific intrinsics are silently turned into
    // symbolic-returning stubs here, which *loses the operation's semantics*.
    // The most important case today is ARM (retargeting from ARM32 to x86_64
    // is common for KLEE): `llvm.arm.qadd`, `llvm.arm.qsub`, `llvm.arm.ssat`,
    // `llvm.arm.usat`, the NEON saturating family, MRS/MSR co-processor reads,
    // etc. all arrive here as unresolved decls and get a `klee_make_symbolic`
    // body — the harness then "verifies" against an unconstrained return
    // value rather than the real saturation / clamp / register semantics.
    //
    // Planned fix (see design note in generateHarness): ship a per-arch
    // model library under `lib/patchestry/klee/models/` (e.g. `arm32.c`)
    // that defines correct bodies for the common intrinsics, link it via
    // the existing `--model-library` plumbing *before* this pass runs, and
    // only fall through to symbolic stubbing for names that neither the
    // model library nor KLEE's own runtime provides. Since `llvm.*` symbols
    // are reserved, the pre-pass needs to rewrite calls to known ARM
    // intrinsics to calls to `arm_model_*` wrapper symbols first — then
    // the linker binds them to the model library normally. Until that
    // lands, authors should be aware that contracts exercising ARM-specific
    // behavior (saturation, strict-align semantics, co-processor state)
    // are under-constrained and may silently pass.
    unsigned stubExternalFunctions(llvm::Module &M, llvm::Function *target_fn) {
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

    bool instrumentStaticContracts(llvm::Module &M) {
        auto sites = collectContractSites(M);
        if (sites.empty()) {
            if (verbose)
                llvm::outs() << "No static contracts found\n";
            return true;
        }

        unsigned dropped_total = 0;
        unsigned instrumented = 0;
        for (auto &site : sites) {
            unsigned dropped = 0;
            auto preds = parseStaticContractText(site.body, dropped);
            dropped_total += dropped;
            if (preds.empty())
                continue;

            // Resolve a predicate's `Arg(N)` reference to the i-th operand of
            // the contracted call site. `ReturnValue` resolves to the call's
            // SSA result (skipped on void calls — caller checks).
            auto resolve_arg = [&](const ParsedPredicate &P) -> llvm::Value * {
                if (P.target.substr(0, 3) != "Arg")
                    return nullptr;
                if (P.arg_index >= site.call->arg_size())
                    return nullptr;
                return site.call->getArgOperand(P.arg_index);
            };

            // Preconditions: emit klee_assume just before the call. No bb
            // surgery needed — klee_assume is a non-terminator call.
            {
                llvm::IRBuilder<> B(site.call);
                for (auto &P : preds) {
                    if (!P.is_precondition)
                        continue;
                    // ReturnValue in a precondition is meaningless (the call
                    // hasn't run yet). emitKleePredicate skips when V is null.
                    llvm::Value *arg_val = resolve_arg(P);
                    emitKleePredicate(B, M, P, arg_val, /*ret_val=*/nullptr);
                }
            }

            // Postconditions: emit klee_assert just after the call. Each
            // assertion expands to `if (!cond) klee_abort()`, which needs a
            // terminator — split the bb after the call so the assertion
            // chain owns its own terminators, then close with a branch to
            // the post-call tail.
            bool has_post = false;
            for (auto &P : preds) {
                if (!P.is_precondition) {
                    has_post = true;
                    break;
                }
            }
            if (has_post) {
                auto *parent_bb = site.call->getParent();
                auto *post_bb   = parent_bb->splitBasicBlock(
                    site.call->getNextNode(), "after.contract"
                );
                // splitBasicBlock added an unconditional br; replace it with
                // our own terminator chain.
                parent_bb->getTerminator()->eraseFromParent();

                llvm::IRBuilder<> B(parent_bb);
                llvm::Value *ret_val = site.call->getType()->isVoidTy()
                    ? nullptr
                    : static_cast< llvm::Value * >(site.call);
                for (auto &P : preds) {
                    if (P.is_precondition)
                        continue;
                    llvm::Value *arg_val = resolve_arg(P);
                    emitKleePredicate(B, M, P, arg_val, ret_val);
                }
                B.CreateBr(post_bb);
            }

            ++instrumented;
        }

        if (verbose) {
            llvm::outs() << "Instrumented " << instrumented << " of " << sites.size()
                         << " contract site(s)\n";
        }

        if (dropped_total > 0) {
            if (strict_contracts) {
                LOG(ERROR)
                    << dropped_total
                    << " predicate(s) failed to parse — refusing to emit an "
                       "under-constrained harness (pass --strict-contracts=false to override)\n";
                return false;
            }
            LOG(WARNING) << dropped_total
                         << " predicate(s) dropped during parsing — harness may be "
                            "under-constrained\n";
        }
        return true;
    }

    bool generateHarness(llvm::Module &M, llvm::Function *target_fn) {
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
        //
        // HARD-FAULT RISK: this codepath assumes every remaining use of the
        // original `main` is on a dead startup path that the target function
        // will never execute. That assumption can be wrong — a decompiled
        // interrupt vector, test driver, or re-entrant startup stub can call
        // `main` from inside a chain the target *does* reach (directly, or
        // transitively through an indirect dispatch we cannot see). When
        // that happens, the call site we RAUW'd to a poison callee executes
        // under KLEE as "call poison", which is undefined behavior: KLEE
        // reports a hard fault (typically an out-of-bound pointer or an
        // assertion inside the executor), and *the verdict for that run is
        // unsound* — we terminated on our own injected poison rather than
        // on a real property violation by the target.
        //
        // Mitigations to consider if this bites in practice:
        //   * rename `main` to `__klee_orig_main` unconditionally (preserves
        //     callers; old body becomes harmless dead code if nothing else
        //     calls it — the same treatment we give the target-is-main
        //     branch above);
        //   * reachability-gate the erase (walk direct calls from target_fn;
        //     rename if main is reachable, erase if not — limited by not
        //     tracking indirect dispatch).
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

        // 1. Symbolically initialize module-wide globals via the per-type
        //    init machinery (see GlobalsInit.cpp for the four-stage design).
        //    Emits a single call to the dispatcher at the top of main(),
        //    before argument symbolization so the target sees initialized
        //    globals when it runs.
        GlobalsInitStats globals_stats;
        llvm::Function *dispatcher = installGlobalsInit(M, globals_stats);
        B.CreateCall(dispatcher, {});

        if (verbose) {
            llvm::outs() << "Globals init: " << globals_stats.collected
                         << " global(s) ("
                         << globals_stats.materialized << " materialized from external), "
                         << globals_stats.type_init_fns
                         << " per-type init function(s), "
                         << globals_stats.pointer_fields
                         << " pointer field(s) inferred\n";
        }

        // 2. Create symbolic arguments for the target function
        llvm::FunctionType *target_ft = target_fn->getFunctionType();
        std::vector< llvm::Value * > args;

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
            } else if (param_size == 0) {
                // Zero-sized non-pointer type (e.g. empty aggregate passed
                // by value under some ABIs). klee_make_symbolic(ptr, 0,
                // name) is a no-op and the subsequent load would read
                // undef anyway; synthesize undef directly and skip the
                // wasted call.
                llvm::Value *Val = llvm::UndefValue::get(param_ty);
                args.push_back(Val);
            } else {
                // For non-pointer args: alloca, make symbolic, load
                auto *Alloca = B.CreateAlloca(param_ty);

                llvm::Value *cast_ptr = B.CreateBitCast(Alloca, ptrTy);
                llvm::Value *size_val = llvm::ConstantInt::get(sizeTy, param_size);
                llvm::Value *name_str = B.CreateGlobalString(arg_name);
                B.CreateCall(make_sym, { cast_ptr, size_val, name_str });

                llvm::Value *Val = B.CreateLoad(param_ty, Alloca);
                args.push_back(Val);
            }

            if (verbose) {
                llvm::outs() << "  Symbolic arg" << i << ": ";
                param_ty->print(llvm::outs());
                llvm::outs() << "\n";
            }
        }

        // 3. Call the target function. Contract predicates have already been
        //    instrumented around their respective call sites by
        //    instrumentStaticContracts(); the harness no longer needs to
        //    reason about contracts at this level.
        if (target_ft->getReturnType()->isVoidTy()) {
            B.CreateCall(target_fn, args);
        } else {
            B.CreateCall(target_fn, args);
        }

        // 4. Return 0
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
    bool writeModuleToFile(llvm::Module &module, llvm::StringRef out) {
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

    namespace {

        // Scan the retargeted module for ARM32 ABI-specific artifacts that
        // would silently invalidate contracts under x86_64 execution. Two
        // classes of finding are reported as LOG(ERROR):
        //
        //   1. ARM-specific LLVM intrinsic declarations (`llvm.arm.*`) —
        //      these have no x86_64 implementation, get symbolic-stubbed by
        //      `stubExternalFunctions`, and silently drop saturation /
        //      co-processor / strict-align semantics. See the TODO above
        //      stubExternalFunctions for the planned model-library fix.
        //
        //   2. Register-alias parameter and global names produced by the
        //      Ghidra P-Code importer (`extraout_rN`, `unaff_*`, `in_rN`,
        //      `in_lr`, `in_sp`). PR #193 ("sanitize extraout_rN / unaff_*
        //      / in_* register-alias artifacts") removes most of these at
        //      decomp time, but anything slipping through means a contract
        //      targeting that parameter is asserting over an ARM AAPCS
        //      register that has no x86_64 analogue — the predicate fires
        //      against an unconstrained symbolic value.
        void scanNarrowPointerCasts(const llvm::Module &M) {
            unsigned narrow_inttoptr = 0;
            unsigned narrow_ptrtoint = 0;
            for (const auto &F : M) {
                for (const auto &BB : F) {
                    for (const auto &I : BB) {
                        if (auto *ITP = llvm::dyn_cast< llvm::IntToPtrInst >(&I)) {
                            unsigned src_bits = ITP->getOperand(0)
                                                    ->getType()
                                                    ->getIntegerBitWidth();
                            if (src_bits < 64)
                                ++narrow_inttoptr;
                        } else if (auto *PTI = llvm::dyn_cast< llvm::PtrToIntInst >(&I)) {
                            unsigned dst_bits = PTI->getType()->getIntegerBitWidth();
                            if (dst_bits < 64)
                                ++narrow_ptrtoint;
                        }
                    }
                }
            }

            if (narrow_inttoptr || narrow_ptrtoint) {
                LOG(WARNING)
                    << "found " << narrow_inttoptr
                    << " inttoptr and " << narrow_ptrtoint
                    << " ptrtoint instruction(s) through sub-64-bit integers; "
                       "these silently truncate pointer values under the "
                       "x86_64 datalayout and will produce incorrect addresses "
                       "at KLEE execution time\n";
            }
        }

        void scanARM32ABIArtifacts(const llvm::Module &M) {
            static constexpr llvm::StringLiteral kARMIntrinsicPrefix = "llvm.arm.";
            static const llvm::StringRef kARMAbiPrefixes[] = {
                "extraout_r", "unaff_", "in_r", "in_lr", "in_sp",
            };
            auto matchesAbiPrefix = [](llvm::StringRef name) {
                for (auto p : kARMAbiPrefixes) {
                    if (name.starts_with(p))
                        return true;
                }
                return false;
            };

            unsigned intrinsic_count = 0;
            unsigned abi_artifact_count = 0;

            for (const auto &F : M) {
                if (F.getName().starts_with(kARMIntrinsicPrefix)) {
                    LOG(ERROR)
                        << "ARM32 intrinsic '" << F.getName().str()
                        << "' remains in retargeted module; "
                           "stubExternalFunctions will replace it with a "
                           "symbolic-return body and its target semantics "
                           "(saturation / co-processor / strict-align) will "
                           "be lost — provide a model via --model-library "
                           "or rewrite the call at decomp time\n";
                    ++intrinsic_count;
                }
                if (F.isDeclaration())
                    continue;
                for (const auto &arg : F.args()) {
                    if (matchesAbiPrefix(arg.getName())) {
                        LOG(ERROR)
                            << "function '" << F.getName().str()
                            << "' has ARM32 ABI-specific parameter '"
                            << arg.getName().str()
                            << "' (register-alias artifact); any contract "
                               "predicate targeting this argument asserts "
                               "over an AAPCS register slot that has no "
                               "x86_64 analogue and will be evaluated "
                               "against an unconstrained symbolic value\n";
                        ++abi_artifact_count;
                    }
                }
            }

            for (const auto &GV : M.globals()) {
                if (matchesAbiPrefix(GV.getName())) {
                    LOG(ERROR)
                        << "global '" << GV.getName().str()
                        << "' is an ARM32 ABI-specific artifact "
                           "(register-alias). Contracts or code relying on "
                           "its contents are meaningless under x86_64 "
                           "retargeting\n";
                    ++abi_artifact_count;
                }
            }

            if (intrinsic_count || abi_artifact_count) {
                LOG(ERROR)
                    << "found " << intrinsic_count
                    << " ARM32 intrinsic(s) and " << abi_artifact_count
                    << " ABI-specific artifact(s) post-retargeting; "
                       "verification results involving them are unsound\n";
            }
        }

        // After retargeting, pre-existing allocator declarations from the
        // source datalayout (e.g. `declare ptr @malloc(i32)` from an ARM32
        // decompile) still carry the old size-argument width. In opaque-
        // pointer LLVM this is a legal (if inconsistent) state — call
        // sites carry their own FunctionType — but KLEE's native allocator
        // intercept uses host size_t, and any pre-existing call that
        // passed an i32 size will be re-interpreted against the i64 ABI
        // at execution time. Warn the user; the robust fix is to rebuild
        // the IR at the host pointer width or supply a model library.
        void scanAllocatorSignatureDrift(const llvm::Module &M) {
            auto &Ctx       = M.getContext();
            auto *sizeTy    = M.getDataLayout().getIntPtrType(Ctx);
            auto *ptrTy     = llvm::PointerType::getUnqual(Ctx);
            auto *voidTy    = llvm::Type::getVoidTy(Ctx);
            unsigned size_bits = sizeTy->getIntegerBitWidth();

            struct AllocatorSig {
                llvm::StringRef name;
                llvm::FunctionType *expected;
            };

            AllocatorSig sigs[] = {
                { "malloc",  llvm::FunctionType::get(ptrTy,  { sizeTy }, false) },
                { "calloc",  llvm::FunctionType::get(ptrTy,  { sizeTy, sizeTy }, false) },
                { "realloc", llvm::FunctionType::get(ptrTy,  { ptrTy, sizeTy }, false) },
                { "free",    llvm::FunctionType::get(voidTy, { ptrTy }, false) },
            };

            for (auto &sig : sigs) {
                auto *fn = M.getFunction(sig.name);
                if (!fn)
                    continue;
                if (fn->getFunctionType() == sig.expected)
                    continue;
                LOG(WARNING)
                    << "allocator '" << sig.name.str()
                    << "' declaration does not match the post-retarget "
                       "host ABI (expected size_t = i" << size_bits
                    << "); KLEE's native intercept uses host size_t, so "
                       "pre-existing call sites that pass a narrower size "
                       "argument will be misinterpreted. Rebuild the IR "
                       "at the host pointer width or link a model library "
                       "via --model-library.\n";
            }
        }

    } // namespace

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
    void retargetModuleToX86_64(llvm::Module &M, bool verbose_flag) {
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

        if (verbose_flag) {
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
               "through narrow integers will be miscompiled. See "
               "docs/klee-integration.md#architecture-retargeting.\n";

        scanNarrowPointerCasts(M);

        // Surface ARM32 ABI-specific artifacts that survive retargeting.
        // Only meaningful when the original pointer width was 32 (strong
        // proxy for an ARM32 source); other 32-bit architectures may emit
        // false positives here, which is acceptable noise — the patterns
        // we flag are ARM-specific names.
        scanARM32ABIArtifacts(M);

        // Surface allocator declarations whose size-argument width is stale
        // after retargeting — KLEE's native intercept uses host size_t and
        // pre-existing call sites with the old width will be misinterpreted.
        scanAllocatorSignatureDrift(M);
    }

} // namespace patchestry::klee_verifier
