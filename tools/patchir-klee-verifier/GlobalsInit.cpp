/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "GlobalsInit.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/Util/Log.hpp>

#include "Options.hpp"
#include "Runtime.hpp"

namespace patchestry::klee_verifier {

    // Get or declare klee_make_symbolic: void klee_make_symbolic(void*, size_t, const char*)
    extern llvm::FunctionCallee getKleeMakeSymbolic(llvm::Module &M);

    // Get or declare malloc: i8* malloc(size_t). KLEE intercepts this at
    // runtime and returns a tracked heap allocation that survives the
    // caller's stack frame.
    extern llvm::FunctionCallee getMalloc(llvm::Module &M);

    namespace {

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
        //      as a counted loop over the descriptor table. The harness's
        //      entry point issues a single call to this dispatcher.
        //
        // A runtime depth cap (`--klee-init-max-depth`, default 2) bounds the
        // live recursion of cyclic type initializers: at `depth >= max` the
        // pointer field is set to null and the recursive call is skipped,
        // so KLEE doesn't explode on self-referential lists/trees.
        // ========================================================================

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
        void mergeInference(
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
        PointerFieldInferenceMap inferPointerFieldTypes(llvm::Module &M) {
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
        // Filters:
        //   * skip external declarations (no storage to symbolize)
        //   * skip constants (their value is authoritative)
        //   * skip zero-sized / unsized globals
        //   * skip LLVM-internal globals (@llvm.global_ctors etc.)
        //   * skip the tool's own synthesized descriptor/name storage, so we
        //     don't accidentally symbolize our own runtime data on re-runs
        bool isToolSynthesizedGlobal(const llvm::GlobalVariable *GV) {
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
        unsigned materializeExternalGlobals(llvm::Module &M) {
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

        void collectModuleGlobals(
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
        void logCollectedGlobals(
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

        // Forward declarations — these functions form a tight recursive
        // cluster (per-type init bodies invoke emitPointerField, which calls
        // getOrCreatePerTypeInit, which calls back into buildTypeInitBody).
        // emitInitWithInitializer is the initializer-aware top-level walker
        // that delegates to buildTypeInitBody for trivial-init subregions.
        llvm::Function *getOrCreatePerTypeInit(
            llvm::Module &M, llvm::Type *T,
            const PointerFieldInferenceMap &im, TypeInitCache &cache
        );
        void buildTypeInitBody(
            llvm::IRBuilder<> &B, llvm::Value *addr, llvm::Value *depth,
            llvm::Type *T, const PointerFieldInferenceMap &im,
            TypeInitCache &cache, const llvm::Twine &name, llvm::Module &M
        );
        void emitInitWithInitializer(
            llvm::IRBuilder<> &B, llvm::Value *addr, llvm::Type *T,
            llvm::Constant *init, const PointerFieldInferenceMap &im,
            TypeInitCache &cache, const llvm::Twine &name, llvm::Module &M
        );

        // Sanitize an LLVM type name so it can be embedded in a function
        // symbol. Non-identifier characters are replaced with '_'. Anonymous
        // and literal struct types get a "anon_<N>" suffix based on a running
        // counter so the generated symbols stay unique per type instance.
        std::string mangleTypeName(llvm::Type *T) {
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
                static unsigned anon_counter = 0;
                os << "anon_" << anon_counter++;
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
            static unsigned unknown_counter = 0;
            os << "ty" << unknown_counter++;
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
        void emitPointerField(
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
            if (!pointee || !pointee->isSized()) {
                llvm::Value *size_val = llvm::ConstantInt::get(sizeTy, symbolic_ptr_size);
                llvm::Value *buf      = B.CreateCall(malloc_fn, { size_val });
                llvm::Value *name_str = B.CreateGlobalString(name.str());
                B.CreateCall(make_sym, { buf, size_val, name_str });
                B.CreateStore(buf, field_ptr);
                return;
            }
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

            auto *insert_bb = B.GetInsertBlock();
            assert(insert_bb && "IRBuilder must have an active insertion point");
            llvm::Function *parent_fn = insert_bb->getParent();
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
        void buildTypeInitBody(
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
                    std::string child_name = (name + ".f" + llvm::Twine(i)).str();
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
                        std::string elt_name = (name + "[" + llvm::Twine(j) + "]").str();
                        buildTypeInitBody(
                            B, elt_ptr, depth, ET, im, cache,
                            elt_name, M
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

        // Return true when the initializer carries no meaningful bytes that
        // need to be preserved. Treats null pointer, undef, zero-aggregate,
        // and zero-valued scalar constants as trivial — symbolizing over them
        // has no observable effect since any concrete seed (including zero)
        // is overwritten by klee_make_symbolic anyway.
        bool isTrivialInit(llvm::Constant *c) {
            if (!c)
                return true;
            if (llvm::isa< llvm::UndefValue >(c))
                return true;
            if (c->isNullValue())
                return true;
            return false;
        }

        // Walk a global's type and initializer in lockstep, emitting
        // symbolization only for sub-regions that are trivially initialized
        // (null / undef / zero). Concrete non-trivial pointer values
        // (function pointers, cross-global references, ConstantExpr ptrs)
        // and concrete non-trivial scalar values are left untouched so the
        // global's declared starting state survives into the symbolic run.
        //
        // This is the top-level entry for per-global init. Recursive pointee
        // allocation (pointer fields whose inferred kind is PointeeType)
        // still goes through emitPointerField → getOrCreatePerTypeInit,
        // because the malloc'd pointee has no initializer to preserve and
        // the per-type cache is needed to close type cycles.
        void emitInitWithInitializer(
            llvm::IRBuilder<> &B, llvm::Value *addr, llvm::Type *T,
            llvm::Constant *init, const PointerFieldInferenceMap &im,
            TypeInitCache &cache, const llvm::Twine &name, llvm::Module &M
        ) {
            auto &Ctx = M.getContext();

            // Trivial init — fall through to the full symbolization path
            // (flat fast-path for pointer-free types, recursive for the
            // rest). Matches the pre-fix behavior for zero-initialized
            // globals.
            if (isTrivialInit(init)) {
                llvm::Value *zero =
                    llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 0);
                buildTypeInitBody(B, addr, zero, T, im, cache, name, M);
                return;
            }

            if (auto *ST = llvm::dyn_cast< llvm::StructType >(T)) {
                for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
                    llvm::Type *field_ty = ST->getElementType(i);
                    llvm::Constant *field_init = init->getAggregateElement(i);
                    llvm::Value *field_ptr = B.CreateStructGEP(ST, addr, i);
                    std::string child_name = (name + ".f" + llvm::Twine(i)).str();

                    if (field_ty->isPointerTy()) {
                        // Concrete pointer (Function, GlobalVariable,
                        // ConstantExpr) → preserve the initializer's value;
                        // do nothing. Null/undef pointer → symbolize through
                        // the existing path.
                        if (isTrivialInit(field_init)) {
                            llvm::Value *depth_zero =
                                llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 0);
                            emitPointerField(
                                B, field_ptr, depth_zero,
                                std::make_pair(ST, i),
                                im, cache, child_name, M
                            );
                        }
                    } else {
                        emitInitWithInitializer(
                            B, field_ptr, field_ty, field_init, im, cache,
                            child_name, M
                        );
                    }
                }
                return;
            }

            if (auto *AT = llvm::dyn_cast< llvm::ArrayType >(T)) {
                uint64_t N = AT->getNumElements();
                llvm::Type *ET = AT->getElementType();
                // Non-trivial array init: walk each element so per-element
                // concrete values survive. We intentionally ignore
                // klee_init_array_expand_limit here — that limit guards the
                // trivial-init fast path (flat symbolic over N elements);
                // for a non-trivial init there is no safe flat path, so
                // correctness wins over IR bound.
                for (uint64_t j = 0; j < N; ++j) {
                    llvm::Value *idxs[] = {
                        llvm::ConstantInt::get(llvm::Type::getInt64Ty(Ctx), 0),
                        llvm::ConstantInt::get(llvm::Type::getInt64Ty(Ctx), j),
                    };
                    llvm::Value *elt_ptr = B.CreateInBoundsGEP(AT, addr, idxs);
                    llvm::Constant *elt_init =
                        init->getAggregateElement(static_cast< unsigned >(j));
                    std::string elt_name =
                        (name + "[" + llvm::Twine(j) + "]").str();
                    emitInitWithInitializer(
                        B, elt_ptr, ET, elt_init, im, cache, elt_name, M
                    );
                }
                return;
            }

            // Top-level scalar (integer/float/pointer/vector) with a
            // non-trivial constant initializer — preserve as-is. The
            // declared value lives in the global's storage already; no
            // store/symbolize needed.
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
        llvm::Function *getOrCreatePerTypeInit(
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

        // Create an internal `void @__klee_init_g_<name>()` wrapper per
        // global. Unlike the earlier design that always called the cached
        // per-type init (which unconditionally symbolized every sub-field),
        // the wrapper now walks the global's concrete initializer so that
        // preinitialized pointer fields — function-pointer dispatch tables,
        // cross-global references, vtables — survive into the symbolic run.
        // Sub-regions that are trivially initialized (null/undef/zero) still
        // delegate to the cached per-type init, so the fast path and cycle-
        // breaking machinery is preserved for self-referential types.
        //
        // Caller owns the per-GV cache (the `wrappers` map in
        // `installGlobalsInit`), keyed by GlobalVariable* identity. We do
        // NOT do a name-based `M.getFunction` lookup here: the sanitizer
        // collapses `.` / `$` / other non-identifier characters to `_`, so
        // two distinct globals like `@foo.bar` and `@foo_bar` would both
        // map to `__klee_init_g_foo_bar`. A name-based cache would hand
        // the second caller the first caller's wrapper, and the descriptor
        // table would double-initialize one global while skipping the
        // other. `Function::Create` auto-suffixes duplicate base names
        // (`.1`, `.2`, ...) so each global gets a uniquely-named wrapper.
        llvm::Function *getOrCreatePerGlobalInit(
            llvm::Module &M, llvm::GlobalVariable *GV,
            const PointerFieldInferenceMap &im, TypeInitCache &cache
        ) {
            auto &Ctx    = M.getContext();
            auto *voidTy = llvm::Type::getVoidTy(Ctx);

            std::string wrapper_name = ("__klee_init_g_" + GV->getName()).str();
            // Sanitize the global's name the same way as types.
            for (char &c : wrapper_name) {
                if (!std::isalnum(static_cast< unsigned char >(c)) && c != '_')
                    c = '_';
            }

            auto *FT = llvm::FunctionType::get(voidTy, {}, false);
            auto *F  = llvm::Function::Create(
                FT, llvm::GlobalValue::InternalLinkage, wrapper_name, &M
            );

            auto *entry_bb = llvm::BasicBlock::Create(Ctx, "entry", F);
            llvm::IRBuilder<> B(entry_bb);

            // `materializeExternalGlobals` stamps a zero initializer on
            // previously-external globals, so `getInitializer()` is safe.
            // A defensive nullptr check still matters in case a future
            // pass adds a global without materializing an initializer.
            llvm::Constant *init =
                GV->hasInitializer() ? GV->getInitializer() : nullptr;
            emitInitWithInitializer(
                B, GV, GV->getValueType(), init, im, cache, GV->getName(), M
            );
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
        std::pair< llvm::GlobalVariable *, uint64_t >
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
        llvm::Function *getOrCreateInitGlobalsDispatcher(
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

    } // namespace

    llvm::Function *installGlobalsInit(llvm::Module &M, GlobalsInitStats &stats) {
        auto inference_map = inferPointerFieldTypes(M);

        // Pre-pass: promote `external global` declarations referenced by the
        // code under analysis into internally-linked definitions with zero
        // initializers. Without this, decompiled-firmware globals that the
        // JSON exporter left as externs (common when the source binary had no
        // initializer in its own translation unit) are skipped by
        // collectModuleGlobals and the target function then references an
        // unlinked symbol at KLEE time — the exact failure shape seen on
        // `@usb_g` in bl_usb__send_message.
        stats.materialized = materializeExternalGlobals(M);

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

        stats.collected      = module_globals.size();
        stats.type_init_fns  = type_init_cache.size();
        stats.pointer_fields = inference_map.size();
        return dispatcher;
    }

} // namespace patchestry::klee_verifier
