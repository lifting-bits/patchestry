/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "Runtime.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>


namespace patchestry::klee_verifier {

    llvm::FunctionCallee getKleeMakeSymbolic(llvm::Module &M) {
        auto &Ctx    = M.getContext();
        auto *voidTy = llvm::Type::getVoidTy(Ctx);
        auto *ptrTy  = llvm::PointerType::getUnqual(Ctx);
        auto *sizeTy = M.getDataLayout().getIntPtrType(Ctx);
        auto *FT     = llvm::FunctionType::get(voidTy, { ptrTy, sizeTy, ptrTy }, false);
        return M.getOrInsertFunction("klee_make_symbolic", FT);
    }

    llvm::FunctionCallee getKleeAssume(llvm::Module &M) {
        auto &Ctx      = M.getContext();
        auto *voidTy   = llvm::Type::getVoidTy(Ctx);
        auto *intptrTy = M.getDataLayout().getIntPtrType(Ctx);
        auto *FT       = llvm::FunctionType::get(voidTy, { intptrTy }, false);
        return M.getOrInsertFunction("klee_assume", FT);
    }

    llvm::FunctionCallee getKleeAbort(llvm::Module &M) {
        auto &Ctx    = M.getContext();
        auto *voidTy = llvm::Type::getVoidTy(Ctx);
        auto *FT     = llvm::FunctionType::get(voidTy, {}, false);
        return M.getOrInsertFunction("klee_abort", FT);
    }

    llvm::FunctionCallee getMalloc(llvm::Module &M) {
        auto &Ctx    = M.getContext();
        auto *ptrTy  = llvm::PointerType::getUnqual(Ctx);
        auto *sizeTy = M.getDataLayout().getIntPtrType(Ctx);
        auto *FT     = llvm::FunctionType::get(ptrTy, { sizeTy }, /*isVarArg=*/false);
        return M.getOrInsertFunction("malloc", FT);
    }

    bool typeContainsPointer(
        llvm::Type *T, llvm::SmallPtrSetImpl< llvm::Type * > &seen
    ) {
        llvm::SmallVector< llvm::Type *, 8 > worklist;
        if (T)
            worklist.push_back(T);

        while (!worklist.empty()) {
            llvm::Type *cur = worklist.pop_back_val();
            if (!cur || !seen.insert(cur).second)
                continue;
            if (cur->isPointerTy())
                return true;
            if (auto *ST = llvm::dyn_cast< llvm::StructType >(cur)) {
                if (!ST->isOpaque()) {
                    for (llvm::Type *elt : ST->elements())
                        worklist.push_back(elt);
                }
            } else if (auto *AT = llvm::dyn_cast< llvm::ArrayType >(cur)) {
                worklist.push_back(AT->getElementType());
            } else if (auto *VT = llvm::dyn_cast< llvm::VectorType >(cur)) {
                worklist.push_back(VT->getElementType());
            }
        }
        return false;
    }

} // namespace patchestry::klee_verifier
