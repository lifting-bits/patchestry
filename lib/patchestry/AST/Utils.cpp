/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <cassert>
#include <cctype>
#include <unordered_map>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Type.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/SourceManager.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>

#include <patchestry/AST/Utils.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {

    clang::SourceLocation SourceLocation(clang::SourceManager &sm, std::string key) {
        auto &fm = sm.getFileManager();
        auto fe  = fm.getVirtualFileRef(key, static_cast< int >(key.size()), 0);
        std::unique_ptr< llvm::MemoryBuffer > buffer = llvm::MemoryBuffer::getMemBufferCopy(key);
        sm.overrideFileContents(fe, std::move(buffer));
        auto fid = sm.createFileID(fe, clang::SourceLocation(), clang::SrcMgr::C_User, 0);
        return sm.getLocForStartOfFile(fid);
    }

    clang::SourceLocation VirtualLoc(clang::ASTContext &ctx) {
        static std::unordered_map< clang::SourceManager *, clang::SourceLocation > cache;
        auto *sm = &ctx.getSourceManager();
        auto it  = cache.find(sm);
        if (it != cache.end()) {
            return it->second;
        }
        auto loc  = SourceLocation(*sm, "<patchestry-virtual>");
        cache[sm] = loc;
        return loc;
    }

    clang::QualType GetTypeFromSize(
        clang::ASTContext &ctx, unsigned bit_size, bool is_signed, bool is_integer
    ) {
        if (is_integer) {
            auto intty =
                ctx.getIntTypeForBitwidth(bit_size, static_cast< unsigned int >(is_signed));
            if (intty.isNull()) {
                return ctx.IntTy;
            }
            return intty;
        }

        switch (bit_size) {
            case 32:
                return ctx.FloatTy;
            case 64:
                return ctx.DoubleTy;
            case 80:
                return ctx.LongDoubleTy;
            default:
                llvm_unreachable("Unsupported float bit size in GetTypeFromSize");
        }
    }

    std::string LabelNameFromKey(std::string key) {
        std::replace(key.begin(), key.end(), ':', '_');
        return key;
    }

    std::string SanitizeKeyToIdent(std::string_view key) {
        std::string result;
        result.reserve(key.size());
        for (char c : key) {
            result += (std::isalnum(static_cast< unsigned char >(c)) || c == '_') ? c : '_';
        }
        return result;
    }

    clang::CastKind GetCastKind(
        clang::ASTContext &ctx, const clang::QualType &from_type, const clang::QualType &to_type
    ) {
        // Release-safe null guards.  Previously these were asserts, but
        // asserts compile out in Release builds and the very next line
        // (to_type->isVoidType()) would then segfault on a null QualType.
        if (to_type.isNull() || from_type.isNull()) {
            LOG(ERROR) << "GetCastKind called with null QualType: from_null="
                       << from_type.isNull() << " to_null=" << to_type.isNull()
                       << " -- returning CK_NoOp.";
            return clang::CK_NoOp;
        }

        // 1. Identity.
        if (ctx.hasSameUnqualifiedType(from_type, to_type)) {
            return clang::CK_NoOp;
        }

        // 2. Void.
        if (to_type->isVoidType()) {
            return clang::CK_ToVoid;
        }

        // 3. Boolean.
        if (to_type->isBooleanType()) {
            if (from_type->isIntegralOrEnumerationType()) {
                return clang::CK_IntegralToBoolean;
            }
            if (from_type->isFloatingType()) {
                return clang::CK_FloatingToBoolean;
            }
            if (from_type->isAnyPointerType() || from_type->isArrayType()
                || from_type->isFunctionType() || from_type->isNullPtrType())
            {
                return clang::CK_PointerToBoolean;
            }
            if (from_type->isMemberPointerType()) {
                return clang::CK_MemberPointerToBoolean;
            }
            // Complex / record → fall through to CK_BitCast guard.
        }

        // 4. Integral (includes char and unscoped enum).
        if (to_type->isIntegralOrEnumerationType()) {
            if (from_type->isIntegralOrEnumerationType()) {
                return clang::CK_IntegralCast;
            }
            if (from_type->isFloatingType()) {
                return clang::CK_FloatingToIntegral;
            }
            if (from_type->isAnyPointerType() || from_type->isArrayType()
                || from_type->isFunctionType() || from_type->isNullPtrType())
            {
                return clang::CK_PointerToIntegral;
            }
            // Complex / record → fall through.
        }

        // 5. Floating.
        if (to_type->isFloatingType()) {
            if (from_type->isIntegralOrEnumerationType()) {
                return clang::CK_IntegralToFloating;
            }
            if (from_type->isFloatingType()) {
                return clang::CK_FloatingCast;
            }
            // Pointer / array / complex → fall through.
        }

        // 6. Pointer.
        if (to_type->isAnyPointerType()) {
            if (from_type->isNullPtrType()) {
                return clang::CK_NullToPointer;
            }
            if (from_type->isIntegralOrEnumerationType()) {
                return clang::CK_IntegralToPointer;
            }
            if (from_type->isArrayType()) {
                return clang::CK_ArrayToPointerDecay;
            }
            if (from_type->isFunctionType()) {
                return clang::CK_FunctionToPointerDecay;
            }
            if (from_type->isAnyPointerType()) {
                return clang::CK_BitCast;
            }
            // Floating / complex / record → fall through.
        }

        // 7. Array destination intentionally not handled.
        //
        // Returning CK_ArrayToPointerDecay with `to_type` set to an array
        // is internally inconsistent: the cast kind produces a pointer
        // type but the ImplicitCastExpr's stated type would remain the
        // array, tripping downstream consumers (CIR lowering, pretty
        // printers).  More importantly, this section is effectively
        // unreachable: OpBuilder::make_cast pre-decays `from=array ->
        // to=pointer` before calling GetCastKind, and ShouldReinterpretCast
        // routes `arithmetic|pointer -> array` through reinterpret casts
        // upstream.  You cannot cast to an array type in C at the source
        // level, and Ghidra P-Code for C input never produces one as a
        // CAST destination.  Any exotic combination that does reach this
        // point (e.g. complex -> array) falls through to the CK_BitCast
        // guard below with a diagnostic log.

        // 8. Complex.
        if (to_type->isAnyComplexType()) {
            if (from_type->isIntegralOrEnumerationType()) {
                return clang::CK_IntegralRealToComplex;
            }
            if (from_type->isFloatingType()) {
                return clang::CK_FloatingRealToComplex;
            }
            if (from_type->isAnyComplexType()) {
                const bool from_is_int_complex = from_type->isComplexIntegerType();
                const bool to_is_int_complex   = to_type->isComplexIntegerType();
                if (from_is_int_complex && to_is_int_complex) {
                    return clang::CK_IntegralComplexCast;
                }
                if (!from_is_int_complex && !to_is_int_complex) {
                    return clang::CK_FloatingComplexCast;
                }
                return from_is_int_complex
                    ? clang::CK_IntegralComplexToFloatingComplex
                    : clang::CK_FloatingComplexToIntegralComplex;
            }
        }

        // 9. Member pointer.
        //
        // Member-pointer handling is intentionally minimal: Patchestry
        // decompiles C binaries via Ghidra P-Code, and the P-Code
        // serializer never emits `MemberPointerType` for C inputs, so
        // this branch is effectively dead for every current fixture.
        // When/if C++ decomp support is added, this section needs a
        // broader rewrite to distinguish:
        //   - nullptr -> T::*          (CK_NullToMemberPointer)
        //   - Base::* -> Derived::*    (CK_BaseToDerivedMemberPointer)
        //   - Derived::* -> Base::*    (CK_DerivedToBaseMemberPointer)
        //   - reinterpret_cast casts   (CK_ReinterpretMemberPointer)
        // which requires walking the class hierarchy, not just type
        // predicates.  For now, treat any memberptr-to-memberptr as
        // the base-to-derived form — this is better than the earlier
        // incorrect CK_BaseToDerived (for classes, not member pointers)
        // it replaced, and matches the only shape P-Code could ever
        // produce.  Other memberptr combinations fall through to the
        // CK_BitCast guard.
        //
        // Note on Clang's type predicates: isAnyPointerType() is
        // isPointerType() || isObjCObjectPointerType() — it does NOT
        // include MemberPointerType, so this branch is reachable
        // (the earlier to=AnyPointer dispatch does not shadow it).
        if (to_type->isMemberPointerType() && from_type->isMemberPointerType()) {
            return clang::CK_BaseToDerivedMemberPointer;
        }

        // 10. Ultimate fallback. Reached only by genuinely exotic pairs:
        //     float ↔ pointer, pointer ↔ complex, record-involved pairs
        //     that slipped past ShouldReinterpretCast, etc. Log so the
        //     specific pair is visible in stderr, then emit CK_BitCast so
        //     the decomp does not abort. OpBuilder::make_cast has a
        //     reinterpret-cast final fallback that can still rescue the
        //     result upstream if BitCast is wrong.
        LOG(ERROR) << "GetCastKind: no specific handler for cast: from='"
                   << from_type.getAsString() << "' to='"
                   << to_type.getAsString()
                   << "' -- falling back to CK_BitCast.";
        return clang::CK_BitCast;
    }

    bool
    ShouldReinterpretCast(const clang::QualType &from_type, const clang::QualType &to_type) {
        if (from_type->isRecordType()) {
            return to_type->isArithmeticType() || to_type->isAnyPointerType()
                || to_type->isArrayType() || to_type->isRecordType();
        }
        if (from_type->isArithmeticType() || from_type->isPointerType()) {
            return to_type->isArrayType() || to_type->isRecordType();
        }

        if (from_type->isArrayType()) {
            return to_type->isIntegerType() || to_type->isPointerType()
                || to_type->isRecordType();
        }

        return false;
    }

    clang::Expr *EnsureRValue(clang::ASTContext &ctx, clang::Expr *expr) {
        if (!expr || !expr->isGLValue()) {
            return expr;
        }
        return clang::ImplicitCastExpr::Create(
            ctx, expr->getType(), clang::CK_LValueToRValue, expr, nullptr,
            clang::VK_PRValue, clang::FPOptionsOverride()
        );
    }

    clang::Expr *NegateExpr(clang::ASTContext &ctx, clang::Expr *expr) {
        LOG_FATAL_IF(!expr, "NegateExpr called with null expression");
        auto *rv   = EnsureRValue(ctx, expr);
        auto loc   = expr->getExprLoc();
        auto *paren = new (ctx) clang::ParenExpr(loc, loc, rv);
        return clang::UnaryOperator::Create(
            ctx, paren, clang::UO_LNot, ctx.BoolTy, clang::VK_PRValue, clang::OK_Ordinary, loc,
            false, clang::FPOptionsOverride()
        );
    }

    clang::Expr *CloneExpr(clang::ASTContext &ctx, clang::Expr *expr) {
        if (!expr) return nullptr;

        auto loc = expr->getExprLoc();

        if (auto *il = llvm::dyn_cast< clang::IntegerLiteral >(expr)) {
            return clang::IntegerLiteral::Create(
                ctx, il->getValue(), il->getType(), loc
            );
        }
        if (auto *dre = llvm::dyn_cast< clang::DeclRefExpr >(expr)) {
            return clang::DeclRefExpr::Create(
                ctx, dre->getQualifierLoc(), dre->getTemplateKeywordLoc(),
                dre->getDecl(), dre->refersToEnclosingVariableOrCapture(),
                loc, dre->getType(), dre->getValueKind()
            );
        }
        if (auto *bo = llvm::dyn_cast< clang::BinaryOperator >(expr)) {
            return clang::BinaryOperator::Create(
                ctx, CloneExpr(ctx, bo->getLHS()), CloneExpr(ctx, bo->getRHS()),
                bo->getOpcode(), bo->getType(), bo->getValueKind(),
                bo->getObjectKind(), loc, clang::FPOptionsOverride()
            );
        }
        if (auto *uo = llvm::dyn_cast< clang::UnaryOperator >(expr)) {
            return clang::UnaryOperator::Create(
                ctx, CloneExpr(ctx, uo->getSubExpr()), uo->getOpcode(),
                uo->getType(), uo->getValueKind(), uo->getObjectKind(),
                loc, false, clang::FPOptionsOverride()
            );
        }
        if (auto *ice = llvm::dyn_cast< clang::ImplicitCastExpr >(expr)) {
            return clang::ImplicitCastExpr::Create(
                ctx, ice->getType(), ice->getCastKind(),
                CloneExpr(ctx, ice->getSubExpr()), nullptr,
                ice->getValueKind(), clang::FPOptionsOverride()
            );
        }
        if (auto *pe = llvm::dyn_cast< clang::ParenExpr >(expr)) {
            return new (ctx) clang::ParenExpr(
                loc, loc, CloneExpr(ctx, pe->getSubExpr())
            );
        }
        if (auto *cse = llvm::dyn_cast< clang::CStyleCastExpr >(expr)) {
            auto *cloned_sub = CloneExpr(ctx, cse->getSubExpr());
            return clang::CStyleCastExpr::Create(
                ctx, cse->getType(), cse->getValueKind(), cse->getCastKind(),
                cloned_sub, nullptr, clang::FPOptionsOverride(),
                ctx.getTrivialTypeSourceInfo(cse->getType()),
                cse->getLParenLoc(), cse->getRParenLoc()
            );
        }
        if (auto *ase = llvm::dyn_cast< clang::ArraySubscriptExpr >(expr)) {
            return new (ctx) clang::ArraySubscriptExpr(
                CloneExpr(ctx, ase->getLHS()),
                CloneExpr(ctx, ase->getRHS()),
                ase->getType(), ase->getValueKind(),
                ase->getObjectKind(), loc
            );
        }
        if (auto *me = llvm::dyn_cast< clang::MemberExpr >(expr)) {
            return clang::MemberExpr::CreateImplicit(
                ctx, CloneExpr(ctx, me->getBase()),
                me->isArrow(), me->getMemberDecl(),
                me->getType(), me->getValueKind(),
                me->getObjectKind()
            );
        }
        if (auto *ce = llvm::dyn_cast< clang::CallExpr >(expr)) {
            llvm::SmallVector< clang::Expr *, 4 > args;
            for (auto *a : ce->arguments())
                args.push_back(CloneExpr(ctx, a));
            return clang::CallExpr::Create(
                ctx, CloneExpr(ctx, ce->getCallee()), args,
                ce->getType(), ce->getValueKind(), loc,
                clang::FPOptionsOverride()
            );
        }
        if (auto *co = llvm::dyn_cast< clang::ConditionalOperator >(expr)) {
            return new (ctx) clang::ConditionalOperator(
                CloneExpr(ctx, co->getCond()),
                loc, CloneExpr(ctx, co->getTrueExpr()),
                loc, CloneExpr(ctx, co->getFalseExpr()),
                co->getType(), co->getValueKind(),
                co->getObjectKind()
            );
        }
        // Fallback: wrap in a ParenExpr to force a unique AST node.
        // This prevents shared Expr* pointers from causing CIR lowering
        // assertions when the same condition is used in multiple places.
        LOG(WARNING) << "CloneExpr: unhandled expression type "
                     << expr->getStmtClassName() << ", wrapping in ParenExpr\n";
        return new (ctx) clang::ParenExpr(loc, loc, expr);
    }

} // namespace patchestry::ast
