/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <cctype>

#include <clang/AST/ASTContext.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Type.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/SourceManager.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>

#include <patchestry/AST/Utils.hpp>

namespace patchestry::ast {

    clang::SourceLocation sourceLocation(clang::SourceManager &sm, std::string key) {
        auto &fm = sm.getFileManager();
        auto fe  = fm.getVirtualFileRef(key, static_cast< int >(key.size()), 0);
        std::unique_ptr< llvm::MemoryBuffer > buffer = llvm::MemoryBuffer::getMemBuffer(key);
        sm.overrideFileContents(fe, std::move(buffer));
        auto fid = sm.createFileID(fe, clang::SourceLocation(), clang::SrcMgr::C_User, 0);
        return sm.getLocForStartOfFile(fid);
    }

    clang::QualType getTypeFromSize(
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
                llvm_unreachable("Unsupported float bit size in getTypeFromSize");
        }
    }

    std::string labelNameFromKey(std::string key) {
        std::replace(key.begin(), key.end(), ':', '_');
        return key;
    }

    std::string sanitize_key_to_ident(std::string_view key) {
        std::string result;
        result.reserve(key.size());
        for (char c : key) {
            result += (std::isalnum(static_cast< unsigned char >(c)) || c == '_') ? c : '_';
        }
        return result;
    }

    clang::CastKind getCastKind(
        clang::ASTContext &ctx, const clang::QualType &from_type, const clang::QualType &to_type
    ) {
        assert(!to_type.isNull() && "to_type is null");
        assert(!from_type.isNull() && "from_type is null");

        // Identity cast
        if (ctx.hasSameUnqualifiedType(from_type, to_type)) {
            return clang::CastKind::CK_NoOp;
        }

        // Void cast
        if (to_type->isVoidType()) {
            return clang::CastKind::CK_ToVoid;
        }

        // Boolean conversion
        if (to_type->isBooleanType()) {
            if (from_type->isIntegerType()) {
                return clang::CK_IntegralToBoolean;
            }
            if (from_type->isFloatingType()) {
                return clang::CK_FloatingToBoolean;
            }
            if (from_type->isPointerType()) {
                return clang::CK_PointerToBoolean;
            }
            if (from_type->isMemberPointerType()) {
                return clang::CK_MemberPointerToBoolean;
            }
        }

        if (from_type->isIntegerType() && to_type->isCharType()) {
            return clang::CK_IntegralCast;
        }

        if (from_type->isIntegerType() && to_type->isArrayType()) {
            return clang::CK_IntegralToPointer;
        }

        if (from_type->isIntegerType() && to_type->isRecordType()) {
            return clang::CK_BitCast;
        }

        // Integer conversion
        if (to_type->isIntegerType()) {
            if (from_type->isBooleanType()) {
                return clang::CK_BooleanToSignedIntegral;
            }
            if (from_type->isFloatingType()) {
                return clang::CK_FloatingToIntegral;
            }
            if (from_type->isPointerType()) {
                return clang::CK_PointerToIntegral;
            }
            if (from_type->isEnumeralType() || from_type->isIntegerType()) {
                return clang::CK_IntegralCast;
            }
        }

        // Floating conversion
        if (to_type->isFloatingType()) {
            if (from_type->isIntegerType()) {
                return clang::CK_IntegralToFloating;
            }
            if (from_type->isFloatingType()) {
                return clang::CK_FloatingCast;
            }
        }

        // Handle pointer target type
        if (to_type->isAnyPointerType()) {
            if (from_type->isIntegerType()) {
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
        }
        // Array conversions
        if (to_type->isArrayType()) {
            if (from_type->canDecayToPointerType()) {
                return clang::CK_ArrayToPointerDecay;
            }
        }

        // Complex numbers
        if (to_type->isComplexType()) {
            if (from_type->isIntegerType()) {
                return clang::CK_IntegralComplexCast;
            }
            if (from_type->isFloatingType()) {
                return clang::CK_FloatingComplexCast;
            }
        }

        // Member pointer conversions
        if (to_type->isMemberPointerType() && from_type->isMemberPointerType()) {
            return clang::CK_BaseToDerived;
        }

        assert(false && "Failed to find implicit cast kind");

        return clang::CastKind::CK_NoOp;
    }

    bool
    shouldReinterpretCast(const clang::QualType &from_type, const clang::QualType &to_type) {
        if (from_type->isRecordType()) {
            return to_type->isArithmeticType() || to_type->isAnyPointerType()
                || to_type->isArrayType();
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

} // namespace patchestry::ast
