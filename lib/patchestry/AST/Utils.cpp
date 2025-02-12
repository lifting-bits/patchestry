/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <clang/AST/ASTContext.h>
#include <clang/AST/Type.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/SourceManager.h>
#include <llvm/Support/MemoryBuffer.h>

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
            return ctx.getIntTypeForBitwidth(bit_size, static_cast< unsigned int >(is_signed));
        }

        switch (bit_size) {
            case 32:
                return ctx.FloatTy;
            case 64:
                return ctx.DoubleTy;
            case 80:
                return ctx.LongDoubleTy;
            default:
                assert(false);
                return clang::QualType();
        }
    }

    std::string label_name_from_key(std::string key) {
        std::replace(key.begin(), key.end(), ':', '_');
        return key;
    }

} // namespace patchestry::ast
