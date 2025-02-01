/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <clang/AST/ASTContext.h>
#include <mlir/IR/MLIRContext.h>

#define GAP_ENABLE_COROUTINES
#include <vast/CodeGen/CodeGenMetaGenerator.hpp>

#include <patchestry/Codegen/Codegen.hpp>

namespace patchestry::codegen {

    struct MetaGenerator final : vast::cg::meta_generator
    {
        MetaGenerator(
            clang::ASTContext *actx, mlir::MLIRContext *mctx, const LocationMap &locs
        );

        void *raw_pointer(const clang::Decl *decl) const;

        void *raw_pointer(const clang::Stmt *stmt) const;

        void *raw_pointer(const clang::Expr *expr) const;

        mlir::Location location(const clang::Decl *decl) const override;

        mlir::Location location(const clang::Stmt *stmt) const override;

        mlir::Location location(const clang::Expr *expr) const override;

      private:
        uint64_t address_from_location(const std::string &str, char delimiter) const;

        mlir::Location location(void *data, const clang::SourceLocation &loc) const;

        clang::ASTContext *actx;

        mlir::MLIRContext *mctx;
        const LocationMap &locations;
    };
} // namespace patchestry::codegen
