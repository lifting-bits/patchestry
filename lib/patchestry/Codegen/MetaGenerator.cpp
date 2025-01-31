/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#define VAST_ENABLE_EXCEPTIONS
#include <vast/Util/Warnings.hpp>

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <mlir/IR/MLIRContext.h>

VAST_UNRELAX_WARNINGS

#define GAP_ENABLE_COROUTINES

#include <vast/CodeGen/DefaultMetaGenerator.hpp>
#include <vast/Dialect/Meta/MetaAttributes.hpp>

#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Codegen/MetaGenerator.hpp>

namespace patchestry::codegen {

    namespace {

        std::string get_path_to_source(clang::ASTContext *actx) { // NOLINT
            auto main_file_id     = actx->getSourceManager().getMainFileID();
            const auto &main_file = *actx->getSourceManager().getFileEntryForID(main_file_id);
            return main_file.tryGetRealPathName().str();
        }

        mlir::Location
        make_loc_name(mlir::MLIRContext *mctx, clang::ASTContext *actx) { // NOLINT
            if (auto path = get_path_to_source(actx); !path.empty()) {
                return mlir::FileLineColLoc::get(mctx, path, 0, 0);
            }
            return mlir::UnknownLoc::get(mctx);
        }
    } // namespace

    MetaGenerator::MetaGenerator(
        clang::ASTContext *actx, mlir::MLIRContext *mctx, const LocationMap &locs
    )
        : actx(actx), mctx(mctx), locations(locs) {}

    void *MetaGenerator::raw_pointer(const clang::Decl *decl) const {
        return static_cast< void * >(const_cast< clang::Decl * >(decl));
    }

    void *MetaGenerator::raw_pointer(const clang::Stmt *stmt) const {
        return static_cast< void * >(const_cast< clang::Stmt * >(stmt));
    }

    void *MetaGenerator::raw_pointer(const clang::Expr *expr) const {
        return static_cast< void * >(const_cast< clang::Expr * >(expr));
    }

    mlir::Location MetaGenerator::location(const clang::Decl *decl) const {
        return location(raw_pointer(decl), decl->getLocation());
    }

    mlir::Location MetaGenerator::location(const clang::Stmt *stmt) const {
        return location(raw_pointer(stmt), stmt->getBeginLoc());
    }

    mlir::Location MetaGenerator::location(const clang::Expr *expr) const {
        return location(raw_pointer(expr), expr->getExprLoc());
    }

    uint64_t
    MetaGenerator::address_from_location(const std::string &str, char delimiter) const {
        std::stringstream ss(str);
        std::string token;
        int count = 0;

        while (std::getline(ss, token, delimiter)) {
            ++count;
            if (count == 2) {
                return std::stoi(token, nullptr, 16);
            }
        }
        return 0;
    }

    mlir::Location MetaGenerator::location(void *data, const clang::SourceLocation &loc) const {
        mlir::StringAttr string_attr;
        if (locations.contains(data)) {
            const auto &location_str = locations.at(data);
            string_attr              = mlir::StringAttr::get(mctx, location_str);
        } else {
            string_attr = mlir::StringAttr::get(mctx, "unknown_location");
        }

        mlir::DictionaryAttr metadata = mlir::DictionaryAttr::get(
            mctx,
            {
                {mlir::StringAttr::get(mctx, "pcode"), string_attr}
        }
        );
        return mlir::FusedLoc::get(make_loc_name(mctx, actx), metadata, mctx);
    }

} // namespace patchestry::codegen
