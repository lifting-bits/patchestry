#pragma once

#include <clang/AST/ASTContext.h>

#include <patchestry/Util/Options.hpp>

namespace patchestry::ast {

    bool runASTNormalizationPipeline(clang::ASTContext &ctx, const patchestry::Options &options);

} // namespace patchestry::ast
