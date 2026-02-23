#pragma once

#include <memory>
#include <vector>

#include <clang/AST/ASTContext.h>

#include <patchestry/Util/Options.hpp>

namespace patchestry::ast {

    class ASTPass
    {
      public:
        virtual ~ASTPass() = default;

        virtual const char *name(void) const = 0;
        virtual bool run(clang::ASTContext &ctx, const patchestry::Options &options) = 0;
    };

    class ASTPassManager
    {
      public:
        void add_pass(std::unique_ptr< ASTPass > pass);
        bool run(clang::ASTContext &ctx, const patchestry::Options &options);

      private:
        std::vector< std::unique_ptr< ASTPass > > passes;
    };

} // namespace patchestry::ast
