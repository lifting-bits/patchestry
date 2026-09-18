/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/TUPrinter.hpp>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/PrettyPrinter.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

#include <string>
#include <unordered_set>
#include <vector>

namespace patchestry::ast {

    namespace {

        // ------------------------------------------------------------------
        // Reparenthesization
        //
        // StmtPrinter never inserts parentheses; it relies on the ParenExpr
        // nodes Sema creates while parsing source.  Expressions built by the
        // lifter have none.  The ranks below mirror the C grammar: binary
        // operators reuse clang's prec::Level values, and unary, postfix and
        // primary expressions sit above them.
        // ------------------------------------------------------------------

        enum Rank : int {
            kComma       = 1,
            kAssignment  = 2,
            kConditional = 3,
            kLogicalOr   = 4,
            kUnary       = 20,
            kPostfix     = 30,
            kPrimary     = 40,
        };

        int BinaryRank(clang::BinaryOperatorKind op) {
            switch (op) {
                case clang::BO_PtrMemD:
                case clang::BO_PtrMemI:
                    return 15;
                case clang::BO_Mul:
                case clang::BO_Div:
                case clang::BO_Rem:
                    return 14;
                case clang::BO_Add:
                case clang::BO_Sub:
                    return 13;
                case clang::BO_Shl:
                case clang::BO_Shr:
                    return 12;
                case clang::BO_Cmp:
                    return 11;
                case clang::BO_LT:
                case clang::BO_GT:
                case clang::BO_LE:
                case clang::BO_GE:
                    return 10;
                case clang::BO_EQ:
                case clang::BO_NE:
                    return 9;
                case clang::BO_And:
                    return 8;
                case clang::BO_Xor:
                    return 7;
                case clang::BO_Or:
                    return 6;
                case clang::BO_LAnd:
                    return 5;
                case clang::BO_LOr:
                    return kLogicalOr;
                case clang::BO_Assign:
                case clang::BO_MulAssign:
                case clang::BO_DivAssign:
                case clang::BO_RemAssign:
                case clang::BO_AddAssign:
                case clang::BO_SubAssign:
                case clang::BO_ShlAssign:
                case clang::BO_ShrAssign:
                case clang::BO_AndAssign:
                case clang::BO_XorAssign:
                case clang::BO_OrAssign:
                    return kAssignment;
                case clang::BO_Comma:
                    return kComma;
            }
            return kPrimary;
        }

        // Object pointer to `void *` without dropping qualifiers.  C converts
        // that implicitly, and the lifter never does arithmetic on a `void *`
        // operand (bases are cast to `char *` first), so the printed C means
        // the same without the cast.
        bool IsImplicitVoidPointerConversion(const clang::ImplicitCastExpr *cast) {
            const auto *to   = cast->getType()->getAs< clang::PointerType >();
            const auto *from = cast->getSubExpr()->getType()->getAs< clang::PointerType >();
            if (to == nullptr || from == nullptr) { return false; }
            const auto to_pointee   = to->getPointeeType();
            const auto from_pointee = from->getPointeeType();
            if (!to_pointee->isVoidType() || from_pointee->isFunctionType()) { return false; }
            return (from_pointee.getCVRQualifiers() & ~to_pointee.getCVRQualifiers()) == 0;
        }

        // Implicit conversions that change the representation are printed as
        // explicit casts: StmtPrinter drops them, and the resulting C would be
        // ill-typed (pointer mismatches, int-to-pointer) or mean something
        // else (`void *` arithmetic where the lifter had `char *`).  The one
        // exception is the conversion to `void *` C makes on its own.
        bool IsPrintedConversion(const clang::ImplicitCastExpr *cast) {
            switch (cast->getCastKind()) {
                case clang::CK_BitCast:
                    return !IsImplicitVoidPointerConversion(cast);
                case clang::CK_IntegralToPointer:
                case clang::CK_PointerToIntegral:
                    return true;
                default:
                    return false;
            }
        }

        // Rank at which StmtPrinter prints `expr`.  Implicit nodes print
        // nothing of their own, so look through them, except for the
        // conversions printed as casts.
        int PrintedRank(const clang::Expr *expr) {
            while (true) {
                if (const auto *ice = llvm::dyn_cast< clang::ImplicitCastExpr >(expr)) {
                    if (IsPrintedConversion(ice)) { return kUnary; }
                    expr = ice->getSubExpr();
                    continue;
                }
                if (const auto *full = llvm::dyn_cast< clang::FullExpr >(expr)) {
                    expr = full->getSubExpr();
                    continue;
                }
                break;
            }
            if (llvm::isa< clang::ParenExpr >(expr)) { return kPrimary; }
            if (const auto *bin = llvm::dyn_cast< clang::BinaryOperator >(expr)) {
                return BinaryRank(bin->getOpcode());
            }
            if (llvm::isa< clang::AbstractConditionalOperator >(expr)) { return kConditional; }
            if (const auto *un = llvm::dyn_cast< clang::UnaryOperator >(expr)) {
                return un->isPostfix() ? kPostfix : kUnary;
            }
            if (llvm::isa< clang::CStyleCastExpr, clang::UnaryExprOrTypeTraitExpr >(expr)) {
                return kUnary;
            }
            if (llvm::isa<
                    clang::CallExpr, clang::ArraySubscriptExpr, clang::MemberExpr,
                    clang::CompoundLiteralExpr >(expr))
            {
                return kPostfix;
            }
            return kPrimary;
        }

        // Parenthesize the written expression under `slot`.  Implicit casts
        // stay on the outside so the tree keeps the ImplicitCast(Paren(...))
        // shape Sema would have produced.  Returns the value to store in the
        // parent's slot.
        clang::Expr *Parenthesize(clang::ASTContext &ctx, clang::Expr *slot) {
            clang::Expr *parent = nullptr;
            clang::Expr *inner  = slot;
            while (true) {
                clang::Expr *next = nullptr;
                if (auto *ice = llvm::dyn_cast< clang::ImplicitCastExpr >(inner)) {
                    if (IsPrintedConversion(ice)) { break; } // printed: wrap it whole
                    next = ice->getSubExpr();
                } else if (auto *full = llvm::dyn_cast< clang::FullExpr >(inner)) {
                    next = full->getSubExpr();
                }
                if (next == nullptr) { break; }
                parent = inner;
                inner  = next;
            }

            auto *paren =
                new (ctx) clang::ParenExpr(inner->getBeginLoc(), inner->getEndLoc(), inner);
            if (parent == nullptr) { return paren; }
            if (auto *ice = llvm::dyn_cast< clang::ImplicitCastExpr >(parent)) {
                ice->setSubExpr(paren);
            } else {
                llvm::cast< clang::FullExpr >(parent)->setSubExpr(paren);
            }
            return slot;
        }

        clang::Expr *WrapIfBelow(clang::ASTContext &ctx, clang::Expr *expr, int min_rank) {
            return PrintedRank(expr) < min_rank ? Parenthesize(ctx, expr) : expr;
        }

        bool IsAddressOf(const clang::Expr *expr) {
            const auto *un = llvm::dyn_cast< clang::UnaryOperator >(expr->IgnoreImplicit());
            return un != nullptr && un->getOpcode() == clang::UO_AddrOf;
        }

        void Reparenthesize(clang::ASTContext &ctx, clang::Stmt *stmt) {
            if (stmt == nullptr) { return; }
            for (clang::Stmt *child : stmt->children()) { Reparenthesize(ctx, child); }
            auto *expr = llvm::dyn_cast< clang::Expr >(stmt);
            if (expr == nullptr) { return; }

            if (auto *bin = llvm::dyn_cast< clang::BinaryOperator >(expr)) {
                const int rank         = BinaryRank(bin->getOpcode());
                const bool right_assoc = rank == kAssignment;
                const int lhs_rank     = PrintedRank(bin->getLHS());
                const int rhs_rank     = PrintedRank(bin->getRHS());
                if (lhs_rank < rank || (lhs_rank == rank && right_assoc)) {
                    bin->setLHS(Parenthesize(ctx, bin->getLHS()));
                }
                if (rhs_rank < rank || (rhs_rank == rank && !right_assoc)) {
                    bin->setRHS(Parenthesize(ctx, bin->getRHS()));
                }
                return;
            }
            if (auto *cond = llvm::dyn_cast< clang::ConditionalOperator >(expr)) {
                // logical-OR-expression ? expression : conditional-expression
                unsigned index = 0;
                for (clang::Stmt *&slot : cond->children()) {
                    auto *sub = llvm::cast< clang::Expr >(slot);
                    if (index == 0) {
                        slot = WrapIfBelow(ctx, sub, kLogicalOr);
                    } else if (index == 2) {
                        slot = WrapIfBelow(ctx, sub, kConditional);
                    }
                    ++index;
                }
                return;
            }
            if (auto *un = llvm::dyn_cast< clang::UnaryOperator >(expr)) {
                auto *sub          = un->getSubExpr();
                const int min_rank = un->isPostfix() ? kPostfix : kUnary;
                // `& &x` would print as the `&&` token.
                const bool addr_of_addr =
                    un->getOpcode() == clang::UO_AddrOf && IsAddressOf(sub);
                if (PrintedRank(sub) < min_rank || addr_of_addr) {
                    un->setSubExpr(Parenthesize(ctx, sub));
                }
                return;
            }
            if (auto *cast = llvm::dyn_cast< clang::CStyleCastExpr >(expr)) {
                cast->setSubExpr(WrapIfBelow(ctx, cast->getSubExpr(), kUnary));
                return;
            }
            if (auto *ice = llvm::dyn_cast< clang::ImplicitCastExpr >(expr);
                ice != nullptr && IsPrintedConversion(ice))
            {
                ice->setSubExpr(WrapIfBelow(ctx, ice->getSubExpr(), kUnary));
                return;
            }
            if (auto *trait = llvm::dyn_cast< clang::UnaryExprOrTypeTraitExpr >(expr)) {
                if (!trait->isArgumentType()) {
                    trait->setArgument(WrapIfBelow(ctx, trait->getArgumentExpr(), kUnary));
                }
                return;
            }
            if (auto *subscript = llvm::dyn_cast< clang::ArraySubscriptExpr >(expr)) {
                if (subscript->getLHS() == subscript->getBase()) {
                    subscript->setLHS(WrapIfBelow(ctx, subscript->getLHS(), kPostfix));
                } else {
                    subscript->setRHS(WrapIfBelow(ctx, subscript->getRHS(), kPostfix));
                }
                return;
            }
            if (auto *member = llvm::dyn_cast< clang::MemberExpr >(expr)) {
                member->setBase(WrapIfBelow(ctx, member->getBase(), kPostfix));
                return;
            }
            if (auto *call = llvm::dyn_cast< clang::CallExpr >(expr)) {
                call->setCallee(WrapIfBelow(ctx, call->getCallee(), kPostfix));
                for (unsigned i = 0; i < call->getNumArgs(); ++i) {
                    // A comma expression as an argument would split the list.
                    call->setArg(i, WrapIfBelow(ctx, call->getArg(i), kAssignment));
                }
                return;
            }
            if (auto *init = llvm::dyn_cast< clang::InitListExpr >(expr)) {
                for (unsigned i = 0; i < init->getNumInits(); ++i) {
                    if (auto *elem = init->getInit(i)) {
                        init->setInit(i, WrapIfBelow(ctx, elem, kAssignment));
                    }
                }
                return;
            }
        }

        // ------------------------------------------------------------------
        // Literal printing
        // ------------------------------------------------------------------

        class CPrinterHelper : public clang::PrinterHelper
        {
          public:
            explicit CPrinterHelper(clang::ASTContext &ctx) : ctx_(ctx) {}

            bool handledStmt(clang::Stmt *stmt, llvm::raw_ostream &os) override {
                if (auto *lit = llvm::dyn_cast< clang::IntegerLiteral >(stmt)) {
                    return PrintNarrowInt(lit, os);
                }
                if (auto *lit = llvm::dyn_cast< clang::FloatingLiteral >(stmt)) {
                    return PrintNonFinite(lit, os);
                }
                if (auto *cast = llvm::dyn_cast< clang::ImplicitCastExpr >(stmt)) {
                    return PrintConversion(cast, os);
                }
                return false;
            }

          private:
            bool PrintConversion(clang::ImplicitCastExpr *cast, llvm::raw_ostream &os) {
                if (!IsPrintedConversion(cast)) { return false; }
                os << "(" << cast->getType().getAsString(ctx_.getPrintingPolicy()) << ")";
                cast->getSubExpr()->printPretty(
                    os, this, ctx_.getPrintingPolicy(), 0, "\n", &ctx_
                );
                return true;
            }

            // StmtPrinter spells char/short literals with the MS-only
            // i8/Ui8/i16/Ui16 suffixes.  The value fits in int, so a plain
            // decimal literal keeps its meaning.
            static bool PrintNarrowInt(clang::IntegerLiteral *lit, llvm::raw_ostream &os) {
                const auto *builtin = lit->getType()->getAs< clang::BuiltinType >();
                if (builtin == nullptr) { return false; }
                switch (builtin->getKind()) {
                    case clang::BuiltinType::Char_S:
                    case clang::BuiltinType::Char_U:
                    case clang::BuiltinType::SChar:
                    case clang::BuiltinType::UChar:
                    case clang::BuiltinType::Short:
                    case clang::BuiltinType::UShort:
                        break;
                    default:
                        return false;
                }
                llvm::SmallString< 24 > buffer;
                lit->getValue().toString(buffer, 10, builtin->isSignedInteger());
                os << buffer;
                return true;
            }

            // APFloat spells non-finite values as `NaN`/`inf`, which are not
            // C literals.  Spell them as expressions rather than builtins so
            // they parse under any frontend configuration.
            static bool PrintNonFinite(clang::FloatingLiteral *lit, llvm::raw_ostream &os) {
                const auto &value = lit->getValue();
                if (value.isFinite()) { return false; }
                llvm::StringRef suffix;
                if (const auto *builtin = lit->getType()->getAs< clang::BuiltinType >()) {
                    if (builtin->getKind() == clang::BuiltinType::Float) {
                        suffix = "f";
                    } else if (builtin->getKind() == clang::BuiltinType::LongDouble) {
                        suffix = "L";
                    }
                }
                if (value.isNaN()) {
                    os << "(0.0" << suffix << " / 0.0" << suffix << ")";
                } else {
                    os << "(" << (value.isNegative() ? "-1.0" : "1.0") << suffix << " / 0.0"
                       << suffix << ")";
                }
                return true;
            }

            clang::ASTContext &ctx_;
        };

        // ------------------------------------------------------------------
        // Declaration ordering
        // ------------------------------------------------------------------

        // The declaration that defines `tag`, or null when it is only ever
        // forward-declared.
        clang::Decl *DefinitionOf(const clang::TagDecl *tag) {
            if (const auto *record = llvm::dyn_cast< clang::RecordDecl >(tag)) {
                return record->getDefinition();
            }
            if (const auto *enumeration = llvm::dyn_cast< clang::EnumDecl >(tag)) {
                return enumeration->getDefinition();
            }
            return nullptr;
        }

        // Type declarations that must precede a use of `type`.  A typedef must
        // always be declared first; a record needs its definition only when
        // used by value (fields, array elements); an enum has no forward
        // declaration in C, so its definition is always required.
        void CollectTypeDeps(
            clang::QualType type, bool need_complete,
            llvm::SmallVectorImpl< clang::Decl * > &out
        ) {
            const clang::Type *ty = type.getTypePtrOrNull();
            if (ty == nullptr) { return; }
            if (const auto *typedef_type = llvm::dyn_cast< clang::TypedefType >(ty)) {
                auto *decl = typedef_type->getDecl();
                out.push_back(decl);
                if (need_complete) { CollectTypeDeps(decl->getUnderlyingType(), true, out); }
                return;
            }
            if (const auto *pointer = ty->getAs< clang::PointerType >()) {
                CollectTypeDeps(pointer->getPointeeType(), false, out);
                return;
            }
            if (const auto *array = ty->getAsArrayTypeUnsafe()) {
                CollectTypeDeps(array->getElementType(), true, out);
                return;
            }
            if (const auto *function = ty->getAs< clang::FunctionType >()) {
                CollectTypeDeps(function->getReturnType(), false, out);
                if (const auto *proto = llvm::dyn_cast< clang::FunctionProtoType >(function)) {
                    for (auto param : proto->getParamTypes()) {
                        CollectTypeDeps(param, false, out);
                    }
                }
                return;
            }
            if (const auto *tag = ty->getAsTagDecl()) {
                if (need_complete || llvm::isa< clang::EnumDecl >(tag)) {
                    if (auto *definition = DefinitionOf(tag)) { out.push_back(definition); }
                }
            }
        }

        // Depth-first post-order over the "must be declared before" edges,
        // restricted to the type declarations of this translation unit.
        class TypeOrderer
        {
          public:
            explicit TypeOrderer(const std::unordered_set< const clang::Decl * > &candidates)
                : candidates_(candidates) {}

            void Visit(clang::Decl *decl, std::vector< clang::Decl * > &out) {
                if (!candidates_.contains(decl) || done_.contains(decl)) { return; }
                if (!active_.insert(decl).second) {
                    return; // cycle through by-value uses: invalid C, leave order as-is
                }
                llvm::SmallVector< clang::Decl *, 8 > deps;
                if (auto *typedef_decl = llvm::dyn_cast< clang::TypedefDecl >(decl)) {
                    CollectTypeDeps(typedef_decl->getUnderlyingType(), false, deps);
                } else if (auto *record = llvm::dyn_cast< clang::RecordDecl >(decl)) {
                    for (auto *field : record->fields()) {
                        CollectTypeDeps(field->getType(), true, deps);
                    }
                }
                for (auto *dep : deps) { Visit(dep, out); }
                active_.erase(decl);
                done_.insert(decl);
                out.push_back(decl);
            }

          private:
            const std::unordered_set< const clang::Decl * > &candidates_;
            std::unordered_set< const clang::Decl * > done_;
            std::unordered_set< const clang::Decl * > active_;
        };

        struct TUSections
        {
            std::vector< clang::TagDecl * > tags; // forward declarations
            std::vector< clang::Decl * > types;   // definitions, dependency order
            std::vector< clang::Decl * > others;  // globals and anything else
            std::vector< clang::FunctionDecl * > prototypes;
            std::vector< clang::FunctionDecl * > definitions;
        };

        TUSections CollectSections(clang::ASTContext &ctx) {
            TUSections sections;
            std::unordered_set< const clang::Decl * > candidates;
            std::unordered_set< const clang::Decl * > seen_tags;
            std::unordered_set< const clang::Decl * > seen_prototypes;
            std::vector< clang::Decl * > type_decls;

            for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                if (decl->isImplicit()) {
                    // Library builtins the lifter declared (ceilf, memcpy) need a
                    // prototype to re-parse; compiler builtins (__builtin_*) do not.
                    auto *function = llvm::dyn_cast< clang::FunctionDecl >(decl);
                    if (function == nullptr || function->getIdentifier() == nullptr
                        || function->getName().starts_with("__"))
                    {
                        continue;
                    }
                    if (seen_prototypes.insert(function->getCanonicalDecl()).second) {
                        sections.prototypes.push_back(function);
                    }
                    continue;
                }
                if (auto *tag = llvm::dyn_cast< clang::TagDecl >(decl)) {
                    if (llvm::isa< clang::RecordDecl >(tag)
                        && seen_tags.insert(tag->getCanonicalDecl()).second)
                    {
                        sections.tags.push_back(tag);
                    }
                    if (tag->isCompleteDefinition()) {
                        candidates.insert(decl);
                        type_decls.push_back(decl);
                    }
                    continue;
                }
                if (llvm::isa< clang::TypedefDecl >(decl)) {
                    candidates.insert(decl);
                    type_decls.push_back(decl);
                    continue;
                }
                if (auto *function = llvm::dyn_cast< clang::FunctionDecl >(decl)) {
                    if (function->doesThisDeclarationHaveABody()) {
                        sections.definitions.push_back(function);
                    } else if (seen_prototypes.insert(function->getCanonicalDecl()).second) {
                        sections.prototypes.push_back(function);
                    }
                    continue;
                }
                sections.others.push_back(decl);
            }

            TypeOrderer orderer(candidates);
            for (auto *decl : type_decls) { orderer.Visit(decl, sections.types); }
            return sections;
        }

        std::string SymbolOf(const clang::FunctionDecl *function) {
            if (const auto *label = function->getAttr< clang::AsmLabelAttr >()) {
                return label->getLabel().str();
            }
            return function->getNameAsString();
        }

        std::string EscapeComment(const std::string &text) {
            std::string out = text;
            for (auto pos = out.find("*/"); pos != std::string::npos; pos = out.find("*/", pos))
            {
                out.replace(pos, 2, "* /");
                pos += 3;
            }
            return out;
        }

    } // namespace

    void ReparenthesizeForPrint(clang::ASTContext &ctx, clang::Stmt *body) {
        Reparenthesize(ctx, body);
    }

    void PrintTranslationUnit(
        llvm::raw_ostream &os, clang::ASTContext &ctx, const DefinitionMap &defs,
        const TUPrintOptions &opts
    ) {
        const auto &policy = ctx.getPrintingPolicy();
        auto terse         = policy;
        terse.TerseOutput  = true;
        CPrinterHelper helper(ctx);

        if (opts.emit_markers) {
            os << "// patchestry:tu format=1";
            if (!opts.lang_id.empty()) { os << " target=" << opts.lang_id; }
            if (!opts.arch.empty()) { os << " arch=" << opts.arch; }
            os << "\n";
        }

        auto sections = CollectSections(ctx);
        for (auto *tag : sections.tags) {
            if (tag->getName().empty()) {
                continue; // anonymous tags cannot be forward-declared
            }
            os << tag->getKindName() << " " << tag->getName() << ";\n";
        }
        for (auto *decl : sections.types) {
            decl->print(os, policy);
            os << ";\n";
        }
        for (auto *decl : sections.others) {
            decl->print(os, policy);
            os << ";\n";
        }
        for (auto *function : sections.prototypes) {
            function->print(os, policy);
            os << ";\n";
        }
        for (auto *function : sections.definitions) {
            const ghidra::Function *lifted = nullptr;
            if (auto iter = defs.find(function); iter != defs.end()) { lifted = iter->second; }
            const std::string key = lifted != nullptr ? lifted->key : std::string("-");
            if (opts.emit_markers) {
                os << "// patchestry:function-begin " << key
                   << " name=" << function->getNameAsString()
                   << " symbol=" << SymbolOf(function) << "\n";
            }
            if (lifted != nullptr && !lifted->comment.empty()) {
                os << "/* " << EscapeComment(lifted->comment) << " */\n";
            }
            function->print(os, terse);
            function->getBody()->printPrettyControlled(os, &helper, policy, 0, "\n", &ctx);
            if (opts.emit_markers) { os << "// patchestry:function-end " << key << "\n"; }
        }
    }

} // namespace patchestry::ast
