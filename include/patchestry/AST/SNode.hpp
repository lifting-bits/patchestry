/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <memory>
#include <string_view>
#include <vector>

#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>

#include <llvm/Support/Allocator.h>
#include <llvm/Support/raw_ostream.h>

namespace patchestry::ast {

    enum class SNodeKind {
        Seq,
        Block,
        IfThenElse,
        While,
        DoWhile,
        For,
        Switch,
        Goto,
        Label,
        Break,
        Continue,
        Return,
    };

    class SNode
    {
      public:
        SNode(SNodeKind kind) : kind_(kind) {}
        virtual ~SNode() = default;

        SNodeKind kind() const { return kind_; }
        static const char *kindName(SNodeKind k);
        const char *kindName() const { return kindName(kind_); }

        SNode *parent() const { return parent_; }
        void setParent(SNode *p) { parent_ = p; }

        void dump(llvm::raw_ostream &os, unsigned indent = 0) const;

        template< typename T >
        bool isa() const { return T::classof(this); }
        template< typename T >
        T *as() { assert(isa< T >()); return static_cast< T * >(this); }
        template< typename T >
        const T *as() const { assert(isa< T >()); return static_cast< const T * >(this); }
        template< typename T >
        T *dyn_cast() { return isa< T >() ? static_cast< T * >(this) : nullptr; }
        template< typename T >
        const T *dyn_cast() const { return isa< T >() ? static_cast< const T * >(this) : nullptr; }

      protected:
        virtual void dumpChildren(llvm::raw_ostream & /*os*/, unsigned /*indent*/) const {}

      private:
        SNodeKind kind_;
        SNode *parent_ = nullptr;
    };

    // Sequence of SNodes (analogous to CompoundStmt)
    class SSeq : public SNode
    {
      public:
        SSeq() : SNode(SNodeKind::Seq) {}

        const std::vector< SNode * > &children() const { return children_; }
        std::vector< SNode * > &children() { return children_; }

        void addChild(SNode *child) {
            child->setParent(this);
            children_.push_back(child);
        }

        void insertChild(size_t pos, SNode *child) {
            child->setParent(this);
            children_.insert(children_.begin() + static_cast< ptrdiff_t >(pos), child);
        }

        void removeChild(size_t pos) {
            children_.erase(children_.begin() + static_cast< ptrdiff_t >(pos));
        }

        void replaceChild(size_t pos, SNode *child) {
            child->setParent(this);
            children_[pos] = child;
        }

        // Replace a range [from, to) with a single node
        void replaceRange(size_t from, size_t to, SNode *replacement) {
            replacement->setParent(this);
            auto begin = children_.begin();
            children_.erase(begin + static_cast< ptrdiff_t >(from) + 1,
                            begin + static_cast< ptrdiff_t >(to));
            children_[from] = replacement;
        }

        // Replace a range [from, to) with multiple nodes
        void replaceRange(size_t from, size_t to,
                          const std::vector< SNode * > &replacements) {
            for (auto *r : replacements) r->setParent(this);
            auto begin = children_.begin();
            children_.erase(begin + static_cast< ptrdiff_t >(from),
                            begin + static_cast< ptrdiff_t >(to));
            children_.insert(children_.begin() + static_cast< ptrdiff_t >(from),
                             replacements.begin(), replacements.end());
        }

        size_t size() const { return children_.size(); }
        bool empty() const { return children_.empty(); }
        SNode *operator[](size_t i) { return children_[i]; }
        const SNode *operator[](size_t i) const { return children_[i]; }

        static bool classof(const SNode *n) { return n->kind() == SNodeKind::Seq; }

      protected:
        void dumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        std::vector< SNode * > children_;
    };

    // Basic block: holds raw Clang Stmt* and an optional label
    class SBlock : public SNode
    {
      public:
        SBlock() : SNode(SNodeKind::Block) {}

        std::string_view label() const { return label_; }
        void setLabel(std::string_view l) { label_ = l; }

        const std::vector< clang::Stmt * > &stmts() const { return stmts_; }
        std::vector< clang::Stmt * > &stmts() { return stmts_; }

        void addStmt(clang::Stmt *s) { stmts_.push_back(s); }
        bool empty() const { return stmts_.empty(); }
        size_t size() const { return stmts_.size(); }

        static bool classof(const SNode *n) { return n->kind() == SNodeKind::Block; }

      protected:
        void dumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        std::string label_;
        std::vector< clang::Stmt * > stmts_;
    };

    // If-then-else
    class SIfThenElse : public SNode
    {
      public:
        SIfThenElse(clang::Expr *cond, SNode *then_branch, SNode *else_branch = nullptr)
            : SNode(SNodeKind::IfThenElse)
            , cond_(cond), then_(then_branch), else_(else_branch)
        {
            if (then_) then_->setParent(this);
            if (else_) else_->setParent(this);
        }

        clang::Expr *cond() const { return cond_; }
        void setCond(clang::Expr *c) { cond_ = c; }

        SNode *thenBranch() const { return then_; }
        void setThenBranch(SNode *n) { then_ = n; if (n) n->setParent(this); }

        SNode *elseBranch() const { return else_; }
        void setElseBranch(SNode *n) { else_ = n; if (n) n->setParent(this); }

        static bool classof(const SNode *n) { return n->kind() == SNodeKind::IfThenElse; }

      protected:
        void dumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        clang::Expr *cond_;
        SNode *then_;
        SNode *else_;
    };

    // While loop
    class SWhile : public SNode
    {
      public:
        SWhile(clang::Expr *cond, SNode *body)
            : SNode(SNodeKind::While), cond_(cond), body_(body)
        {
            if (body_) body_->setParent(this);
        }

        clang::Expr *cond() const { return cond_; }
        void setCond(clang::Expr *c) { cond_ = c; }

        SNode *body() const { return body_; }
        void setBody(SNode *n) { body_ = n; if (n) n->setParent(this); }

        static bool classof(const SNode *n) { return n->kind() == SNodeKind::While; }

      protected:
        void dumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        clang::Expr *cond_;
        SNode *body_;
    };

    // Do-while loop
    class SDoWhile : public SNode
    {
      public:
        SDoWhile(SNode *body, clang::Expr *cond)
            : SNode(SNodeKind::DoWhile), body_(body), cond_(cond)
        {
            if (body_) body_->setParent(this);
        }

        SNode *body() const { return body_; }
        void setBody(SNode *n) { body_ = n; if (n) n->setParent(this); }

        clang::Expr *cond() const { return cond_; }
        void setCond(clang::Expr *c) { cond_ = c; }

        static bool classof(const SNode *n) { return n->kind() == SNodeKind::DoWhile; }

      protected:
        void dumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        SNode *body_;
        clang::Expr *cond_;
    };

    // For loop
    class SFor : public SNode
    {
      public:
        SFor(clang::Stmt *init, clang::Expr *cond, clang::Expr *inc, SNode *body)
            : SNode(SNodeKind::For)
            , init_(init), cond_(cond), inc_(inc), body_(body)
        {
            if (body_) body_->setParent(this);
        }

        clang::Stmt *init() const { return init_; }
        void setInit(clang::Stmt *s) { init_ = s; }

        clang::Expr *cond() const { return cond_; }
        void setCond(clang::Expr *c) { cond_ = c; }

        clang::Expr *inc() const { return inc_; }
        void setInc(clang::Expr *e) { inc_ = e; }

        SNode *body() const { return body_; }
        void setBody(SNode *n) { body_ = n; if (n) n->setParent(this); }

        static bool classof(const SNode *n) { return n->kind() == SNodeKind::For; }

      protected:
        void dumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        clang::Stmt *init_;
        clang::Expr *cond_;
        clang::Expr *inc_;
        SNode *body_;
    };

    // Switch case
    struct SCase {
        clang::Expr *value;  // nullptr for default
        SNode *body;
    };

    // Switch statement
    class SSwitch : public SNode
    {
      public:
        SSwitch(clang::Expr *discriminant)
            : SNode(SNodeKind::Switch), discriminant_(discriminant)
        {}

        clang::Expr *discriminant() const { return discriminant_; }
        void setDiscriminant(clang::Expr *e) { discriminant_ = e; }

        const std::vector< SCase > &cases() const { return cases_; }
        std::vector< SCase > &cases() { return cases_; }

        void addCase(clang::Expr *value, SNode *body) {
            if (body) body->setParent(this);
            cases_.push_back({value, body});
        }

        SNode *defaultBody() const { return default_; }
        void setDefaultBody(SNode *n) { default_ = n; if (n) n->setParent(this); }

        static bool classof(const SNode *n) { return n->kind() == SNodeKind::Switch; }

      protected:
        void dumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        clang::Expr *discriminant_;
        std::vector< SCase > cases_;
        SNode *default_ = nullptr;
    };

    // Goto
    class SGoto : public SNode
    {
      public:
        SGoto(std::string_view target) : SNode(SNodeKind::Goto), target_(target) {}

        std::string_view target() const { return target_; }
        void setTarget(std::string_view t) { target_ = t; }

        static bool classof(const SNode *n) { return n->kind() == SNodeKind::Goto; }

      private:
        std::string target_;
    };

    // Label
    class SLabel : public SNode
    {
      public:
        SLabel(std::string_view name, SNode *body = nullptr)
            : SNode(SNodeKind::Label), name_(name), body_(body)
        {
            if (body_) body_->setParent(this);
        }

        std::string_view name() const { return name_; }
        void setName(std::string_view n) { name_ = n; }

        SNode *body() const { return body_; }
        void setBody(SNode *n) { body_ = n; if (n) n->setParent(this); }

        static bool classof(const SNode *n) { return n->kind() == SNodeKind::Label; }

      protected:
        void dumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        std::string name_;
        SNode *body_;
    };

    // Break (with optional depth for multi-level breaks)
    class SBreak : public SNode
    {
      public:
        SBreak(unsigned depth = 1) : SNode(SNodeKind::Break), depth_(depth) {}

        unsigned depth() const { return depth_; }
        void setDepth(unsigned d) { depth_ = d; }

        static bool classof(const SNode *n) { return n->kind() == SNodeKind::Break; }

      private:
        unsigned depth_;
    };

    // Continue
    class SContinue : public SNode
    {
      public:
        SContinue() : SNode(SNodeKind::Continue) {}

        static bool classof(const SNode *n) { return n->kind() == SNodeKind::Continue; }
    };

    // Return
    class SReturn : public SNode
    {
      public:
        SReturn(clang::Expr *value = nullptr)
            : SNode(SNodeKind::Return), value_(value)
        {}

        clang::Expr *value() const { return value_; }
        void setValue(clang::Expr *e) { value_ = e; }

        static bool classof(const SNode *n) { return n->kind() == SNodeKind::Return; }

      private:
        clang::Expr *value_;
    };

    // Arena allocator factory for SNodes
    class SNodeFactory
    {
      public:
        template< typename T, typename... Args >
        T *make(Args &&...args) {
            void *mem = allocator_.Allocate(sizeof(T), alignof(T));
            return new (mem) T(std::forward< Args >(args)...);
        }

        // Allocate a copy of a string that lives as long as the factory
        std::string_view intern(std::string_view s) {
            char *buf = static_cast< char * >(allocator_.Allocate(s.size(), 1));
            std::memcpy(buf, s.data(), s.size());
            return std::string_view(buf, s.size());
        }

        void reset() { allocator_.Reset(); }

      private:
        llvm::BumpPtrAllocator allocator_;
    };

} // namespace patchestry::ast
