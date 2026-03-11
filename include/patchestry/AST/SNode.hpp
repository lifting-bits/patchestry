/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>

#include <llvm/Support/Allocator.h>
#include <llvm/Support/raw_ostream.h>

namespace patchestry::ast {

    enum class SNodeKind {
        SEQ,
        BLOCK,
        IF_THEN_ELSE,
        WHILE,
        DO_WHILE,
        FOR,
        SWITCH,
        GOTO,
        LABEL,
        BREAK,
        CONTINUE,
        RETURN,
    };

    class SNode
    {
      public:
        SNode(SNodeKind kind) : kind_(kind) {}
        virtual ~SNode() = default;

        SNodeKind Kind() const { return kind_; }
        static const char *KindName(SNodeKind k);
        const char *KindName() const { return KindName(kind_); }

        SNode *Parent() const { return parent_; }
        void SetParent(SNode *p) { parent_ = p; }

        void Dump(llvm::raw_ostream &os, unsigned indent = 0) const;

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
        virtual void DumpChildren(llvm::raw_ostream & /*os*/, unsigned /*indent*/) const {}

      private:
        SNodeKind kind_;
        SNode *parent_ = nullptr;
    };

    // Sequence of SNodes (analogous to CompoundStmt)
    class SSeq : public SNode
    {
      public:
        SSeq() : SNode(SNodeKind::SEQ) {}

        const std::vector< SNode * > &Children() const { return children_; }
        std::vector< SNode * > &Children() { return children_; }

        void AddChild(SNode *child) {
            child->SetParent(this);
            children_.push_back(child);
        }

        void InsertChild(size_t pos, SNode *child) {
            child->SetParent(this);
            children_.insert(children_.begin() + static_cast< ptrdiff_t >(pos), child);
        }

        void RemoveChild(size_t pos) {
            children_.erase(children_.begin() + static_cast< ptrdiff_t >(pos));
        }

        void ReplaceChild(size_t pos, SNode *child) {
            child->SetParent(this);
            children_[pos] = child;
        }

        // Replace a range [from, to) with a single node
        void ReplaceRange(size_t from, size_t to, SNode *replacement) {
            replacement->SetParent(this);
            auto begin = children_.begin();
            children_.erase(begin + static_cast< ptrdiff_t >(from) + 1,
                            begin + static_cast< ptrdiff_t >(to));
            children_[from] = replacement;
        }

        // Replace a range [from, to) with multiple nodes
        void ReplaceRange(size_t from, size_t to,
                           const std::vector< SNode * > &replacements) {
            for (auto *r : replacements) r->SetParent(this);
            auto begin = children_.begin();
            children_.erase(begin + static_cast< ptrdiff_t >(from),
                            begin + static_cast< ptrdiff_t >(to));
            children_.insert(children_.begin() + static_cast< ptrdiff_t >(from),
                             replacements.begin(), replacements.end());
        }

        size_t Size() const { return children_.size(); }
        bool Empty() const { return children_.empty(); }
        SNode *operator[](size_t i) { return children_[i]; }
        const SNode *operator[](size_t i) const { return children_[i]; }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::SEQ; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        std::vector< SNode * > children_;
    };

    // Basic block: holds raw Clang Stmt* and an optional label
    class SBlock : public SNode
    {
      public:
        SBlock() : SNode(SNodeKind::BLOCK) {}

        std::string_view Label() const { return label_; }
        void SetLabel(std::string_view l) { label_ = l; }

        const std::vector< clang::Stmt * > &Stmts() const { return stmts_; }
        std::vector< clang::Stmt * > &Stmts() { return stmts_; }

        void AddStmt(clang::Stmt *s) { stmts_.push_back(s); }
        bool Empty() const { return stmts_.empty(); }
        size_t Size() const { return stmts_.size(); }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::BLOCK; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        std::string label_;
        std::vector< clang::Stmt * > stmts_;
    };

    // If-then-else
    class SIfThenElse : public SNode
    {
      public:
        SIfThenElse(clang::Expr *cond, SNode *then_branch, SNode *else_branch = nullptr)
            : SNode(SNodeKind::IF_THEN_ELSE)
            , cond_(cond), then_(then_branch), else_(else_branch)
        {
            if (then_) then_->SetParent(this);
            if (else_) else_->SetParent(this);
        }

        clang::Expr *Cond() const { return cond_; }
        void SetCond(clang::Expr *c) { cond_ = c; }

        SNode *ThenBranch() const { return then_; }
        void SetThenBranch(SNode *n) { then_ = n; if (n) n->SetParent(this); }

        SNode *ElseBranch() const { return else_; }
        void SetElseBranch(SNode *n) { else_ = n; if (n) n->SetParent(this); }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::IF_THEN_ELSE; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

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
            : SNode(SNodeKind::WHILE), cond_(cond), body_(body)
        {
            if (body_) body_->SetParent(this);
        }

        clang::Expr *Cond() const { return cond_; }
        void SetCond(clang::Expr *c) { cond_ = c; }

        SNode *Body() const { return body_; }
        void SetBody(SNode *n) { body_ = n; if (n) n->SetParent(this); }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::WHILE; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        clang::Expr *cond_;
        SNode *body_;
    };

    // Do-while loop
    class SDoWhile : public SNode
    {
      public:
        SDoWhile(SNode *body, clang::Expr *cond)
            : SNode(SNodeKind::DO_WHILE), body_(body), cond_(cond)
        {
            if (body_) body_->SetParent(this);
        }

        SNode *Body() const { return body_; }
        void SetBody(SNode *n) { body_ = n; if (n) n->SetParent(this); }

        clang::Expr *Cond() const { return cond_; }
        void SetCond(clang::Expr *c) { cond_ = c; }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::DO_WHILE; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        SNode *body_;
        clang::Expr *cond_;
    };

    // For loop
    class SFor : public SNode
    {
      public:
        SFor(clang::Stmt *init, clang::Expr *cond, clang::Expr *inc, SNode *body)
            : SNode(SNodeKind::FOR)
            , init_(init), cond_(cond), inc_(inc), body_(body)
        {
            if (body_) body_->SetParent(this);
        }

        clang::Stmt *Init() const { return init_; }
        void SetInit(clang::Stmt *s) { init_ = s; }

        clang::Expr *Cond() const { return cond_; }
        void SetCond(clang::Expr *c) { cond_ = c; }

        clang::Expr *Inc() const { return inc_; }
        void SetInc(clang::Expr *e) { inc_ = e; }

        SNode *Body() const { return body_; }
        void SetBody(SNode *n) { body_ = n; if (n) n->SetParent(this); }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::FOR; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

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
            : SNode(SNodeKind::SWITCH), discriminant_(discriminant)
        {}

        clang::Expr *Discriminant() const { return discriminant_; }
        void SetDiscriminant(clang::Expr *e) { discriminant_ = e; }

        const std::vector< SCase > &Cases() const { return cases_; }
        std::vector< SCase > &Cases() { return cases_; }

        void AddCase(clang::Expr *value, SNode *body) {
            if (body) body->SetParent(this);
            cases_.push_back({value, body});
        }

        SNode *DefaultBody() const { return default_; }
        void SetDefaultBody(SNode *n) { default_ = n; if (n) n->SetParent(this); }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::SWITCH; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        clang::Expr *discriminant_;
        std::vector< SCase > cases_;
        SNode *default_ = nullptr;
    };

    // Goto
    class SGoto : public SNode
    {
      public:
        SGoto(std::string_view target) : SNode(SNodeKind::GOTO), target_(target) {}

        std::string_view Target() const { return target_; }
        void SetTarget(std::string_view t) { target_ = t; }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::GOTO; }

      private:
        std::string target_;
    };

    // Label
    class SLabel : public SNode
    {
      public:
        SLabel(std::string_view name, SNode *body = nullptr)
            : SNode(SNodeKind::LABEL), name_(name), body_(body)
        {
            if (body_) body_->SetParent(this);
        }

        std::string_view Name() const { return name_; }
        void SetName(std::string_view n) { name_ = n; }

        SNode *Body() const { return body_; }
        void SetBody(SNode *n) { body_ = n; if (n) n->SetParent(this); }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::LABEL; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        std::string name_;
        SNode *body_;
    };

    // Break (with optional depth for multi-level breaks)
    class SBreak : public SNode
    {
      public:
        SBreak(unsigned depth = 1) : SNode(SNodeKind::BREAK), depth_(depth) {}

        unsigned Depth() const { return depth_; }
        void SetDepth(unsigned d) { depth_ = d; }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::BREAK; }

      private:
        unsigned depth_;
    };

    // Continue
    class SContinue : public SNode
    {
      public:
        SContinue() : SNode(SNodeKind::CONTINUE) {}

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::CONTINUE; }
    };

    // Return
    class SReturn : public SNode
    {
      public:
        SReturn(clang::Expr *value = nullptr)
            : SNode(SNodeKind::RETURN), value_(value)
        {}

        clang::Expr *Value() const { return value_; }
        void SetValue(clang::Expr *e) { value_ = e; }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::RETURN; }

      private:
        clang::Expr *value_;
    };

    // Owning factory for SNodes.
    //
    // SNode subclasses hold std:: members (std::string, std::vector) whose heap
    // allocations must be released via their destructors.  A raw BumpPtrAllocator
    // would reclaim the slab memory without ever calling destructors, leaking every
    // such sub-allocation.
    //
    // Nodes are therefore owned through std::unique_ptr so that Reset() and the
    // factory destructor both invoke the full virtual destructor chain.  The bump
    // allocator is kept only for Intern(), whose raw char bytes carry no destructors.
    class SNodeFactory
    {
      public:
        template< typename T, typename... Args >
        T *Make(Args &&...args) {
            static_assert(
                std::is_base_of_v< SNode, T >,
                "SNodeFactory::Make may only create SNode subclasses"
            );
            auto node = std::make_unique< T >(std::forward< Args >(args)...);
            T *ptr    = node.get();
            nodes_.push_back(std::move(node));
            return ptr;
        }

        // Intern a copy of a string; the returned view is valid until Reset().
        // Uses a bump allocator because raw char data carries no destructor.
        std::string_view Intern(std::string_view s) {
            if (s.empty()) {
                return {};
            }
            char *buf = static_cast< char * >(string_alloc_.Allocate(s.size(), 1));
            std::memcpy(buf, s.data(), s.size());
            return std::string_view(buf, s.size());
        }

        // Destroy all nodes and release interned string memory.
        void Reset() {
            nodes_.clear(); // invokes virtual destructor on every node
            string_alloc_.Reset();
        }

        size_t NodeCount() const { return nodes_.size(); }

      private:
        // Owns all allocated SNodes; clear() triggers the full destructor chain.
        std::vector< std::unique_ptr< SNode > > nodes_;
        // Raw slab for interned string bytes only — no destructors needed.
        llvm::BumpPtrAllocator string_alloc_;
    };

} // namespace patchestry::ast
