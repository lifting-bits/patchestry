/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <cstring>
#include <functional>
#include <initializer_list>
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

#include <patchestry/AST/SourceOrigin.hpp>

namespace patchestry::ast {

    enum class SNodeKind {
        kStmt,
        kIfThenElse,
        kWhile,
        kDoWhile,
        kFor,
        kSwitch,
        kGoto,
        kLabel,
        kBreak,
        kContinue,
        kReturn,
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

        const std::vector< StmtOrigin > &Origins() const { return origins_; }

        std::vector< StmtOrigin > &Origins() { return origins_; }

        bool HasOrigins() const { return !origins_.empty(); }

        void SetOrigins(std::vector< StmtOrigin > origins) { origins_ = std::move(origins); }

        void ClearOrigins() { origins_.clear(); }

        void AddOrigin(StmtOrigin origin) { origins_.push_back(std::move(origin)); }

        // ChildVisitor API.
        // Each subtype iterates over its immediate SNode children
        // (slots that hold an SNode* — not raw clang::Stmt members
        // like SStmt::Stmt or SFor::Init/Cond/Inc).
        //
        // Used by uniform recursion in cleanup walks.  Read-only form
        // takes SNode*; mutable form takes SNode*& so callers can
        // reassign the slot.
        //
        // Default: no children.  Each subtype with children overrides.
        using ChildFn    = std::function< void(SNode *) >;
        using ChildMutFn = std::function< void(SNode *&) >;

        virtual void for_each_child(const ChildFn &) const {}

        virtual void for_each_child_mut(const ChildMutFn &) {}

        void Dump(llvm::raw_ostream &os, unsigned indent = 0) const;

        template< typename T >
        bool isa() const {
            return T::classof(this);
        }

        template< typename T >
        T *as() {
            assert(isa< T >());
            return static_cast< T * >(this);
        }

        template< typename T >
        const T *as() const {
            assert(isa< T >());
            return static_cast< const T * >(this);
        }

        template< typename T >
        T *dyn_cast() {
            return isa< T >() ? static_cast< T * >(this) : nullptr;
        }

        template< typename T >
        const T *dyn_cast() const {
            return isa< T >() ? static_cast< const T * >(this) : nullptr;
        }

      protected:
        virtual void DumpChildren(llvm::raw_ostream & /*os*/, unsigned /*indent*/) const {}

      private:
        SNodeKind kind_;
        SNode *parent_ = nullptr;
        std::vector< StmtOrigin > origins_;
    };

    // Single-statement leaf — holds one raw clang::Stmt*.  The
    // statement-level counterpart to the control-flow SNode kinds:
    // a "block" of N statements is represented as N SStmt siblings
    // in a body vector.
    class SStmt : public SNode
    {
      public:
        explicit SStmt(clang::Stmt *stmt) : SNode(SNodeKind::kStmt), stmt_(stmt) {}

        clang::Stmt *Stmt() const { return stmt_; }

        void SetStmt(clang::Stmt *s) {
            stmt_ = s;
            ClearOrigins();
        }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::kStmt; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        clang::Stmt *stmt_;
    };

    // If-then-else — then_/else_ are std::vector<SNode*>.  See SLabel
    // docstring for the back-compat API rationale.
    class SIfThenElse : public SNode
    {
      public:
        SIfThenElse(clang::Expr *cond, SNode *then_branch, SNode *else_branch = nullptr)
            : SNode(SNodeKind::kIfThenElse), cond_(cond) {
            if (then_branch) {
                then_.push_back(then_branch);
                then_branch->SetParent(this);
            }
            if (else_branch) {
                else_.push_back(else_branch);
                else_branch->SetParent(this);
            }
        }

        SIfThenElse(
            clang::Expr *cond, std::vector< SNode * > then_list,
            std::vector< SNode * > else_list
        )
            : SNode(SNodeKind::kIfThenElse)
            , cond_(cond)
            , then_(std::move(then_list))
            , else_(std::move(else_list)) {
            for (auto *c : then_) {
                if (c) { c->SetParent(this); }
            }
            for (auto *c : else_) {
                if (c) { c->SetParent(this); }
            }
        }

        clang::Expr *Cond() const { return cond_; }

        void SetCond(clang::Expr *c) { cond_ = c; }

        SNode *ThenBranch() const { return then_.empty() ? nullptr : then_[0]; }

        void SetThenBranch(SNode *n) {
            then_.clear();
            if (n) {
                then_.push_back(n);
                n->SetParent(this);
            }
        }

        void SetThenBranch(std::vector< SNode * > body) {
            then_ = std::move(body);
            for (auto *c : then_) {
                if (c) { c->SetParent(this); }
            }
        }

        SNode *ElseBranch() const { return else_.empty() ? nullptr : else_[0]; }

        void SetElseBranch(SNode *n) {
            else_.clear();
            if (n) {
                else_.push_back(n);
                n->SetParent(this);
            }
        }

        void SetElseBranch(std::vector< SNode * > body) {
            else_ = std::move(body);
            for (auto *c : else_) {
                if (c) { c->SetParent(this); }
            }
        }

        const std::vector< SNode * > &ThenList() const { return then_; }

        std::vector< SNode * > &ThenList() { return then_; }

        const std::vector< SNode * > &ElseList() const { return else_; }

        std::vector< SNode * > &ElseList() { return else_; }

        void for_each_child(const ChildFn &fn) const override {
            for (auto *c : then_) {
                if (c) { fn(c); }
            }
            for (auto *c : else_) {
                if (c) { fn(c); }
            }
        }

        void for_each_child_mut(const ChildMutFn &fn) override {
            for (auto &slot : then_) {
                if (slot) { fn(slot); }
            }
            for (auto &slot : else_) {
                if (slot) { fn(slot); }
            }
        }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::kIfThenElse; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        clang::Expr *cond_;
        std::vector< SNode * > then_;
        std::vector< SNode * > else_;
    };

    // While loop — body_ is std::vector<SNode*>.
    // See SLabel docstring for the back-compat API rationale.
    class SWhile : public SNode
    {
      public:
        SWhile(clang::Expr *cond, SNode *body) : SNode(SNodeKind::kWhile), cond_(cond) {
            if (body) {
                body_.push_back(body);
                body->SetParent(this);
            }
        }

        SWhile(clang::Expr *cond, std::vector< SNode * > body)
            : SNode(SNodeKind::kWhile), cond_(cond), body_(std::move(body)) {
            for (auto *c : body_) {
                if (c) { c->SetParent(this); }
            }
        }

        clang::Expr *Cond() const { return cond_; }

        void SetCond(clang::Expr *c) { cond_ = c; }

        SNode *Body() const { return body_.empty() ? nullptr : body_[0]; }

        void SetBody(SNode *n) {
            body_.clear();
            if (n) {
                body_.push_back(n);
                n->SetParent(this);
            }
        }

        const std::vector< SNode * > &BodyList() const { return body_; }

        std::vector< SNode * > &BodyList() { return body_; }

        // Scope labels for break/continue resolution (set by fold rules)
        std::string_view ExitLabel() const { return exit_label_; }

        void SetExitLabel(std::string_view l) { exit_label_ = l; }

        std::string_view HeaderLabel() const { return header_label_; }

        void SetHeaderLabel(std::string_view l) { header_label_ = l; }

        void for_each_child(const ChildFn &fn) const override {
            for (auto *c : body_) {
                if (c) { fn(c); }
            }
        }

        void for_each_child_mut(const ChildMutFn &fn) override {
            for (auto &slot : body_) {
                if (slot) { fn(slot); }
            }
        }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::kWhile; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        clang::Expr *cond_;
        std::vector< SNode * > body_;
        std::string_view exit_label_;
        std::string_view header_label_;
    };

    // Do-while loop — body_ is std::vector<SNode*>.
    class SDoWhile : public SNode
    {
      public:
        SDoWhile(SNode *body, clang::Expr *cond) : SNode(SNodeKind::kDoWhile), cond_(cond) {
            if (body) {
                body_.push_back(body);
                body->SetParent(this);
            }
        }

        SDoWhile(std::vector< SNode * > body, clang::Expr *cond)
            : SNode(SNodeKind::kDoWhile), cond_(cond), body_(std::move(body)) {
            for (auto *c : body_) {
                if (c) { c->SetParent(this); }
            }
        }

        SNode *Body() const { return body_.empty() ? nullptr : body_[0]; }

        void SetBody(SNode *n) {
            body_.clear();
            if (n) {
                body_.push_back(n);
                n->SetParent(this);
            }
        }

        const std::vector< SNode * > &BodyList() const { return body_; }

        std::vector< SNode * > &BodyList() { return body_; }

        clang::Expr *Cond() const { return cond_; }

        void SetCond(clang::Expr *c) { cond_ = c; }

        std::string_view ExitLabel() const { return exit_label_; }

        void SetExitLabel(std::string_view l) { exit_label_ = l; }

        std::string_view HeaderLabel() const { return header_label_; }

        void SetHeaderLabel(std::string_view l) { header_label_ = l; }

        void for_each_child(const ChildFn &fn) const override {
            for (auto *c : body_) {
                if (c) { fn(c); }
            }
        }

        void for_each_child_mut(const ChildMutFn &fn) override {
            for (auto &slot : body_) {
                if (slot) { fn(slot); }
            }
        }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::kDoWhile; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        clang::Expr *cond_;
        std::vector< SNode * > body_;
        std::string_view exit_label_;
        std::string_view header_label_;
    };

    // For loop — body_ is std::vector<SNode*>.
    class SFor : public SNode
    {
      public:
        SFor(clang::Stmt *init, clang::Expr *cond, clang::Expr *inc, SNode *body)
            : SNode(SNodeKind::kFor), init_(init), cond_(cond), inc_(inc) {
            if (body) {
                body_.push_back(body);
                body->SetParent(this);
            }
        }

        SFor(
            clang::Stmt *init, clang::Expr *cond, clang::Expr *inc, std::vector< SNode * > body
        )
            : SNode(SNodeKind::kFor)
            , init_(init)
            , cond_(cond)
            , inc_(inc)
            , body_(std::move(body)) {
            for (auto *c : body_) {
                if (c) { c->SetParent(this); }
            }
        }

        clang::Stmt *Init() const { return init_; }

        void SetInit(clang::Stmt *s) { init_ = s; }

        clang::Expr *Cond() const { return cond_; }

        void SetCond(clang::Expr *c) { cond_ = c; }

        clang::Expr *Inc() const { return inc_; }

        void SetInc(clang::Expr *e) { inc_ = e; }

        SNode *Body() const { return body_.empty() ? nullptr : body_[0]; }

        void SetBody(SNode *n) {
            body_.clear();
            if (n) {
                body_.push_back(n);
                n->SetParent(this);
            }
        }

        const std::vector< SNode * > &BodyList() const { return body_; }

        std::vector< SNode * > &BodyList() { return body_; }

        std::string_view ExitLabel() const { return exit_label_; }

        void SetExitLabel(std::string_view l) { exit_label_ = l; }

        std::string_view HeaderLabel() const { return header_label_; }

        void SetHeaderLabel(std::string_view l) { header_label_ = l; }

        void for_each_child(const ChildFn &fn) const override {
            for (auto *c : body_) {
                if (c) { fn(c); }
            }
        }

        void for_each_child_mut(const ChildMutFn &fn) override {
            for (auto &slot : body_) {
                if (slot) { fn(slot); }
            }
        }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::kFor; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        clang::Stmt *init_;
        clang::Expr *cond_;
        clang::Expr *inc_;
        std::vector< SNode * > body_;
        std::string_view exit_label_;
        std::string_view header_label_;
    };

    // Switch case — body is std::vector<SNode*>.
    //
    // Back-compat helpers preserve the prior field-style read pattern:
    //   c.body() returns the first element of body_list (or nullptr)
    //   c.set_body(n) clears body_list and pushes n (caller manages SetParent
    //                                                on the enclosing SSwitch)
    // body_list is the storage; callers that need direct access use it.
    struct SCase
    {
        clang::Expr *value; // nullptr for default
        std::vector< SNode * > body_list;

        SNode *body() const { return body_list.empty() ? nullptr : body_list[0]; }

        void set_body(SNode *n) {
            body_list.clear();
            if (n) { body_list.push_back(n); }
        }
    };

    // Switch statement — default_ is std::vector<SNode*>
    // (matches the cases_ list-of-list shape).
    class SSwitch : public SNode
    {
      public:
        SSwitch(clang::Expr *discriminant)
            : SNode(SNodeKind::kSwitch), discriminant_(discriminant) {}

        clang::Expr *Discriminant() const { return discriminant_; }

        void SetDiscriminant(clang::Expr *e) { discriminant_ = e; }

        const std::vector< SCase > &Cases() const { return cases_; }

        std::vector< SCase > &Cases() { return cases_; }

        void AddCase(clang::Expr *value, SNode *body) {
            if (body) { body->SetParent(this); }
            SCase c{ value, {} };
            if (body) { c.body_list.push_back(body); }
            cases_.push_back(std::move(c));
        }

        void AddCase(clang::Expr *value, std::vector< SNode * > body_list) {
            for (auto *b : body_list) {
                if (b) { b->SetParent(this); }
            }
            cases_.push_back({ value, std::move(body_list) });
        }

        SNode *DefaultBody() const { return default_.empty() ? nullptr : default_[0]; }

        void SetDefaultBody(SNode *n) {
            default_.clear();
            if (n) {
                default_.push_back(n);
                n->SetParent(this);
            }
        }

        void SetDefaultBody(std::vector< SNode * > body) {
            default_ = std::move(body);
            for (auto *c : default_) {
                if (c) { c->SetParent(this); }
            }
        }

        const std::vector< SNode * > &DefaultBodyList() const { return default_; }

        std::vector< SNode * > &DefaultBodyList() { return default_; }

        void for_each_child(const ChildFn &fn) const override {
            for (auto &c : cases_) {
                for (auto *b : c.body_list) {
                    if (b) { fn(b); }
                }
            }
            for (auto *b : default_) {
                if (b) { fn(b); }
            }
        }

        void for_each_child_mut(const ChildMutFn &fn) override {
            for (auto &c : cases_) {
                for (auto &slot : c.body_list) {
                    if (slot) { fn(slot); }
                }
            }
            for (auto &slot : default_) {
                if (slot) { fn(slot); }
            }
        }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::kSwitch; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        clang::Expr *discriminant_;
        std::vector< SCase > cases_;
        std::vector< SNode * > default_;
    };

    // Goto
    class SGoto : public SNode
    {
      public:
        SGoto(std::string_view target) : SNode(SNodeKind::kGoto), target_(target) {}

        std::string_view Target() const { return target_; }

        void SetTarget(std::string_view t) { target_ = t; }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::kGoto; }

      private:
        std::string target_;
    };

    // Label
    //
    // body_ is a std::vector<SNode*> so a label body can directly hold a
    // multi-statement sequence.  Public Body()/SetBody() preserve the
    // single-body API for back-compat (Body() returns the first child
    // or nullptr; SetBody replaces the whole list); construction with a
    // single SNode* still works via the existing constructor.  BodyList
    // accessors expose the vector directly for append/iterate callers.
    class SLabel : public SNode
    {
      public:
        SLabel(std::string_view name, SNode *body = nullptr)
            : SNode(SNodeKind::kLabel), name_(name) {
            if (body) {
                body_.push_back(body);
                body->SetParent(this);
            }
        }

        SLabel(std::string_view name, std::vector< SNode * > body)
            : SNode(SNodeKind::kLabel), name_(name), body_(std::move(body)) {
            for (auto *c : body_) {
                if (c) { c->SetParent(this); }
            }
        }

        std::string_view Name() const { return name_; }

        void SetName(std::string_view n) { name_ = n; }

        // Back-compat single-body view: returns the sole child or
        // nullptr.  Callers handling multi-child bodies should use
        // BodyList() instead.
        SNode *Body() const { return body_.empty() ? nullptr : body_[0]; }

        void SetBody(SNode *n) {
            body_.clear();
            if (n) {
                body_.push_back(n);
                n->SetParent(this);
            }
        }

        const std::vector< SNode * > &BodyList() const { return body_; }

        std::vector< SNode * > &BodyList() { return body_; }

        void for_each_child(const ChildFn &fn) const override {
            for (auto *c : body_) {
                if (c) { fn(c); }
            }
        }

        void for_each_child_mut(const ChildMutFn &fn) override {
            for (auto &slot : body_) {
                if (slot) { fn(slot); }
            }
        }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::kLabel; }

      protected:
        void DumpChildren(llvm::raw_ostream &os, unsigned indent) const override;

      private:
        std::string name_;
        std::vector< SNode * > body_;
    };

    // Break (with optional depth for multi-level breaks)
    class SBreak : public SNode
    {
      public:
        SBreak(unsigned depth = 1) : SNode(SNodeKind::kBreak), depth_(depth) {}

        unsigned Depth() const { return depth_; }

        void SetDepth(unsigned d) { depth_ = d; }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::kBreak; }

      private:
        unsigned depth_;
    };

    // Continue
    class SContinue : public SNode
    {
      public:
        SContinue() : SNode(SNodeKind::kContinue) {}

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::kContinue; }
    };

    // Return
    class SReturn : public SNode
    {
      public:
        SReturn(clang::Expr *value = nullptr) : SNode(SNodeKind::kReturn), value_(value) {}

        clang::Expr *Value() const { return value_; }

        void SetValue(clang::Expr *e) { value_ = e; }

        static bool classof(const SNode *n) { return n->Kind() == SNodeKind::kReturn; }

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

        // Build a sequence (std::vector<SNode*>) with normalization:
        // nullptr children are dropped.  A "sequence" is a plain
        // std::vector<SNode*> — the SSeq node kind no longer exists.
        std::vector< SNode * > MakeSeq(std::initializer_list< SNode * > children) {
            return MakeSeq(std::vector< SNode * >(children));
        }

        std::vector< SNode * > MakeSeq(std::vector< SNode * > children);

        // Intern a copy of a string; the returned view is valid until Reset().
        // Uses a bump allocator because raw char data carries no destructor.
        std::string_view Intern(std::string_view s) {
            if (s.empty()) { return {}; }
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
