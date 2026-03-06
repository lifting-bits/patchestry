/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/SNode.hpp>

#include <clang/AST/Stmt.h>

namespace patchestry::ast {

    const char *SNode::kindName(SNodeKind k) {
        switch (k) {
            case SNodeKind::Seq:         return "Seq";
            case SNodeKind::Block:       return "Block";
            case SNodeKind::IfThenElse:  return "IfThenElse";
            case SNodeKind::While:       return "While";
            case SNodeKind::DoWhile:     return "DoWhile";
            case SNodeKind::For:         return "For";
            case SNodeKind::Switch:      return "Switch";
            case SNodeKind::Goto:        return "Goto";
            case SNodeKind::Label:       return "Label";
            case SNodeKind::Break:       return "Break";
            case SNodeKind::Continue:    return "Continue";
            case SNodeKind::Return:      return "Return";
        }
        return "Unknown";
    }

    static void printIndent(llvm::raw_ostream &os, unsigned indent) {
        for (unsigned i = 0; i < indent; ++i) os << "  ";
    }

    void SNode::dump(llvm::raw_ostream &os, unsigned indent) const {
        printIndent(os, indent);
        os << kindName() << "\n";
        dumpChildren(os, indent);
    }

    void SSeq::dumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        for (const auto *child : children_) {
            child->dump(os, indent + 1);
        }
    }

    void SBlock::dumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        if (!label_.empty()) {
            printIndent(os, indent + 1);
            os << "label: " << label_ << "\n";
        }
        printIndent(os, indent + 1);
        os << "stmts: " << stmts_.size() << "\n";
    }

    void SIfThenElse::dumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        printIndent(os, indent + 1);
        os << "cond: <expr>\n";
        if (then_) {
            printIndent(os, indent + 1);
            os << "then:\n";
            then_->dump(os, indent + 2);
        }
        if (else_) {
            printIndent(os, indent + 1);
            os << "else:\n";
            else_->dump(os, indent + 2);
        }
    }

    void SWhile::dumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        printIndent(os, indent + 1);
        os << "cond: <expr>\n";
        if (body_) {
            printIndent(os, indent + 1);
            os << "body:\n";
            body_->dump(os, indent + 2);
        }
    }

    void SDoWhile::dumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        if (body_) {
            printIndent(os, indent + 1);
            os << "body:\n";
            body_->dump(os, indent + 2);
        }
        printIndent(os, indent + 1);
        os << "cond: <expr>\n";
    }

    void SFor::dumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        printIndent(os, indent + 1);
        os << "init: " << (init_ ? "<stmt>" : "null") << "\n";
        printIndent(os, indent + 1);
        os << "cond: " << (cond_ ? "<expr>" : "null") << "\n";
        printIndent(os, indent + 1);
        os << "inc: " << (inc_ ? "<expr>" : "null") << "\n";
        if (body_) {
            printIndent(os, indent + 1);
            os << "body:\n";
            body_->dump(os, indent + 2);
        }
    }

    void SSwitch::dumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        printIndent(os, indent + 1);
        os << "discriminant: <expr>\n";
        for (size_t i = 0; i < cases_.size(); ++i) {
            printIndent(os, indent + 1);
            os << "case " << i << ":\n";
            if (cases_[i].body) {
                cases_[i].body->dump(os, indent + 2);
            }
        }
        if (default_) {
            printIndent(os, indent + 1);
            os << "default:\n";
            default_->dump(os, indent + 2);
        }
    }

    void SLabel::dumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        printIndent(os, indent + 1);
        os << "name: " << name_ << "\n";
        if (body_) {
            body_->dump(os, indent + 1);
        }
    }

} // namespace patchestry::ast
