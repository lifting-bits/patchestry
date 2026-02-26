/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/SNode.hpp>

#include <clang/AST/Stmt.h>

namespace patchestry::ast {

    const char *SNode::KindName(SNodeKind k) {
        switch (k) {
            case SNodeKind::SEQ:         return "Seq";
            case SNodeKind::BLOCK:       return "Block";
            case SNodeKind::IF_THEN_ELSE:return "IfThenElse";
            case SNodeKind::WHILE:       return "While";
            case SNodeKind::DO_WHILE:    return "DoWhile";
            case SNodeKind::FOR:         return "For";
            case SNodeKind::SWITCH:      return "Switch";
            case SNodeKind::GOTO:        return "Goto";
            case SNodeKind::LABEL:       return "Label";
            case SNodeKind::BREAK:       return "Break";
            case SNodeKind::CONTINUE:    return "Continue";
            case SNodeKind::RETURN:      return "Return";
        }
        return "Unknown";
    }

    static void PrintIndent(llvm::raw_ostream &os, unsigned indent) {
        for (unsigned i = 0; i < indent; ++i) os << "  ";
    }

    void SNode::Dump(llvm::raw_ostream &os, unsigned indent) const {
        PrintIndent(os, indent);
        os << KindName() << "\n";
        DumpChildren(os, indent);
    }

    void SSeq::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        for (const auto *child : children_) {
            child->Dump(os, indent + 1);
        }
    }

    void SBlock::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        if (!label_.empty()) {
            PrintIndent(os, indent + 1);
            os << "label: " << label_ << "\n";
        }
        PrintIndent(os, indent + 1);
        os << "stmts: " << stmts_.size() << "\n";
    }

    void SIfThenElse::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        PrintIndent(os, indent + 1);
        os << "cond: <expr>\n";
        if (then_) {
            PrintIndent(os, indent + 1);
            os << "then:\n";
            then_->Dump(os, indent + 2);
        }
        if (else_) {
            PrintIndent(os, indent + 1);
            os << "else:\n";
            else_->Dump(os, indent + 2);
        }
    }

    void SWhile::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        PrintIndent(os, indent + 1);
        os << "cond: <expr>\n";
        if (body_) {
            PrintIndent(os, indent + 1);
            os << "body:\n";
            body_->Dump(os, indent + 2);
        }
    }

    void SDoWhile::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        if (body_) {
            PrintIndent(os, indent + 1);
            os << "body:\n";
            body_->Dump(os, indent + 2);
        }
        PrintIndent(os, indent + 1);
        os << "cond: <expr>\n";
    }

    void SFor::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        PrintIndent(os, indent + 1);
        os << "init: " << (init_ ? "<stmt>" : "null") << "\n";
        PrintIndent(os, indent + 1);
        os << "cond: " << (cond_ ? "<expr>" : "null") << "\n";
        PrintIndent(os, indent + 1);
        os << "inc: " << (inc_ ? "<expr>" : "null") << "\n";
        if (body_) {
            PrintIndent(os, indent + 1);
            os << "body:\n";
            body_->Dump(os, indent + 2);
        }
    }

    void SSwitch::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        PrintIndent(os, indent + 1);
        os << "discriminant: <expr>\n";
        for (size_t i = 0; i < cases_.size(); ++i) {
            PrintIndent(os, indent + 1);
            os << "case " << i << ":\n";
            if (cases_[i].body) {
                cases_[i].body->Dump(os, indent + 2);
            }
        }
        if (default_) {
            PrintIndent(os, indent + 1);
            os << "default:\n";
            default_->Dump(os, indent + 2);
        }
    }

    void SLabel::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        PrintIndent(os, indent + 1);
        os << "name: " << name_ << "\n";
        if (body_) {
            body_->Dump(os, indent + 1);
        }
    }

} // namespace patchestry::ast
