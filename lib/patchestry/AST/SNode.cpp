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
            case SNodeKind::kStmt:        return "Stmt";
            case SNodeKind::kIfThenElse:return "IfThenElse";
            case SNodeKind::kWhile:       return "While";
            case SNodeKind::kDoWhile:    return "DoWhile";
            case SNodeKind::kFor:         return "For";
            case SNodeKind::kSwitch:      return "Switch";
            case SNodeKind::kGoto:        return "Goto";
            case SNodeKind::kLabel:       return "Label";
            case SNodeKind::kBreak:       return "Break";
            case SNodeKind::kContinue:    return "Continue";
            case SNodeKind::kReturn:      return "Return";
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

    void SStmt::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        PrintIndent(os, indent + 1);
        os << "stmt: " << (stmt_ ? "<stmt>" : "null") << "\n";
    }

    void SIfThenElse::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        PrintIndent(os, indent + 1);
        os << "cond: <expr>\n";
        if (!then_.empty()) {
            PrintIndent(os, indent + 1);
            os << "then:\n";
            for (const auto *c : then_) if (c) c->Dump(os, indent + 2);
        }
        if (!else_.empty()) {
            PrintIndent(os, indent + 1);
            os << "else:\n";
            for (const auto *c : else_) if (c) c->Dump(os, indent + 2);
        }
    }

    void SWhile::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        PrintIndent(os, indent + 1);
        os << "cond: <expr>\n";
        if (!body_.empty()) {
            PrintIndent(os, indent + 1);
            os << "body:\n";
            for (const auto *c : body_) if (c) c->Dump(os, indent + 2);
        }
    }

    void SDoWhile::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        if (!body_.empty()) {
            PrintIndent(os, indent + 1);
            os << "body:\n";
            for (const auto *c : body_) if (c) c->Dump(os, indent + 2);
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
        if (!body_.empty()) {
            PrintIndent(os, indent + 1);
            os << "body:\n";
            for (const auto *c : body_) if (c) c->Dump(os, indent + 2);
        }
    }

    void SSwitch::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        PrintIndent(os, indent + 1);
        os << "discriminant: <expr>\n";
        for (size_t i = 0; i < cases_.size(); ++i) {
            PrintIndent(os, indent + 1);
            os << "case " << i << ":\n";
            for (const auto *c : cases_[i].body_list)
                if (c) c->Dump(os, indent + 2);
        }
        if (!default_.empty()) {
            PrintIndent(os, indent + 1);
            os << "default:\n";
            for (const auto *c : default_) if (c) c->Dump(os, indent + 2);
        }
    }

    void SLabel::DumpChildren(llvm::raw_ostream &os, unsigned indent) const {
        PrintIndent(os, indent + 1);
        os << "name: " << name_ << "\n";
        for (const auto *c : body_) if (c) c->Dump(os, indent + 1);
    }

    // ---------------------------------------------------------------
    // SNodeFactory::MakeSeq — normalize a child sequence.
    //
    // Since the SSeq node kind was removed, a "sequence" is just a
    // std::vector<SNode*>.  MakeSeq drops nullptr children and returns
    // the resulting vector; callers store it directly into a body
    // slot, into CNode::structured, or pass it to IdentifyInternal.
    // ---------------------------------------------------------------
    std::vector< SNode * > SNodeFactory::MakeSeq(std::vector< SNode * > children) {
        std::vector< SNode * > out;
        out.reserve(children.size());
        for (SNode *c : children) {
            if (c) out.push_back(c);
        }
        return out;
    }

} // namespace patchestry::ast
