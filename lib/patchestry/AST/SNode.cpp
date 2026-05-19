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
            case SNodeKind::kSeq:         return "Seq";
            case SNodeKind::kBlock:       return "Block";
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

    // ---------------------------------------------------------------
    // SNodeFactory::MakeSeq — Layer C migration Stage 0
    //
    // Normalizes children before constructing an SSeq.  See header
    // for rules.  Out-of-line so we can inspect SSeq's internals
    // without forcing a chain of inline includes.
    //
    // NOTE: Empty unlabeled SBlocks are intentionally PRESERVED here.
    // They are topologically meaningful — they correspond to CFG nodes
    // with no statements, and downstream cleanup passes use sibling
    // position to reason about fallthrough vs goto.  Dropping them at
    // construction was tried in an earlier draft of Stage 1 and broke
    // 4 fixtures (cwe121_eeprom_handler_write DUP_ASSIGN +
    // decode_frame / load_descriptor_values / pb_decode_inner Fall=N)
    // by promoting goto-only labels to sequential successors.  The
    // cosmetic `{ }` artifact in the emitted C is handled by
    // RemoveEmptyBlocks at the Clang-AST level, which understands
    // topology because it operates after IR materialization.
    // ---------------------------------------------------------------
    SNode *SNodeFactory::MakeSeq(std::vector< SNode * > children) {
        // Conservative normalization for Stage 1:
        //   - Drop nullptr children.
        //   - If size 0 → return nullptr (caller decides substitute).
        //   - If size 1 → return the single child directly (no wrap).
        //   - Otherwise → allocate an SSeq and AddChild each.
        //
        // Two more aggressive normalizations were tried and reverted:
        //
        //   * Dropping empty unlabeled SBlock children — broke 4 fixtures
        //     (cwe121_eeprom_handler_write DUP_ASSIGN +
        //      decode_frame / load_descriptor_values / pb_decode_inner Fall=N)
        //     by promoting goto-only labels to sequential successors.
        //     Empty SBlocks correspond to CFG nodes with no statements
        //     and downstream cleanup uses sibling distance to reason
        //     about fallthrough vs goto.
        //
        //   * Flattening nested SSeq inline — also broke the same 4
        //     fixtures.  Flattening exposed label adjacency that
        //     downstream EliminateGotoToNextLabel / IfStmtGotoArm
        //     elision logic mis-handles when the label has live
        //     non-goto predecessors.  The aggressive goto reduction it
        //     delivered (cve_2016_6563 38→6) is real but requires a
        //     separate fix to the elision predecessor-set check before
        //     it can be enabled safely.
        //
        // Both deferrals keep Stage 1 semantics-preserving so the SSeq
        // → vector-slot migration can land without entangling policy
        // fixes with the structural refactor.
        std::vector< SNode * > out;
        out.reserve(children.size());
        for (SNode *c : children) {
            if (c) out.push_back(c);
        }
        if (out.empty()) return nullptr;
        if (out.size() == 1) return out[0];
        auto *seq = Make< SSeq >();
        for (auto *c : out) seq->AddChild(c);
        return seq;
    }

} // namespace patchestry::ast
