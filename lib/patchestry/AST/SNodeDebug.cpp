/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/SNodeDebug.hpp>

namespace patchestry::ast {

    namespace {
        struct DotEmitter {
            llvm::raw_ostream &os;
            unsigned next_id = 0;

            unsigned emit(const SNode *node) {
                if (!node) return 0;
                unsigned id = next_id++;

                os << "  n" << id << " [label=\"" << node->KindName();

                if (auto *lbl = node->dyn_cast< SLabel >()) {
                    os << "\\n" << lbl->Name();
                } else if (auto *g = node->dyn_cast< SGoto >()) {
                    os << "\\n-> " << g->Target();
                } else if (auto *blk = node->dyn_cast< SBlock >()) {
                    os << "\\n(" << blk->Size() << " stmts)";
                    if (!blk->Label().empty()) os << "\\nlabel: " << blk->Label();
                }

                os << "\"];\n";

                switch (node->Kind()) {
                case SNodeKind::SEQ: {
                    auto *seq = node->as< SSeq >();
                    for (const auto *child : seq->Children()) {
                        unsigned cid = emit(child);
                        os << "  n" << id << " -> n" << cid << ";\n";
                    }
                    break;
                }
                case SNodeKind::IF_THEN_ELSE: {
                    auto *ite = node->as< SIfThenElse >();
                    if (ite->ThenBranch()) {
                        unsigned tid = emit(ite->ThenBranch());
                        os << "  n" << id << " -> n" << tid << " [label=\"then\"];\n";
                    }
                    if (ite->ElseBranch()) {
                        unsigned eid = emit(ite->ElseBranch());
                        os << "  n" << id << " -> n" << eid << " [label=\"else\"];\n";
                    }
                    break;
                }
                case SNodeKind::WHILE: {
                    auto *w = node->as< SWhile >();
                    if (w->Body()) {
                        unsigned bid = emit(w->Body());
                        os << "  n" << id << " -> n" << bid << " [label=\"body\"];\n";
                    }
                    break;
                }
                case SNodeKind::DO_WHILE: {
                    auto *dw = node->as< SDoWhile >();
                    if (dw->Body()) {
                        unsigned bid = emit(dw->Body());
                        os << "  n" << id << " -> n" << bid << " [label=\"body\"];\n";
                    }
                    break;
                }
                case SNodeKind::FOR: {
                    auto *f = node->as< SFor >();
                    if (f->Body()) {
                        unsigned bid = emit(f->Body());
                        os << "  n" << id << " -> n" << bid << " [label=\"body\"];\n";
                    }
                    break;
                }
                case SNodeKind::SWITCH: {
                    auto *sw = node->as< SSwitch >();
                    for (size_t i = 0; i < sw->Cases().size(); ++i) {
                        if (sw->Cases()[i].body) {
                            unsigned cid = emit(sw->Cases()[i].body);
                            os << "  n" << id << " -> n" << cid
                               << " [label=\"case " << i << "\"];\n";
                        }
                    }
                    if (sw->DefaultBody()) {
                        unsigned did = emit(sw->DefaultBody());
                        os << "  n" << id << " -> n" << did
                           << " [label=\"default\"];\n";
                    }
                    break;
                }
                case SNodeKind::LABEL: {
                    auto *lbl = node->as< SLabel >();
                    if (lbl->Body()) {
                        unsigned bid = emit(lbl->Body());
                        os << "  n" << id << " -> n" << bid << ";\n";
                    }
                    break;
                }
                default:
                    break;
                }

                return id;
            }
        };
    } // namespace

    void EmitDot(const SNode *node, llvm::raw_ostream &os) {
        os << "digraph SNodeTree {\n";
        os << "  node [shape=box, fontname=\"Courier\"];\n";
        DotEmitter emitter{os};
        emitter.emit(node);
        os << "}\n";
    }

} // namespace patchestry::ast
