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

            unsigned Emit(const SNode *node) {
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
                case SNodeKind::kSeq: {
                    auto *seq = node->as< SSeq >();
                    for (const auto *child : seq->Children()) {
                        unsigned cid = Emit(child);
                        os << "  n" << id << " -> n" << cid << ";\n";
                    }
                    break;
                }
                case SNodeKind::kIfThenElse: {
                    auto *ite = node->as< SIfThenElse >();
                    if (ite->ThenBranch()) {
                        unsigned tid = Emit(ite->ThenBranch());
                        os << "  n" << id << " -> n" << tid << " [label=\"then\"];\n";
                    }
                    if (ite->ElseBranch()) {
                        unsigned eid = Emit(ite->ElseBranch());
                        os << "  n" << id << " -> n" << eid << " [label=\"else\"];\n";
                    }
                    break;
                }
                case SNodeKind::kWhile: {
                    auto *w = node->as< SWhile >();
                    if (w->Body()) {
                        unsigned bid = Emit(w->Body());
                        os << "  n" << id << " -> n" << bid << " [label=\"body\"];\n";
                    }
                    break;
                }
                case SNodeKind::kDoWhile: {
                    auto *dw = node->as< SDoWhile >();
                    if (dw->Body()) {
                        unsigned bid = Emit(dw->Body());
                        os << "  n" << id << " -> n" << bid << " [label=\"body\"];\n";
                    }
                    break;
                }
                case SNodeKind::kFor: {
                    auto *f = node->as< SFor >();
                    if (f->Body()) {
                        unsigned bid = Emit(f->Body());
                        os << "  n" << id << " -> n" << bid << " [label=\"body\"];\n";
                    }
                    break;
                }
                case SNodeKind::kSwitch: {
                    auto *sw = node->as< SSwitch >();
                    for (size_t i = 0; i < sw->Cases().size(); ++i) {
                        if (sw->Cases()[i].body) {
                            unsigned cid = Emit(sw->Cases()[i].body);
                            os << "  n" << id << " -> n" << cid
                               << " [label=\"case " << i << "\"];\n";
                        }
                    }
                    if (sw->DefaultBody()) {
                        unsigned did = Emit(sw->DefaultBody());
                        os << "  n" << id << " -> n" << did
                           << " [label=\"default\"];\n";
                    }
                    break;
                }
                case SNodeKind::kLabel: {
                    auto *lbl = node->as< SLabel >();
                    if (lbl->Body()) {
                        unsigned bid = Emit(lbl->Body());
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
        emitter.Emit(node);
        os << "}\n";
    }

} // namespace patchestry::ast
