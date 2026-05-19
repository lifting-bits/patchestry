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
                } else if (node->dyn_cast< SStmt >()) {
                    os << "\\n(stmt)";
                }

                os << "\"];\n";

                // Emit one edge per child for a body slot.  Body vectors
                // may hold many siblings, not just the first — the
                // single-body Body()/ThenBranch() accessors would drop
                // everything after the first child.
                auto emit_list = [&](const std::vector< SNode * > &list,
                                     const std::string &edge) {
                    for (const auto *c : list) {
                        if (!c) continue;
                        unsigned cid = Emit(c);
                        os << "  n" << id << " -> n" << cid;
                        if (!edge.empty()) {
                            os << " [label=\"" << edge << "\"]";
                        }
                        os << ";\n";
                    }
                };

                switch (node->Kind()) {
                case SNodeKind::kIfThenElse: {
                    auto *ite = node->as< SIfThenElse >();
                    emit_list(ite->ThenList(), "then");
                    emit_list(ite->ElseList(), "else");
                    break;
                }
                case SNodeKind::kWhile:
                    emit_list(node->as< SWhile >()->BodyList(), "body");
                    break;
                case SNodeKind::kDoWhile:
                    emit_list(node->as< SDoWhile >()->BodyList(), "body");
                    break;
                case SNodeKind::kFor:
                    emit_list(node->as< SFor >()->BodyList(), "body");
                    break;
                case SNodeKind::kSwitch: {
                    auto *sw = node->as< SSwitch >();
                    for (size_t i = 0; i < sw->Cases().size(); ++i) {
                        emit_list(sw->Cases()[i].body_list,
                                  "case " + std::to_string(i));
                    }
                    emit_list(sw->DefaultBodyList(), "default");
                    break;
                }
                case SNodeKind::kLabel:
                    emit_list(node->as< SLabel >()->BodyList(), "");
                    break;
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
