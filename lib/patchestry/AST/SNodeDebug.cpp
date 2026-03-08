/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/SNodeDebug.hpp>

#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/PrettyPrinter.h>

#include <string>

namespace patchestry::ast {

    namespace {
        // Convert std::string_view to llvm::StringRef for LLVM JSON APIs
        llvm::StringRef toRef(std::string_view sv) {
            return llvm::StringRef(sv.data(), sv.size());
        }
    } // namespace

    // ========================================================================
    // Pseudo-C printer
    // ========================================================================

    static void indent(llvm::raw_ostream &os, unsigned level) {
        for (unsigned i = 0; i < level; ++i) os << "    ";
    }

    static void printExpr(const clang::Expr *e, llvm::raw_ostream &os,
                          clang::ASTContext *ctx) {
        if (!e) { os << "/*null*/"; return; }
        if (ctx) {
            clang::PrintingPolicy pp(ctx->getLangOpts());
            pp.SuppressImplicitBase = true;
            e->printPretty(os, nullptr, pp);
        } else {
            os << "<expr>";
        }
    }

    static void printStmt(const clang::Stmt *s, llvm::raw_ostream &os,
                          clang::ASTContext *ctx, unsigned ind) {
        if (!s) return;
        if (ctx) {
            clang::PrintingPolicy pp(ctx->getLangOpts());
            pp.SuppressImplicitBase = true;
            indent(os, ind);
            s->printPretty(os, nullptr, pp, ind);
            // printPretty may or may not add newline — ensure one
            os << "\n";
        } else {
            indent(os, ind);
            os << "<stmt>;\n";
        }
    }

    void printPseudoC(const SNode *node, llvm::raw_ostream &os,
                      clang::ASTContext *ctx, unsigned ind) {
        if (!node) return;

        switch (node->kind()) {
        case SNodeKind::Seq: {
            auto *seq = node->as< SSeq >();
            for (const auto *child : seq->children()) {
                printPseudoC(child, os, ctx, ind);
            }
            break;
        }
        case SNodeKind::Block: {
            auto *blk = node->as< SBlock >();
            for (auto *s : blk->stmts()) {
                printStmt(s, os, ctx, ind);
            }
            break;
        }
        case SNodeKind::IfThenElse: {
            auto *ite = node->as< SIfThenElse >();
            indent(os, ind);
            os << "if (";
            printExpr(ite->cond(), os, ctx);
            os << ") {\n";
            printPseudoC(ite->thenBranch(), os, ctx, ind + 1);
            if (ite->elseBranch()) {
                indent(os, ind);
                os << "} else {\n";
                printPseudoC(ite->elseBranch(), os, ctx, ind + 1);
            }
            indent(os, ind);
            os << "}\n";
            break;
        }
        case SNodeKind::While: {
            auto *w = node->as< SWhile >();
            indent(os, ind);
            os << "while (";
            printExpr(w->cond(), os, ctx);
            os << ") {\n";
            printPseudoC(w->body(), os, ctx, ind + 1);
            indent(os, ind);
            os << "}\n";
            break;
        }
        case SNodeKind::DoWhile: {
            auto *dw = node->as< SDoWhile >();
            indent(os, ind);
            os << "do {\n";
            printPseudoC(dw->body(), os, ctx, ind + 1);
            indent(os, ind);
            os << "} while (";
            printExpr(dw->cond(), os, ctx);
            os << ");\n";
            break;
        }
        case SNodeKind::For: {
            auto *f = node->as< SFor >();
            indent(os, ind);
            os << "for (";
            if (f->init()) {
                if (ctx) {
                    clang::PrintingPolicy pp(ctx->getLangOpts());
                    f->init()->printPretty(os, nullptr, pp);
                } else {
                    os << "<init>";
                }
            }
            os << "; ";
            printExpr(f->cond(), os, ctx);
            os << "; ";
            printExpr(f->inc(), os, ctx);
            os << ") {\n";
            printPseudoC(f->body(), os, ctx, ind + 1);
            indent(os, ind);
            os << "}\n";
            break;
        }
        case SNodeKind::Switch: {
            auto *sw = node->as< SSwitch >();
            indent(os, ind);
            os << "switch (";
            printExpr(sw->discriminant(), os, ctx);
            os << ") {\n";
            for (const auto &c : sw->cases()) {
                indent(os, ind);
                os << "case ";
                printExpr(c.value, os, ctx);
                os << ":\n";
                if (c.body) printPseudoC(c.body, os, ctx, ind + 1);
                indent(os, ind + 1);
                os << "break;\n";
            }
            if (sw->defaultBody()) {
                indent(os, ind);
                os << "default:\n";
                printPseudoC(sw->defaultBody(), os, ctx, ind + 1);
            }
            indent(os, ind);
            os << "}\n";
            break;
        }
        case SNodeKind::Goto: {
            auto *g = node->as< SGoto >();
            indent(os, ind);
            os << "goto " << g->target() << ";\n";
            break;
        }
        case SNodeKind::Label: {
            auto *lbl = node->as< SLabel >();
            indent(os, ind > 0 ? ind - 1 : 0);
            os << lbl->name() << ":\n";
            if (lbl->body()) printPseudoC(lbl->body(), os, ctx, ind);
            break;
        }
        case SNodeKind::Break: {
            indent(os, ind);
            os << "break;\n";
            break;
        }
        case SNodeKind::Continue: {
            indent(os, ind);
            os << "continue;\n";
            break;
        }
        case SNodeKind::Return: {
            auto *ret = node->as< SReturn >();
            indent(os, ind);
            os << "return";
            if (ret->value()) {
                os << " ";
                printExpr(ret->value(), os, ctx);
            }
            os << ";\n";
            break;
        }
        }
    }

    // ========================================================================
    // DOT emitter
    // ========================================================================

    namespace {
        struct DotEmitter {
            llvm::raw_ostream &os;
            unsigned next_id = 0;

            unsigned emit(const SNode *node) {
                if (!node) return 0;
                unsigned id = next_id++;

                os << "  n" << id << " [label=\"" << node->kindName();

                if (auto *lbl = node->dyn_cast< SLabel >()) {
                    os << "\\n" << lbl->name();
                } else if (auto *g = node->dyn_cast< SGoto >()) {
                    os << "\\n-> " << g->target();
                } else if (auto *blk = node->dyn_cast< SBlock >()) {
                    os << "\\n(" << blk->size() << " stmts)";
                    if (!blk->label().empty()) os << "\\nlabel: " << blk->label();
                }

                os << "\"];\n";

                switch (node->kind()) {
                case SNodeKind::Seq: {
                    auto *seq = node->as< SSeq >();
                    for (const auto *child : seq->children()) {
                        unsigned cid = emit(child);
                        os << "  n" << id << " -> n" << cid << ";\n";
                    }
                    break;
                }
                case SNodeKind::IfThenElse: {
                    auto *ite = node->as< SIfThenElse >();
                    if (ite->thenBranch()) {
                        unsigned tid = emit(ite->thenBranch());
                        os << "  n" << id << " -> n" << tid << " [label=\"then\"];\n";
                    }
                    if (ite->elseBranch()) {
                        unsigned eid = emit(ite->elseBranch());
                        os << "  n" << id << " -> n" << eid << " [label=\"else\"];\n";
                    }
                    break;
                }
                case SNodeKind::While: {
                    auto *w = node->as< SWhile >();
                    if (w->body()) {
                        unsigned bid = emit(w->body());
                        os << "  n" << id << " -> n" << bid << " [label=\"body\"];\n";
                    }
                    break;
                }
                case SNodeKind::DoWhile: {
                    auto *dw = node->as< SDoWhile >();
                    if (dw->body()) {
                        unsigned bid = emit(dw->body());
                        os << "  n" << id << " -> n" << bid << " [label=\"body\"];\n";
                    }
                    break;
                }
                case SNodeKind::For: {
                    auto *f = node->as< SFor >();
                    if (f->body()) {
                        unsigned bid = emit(f->body());
                        os << "  n" << id << " -> n" << bid << " [label=\"body\"];\n";
                    }
                    break;
                }
                case SNodeKind::Switch: {
                    auto *sw = node->as< SSwitch >();
                    for (size_t i = 0; i < sw->cases().size(); ++i) {
                        if (sw->cases()[i].body) {
                            unsigned cid = emit(sw->cases()[i].body);
                            os << "  n" << id << " -> n" << cid
                               << " [label=\"case " << i << "\"];\n";
                        }
                    }
                    if (sw->defaultBody()) {
                        unsigned did = emit(sw->defaultBody());
                        os << "  n" << id << " -> n" << did
                           << " [label=\"default\"];\n";
                    }
                    break;
                }
                case SNodeKind::Label: {
                    auto *lbl = node->as< SLabel >();
                    if (lbl->body()) {
                        unsigned bid = emit(lbl->body());
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

    void emitDOT(const SNode *node, llvm::raw_ostream &os) {
        os << "digraph SNodeTree {\n";
        os << "  node [shape=box, fontname=\"Courier\"];\n";
        DotEmitter emitter{os};
        emitter.emit(node);
        os << "}\n";
    }

    // ========================================================================
    // JSON emitter
    // ========================================================================

    void emitJSON(const SNode *node, llvm::json::OStream &jos,
                  clang::ASTContext *ctx) {
        if (!node) { jos.value(nullptr); return; }

        jos.object([&] {
            jos.attribute("kind", node->kindName());

            switch (node->kind()) {
            case SNodeKind::Seq: {
                auto *seq = node->as< SSeq >();
                jos.attributeArray("children", [&] {
                    for (const auto *child : seq->children()) {
                        emitJSON(child, jos, ctx);
                    }
                });
                break;
            }
            case SNodeKind::Block: {
                auto *blk = node->as< SBlock >();
                if (!blk->label().empty()) {
                    jos.attribute("label", toRef(blk->label()));
                }
                jos.attribute("stmt_count", static_cast< int64_t >(blk->size()));
                break;
            }
            case SNodeKind::IfThenElse: {
                auto *ite = node->as< SIfThenElse >();
                if (ite->cond() && ctx) {
                    std::string cond_str;
                    llvm::raw_string_ostream rso(cond_str);
                    ite->cond()->printPretty(rso, nullptr, ctx->getPrintingPolicy());
                    jos.attribute("cond", cond_str);
                }
                jos.attributeBegin("then");
                emitJSON(ite->thenBranch(), jos, ctx);
                jos.attributeEnd();
                if (ite->elseBranch()) {
                    jos.attributeBegin("else");
                    emitJSON(ite->elseBranch(), jos, ctx);
                    jos.attributeEnd();
                }
                break;
            }
            case SNodeKind::While: {
                auto *w = node->as< SWhile >();
                jos.attributeBegin("body");
                emitJSON(w->body(), jos, ctx);
                jos.attributeEnd();
                break;
            }
            case SNodeKind::DoWhile: {
                auto *dw = node->as< SDoWhile >();
                jos.attributeBegin("body");
                emitJSON(dw->body(), jos, ctx);
                jos.attributeEnd();
                break;
            }
            case SNodeKind::For: {
                auto *f = node->as< SFor >();
                jos.attributeBegin("body");
                emitJSON(f->body(), jos, ctx);
                jos.attributeEnd();
                break;
            }
            case SNodeKind::Switch: {
                auto *sw = node->as< SSwitch >();
                jos.attributeArray("cases", [&] {
                    for (const auto &c : sw->cases()) {
                        jos.object([&] {
                            jos.attributeBegin("body");
                            emitJSON(c.body, jos, ctx);
                            jos.attributeEnd();
                        });
                    }
                });
                if (sw->defaultBody()) {
                    jos.attributeBegin("default");
                    emitJSON(sw->defaultBody(), jos, ctx);
                    jos.attributeEnd();
                }
                break;
            }
            case SNodeKind::Goto: {
                auto *g = node->as< SGoto >();
                jos.attribute("target", toRef(g->target()));
                break;
            }
            case SNodeKind::Label: {
                auto *lbl = node->as< SLabel >();
                jos.attribute("name", toRef(lbl->name()));
                jos.attributeBegin("body");
                emitJSON(lbl->body(), jos, ctx);
                jos.attributeEnd();
                break;
            }
            case SNodeKind::Break: {
                auto *brk = node->as< SBreak >();
                if (brk->depth() > 1)
                    jos.attribute("depth", static_cast< int64_t >(brk->depth()));
                break;
            }
            case SNodeKind::Continue:
                break;
            case SNodeKind::Return: {
                auto *ret = node->as< SReturn >();
                jos.attribute("has_value", ret->value() != nullptr);
                break;
            }
            }
        });
    }

} // namespace patchestry::ast
