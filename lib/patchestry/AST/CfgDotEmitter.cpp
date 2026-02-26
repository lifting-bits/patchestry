/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/CfgDotEmitter.hpp>
#include <patchestry/Util/Log.hpp>

#include <clang/AST/Expr.h>
#include <clang/AST/PrettyPrinter.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/LangOptions.h>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <iomanip>
#include <sstream>
#include <unordered_set>

namespace patchestry::ast {

    namespace {
        /// Print a Clang Stmt as a single-line string, escaped for DOT labels.
        /// Truncates to max_len characters to keep nodes readable.
        std::string StmtToOneLine(const clang::Stmt *s, unsigned max_len = 60) {
            if (!s) return "";
            std::string buf;
            llvm::raw_string_ostream rso(buf);
            clang::LangOptions lo;
            clang::PrintingPolicy pp(lo);
            pp.SuppressImplicitBase = true;
            s->printPretty(rso, nullptr, pp);
            rso.flush();

            // Replace newlines, quotes, backslashes for DOT compatibility
            std::string result;
            result.reserve(buf.size());
            for (char c : buf) {
                if (c == '\n' || c == '\r') { result += ' '; continue; }
                if (c == '"') { result += "\\\""; continue; }
                if (c == '\\') { result += "\\\\"; continue; }
                if (c == '<' || c == '>' || c == '{' || c == '}' || c == '|') {
                    result += c;  // safe in quoted labels
                    continue;
                }
                result += c;
            }

            // Truncate long statements
            if (result.size() > max_len) {
                result.resize(max_len - 3);
                // Don't leave a trailing backslash (splits escape sequence)
                if (!result.empty() && result.back() == '\\') {
                    result.pop_back();
                }
                result += "...";
            }
            return result;
        }
        /// Recursively dump an SNode tree as indented text for DOT labels.
        /// Each line is left-aligned with \l.
        void DumpSNode(const SNode *node, llvm::raw_ostream &os,
                       unsigned indent = 0) {
            if (!node) return;
            auto pad = [&]() {
                for (unsigned i = 0; i < indent; ++i) os << "  ";
            };

            switch (node->Kind()) {
            case SNodeKind::kBlock: {
                auto *blk = node->as<SBlock>();
                for (const auto *s : blk->Stmts()) {
                    pad(); os << StmtToOneLine(s) << "\\l";
                }
                break;
            }
            case SNodeKind::kSeq: {
                for (const auto *child : node->as<SSeq>()->Children())
                    DumpSNode(child, os, indent);
                break;
            }
            case SNodeKind::kIfThenElse: {
                auto *ite = node->as<SIfThenElse>();
                pad(); os << "if (" << StmtToOneLine(ite->Cond()) << ") {\\l";
                DumpSNode(ite->ThenBranch(), os, indent + 1);
                if (ite->ElseBranch()) {
                    pad(); os << "} else {\\l";
                    DumpSNode(ite->ElseBranch(), os, indent + 1);
                }
                pad(); os << "}\\l";
                break;
            }
            case SNodeKind::kWhile: {
                auto *w = node->as<SWhile>();
                pad(); os << "while (" << StmtToOneLine(w->Cond()) << ") {\\l";
                DumpSNode(w->Body(), os, indent + 1);
                pad(); os << "}\\l";
                break;
            }
            case SNodeKind::kDoWhile: {
                auto *dw = node->as<SDoWhile>();
                pad(); os << "do {\\l";
                DumpSNode(dw->Body(), os, indent + 1);
                pad(); os << "} while (" << StmtToOneLine(dw->Cond()) << ");\\l";
                break;
            }
            case SNodeKind::kFor: {
                auto *f = node->as<SFor>();
                pad(); os << "for (...) {\\l";
                DumpSNode(f->Body(), os, indent + 1);
                pad(); os << "}\\l";
                break;
            }
            case SNodeKind::kSwitch: {
                auto *sw = node->as<SSwitch>();
                pad(); os << "switch (" << StmtToOneLine(sw->Discriminant()) << ") {\\l";
                for (size_t i = 0; i < sw->Cases().size(); ++i) {
                    const auto &c = sw->Cases()[i];
                    pad(); os << "  case " << StmtToOneLine(c.value) << ":\\l";
                    DumpSNode(c.body, os, indent + 2);
                }
                if (sw->DefaultBody()) {
                    pad(); os << "  default:\\l";
                    DumpSNode(sw->DefaultBody(), os, indent + 2);
                }
                pad(); os << "}\\l";
                break;
            }
            case SNodeKind::kLabel: {
                auto *lbl = node->as<SLabel>();
                pad(); os << lbl->Name() << ":\\l";
                DumpSNode(lbl->Body(), os, indent);
                break;
            }
            case SNodeKind::kGoto: {
                auto *g = node->as<SGoto>();
                pad(); os << "goto " << g->Target() << ";\\l";
                break;
            }
            case SNodeKind::kBreak:
                pad(); os << "break;\\l";
                break;
            case SNodeKind::kContinue:
                pad(); os << "continue;\\l";
                break;
            case SNodeKind::kReturn:
                pad(); os << "return;\\l";
                break;
            }
        }

    } // namespace

    // -----------------------------------------------------------------------
    // EmitCfgDot — Cfg (CfgBlock-based) visualization
    // -----------------------------------------------------------------------
    void EmitCfgDot(const Cfg &cfg, llvm::raw_ostream &os) {
        os << "digraph Cfg {\n";
        os << "  node [shape=box, fontname=\"Courier\", fontsize=10];\n";
        os << "  edge [fontname=\"Courier\", fontsize=9];\n";

        // Entry arrow
        os << "  entry [shape=point];\n";
        os << "  entry -> B" << cfg.entry << ";\n";

        for (size_t i = 0; i < cfg.blocks.size(); ++i) {
            const auto &blk = cfg.blocks[i];

            // Node label: B<idx> header, then label: and statements in order
            os << "  B" << i << " [label=\"B" << i << "\\l---\\l";
            if (!blk.label.empty()) {
                os << blk.label << ":\\l";
            }
            for (const auto *s : blk.stmts) {
                os << StmtToOneLine(s) << "\\l";
            }

            // Synthesize the terminal statement that was popped by
            // ResolveEdges (goto / if-goto / switch / return).
            auto succ_label = [&](size_t idx) -> std::string {
                if (idx < cfg.blocks.size() && !cfg.blocks[idx].label.empty())
                    return cfg.blocks[idx].label;
                return "B" + std::to_string(idx);
            };

            if (blk.is_conditional && blk.branch_cond && blk.succs.size() == 2) {
                os << "if (" << StmtToOneLine(blk.branch_cond) << ") "
                   << "goto " << succ_label(blk.taken_succ) << "; "
                   << "else goto " << succ_label(blk.fallthrough_succ)
                   << ";\\l";
            } else if (!blk.switch_cases.empty()) {
                // switch terminal already in stmts (not popped)
            } else if (blk.succs.size() == 1) {
                os << "goto " << succ_label(blk.succs[0]) << ";\\l";
            } else if (blk.succs.empty() && blk.stmts.empty()) {
                os << "(empty)\\l";
            }
            // succs.empty() with stmts = block ends with return (already in stmts)

            os << "\", fontsize=8];\n";

            // Edges
            if (blk.is_conditional && blk.succs.size() == 2) {
                // Conditional: taken (green) and fallthrough (red)
                for (size_t j = 0; j < blk.succs.size(); ++j) {
                    bool is_taken = (blk.succs[j] == blk.taken_succ);
                    os << "  B" << i << " -> B" << blk.succs[j];
                    if (is_taken) {
                        os << " [label=\"true\", color=green]";
                    } else {
                        os << " [label=\"false\", color=red]";
                    }
                    os << ";\n";
                }
            } else if (!blk.switch_cases.empty()) {
                // Switch edges with case labels
                std::unordered_set< size_t > labeled;
                for (const auto &sc : blk.switch_cases) {
                    if (sc.succ_index < blk.succs.size()) {
                        os << "  B" << i << " -> B" << blk.succs[sc.succ_index]
                           << " [label=\"case " << sc.value << "\"];\n";
                        labeled.insert(sc.succ_index);
                    }
                }
                // Unlabeled successors (default)
                for (size_t j = 0; j < blk.succs.size(); ++j) {
                    if (!labeled.count(j)) {
                        os << "  B" << i << " -> B" << blk.succs[j]
                           << " [label=\"default\", style=dashed];\n";
                    }
                }
            } else {
                // Unconditional edges
                for (auto succ : blk.succs) {
                    os << "  B" << i << " -> B" << succ << ";\n";
                }
            }
        }

        os << "}\n";
    }

    // -----------------------------------------------------------------------
    // EmitCGraphDot — CGraph (CNode-based) visualization
    // -----------------------------------------------------------------------
    void EmitCGraphDot(const detail::CGraph &g, llvm::raw_ostream &os) {
        os << "digraph CGraph {\n";
        os << "  node [shape=box, fontname=\"Courier\", fontsize=10];\n";
        os << "  edge [fontname=\"Courier\", fontsize=9];\n";

        // Entry arrow
        os << "  entry [shape=point];\n";
        // Find the actual entry (may have been collapsed into another)
        size_t entry_id = g.entry;
        if (entry_id < g.nodes.size() && g.nodes[entry_id].collapsed) {
            // Walk collapsed_into chain
            while (entry_id < g.nodes.size() && g.nodes[entry_id].collapsed
                   && g.nodes[entry_id].collapsed_into != detail::CNode::kNone) {
                entry_id = g.nodes[entry_id].collapsed_into;
            }
        }
        os << "  entry -> N" << entry_id << ";\n";

        for (const auto &n : g.nodes) {
            if (n.collapsed) continue;

            // Node label: N<id> header, then label: and statements in order,
            // followed by structured content from fold rules
            os << "  N" << n.id << " [label=\"N" << n.id << "\\l---\\l";
            if (!n.label.empty()) {
                os << n.label << ":\\l";
            }
            for (const auto *s : n.stmts) {
                os << StmtToOneLine(s) << "\\l";
            }
            if (n.structured) {
                if (!n.stmts.empty()) os << "---\\l";
                DumpSNode(n.structured, os, 0);
            }

            // Show synthesized control flow from edge metadata.
            // This is more useful for debugging than the raw terminal
            // stmt, since it reflects the current graph state (edges
            // may have been added/removed/marked by fold rules).
            if (!n.structured) {
                auto node_label = [&](size_t id) -> std::string {
                    if (id < g.nodes.size() && !g.nodes[id].label.empty())
                        return g.nodes[id].label;
                    return "N" + std::to_string(id);
                };
                if (n.is_conditional && n.branch_cond && n.succs.size() == 2) {
                    if (!n.stmts.empty()) os << "---\\l";
                    os << "if (" << StmtToOneLine(n.branch_cond) << ")\\l"
                       << "  T: goto " << node_label(n.succs[1]);
                    if (n.IsGotoOut(1)) os << " [G]";
                    if (n.IsBackEdge(1)) os << " [B]";
                    os << "\\l"
                       << "  F: goto " << node_label(n.succs[0]);
                    if (n.IsGotoOut(0)) os << " [G]";
                    if (n.IsBackEdge(0)) os << " [B]";
                    os << "\\l";
                } else if (!n.switch_cases.empty()) {
                    if (!n.stmts.empty()) os << "---\\l";
                    os << "switch:";
                    for (const auto &sc : n.switch_cases) {
                        os << " " << sc.value << "->N";
                        if (sc.succ_index < n.succs.size())
                            os << n.succs[sc.succ_index];
                        else
                            os << "?";
                    }
                    os << "\\l";
                } else if (n.succs.size() == 1) {
                    if (!n.stmts.empty()) os << "---\\l";
                    os << "goto " << node_label(n.succs[0]);
                    if (n.IsGotoOut(0)) os << " [G]";
                    if (n.IsBackEdge(0)) os << " [B]";
                    os << "\\l";
                } else if (n.succs.size() > 1) {
                    if (!n.stmts.empty()) os << "---\\l";
                    for (size_t si = 0; si < n.succs.size(); ++si) {
                        os << "succ[" << si << "]: " << node_label(n.succs[si]);
                        if (n.IsGotoOut(si)) os << " [G]";
                        if (n.IsBackEdge(si)) os << " [B]";
                        os << "\\l";
                    }
                } else if (n.succs.empty() && n.stmts.empty()) {
                    os << "(empty)\\l";
                }
            }

            os << "\", fontsize=8];\n";

            // Edges with flag styling
            for (size_t i = 0; i < n.succs.size(); ++i) {
                os << "  N" << n.id << " -> N" << n.succs[i];

                bool is_goto = n.IsGotoOut(i);
                bool is_back = n.IsBackEdge(i);
                bool is_exit = n.IsLoopExit(i);

                if (is_back) {
                    os << " [style=bold, color=blue, label=\"back\"]";
                } else if (is_goto) {
                    os << " [style=dashed, color=gray, label=\"goto\"]";
                } else if (is_exit) {
                    os << " [style=dotted, color=orange, label=\"loop-exit\"]";
                }

                os << ";\n";
            }
        }

        os << "}\n";
    }

    // -----------------------------------------------------------------------
    // CGraphDotTracer — numbered step dumps
    // -----------------------------------------------------------------------
    void CGraphDotTracer::Dump(const detail::CGraph &g, llvm::StringRef rule_name,
                               bool is_phase_boundary) {
        if (!enabled && !audit) return;

        // Only emit DOT at phase boundaries — reduces 90+ files to ~15.
        if (enabled && is_phase_boundary) {
            // Format: <fn_name>.step_<NNN>.<rule>.dot
            std::ostringstream filename;
            filename << fn_name << ".step_"
                     << std::setfill('0') << std::setw(3) << step
                     << "." << rule_name.str() << ".dot";

            std::error_code ec;
            llvm::raw_fd_ostream out(filename.str(), ec, llvm::sys::fs::OF_Text);
            if (ec) {
                LOG(WARNING) << "CGraphDotTracer: failed to write " << filename.str()
                             << ": " << ec.message() << "\n";
            } else {
                EmitCGraphDot(g, out);
            }
        }

        ++step;
        AuditAfterFold(g, rule_name);
    }

    void CGraphDotTracer::AuditAfterFold(const detail::CGraph &g,
                                          llvm::StringRef rule_name) {
        if (!audit) return;

        size_t current = CountCGraphStmts(g);
        if (current < original_stmt_count) {
            LOG(WARNING) << "STMT DROP after " << rule_name.str()
                         << ": " << current << " / " << original_stmt_count
                         << " (lost " << (original_stmt_count - current) << ")\n";

            // Pointer-level diff: identify exactly which stmts were lost
            if (!baseline_stmts.empty()) {
                std::unordered_set<const clang::Stmt *> current_stmts;
                CollectCGraphStmtPtrs(g, current_stmts);
                ReportMissingStmts(baseline_stmts, current_stmts, rule_name);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Statement counting utilities
    // -----------------------------------------------------------------------

    size_t CountCfgStmts(const Cfg &cfg) {
        size_t total = 0;
        for (const auto &blk : cfg.blocks) {
            total += blk.stmts.size();
        }
        return total;
    }

    /// Count expression-stmts embedded in a condition via comma-operator chains.
    /// E.g. `(a = 1, b = 2, cond)` has 2 embedded stmts (the assignments).
    /// A non-comma expression counts as 0 (the condition itself is structural).
    /// A SwitchStmt counts as 1 (FoldSwitch strips + rebuilds it).
    static size_t CountCommaChainStmts(const clang::Expr *e) {
        if (!e) return 0;
        if (auto *bo = llvm::dyn_cast<clang::BinaryOperator>(e)) {
            if (bo->getOpcode() == clang::BO_Comma) {
                // Left side is an embedded stmt, recurse right for more
                return 1 + CountCommaChainStmts(bo->getRHS());
            }
        }
        return 0;
    }

    size_t CountSNodeStmts(const SNode *root) {
        if (!root) return 0;

        switch (root->Kind()) {
        case SNodeKind::kBlock:
            return root->as<SBlock>()->Stmts().size();
        case SNodeKind::kSeq: {
            size_t total = 0;
            for (const auto *child : root->as<SSeq>()->Children())
                total += CountSNodeStmts(child);
            return total;
        }
        case SNodeKind::kIfThenElse: {
            auto *ite = root->as<SIfThenElse>();
            return CountSNodeStmts(ite->ThenBranch())
                 + CountSNodeStmts(ite->ElseBranch());
        }
        case SNodeKind::kWhile:
            return CountCommaChainStmts(root->as<SWhile>()->Cond())
                 + CountSNodeStmts(root->as<SWhile>()->Body());
        case SNodeKind::kDoWhile:
            return CountCommaChainStmts(root->as<SDoWhile>()->Cond())
                 + CountSNodeStmts(root->as<SDoWhile>()->Body());
        case SNodeKind::kFor:
            return CountSNodeStmts(root->as<SFor>()->Body());
        case SNodeKind::kSwitch: {
            auto *sw = root->as<SSwitch>();
            // Count 1 for the discriminant — compensates for the original
            // SwitchStmt that FoldSwitch strips from the head block's stmts.
            size_t total = sw->Discriminant() ? 1 : 0;
            for (const auto &c : sw->Cases())
                total += CountSNodeStmts(c.body);
            total += CountSNodeStmts(sw->DefaultBody());
            return total;
        }
        case SNodeKind::kLabel:
            return CountSNodeStmts(root->as<SLabel>()->Body());
        default:
            return 0;
        }
    }

    size_t CountCGraphStmts(const detail::CGraph &g) {
        size_t total = 0;
        for (const auto &n : g.nodes) {
            if (n.collapsed) continue;
            total += n.stmts.size();
            if (n.structured) {
                total += CountSNodeStmts(n.structured);
            }
        }
        return total;
    }

    // -----------------------------------------------------------------------
    // Stmt-pointer tracking
    // -----------------------------------------------------------------------

    void CollectCfgStmtPtrs(const Cfg &cfg,
                             std::unordered_set<const clang::Stmt *> &out) {
        for (const auto &blk : cfg.blocks) {
            for (auto *s : blk.stmts) {
                if (s) out.insert(s);
            }
        }
    }

    static void CollectSNodeStmtPtrsImpl(const SNode *root,
                                          std::unordered_set<const clang::Stmt *> &out) {
        if (!root) return;
        switch (root->Kind()) {
        case SNodeKind::kBlock:
            for (auto *s : root->as<SBlock>()->Stmts())
                if (s) out.insert(s);
            break;
        case SNodeKind::kSeq:
            for (const auto *child : root->as<SSeq>()->Children())
                CollectSNodeStmtPtrsImpl(child, out);
            break;
        case SNodeKind::kIfThenElse: {
            auto *ite = root->as<SIfThenElse>();
            // The condition may embed stmts via comma-operator
            if (ite->Cond()) out.insert(ite->Cond());
            CollectSNodeStmtPtrsImpl(ite->ThenBranch(), out);
            CollectSNodeStmtPtrsImpl(ite->ElseBranch(), out);
            break;
        }
        case SNodeKind::kWhile: {
            auto *w = root->as<SWhile>();
            if (w->Cond()) out.insert(w->Cond());
            CollectSNodeStmtPtrsImpl(w->Body(), out);
            break;
        }
        case SNodeKind::kDoWhile: {
            auto *dw = root->as<SDoWhile>();
            if (dw->Cond()) out.insert(dw->Cond());
            CollectSNodeStmtPtrsImpl(dw->Body(), out);
            break;
        }
        case SNodeKind::kFor: {
            auto *f = root->as<SFor>();
            if (f->Cond()) out.insert(f->Cond());
            CollectSNodeStmtPtrsImpl(f->Body(), out);
            break;
        }
        case SNodeKind::kSwitch: {
            auto *sw = root->as<SSwitch>();
            if (sw->Discriminant()) out.insert(sw->Discriminant());
            for (const auto &c : sw->Cases())
                CollectSNodeStmtPtrsImpl(c.body, out);
            CollectSNodeStmtPtrsImpl(sw->DefaultBody(), out);
            break;
        }
        case SNodeKind::kLabel:
            CollectSNodeStmtPtrsImpl(root->as<SLabel>()->Body(), out);
            break;
        default:
            break;
        }
    }

    void CollectSNodeStmtPtrs(const SNode *root,
                               std::unordered_set<const clang::Stmt *> &out) {
        CollectSNodeStmtPtrsImpl(root, out);
    }

    void CollectCGraphStmtPtrs(const detail::CGraph &g,
                                std::unordered_set<const clang::Stmt *> &out) {
        for (const auto &n : g.nodes) {
            if (n.collapsed) continue;
            for (auto *s : n.stmts)
                if (s) out.insert(s);
            if (n.structured)
                CollectSNodeStmtPtrsImpl(n.structured, out);
        }
    }

    size_t ReportMissingStmts(
            const std::unordered_set<const clang::Stmt *> &baseline,
            const std::unordered_set<const clang::Stmt *> &current,
            llvm::StringRef context_label) {
        size_t missing = 0;
        clang::LangOptions lo;
        clang::PrintingPolicy pp(lo);
        pp.SuppressInitializers = true;

        for (const auto *s : baseline) {
            if (current.count(s)) continue;
            ++missing;

            // Print a one-line summary of the lost stmt
            std::string buf;
            llvm::raw_string_ostream rso(buf);
            s->printPretty(rso, nullptr, pp);
            // Truncate to first line, max 80 chars
            auto nl = buf.find('\n');
            if (nl != std::string::npos) buf.resize(nl);
            if (buf.size() > 80) { buf.resize(77); buf += "..."; }

            LOG(WARNING) << "LOST STMT [" << context_label.str() << "]: "
                         << s->getStmtClassName() << ": " << buf << "\n";
        }
        return missing;
    }

} // namespace patchestry::ast
