/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/CGraph.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <limits>
#include <list>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace patchestry::ast {

    // Forward decl — defined later in this file; AppendSourceSuccs needs it.
    const ghidra::Operation *FindSourceTerminal(const ghidra::BasicBlock &block);

    namespace {

        std::string NodeLabel(size_t id) {
            if (id == CNode::kNone) return "none";
            std::ostringstream os;
            os << id;
            return os.str();
        }

        void AddDiagnostic(
            std::vector<std::string> &diagnostics,
            const std::string &message
        ) {
            diagnostics.push_back(message);
        }

        bool ContainsId(const std::vector<size_t> &ids, size_t id) {
            return std::find(ids.begin(), ids.end(), id) != ids.end();
        }

        std::string EdgeKey(const std::string &from, const std::string &to) {
            return from + " -> " + to;
        }

        /// Parse the hex address out of a P-Code basic block key of the form
        /// "...:HEX:...".  Minimal helper local to CGraph validation (the
        /// project-wide Utils::ParseBlockAddress is not yet on this branch).
        std::optional<uint64_t> ParseBlockAddress(const std::string &key) {
            auto p1 = key.find(':');
            if (p1 == std::string::npos) {
                return std::nullopt;
            }
            auto p2 = key.find(':', p1 + 1);
            if (p2 == std::string::npos) {
                return std::nullopt;
            }
            auto hex_str = key.substr(p1 + 1, p2 - p1 - 1);
            if (hex_str.empty()) {
                return std::nullopt;
            }
            try {
                return std::stoull(hex_str, nullptr, 16);
            } catch (...) {
                return std::nullopt;
            }
        }

        std::string SwitchCaseKey(
            const std::string &switch_block,
            bool is_default,
            int64_t value,
            const std::string &target_block,
            bool has_exit
        ) {
            std::ostringstream os;
            os << switch_block << " :: ";
            if (is_default)
                os << "default";
            else
                os << "case " << value;
            os << " -> " << target_block
               << " exit=" << (has_exit ? "true" : "false");
            return os.str();
        }

        void AppendSourceSuccs(const ghidra::Function &function,
                               const std::string &block_key,
                               std::vector<std::string> &succs) {
            if (!function.basic_blocks.contains(block_key)) return;

            const auto &block = function.basic_blocks.at(block_key);
            const auto *op = FindSourceTerminal(block);
            if (!op) return;
            const auto &blocks = function.basic_blocks;
            if (op->taken_block && blocks.contains(*op->taken_block))
                succs.push_back(*op->taken_block);
            if (op->not_taken_block && blocks.contains(*op->not_taken_block))
                succs.push_back(*op->not_taken_block);
            if (op->target_block && blocks.contains(*op->target_block))
                succs.push_back(*op->target_block);
            for (const auto &s : op->successor_blocks) {
                if (blocks.contains(s)) succs.push_back(s);
            }
            for (const auto &sc : op->switch_cases) {
                if (blocks.contains(sc.target_block))
                    succs.push_back(sc.target_block);
            }
            if (op->fallback_block && blocks.contains(*op->fallback_block))
                succs.push_back(*op->fallback_block);
        }

        std::vector<std::string>
        ComputeSourceRPO(const ghidra::Function &function) {
            std::unordered_set<std::string> visited;
            std::vector<std::string> post_order;

            struct Frame {
                std::string key;
                size_t child_idx = 0;
                std::vector<std::string> succs;
            };
            std::vector<Frame> stack;

            if (!function.entry_block.empty()
                && function.basic_blocks.contains(function.entry_block)) {
                Frame entry;
                entry.key = function.entry_block;
                AppendSourceSuccs(function, entry.key, entry.succs);
                stack.push_back(std::move(entry));
                visited.insert(function.entry_block);
            }

            while (!stack.empty()) {
                auto &top = stack.back();
                if (top.child_idx < top.succs.size()) {
                    std::string child = top.succs[top.child_idx++];
                    if (visited.insert(child).second) {
                        Frame next;
                        next.key = std::move(child);
                        AppendSourceSuccs(function, next.key, next.succs);
                        stack.push_back(std::move(next));
                    }
                } else {
                    post_order.push_back(top.key);
                    stack.pop_back();
                }
            }

            std::reverse(post_order.begin(), post_order.end());

            if (post_order.size() < function.basic_blocks.size()) {
                std::vector<std::string> unreachable;
                for (const auto &[key, _] : function.basic_blocks) {
                    if (!visited.contains(key)) unreachable.push_back(key);
                }
                // Sort by parsed (addr, idx) to match CGraphBuilder's
                // ordering — purely-lexicographic sort puts "ram:10:..."
                // before "ram:2:..." and the validator oracle would
                // disagree with the actual graph.
                auto parse_key = [](const std::string &k)
                    -> std::pair<uint64_t, uint64_t> {
                    auto p1 = k.find(':');
                    if (p1 == std::string::npos) return {0, 0};
                    auto p2 = k.find(':', p1 + 1);
                    if (p2 == std::string::npos) return {0, 0};
                    auto p3 = k.find(':', p2 + 1);
                    uint64_t addr = 0, idx = 0;
                    try { addr = std::stoull(k.substr(p1 + 1, p2 - p1 - 1), nullptr, 16); }
                    catch (...) {}
                    try {
                        size_t end_pos = p3 == std::string::npos ? k.size() : p3;
                        idx = std::stoull(k.substr(p2 + 1, end_pos - p2 - 1));
                    } catch (...) {}
                    return {addr, idx};
                };
                std::sort(unreachable.begin(), unreachable.end(),
                    [&](const std::string &a, const std::string &b) {
                        return parse_key(a) < parse_key(b);
                    });
                post_order.insert(post_order.end(), unreachable.begin(),
                                  unreachable.end());
            }

            return post_order;
        }

        void AddOracleEdge(
            std::unordered_map<std::string, unsigned> &edges,
            const std::string &from,
            const std::string &to
        ) {
            // The oracle is a source edge set. Multiple switch cases may
            // target the same block, but CGraph intentionally stores one
            // successor edge per distinct target.
            edges.try_emplace(EdgeKey(from, to), 1);
        }

        std::unordered_map<std::string, unsigned>
        BuildOracleEdges(const ghidra::Function &function,
                         std::vector<std::string> &diagnostics) {
            std::unordered_map<std::string, unsigned> edges;
            auto rpo = ComputeSourceRPO(function);
            std::unordered_map<std::string, size_t> rpo_pos;
            for (size_t i = 0; i < rpo.size(); ++i)
                rpo_pos.emplace(rpo[i], i);

            auto add_if_valid = [&](const std::string &from,
                                    const std::optional<std::string> &to,
                                    const char *kind) {
                if (!to) return false;
                if (!function.basic_blocks.contains(*to)) {
                    AddDiagnostic(diagnostics,
                                  "source " + std::string(kind)
                                      + " edge from " + from
                                      + " targets missing block " + *to);
                    return false;
                }
                AddOracleEdge(edges, from, *to);
                return true;
            };

            for (const auto &block_key : rpo) {
                if (!function.basic_blocks.contains(block_key)) continue;
                const auto &block = function.basic_blocks.at(block_key);
                const auto *term = FindSourceTerminal(block);
                if (!term) {
                    auto pos = rpo_pos.find(block_key);
                    if (pos != rpo_pos.end() && pos->second + 1 < rpo.size())
                        AddOracleEdge(edges, block_key, rpo[pos->second + 1]);
                    continue;
                }

                using M = ghidra::Mnemonic;
                if (term->mnemonic == M::OP_BRANCH) {
                    add_if_valid(block_key, term->target_block, "branch");
                } else if (term->mnemonic == M::OP_CBRANCH) {
                    if (!add_if_valid(block_key, term->not_taken_block,
                                      "conditional-not-taken")) {
                        auto pos = rpo_pos.find(block_key);
                        if (pos != rpo_pos.end() && pos->second + 1 < rpo.size())
                            AddOracleEdge(edges, block_key,
                                          rpo[pos->second + 1]);
                    }
                    add_if_valid(block_key, term->taken_block,
                                 "conditional-taken");
                } else if (term->mnemonic == M::OP_BRANCHIND) {
                    for (const auto &sc : term->switch_cases) {
                        if (!function.basic_blocks.contains(sc.target_block)) {
                            AddDiagnostic(
                                diagnostics,
                                "source switch-case edge from " + block_key
                                    + " targets missing block "
                                    + sc.target_block);
                            continue;
                        }
                        AddOracleEdge(edges, block_key, sc.target_block);
                    }
                    add_if_valid(block_key, term->fallback_block,
                                 "switch-default");
                    for (const auto &succ : term->successor_blocks) {
                        if (!function.basic_blocks.contains(succ)) {
                            AddDiagnostic(
                                diagnostics,
                                "source switch-successor-block edge from "
                                    + block_key + " targets missing block "
                                    + succ);
                            continue;
                        }
                        AddOracleEdge(edges, block_key, succ);
                    }
                }
            }

            return edges;
        }

        std::unordered_map<std::string, unsigned>
        BuildOracleSwitchCases(
            const ghidra::Function &function,
            std::unordered_set<std::string> &switch_blocks,
            std::vector<std::string> &diagnostics
        ) {
            std::unordered_map<std::string, unsigned> cases;
            for (const auto &[block_key, block] : function.basic_blocks) {
                const auto *term = FindSourceTerminal(block);
                if (!term || term->mnemonic != ghidra::Mnemonic::OP_BRANCHIND)
                    continue;

                switch_blocks.insert(block_key);

                for (const auto &sc : term->switch_cases) {
                    if (!function.basic_blocks.contains(sc.target_block)) {
                        AddDiagnostic(
                            diagnostics,
                            "source switch case from " + block_key
                                + " targets missing block "
                                + sc.target_block);
                        continue;
                    }
                    ++cases[SwitchCaseKey(block_key, /*is_default=*/false,
                                          sc.value, sc.target_block,
                                          sc.has_exit)];
                }

                if (term->fallback_block) {
                    if (!function.basic_blocks.contains(*term->fallback_block)) {
                        AddDiagnostic(
                            diagnostics,
                            "source switch default from " + block_key
                                + " targets missing block "
                                + *term->fallback_block);
                    } else {
                        ++cases[SwitchCaseKey(block_key, /*is_default=*/true,
                                              0, *term->fallback_block,
                                              /*has_exit=*/false)];
                    }
                }

                for (const auto &succ : term->successor_blocks) {
                    if (!function.basic_blocks.contains(succ)) {
                        AddDiagnostic(
                            diagnostics,
                            "source switch successor-block case from "
                                + block_key + " targets missing block "
                                + succ);
                        continue;
                    }
                    auto addr = ParseBlockAddress(succ);
                    if (!addr) {
                        AddDiagnostic(
                            diagnostics,
                            "source switch successor-block case from "
                                + block_key
                                + " has unparseable target address " + succ);
                        continue;
                    }
                    ++cases[SwitchCaseKey(
                        block_key, /*is_default=*/false,
                        static_cast<int64_t>(*addr), succ,
                        /*has_exit=*/false)];
                }
            }
            return cases;
        }

        std::unordered_map<std::string, unsigned>
        BuildActualSwitchCases(
            const CGraph &g,
            std::unordered_set<std::string> &switch_blocks,
            std::vector<std::string> &diagnostics
        ) {
            std::unordered_map<std::string, unsigned> cases;
            for (const auto &node : g.nodes) {
                if (!node.switch_cases.empty() || node.IsSwitchOut()) {
                    if (!node.source_key.empty())
                        switch_blocks.insert(node.source_key);
                }

                if (node.switch_cases.empty())
                    continue;
                if (node.source_key.empty())
                    continue;

                for (const auto &sc : node.switch_cases) {
                    if (sc.succ_index >= node.succs.size()) {
                        // Core structural validation reports this too.
                        continue;
                    }
                    size_t succ = node.succs[sc.succ_index];
                    if (succ >= g.nodes.size()) continue;
                    const auto &target = g.nodes[succ];
                    if (target.source_key.empty()) {
                        AddDiagnostic(
                            diagnostics,
                            "emitted switch case from " + node.source_key
                                + " targets node " + NodeLabel(succ)
                                + " without source block key");
                        continue;
                    }

                    ++cases[SwitchCaseKey(node.source_key, sc.is_default,
                                          sc.value, target.source_key,
                                          sc.has_exit)];
                }
            }
            return cases;
        }

    } // namespace

    const ghidra::Operation *FindSourceTerminal(const ghidra::BasicBlock &block) {
        if (block.ordered_operations.empty()) {
            return nullptr;
        }
        const auto &last_key = block.ordered_operations.back();
        if (!block.operations.contains(last_key)) {
            return nullptr;
        }
        const auto &op = block.operations.at(last_key);
        using M        = ghidra::Mnemonic;
        if (op.mnemonic == M::OP_BRANCH || op.mnemonic == M::OP_CBRANCH
            || op.mnemonic == M::OP_BRANCHIND || op.mnemonic == M::OP_RETURN
            || op.mnemonic == M::OP_TAIL_CALL)
        {
            return &op;
        }
        return nullptr;
    }

    CGraphValidationReport ValidateCGraph(
        const CGraph &g,
        const ghidra::Function *source
    ) {
        CGraphValidationReport report;
        report.node_count = g.nodes.size();

        if (g.nodes.empty()) {
            AddDiagnostic(report.diagnostics, "graph has no nodes");
            return report;
        }

        if (g.entry >= g.nodes.size()) {
            AddDiagnostic(report.diagnostics,
                          "entry node " + NodeLabel(g.entry)
                              + " is outside node range");
        }

        for (size_t id = 0; id < g.nodes.size(); ++id) {
            const auto &node = g.nodes[id];
            if (node.id != id) {
                AddDiagnostic(report.diagnostics,
                              "node slot " + NodeLabel(id)
                                  + " has mismatched node.id "
                                  + NodeLabel(node.id));
            }
            if (node.source_key.empty()) {
                AddDiagnostic(report.diagnostics,
                              "node " + NodeLabel(id)
                                  + " has no source block key");
            } else {
                ++report.emitted_blocks;
            }

            if (node.IsCollapsed()) {
                ++report.collapsed_nodes;
                if (node.collapsed_into >= g.nodes.size()) {
                    AddDiagnostic(report.diagnostics,
                                  "node " + NodeLabel(id)
                                      + " collapsed_into invalid node "
                                      + NodeLabel(node.collapsed_into));
                } else if (node.collapsed_into == id) {
                    AddDiagnostic(report.diagnostics,
                                  "node " + NodeLabel(id)
                                      + " is collapsed into itself");
                }
            } else {
                ++report.active_nodes;
            }

            if (node.edge_flags.size() != node.succs.size()) {
                AddDiagnostic(report.diagnostics,
                              "node " + NodeLabel(id)
                                  + " has " + NodeLabel(node.succs.size())
                                  + " successor(s) but "
                                  + NodeLabel(node.edge_flags.size())
                                  + " edge flag(s)");
            }

            report.edge_count += node.succs.size();
            report.emitted_edges += node.succs.size();
            if (node.is_conditional) {
                ++report.conditional_nodes;
                if (node.succs.size() != 2) {
                    AddDiagnostic(report.diagnostics,
                                  "conditional node " + NodeLabel(id)
                                      + " has " + NodeLabel(node.succs.size())
                                      + " successor(s)");
                }
                if (!node.branch_cond) {
                    AddDiagnostic(report.diagnostics,
                                  "conditional node " + NodeLabel(id)
                                      + " has no branch condition");
                }
                if (node.branch_roles.normalized) {
                    ++report.normalized_conditions;
                    if (node.branch_roles.merge >= g.nodes.size()) {
                        AddDiagnostic(report.diagnostics,
                                      "normalized conditional node "
                                          + NodeLabel(id)
                                          + " has invalid merge role "
                                          + NodeLabel(node.branch_roles.merge));
                    } else if (node.succs.size() == 2
                               && node.branch_roles.merge != node.succs[0]) {
                        AddDiagnostic(report.diagnostics,
                                      "normalized conditional node "
                                          + NodeLabel(id)
                                          + " does not have merge on succs[0]");
                    }
                }
                if (node.branch_roles.swapped)
                    ++report.branch_swaps;
                if (node.branch_roles.condition_negated)
                    ++report.condition_negations;
                if (node.branch_roles.condition_negated
                    && !node.branch_roles.swapped) {
                    AddDiagnostic(report.diagnostics,
                                  "conditional node " + NodeLabel(id)
                                      + " negated condition without branch swap");
                }
            }

            if (node.region_kind == CNode::RegionKind::kIrreducible)
                ++report.irreducible_regions;

            if (node.IsSwitchOut()) {
                ++report.switch_nodes;
                if (!node.branch_cond) {
                    AddDiagnostic(report.diagnostics,
                                  "switch node " + NodeLabel(id)
                                      + " has no discriminant");
                }
            }

            for (size_t si = 0; si < node.succs.size(); ++si) {
                size_t succ = node.succs[si];
                if (succ >= g.nodes.size()) {
                    AddDiagnostic(report.diagnostics,
                                  "node " + NodeLabel(id)
                                      + " has invalid successor "
                                      + NodeLabel(succ));
                    continue;
                }
                const auto &succ_node = g.nodes[succ];
                if (!ContainsId(succ_node.preds, id)) {
                    AddDiagnostic(report.diagnostics,
                                  "node " + NodeLabel(id)
                                      + " successor " + NodeLabel(succ)
                                      + " does not list reverse predecessor");
                }
            }

            for (size_t pred : node.preds) {
                if (pred >= g.nodes.size()) {
                    AddDiagnostic(report.diagnostics,
                                  "node " + NodeLabel(id)
                                      + " has invalid predecessor "
                                      + NodeLabel(pred));
                    continue;
                }
                const auto &pred_node = g.nodes[pred];
                if (!ContainsId(pred_node.succs, id)) {
                    AddDiagnostic(report.diagnostics,
                                  "node " + NodeLabel(id)
                                      + " predecessor " + NodeLabel(pred)
                                      + " does not list forward successor");
                }
            }

            std::unordered_set<size_t> seen_preds;
            for (size_t pred : node.preds) {
                if (!seen_preds.insert(pred).second) {
                    AddDiagnostic(report.diagnostics,
                                  "node " + NodeLabel(id)
                                      + " has duplicate predecessor "
                                      + NodeLabel(pred));
                }
            }

            for (const auto &sc : node.switch_cases) {
                if (sc.succ_index >= node.succs.size()) {
                    AddDiagnostic(report.diagnostics,
                                  "switch node " + NodeLabel(id)
                                      + " case target index "
                                      + NodeLabel(sc.succ_index)
                                      + " is outside successor range "
                                      + NodeLabel(node.succs.size()));
                }
            }
        }

        std::unordered_map<std::string, unsigned> expected_edges;
        if (source) {
            report.input_blocks = source->basic_blocks.size();
            std::unordered_set<std::string> emitted_blocks;
            for (const auto &node : g.nodes) {
                if (!node.source_key.empty())
                    emitted_blocks.insert(node.source_key);
            }
            for (const auto &[block_key, _] : source->basic_blocks) {
                if (!emitted_blocks.contains(block_key)) {
                    report.missing_blocks.push_back(block_key);
                    AddDiagnostic(report.diagnostics,
                                  "missing source block " + block_key);
                }
            }
            for (const auto &block_key : emitted_blocks) {
                if (!source->basic_blocks.contains(block_key)) {
                    report.extra_blocks.push_back(block_key);
                    AddDiagnostic(report.diagnostics,
                                  "extra graph block " + block_key);
                }
            }
            std::sort(report.missing_blocks.begin(),
                      report.missing_blocks.end());
            std::sort(report.extra_blocks.begin(), report.extra_blocks.end());

            expected_edges = BuildOracleEdges(*source, report.diagnostics);

            std::unordered_set<std::string> expected_switches;
            auto expected_cases = BuildOracleSwitchCases(
                *source, expected_switches, report.diagnostics);
            std::unordered_set<std::string> actual_switches;
            auto actual_cases = BuildActualSwitchCases(
                g, actual_switches, report.diagnostics);

            report.input_switches = expected_switches.size();
            report.emitted_switches = actual_switches.size();
            report.input_cases = 0;
            for (const auto &[_, count] : expected_cases)
                report.input_cases += count;
            report.emitted_cases = 0;
            for (const auto &[_, count] : actual_cases)
                report.emitted_cases += count;

            for (const auto &sw : expected_switches) {
                if (!actual_switches.contains(sw)) {
                    report.missing_switches.push_back(sw);
                    AddDiagnostic(report.diagnostics,
                                  "missing emitted switch for source block "
                                      + sw);
                }
            }
            for (const auto &sw : actual_switches) {
                if (!expected_switches.contains(sw)) {
                    report.extra_switches.push_back(sw);
                    AddDiagnostic(report.diagnostics,
                                  "extra emitted switch for source block "
                                      + sw);
                }
            }

            for (const auto &[case_key, expected_count] : expected_cases) {
                unsigned actual_count = 0;
                if (auto it = actual_cases.find(case_key);
                    it != actual_cases.end())
                    actual_count = it->second;
                if (actual_count < expected_count) {
                    report.missing_cases.push_back(case_key);
                    AddDiagnostic(report.diagnostics,
                                  "missing switch case " + case_key
                                      + " expected "
                                      + NodeLabel(expected_count)
                                      + " occurrence(s), saw "
                                      + NodeLabel(actual_count));
                }
            }
            for (const auto &[case_key, actual_count] : actual_cases) {
                unsigned expected_count = 0;
                if (auto it = expected_cases.find(case_key);
                    it != expected_cases.end())
                    expected_count = it->second;
                if (expected_count == 0) {
                    report.extra_cases.push_back(case_key);
                    AddDiagnostic(report.diagnostics,
                                  "extra emitted switch case " + case_key);
                } else if (actual_count > expected_count) {
                    report.duplicated_cases.push_back(case_key);
                    AddDiagnostic(report.diagnostics,
                                  "duplicated emitted switch case "
                                      + case_key + " expected "
                                      + NodeLabel(expected_count)
                                      + " occurrence(s), saw "
                                      + NodeLabel(actual_count));
                }
                // The else-if branch above already records every genuine
                // duplicate.  An earlier unconditional `if (actual_count
                // > 1)` block that also pushed to duplicated_cases was a
                // bug: it double-classified extras (expected_count == 0)
                // and falsely flagged exact-match cases (expected_count
                // > 1 && actual_count == expected_count, e.g. fallthrough
                // chains repeating a value the source also repeats).
            }

            std::sort(report.missing_switches.begin(),
                      report.missing_switches.end());
            std::sort(report.extra_switches.begin(),
                      report.extra_switches.end());
            std::sort(report.missing_cases.begin(),
                      report.missing_cases.end());
            std::sort(report.extra_cases.begin(), report.extra_cases.end());
            std::sort(report.duplicated_cases.begin(),
                      report.duplicated_cases.end());
        } else {
            // Dedupe by distinct (from, to) to match AddOracleEdge above
            // and actual_edges below — CGraph stores one successor edge
            // per distinct target, so multiplicities in source_edges
            // (e.g., switch cases sharing a target) must not inflate
            // the expected count.
            for (const auto &edge : g.source_edges) {
                if (edge.from_key.empty() || edge.to_key.empty()) {
                    AddDiagnostic(report.diagnostics,
                                  "source edge has empty endpoint for reason "
                                      + edge.reason);
                    continue;
                }
                expected_edges.try_emplace(
                    EdgeKey(edge.from_key, edge.to_key), 1);
            }
        }
        report.input_edges = expected_edges.size();

        std::unordered_map<std::string, unsigned> actual_edges;
        for (const auto &node : g.nodes) {
            if (node.source_key.empty()) continue;
            for (size_t succ : node.succs) {
                if (succ >= g.nodes.size()) continue;
                const auto &succ_node = g.nodes[succ];
                if (succ_node.source_key.empty()) continue;
                actual_edges.try_emplace(
                    EdgeKey(node.source_key, succ_node.source_key), 1);
            }
        }

        for (const auto &[edge, expected_count] : expected_edges) {
            unsigned actual_count = 0;
            if (auto it = actual_edges.find(edge); it != actual_edges.end())
                actual_count = it->second;
            if (actual_count < expected_count) {
                report.missing_edges.push_back(edge);
                AddDiagnostic(report.diagnostics,
                              "missing source edge " + edge + " expected "
                                  + NodeLabel(expected_count) + " occurrence(s), saw "
                                  + NodeLabel(actual_count));
            }
        }
        for (const auto &[edge, actual_count] : actual_edges) {
            unsigned expected_count = 0;
            if (auto it = expected_edges.find(edge); it != expected_edges.end())
                expected_count = it->second;
            if (expected_count == 0) {
                report.extra_edges.push_back(edge);
                AddDiagnostic(report.diagnostics,
                              "extra graph edge " + edge);
            } else if (actual_count > expected_count) {
                report.duplicated_edges.push_back(edge);
                AddDiagnostic(report.diagnostics,
                              "duplicated graph edge " + edge + " expected "
                                  + NodeLabel(expected_count)
                                  + " occurrence(s), saw "
                                  + NodeLabel(actual_count));
            }
        }
        std::sort(report.missing_edges.begin(), report.missing_edges.end());
        std::sort(report.extra_edges.begin(), report.extra_edges.end());
        std::sort(report.duplicated_edges.begin(),
                  report.duplicated_edges.end());

        return report;
    }

    // IdentifyInternal — absorb component nodes into a hierarchical
    // structured block (structural).
    size_t CGraph::IdentifyInternal(const std::vector<size_t> &ids,
                                        CNode::BlockType type,
                                        std::vector< SNode * > snodes) {
        if (ids.empty()) return CNode::kNone;
        size_t rep = ids[0];
        nodes[rep].structured = std::move(snodes);
        nodes[rep].block_type = type;
        nodes[rep].branch_roles = CNode::BranchRoles{};

        std::unordered_set<size_t> idset(ids.begin(), ids.end());

        // Invariant: succs[] and edge_flags[] are indexed in parallel.
        // The validator (Validate, ~line 469) reports a diagnostic when
        // this breaks, so guard the OOB read at line 794 below before
        // we trust the indices.  Guarded by NDEBUG so the loop variable
        // is not flagged as unused under -Werror=unused-variable in
        // release builds where assert() becomes a no-op.
#ifndef NDEBUG
        for (size_t nid : ids) {
            assert(nodes[nid].succs.size() == nodes[nid].edge_flags.size()
                   && "CNode succs/edge_flags size mismatch");
        }
#endif

        // Collect external predecessors
        std::vector<size_t> ext_preds;
        for (size_t nid : ids) {
            for (size_t p : nodes[nid].preds) {
                if (idset.count(p) == 0) ext_preds.push_back(p);
            }
        }

        // Collect external successors (first-seen flags win)
        std::vector<size_t> ext_succs;
        std::vector<uint32_t> ext_succ_flags;
        for (size_t nid : ids) {
            for (size_t i = 0; i < nodes[nid].succs.size(); ++i) {
                size_t s = nodes[nid].succs[i];
                if (idset.count(s) == 0) {
                    if (std::find(ext_succs.begin(), ext_succs.end(), s) == ext_succs.end()) {
                        ext_succs.push_back(s);
                        ext_succ_flags.push_back(nodes[nid].edge_flags[i]);
                    }
                }
            }
        }

        // Mark non-rep nodes as collapsed, record as children
        nodes[rep].children.clear();
        for (size_t nid : ids) {
            if (nid != rep) {
                nodes[nid].collapsed_into = rep;
                nodes[rep].children.push_back(nid);
            }
        }

        // Rewire edges
        for (size_t nid : ids) {
            for (size_t s : nodes[nid].succs) {
                if (idset.count(s) == 0) {
                    auto &p = nodes[s].preds;
                    p.erase(std::remove(p.begin(), p.end(), nid), p.end());
                }
            }
            for (size_t p : nodes[nid].preds) {
                if (idset.count(p) == 0) {
                    auto &ss = nodes[p].succs;
                    for (size_t i = 0; i < ss.size(); ++i) {
                        if (ss[i] == nid) {
                            ss[i] = rep;
                        }
                    }
                }
            }
        }

        // Deduplicate predecessor succs
        for (size_t p : ext_preds) {
            auto &ss = nodes[p].succs;
            auto &sf = nodes[p].edge_flags;
            std::unordered_set<size_t> seen;
            size_t write = 0;
            for (size_t i = 0; i < ss.size(); ++i) {
                if (seen.insert(ss[i]).second) {
                    ss[write] = ss[i];
                    sf[write] = sf[i];
                    ++write;
                }
            }
            ss.resize(write);
            sf.resize(write);
        }

        // Deduplicate convergent ext_succs.
        //
        // When two collapsed nodes independently exit to different
        // external blocks that lie on the same sequential path (e.g.,
        // switch-default→N17 and loop-exit→N14→...→N17), both appear
        // as ext_succs.  The representative isn't a true conditional —
        // the two exits are sequential, not alternatives.
        //
        // Detection: for each pair (A, B) in ext_succs, walk from A
        // through single-in/single-out blocks.  If B is reachable, A→B
        // is sequential — drop B from ext_succs (A will eventually
        // reach it via the graph).
        if (ext_succs.size() == 2) {
            // Check if `target` is reachable from `from` through a
            // bounded forward walk.  Follows single-successor chains
            // and also handles conditionals where ALL branches converge
            // to the target (if-then-else diamond → same exit).
            auto reaches = [&](size_t from, size_t target, size_t limit = 20) -> bool {
                size_t cur = from;
                for (size_t step = 0; step < limit; ++step) {
                    auto &cn = nodes[cur];
                    if (cn.IsCollapsed()) break;
                    // Never reason through the node being formed or any
                    // member of the collapse set: their succs[] are
                    // mid-rewrite and still carry pre-collapse edges.  A
                    // walk that crosses a loop back-edge into `rep` would
                    // read a stale successor and wrongly conclude two
                    // external exits are sequential — dropping a loop's
                    // real exit edge (issue #259).
                    if (cur == rep || idset.count(cur)) break;
                    for (size_t s : cn.succs) {
                        if (s == target) return true;
                    }
                    if (cn.succs.size() == 1) {
                        cur = cn.succs[0];
                        if (cur == from) break;
                        continue;
                    }
                    // Conditional: check if ALL successors reach the
                    // target within a short walk (diamond pattern).
                    if (cn.succs.size() == 2) {
                        bool all_reach = true;
                        for (size_t s : cn.succs) {
                            bool found = (s == target);
                            if (!found && s != rep && idset.count(s) == 0) {
                                // One-hop check from each branch.  Skip
                                // `rep` / collapse-set members — their
                                // succs[] are stale mid-rewrite (#259).
                                auto &sn = nodes[s];
                                if (!sn.IsCollapsed()) {
                                    for (size_t ss : sn.succs) {
                                        if (ss == target) { found = true; break; }
                                    }
                                }
                            }
                            if (!found) { all_reach = false; break; }
                        }
                        if (all_reach) return true;
                    }
                    break;
                }
                return false;
            };

            if (reaches(ext_succs[0], ext_succs[1])) {
                // A reaches B — keep only A
                auto &bp = nodes[ext_succs[1]].preds;
                bp.erase(std::remove(bp.begin(), bp.end(), rep), bp.end());
                for (size_t nid : ids) {
                    bp.erase(std::remove(bp.begin(), bp.end(), nid), bp.end());
                }
                ext_succs.erase(ext_succs.begin() + 1);
                ext_succ_flags.erase(ext_succ_flags.begin() + 1);
            } else if (reaches(ext_succs[1], ext_succs[0])) {
                // B reaches A — keep only B
                auto &ap = nodes[ext_succs[0]].preds;
                ap.erase(std::remove(ap.begin(), ap.end(), rep), ap.end());
                for (size_t nid : ids) {
                    ap.erase(std::remove(ap.begin(), ap.end(), nid), ap.end());
                }
                ext_succs.erase(ext_succs.begin());
                ext_succ_flags.erase(ext_succ_flags.begin());
            } else {
                // Check if both reach a common descendant — if so,
                // they're fan-out paths to a shared merge point.
                // Find common target by checking 1-hop successors.
                //
                // A node is "stale" while IdentifyInternal is mid-collapse
                // if it is `rep` or a collapse-set member: their succs[]
                // are not rewired until the "Install edges" step below,
                // and they are the loop being formed — never a valid
                // downstream merge point.  first_succ must neither read
                // their succs[] nor return them as a target, or it drops
                // a loop's real exit edge (same hazard as issue #259).
                auto is_stale = [&](size_t nid) {
                    return nid == rep || idset.count(nid) != 0;
                };
                auto first_succ = [&](size_t nid) -> size_t {
                    auto &n = nodes[nid];
                    if (n.IsCollapsed() || n.succs.empty()) return CNode::kNone;
                    if (n.succs.size() == 1) {
                        return is_stale(n.succs[0]) ? CNode::kNone : n.succs[0];
                    }
                    // For conditionals, check if both branches go to same target
                    if (n.succs.size() == 2) {
                        // Bail if either branch is `rep`/collapse-set:
                        // their succs[] are stale mid-rewrite (#259).
                        if (is_stale(n.succs[0]) || is_stale(n.succs[1])) {
                            return CNode::kNone;
                        }
                        auto &s0 = nodes[n.succs[0]];
                        auto &s1 = nodes[n.succs[1]];
                        if (!s0.IsCollapsed() && s0.succs.size() == 1
                            && !s1.IsCollapsed() && s1.succs.size() == 1
                            && s0.succs[0] == s1.succs[0]
                            && !is_stale(s0.succs[0])) {
                            return s0.succs[0];
                        }
                    }
                    return CNode::kNone;
                };

                size_t dest0 = first_succ(ext_succs[0]);
                size_t dest1 = first_succ(ext_succs[1]);
                if (dest0 != CNode::kNone && dest0 == dest1) {
                    // Both converge — keep only the first (it will
                    // reach the common merge sequentially).
                    auto &bp = nodes[ext_succs[1]].preds;
                    bp.erase(std::remove(bp.begin(), bp.end(), rep), bp.end());
                    for (size_t nid : ids) {
                        bp.erase(std::remove(bp.begin(), bp.end(), nid), bp.end());
                    }
                    ext_succs.erase(ext_succs.begin() + 1);
                    ext_succ_flags.erase(ext_succ_flags.begin() + 1);
                }
            }
        }

        // Install edges on representative
        nodes[rep].succs = ext_succs;
        nodes[rep].edge_flags = ext_succ_flags;

        std::sort(ext_preds.begin(), ext_preds.end());
        ext_preds.erase(std::unique(ext_preds.begin(), ext_preds.end()), ext_preds.end());
        nodes[rep].preds = ext_preds;

        // Ensure rep is listed in each successor's preds
        for (size_t s : ext_succs) {
            auto &p = nodes[s].preds;
            if (std::find(p.begin(), p.end(), rep) == p.end()) {
                p.push_back(rep);
            }
        }

        nodes[rep].is_conditional = !ext_succs.empty() && ext_succs.size() == 2;
        switch (type) {
            case CNode::BlockType::kWhile:
            case CNode::BlockType::kDoWhile:
            case CNode::BlockType::kInfLoop:
                nodes[rep].region_kind = CNode::RegionKind::kLoop;
                break;
            case CNode::BlockType::kSwitch:
                nodes[rep].region_kind = CNode::RegionKind::kSwitch;
                break;
            default:
                nodes[rep].region_kind = CNode::RegionKind::kAcyclic;
                break;
        }

        nodes[rep].stmts.clear();
        nodes[rep].label.clear();

        // Preserve branch_cond from the collapsed node that owns the conditional split
        nodes[rep].branch_cond = nullptr;
        if (nodes[rep].is_conditional) {
            for (auto it = ids.rbegin(); it != ids.rend(); ++it) {
                if (nodes[*it].branch_cond) {
                    nodes[rep].branch_cond = nodes[*it].branch_cond;
                    break;
                }
            }
        }

        // Preserve switch metadata from collapsed nodes
        nodes[rep].switch_cases.clear();
        for (size_t nid : ids) {
            if (nid != rep && !nodes[nid].switch_cases.empty()) {
                nodes[rep].switch_cases = std::move(nodes[nid].switch_cases);
                if (!nodes[rep].branch_cond) {
                    nodes[rep].branch_cond = nodes[nid].branch_cond;
                }
                break;
            }
        }

        return rep;
    }



    // Detect back-edges using iterative DFS
    void MarkBackEdges(CGraph &g) {
        // Clear previous back-edge marks so stale flags from earlier
        // calls don't accumulate after topology changes.
        for (auto &n : g.nodes) {
            for (auto &f : n.edge_flags)
                f &= ~CNode::kBack;
        }

        enum Color { WHITE, GRAY, BLACK };
        std::vector<Color> color(g.nodes.size(), WHITE);

        struct Frame { size_t u; size_t i; };
        std::vector<Frame> stack;
        stack.push_back({g.entry, 0});
        color[g.entry] = GRAY;

        while (!stack.empty()) {
            // Copy out u and i — stack.push_back below may reallocate
            // the vector and dangle any reference into the previous
            // storage.  We write the advanced `i` back through the
            // index since the prior frame might have moved.
            const size_t top = stack.size() - 1;
            size_t u = stack[top].u;
            size_t i = stack[top].i;
            auto &nd = g.Node(u);
            if (i < nd.succs.size()) {
                size_t v = nd.succs[i];
                stack[top].i = i + 1;
                // Skip collapsed nodes — not part of the active graph.
                if (g.Node(v).IsCollapsed()) continue;
                if (color[v] == GRAY) {
                    nd.edge_flags[i] |= CNode::kBack;
                } else if (color[v] == WHITE) {
                    color[v] = GRAY;
                    stack.push_back({v, 0});
                }
            } else {
                color[u] = BLACK;
                stack.pop_back();
            }
        }
    }

    // LoopBody core methods

    void ClearMarks(CGraph &g, const std::vector<size_t> &body) {
        for (size_t id : body) {
            g.Node(id).mark = false;
        }
    }

    void LoopBody::FindBase(CGraph &g, std::vector<size_t> &body) const {
        body.clear();

        g.Node(head).mark = true;
        body.push_back(head);

        for (size_t t : tails) {
            if (!g.Node(t).mark) {
                g.Node(t).mark = true;
                body.push_back(t);
            }
        }

        // BFS backward from tails
        for (size_t idx = 1; idx < body.size(); ++idx) {
            auto &nd = g.Node(body[idx]);
            for (size_t p : nd.preds) {
                if (g.Node(p).mark) continue;
                if (g.Node(p).IsCollapsed()) continue;

                bool is_goto = false;
                auto &pn = g.Node(p);
                for (size_t si = 0; si < pn.succs.size(); ++si) {
                    if (pn.succs[si] == body[idx]) {
                        if (pn.IsGotoOut(si)) is_goto = true;
                        break;
                    }
                }
                if (is_goto) continue;

                g.Node(p).mark = true;
                body.push_back(p);
            }
        }
    }

    void LoopBody::LabelContainments(
        const CGraph &g, const std::vector<size_t> & /*body*/,
        const std::vector<LoopBody *> &looporder
    ) {
        for (LoopBody *lb : looporder) {
            if (lb == this) continue;
            if (!g.Node(lb->head).mark) continue;

            if (lb->immed_container == nullptr) {
                lb->immed_container = this;
            } else if (!g.Node(lb->immed_container->head).mark) {
                lb->immed_container = this;
            }
        }
    }

    void LoopBody::MergeIdenticalHeads(
        std::vector<LoopBody *> &looporder, std::list<LoopBody> &storage
    ) {
        std::sort(looporder.begin(), looporder.end(),
            [](const LoopBody *a, const LoopBody *b) { return a->head < b->head; });

        size_t write = 0;
        for (size_t read = 0; read < looporder.size(); ++read) {
            if (write > 0 && looporder[write - 1]->head == looporder[read]->head) {
                auto *dst = looporder[write - 1];
                auto *src = looporder[read];
                for (size_t t : src->tails) {
                    dst->AddTail(t);
                }
                for (auto it = storage.begin(); it != storage.end(); ++it) {
                    if (&(*it) == src) {
                        storage.erase(it);
                        break;
                    }
                }
            } else {
                looporder[write++] = looporder[read];
            }
        }
        looporder.resize(write);
    }

    void LabelLoops(
        CGraph &g, std::list<LoopBody> &loopbody, std::vector<LoopBody *> &looporder
    ) {
        for (auto &n : g.nodes) {
            if (n.IsCollapsed()) continue;
            for (size_t i = 0; i < n.succs.size(); ++i) {
                if (n.IsBackEdge(i)) {
                    size_t hd = n.succs[i];
                    loopbody.emplace_back(hd);
                    loopbody.back().AddTail(n.id);
                    looporder.push_back(&loopbody.back());
                }
            }
        }

        LoopBody::MergeIdenticalHeads(looporder, loopbody);

        for (LoopBody *lb : looporder) {
            std::vector<size_t> body;
            lb->FindBase(g, body);
            lb->unique_count = static_cast<int>(body.size());
            lb->LabelContainments(g, body, looporder);
            ClearMarks(g, body);
        }

        for (LoopBody *lb : looporder) {
            int d = 0;
            for (LoopBody *c = lb->immed_container; c != nullptr; c = c->immed_container) {
                ++d;
            }
            lb->depth = d;
        }

        std::sort(looporder.begin(), looporder.end(),
            [](const LoopBody *a, const LoopBody *b) { return *a < *b; });
    }

    // LoopBody exit detection, tail ordering, extension, exit labeling

    void LoopBody::FindExit(CGraph &g, const std::vector<size_t> &body) {
        std::vector<size_t> candidates;

        // Scan tails for exits
        for (size_t t : tails) {
            const auto &tn = g.Node(t);
            for (size_t i = 0; i < tn.succs.size(); ++i) {
                size_t s = tn.succs[i];
                if (!g.Node(s).mark && !tn.IsGotoOut(i) && !tn.IsBackEdge(i)) {
                    if (!immed_container) {
                        exit_block = s;
                        return;
                    }
                    candidates.push_back(s);
                }
            }
        }

        // Scan head and middle body nodes
        {
            const auto &hd = g.Node(body[0]);
            for (size_t i = 0; i < hd.succs.size(); ++i) {
                size_t s = hd.succs[i];
                if (!g.Node(s).mark && !hd.IsGotoOut(i) && !hd.IsBackEdge(i)) {
                    if (!immed_container) {
                        exit_block = s;
                        return;
                    }
                    candidates.push_back(s);
                }
            }
            for (size_t idx = static_cast<size_t>(unique_count); idx < body.size(); ++idx) {
                const auto &bn = g.Node(body[idx]);
                for (size_t i = 0; i < bn.succs.size(); ++i) {
                    size_t s = bn.succs[i];
                    if (!g.Node(s).mark && !bn.IsGotoOut(i) && !bn.IsBackEdge(i)) {
                        if (!immed_container) {
                            exit_block = s;
                            return;
                        }
                        candidates.push_back(s);
                    }
                }
            }
        }

        if (candidates.empty()) {
            exit_block = kNone;
            return;
        }

        if (!immed_container) {
            exit_block = candidates[0];
            return;
        }

        // Container filtering (structural)
        std::vector<size_t> container_body;
        {
            g.Node(immed_container->head).visit_count = 1;
            container_body.push_back(immed_container->head);
            for (size_t t : immed_container->tails) {
                if (g.Node(t).visit_count == 0) {
                    g.Node(t).visit_count = 1;
                    container_body.push_back(t);
                }
            }
            for (size_t idx = 1; idx < container_body.size(); ++idx) {
                auto &nd = g.Node(container_body[idx]);
                for (size_t p : nd.preds) {
                    if (g.Node(p).visit_count != 0) continue;
                    if (g.Node(p).IsCollapsed()) continue;
                    bool is_goto = false;
                    auto &pn = g.Node(p);
                    for (size_t si = 0; si < pn.succs.size(); ++si) {
                        if (pn.succs[si] == container_body[idx]) {
                            if (pn.IsGotoOut(si)) is_goto = true;
                            break;
                        }
                    }
                    if (is_goto) continue;
                    g.Node(p).visit_count = 1;
                    container_body.push_back(p);
                }
            }
        }

        exit_block = kNone;
        for (size_t c : candidates) {
            if (g.Node(c).visit_count != 0) {
                exit_block = c;
                break;
            }
        }

        if (exit_block == kNone && !candidates.empty()) {
            exit_block = candidates[0];
        }

        for (size_t nid : container_body) {
            g.Node(nid).visit_count = 0;
        }
    }

    void LoopBody::OrderTails(const CGraph &g) {
        if (tails.size() <= 1 || exit_block == kNone) return;

        for (size_t ti = 0; ti < tails.size(); ++ti) {
            const auto &tn = g.Node(tails[ti]);
            for (size_t s : tn.succs) {
                if (s == exit_block) {
                    if (ti != 0) std::swap(tails[0], tails[ti]);
                    return;
                }
            }
        }
    }

    void LoopBody::Extend(CGraph &g, std::vector<size_t> &body) const {
        std::vector<size_t> trial;

        size_t idx = 0;
        while (idx < body.size()) {
            auto &bn = g.Node(body[idx]);
            ++idx;
            for (size_t i = 0; i < bn.succs.size(); ++i) {
                if (bn.IsGotoOut(i)) continue;
                size_t s = bn.succs[i];
                auto &sn = g.Node(s);
                if (sn.mark) continue;
                if (sn.IsCollapsed()) continue;
                if (s == exit_block) continue;

                if (sn.visit_count == 0) trial.push_back(s);
                sn.visit_count++;

                if (sn.visit_count == static_cast<int>(sn.SizeIn())) {
                    sn.mark = true;
                    body.push_back(s);
                }
            }
        }

        for (size_t s : trial) {
            g.Node(s).visit_count = 0;
        }
    }

    void LoopBody::LabelExitEdges(CGraph &g, const std::vector<size_t> &body) const {
        for (size_t bid : body) {
            auto &bn = g.Node(bid);
            for (size_t i = 0; i < bn.succs.size(); ++i) {
                size_t s = bn.succs[i];
                if (!g.Node(s).mark && !bn.IsGotoOut(i)) {
                    bn.edge_flags[i] |= CNode::kLoopExit;
                }
            }
        }
    }

    // OrderLoopBodies: orchestrate full loop detection pipeline

    void OrderLoopBodies(CGraph &g, std::list<LoopBody> &loopbody) {
        std::vector<LoopBody *> looporder;
        LabelLoops(g, loopbody, looporder);
        if (loopbody.empty()) return;

        for (LoopBody *lb : looporder) {
            std::vector<size_t> body;
            lb->FindBase(g, body);
            lb->FindExit(g, body);
            lb->OrderTails(g);
            lb->Extend(g, body);
            lb->LabelExitEdges(g, body);
            ClearMarks(g, body);
        }
    }

    // LoopBody: exit mark management and update

    void LoopBody::SetExitMarks(CGraph &g, const std::vector<size_t> &body) const {
        std::unordered_set<size_t> bodyset(body.begin(), body.end());
        for (size_t nid : body) {
            auto &n = g.Node(nid);
            for (size_t i = 0; i < n.succs.size(); ++i) {
                if (bodyset.count(n.succs[i]) == 0) {
                    n.SetLoopExit(i);
                }
            }
        }
    }

    void LoopBody::ClearExitMarks(CGraph &g, const std::vector<size_t> &body) const {
        for (size_t nid : body) {
            auto &n = g.Node(nid);
            for (size_t i = 0; i < n.succs.size(); ++i) {
                n.ClearLoopExit(i);
            }
        }
    }

    bool LoopBody::Update(const CGraph &g) const { return !g.Node(head).IsCollapsed(); }

    // FloatingEdge

    std::pair<size_t, size_t> FloatingEdge::GetCurrentEdge(const CGraph &g) const {
        size_t top = top_id;
        while (top < g.nodes.size() && g.Node(top).IsCollapsed()) {
            size_t next = g.Node(top).collapsed_into;
            if (next == CNode::kNone || next == top) break;
            top = next;
        }
        size_t bot = bottom_id;
        while (bot < g.nodes.size() && g.Node(bot).IsCollapsed()) {
            size_t next = g.Node(bot).collapsed_into;
            if (next == CNode::kNone || next == bot) break;
            bot = next;
        }

        if (top >= g.nodes.size() || bot >= g.nodes.size())
            return {CNode::kNone, 0};
        if (g.Node(top).IsCollapsed() || g.Node(bot).IsCollapsed())
            return {CNode::kNone, 0};
        if (top == bot)
            return {CNode::kNone, 0};

        const auto &succs = g.Node(top).succs;
        for (size_t i = 0; i < succs.size(); ++i) {
            if (succs[i] == bot) return {top, i};
        }
        return {CNode::kNone, 0};
    }

    // TraceDAG implementation

    TraceDAG::~TraceDAG() {
        for (auto *bp : branchlist_) {
            for (auto *bt : bp->paths) {
                delete bt;
            }
            delete bp;
        }
    }

    void TraceDAG::BranchPoint::MarkPath() {
        ismark = true;
        if (parent != nullptr && !parent->ismark)
            parent->MarkPath();
    }

    int TraceDAG::BranchPoint::Distance(BranchPoint *op2) {
        for (auto *cur = this; cur != nullptr; cur = cur->parent)
            cur->ismark = false;
        for (auto *cur = op2; cur != nullptr; cur = cur->parent)
            cur->ismark = false;

        MarkPath();

        int dist = 0;
        auto *cur = op2;
        while (cur != nullptr && !cur->ismark) {
            ++dist;
            cur = cur->parent;
        }
        if (cur == nullptr) return dist;

        auto *cur2 = this;
        while (cur2 != cur) {
            ++dist;
            cur2 = cur2->parent;
        }
        return dist;
    }

    void TraceDAG::InsertActive(BlockTrace *trace) {
        trace->flags |= BlockTrace::kActive;
        activetrace_.push_back(trace);
        trace->activeiter = std::prev(activetrace_.end());
        ++activecount_;
    }

    void TraceDAG::RemoveActive(BlockTrace *trace) {
        if (trace->IsActive()) {
            activetrace_.erase(trace->activeiter);
            trace->flags &= ~BlockTrace::kActive;
            --activecount_;
        }
    }

    void TraceDAG::RemoveTrace(BlockTrace *trace) {
        RemoveActive(trace);
        if (trace->derivedbp != nullptr) {
            for (auto *bt : trace->derivedbp->paths) {
                if (bt != trace)
                    RemoveTrace(bt);
            }
        }
    }

    void TraceDAG::Initialize() {
        for (size_t root_id : rootlist_) {
            auto *bp = new BranchPoint();
            bp->top_id = root_id;
            bp->depth = 0;
            branchlist_.push_back(bp);

            auto *bt = new BlockTrace();
            bt->top = bp;
            bt->pathout = 0;
            bt->bottom_id = CNode::kNone;
            bt->dest_id = root_id;
            bp->paths.push_back(bt);
            InsertActive(bt);
        }
    }

    bool TraceDAG::CheckOpen(CGraph &g, BlockTrace *trace) {
        size_t dest = trace->dest_id;
        if (dest == CNode::kNone || dest >= g.nodes.size()) {
            return true;
        }

        auto &n = g.Node(dest);
        if (n.IsCollapsed()) {
            trace->flags |= BlockTrace::kTerminal;
            return true;
        }

        if (dest == finishblock_id_) {
            trace->flags |= BlockTrace::kTerminal;
            return true;
        }

        size_t dag_out_count = 0;
        size_t single_succ = CNode::kNone;
        for (size_t i = 0; i < n.succs.size(); ++i) {
            if (n.IsLoopDagOut(i)) {
                ++dag_out_count;
                single_succ = n.succs[i];
            }
        }

        if (dag_out_count == 0) {
            trace->flags |= BlockTrace::kTerminal;
            return true;
        }

        if (dag_out_count == 1) {
            trace->bottom_id = dest;
            trace->dest_id = single_succ;
            if (single_succ < g.nodes.size() && !g.Node(single_succ).IsCollapsed()) {
                g.Node(single_succ).visit_count += 1;
            }
            return true;
        }

        return false;
    }

    std::list<TraceDAG::BlockTrace *>::iterator
    TraceDAG::OpenBranch(CGraph &g, BlockTrace *parent) {
        size_t branch_id = parent->dest_id;
        const auto &n = g.Node(branch_id);

        auto *bp = new BranchPoint();
        bp->parent = parent->top;
        bp->pathout = parent->pathout;
        bp->top_id = branch_id;
        bp->depth = parent->top->depth + 1;
        branchlist_.push_back(bp);
        parent->derivedbp = bp;

        auto next_iter = std::next(parent->activeiter);
        RemoveActive(parent);

        int pathindex = 0;
        for (size_t i = 0; i < n.succs.size(); ++i) {
            if (!n.IsLoopDagOut(i)) continue;

            size_t succ_id = n.succs[i];
            auto &succ_node = g.Node(succ_id);

            auto *bt = new BlockTrace();
            bt->top = bp;
            bt->pathout = pathindex++;
            bt->bottom_id = branch_id;
            bt->dest_id = succ_id;
            bp->paths.push_back(bt);

            if (!succ_node.IsCollapsed() && succ_node.visit_count > 0) {
                for (auto *existing : activetrace_) {
                    if (existing->dest_id == succ_id || existing->bottom_id == succ_id) {
                        existing->edgelump += 1;
                        bt->flags |= BlockTrace::kTerminal;
                        break;
                    }
                }
                if (!bt->IsTerminal()) {
                    InsertActive(bt);
                }
            } else {
                InsertActive(bt);
            }

            if (!succ_node.IsCollapsed()) {
                succ_node.visit_count += 1;
            }
        }

        return next_iter;
    }

    bool TraceDAG::CheckRetirement(BlockTrace *trace, size_t &exitblock_id) {
        if (trace->pathout != 0) return false;
        BranchPoint *bp = trace->top;
        if (bp == nullptr) return false;

        if (bp->depth == 0) {
            for (auto *bt : bp->paths) {
                if (!bt->IsActive()) return false;
                if (!bt->IsTerminal()) return false;
            }
            exitblock_id = CNode::kNone;
            return true;
        }

        size_t outblock = CNode::kNone;
        for (auto *bt : bp->paths) {
            if (!bt->IsActive()) return false;
            if (bt->IsTerminal()) continue;
            if (outblock == bt->dest_id) continue;
            if (outblock != CNode::kNone) return false;
            outblock = bt->dest_id;
        }
        exitblock_id = outblock;
        return true;
    }

    std::list<TraceDAG::BlockTrace *>::iterator
    TraceDAG::RetireBranch(BranchPoint *bp, size_t exitblock_id) {
        std::list<BlockTrace *>::iterator next_iter = current_activeiter_;
        for (auto *bt : bp->paths) {
            if (bt->IsActive()) {
                auto it = bt->activeiter;
                if (it == next_iter) ++next_iter;
                RemoveActive(bt);
            }
        }

        if (bp->parent != nullptr) {
            for (auto *pt : bp->parent->paths) {
                if (pt->derivedbp == bp) {
                    pt->derivedbp = nullptr;
                    if (exitblock_id != CNode::kNone) {
                        pt->dest_id = exitblock_id;
                        InsertActive(pt);
                    } else {
                        pt->flags |= BlockTrace::kTerminal;
                    }
                    break;
                }
            }
        }

        return next_iter;
    }

    bool TraceDAG::BadEdgeScore::CompareFinal(const BadEdgeScore &op2) const {
        if (siblingedge != op2.siblingedge)
            return (op2.siblingedge < siblingedge);
        if (terminal != op2.terminal)
            return (terminal < op2.terminal);
        if (distance != op2.distance)
            return (distance < op2.distance);
        if (trace->top && op2.trace->top)
            return (trace->top->depth < op2.trace->top->depth);
        return false;
    }

    bool TraceDAG::BadEdgeScore::operator<(const BadEdgeScore &op2) const {
        return exitproto_id < op2.exitproto_id;
    }

    void TraceDAG::ProcessExitConflict(
        std::list<BadEdgeScore>::iterator start, std::list<BadEdgeScore>::iterator end
    ) {
        int count = 0;
        for (auto it = start; it != end; ++it) ++count;
        if (count <= 1) return;

        for (auto it = start; it != end; ++it) {
            it->siblingedge = count - 1;
            for (auto jt = start; jt != end; ++jt) {
                if (it == jt) continue;
                int d = it->trace->top->Distance(jt->trace->top);
                if (it->distance < 0 || d < it->distance) {
                    it->distance = d;
                }
            }
        }
    }

    TraceDAG::BlockTrace *TraceDAG::SelectBadEdge() {
        std::list<BadEdgeScore> scores;
        for (auto *bt : activetrace_) {
            if (bt->IsTerminal()) continue;
            BadEdgeScore score;
            score.exitproto_id = bt->dest_id;
            score.trace = bt;
            score.terminal = 0;
            scores.push_back(score);
        }

        if (scores.empty()) return nullptr;

        scores.sort();

        auto group_start = scores.begin();
        while (group_start != scores.end()) {
            auto group_end = group_start;
            while (group_end != scores.end() &&
                   group_end->exitproto_id == group_start->exitproto_id) {
                ++group_end;
            }
            ProcessExitConflict(group_start, group_end);
            group_start = group_end;
        }

        BlockTrace *worst = nullptr;
        BadEdgeScore worst_score;
        for (auto &s : scores) {
            if (worst == nullptr || s.CompareFinal(worst_score)) {
                worst = s.trace;
                worst_score = s;
            }
        }

        return worst;
    }

    void TraceDAG::PushBranches(CGraph &g) {
        ClearVisitCount(g);

        for (size_t root_id : rootlist_) {
            if (root_id < g.nodes.size() && !g.Node(root_id).IsCollapsed()) {
                g.Node(root_id).visit_count = 1;
            }
        }

        // Single-loop structure matching Ghidra's pushBranches():
        // one while(activecount>0) with wrap-around, no nested loops.
        // Convergence is detected by missed_count >= activecount_ —
        // when all active traces are stuck, selectBadEdge removes the
        // worst trace (reducing activecount_) and retries.
        //
        // Safety bound: scale by edges (N*E*4) rather than nodes squared
        // to handle switch-heavy graphs where edge count dominates.
        // This should never trigger in correct operation but prevents
        // hangs from bugs in trace logic.
        size_t total_edges = 0;
        for (auto &n : g.nodes) {
            if (!n.IsCollapsed()) {
                total_edges += n.succs.size();
            }
        }
        // Saturate to avoid overflow on huge graphs.
        constexpr size_t kLimit = std::numeric_limits< size_t >::max() / 8;
        size_t n               = g.nodes.size() + 1;
        size_t e               = total_edges + 1;
        const size_t max_outer = (n <= kLimit / e) ? n * e * 4 + 512 : kLimit;
        size_t outer_iter      = 0;
        int missed_count        = 0;
        current_activeiter_    = activetrace_.begin();

        while (activecount_ > 0 && outer_iter < max_outer) {
            ++outer_iter;
            if (current_activeiter_ == activetrace_.end()) {
                if (activetrace_.empty()) {
                    break;
                }
                current_activeiter_ = activetrace_.begin();
            }

            BlockTrace *bt = *current_activeiter_;

            if (missed_count >= activecount_) {
                BlockTrace *bad = SelectBadEdge();
                if (bad == nullptr) {
                    ClearVisitCount(g);
                    return;
                }
                if (bad->bottom_id != CNode::kNone && bad->dest_id != CNode::kNone) {
                    likelygoto_.emplace_back(bad->bottom_id, bad->dest_id);
                }
                RemoveTrace(bad);
                missed_count         = 0;
                current_activeiter_ = activetrace_.begin();
                continue;
            }

            {
                size_t exit_id = CNode::kNone;
                if (CheckRetirement(bt, exit_id)) {
                    current_activeiter_ = RetireBranch(bt->top, exit_id);
                    missed_count         = 0;
                    continue;
                }
            }

            {
                bool was_terminal = bt->IsTerminal();
                if (CheckOpen(g, bt)) {
                    ++current_activeiter_;
                    if (was_terminal) {
                        ++missed_count;
                    } else {
                        missed_count = 0;
                    }
                    continue;
                }
            }

            {
                const auto &dest_node = g.Node(bt->dest_id);
                size_t dag_preds      = 0;
                for (size_t p : dest_node.preds) {
                    if (!g.Node(p).IsCollapsed()) {
                        ++dag_preds;
                    }
                }
                if (dag_preds > 1 && dest_node.visit_count < static_cast< int >(dag_preds))
                {
                    ++current_activeiter_;
                    ++missed_count;
                    continue;
                }
            }

            current_activeiter_ = OpenBranch(g, bt);
            missed_count         = 0;
        }

        if (outer_iter >= max_outer) {
            LOG(WARNING) << "PushBranches: safety limit reached (" << max_outer
                         << " iterations on " << g.nodes.size() << " nodes, " << total_edges
                         << " edges)\n";
        }

        ClearVisitCount(g);
    }

    void TraceDAG::ClearVisitCount(CGraph &g) {
        for (auto &n : g.nodes) {
            n.visit_count = 0;
        }
    }

} // namespace patchestry::ast
