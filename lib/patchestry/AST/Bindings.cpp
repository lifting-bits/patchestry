/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/Bindings.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>

#include <algorithm>
#include <tuple>
#include <utility>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/JSON.h>

namespace patchestry::ast {

    std::vector< UnresolvedCall > CollectUnresolvedCalls(const ghidra::Program &program) {
        std::vector< UnresolvedCall > calls;
        for (const auto &[function_key, function] : program.serialized_functions) {
            for (const auto &[block_key, block] : function.basic_blocks) {
                for (const auto &[op_key, op] : block.operations) {
                    UnresolvedCall call;
                    if (op.mnemonic == ghidra::Mnemonic::OP_CALLIND) {
                        call.kind = "indirect";
                        if (op.target && op.target->global) {
                            const auto &globals = program.serialized_globals;
                            auto global         = globals.find(*op.target->global);
                            call.through        = global != globals.end()
                                ? global->second.name
                                : *op.target->global;
                        }
                    } else if (op.mnemonic == ghidra::Mnemonic::OP_TAIL_CALL
                               && (!op.target || !op.target->function))
                    {
                        call.kind = "tail-call";
                        if (op.target && op.target->address) {
                            call.target_address = *op.target->address;
                        }
                    } else {
                        continue;
                    }
                    call.function = function.name;
                    call.key      = op_key;

                    // `<space>:<hex offset>[:...]` -> (space, offset).
                    llvm::StringRef rest(op_key);
                    auto [space, tail] = rest.split(':');
                    uint64_t offset    = 0;
                    if (!tail.empty() && !tail.split(':').first.getAsInteger(16, offset)) {
                        call.address_space = space.str();
                        call.address       = offset;
                    }
                    calls.push_back(std::move(call));
                }
            }
        }
        std::sort(calls.begin(), calls.end(), [](const auto &lhs, const auto &rhs) {
            return std::tie(lhs.function, lhs.key) < std::tie(rhs.function, rhs.key);
        });
        return calls;
    }

    void WriteBindings(
        llvm::raw_ostream &out, std::vector< Binding > bindings,
        const std::vector< UnresolvedCall > &unresolved_calls,
        const std::string &architecture, const std::string &language,
        const std::string &format
    ) {
        // Objects before functions, each by address, then by name.  Entries
        // without an address sort last within their kind.
        std::sort(bindings.begin(), bindings.end(), [](const Binding &lhs, const Binding &rhs) {
            auto rank = [](const Binding &binding) {
                return std::make_tuple(
                    binding.kind != "object", !binding.address.has_value(),
                    binding.address_space, binding.address.value_or(0), binding.name
                );
            };
            return rank(lhs) < rank(rhs);
        });

        llvm::json::Array entries;
        for (const auto &binding : bindings) {
            llvm::json::Object entry{
                { "name", binding.name },
                { "symbol", binding.symbol },
                { "kind", binding.kind },
                { "origin", binding.origin },
                { "size", static_cast< int64_t >(binding.size) },
            };
            if (!binding.key.empty()) {
                entry["key"] = binding.key;
            }
            if (binding.address) {
                entry["address_space"] = binding.address_space;
                // A string: JSON numbers do not hold 64-bit addresses exactly.
                entry["address"] = "0x" + llvm::utohexstr(*binding.address, /*LowerCase=*/true);
            }
            if (binding.kind == "object") {
                entry["lifted_size"] = static_cast< int64_t >(binding.lifted_size);
            } else {
                entry["defined"] = binding.defined;
            }
            entries.push_back(std::move(entry));
        }

        llvm::json::Array calls;
        for (const auto &call : unresolved_calls) {
            llvm::json::Object entry{
                { "function", call.function },
                { "key", call.key },
                { "kind", call.kind },
                { "target", "unresolved" },
            };
            if (call.address) {
                entry["address_space"] = call.address_space;
                entry["address"] = "0x" + llvm::utohexstr(*call.address, /*LowerCase=*/true);
            }
            if (!call.through.empty()) {
                entry["through"] = call.through;
            }
            if (!call.target_address.empty()) {
                entry["target_address"] = call.target_address;
            }
            calls.push_back(std::move(entry));
        }

        llvm::json::Object root{
            { "architecture", architecture },
            { "language", language },
            { "format", format },
            { "bindings", std::move(entries) },
            { "unresolved_calls", std::move(calls) },
        };
        out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
    }

} // namespace patchestry::ast
