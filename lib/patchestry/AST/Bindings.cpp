/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/Bindings.hpp>

#include <algorithm>
#include <tuple>
#include <utility>

#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/JSON.h>

namespace patchestry::ast {

    void WriteBindings(
        llvm::raw_ostream &out, std::vector< Binding > bindings,
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

        llvm::json::Object root{
            { "architecture", architecture },
            { "language", language },
            { "format", format },
            { "bindings", std::move(entries) },
        };
        out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
    }

} // namespace patchestry::ast
