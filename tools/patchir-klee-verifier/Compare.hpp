/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/JSON.h>

namespace patchestry::klee_verifier {

    // A callee swap the patch declares equivalent: a call of `patched` in the
    // patched version stands for a call of `original`, and the two agree on
    // their first `compare_args` arguments.
    struct Substitution
    {
        std::string original;
        std::string patched;
        unsigned compare_args = 0;
    };

    struct CompareOptions
    {
        // Functions whose call means the caller rejected its input.
        std::vector< std::string > reject_handlers;
        // Return values of the target that mean it rejected its input.
        std::vector< int64_t > reject_returns;
        // Argument index -> size of the buffer the argument points at.
        std::map< unsigned, uint64_t > arg_buffers;
        // Argument index -> inclusive range of an integer argument.
        std::map< unsigned, std::pair< int64_t, int64_t > > arg_ranges;
        // Buffer size for a pointer argument with no declared size.
        uint64_t pointer_size = 256;
        // Callees whose calls are left out of the call trace.
        std::vector< std::string > pure_callees;
        std::vector< Substitution > substitutions;
        // External functions the runner implements (`__patchestry_model_<name>`):
        // their calls are traced and then executed.
        std::set< std::string > models;
        // How many distinct values an external function returns before it
        // repeats the last one.
        unsigned stub_values = 8;
    };

    // Link `patched` into `original` and add a main() that runs both versions
    // of `target` on one input.  Returns the product module, or null after
    // logging why the pair cannot be modelled.  `info` receives what the
    // harness assumes and bounds, for the evidence.
    std::unique_ptr< llvm::Module > buildProductHarness(
        std::unique_ptr< llvm::Module > original, std::unique_ptr< llvm::Module > patched,
        const std::string &target, const CompareOptions &options, llvm::json::Object &info
    );

    bool writeHarnessInfo(const llvm::json::Object &info, llvm::StringRef path);

} // namespace patchestry::klee_verifier
