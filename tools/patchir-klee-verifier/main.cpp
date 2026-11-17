/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>

#include <patchestry/Util/Log.hpp>

#include "Compare.hpp"
#include "HarnessGen.hpp"
#include "Options.hpp"

namespace patchestry::klee_verifier {

    llvm::cl::opt< std::string > input_filename(
        llvm::cl::Positional, llvm::cl::desc("<input LLVM IR file>"), llvm::cl::init("-")
    );

    llvm::cl::opt< std::string > output_filename(
        "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
        llvm::cl::init("-")
    );

    llvm::cl::opt< std::string > target_function(
        "target-function", llvm::cl::desc("Function to verify (required)"),
        llvm::cl::value_desc("name"), llvm::cl::Required
    );

    llvm::cl::opt< bool >
        emit_ll("S", llvm::cl::desc("Emit LLVM IR instead of bitcode"), llvm::cl::init(false));

    llvm::cl::opt< std::string > model_library(
        "model-library",
        llvm::cl::desc("LLVM IR/bitcode file with external function models (linked before stubbing)"),
        llvm::cl::value_desc("file"), llvm::cl::init("")
    );

    llvm::cl::opt< bool >
        verbose("v", llvm::cl::desc("Enable verbose output"), llvm::cl::init(false));

    llvm::cl::opt< unsigned > symbolic_ptr_size(
        "symbolic-ptr-size",
        llvm::cl::desc("Default symbolic buffer size (bytes) for pointer arguments"),
        llvm::cl::value_desc("bytes"), llvm::cl::init(256)
    );

    // Runtime depth cap for the recursive globals initializer. Self-
    // referential and mutually-recursive types emit a call to the
    // already-being-built per-type init function at codegen time (the
    // TypeInitCache is the cycle breaker), so a runtime guard is needed
    // to stop KLEE from evaluating malloc -> init -> malloc -> init …
    // forever. At depth >= max the pointer field is set to null and
    // the recursive call is skipped. 2 is enough for list.next.next;
    // raise for deeper structures (trees, chains) at the cost of larger
    // symbolic states.
    llvm::cl::opt< unsigned > klee_init_max_depth(
        "klee-init-max-depth",
        llvm::cl::desc("Runtime depth cap for recursive pointee initialization"),
        llvm::cl::value_desc("depth"), llvm::cl::init(2)
    );

    // Arrays longer than this collapse to a single flat klee_make_symbolic
    // call over the whole storage (matching the pre-change behavior for
    // the whole module). Keeps generated IR bounded on modules with huge
    // static tables. Pointer fields inside the fall-back array are then
    // left as symbolic bytes — a known limitation documented in the plan.
    llvm::cl::opt< unsigned > klee_init_array_expand_limit(
        "klee-init-array-expand-limit",
        llvm::cl::desc("Above this length, arrays of aggregates fall back to flat byte symbolization"),
        llvm::cl::value_desc("elements"), llvm::cl::init(64)
    );

    // When any contract predicate fails to parse, a silently-passing
    // verifier run is worse than one that refuses to run: the resulting
    // harness is under-constrained and can "verify" against a weaker
    // condition than the author intended. Default-on so typos in contract
    // YAML always surface; set to false to fall back to the old
    // warn-and-continue behavior (e.g. for CI jobs that parse partial
    // contracts intentionally).
    llvm::cl::opt< bool > strict_contracts(
        "strict-contracts",
        llvm::cl::desc("Exit non-zero when any contract predicate fails to parse"),
        llvm::cl::init(true)
    );


    // Compare mode: the positional input is the patched module, this is the
    // lifted original.  See Compare.cpp.
    llvm::cl::opt< std::string > compare_original(
        "compare-original",
        llvm::cl::desc("Lifted original module; build a product harness of it and the input"),
        llvm::cl::value_desc("file"), llvm::cl::init("")
    );

    llvm::cl::list< std::string > reject_handlers(
        "reject-handler",
        llvm::cl::desc("Function whose call means the input was rejected (compare mode)"),
        llvm::cl::value_desc("name")
    );

    llvm::cl::list< long long > reject_returns(
        "reject-return",
        llvm::cl::desc("Return value that means the input was rejected (compare mode)"),
        llvm::cl::value_desc("value")
    );

    llvm::cl::list< std::string > arg_buffers(
        "arg-buffer",
        llvm::cl::desc("Argument INDEX points at a buffer of SIZE bytes (compare mode)"),
        llvm::cl::value_desc("index:size")
    );

    llvm::cl::list< std::string > arg_ranges(
        "arg-range",
        llvm::cl::desc("Integer argument INDEX takes values LOW to HIGH inclusive (compare mode)"),
        llvm::cl::value_desc("index:low:high")
    );

    llvm::cl::list< std::string > noreturn_handlers(
        "noreturn-handler",
        llvm::cl::desc("Reject handler that does not return (compare mode)"),
        llvm::cl::value_desc("name")
    );

    llvm::cl::list< std::string > reject_skips(
        "reject-skip",
        llvm::cl::desc("Function the patched version rejects an input by not calling "
                       "(compare mode)"),
        llvm::cl::value_desc("name")
    );

    llvm::cl::list< std::string > reject_effects(
        "reject-effect",
        llvm::cl::desc("Function a version may call before it rejects (compare mode)"),
        llvm::cl::value_desc("name")
    );

    llvm::cl::list< std::string > mmio_regions(
        "mmio-region",
        llvm::cl::desc("Device registers LOW to HIGH inclusive; writes are compared in order "
                       "(compare mode)"),
        llvm::cl::value_desc("low:high")
    );

    llvm::cl::list< std::string > balance(
        "balance",
        llvm::cl::desc("Calls of ACQUIRE and RELEASE must balance on every exit (compare mode)"),
        llvm::cl::value_desc("acquire:release")
    );

    llvm::cl::list< std::string > pure_callees(
        "pure-callee",
        llvm::cl::desc("External function whose calls are not compared (compare mode)"),
        llvm::cl::value_desc("name")
    );

    llvm::cl::list< std::string > substitutions(
        "substitution",
        llvm::cl::desc(
            "PATCHED called by the patched version stands for ORIGINAL; the first COUNT "
            "arguments are compared (compare mode)"
        ),
        llvm::cl::value_desc("original:patched:count")
    );

    llvm::cl::list< std::string > models(
        "model",
        llvm::cl::desc("External function the runner implements as __patchestry_model_NAME "
                       "(compare mode)"),
        llvm::cl::value_desc("name")
    );

    llvm::cl::opt< unsigned > stub_values(
        "stub-values",
        llvm::cl::desc("Distinct values an external function returns (compare mode)"),
        llvm::cl::value_desc("count"), llvm::cl::init(8)
    );

    llvm::cl::opt< bool > keep_target(
        "keep-target",
        llvm::cl::desc("Do not retarget to x86_64; keep the modules' triple and layout"),
        llvm::cl::init(false)
    );

    llvm::cl::opt< std::string > harness_info(
        "harness-info",
        llvm::cl::desc("Write what the product harness assumes and bounds, as JSON"),
        llvm::cl::value_desc("file"), llvm::cl::init("")
    );

    static int compareMain(
        const char *program, std::unique_ptr< llvm::Module > patched, llvm::LLVMContext &context
    ) {
        llvm::SMDiagnostic err;
        auto original = llvm::parseIRFile(compare_original, err, context);
        if (!original) {
            err.print(program, llvm::errs());
            return EXIT_FAILURE;
        }
        CompareOptions options;
        options.reject_handlers.assign(reject_handlers.begin(), reject_handlers.end());
        options.reject_returns.assign(reject_returns.begin(), reject_returns.end());
        options.pointer_size = symbolic_ptr_size;
        options.stub_values  = stub_values == 0 ? 1 : stub_values;
        for (const auto &text : arg_buffers) {
            unsigned index = 0;
            unsigned long long size = 0;
            auto [left, right] = llvm::StringRef(text).split(':');
            if (left.getAsInteger(10, index) || right.getAsInteger(0, size)) {
                LOG(ERROR) << "--arg-buffer expects INDEX:SIZE, got '" << text << "'\n";
                return EXIT_FAILURE;
            }
            options.arg_buffers[index] = size;
        }
        for (const auto &text : arg_ranges) {
            unsigned index = 0;
            long long low = 0;
            long long high = 0;
            auto [left, rest]  = llvm::StringRef(text).split(':');
            auto [middle, end] = rest.split(':');
            if (left.getAsInteger(10, index) || middle.getAsInteger(0, low)
                || end.getAsInteger(0, high))
            {
                LOG(ERROR) << "--arg-range expects INDEX:LOW:HIGH, got '" << text << "'\n";
                return EXIT_FAILURE;
            }
            options.arg_ranges[index] = { low, high };
        }
        options.noreturn_handlers.assign(noreturn_handlers.begin(), noreturn_handlers.end());
        options.reject_effects.assign(reject_effects.begin(), reject_effects.end());
        options.reject_skips.assign(reject_skips.begin(), reject_skips.end());
        for (const auto &text : mmio_regions) {
            unsigned long long low  = 0;
            unsigned long long high = 0;
            auto [left, right]      = llvm::StringRef(text).split(':');
            if (left.getAsInteger(0, low) || right.getAsInteger(0, high) || low > high) {
                LOG(ERROR) << "--mmio-region expects LOW:HIGH, got '" << text << "'\n";
                return EXIT_FAILURE;
            }
            options.mmio_regions.emplace_back(low, high);
        }
        for (const auto &text : balance) {
            auto [left, right] = llvm::StringRef(text).split(':');
            if (left.empty() || right.empty() || left == right) {
                LOG(ERROR) << "--balance expects ACQUIRE:RELEASE, got '" << text << "'\n";
                return EXIT_FAILURE;
            }
            options.balance.emplace_back(left.str(), right.str());
        }
        options.pure_callees.assign(pure_callees.begin(), pure_callees.end());
        options.models.insert(models.begin(), models.end());
        for (const auto &text : substitutions) {
            Substitution swap;
            auto [left, rest]  = llvm::StringRef(text).split(':');
            auto [middle, end] = rest.split(':');
            if (left.empty() || middle.empty() || end.getAsInteger(10, swap.compare_args)) {
                LOG(ERROR) << "--substitution expects ORIGINAL:PATCHED:COUNT, got '" << text
                           << "'\n";
                return EXIT_FAILURE;
            }
            swap.original = left.str();
            swap.patched  = middle.str();
            options.substitutions.push_back(std::move(swap));
        }
        if (!keep_target) {
            retargetModuleToX86_64(*original, verbose);
            retargetModuleToX86_64(*patched, verbose);
        }
        llvm::json::Object info;
        auto product = buildProductHarness(
            std::move(original), std::move(patched), target_function, options, info
        );
        if (!product) {
            return EXIT_FAILURE;
        }
        info["retargeted"] = !keep_target;
        if (!harness_info.empty() && !writeHarnessInfo(info, harness_info)) {
            return EXIT_FAILURE;
        }
        return writeModuleToFile(*product, output_filename) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

} // namespace patchestry::klee_verifier

int main(int argc, char **argv) {
    using namespace patchestry::klee_verifier;

    llvm::InitLLVM y(argc, argv);
    llvm::cl::ParseCommandLineOptions(argc, argv, "patchir-klee-verifier\n");

    llvm::LLVMContext context;
    llvm::SMDiagnostic err;

    std::unique_ptr< llvm::Module > module = llvm::parseIRFile(input_filename, err, context);
    if (!module) {
        err.print(argv[0], llvm::errs());
        return EXIT_FAILURE;
    }

    if (!compare_original.empty()) {
        return compareMain(argv[0], std::move(module), context);
    }

    // Retarget module to x86_64 for KLEE compatibility. See the comment on
    // retargetModuleToX86_64 for the layout-reshaping caveat — this call
    // will emit a prominent warning if the original pointer width differed
    // from 64 and any struct types had to be silently reshaped.
    retargetModuleToX86_64(*module, verbose);

    if (verbose) {
        llvm::outs() << "Loaded module: " << module->getName() << "\n";
        llvm::outs() << "Target function: " << target_function << "\n";
    }

    // Find the target function
    llvm::Function *target_fn = module->getFunction(target_function);
    if (!target_fn) {
        LOG(ERROR) << "target function '" << target_function << "' not found in module\n";
        return EXIT_FAILURE;
    }

    if (target_fn->isDeclaration()) {
        LOG(ERROR) << "target function '" << target_function
                   << "' is a declaration (no body)\n";
        return EXIT_FAILURE;
    }

    // 1. Link model library if provided (replaces declarations with definitions)
    if (!model_library.empty()) {
        auto model_mod = llvm::parseIRFile(model_library, err, context);
        if (!model_mod) {
            LOG(ERROR) << "failed to load model library '" << model_library << "'\n";
            err.print(argv[0], llvm::errs());
            return EXIT_FAILURE;
        }

        // The model must be compatible with the retargeted datalayout —
        // mismatched pointer widths or struct alignments would produce
        // miscompiled code after linking. For the triple, we only require
        // matching arch+vendor+OS; the environment (libc) field differs
        // between musl/gnu but does not affect IR-level layout, and
        // rejecting on it would force model authors to rebuild for every
        // libc variant. Empty triple/DL (older bitcode) is stamped with
        // the host values.
        const llvm::Triple &model_triple = model_mod->getTargetTriple();
        const std::string &model_dl      = model_mod->getDataLayoutStr();
        const llvm::Triple &host_triple  = module->getTargetTriple();
        const std::string &host_dl       = module->getDataLayoutStr();
        const std::string model_triple_str = model_triple.str();
        const std::string host_triple_str  = host_triple.str();

        if (model_triple_str.empty() && model_dl.empty()) {
            model_mod->setTargetTriple(host_triple);
            model_mod->setDataLayout(module->getDataLayout());
        } else {
            bool triple_compatible =
                model_triple.getArch()   == host_triple.getArch()   &&
                model_triple.getVendor() == host_triple.getVendor() &&
                model_triple.getOS()     == host_triple.getOS();
            if (!triple_compatible || model_dl != host_dl) {
                LOG(ERROR) << "model library '" << model_library
                           << "' targets '" << model_triple_str << "' / '" << model_dl
                           << "', expected arch+OS of '" << host_triple_str << "' / DL '"
                           << host_dl << "'; rebuild the model for the host layout\n";
                return EXIT_FAILURE;
            }
        }

        if (llvm::Linker::linkModules(*module, std::move(model_mod),
                                      llvm::Linker::Flags::LinkOnlyNeeded)) {
            LOG(ERROR) << "failed to link model library '" << model_library << "'\n";
            return EXIT_FAILURE;
        }

        // Re-fetch: linkModules may have replaced the original Function*.
        target_fn = module->getFunction(target_function);
        if (!target_fn || target_fn->isDeclaration()) {
            LOG(ERROR) << "target function '" << target_function
                       << "' lost or became a declaration after linking model library\n";
            return EXIT_FAILURE;
        }

        if (verbose) {
            llvm::outs() << "Linked model library: " << model_library << "\n";
        }
    }

    // 2. Rewrite abort-like declarations to call klee_abort. Must run
    //    before stubExternalFunctions — otherwise the stubber gives
    //    abort-like decls a symbolic body instead of the klee_abort
    //    redirect.
    unsigned abort_count = rewriteAbortCalls(*module);
    if (verbose) {
        llvm::outs() << "Rewrote " << abort_count << " abort-like call(s) to klee_abort\n";
    }

    // 3. Stub external functions (must do before harness so stubs exist)
    unsigned stub_count = stubExternalFunctions(*module, target_fn);
    if (verbose) {
        llvm::outs() << "Stubbed " << stub_count << " external function(s)\n";
    }

    // 4. Instrument static contracts at each call site that carries
    //    `!static_contract` metadata. Runs before harness generation so the
    //    harness's call to target_fn flows through the now-instrumented
    //    bodies of target_fn and its callees.
    if (!instrumentStaticContracts(*module)) {
        LOG(ERROR) << "failed to instrument static contracts\n";
        return EXIT_FAILURE;
    }

    // 5. Generate main() harness
    if (!generateHarness(*module, target_fn)) {
        LOG(ERROR) << "failed to generate harness\n";
        return EXIT_FAILURE;
    }

    // 6. Write output
    if (!writeModuleToFile(*module, output_filename)) {
        LOG(ERROR) << "failed to write output\n";
        return EXIT_FAILURE;
    }

    if (verbose) {
        llvm::outs() << "Successfully wrote output to: " << output_filename << "\n";
    }

    return EXIT_SUCCESS;
}
