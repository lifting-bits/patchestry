/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <memory>

#include <clang/Basic/TargetInfo.h>
#include <clang/Parse/ParseAST.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Verifier.h>

#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Frontend/ClangFrontend.hpp>
#include <patchestry/Ghidra/Target.hpp>

using namespace patchestry;

int main(int argc, char **argv) {
    if (argc != 2) { return 1; }
    bool passed = true;
    auto check  = [&](bool ok, const char *message) {
        if (!ok) {
            llvm::errs() << message << "\n";
            passed = false;
        }
    };

    using ghidra::VariantPolicy;
    check(
        ghidra::targetTriple("ARM:LE:32:Cortex", VariantPolicy::Ignore)
            == "arm-unknown-linux-gnueabihf",
        "Lifter must ignore ARM variant"
    );
    check(
        ghidra::targetTriple("ARM:LE:32:Cortex", VariantPolicy::Preserve)
            == "arm-unknown-linux-gnueabihf",
        "Preserve existing Cortex triple spelling"
    );
    check(
        ghidra::targetTriple("AARCH64:LE:64:AppleSilicon", VariantPolicy::Ignore)
            == "aarch64-unknown-linux",
        "Lifter must ignore AArch64 variant"
    );
    check(
        ghidra::targetTriple("AARCH64:LE:64:AppleSilicon", VariantPolicy::Preserve)
            == "arm64e-unknown-linux",
        "Patch compiler must preserve AArch64 variant"
    );
    check(
        ghidra::targetTriple("x86:LE:32:default", VariantPolicy::Ignore, "ARM")
            == "arm-unknown-linux-gnueabihf",
        "Program architecture must override language"
    );
    check(
        ghidra::targetTriple("MIPS:BE:64:default", VariantPolicy::Ignore)
            == "mips64-unknown-linux",
        "Preserve big-endian MIPS mapping"
    );
    check(
        ghidra::targetTriple("bad", VariantPolicy::Preserve).empty(),
        "Reject malformed language ids"
    );
    check(
        ghidra::targetTriple("ARM:LE:bad:Cortex", VariantPolicy::Ignore).empty(),
        "Reject invalid bit sizes"
    );

    for (auto policy :
         { frontend::CompilationPolicy::LiftedCode, frontend::CompilationPolicy::PatchCode })
    {
        frontend::FrontendConfig config("x86_64-unknown-linux", policy);
        auto synthetic =
            frontend::createSyntheticCompilerInstance(config, [](clang::CompilerInstance &) {
                return std::make_unique< clang::ASTConsumer >();
            });
        auto parsed = frontend::createCompilerInstance(argv[1], config);
        if (!synthetic || !parsed) { return 1; }
        const bool lifted = policy == frontend::CompilationPolicy::LiftedCode;
        for (auto *ci : { synthetic.get(), parsed.get() }) {
            check(
                ci->getCodeGenOpts().StrictReturn == !lifted,
                "StrictReturn must depend on policy, not AST source"
            );
            check(!ci->getCodeGenOpts().StrictEnums, "StrictEnums must remain disabled");
            check(ci->getCodeGenOpts().OptimizationLevel == 0, "Preserve optimization level");
            check(ci->getLangOpts().GNUMode == !lifted, "GNU mode must depend on policy");
            check(
                &ci->getTarget().getTargetOpts() == &ci->getTargetOpts(),
                "Target options must be invocation-owned"
            );
        }
        clang::ParseAST(parsed->getSema());
        check(
            !parsed->getDiagnostics().hasErrorOccurred(), "Parse the test C translation unit"
        );
        codegen::CodeGenerator generator(parsed->getASTContext(), parsed->getCodeGenOpts());
        auto module = generator.lower_ast_to_mlir();
        if (!module) { return 1; }
        // Duplicate a symbol to exercise the public API's verification failure.
        auto &ops = module->getBody()->getOperations();
        if (ops.empty()) { return 1; }
        ops.push_back(ops.front().clone());
        check(!generator.emit_outputs(*module, {}), "Reject an invalid module before output");
        ops.back().erase();

        // Well-formed but unsupported IR must fail conversion/translation without
        // invoking the fatal-error wrapper used by direct CIR lowering.
        module->getContext()->allowUnregisteredDialects();
        mlir::OperationState unsupported(module->getLoc(), "test.unsupported");
        ops.push_back(mlir::Operation::create(unsupported));
        check(mlir::succeeded(mlir::verify(*module)), "Unsupported op must be valid input IR");
        check(
            !generator.emit_outputs(*module, { .emit_llvm = true }),
            "Return conversion/translation failure to the caller"
        );
    }
    return passed ? 0 : 1;
}
