/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <system_error>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/AST/LiftOptions.hpp>
#include <patchestry/AST/PcodeLifter.hpp>
#include <patchestry/AST/TranslationUnit.hpp>
#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Util/Log.hpp>
#include <patchestry/Util/Options.hpp>

namespace {

    const llvm::cl::opt< bool > emit_mlir( // NOLINT(cert-err58-cpp)
        "emit-mlir", llvm::cl::desc("Emit High-level MLIR representation"),
        llvm::cl::init(false)
    );

    const llvm::cl::opt< bool > emit_cir( // NOLINT(cert-err58-cpp)
        "emit-cir", llvm::cl::desc("Emit CIR MLIR representation"), llvm::cl::init(false)
    );

    const llvm::cl::opt< bool > emit_llvm( // NOLINT(cert-err58-cpp)
        "emit-llvm", llvm::cl::desc("Emit LLVM IR Representation"), llvm::cl::init(false)
    );

    const llvm::cl::opt< std::string > input_filename( // NOLINT(cert-err58-cpp)
        "input", llvm::cl::desc("Input JSON file"), llvm::cl::Required
    );

    const llvm::cl::opt< std::string > output_filename( // NOLINT(cert-err58-cpp)
        "output", llvm::cl::desc("Specify output filename"), llvm::cl::value_desc("filename"),
        llvm::cl::init("") // Initialize with empty string
    );

    const llvm::cl::opt< bool > verbose( // NOLINT(cert-err58-cpp)
        "verbose", llvm::cl::desc("Enable debug logs"), llvm::cl::init(false)
    ); // NOLINT(cert-err58-cpp)

    const llvm::cl::opt< bool > print_tu( // NOLINT(cert-err58-cpp)
        "print-tu", llvm::cl::desc("Pretty print translation unit"), llvm::cl::init(false)
    );

    const llvm::cl::opt< bool > emit_flat_baseline( // NOLINT(cert-err58-cpp)
        "emit-flat-baseline",
        llvm::cl::desc(
            "Emit the raw flat CGraph with goto-based control flow "
            "(skips the structuring pass and all post-pass cleanup). "
            "Debug-only — used by /patchir-inspect --debug for parity "
            "diffs against the structured output."
        ),
        llvm::cl::init(false),
        llvm::cl::Hidden
    );

    const llvm::cl::opt< bool > emit_dot_cfg( // NOLINT(cert-err58-cpp)
        "emit-dot-cfg",
        llvm::cl::desc("Dump DOT graphs at phase boundaries (debug)"),
        llvm::cl::init(false)
    );

    const llvm::cl::opt< bool > clang_ast_cleanup( // NOLINT(cert-err58-cpp)
        "clang-ast-cleanup",
        llvm::cl::desc(
            "Run the Clang-AST post-emission cleanup pipeline after "
            "EmitClangAST.  Default on; pass =false to skip it and "
            "emit the raw post-emission AST."),
        llvm::cl::init(true)
    );

    patchestry::Options parseCommandLineOptions(int argc, char **argv) {
        llvm::cl::ParseCommandLineOptions(
            argc, argv, "patche-lifter to represent high pcode into mlir representations\n"
        );

        // --verbose enables DEBUG/INFO; otherwise only WARNING/ERROR/FATAL.
        ::patchestry::logging::MinLogLevel() = verbose.getValue() ? DEBUG : WARNING;

        return {
            .emit_cir                   = emit_cir.getValue(),
            .emit_mlir                  = emit_mlir.getValue(), // It is set to true by default
            .emit_llvm                  = emit_llvm.getValue(),
            .verbose                    = verbose.getValue(),
            .emit_flat_baseline         = emit_flat_baseline.getValue(),
            .clang_ast_cleanup          = clang_ast_cleanup.getValue(),
            .output_file                = output_filename.getValue(),
            .input_file                 = input_filename.getValue(),
            .print_tu                   = print_tu.getValue(),
            .emit_dot_cfg               = emit_dot_cfg.getValue(),
        };
    }

    bool validateBranchindSwitchMetadata(const patchestry::ghidra::Program &program) {
        for (const auto &[func_key, function] : program.serialized_functions) {
            (void)func_key;
            for (const auto &[block_key, block] : function.basic_blocks) {
                (void)block_key;
                for (const auto &operation_key : block.ordered_operations) {
                    auto op_it = block.operations.find(operation_key);
                    if (op_it == block.operations.end()) continue;

                    const auto &op = op_it->second;
                    if (op.mnemonic != patchestry::ghidra::Mnemonic::OP_BRANCHIND) continue;
                    if (op.switch_cases.empty()) continue;

                    size_t valid_targets = 0;
                    for (const auto &sc : op.switch_cases) {
                        if (function.basic_blocks.contains(sc.target_block)) {
                            ++valid_targets;
                        }
                    }

                    const bool has_successor_fallback = !op.successor_blocks.empty();
                    const bool has_valid_fallback_block =
                        op.fallback_block.has_value()
                        && function.basic_blocks.contains(*op.fallback_block);

                    if (valid_targets == 0 && !has_successor_fallback && !has_valid_fallback_block) {
                        LOG(ERROR) << "BRANCHIND switch_cases has no valid target blocks in "
                                   << function.name << " at operation " << op.key
                                   << "; add successor_blocks or at least one valid switch_cases "
                                      "target.\n";
                        return false;
                    }
                }
            }
        }

        return true;
    }

    patchestry::ast::LiftOptions liftOptions(const patchestry::Options &options) {
        return {
            .emit_flat_baseline = options.emit_flat_baseline,
            .clang_ast_cleanup  = options.clang_ast_cleanup,
            .emit_dot_cfg       = options.emit_dot_cfg,
        };
    }

    patchestry::codegen::LoweringOptions loweringOptions(const patchestry::Options &options) {
        return {
            .emit_cir      = options.emit_cir,
            .emit_mlir     = options.emit_mlir,
            .emit_llvm     = options.emit_llvm,
            .output_prefix = options.output_file,
        };
    }

    // Read, parse and deserialize a P-Code JSON file.
    std::optional< patchestry::ghidra::Program > loadProgram(const std::string &path) {
        llvm::ErrorOr< std::unique_ptr< llvm::MemoryBuffer > > file_or_err =
            llvm::MemoryBuffer::getFile(path);
        if (std::error_code error_code = file_or_err.getError()) {
            LOG(ERROR) << "Error reading json file : " << error_code.message() << "\n";
            return std::nullopt;
        }

        auto json = llvm::json::parse(file_or_err.get()->getBuffer());
        if (!json) {
            LOG(ERROR) << "Failed to parse pcode JSON: " << json.takeError();
            return std::nullopt;
        }

        const auto *json_obj = json->getAsObject();
        if (json_obj == nullptr) {
            LOG(ERROR) << "Input JSON is not an object\n";
            return std::nullopt;
        }

        auto program = patchestry::ghidra::JsonParser().deserialize_program(*json_obj);
        if (!program.has_value()) {
            LOG(ERROR) << "Failed to deserialize JSON file '" << path
                       << "' as patchestry program\n";
        }
        return program;
    }

    // `-print-tu`: write the unit as C to `<prefix>.c`, or to stdout when no
    // output prefix was given.  Runs before lowering so the C file is produced
    // even when CIR lowering later fails.
    bool printTranslationUnit(
        patchestry::ast::TranslationUnit &unit, const std::string &output_file
    ) {
        auto &ctx = unit.context();
        if (!output_file.empty()) {
            std::error_code ec;
            llvm::raw_fd_ostream out(output_file + ".c", ec, llvm::sys::fs::OF_Text);
            if (ec) {
                LOG(ERROR) << "Failed to write C output: " << ec.message() << "\n";
                return false;
            }
            ctx.getTranslationUnitDecl()->print(out, ctx.getPrintingPolicy(), 0);
            out.close();
            if (out.has_error()) {
                LOG(ERROR) << "Failed to write C output: " << out.error().message() << "\n";
                out.clear_error();
                return false;
            }
        } else {
            ctx.getTranslationUnitDecl()->print(
                llvm::outs(), ctx.getPrintingPolicy(), /*Indentation=*/0
            );
        }
        return true;
    }

} // namespace

int main(int argc, char **argv) {
    auto options = parseCommandLineOptions(argc, argv);

    auto program = loadProgram(options.input_file);
    if (!program.has_value()) { return EXIT_FAILURE; }

    if (!validateBranchindSwitchMetadata(*program)) { return EXIT_FAILURE; }

    // AST source: lift the P-Code model into a Clang translation unit.
    auto unit = patchestry::ast::LiftProgram(*program, liftOptions(options));
    if (!unit) { return EXIT_FAILURE; }

    if (options.print_tu && !printTranslationUnit(*unit, options.output_file)) {
        return EXIT_FAILURE;
    }

    if (unit->has_errors()) {
        LOG(ERROR) << "Skipping code generation due to prior diagnostics errors.\n";
        return EXIT_FAILURE;
    }

    // Lowering: the same call serves any AST source.
    patchestry::codegen::CodeGenerator codegen(unit->context(), unit->codegen_options());
    auto module = codegen.lower_ast_to_mlir();
    if (!module.has_value()) {
        LOG(ERROR) << "Failed to emit mlir module\n";
        return EXIT_FAILURE;
    }
    return codegen.emit_outputs(*module, loweringOptions(options)) ? EXIT_SUCCESS
                                                                   : EXIT_FAILURE;
}
