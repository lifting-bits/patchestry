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
#include <vector>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

#include <patchestry/AST/CSourceUnit.hpp>
#include <patchestry/AST/LiftOptions.hpp>
#include <patchestry/AST/PcodeLifter.hpp>
#include <patchestry/AST/PcodeValidator.hpp>
#include <patchestry/AST/TUPrinter.hpp>
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
        "input", llvm::cl::desc("Input P-Code JSON file (required unless -from-c is given)"),
        llvm::cl::init("")
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
            "Lean lift: emit the flat CGraph with goto-based control flow "
            "(mechanical opcode lifting plus one label/goto per CFG edge; "
            "skips the structuring pass and all post-pass cleanup).  Input "
            "for the out-of-process LLM structuring stage and the parity "
            "baseline for /patchir-inspect --debug."
        ),
        llvm::cl::init(false)
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

    const llvm::cl::opt< std::string > from_c_filename( // NOLINT(cert-err58-cpp)
        "from-c",
        llvm::cl::desc(
            "Compile a C translation unit written by -print-tu (or refined from it) "
            "instead of lifting JSON.  The target comes from -target-lang, else the "
            "`patchestry:tu` header, else -input."
        ),
        llvm::cl::value_desc("file.c"), llvm::cl::init("")
    );

    const llvm::cl::opt< std::string > target_lang( // NOLINT(cert-err58-cpp)
        "target-lang",
        llvm::cl::desc("Ghidra language id for -from-c, e.g. ARM:LE:32:Cortex"),
        llvm::cl::init("")
    );

    const llvm::cl::opt< bool > strict_symbols( // NOLINT(cert-err58-cpp)
        "strict-symbols",
        llvm::cl::desc(
            "-from-c: fail when a function named by a patchestry:function-begin marker "
            "is missing or has no body (default true)"
        ),
        llvm::cl::init(true)
    );

    const llvm::cl::opt< std::string > validate_pcode( // NOLINT(cert-err58-cpp)
        "validate-pcode",
        llvm::cl::desc(
            "-from-c: validate the parsed C against the -input P-Code model and write "
            "<output>.validation.json.  Bare flag fails on any critical finding; "
            "=report writes the report and exits 0."
        ),
        llvm::cl::ValueOptional, llvm::cl::init("")
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
            .from_c_file                = from_c_filename.getValue(),
            .target_lang                = target_lang.getValue(),
            .strict_symbols             = strict_symbols.getValue(),
            .validate_pcode             = validate_pcode.getNumOccurrences() > 0,
            .validate_report_only       = validate_pcode.getValue() == "report",
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

    // `-print-tu`: write the unit as re-parseable C to `<prefix>.c`, or to
    // stdout when no output prefix was given.  Runs before lowering so the C
    // file is produced even when CIR lowering later fails.  ParenExpr is
    // transparent to CIRGen, so reparenthesizing here leaves the lowered IR
    // unchanged.
    bool printTranslationUnit(
        patchestry::ast::TranslationUnit &unit, const std::string &output_file
    ) {
        auto &ctx = unit.context();
        for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
            if (auto *fn = llvm::dyn_cast< clang::FunctionDecl >(decl);
                fn != nullptr && fn->doesThisDeclarationHaveABody())
            {
                patchestry::ast::ReparenthesizeForPrint(ctx, fn->getBody());
            }
        }
        patchestry::ast::TUPrintOptions print_opts;
        print_opts.lang_id = unit.lang_id;
        print_opts.arch    = unit.arch;
        if (!output_file.empty()) {
            std::error_code ec;
            llvm::raw_fd_ostream out(output_file + ".c", ec, llvm::sys::fs::OF_Text);
            if (ec) {
                LOG(ERROR) << "Failed to write C output: " << ec.message() << "\n";
                return false;
            }
            patchestry::ast::PrintTranslationUnit(out, ctx, unit.definitions, print_opts);
            out.close();
            if (out.has_error()) {
                LOG(ERROR) << "Failed to write C output: " << out.error().message() << "\n";
                out.clear_error();
                return false;
            }
        } else {
            patchestry::ast::PrintTranslationUnit(
                llvm::outs(), ctx, unit.definitions, print_opts
            );
        }
        return true;
    }

    // `-from-c`: every `patchestry:function-begin` marker must name a function
    // with a body in the parsed C.  `-strict-symbols=false` only logs a miss.
    bool checkMarkerDefinitions(patchestry::ast::TranslationUnit &unit, bool strict) {
        bool missing = false;
        for (const auto &marker : unit.markers) {
            if (patchestry::ast::FindFunctionDefinition(unit.context(), marker.name) == nullptr)
            {
                LOG(ERROR) << "from-c: no definition for marker function '" << marker.name
                           << "' (" << marker.key << ")\n";
                missing = true;
            }
        }
        return !(missing && strict);
    }

    // `-validate-pcode`: check the unit against the P-Code model and write
    // `<prefix>.validation.json`, or the report to stdout without a prefix.
    // Returns false when the report cannot be written; `critical` reports
    // whether any finding is critical.
    bool validateAgainstPcode(
        patchestry::ast::TranslationUnit &unit, const patchestry::ghidra::Program &program,
        const std::string &output_file, bool &critical
    ) {
        auto report =
            patchestry::ast::ValidateAgainstPcode(unit.context(), program, unit.markers, {});
        if (!output_file.empty()) {
            std::error_code ec;
            llvm::raw_fd_ostream out(
                output_file + ".validation.json", ec, llvm::sys::fs::OF_Text
            );
            if (ec) {
                LOG(ERROR) << "from-c: cannot write validation report: " << ec.message()
                           << "\n";
                return false;
            }
            patchestry::ast::WriteValidationReport(out, report);
        } else {
            patchestry::ast::WriteValidationReport(llvm::outs(), report);
        }
        LOG(WARNING) << "validate: " << report.functions.size() << " function(s), "
                     << report.failed << " failed, " << report.warned << " with warnings\n";
        critical = report.HasCritical();
        return true;
    }

    // `-from-c`: every marker symbol must have a body in the lowered module.
    bool checkMarkerSymbols(
        mlir::ModuleOp module, const std::vector< patchestry::ast::MarkerEntry > &markers,
        bool strict
    ) {
        bool missing = false;
        for (const auto &marker : markers) {
            auto *op = mlir::SymbolTable::lookupSymbolIn(module, marker.symbol);
            const bool defined =
                op != nullptr && op->getNumRegions() > 0 && !op->getRegion(0).empty();
            if (!defined) {
                LOG(ERROR) << "from-c: CIR symbol '" << marker.symbol
                           << "' is missing or has no body\n";
                missing = true;
            }
        }
        return !(missing && strict);
    }

} // namespace

int main(int argc, char **argv) {
    auto options      = parseCommandLineOptions(argc, argv);
    const bool from_c = !options.from_c_file.empty();
    if (!from_c && options.input_file.empty()) {
        LOG(ERROR) << "-input <json> is required unless -from-c is given\n";
        return EXIT_FAILURE;
    }

    // `-input` is the program to lift, or, with `-from-c`, the target
    // fallback and the model for `-validate-pcode`.
    std::optional< patchestry::ghidra::Program > program;
    if (!options.input_file.empty()) {
        program = loadProgram(options.input_file);
        if (!program.has_value()) { return EXIT_FAILURE; }
    }

    // AST source: re-enter printed or refined C, or lift the P-Code model.
    std::unique_ptr< patchestry::ast::TranslationUnit > unit;
    if (from_c) {
        unit = patchestry::ast::ParseCTranslationUnit(
            { .path = options.from_c_file, .target_lang = options.target_lang },
            program.has_value() ? &*program : nullptr
        );
    } else {
        if (!validateBranchindSwitchMetadata(*program)) { return EXIT_FAILURE; }
        unit = patchestry::ast::LiftProgram(*program, liftOptions(options));
    }
    if (!unit) { return EXIT_FAILURE; }

    if (!checkMarkerDefinitions(*unit, options.strict_symbols)) { return EXIT_FAILURE; }

    int status = EXIT_SUCCESS;
    if (options.validate_pcode) {
        if (!program.has_value()) {
            LOG(ERROR) << "-validate-pcode requires -input <json>\n";
            return EXIT_FAILURE;
        }
        bool critical = false;
        if (!validateAgainstPcode(*unit, *program, options.output_file, critical)) {
            return EXIT_FAILURE;
        }
        if (critical && !options.validate_report_only) { status = EXIT_FAILURE; }
    }

    if (options.print_tu) {
        if (from_c
            && (options.output_file.empty()
                || options.output_file + ".c" == options.from_c_file))
        {
            LOG(ERROR) << "from-c: -print-tu needs an -output prefix different from "
                          "the input file\n";
            return EXIT_FAILURE;
        }
        if (!printTranslationUnit(*unit, options.output_file)) { return EXIT_FAILURE; }
    }

    if (unit->has_errors()) {
        LOG(ERROR) << "Skipping code generation due to prior diagnostics errors.\n";
        return EXIT_FAILURE;
    }

    // Lowering: the same call serves both AST sources.
    patchestry::codegen::CodeGenerator codegen(unit->context(), unit->codegen_options());
    auto module = codegen.lower_ast_to_mlir();
    if (!module.has_value()) {
        LOG(ERROR) << "Failed to emit mlir module\n";
        return EXIT_FAILURE;
    }
    if (!checkMarkerSymbols(*module, unit->markers, options.strict_symbols)) {
        return EXIT_FAILURE;
    }
    if (!codegen.emit_outputs(*module, loweringOptions(options))) { return EXIT_FAILURE; }
    return status;
}
