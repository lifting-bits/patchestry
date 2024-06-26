/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Util/Warnings.hpp"

PATCHESTRY_RELAX_WARNINGS
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/InitAllDialects.h>
PATCHESTRY_UNRELAX_WARNINGS

#include <cstdlib>
#include <iostream>
#include <span>
#include <string>

#include "patchestry/Dialect/Pcode/PcodeDialect.hpp"
#include "patchestry/Ghidra/Codegen.hpp"
#include "patchestry/Ghidra/Deserialize.hpp"

void print_help() {
    std::stringstream ss;
    ss << "Usage:\n";
    ss << "\tPCODE_JSON_FILE\n";
    ss << "\t[OUTPUT_FILE]\n";
    std::cout << ss.str();
}

auto main(int argc, char **argv) -> int try
{
    const std::span args(argv, static_cast< size_t >(argc));

    if (argc == 1 || std::string(args[1]) == "--help") {
        print_help();
        return EXIT_FAILURE;
    }

    auto ibuf = llvm::MemoryBuffer::getFileOrSTDIN(args[1], /*IsText=*/true);
    if (!ibuf) {
        auto msg = ibuf.getError().message();
        llvm::errs() << "Failed to open pcode file: " << msg << '\n';
        return EXIT_FAILURE;
    }

    auto root = llvm::json::parse(ibuf.get()->getBuffer());
    if (!root) {
        auto msg = llvm::toString(root.takeError());
        llvm::errs() << "Failed to parse pcode file: " << msg << '\n';
        return EXIT_FAILURE;
    }

    assert(root->kind() == llvm::json::Value::Kind::Object);

    auto func = patchestry::ghidra::function_t::from_json(*root->getAsObject());
    if (!func) {
        llvm::errs() << "Failed to parse pcode file: " << llvm::toString(func.takeError())
                     << '\n';
        return EXIT_FAILURE;
    }

    std::error_code err;
    llvm::raw_fd_ostream ofs(args[2], err, llvm::sys::fs::OF_Text);

    mlir::DialectRegistry registry;
    registry.insert< patchestry::pc::PcodeDialect >();
    mlir::registerAllDialects(registry);

    mlir::MLIRContext ctx(registry);

    ctx.loadAllAvailableDialects();

    auto loc = mlir::UnknownLoc::get(&ctx);
    auto mod = mlir::OwningOpRef< mlir::ModuleOp >(mlir::ModuleOp::create(loc));

    patchestry::ghidra::mlir_codegen_visitor(*mod).visit(*func);

    mlir::OpPrintingFlags flags;
    flags.enableDebugInfo(/* print locations */ false, /* prettyForm */ true);

    mod->print(ofs, flags);

    return EXIT_SUCCESS;
} catch (std::exception &err) {
    llvm::errs() << "error: " << err.what() << '\n';
    return EXIT_FAILURE;
}
