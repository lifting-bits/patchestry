/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <llvm/Support/MemoryBuffer.h>
#pragma clang diagnostic pop

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>

#include "ghidra.hpp"

auto main(int argc, char **argv) -> int {
    const std::span args(argv, static_cast< size_t >(argc));

    if (argc != 3) {
        std::cerr << "Usage:  " << args[0] << " PCODE_FILE\n";
        return EXIT_FAILURE;
    }

    auto ibuf = llvm::MemoryBuffer::getFileOrSTDIN(args[1], /*IsText=*/true);
    if (!ibuf) {
        auto msg = ibuf.getError().message();
        std::cerr << "Failed to open pcode file: " << msg << '\n';
        return EXIT_FAILURE;
    }

    auto root = llvm::json::parse(ibuf.get()->getBuffer());
    if (!root) {
        auto msg = llvm::toString(root.takeError());
        std::cerr << "Failed to parse pcode file: " << msg << '\n';
        return EXIT_FAILURE;
    }

    assert(root->kind() == llvm::json::Value::Kind::Object);

    auto func = patchestry::ghidra::function_t::from_json(*root->getAsObject());
    if (!func) {
        std::cerr << "Failed to parse pcode file: " << llvm::toString(func.takeError()) << '\n';
        return EXIT_FAILURE;
    }

    std::ofstream ofs(args[2]);

    ofs << "Function name: " << func->name << '\n';
    ofs << "Number of basic blocks: " << func->basic_blocks.size() << '\n';

    const size_t num_insts = [&] {
        size_t sum = 0;
        for (const auto &block : func->basic_blocks) {
            sum += block.instructions.size();
        }
        return sum;
    }();

    ofs << "Number of instructions: " << num_insts << '\n';

    return EXIT_SUCCESS;
}