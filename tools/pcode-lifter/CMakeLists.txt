# Copyright (c) 2024, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

set(LLVM_LINK_COMPONENTS
  Support
  clangFrontend
)


add_executable(pcode-lifter
    main.cpp
)

llvm_update_compile_flags(pcode-lifter)
target_link_libraries(pcode-lifter
    PRIVATE
        patchestry::ghidra
        patchestry::ast
        clangFrontend
)

mlir_check_link_libraries(pcode-lifter)