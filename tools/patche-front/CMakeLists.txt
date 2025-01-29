# Copyright (c) 2024, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

set(LLVM_LINK_COMPONENTS
  Support
  clangFrontend
)


add_executable(patche-front
    main.cpp
)

llvm_update_compile_flags(patche-front)

target_link_libraries(patche-front
  PRIVATE
    patchestry::ghidra
    patchestry::ast
    patchestry::codegen
    clangFrontend
)

mlir_check_link_libraries(patche-front)