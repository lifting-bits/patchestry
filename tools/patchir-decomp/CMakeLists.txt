# Copyright (c) 2024, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

set(LLVM_LINK_COMPONENTS
  Support
  clangFrontend
)


add_executable(patchir-decomp
    main.cpp
)

llvm_update_compile_flags(patchir-decomp)
target_link_libraries(patchir-decomp
    PRIVATE
        patchestry::ghidra
        patchestry::ast
        patchestry::codegen
        rellic::rellic
        glog::glog
        clangFrontend
)


mlir_check_link_libraries(patchir-decomp)