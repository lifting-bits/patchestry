# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

set(LLVM_LINK_COMPONENTS
  Support
  clangFrontend
)


add_executable(patchir-transform
    main.cpp
)

llvm_update_compile_flags(patchir-transform)
target_link_libraries(patchir-transform
    PRIVATE
        patchestry::codegen
        clangFrontend
        patchestry::passes
        MLIROptLib
)

mlir_check_link_libraries(patchir-transform)
