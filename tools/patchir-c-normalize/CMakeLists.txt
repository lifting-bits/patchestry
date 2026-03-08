# Copyright (c) 2024, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

set(LLVM_LINK_COMPONENTS
  Support
  clangFrontend
)

add_executable(patchir-c-normalize
    main.cpp
)

llvm_update_compile_flags(patchir-c-normalize)
target_link_libraries(patchir-c-normalize
    PRIVATE
        patchestry::ast
        glog::glog
        clangFrontend
        clangParse
        clangSema
        clangAST
        clangBasic
        clangLex
)

mlir_check_link_libraries(patchir-c-normalize)

if(PATCHESTRY_INSTALL)
  install(TARGETS patchir-c-normalize
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT patchestry-tools
  )
endif()
