# Copyright (c) 2025, Trail of Bits, Inc. This source code is licensed
# in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

add_library(patchestry_codegen STATIC
  Codegen.cpp
  PassManager.cpp
  Serializer.cpp
)

set(CLANG_IR_LIBS
  clangCIR
  clangCIRLoweringDirectToLLVM
  clangCIRLoweringThroughMLIR
)

add_library(patchestry::codegen ALIAS patchestry_codegen)

target_link_libraries(patchestry_codegen
  PUBLIC
    clangFrontend
  PRIVATE
    LLVMSupport
    patchestry_settings
    rellic::rellic
    glog::glog
    ${CLANG_IR_LIBS}
)

target_include_directories(patchestry_codegen
  PRIVATE
    "$<BUILD_INTERFACE:${RELLIC_INCLUDE_DIRS}>"
)
