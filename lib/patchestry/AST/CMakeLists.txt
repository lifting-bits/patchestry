# Copyright (c) 2024, Trail of Bits, Inc. This source code is licensed
# in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

add_library(patchestry_ast STATIC
  ASTConsumer.cpp
  FunctionBuilder.cpp
  IntrinsicHandlers.cpp
  OperationBuilder.cpp
  OperationStmt.cpp
  TypeBuilder.cpp
  Utils.cpp
)

add_library(patchestry::ast ALIAS patchestry_ast)

target_link_libraries(patchestry_ast
  PUBLIC
    clangFrontend
  PRIVATE
    LLVMSupport
    patchestry_settings
    glog::glog
)

target_include_directories(patchestry_ast
  PRIVATE
    "$<BUILD_INTERFACE:${RELLIC_INCLUDE_DIRS}>"
)
