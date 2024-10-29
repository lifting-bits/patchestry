# Copyright (c) 2024, Trail of Bits, Inc. This source code is licensed
# in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

add_library(patchestry_ast STATIC
  ASTBuilder.cpp
)

add_library(patchestry::ast ALIAS patchestry_ast)

target_link_libraries(patchestry_ast
  PUBLIC
    clangFrontend
  PRIVATE
    LLVMSupport
    patchestry_settings
)