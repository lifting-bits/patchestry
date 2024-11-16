# Copyright (c) 2024, Trail of Bits, Inc. This source code is licensed
# in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

add_library(patchestry_ast STATIC
  ASTConsumer.cpp
  Codegen.cpp
  OperationBuilder.cpp
  OperationStmt.cpp
  TypeBuilder.cpp
  Utils.cpp
)

set(VAST_LIBS
  VAST::VASTTargetLLVMIR
  VAST::VASTToLLVMConversionPasses
  VAST::VASTAliasTypeInterface
  VAST::VASTElementTypeInterface
  VAST::VASTCodeGen
  VAST::VASTFrontend
  VAST::VASTSymbolInterface
  VAST::VASTSymbolTableInterface
  VAST::VASTSymbolRefInterface
  VAST::VASTTypeDefinitionInterface
)

add_library(patchestry::ast ALIAS patchestry_ast)

target_link_libraries(patchestry_ast
  PUBLIC
    clangFrontend
  PRIVATE
    LLVMSupport
    patchestry_settings
    ${VAST_LIBS}
)