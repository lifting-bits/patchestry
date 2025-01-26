# Copyright (c) 2024, Trail of Bits, Inc. This source code is licensed
# in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

add_library(patchestry_codegen STATIC
  Codegen.cpp
  Initializer.cpp
  MetaGenerator.cpp
  PassInstrumentation.cpp
  Serializer.cpp
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
  VAST::VASTTower
  VAST::VASTDeclStorageInterface
)

add_library(patchestry::codegen ALIAS patchestry_codegen)

target_link_libraries(patchestry_codegen
  PUBLIC
    clangFrontend
  PRIVATE
    LLVMSupport
    patchestry_settings
    ${VAST_LIBS}
)