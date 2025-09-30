# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.

add_mlir_dialect_library(MLIRContracts
  ContractsDialect.cpp
  ContractsAttrs.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/patchestry
  DEPENDS
    MLIRContractsIncGen
  LINK_LIBS
    patchestry_settings
  PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)

target_include_directories(obj.MLIRContracts PRIVATE
  ${PROJECT_SOURCE_DIR}/include/patchestry/Dialect/Contracts
  ${PROJECT_BINARY_DIR}/include/patchestry/Dialect/Contracts
)