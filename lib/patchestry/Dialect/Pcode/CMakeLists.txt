# Copyright (c) 2024, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.

add_mlir_dialect_library(MLIRPcode
  PcodeDialect.cpp
  PcodeOps.cpp
  PcodeTypes.cpp
  Deserialize.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/patchestry
  DEPENDS
    MLIRPcodeIncGen
  LINK_LIBS
    patchestry_settings
  PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)
