# Copyright (c) 2024, Trail of Bits, Inc. All rights reserved. This source code
# is licensed in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

add_library(patchestry_ghidra STATIC
  PcodeTranslation.cpp
)

add_library(patchestry::ghidra ALIAS patchestry_ghidra)

target_link_libraries(patchestry_ghidra
  PUBLIC
    ${MLIR_LIBS}
    MLIRPcode
  PRIVATE
    LLVMSupport
    patchestry_settings
)
