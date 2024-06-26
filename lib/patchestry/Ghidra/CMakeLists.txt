# Copyright (c) 2024, Trail of Bits, Inc. All rights reserved. This source code
# is licensed in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

add_library(patchestry_ghidra STATIC Codegen.cpp Deserialize.cpp)

target_link_libraries(patchestry_ghidra
  PUBLIC
    LLVMSupport
    ${MLIR_LIBS}
    MLIRPcode
  PRIVATE
    patchestry_settings
)
