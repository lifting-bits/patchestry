# Copyright (c) 2024, Trail of Bits, Inc. All rights reserved. This source code
# is licensed in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBS ${dialect_libs} MLIRPcode)

add_executable(patchestry_exe main.cpp)

set_property(TARGET patchestry_exe PROPERTY OUTPUT_NAME patchestry)
target_compile_features(patchestry_exe PRIVATE cxx_std_20)
target_link_libraries(patchestry_exe PRIVATE PatchestryGhidra ${LIBS})
