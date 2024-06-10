# Copyright (c) 2024, Trail of Bits, Inc. All rights reserved. This source code
# is licensed in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

add_library(PatchestryGhidra STATIC Deserialize.cpp)

target_include_directories(
  PatchestryGhidra ${warning_guard}
  PUBLIC "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tools>"
         "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "\$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>"
         "\$<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>")

target_compile_features(PatchestryGhidra PUBLIC cxx_std_20)
target_link_libraries(PatchestryGhidra PUBLIC LLVMSupport)
