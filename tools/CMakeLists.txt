# Copyright (c) 2024, Trail of Bits, Inc. All rights reserved. This source code
# is licensed in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

add_library(patchestry_lib OBJECT ghidra.cpp)

target_include_directories(
  patchestry_lib ${warning_guard}
  PUBLIC "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tools>")

target_compile_features(patchestry_lib PUBLIC cxx_std_20)
target_link_libraries(patchestry_lib PUBLIC LLVMSupport)


add_executable(patchestry_exe main.cpp)
add_executable(patchestry::exe ALIAS patchestry_exe)

set_property(TARGET patchestry_exe PROPERTY OUTPUT_NAME patchestry)
target_compile_features(patchestry_exe PRIVATE cxx_std_20)
target_link_libraries(patchestry_exe PRIVATE patchestry_lib)
