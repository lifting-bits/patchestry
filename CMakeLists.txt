# Copyright (c) 2024, Trail of Bits, Inc. All rights reserved. This source code
# is licensed in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
  patchestry
  VERSION 0.1.0
  DESCRIPTION
    "Patchestry is a binary patching framework built with MLIR and Ghidra."
  HOMEPAGE_URL "https://github.com/lifting-bits/patchestry"
  LANGUAGES C CXX)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Add LLVM ----

find_package(LLVM 17 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# ---- Declare library ----

add_library(
    patchestry_lib OBJECT
    source/ghidra.cpp
)

target_include_directories(
    patchestry_lib ${warning_guard}
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
    "\$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>"
)

target_compile_features(patchestry_lib PUBLIC cxx_std_20)

target_link_libraries(patchestry_lib PUBLIC LLVMSupport)

# ---- Declare executable ----

add_executable(patchestry_exe source/main.cpp)
add_executable(patchestry::exe ALIAS patchestry_exe)

set_property(TARGET patchestry_exe PROPERTY OUTPUT_NAME patchestry)

target_compile_features(patchestry_exe PRIVATE cxx_std_20)

target_link_libraries(patchestry_exe PRIVATE patchestry_lib)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT patchestry_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
    AUTHOR_WARNING "Developer mode is intended for developers of patchestry")
endif()

include(cmake/dev-mode.cmake)
