# Copyright (c) 2024, Trail of Bits, Inc. All rights reserved. This source code
# is licensed in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

cmake_minimum_required(VERSION 3.25)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(prevent_in_source_builds)
include(project_settings)

project(
  patchestry
  VERSION 0.1.0
  DESCRIPTION
    "Patchestry is a binary patching framework built with MLIR and Ghidra."
  HOMEPAGE_URL "https://github.com/lifting-bits/patchestry"
  LANGUAGES C CXX)

include(cmake/variables.cmake)

find_package(LLVM 18.1 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(HandleLLVMOptions)

find_package(MLIR ${LLVM_PACKAGE_VERSION} REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(${PROJECT_BINARY_DIR}/include/)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

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
