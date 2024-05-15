# Copyright (c) 2024, Trail of Bits, Inc.
# All rights reserved.
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    patchestry
    VERSION 0.1.0
    DESCRIPTION "Patchestry is a binary patching framework built with MLIR and Ghidra."
    HOMEPAGE_URL "https://github.com/lifting-bits/patchestry"
    LANGUAGES C CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

find_package(LLVM 17 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(HandleLLVMOptions)

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
      AUTHOR_WARNING
      "Developer mode is intended for developers of patchestry"
  )
endif()

include(cmake/dev-mode.cmake)
