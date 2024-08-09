# Copyright (c) 2024, Trail of Bits, Inc. All rights reserved. This source code
# is licensed in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

cmake_minimum_required(VERSION 3.25)

if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif()

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if (POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(prevent_in_source_builds)

# check if patchestry is being used directly or via add_subdirectory,
# but allow overriding
if(NOT DEFINED PATCHESTRY_MASTER_PROJECT)
    if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(PATCHESTRY_MASTER_PROJECT ON)
    else()
        set(PATCHESTRY_MASTER_PROJECT OFF)
    endif()
endif()

project(
  patchestry
  VERSION 0.1.0
  DESCRIPTION
    "Patchestry is a binary patching framework built with MLIR and Ghidra."
  HOMEPAGE_URL "https://github.com/lifting-bits/patchestry"
  LANGUAGES C CXX
)

include(project_settings)
include(project_utils)

#
# Setup package version
#
setup_package_version_variables(patchestry)

set(PATCHESTRY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PATCHESTRY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PATCHESTRY_INCLUDE_DIR ${PATCHESTRY_BINARY_DIR}/include)

#
# CCACHE
#
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

FindAndSelectClangCompiler()

#
# LLVM & MLIR & Clang
#
find_package(LLVM 18.1 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(HandleLLVMOptions)

find_package(MLIR ${LLVM_PACKAGE_VERSION} REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddMLIR)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS})

if (MLIR_LINK_MLIR_DYLIB)
  set(MLIR_LIBS MLIR)
else()
  get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  get_property(MLIR_EXTENSION_LIBS GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
  get_property(MLIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

  set(MLIR_LIBS
    MLIRAnalysis
    MLIRDialect
    MLIRExecutionEngine
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTransformUtils

    ${MLIR_DIALECT_LIBS}
    ${MLIR_CONVERSION_LIBS}
    ${MLIR_EXTENSION_LIBS}
    ${MLIR_TRANSLATION_LIBS}
  )
endif()

add_library(patchestry_settings INTERFACE)

target_include_directories(patchestry_settings INTERFACE
  $<BUILD_INTERFACE:${PATCHESTRY_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PATCHESTRY_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

include(compiler_warnings)
set_project_warnings(patchestry_settings)

# sanitizer options if supported by compiler
include(sanitizers)
enable_sanitizers(patchestry_settings)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

option(PATCHESTRY_ENABLE_TESTING "Enable Test Builds" ON)

if(PATCHESTRY_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()