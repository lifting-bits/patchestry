# Copyright (c) 2024, Trail of Bits, Inc.
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.

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

if(POLICY CMP0127)
  cmake_policy(SET CMP0127 NEW)
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
include(options)

#
# Setup package version
#
setup_package_version_variables(patchestry)

set(PATCHESTRY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PATCHESTRY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PATCHESTRY_INCLUDE_DIR ${PATCHESTRY_SOURCE_DIR}/include)

# Configure Patchestry Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/patchestry/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/patchestry/Version.inc
)

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
  set(CMAKE_INSTALL_RPATH "@executable_path/../${PE_INSTALL_LIB_DIR}")
elseif(UNIX AND NOT APPLE)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../${PE_INSTALL_LIB_DIR}")
endif()

add_subdirectory("vendor")

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
find_package(LLVM 20 CONFIG REQUIRED HINTS "${PE_VENDOR_INSTALL_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(HandleLLVMOptions)

find_package(MLIR 20 CONFIG REQUIRED HINTS "${PE_VENDOR_INSTALL_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

find_package(Clang 20 CONFIG REQUIRED HINTS "${PE_VENDOR_INSTALL_DIR}")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")


set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)


include_directories(SYSTEM "${PE_VENDOR_INSTALL_DIR}/include")
find_package(Z3 4.8 CONFIG REQUIRED HINTS "${PE_VENDOR_INSTALL_DIR}")
message(STATUS "Using Z3Config.cmake in ${Z3_DIR} and Z3 headers in ${PE_VENDOR_INSTALL_DIR}/include")

find_package(rellic REQUIRED HINTS "${PE_VENDOR_INSTALL_DIR}")
message(STATUS "Using RellicConfig.cmake in: ${rellic_DIR}")
get_target_property(RELLIC_INCLUDE_DIRS rellic::rellic INTERFACE_INCLUDE_DIRECTORIES)

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

add_custom_target(patchestry-generic-headers)
set_target_properties(patchestry-generic-headers PROPERTIES FOLDER "Misc")
add_custom_target(patchestry-headers)
set_target_properties(patchestry-headers PROPERTIES FOLDER "Misc")
add_dependencies(patchestry-headers patchestry-generic-headers)
add_custom_target(patchestry-tools)
set_target_properties(patchestry-tools PROPERTIES FOLDER "Misc")
add_custom_target(patchestry-doc)

define_property(GLOBAL PROPERTY VAST_INSTALL_TARGETS)

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

if (PATCHESTRY_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

#
# install settings
#

option(PATCHESTRY_INSTALL "Generate the install target." ${PATCHESTRY_MASTER_PROJECT})

#
# packaging support
#

if (PATCHESTRY_INSTALL)
  set(PATCHESTRY_CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  get_property(PATCHESTRY_INSTALL_TARGETS GLOBAL PROPERTY PATCHESTRY_INSTALL_TARGETS)

  install(DIRECTORY ${PATCHESTRY_INCLUDE_DIR}/patchestry
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT patchestry-headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.h.inc"
    PATTERN "*.hpp.inc"
    PATTERN "CMakeFiles" EXCLUDE
  )

  install(DIRECTORY ${PATCHESTRY_SOURCE_DIR}/include/patchestry
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT patchestry-headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "CMakeFiles" EXCLUDE
  )

  install(TARGETS patchestry_settings EXPORT PATCHESTRYTargets)

  set(PATCHESTRY_EXPORT_NAME PATCHESTRYTargets)

  install(EXPORT PATCHESTRYTargets
    FILE ${PATCHESTRY_EXPORT_NAME}.cmake
    NAMESPACE PATCHESTRY::
    DESTINATION ${PATCHESTRY_CMAKE_INSTALL_DIR}
  )

  set(CPACK_PACKAGE_VENDOR "Trail of Bits")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "Patchestry: an MLIR-based binary patching framework."
  )
  set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH})
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
  set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
  set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package")
  set(CPACK_PACKAGE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  include(CPack)

  set(PATCHESTRY_CONFIG_NAME "${PROJECT_NAME}Config")
  set(PATCHESTRY_PACKAGE_CONFIG_FILE "${PATCHESTRY_CONFIG_NAME}.cmake")
  set(PATCHESTRY_PACKAGE_CONFIG_VERSION_FILE "${PATCHESTRY_CONFIG_NAME}Version.cmake")

  include(CMakePackageConfigHelpers)

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PATCHESTRY_CONFIG_NAME}.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PATCHESTRY_PACKAGE_CONFIG_FILE}"
    INSTALL_DESTINATION ${PATCHESTRY_CMAKE_INSTALL_DIR}
  )

  write_basic_package_version_file(
    ${PATCHESTRY_PACKAGE_CONFIG_VERSION_FILE}
    VERSION ${CPACK_PACKAGE_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PATCHESTRY_PACKAGE_CONFIG_FILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${PATCHESTRY_PACKAGE_CONFIG_VERSION_FILE}"
    DESTINATION ${PATCHESTRY_CMAKE_INSTALL_DIR}
  )

  message(STATUS CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX})

  # Executables are handled in the respective tools/CMakeLists.txt files.
  # TODO: Should we be moving over html docs and other files?
  install(DIRECTORY ${PE_VENDOR_INSTALL_DIR}/
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.h.inc"
    PATTERN "*.hpp.inc"
    PATTERN "*.a"
    PATTERN "*.so"
    PATTERN "*.dylib"
    PATTERN "*.dll"
    PATTERN "*.lib"
    PATTERN "*.cmake"
  )

endif()
