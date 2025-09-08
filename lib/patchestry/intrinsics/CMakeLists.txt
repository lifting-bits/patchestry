# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)

# Support standalone building or as part of patchestry
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(patchestry_intrinsics
    VERSION 0.1.0
    DESCRIPTION "Patchestry Intrinsics Library"
    LANGUAGES C
  )
  set(PATCHESTRY_STANDALONE_BUILD ON)
else()
  set(PATCHESTRY_STANDALONE_BUILD OFF)
endif()

set(PATCHESTRY_INTRINSICS_SOURCES
  runtime.c
  safety.c
  device.c
  patch_utils.c
)

add_library(patchestry_intrinsics STATIC ${PATCHESTRY_INTRINSICS_SOURCES})

# Configure include directories based on build mode
if(PATCHESTRY_STANDALONE_BUILD)
  # For standalone builds, look for headers in relative paths
  target_include_directories(patchestry_intrinsics
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
      ${CMAKE_CURRENT_LIST_DIR}/../../../include
  )
else()
  # For integrated builds, use the main project's include directory
  target_include_directories(patchestry_intrinsics
    PUBLIC
      ${CMAKE_SOURCE_DIR}/include
  )
endif()

# Platform and architecture detection
if(NOT DEFINED TARGET_ARCH)
  set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

# ARMv7 specific configuration
option(PATCHESTRY_TARGET_ARMV7 "Build for ARMv7 architecture" OFF)

if(PATCHESTRY_TARGET_ARMV7)
  set(TARGET_ARCH "armv7")
  message(STATUS "Building patchestry_intrinsics for ARMv7 architecture")
  
  # Only apply ARMv7 flags for appropriate compilers/platforms
  if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR (CMAKE_C_COMPILER_ID MATCHES "Clang" AND NOT APPLE))
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard")
  elseif(APPLE)
    message(WARNING "ARMv7 target requested but Apple Clang doesn't support ARMv7. Use appropriate cross-compilation toolchain.")
    # For Apple platforms, you'd typically use a different approach
    set(CMAKE_OSX_ARCHITECTURES armv7 armv7s)
  endif()
endif()

target_link_libraries(patchestry_intrinsics
  PUBLIC
    pthread
    dl
)

# Set compilation flags for intrinsics
target_compile_options(patchestry_intrinsics PRIVATE
  -Wall
  -Wextra
  -Wno-unused-parameter
  -fPIC
  $<$<COMPILE_LANGUAGE:C>:-std=c99>
  $<$<C_COMPILER_ID:GNU,Clang>:-pedantic>
)

# Cross-compilation support for ARMv7
if(CMAKE_CROSSCOMPILING AND PATCHESTRY_TARGET_ARMV7)
  # Set specific toolchain flags for ARMv7 cross-compilation
  if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set_target_properties(patchestry_intrinsics PROPERTIES
      COMPILE_FLAGS "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard"
    )
  endif()
  
  # Remove pthread and dl linking for bare-metal ARMv7 targets
  option(PATCHESTRY_BARE_METAL "Build for bare-metal target (no pthread/dl)" OFF)
  if(PATCHESTRY_BARE_METAL)
    set_target_properties(patchestry_intrinsics PROPERTIES
      LINK_LIBRARIES ""
    )
  endif()
endif()

# Display build configuration
message(STATUS "Patchestry Intrinsics Configuration:")
message(STATUS "  Target Architecture: ${TARGET_ARCH}")
message(STATUS "  ARMv7 Target: ${PATCHESTRY_TARGET_ARMV7}")
message(STATUS "  Standalone Build: ${PATCHESTRY_STANDALONE_BUILD}")
message(STATUS "  Cross Compiling: ${CMAKE_CROSSCOMPILING}")

# Install the library (conditional for standalone builds)
if(PATCHESTRY_STANDALONE_BUILD)
  install(TARGETS patchestry_intrinsics
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  # Install headers for standalone builds
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/patchestry/intrinsics")
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/patchestry/intrinsics
      DESTINATION include/patchestry
      FILES_MATCHING PATTERN "*.h"
    )
  endif()
else()
  # For integrated builds, let the parent handle installation
  install(TARGETS patchestry_intrinsics
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  # Install headers for integrated builds
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/patchestry/intrinsics
    DESTINATION include/patchestry
    FILES_MATCHING PATTERN "*.h"
  )
endif()
