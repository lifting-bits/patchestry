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

  # Display build configuration
  message(STATUS "Patchestry Intrinsics Configuration:")
  message(STATUS "  Standalone Build: ${PATCHESTRY_STANDALONE_BUILD}")

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
