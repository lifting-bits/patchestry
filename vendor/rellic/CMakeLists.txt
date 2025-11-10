#
# Copyright (c) 2025-present, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
cmake_minimum_required(VERSION 3.20.0)

set(src_dir "${CMAKE_CURRENT_LIST_DIR}/src")
set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/build")

# Fetch the submodule if we don't yet have it.
if(NOT EXISTS "${src_dir}/CMakeLists.txt")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" submodule update --init vendor/rellic/src
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

  execute_process(
    COMMAND "${GIT_EXECUTABLE}" submodule set-branch --branch llvm20 vendor/rellic/src
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

# Only fetch doctest if testing is enabled
if(RELLIC_ENABLE_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
  )
  FetchContent_MakeAvailable(doctest)
endif()

set(RELLIC_PREFIX_PATH ${CMAKE_PREFIX_PATH})

if (PE_USE_VENDORED_CLANG)
  set(RELLIC_PREFIX_PATH ${PE_VENDOR_INSTALL_DIR})
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${build_dir}")

execute_process(COMMAND ${CMAKE_COMMAND}
                        -G "${CMAKE_GENERATOR}"
                        "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
                        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                        "-DCMAKE_INSTALL_PREFIX=${PE_VENDOR_INSTALL_DIR}"
                        "-DCMAKE_PREFIX_PATH=${RELLIC_PREFIX_PATH}"
                        "-DCMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}"
                        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                        "-Dglog_DIR=${PE_VENDOR_INSTALL_DIR}/lib/cmake/glog"
                        -DRELLIC_ENABLE_INSTALL=ON
                        -DRELLIC_ENABLE_TESTING=OFF
                        "${src_dir}"
               WORKING_DIRECTORY "${build_dir}")

# (c) Build just mycomp in the nested build. Don't specify a --config
#     because we cannot know what config the developer will be using
#     at this point. For non-multi-config generators, we've already
#     specified CMAKE_BUILD_TYPE above in (b).
execute_process(COMMAND ${CMAKE_COMMAND} --build . --target install
                WORKING_DIRECTORY "${build_dir}")

if(DEFINED ENV{CI})
  execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${build_dir}")
endif()
