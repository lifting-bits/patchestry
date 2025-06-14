#
# Copyright (c) 2025-present, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
cmake_minimum_required(VERSION 3.20.0)

if(NOT PE_USE_VENDORED_GFLAGS)
  find_package(gflags CONFIG QUIET)
  if(gflags_FOUND)
    if(NOT TARGET gflags::gflags)
      if(TARGET gflags::gflags_static)
        add_library(gflags::gflags ALIAS gflags::gflags_static)
      elseif(TARGET gflags_static)
        add_library(gflags::gflags_static ALIAS gflags_static)
        add_library(gflags::gflags ALIAS gflags_static)
      else()
        message(FATAL_ERROR "Cannot find appropriate gflags target; known targets are ${gflags_LIBRARIES}")
      endif()
    endif()
    return()
  endif()
endif()

set(src_dir "${CMAKE_CURRENT_LIST_DIR}/src")
set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/build")

# Fetch the submodule if we don't yet have it.
if(NOT EXISTS "${src_dir}/CMakeLists.txt")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" submodule update --init vendor/gflags/src
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

# This is the main build, setup and execute the nested build
# to ensure the gflags library exists before continuing

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${build_dir}")

# (b) Nested CMake run. May need more -D... options than shown here.
execute_process(COMMAND ${CMAKE_COMMAND}
                        -G "${CMAKE_GENERATOR}"
                        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                        "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
                        "-DCMAKE_INSTALL_PREFIX=${PE_VENDOR_INSTALL_DIR}"
                        "-DCMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}"
                        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                        "-DCMAKE_LINKER_TYPE=${CMAKE_LINKER_TYPE}"
                        "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
                        -DBUILD_SHARED_LIBS=OFF
                        -DBUILD_STATIC_LIBS=ON
                        -DBUILD_gflags_LIB=ON
                        -DBUILD_gflags_nothreads_LIB=OFF
                        -DBUILD_PACKAGING=ON
                        -DBUILD_TESTING=OFF
                        -DINSTALL_HEADERS=ON
                        -DINSTALL_SHARED_LIBS=OFF
                        -DINSTALL_STATIC_LIBS=ON
                        -DREGISTER_BUILD_DIR=OFF
                        -DREGISTER_INSTALL_PREFIX=OFF
                        "${src_dir}"
               WORKING_DIRECTORY "${build_dir}")

# (c) Build just mycomp in the nested build. Don't specify a --config
#     because we cannot know what config the developer will be using
#     at this point. For non-multi-config generators, we've already
#     specified CMAKE_BUILD_TYPE above in (b).
execute_process(COMMAND ${CMAKE_COMMAND} --build . --target install
                WORKING_DIRECTORY "${build_dir}")

find_package(gflags CONFIG REQUIRED HINTS "${PE_VENDOR_INSTALL_DIR}" NO_DEFAULT_PATH)

if(DEFINED ENV{CI})
  execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${build_dir}")
endif()
