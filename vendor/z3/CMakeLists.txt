#
# Copyright (c) 2025-present, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

set(src_dir "${CMAKE_CURRENT_LIST_DIR}/src")
set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/build")

# Fetch the submodule if we don't yet have it.
if(NOT EXISTS "${src_dir}/CMakeLists.txt")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" submodule update --init vendor/z3/src
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${build_dir}")

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
                        "${src_dir}"
               WORKING_DIRECTORY "${build_dir}")

set(LLVM_Z3_INSTALL_DIR "${PE_VENDOR_INSTALL_DIR}" PARENT_SCOPE)

# (c) Build just mycomp in the nested build. Don't specify a --config
#     because we cannot know what config the developer will be using
#     at this point. For non-multi-config generators, we've already
#     specified CMAKE_BUILD_TYPE above in (b).
execute_process(COMMAND ${CMAKE_COMMAND} --build . --target install
                WORKING_DIRECTORY "${build_dir}")

if(DEFINED ENV{CI})
  execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${build_dir}")
endif()