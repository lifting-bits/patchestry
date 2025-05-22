#
# Copyright (c) 2025-present, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
cmake_minimum_required(VERSION 3.20.0)

find_package(Git REQUIRED)

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${PE_VENDOR_INSTALL_DIR}")

set(CMAKE_LINKER_TYPE LLD CACHE STRING "Linker type for the Patchestry build should overall be LLD!")

# Build components in dependency order
add_subdirectory(gflags)
add_subdirectory(glog)  # Depends on gflags
add_subdirectory(z3)

if(USE_VENDORED_CLANG)
    add_subdirectory(clangir)
endif()

add_subdirectory(rellic)  # Depends on glog and clangir (if vendored)
