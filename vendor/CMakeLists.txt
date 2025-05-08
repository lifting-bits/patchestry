#
# Copyright (c) 2025-present, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

find_package(Git REQUIRED)

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${PE_VENDOR_INSTALL_DIR}")


add_subdirectory(gflags)
add_subdirectory(glog)  # Depends on gflags
if(USE_VENDORED_CLANG)
    add_subdirectory(clangir)
endif()
add_subdirectory(rellic)