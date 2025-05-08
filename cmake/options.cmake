#
# Copyright (c) 2025-present, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
set(CMAKE_EXPORT_COMPILE_COMMANDS true CACHE BOOL "Generate the compile_commands.json file (forced)" FORCE)

set(CMAKE_INSTALL_LIBDIR     "lib"     CACHE PATH "Default library installation directory")
set(CMAKE_INSTALL_BINDIR     "bin"     CACHE PATH "Default binary installation directory")
set(CMAKE_INSTALL_INCLUDEDIR "include" CACHE PATH "Default include installation directory")
set(CMAKE_INSTALL_DATADIR    "share"   CACHE PATH "Default data installation directory")

set(PE_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Directory in which Multiplier libraries will be installed")
set(PE_INSTALL_BIN_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH "Directory in which Multiplier binaries will be installed")
set(PE_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}" CACHE PATH "Directory in which Multiplier headers will be installed")
set(PE_INSTALL_SHARE_DIR "${CMAKE_INSTALL_DATADIR}" CACHE PATH "Directory in which Multiplier CMake files will be installed")
set(PE_VENDOR_INSTALL_DIR "${PROJECT_BINARY_DIR}/vendor/install" CACHE PATH "Directory in which multiplier's vendored dependencies are installed")

option(PE_USE_VENDORED_GLOG "Set to OFF to disable default building of Google glog as a vendored library." ON)
option(PE_USE_VENDORED_GFLAGS "Set to OFF to disable default building of gflags as a vendored library." ON)
option(PE_USE_VENDORED_CLANG "Set to OFF to disable default building of Clang/LLVM as a vendored library." OFF)
