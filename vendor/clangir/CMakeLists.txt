#
# Copyright (c) 2025-present, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

set(src_dir "${CMAKE_CURRENT_LIST_DIR}/src")
set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/build")

# Fetch the submodule if we don't yet have it.
if(NOT EXISTS "${src_dir}/llvm/CMakeLists.txt")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" submodule update --init vendor/llvm-project/src
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

  execute_process(
    COMMAND "${GIT_EXECUTABLE}" submodule set-branch --branch patche-clangir-20 vendor/clangir/src
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

# This is the main build, setup and execute the nested build
# to ensure the Clang and MLIR libraries exist before continuing

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${build_dir}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(LLVM_ENABLE_ASSERTIONS ON)
else()
  set(LLVM_ENABLE_ASSERTIONS OFF)
endif()

if(UNIX AND NOT APPLE)
  set(LLVM_ENABLE_LLD ON)
else()
  set(LLVM_ENABLE_LLD OFF)
endif()

# (b) Nested CMake run. May need more -D... options than shown here.
execute_process(COMMAND ${CMAKE_COMMAND}
                        -G "${CMAKE_GENERATOR}"
                        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                        "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
                        "-DCMAKE_INSTALL_PREFIX=${PE_VENDOR_INSTALL_DIR}"
                        "-DCMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}"
                        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                        "-DCMAKE_ASM_COMPILER=${CMAKE_C_COMPILER}"
                        "-DCMAKE_LINKER_TYPE=${CMAKE_LINKER_TYPE}"
                        -DCMAKE_CXX_STANDARD=17
                        "-DLLVM_ENABLE_PROJECTS=clang;mlir"
                        "-DLLVM_TARGETS_TO_BUILD=AArch64;X86;ARM"
                        "-DLLVM_ENABLE_RUNTIMES=libcxx;libcxxabi"
                        -DLIBCXXABI_USE_LLVM_UNWINDER:BOOL=OFF
                        -DBUILD_SHARED_LIBS:BOOL=OFF
                        -DLLVM_APPEND_VC_REV:BOOL=OFF
                        -DLLVM_INCLUDE_EXAMPLES:BOOL=OFF
                        -DLLVM_BUILD_EXAMPLES:BOOL=OFF
                        -DLLVM_BUILD_LLVM_DYLIB:BOOL=OFF
                        -DLLVM_BUILD_TESTS:BOOL=OFF
                        -DLLVM_ENABLE_BINDINGS:BOOL=OFF
                        -DLLVM_ENABLE_OCAMLDOC:BOOL=OFF
                        -DLLVM_ENABLE_DIA_SDK:BOOL=OFF
                        -DLLVM_ENABLE_EH:BOOL=ON
                        -DLLVM_ENABLE_RTTI:BOOL=ON
                        -DCLANG_ENABLE_CIR:BOOL=ON
                        "-DLLVM_ENABLE_ASSERTIONS:BOOL=${LLVM_ENABLE_ASSERTIONS}"
                        -DLLVM_ENABLE_EXPENSIVE_CHECKS:BOOL=OFF
                        -DLLVM_ENABLE_FFI:BOOL=OFF
                        -DLLVM_ENABLE_WARNINGS:BOOL=ON
                        -DLLVM_INCLUDE_BENCHMARKS:BOOL=OFF
                        -DLLVM_INCLUDE_EXAMPLES:BOOL=OFF
                        -DLLVM_INCLUDE_TESTS:BOOL=OFF
                        -DLLVM_INCLUDE_TOOLS:BOOL=ON
                        -DLLVM_INSTALL_UTILS:BOOL=ON
                        -DLLVM_ENABLE_ZSTD:BOOL=OFF
                        -DLLVM_ENABLE_LLD:BOOL=${LLVM_ENABLE_LLD}
                        "${src_dir}/llvm"
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