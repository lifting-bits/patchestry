# Copyright (c) 2024, Trail of Bits, Inc. All rights reserved. This source code
# is licensed in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

cmake_minimum_required(VERSION 3.25)

include(CTest)

find_package(Python REQUIRED COMPONENTS Interpreter)
if (Python_Interpreter_FOUND)
  set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
endif()

# Configure the lit site configuration file
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Add a lit test suite named "PatchestryTest"
add_lit_testsuite(PatchestryTest "Running Patchestry tests"
  ${CMAKE_CURRENT_SOURCE_DIR}/ghidra
)

# Add a CTest test that runs the lit test suite
add_test(NAME lit
  COMMAND lit -v "${CMAKE_CURRENT_BINARY_DIR}/ghidra"
  --param BUILD_TYPE=$<CONFIG>)