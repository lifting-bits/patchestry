# Copyright (c) 2024, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.

cmake_minimum_required(VERSION 3.25)

include(CTest)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(headless-ghidra
  "Running headless Ghidra decompilation tests"
  ${CMAKE_CURRENT_SOURCE_DIR}/ghidra
)

add_lit_testsuite(pcode-translate-tests
  "Running pcode-translate tests"
  ${CMAKE_CURRENT_SOURCE_DIR}/pcode-translate
)

add_lit_testsuite(patchir-decomp-tests
  "Running patchir-decomp tests"
  ${CMAKE_CURRENT_SOURCE_DIR}/patchir-decomp
)

add_test(NAME lit
  COMMAND lit -v -j 4 "${CMAKE_CURRENT_BINARY_DIR}/ghidra"
  --param BUILD_TYPE=$<CONFIG>
)
