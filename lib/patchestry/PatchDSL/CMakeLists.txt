# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

add_library(patchestry_patchdsl STATIC
    Compiler.cpp
    Lexer.cpp
    Lowering.cpp
    Parser.cpp
)

add_library(patchestry::patchdsl ALIAS patchestry_patchdsl)

target_link_libraries(patchestry_patchdsl
    PUBLIC
        LLVMSupport
        patchestry::yaml
    PRIVATE
        patchestry_settings
)
