# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.

add_library(patchestry_yaml STATIC
    YAMLParser.cpp
)

add_library(patchestry::yaml ALIAS patchestry_yaml)

target_link_libraries(patchestry_yaml
    PUBLIC
        LLVMSupport
        patchestry_passes
    PRIVATE
        patchestry_settings
)
