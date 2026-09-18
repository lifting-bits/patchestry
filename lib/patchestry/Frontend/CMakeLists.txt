# Copyright (c) 2026, Trail of Bits, Inc. This source code is licensed
# in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

add_library(patchestry_frontend STATIC
  ClangFrontend.cpp
)

add_library(patchestry::frontend ALIAS patchestry_frontend)

target_link_libraries(patchestry_frontend
  PUBLIC
    clangFrontend
  PRIVATE
    LLVMSupport
    patchestry_settings
    glog::glog
)
