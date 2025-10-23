# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

set(LLVM_LINK_COMPONENTS
  Support
  clangFrontend
)


add_executable(patchir-yaml-parser
    main.cpp
)

llvm_update_compile_flags(patchir-yaml-parser)
target_link_libraries(patchir-yaml-parser
    PRIVATE
        patchestry::yaml
        patchestry::codegen
        clangFrontend
        patchestry::passes
        MLIROptLib
)

mlir_check_link_libraries(patchir-yaml-parser)

if(PATCHESTRY_INSTALL)
  install(TARGETS patchir-yaml-parser
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT patchestry-tools
  )
endif()
