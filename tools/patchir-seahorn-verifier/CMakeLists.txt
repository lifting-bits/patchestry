# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

add_executable(patchir-seahorn-verifier
    main.cpp
)

target_link_libraries(patchir-seahorn-verifier
    PRIVATE
        LLVMCore
        LLVMSupport
        LLVMIRReader
        LLVMBitWriter
        MLIRContracts
)

if(PATCHESTRY_INSTALL)
  install(TARGETS patchir-seahorn-verifier
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT patchestry-tools
  )
endif()
