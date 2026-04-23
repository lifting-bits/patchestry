# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

add_executable(patchir-klee-verifier
    main.cpp
    PredicateParser.cpp
    Runtime.cpp
    GlobalsInit.cpp
    HarnessGen.cpp
)

target_link_libraries(patchir-klee-verifier
    PRIVATE
        LLVMCore
        LLVMSupport
        LLVMIRReader
        LLVMBitWriter
        LLVMLinker
        patchestry_settings
)

if(PATCHESTRY_INSTALL)
  install(TARGETS patchir-klee-verifier
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT patchestry-tools
  )
endif()
