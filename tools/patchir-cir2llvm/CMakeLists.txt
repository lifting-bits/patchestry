# Copyright (c) 2024, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

add_executable(patchir-cir2llvm
    main.cpp
)

target_link_libraries(patchir-cir2llvm
    PRIVATE
        clangCIRLoweringDirectToLLVM
        MLIRGPUToLLVMIRTranslation
        MLIRAMXToLLVMIRTranslation
        MLIRSPIRVToLLVMIRTranslation
        MLIRArmNeonToLLVMIRTranslation
        MLIRArmSVEToLLVMIRTranslation
        MLIRArmSMEToLLVMIRTranslation
        MLIRVCIXToLLVMIRTranslation
        MLIRX86VectorToLLVMIRTranslation
        MLIROpenACCToLLVMIRTranslation
        MLIRContracts
)

if(PATCHESTRY_INSTALL)
  install(TARGETS patchir-cir2llvm
    COMPONENT patchestry-tools
  )
endif()