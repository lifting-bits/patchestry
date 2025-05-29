# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.

add_library(patchestry_passes STATIC
    InstrumentationPass.cpp
    Compiler.cpp
    NameResolver.cpp
    OperationMatcher.cpp
)

set(CLANG_IR_LIBS
  clangCIR
  clangCIRLoweringDirectToLLVM
  clangCIRLoweringThroughMLIR
)

target_link_libraries(patchestry_passes
    PRIVATE
        patchestry_settings
        MLIRSupport
        MLIRIR
        MLIRPass
        MLIRParser
        MLIRTransforms
        MLIRGPUToLLVMIRTranslation
        MLIRAMXToLLVMIRTranslation
        MLIRSPIRVToLLVMIRTranslation
        MLIRArmNeonToLLVMIRTranslation
        MLIRArmSVEToLLVMIRTranslation
        MLIRArmSMEToLLVMIRTranslation
        MLIRVCIXToLLVMIRTranslation
        MLIRX86VectorToLLVMIRTranslation
        MLIROpenACCToLLVMIRTranslation
        MLIRExecutionEngine
        ${CLANG_IR_LIBS}
)

add_library(patchestry::passes ALIAS patchestry_passes)
