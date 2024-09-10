# Copyright (c) 2024, Trail of Bits, Inc. All rights reserved. This source code
# is licensed in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

set(LLVM_LINK_COMPONENTS
  Support
)


add_executable(pcode-translate
    main.cpp
)

llvm_update_compile_flags(pcode-translate)
target_link_libraries(pcode-translate
    PRIVATE
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTranslateLib
        MLIRSupport
        patchestry::ghidra
)

mlir_check_link_libraries(pcode-translate)
