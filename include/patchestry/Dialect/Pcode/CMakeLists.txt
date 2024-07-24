# Copyright (c) 2024, Trail of Bits, Inc. All rights reserved. This source code
# is licensed in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

add_mlir_dialect(Pcode pc)

if (DEFINED MLIR_BINARY_DIR)
    set(MLIR_OLD_ROOT ${MLIR_BINARY_DIR})
endif()

set(MLIR_BINARY_DIR ${PATCHESTRY_BINARY_DIR})
add_mlir_doc(Pcode Pcode Dialects/ -gen-dialect-doc -dialect=pc)

if (DEFINED MLIR_OLD_ROOT)
    set(MLIR_BINARY_DIR ${MLIR_OLD_ROOT})
endif()