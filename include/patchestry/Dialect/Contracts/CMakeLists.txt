# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.

set(LLVM_TARGET_DEFINITIONS Contract.td)
mlir_tablegen(contracts/ContractsDialect.h.inc -gen-dialect-decls -dialect=contracts)
mlir_tablegen(contracts/ContractsDialect.cpp.inc -gen-dialect-defs -dialect=contracts)
mlir_tablegen(contracts/ContractsEnums.h.inc -gen-enum-decls)
mlir_tablegen(contracts/ContractsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(contracts/ContractsAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(contracts/ContractsAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRContractsIncGen)

if (DEFINED MLIR_BINARY_DIR)
    set(MLIR_OLD_ROOT ${MLIR_BINARY_DIR})
endif()

set(MLIR_BINARY_DIR ${PATCHESTRY_BINARY_DIR})
add_mlir_doc(Contract Contracts Dialects/ -gen-dialect-doc -dialect=contracts)

if (DEFINED MLIR_OLD_ROOT)
    set(MLIR_BINARY_DIR ${MLIR_OLD_ROOT})
endif()