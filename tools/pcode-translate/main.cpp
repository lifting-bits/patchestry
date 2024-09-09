/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Support/LogicalResult.h>

int main(int argc, char **argv) {
    return mlir::failed(mlir::mlirTranslateMain(argc, argv, "P-Code translation driver\n"));
}
