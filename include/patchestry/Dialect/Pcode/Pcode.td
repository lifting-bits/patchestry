/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#ifndef PCODE_DIALECT
#define PCODE_DIALECT

include "mlir/IR/OpBase.td"

def Pcode_Dialect : Dialect {
  let name = "pc";
  let cppNamespace = "::patchestry::pc";

  let summary = "Dialect for representing ghidra pcode operations.";
}

class PcodeOp< string mnemonic, list< Trait > traits = [] >
  : Op< Pcode_Dialect, mnemonic, traits >;

include "PcodeOps.td"
include "PcodeTypes.td"

#endif // PCODE_DIALECT
