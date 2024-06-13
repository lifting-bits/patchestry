/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#ifndef PCODE_DIALECT_TYPES
#define PCODE_DIALECT_TYPES

include "mlir/IR/OpBase.td"

class PcodeType< string name, string _mnemonic, list< Trait > traits = [] >
  : TypeDef< Pcode_Dialect, name, traits >
{
  let mnemonic = _mnemonic;
}

def RegType : PcodeType< "Reg", "reg" >;
def MemType : PcodeType< "Mem", "mem" >;
def VarType : PcodeType< "Var", "var" >;

#endif // PCODE_DIALECT_TYPES
