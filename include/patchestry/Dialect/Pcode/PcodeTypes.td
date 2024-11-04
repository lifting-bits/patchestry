/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#ifndef PCODE_DIALECT_TYPES
#define PCODE_DIALECT_TYPES

include "mlir/IR/OpBase.td"

class Pcode_Type< string type_name, string _mnemonic, list< Trait > traits = [] >
  : TypeDef< Pcode_Dialect, type_name, traits >
{
  let mnemonic = _mnemonic;
}

def Pcode_RegType : Pcode_Type< "Reg", "reg" >;
def Pcode_MemType : Pcode_Type< "Mem", "mem" >;
def Pcode_VarType : Pcode_Type< "Var", "var" >;

#endif // PCODE_DIALECT_TYPES
