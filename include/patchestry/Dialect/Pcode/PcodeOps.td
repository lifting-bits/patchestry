/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#ifndef PCODE_DIALECT_OPS
#define PCODE_DIALECT_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"

def FuncOp
  : Pcode_Op< "function" >
  , Arguments<( ins StrAttr:$name )>
{
  let regions = (region AnyRegion:$blocks);
  let assemblyFormat = [{ attr-dict `:` $blocks }];
}

def BlockOp
  : Pcode_Op< "block" >
  , Arguments<( ins StrAttr:$label )>
{
  let regions = (region AnyRegion:$instructions);
  let assemblyFormat = [{ attr-dict `:` $instructions }];
}

def InstOp
  : Pcode_Op< "instruction" >
  , Arguments<( ins StrAttr:$mnemonic )>
{
  let regions = (region AnyRegion:$semantics);
  let assemblyFormat = [{ attr-dict `:` $semantics }];
}

def VarnodeOp
  : Pcode_Op< "varnode" >
  , Arguments<( ins StrAttr:$addr_space, I64Attr:$addr,  I8Attr:$size)>
  , Results<( outs Builtin_Integer:$result )>
{
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

class UnaryOp< string mnemonic, list< Trait > traits = [] >
  : Pcode_Op< mnemonic, traits >
  , Arguments<( ins Builtin_Integer:$op )>
  , Results<( outs Builtin_Integer:$result )>
{
  let assemblyFormat = [{ $op attr-dict `:` functional-type(operands, results) }];
}

def CopyOp        : UnaryOp < "copy", [SameOperandsAndResultType] >;
def PopcountOp    : UnaryOp < "popcount" >;
def BoolNegateOp  : UnaryOp < "bool_negate", [SameOperandsAndResultType] >;

#endif // PCODE_DIALECT_OPS
