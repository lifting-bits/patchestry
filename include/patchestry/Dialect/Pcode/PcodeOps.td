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

class BinOp< string mnemonic, list< Trait > traits = [] >
  : Pcode_Op< mnemonic, traits >
  , Arguments<( ins Builtin_Integer:$lhs, Builtin_Integer:$rhs )>
  , Results<( outs Builtin_Integer:$result )>
{
  let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];
}

def IntAddOp      : BinOp< "int_add" >;
def IntSubOp      : BinOp< "int_sub" >;
def IntLessOp     : BinOp< "int_less" >;
def IntEqualOp    : BinOp< "int_equal" >;
def IntSBorrowOp  : BinOp< "int_sborrow" >;
def IntSLessOp    : BinOp< "int_sless" >;
def IntAndOp      : BinOp< "int_and" >;

def BranchOp
  : Pcode_Op< "branch" >
  , Arguments<( ins Builtin_Integer:$addr )>
{
  let summary = "Pcode BRANCH operation";
  let description = "TODO(surovic)";
  let assemblyFormat = [{ $addr attr-dict `:` type(operands) }];
}

def CBranchOp
  : Pcode_Op< "cbranch" >
  , Arguments<( ins Builtin_Integer:$addr, Builtin_Integer:$cond )>
{
  let summary = "Pcode CBRANCH operation";
  let description = "TODO(surovic)";
  let assemblyFormat = [{ $addr `,` $cond attr-dict `:` type(operands) }];
}

def CallOp
  : Pcode_Op< "call" >
  , Arguments<( ins Builtin_Integer:$addr )>
{
  let summary = "Pcode CALL operation";
  let description = "TODO(surovic)";
  let assemblyFormat = [{ $addr attr-dict `:` type(operands) }];
}

def ReturnOp
  : Pcode_Op< "return" >
  , Arguments<( ins Builtin_Integer:$varnode )>
{
  let summary = "Pcode RETURN operation";
  let description = "TODO(surovic)";
  let assemblyFormat = [{ $varnode attr-dict `:` type(operands) }];
}

def StoreOp
  : Pcode_Op< "store" >
  , Arguments<( ins Builtin_Integer:$addr_space, Builtin_Integer:$addr, Builtin_Integer:$data )>
{
  let summary = "Pcode STORE operation";
  let description = "TODO(surovic)";
  let assemblyFormat = [{ $addr_space `,` $addr `,` $data attr-dict `:` type(operands) }];
}

def LoadOp
  : Pcode_Op< "load" >
  , Arguments<( ins Builtin_Integer:$addr_space, Builtin_Integer:$addr )>
  , Results<( outs Builtin_Integer:$result )>
{
  let summary = "Pcode LOAD operation";
  let description = "TODO(surovic)";
  let assemblyFormat = [{ $addr_space `,` $addr attr-dict `:` functional-type(operands, results) }];
}

#endif // PCODE_DIALECT_OPS
