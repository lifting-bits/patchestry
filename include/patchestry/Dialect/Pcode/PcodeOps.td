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

include "patchestry/Dialect/Pcode/PcodeTypes.td"

def FuncOp
  : PcodeOp< "function" >
  , Arguments<( ins StrAttr:$func_name )>
{
  let regions = (region AnyRegion:$blocks);
  let assemblyFormat = [{ attr-dict `:` $blocks }];
}

def BlockOp
  : PcodeOp< "block" >
  , Arguments<( ins StrAttr:$block_label )>
{
  let regions = (region AnyRegion:$instructions);
  let assemblyFormat = [{ attr-dict `:` $instructions }];
}

def InstOp
  : PcodeOp< "instruction" >
  , Arguments<( ins StrAttr:$inst_mnemonic )>
{
  let regions = (region AnyRegion:$semantics);
  let assemblyFormat = [{ attr-dict `:` $semantics }];
}

def ConstOp
  : PcodeOp< "const", [ConstantLike, AllTypesMatch< ["value", "result"] >] >
  , Arguments<( ins TypedAttrInterface:$value )>
  , Results<( outs AnyType:$result )>
{
  let hasFolder = 1;
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

class VarnodeTypeOp< string varnode_mnemonic, list< Trait > traits = [] >
  : PcodeOp< varnode_mnemonic, traits >
  , Arguments<( ins StrAttr:$addr_space, I64Attr:$addr, I8Attr:$size )>
  , Results<( outs AnyType:$result )>
{
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def RegOp : VarnodeTypeOp< "reg" >;
def MemOp : VarnodeTypeOp< "mem" >;
def VarOp : VarnodeTypeOp< "var" >;

class UnaryOp< string op_mnemonic, list< Trait > traits = [] >
  : PcodeOp< op_mnemonic, traits >
  , Arguments<( ins AnySignlessInteger:$op )>
  , Results<( outs AnySignlessInteger:$result )>
{
  let assemblyFormat = [{ $op attr-dict `:` functional-type(operands, results) }];
}

def CopyOp        : UnaryOp < "copy", [SameOperandsAndResultType] >;
def PopcountOp    : UnaryOp < "popcount" >;
def BoolNegateOp  : UnaryOp < "bool_negate", [SameOperandsAndResultType] >;

class BinOp< string op_mnemonic, list< Trait > traits = [] >
  : PcodeOp< op_mnemonic, traits >
  , Arguments<( ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs )>
  , Results<( outs AnySignlessInteger:$result )>
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
  : PcodeOp< "branch" >
  , Arguments<( ins AnySignlessInteger:$addr )>
{
  let summary = "Pcode BRANCH operation";
  let description = "TODO(surovic)";
  let assemblyFormat = [{ $addr attr-dict `:` type(operands) }];
}

def CBranchOp
  : PcodeOp< "cbranch" >
  , Arguments<( ins AnySignlessInteger:$addr, AnySignlessInteger:$cond )>
{
  let summary = "Pcode CBRANCH operation";
  let description = "TODO(surovic)";
  let assemblyFormat = [{ $addr `,` $cond attr-dict `:` type(operands) }];
}

def CallOp
  : PcodeOp< "call" >
  , Arguments<( ins AnySignlessInteger:$addr )>
{
  let summary = "Pcode CALL operation";
  let description = "TODO(surovic)";
  let assemblyFormat = [{ $addr attr-dict `:` type(operands) }];
}

def ReturnOp
  : PcodeOp< "return" >
  , Arguments<( ins AnySignlessInteger:$varnode )>
{
  let summary = "Pcode RETURN operation";
  let description = "TODO(surovic)";
  let assemblyFormat = [{ $varnode attr-dict `:` type(operands) }];
}

def StoreOp
  : PcodeOp< "store" >
  , Arguments<( ins AnySignlessInteger:$addr_space, AnySignlessInteger:$addr, AnySignlessInteger:$data )>
{
  let summary = "Pcode STORE operation";
  let description = "TODO(surovic)";
  let assemblyFormat = [{ $addr_space `,` $addr `,` $data attr-dict `:` type(operands) }];
}

def LoadOp
  : PcodeOp< "load" >
  , Arguments<( ins AnySignlessInteger:$addr_space, AnySignlessInteger:$addr )>
  , Results<( outs AnySignlessInteger:$result )>
{
  let summary = "Pcode LOAD operation";
  let description = "TODO(surovic)";
  let assemblyFormat = [{ $addr_space `,` $addr attr-dict `:` functional-type(operands, results) }];
}

#endif // PCODE_DIALECT_OPS
