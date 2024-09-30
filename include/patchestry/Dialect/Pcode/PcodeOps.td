/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#ifndef PCODE_DIALECT_OPS
#define PCODE_DIALECT_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"

include "patchestry/Dialect/Pcode/PcodeTypes.td"

def Pcode_FuncOp
  : Pcode_Op< "func", [
      NoTerminator, SymbolTable, Symbol, IsolatedFromAbove
    ] >
  , Arguments<( ins SymbolNameAttr:$sym_name )>
{
  let regions = (region SizedRegion<1>:$blocks);
  let assemblyFormat = [{ $sym_name attr-dict `:` $blocks }];
}

def Pcode_BlockOp
  : Pcode_Op< "block", [NoTerminator] >
  , Arguments<( ins SymbolNameAttr:$sym_name )>
{
  let regions = (region SizedRegion<1>:$instructions);
  let assemblyFormat = [{ $sym_name attr-dict `:` $instructions }];
}

def Pcode_InstOp
  : Pcode_Op< "instruction", [NoTerminator] >
  , Arguments<( ins StrAttr:$inst_mnemonic )>
{
  let regions = (region SizedRegion<1>:$semantics);
  let assemblyFormat = [{ attr-dict `:` $semantics }];
}

def Pcode_ConstOp
  : Pcode_Op< "const", [
    ConstantLike, AllTypesMatch< ["value", "result"] >
  ] >
  , Arguments<( ins TypedAttrInterface:$value )>
  , Results<( outs AnyType:$result )>
{
  let hasFolder = 1;
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

class Pcode_VarnodeTypeOp< string mnemonic, list< Trait > traits = [] >
  : Pcode_Op< mnemonic, traits >
  , Arguments<( ins StrAttr:$addr_space, I64Attr:$addr, I8Attr:$size )>
  , Results<( outs AnyType:$result )>
{
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def Pcode_RegOp : Pcode_VarnodeTypeOp< "reg" >;
def Pcode_MemOp : Pcode_VarnodeTypeOp< "mem" >;
def Pcode_VarOp : Pcode_VarnodeTypeOp< "var" >;

class Pcode_UnaryOp< string mnemonic, list< Trait > traits = [] >
  : Pcode_Op< mnemonic, traits >
  , Arguments<( ins AnySignlessInteger:$op )>
  , Results<( outs AnySignlessInteger:$result )>
{
  let assemblyFormat = [{
    $op attr-dict `:` functional-type(operands, results)
  }];
}

def Pcode_CopyOp        : Pcode_UnaryOp < "copy", [SameOperandsAndResultType] >;
def Pcode_PopcountOp    : Pcode_UnaryOp < "popcount" >;
def Pcode_BoolNegateOp  : Pcode_UnaryOp < "bool_negate", [SameOperandsAndResultType] >;

class Pcode_BinOp< string mnemonic, list< Trait > traits = [] >
  : Pcode_Op< mnemonic, traits >
  , Arguments<( ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs )>
  , Results<( outs AnySignlessInteger:$result )>
{
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def Pcode_IntAddOp      : Pcode_BinOp< "int_add" >;
def Pcode_IntSubOp      : Pcode_BinOp< "int_sub" >;
def Pcode_IntLessOp     : Pcode_BinOp< "int_less" >;
def Pcode_IntEqualOp    : Pcode_BinOp< "int_equal" >;
def Pcode_IntSBorrowOp  : Pcode_BinOp< "int_sborrow" >;
def Pcode_IntSLessOp    : Pcode_BinOp< "int_sless" >;
def Pcode_IntAndOp      : Pcode_BinOp< "int_and" >;

def Pcode_BranchOp
  : Pcode_Op< "branch" >
  , Arguments<( ins AnySignlessInteger:$addr )>
{
  let summary = "Pcode BRANCH operation";
  let description = "TBD";
  let assemblyFormat = [{ $addr attr-dict `:` type(operands) }];
}

def Pcode_CBranchOp
  : Pcode_Op< "cbranch" >
  , Arguments<( ins AnySignlessInteger:$addr, AnySignlessInteger:$cond )>
{
  let summary = "Pcode CBRANCH operation";
  let description = "TBD";
  let assemblyFormat = [{ $addr `,` $cond attr-dict `:` type(operands) }];
}

def Pcode_CallOp
  : Pcode_Op< "call" >
  , Arguments<( ins AnySignlessInteger:$addr )>
{
  let summary = "Pcode CALL operation";
  let description = "TBD";
  let assemblyFormat = [{ $addr attr-dict `:` type(operands) }];
}

def Pcode_ReturnOp
  : Pcode_Op< "return" >
  , Arguments<( ins AnySignlessInteger:$varnode )>
{
  let summary = "Pcode RETURN operation";
  let description = "TBD";
  let assemblyFormat = [{ $varnode attr-dict `:` type(operands) }];
}

def Pcode_StoreOp
  : Pcode_Op< "store" >
  , Arguments<( ins
    AnySignlessInteger:$addr_space,
    AnySignlessInteger:$addr,
    AnySignlessInteger:$data
  )>
{
  let summary = "Pcode STORE operation";
  let description = "TBD";
  let assemblyFormat = [{
    $addr_space `,` $addr `,` $data attr-dict `:` type(operands)
  }];
}

def Pcode_LoadOp
  : Pcode_Op< "load" >
  , Arguments<( ins AnySignlessInteger:$addr_space, AnySignlessInteger:$addr )>
  , Results<( outs AnySignlessInteger:$result )>
{
  let summary = "Pcode LOAD operation";
  let description = "TBD";
  let assemblyFormat = [{
    $addr_space `,` $addr attr-dict `:` functional-type(operands, results)
  }];
}

#endif // PCODE_DIALECT_OPS
