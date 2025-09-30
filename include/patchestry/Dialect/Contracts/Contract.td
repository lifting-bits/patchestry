/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#ifndef CONTRACT_DIALECT
#define CONTRACT_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

// Dialect
def Contract_Dialect : Dialect {
    let name = "contracts";
    let cppNamespace = "contracts";
    let summary = "Static contracts as attributes";
    let description = [{ Declarative pre/post conditions attached to operations. }];
}

// cmp predicate
def CmpKind : I32EnumAttr<"CmpKind", "comparison predicate", [
  I32EnumAttrCase<"eq", 0>,
  I32EnumAttrCase<"ne", 1>,
  I32EnumAttrCase<"slt", 2>,
  I32EnumAttrCase<"sle", 3>,
  I32EnumAttrCase<"sgt", 4>,
  I32EnumAttrCase<"sge", 5>,
  I32EnumAttrCase<"ult", 6>,
  I32EnumAttrCase<"ule", 7>,
  I32EnumAttrCase<"ugt", 8>,
  I32EnumAttrCase<"uge", 9>
]> { let cppNamespace = "::contracts"; }

// Positional operand reference: $0, $1, ...
def VarRefAttr : AttrDef<Contract_Dialect, "varref"> {
  let summary = "Positional operand reference";
  let parameters = (ins "uint64_t":$index);
  let mnemonic = "varref";
  let assemblyFormat = "`<` $index `>`";
}

// Integer literal
def ConstIntAttr : AttrDef<Contract_Dialect, "iconst"> {
  let summary = "Integer constant";
  let parameters = (ins "mlir::IntegerAttr":$value);
  let mnemonic = "iconst";
  let assemblyFormat = "`<` $value `>`";
}

// cmp clause: (lhs pred rhs) where rhs is var or const
def CmpClauseAttr : AttrDef<Contract_Dialect, "cmp"> {
  let summary = "Integer comparison clause";
  let parameters = (ins
    "::contracts::CmpKind":$pred,
    "::contracts::VarRefAttr":$lhs,
    OptionalParameter<"::contracts::VarRefAttr">:$rhsVar,
    OptionalParameter<"::contracts::ConstIntAttr">:$rhsConst
  );
  let mnemonic = "cmp";
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $pred `,` $lhs (`,` $rhsVar^)? (`,` $rhsConst^)? `>`";
}

// Boolean aggregators
def AllOfAttr : AttrDef<Contract_Dialect, "all_of"> {
  let summary = "Conjunction of clauses";
  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$clauses);
  let mnemonic = "all_of";
  let assemblyFormat = "`<` `[` $clauses `]` `>`";
}
def AnyOfAttr : AttrDef<Contract_Dialect, "any_of"> {
  let summary = "Disjunction of clauses";
  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$clauses);
  let mnemonic = "any_of";
  let assemblyFormat = "`<` `[` $clauses `]` `>`";
}

// Top-level contract you attach to an op
def StaticContractAttr : AttrDef<Contract_Dialect, "static"> {
  let summary = "Static contract container";
  let parameters = (ins
    "mlir::StringAttr":$message,
    "mlir::Attribute":$expr   // CmpClauseAttr | AllOfAttr | AnyOfAttr
  );
  let mnemonic = "static";
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` `message` `=` $message `,` $expr `>`";
}

#endif // CONTRACT_DIALECT