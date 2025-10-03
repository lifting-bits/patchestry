/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#ifndef CONTRACT_DIALECT
#define CONTRACT_DIALECT

//===- ContractDialect.td -------------------------------------*- tablegen -*-===//
//
// Contracts dialect: declarative pre/post conditions as attributes.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Contract_Dialect : Dialect {
  let name = "contracts";
  let cppNamespace = "contracts";
  let summary = "Static contracts as attributes";
  let description = [{
    Declarative pre/post conditions attached to operations.
  }];
}

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

// Target type for predicate
def TargetKind : I32EnumAttr<"TargetKind", "predicate target", [
  I32EnumAttrCase<"Arg",          0>,
  I32EnumAttrCase<"ReturnValue",  1>,
  I32EnumAttrCase<"Symbol",       2>
]> { let cppNamespace = "::contracts"; }

def PredicateKind : I32EnumAttr<"PredicateKind", "predicate kind", [
  I32EnumAttrCase<"nonnull",  0>,
  I32EnumAttrCase<"relation", 1>,
  I32EnumAttrCase<"alignment",2>,
  I32EnumAttrCase<"expr",     3>,
  I32EnumAttrCase<"range",    4>
]> { let cppNamespace = "::contracts"; }

def RelationKind : I32EnumAttr<"RelationKind", "relation predicate", [
  I32EnumAttrCase<"none", 0>,
  I32EnumAttrCase<"lt",   1>,
  I32EnumAttrCase<"lte",  2>,
  I32EnumAttrCase<"gt",   3>,
  I32EnumAttrCase<"gte",  4>,
  I32EnumAttrCase<"eq",   5>,
  I32EnumAttrCase<"neq",  6>
]> { let cppNamespace = "::contracts"; }

//===----------------------------------------------------------------------===//
// Core attrs
//===----------------------------------------------------------------------===//

// Variant-like target:
//   #contracts.target<arg3>
//   #contracts.target<return_value>
//   #contracts.target<symbol @usb_device>
def Contract_TargetAttr : AttrDef<Contract_Dialect, "target"> {
  let summary = "Target selector (arg<N> | return_value | symbol @name)";
  let parameters = (ins
    "::contracts::TargetKind":$kind,
    OptionalParameter<"uint64_t">:$index,                          // only when Arg
    OptionalParameter<"mlir::FlatSymbolRefAttr">:$symbol          // only when Symbol
  );
  let cppClassName = "TargetAttr";
  let mnemonic = "target";
  let hasCustomAssemblyFormat = 1; // <-- add this
  // NOTE: implement custom parse()/print() in C++ to support the compact forms above.
}

// Symbolic variable reference (string-based; useful as a value payload)
// Kept distinct from FlatSymbolRefAttr when you want a free-form name.
def Contract_SymbolNameAttr : AttrDef<Contract_Dialect, "symname"> {
  let summary = "Symbolic variable name";
  let parameters = (ins "mlir::StringAttr":$name);
  let mnemonic = "symname";
  let cppClassName = "SymbolNameAttr";
  let assemblyFormat = "`<` $name `>`";
}

// Range constraint: min <= value <= max
def Contract_RangeAttr : AttrDef<Contract_Dialect, "range"> {
  let summary = "Value range constraint";
  let parameters = (ins
    OptionalParameter<"mlir::IntegerAttr">:$min,
    OptionalParameter<"mlir::IntegerAttr">:$max
  );
  let mnemonic = "range";
  let cppClassName = "ContractRangeAttr";
  let assemblyFormat = "`<` (`min` `=` $min^)? (`,` `max` `=` $max^)? `>`";
}

// Alignment constraint
def Contract_AlignmentAttr : AttrDef<Contract_Dialect, "align"> {
  let summary = "Alignment requirement (bytes)";
  let parameters = (ins "uint64_t":$alignment);
  let mnemonic = "align";
  let cppClassName = "ContractAlignmentAttr";
  let assemblyFormat = "`<` $alignment `>`";
}

// Boolean aggregators
def Contract_AllOfAttr : AttrDef<Contract_Dialect, "all_of"> {
  let summary = "Conjunction of clauses";
  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$clauses);
  let mnemonic = "all_of";
  let cppClassName = "ContractAllOfAttr";
  let assemblyFormat = "`<` `[` $clauses `]` `>`";
}

def Contract_AnyOfAttr : AttrDef<Contract_Dialect, "any_of"> {
  let summary = "Disjunction of clauses";
  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$clauses);
  let mnemonic = "any_of";
  let cppClassName = "ContractAnyOfAttr";
  let assemblyFormat = "`<` `[` $clauses `]` `>`";
}

//===----------------------------------------------------------------------===//
// Predicate & containers
//===----------------------------------------------------------------------===//

def Contract_PredicateAttr : AttrDef<Contract_Dialect, "predicate"> {
  let summary = "Predicate for precondition/postcondition";
  let parameters = (ins
    "::contracts::PredicateKind":$kind,
    OptionalParameter<"contracts::TargetAttr">:$target,
    OptionalParameter<"::contracts::RelationKind">:$relation,
    // 'value' may be IntegerAttr (constant), FlatSymbolRefAttr (named const),
    // or Contract_SymbolNameAttr (free-form symbol name), etc.
    OptionalParameter<"mlir::Attribute">:$value,
    OptionalParameter<"contracts::ContractAlignmentAttr">:$align,
    OptionalParameter<"mlir::StringAttr">:$expr,
    OptionalParameter<"contracts::ContractRangeAttr">:$range
  );
  let mnemonic = "pred";
  let cppClassName = "PredicateAttr";
  let assemblyFormat = [{
    `<` $kind
      ( `,` `target` `=` $target^ )?
      ( `,` `relation` `=` $relation^ )?
      ( `,` `value` `=` $value^ )?
      ( `,` `align` `=` $align^ )?
      ( `,` `expr` `=` $expr^ )?
      ( `,` `range` `=` $range^ )?
    `>`
  }];
}

// Precondition with id and predicate
def Contract_PreconditionAttr : AttrDef<Contract_Dialect, "precondition"> {
  let summary = "Precondition requirement";
  let parameters = (ins
    "mlir::StringAttr":$id,
    "contracts::PredicateAttr":$pred
  );
  let mnemonic = "pre";
  let cppClassName = "PreconditionAttr";
  let assemblyFormat = "`<` `id` `=` $id `,` $pred `>`";
}

// Postcondition with id and predicate
def Contract_PostconditionAttr : AttrDef<Contract_Dialect, "postcondition"> {
  let summary = "Postcondition requirement";
  let parameters = (ins
    "mlir::StringAttr":$id,
    "contracts::PredicateAttr":$pred
  );
  let mnemonic = "post";
  let cppClassName = "PostconditionAttr";
  let assemblyFormat = "`<` `id` `=` $id `,` $pred `>`";
}

// Top-level contract attached to an op
def Contract_StaticContractAttr : AttrDef<Contract_Dialect, "static"> {
  let summary = "Static contract container";
  let parameters = (ins
    ArrayRefParameter<"mlir::Attribute">:$preconditions,   // array of Contract_PreconditionAttr
    ArrayRefParameter<"mlir::Attribute">:$postconditions   // array of Contract_PostconditionAttr
  );
  let mnemonic = "static";
  let cppClassName = "StaticContractAttr";
  let assemblyFormat = "`<` `pre` `=` `[` $preconditions `]` `,` "
                       "`post` `=` `[` $postconditions `]` `>`";
}

#endif // CONTRACT_DIALECT