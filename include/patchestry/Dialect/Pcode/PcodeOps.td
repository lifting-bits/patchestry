/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#ifndef PCODE_DIALECT_OPS
#define PCODE_DIALECT_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"

def Pcode_ConstOp
  : Pcode_Op< "constant", [ConstantLike, NoMemoryEffect] >
  , Results<( outs AnyType:$result )>
{
  let hasFolder = 1;

  let assemblyFormat = [{ attr-dict `:` type($result)}];
}

#endif // PCODE_DIALECT_OPS
