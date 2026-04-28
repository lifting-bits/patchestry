// RUN: %patchir-transform %s --spec %S/rewrite_emit_bump_operation.yaml -o %t1.cir
// RUN: %file-check -vv -check-prefix=BUMP %s --input-file %t1.cir
// BUMP: cir.func @rewrite_host
// BUMP-DAG: cir.binop(mul, %arg0, %arg1) : !u32i
// BUMP-DAG: cir.const #cir.int<1>
// BUMP-DAG: cir.binop(add

// RUN: %patchir-transform %s --spec %S/rewrite_emit_mask_operation.yaml -o %t2.cir
// RUN: %file-check -vv -check-prefix=MASK %s --input-file %t2.cir
// MASK: cir.func @rewrite_host
// MASK-DAG: cir.binop(mul, %arg0, %arg1) : !u32i
// MASK-DAG: cir.const #cir.int<255>
// MASK-DAG: cir.binop(and

// RUN: %patchir-transform %s --spec %S/rewrite_emit_byte_swap_operation.yaml -o %t3.cir
// RUN: %file-check -vv -check-prefix=BSWAP %s --input-file %t3.cir
// BSWAP: cir.func @rewrite_host
// BSWAP-DAG: cir.binop(mul, %arg0, %arg1) : !u32i
// BSWAP-DAG: cir.shift(left
// BSWAP-DAG: cir.const #cir.int<16711680>
// BSWAP-DAG: cir.binop(and
// BSWAP-DAG: cir.binop(or
// BSWAP-DAG: cir.shift(right

// RUN: %patchir-transform %s --spec %S/rewrite_emit_ternary_operation.yaml -o %t4.cir
// RUN: %file-check -vv -check-prefix=TERNARY %s --input-file %t4.cir
// TERNARY: cir.func @rewrite_host
// TERNARY-DAG: cir.binop(mul, %arg0, %arg1) : !u32i
// TERNARY-DAG: cir.cmp(gt
// TERNARY-DAG: cir.ternary

!u32i = !cir.int<u, 32>
module {
  cir.func @rewrite_host(%arg0: !u32i, %arg1: !u32i) -> !u32i {
    %0 = cir.binop(mul, %arg0, %arg1) : !u32i
    cir.return %0 : !u32i
  }
  cir.func private @void_helper()
}