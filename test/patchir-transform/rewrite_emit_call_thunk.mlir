// RUN: %patchir-transform %s --spec %S/rewrite_emit_call_thunk.yaml -o %t.cir
// RUN: %file-check -vv -check-prefix=THUNK %s --input-file %t.cir
// THUNK: cir.func @rewrite_call_host
// THUNK-NOT: cir.call @foo
// THUNK-DAG: cir.call @foo_v2

!u32i = !cir.int<u, 32>
module {
  cir.func private @foo(!u32i, !u32i) -> !u32i
  cir.func private @foo_v2(!u32i, !u32i, !u32i) -> !u32i

  cir.func @rewrite_call_host(%arg0: !u32i, %arg1: !u32i) -> !u32i {
    %0 = cir.call @foo(%arg0, %arg1) : (!u32i, !u32i) -> !u32i
    cir.return %0 : !u32i
  }
}
