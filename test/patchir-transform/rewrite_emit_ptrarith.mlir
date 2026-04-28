// RUN: %patchir-transform %s --spec %S/rewrite_emit_subscript.yaml -o %t1.cir
// RUN: %file-check -vv -check-prefix=SUBSCRIPT %s --input-file %t1.cir
// SUBSCRIPT: cir.func @rewrite_subscript_host
// SUBSCRIPT-NOT: cir.call @sink_subscript
// SUBSCRIPT-DAG: cir.ptr_stride
// SUBSCRIPT-DAG: cir.load

// RUN: %patchir-transform %s --spec %S/rewrite_emit_deref.yaml -o %t2.cir
// RUN: %file-check -vv -check-prefix=DEREF %s --input-file %t2.cir
// DEREF: cir.func @rewrite_deref_host
// DEREF-NOT: cir.call @sink_deref
// DEREF-DAG: cir.load deref

// RUN: %patchir-transform %s --spec %S/rewrite_emit_addrof.yaml -o %t3.cir
// RUN: %file-check -vv -check-prefix=ADDROF %s --input-file %t3.cir
// ADDROF: cir.func @rewrite_addrof_host
// ADDROF-NOT: cir.call @sink_addrof
// ADDROF-DAG: cir.alloca !u32i
// ADDROF-DAG: cir.store

!u32i = !cir.int<u, 32>

module {
  cir.func private @sink_subscript(!cir.ptr<!u32i>, !u32i) -> !u32i
  cir.func @rewrite_subscript_host(%arg0: !cir.ptr<!u32i>, %arg1: !u32i) -> !u32i {
    %call = cir.call @sink_subscript(%arg0, %arg1) : (!cir.ptr<!u32i>, !u32i) -> !u32i
    cir.return %call : !u32i
  }

  cir.func private @sink_deref(!cir.ptr<!u32i>) -> !u32i
  cir.func @rewrite_deref_host(%arg0: !cir.ptr<!u32i>) -> !u32i {
    %call = cir.call @sink_deref(%arg0) : (!cir.ptr<!u32i>) -> !u32i
    cir.return %call : !u32i
  }

  cir.func private @sink_addrof(!u32i) -> !cir.ptr<!u32i>
  cir.func @rewrite_addrof_host(%arg0: !u32i) -> !cir.ptr<!u32i> {
    %call = cir.call @sink_addrof(%arg0) : (!u32i) -> !cir.ptr<!u32i>
    cir.return %call : !cir.ptr<!u32i>
  }
}
