// RUN: %patchir-transform %s --spec %S/rewrite_stmt_free_and_null.yaml -o %t1.cir
// RUN: %file-check -vv -check-prefix=FREE_NULL %s --input-file %t1.cir
// FREE_NULL: cir.func @stmt_free_host(%[[ARG:[a-z0-9]+]]: !cir.ptr<!u32i>)
// FREE_NULL-NOT: ["__cap_P"
// FREE_NULL: %[[P:[0-9]+]] = cir.alloca !cir.ptr<!u32i>, !cir.ptr<!cir.ptr<!u32i>>, ["p", init]
// FREE_NULL: cir.call @free
// FREE_NULL: cir.const #cir.ptr<null>
// FREE_NULL: cir.store {{.*}} %[[P]] : !cir.ptr<!u32i>, !cir.ptr<!cir.ptr<!u32i>>

// RUN: %patchir-transform %s --spec %S/rewrite_stmt_if_guarded.yaml -o %t2.cir
// RUN: %file-check -vv -check-prefix=IF_GUARD %s --input-file %t2.cir
// IF_GUARD: cir.func @stmt_if_host
// IF_GUARD: cir.if
// IF_GUARD: cir.call @free

// RUN: %patchir-transform %s --spec %S/rewrite_stmt_compound_local.yaml -o %t3.cir
// RUN: %file-check -vv -check-prefix=COMPOUND %s --input-file %t3.cir
// COMPOUND: cir.func @stmt_compound_host
// COMPOUND: cir.alloca !u32i, !cir.ptr<!u32i>, ["tmp", init]
// COMPOUND: cir.binop(add
// COMPOUND: cir.call @log_value

// RUN: %patchir-transform %s --spec %S/rewrite_stmt_while.yaml -o %t4.cir
// RUN: %file-check -vv -check-prefix=WHILE %s --input-file %t4.cir
// WHILE: cir.func @stmt_while_host
// WHILE: cir.alloca !u32i, !cir.ptr<!u32i>, ["i", init]
// WHILE: cir.scope
// WHILE: cir.while
// WHILE: cir.call @tick
// WHILE: cir.binop(add

!u32i = !cir.int<u, 32>

module {
  cir.func private @free(!cir.ptr<!u32i>)
  cir.func private @log_value(!u32i)
  cir.func private @tick()
  cir.func private @start_loop(!u32i)

  cir.func @stmt_free_host(%arg0: !cir.ptr<!u32i>) {
    %p = cir.alloca !cir.ptr<!u32i>, !cir.ptr<!cir.ptr<!u32i>>, ["p", init] {alignment = 4 : i64}
    cir.store %arg0, %p : !cir.ptr<!u32i>, !cir.ptr<!cir.ptr<!u32i>>
    %p_val = cir.load %p : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i>
    cir.call @free(%p_val) : (!cir.ptr<!u32i>) -> ()
    cir.return
  }

  cir.func @stmt_if_host(%arg0: !cir.ptr<!u32i>) {
    cir.call @free(%arg0) : (!cir.ptr<!u32i>) -> ()
    cir.return
  }

  cir.func @stmt_compound_host(%arg0: !u32i) {
    cir.call @log_value(%arg0) : (!u32i) -> ()
    cir.return
  }

  cir.func @stmt_while_host(%arg0: !u32i) {
    cir.call @start_loop(%arg0) : (!u32i) -> ()
    cir.return
  }
}
