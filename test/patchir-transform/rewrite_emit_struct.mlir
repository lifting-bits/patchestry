// RUN: %patchir-transform %s --spec %S/rewrite_emit_struct_arrow.yaml -o %t1.cir
// RUN: %file-check -vv -check-prefix=STRUCT_ARROW %s --input-file %t1.cir
// STRUCT_ARROW: cir.func @rewrite_struct_arrow_host(%[[ARG:[a-z0-9]+]]: !cir.ptr<!ty_Point>)
// STRUCT_ARROW-NOT: cir.call @accessor_ptr
// STRUCT_ARROW: cir.alloca !cir.ptr<!ty_Point>, !cir.ptr<!cir.ptr<!ty_Point>>, ["arg_ref"]
// STRUCT_ARROW: cir.get_member %{{[0-9]+}}[0] {name = "x"} : !cir.ptr<!ty_Point> -> !cir.ptr<!u32i>
// STRUCT_ARROW: cir.load %{{[0-9]+}} : !cir.ptr<!u32i>, !u32i

// RUN: %patchir-transform %s --spec %S/rewrite_emit_struct_dot.yaml -o %t2.cir
// RUN: %file-check -vv -check-prefix=STRUCT_DOT %s --input-file %t2.cir
// STRUCT_DOT: cir.func @rewrite_struct_dot_host
// STRUCT_DOT-NOT: cir.call @accessor_val
// STRUCT_DOT: cir.alloca !ty_Point, !cir.ptr<!ty_Point>, ["arg_ref"]
// STRUCT_DOT: cir.store %{{[a-z0-9]+}}, %{{[0-9]+}} : !ty_Point, !cir.ptr<!ty_Point>
// STRUCT_DOT: cir.get_member %{{[0-9]+}}[1] {name = "y"} : !cir.ptr<!ty_Point> -> !cir.ptr<!u32i>
// STRUCT_DOT: cir.load %{{[0-9]+}} : !cir.ptr<!u32i>, !u32i

// RUN: %patchir-transform %s --spec %S/rewrite_emit_struct_dot_ternary.yaml -o %t3.cir
// RUN: %file-check -vv -check-prefix=DOT_TERN %s --input-file %t3.cir
// DOT_TERN: cir.func @rewrite_struct_dot_ternary_host
// DOT_TERN-NOT: cir.call @accessor_cond_val
// DOT_TERN: cir.alloca !ty_Point, !cir.ptr<!ty_Point>, ["arg_ref"]
// DOT_TERN: cir.ternary
// DOT_TERN: cir.get_member %{{[0-9]+}}[0] {name = "x"} : !cir.ptr<!ty_Point> -> !cir.ptr<!u32i>
// DOT_TERN: cir.get_member %{{[0-9]+}}[1] {name = "y"} : !cir.ptr<!ty_Point> -> !cir.ptr<!u32i>

!u32i = !cir.int<u, 32>
!ty_Point = !cir.struct<struct "Point" {!u32i, !u32i}>

module {
  cir.func private @__field_name_seed(%arg0: !cir.ptr<!ty_Point>) {
    %x = cir.get_member %arg0[0] {name = "x"} : !cir.ptr<!ty_Point> -> !cir.ptr<!u32i>
    %y = cir.get_member %arg0[1] {name = "y"} : !cir.ptr<!ty_Point> -> !cir.ptr<!u32i>
    %xv = cir.load %x : !cir.ptr<!u32i>, !u32i
    %yv = cir.load %y : !cir.ptr<!u32i>, !u32i
    cir.return
  }

  cir.func private @accessor_ptr(!cir.ptr<!ty_Point>) -> !u32i
  cir.func @rewrite_struct_arrow_host(%arg0: !cir.ptr<!ty_Point>) -> !u32i {
    %call = cir.call @accessor_ptr(%arg0) : (!cir.ptr<!ty_Point>) -> !u32i
    cir.return %call : !u32i
  }

  cir.func private @accessor_val(!ty_Point) -> !u32i
  cir.func @rewrite_struct_dot_host(%arg0: !cir.ptr<!ty_Point>) -> !u32i {
    %p_val = cir.load %arg0 : !cir.ptr<!ty_Point>, !ty_Point
    %call = cir.call @accessor_val(%p_val) : (!ty_Point) -> !u32i
    cir.return %call : !u32i
  }

  cir.func private @accessor_cond_val(!cir.bool, !ty_Point) -> !u32i
  cir.func @rewrite_struct_dot_ternary_host(%arg0: !cir.bool, %arg1: !ty_Point) -> !u32i {
    %call = cir.call @accessor_cond_val(%arg0, %arg1) : (!cir.bool, !ty_Point) -> !u32i
    cir.return %call : !u32i
  }
}
