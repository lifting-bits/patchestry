// RUN: not %patchir-transform %s --spec %S/rewrite_capture_write_no_storage_reject.yaml -o /dev/null 2>&1 | %file-check -check-prefix=NO_STORAGE_ERR %s
// NO_STORAGE_ERR: capture '$R' is written to inside 'expr:'
// NO_STORAGE_ERR-SAME: has no backing storage at the match site
// NO_STORAGE_ERR-SAME: originates from cir.binop

!u32i = !cir.int<u, 32>
module {
  cir.func @nostorage_host(%arg0: !u32i, %arg1: !u32i) -> !u32i {
    %0 = cir.binop(mul, %arg0, %arg1) : !u32i
    cir.return %0 : !u32i
  }
}
