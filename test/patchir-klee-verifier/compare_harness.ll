; RUN: %patchir-klee-verifier %s --compare-original %S/Inputs/compare_original.ll.in --target-function checked --reject-handler reject_request --reject-return -1 --arg-buffer 0:16 --arg-range 1:-2:13 --keep-target --harness-info %t.json -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll
; RUN: %file-check -check-prefix=INFO %s --input-file %t.json

; Test: product harness of a lifted original and its patched version.
; Both versions are in one module, each with its own copy of every global.

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-none-unknown-eabi"

@counter = global i32 0
@table = constant [2 x i32] [i32 3, i32 5]
@names = global ptr null
@shared_state = external global i32
@patch_limit = constant i32 8

declare i32 @read_sensor(i32)
declare void @log_event()
declare void @reject_request() noreturn

define i32 @helper(i32 %x) {
  %r = add i32 %x, 1
  ret i32 %r
}

define i32 @checked(ptr %buf, i32 %len) {
entry:
  %limit = load i32, ptr @patch_limit
  %bad = icmp ugt i32 %len, %limit
  br i1 %bad, label %reject, label %ok
reject:
  call void @reject_request()
  unreachable
ok:
  %v = call i32 @read_sensor(i32 %len)
  call void @log_event()
  %h = call i32 @helper(i32 %v)
  store i32 %h, ptr @counter
  %s = load i32, ptr @shared_state
  %b = load i8, ptr %buf
  %e = zext i8 %b to i32
  %sum = add i32 %e, %s
  ret i32 %sum
}

; The triple is kept.
; CHECK: target triple = "armv7-none-unknown-eabi"

; Each side has its own globals; a declared global gets storage.
; CHECK-DAG: @counter.orig = {{.*}}global i32 0
; CHECK-DAG: @counter.patched = {{.*}}global i32 0
; CHECK-DAG: @shared_state.orig = internal global i32 0
; CHECK-DAG: @shared_state.patched = internal global i32 0
; CHECK-DAG: @patch_limit.patched = {{.*}}constant i32 8
; CHECK-DAG: @__patchestry_values.read_sensor = internal global [8 x i32] zeroinitializer

; An address inside either copy of a global names the same place.
; CHECK-DAG: @__patchestry_pairs = constant [3 x { ptr, ptr, i32 }] [{ ptr, ptr, i32 } { ptr @counter.orig, ptr @counter.patched, i32 4 }
; CHECK-DAG: @__patchestry_pair_count = constant i32 3
; Constant data is compared by content, whichever side it belongs to.
; CHECK-DAG: @__patchestry_constants = constant [3 x { ptr, i32 }] [{ ptr, i32 } { ptr @table.orig, i32 8 }, { ptr, i32 } { ptr @table.patched, i32 8 }, { ptr, i32 } { ptr @patch_limit.patched, i32 4 }]

; An external function records its call, then returns the same value at the
; same call index on both sides.
; CHECK: define internal i32 @read_sensor(i32 %0)
; CHECK: call void @__patchestry_trace_call(i32 1, i32 1)
; CHECK-NEXT: call void @__patchestry_trace_arg(i32 %0)
; CHECK: load i32, ptr @__patchestry_calls.read_sensor
; CHECK: getelementptr inbounds [8 x i32], ptr @__patchestry_values.read_sensor

; A reject handler that does not return ends the path: accepted by the patch
; and rejected by the original is a P1 violation (10), rejected by the patch is 1.
; CHECK: define internal void @log_event()
; CHECK: call void @__patchestry_trace_call(i32 2, i32 0)
; CHECK: define internal void @reject_request()
; CHECK-NOT: __patchestry_trace_call
; CHECK: load i32, ptr @__patchestry_phase
; CHECK: call void @__patchestry_path_end(i32 10)
; CHECK: call void @__patchestry_path_end(i32 1)

; Both versions of the target and of the functions they define.
; CHECK-DAG: define {{.*}}i32 @checked.orig(ptr %buf, i32 %len)
; CHECK-DAG: define {{.*}}i32 @checked.patched(ptr %buf, i32 %len)
; CHECK-DAG: define {{.*}}i32 @helper.orig(i32 %x)
; CHECK-DAG: define {{.*}}i32 @helper.patched(i32 %x)

; CHECK: define i32 @main()
; Writable globals without pointers start symbolic and equal.
; CHECK-DAG: call void @klee_make_symbolic(ptr @counter.orig, i32 4, ptr
; CHECK-DAG: call void @llvm.memcpy{{.*}}(ptr {{.*}}@counter.patched, ptr {{.*}}@counter.orig, i32 4
; CHECK-DAG: call void @klee_make_symbolic(ptr @shared_state.orig, i32 4, ptr
; CHECK-NOT: klee_make_symbolic(ptr @names
; CHECK-NOT: klee_make_symbolic(ptr @table
; The declared buffer size is used for argument 0.  The harness's copies are
; allocated first, the buffer the versions are given last.
; CHECK: %arg0.patched = call ptr @malloc(i32 16)
; CHECK: %arg0.initial = call ptr @malloc(i32 16)
; CHECK: %arg0 = call ptr @malloc(i32 16)
; A declared range bounds an integer argument: low + (value mod span).
; CHECK: urem i32 %{{.*}}, 16
; CHECK: %arg1.bounded = add i32 %{{.*}}, -2
; The patched version runs first, then the original on restored buffers.
; CHECK: store i32 2, ptr @__patchestry_phase
; CHECK: call i32 @checked.patched(
; CHECK: call void @__patchestry_path_end(i32 1)
; CHECK: store i32 1, ptr @__patchestry_phase
; CHECK: call i32 @checked.orig(
; Run alone, the original reports only whether it accepted the input.
; CHECK: select i1 %only.original, i32 31, i32 10
; CHECK: call void @__patchestry_path_end(i32 30)
; Both accepted: the return values, the calls and the memory are compared,
; and a difference ends the path with 64 plus one bit per property.
; CHECK: %return.differs = select i1 %{{.*}}, i32 1, i32 0
; CHECK: %trace = call i32 @__patchestry_trace_differs()
; CHECK: call void @__patchestry_path_end(i32 20)
; CHECK: %calls.differ = select i1 %{{.*}}, i32 4, i32 0
; CHECK: call i32 @__patchestry_memory_differs(ptr @counter.orig, ptr @counter.patched, i32 4, i32 0)
; CHECK: call i32 @__patchestry_memory_differs(ptr @shared_state.orig, ptr @shared_state.patched, i32 4, i32 1)
; CHECK: call i32 @__patchestry_memory_differs(ptr %arg0, ptr %arg0.patched, i32 16, i32 2)
; CHECK: %[[#CODE:]] = or i32 %{{.*}}, 64
; CHECK: call void @__patchestry_path_end(i32 %[[#CODE]])
; CHECK: call void @__patchestry_path_end(i32 0)

; INFO-DAG: "target": "checked"
; INFO-DAG: "reject_returns": [
; INFO-DAG: "name": "read_sensor"
; INFO-DAG: "names"
; INFO-DAG: "memory_regions": [

; A null pointer as the reject value of a function that returns a pointer.
; RUN: %patchir-klee-verifier %S/Inputs/compare_pointer.ll.in --compare-original %S/Inputs/compare_pointer.ll.in --target-function pick --reject-return 0 --keep-target --harness-info %t.ptr.json -S -o %t.ptr.ll
; RUN: %file-check -check-prefix=POINTER %s --input-file %t.ptr.ll
; RUN: %file-check -check-prefix=LAYOUT %s --input-file %t.ptr.json
; POINTER: %[[#ADDR:]] = ptrtoint ptr %{{.*}} to i32
; POINTER: icmp eq i32 %[[#ADDR]], 0
; LAYOUT: "layout_adopted": false
