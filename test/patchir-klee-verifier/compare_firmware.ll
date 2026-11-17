; RUN: %patchir-klee-verifier %s --compare-original %S/Inputs/compare_send.ll.in --target-function send --reject-handler reject --noreturn-handler reject --reject-effect log_reject --mmio-region 0x4000C000:0x4000CFFF --balance irq_off:irq_on --keep-target --harness-info %t.json -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll
; RUN: %file-check -check-prefix=INFO %s --input-file %t.json
; RUN: not %patchir-klee-verifier %s --compare-original %S/Inputs/compare_send.ll.in --target-function send --noreturn-handler reject --keep-target -S -o %t.ll 2>&1 | %file-check -check-prefix=NORETURN %s
; RUN: not %patchir-klee-verifier %s --compare-original %S/Inputs/compare_send.ll.in --target-function send --balance lock:unlock --keep-target -S -o %t.ll 2>&1 | %file-check -check-prefix=PAIR %s
; RUN: not %patchir-klee-verifier %s --compare-original %S/Inputs/compare_send.ll.in --target-function send --reject-effect absent --keep-target -S -o %t.ll 2>&1 | %file-check -check-prefix=EFFECT %s
; RUN: not %patchir-klee-verifier %s --compare-original %S/Inputs/compare_send.ll.in --target-function send --mmio-region 9:3 --keep-target -S -o %t.ll 2>&1 | %file-check -check-prefix=RANGE %s

; Test: firmware checks of the product harness.  The function writes a device
; register twice with interrupts off; the patched version rejects some inputs
; first, through a handler that is not declared noreturn in the module.

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-none-unknown-eabi"

@state = global i32 0

declare void @irq_off()
declare void @irq_on()
declare void @reject()
declare void @log_reject(i32)

define void @send(i32 %byte) {
entry:
  %slot = alloca i32
  store i32 %byte, ptr %slot
  %bad = icmp uge i32 %byte, 97
  br i1 %bad, label %refuse, label %ok
refuse:
  call void @log_reject(i32 %byte)
  call void @reject()
  br label %ok
ok:
  call void @irq_off()
  %status = load i32, ptr inttoptr (i32 1073790980 to ptr)
  store i32 %byte, ptr inttoptr (i32 1073790976 to ptr)
  store i32 10, ptr inttoptr (i32 1073790976 to ptr)
  call void @irq_on()
  store i32 %status, ptr @state
  ret void
}

; The declared ranges, and the series of values a device read returns.
; CHECK-DAG: @__patchestry_mmio = constant [1 x { i32, i32 }] [{ i32, i32 } { i32 1073790976, i32 1073795071 }]
; CHECK-DAG: @__patchestry_mmio_count = constant i32 1
; CHECK-DAG: @__patchestry_mmio_values = global [8 x i64] zeroinitializer

; An acquire or release is counted; it may be called before a rejection, so
; its id carries the effect bit (0x40000000).
; CHECK: define internal void @irq_off()
; CHECK: call void @__patchestry_balance(i32 0, i32 1)
; CHECK-NEXT: call void @__patchestry_trace_call(i32 1073741825, i32 0)
; CHECK: define internal void @irq_on()
; CHECK: call void @__patchestry_balance(i32 0, i32 -1)

; A load or store that may reach a device asks the runner first.  The store
; to the local and the store to the global are left alone.
; CHECK: define void @send.orig(i32 %byte)
; CHECK: store i32 %byte, ptr %slot
; CHECK: call i32 @__patchestry_mmio_read(i32 ptrtoint (ptr inttoptr (i32 1073790980 to ptr) to i32), i32 4, ptr %device.value)
; CHECK: device.real:
; CHECK-NEXT: %status = load i32, ptr inttoptr (i32 1073790980 to ptr)
; CHECK: device.read:
; CHECK: phi i32 [ %{{.*}}, %device.read ], [ %status, %device.real ]
; CHECK: call i32 @__patchestry_mmio_write(i32 ptrtoint (ptr inttoptr (i32 1073790976 to ptr) to i32), i64 %{{.*}}, i32 4)
; CHECK: device.real1:
; CHECK-NEXT: store i32 %byte, ptr inttoptr (i32 1073790976 to ptr)
; CHECK: call i32 @__patchestry_mmio_write(i32 ptrtoint (ptr inttoptr (i32 1073790976 to ptr) to i32), i64 10, i32 4)
; CHECK: store i32 %{{.*}}, ptr @state.orig
; CHECK-NOT: __patchestry_mmio
; CHECK: define void @send.patched(i32 %byte)

; A declared reject effect carries the effect bit too.
; CHECK: define internal void @log_reject(i32 %0)
; CHECK: call void @__patchestry_trace_call(i32 1073741827, i32 1)

; The handler does not return: in the patched version it ends the path
; through the reject-path check.
; CHECK: define internal void @reject() #[[#ATTR:]]
; CHECK: call void @__patchestry_path_end(i32 10)
; CHECK: call void @__patchestry_reject_end()

; A rejection: memory, calls, device writes and balance are checked, and the
; path ends with 128 plus a bit for each, or with 1.
; CHECK: define internal void @__patchestry_reject_end()
; CHECK: call i32 @__patchestry_memory_differs(ptr @state.orig, ptr @state.patched, i32 4, i32 0)
; CHECK: call i32 @__patchestry_reject_calls()
; CHECK: call i32 @__patchestry_reject_device()
; CHECK: call i32 @__patchestry_reject_unbalanced()
; CHECK: %[[#DIRTY:]] = or i32 %{{.*}}, 128
; CHECK: select i1 %{{.*}}, i32 %[[#DIRTY]], i32 1

; CHECK: define i32 @main()
; CHECK: call void @klee_make_symbolic(ptr @__patchestry_mmio_values, i32 64, ptr
; Both accepted: device writes and balance are compared too.
; CHECK: %device.differs = select i1 %{{.*}}, i32 8, i32 0
; CHECK: %balance.differs = select i1 %{{.*}}, i32 16, i32 0

; INFO-DAG: "device_accesses": 6
; INFO-DAG: "reject_effect": true
; INFO-DAG: "balance": "acquire"
; INFO-DAG: "kind": "reject-noreturn"
; NORETURN: 'reject' is declared not to return and is not a reject handler
; PAIR: balanced callee 'lock' is not called by either version
; EFFECT: reject effect 'absent' is not called by either version
; RANGE: --mmio-region expects LOW:HIGH
