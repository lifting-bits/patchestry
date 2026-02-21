; RUN: %patchir-seahorn-verifier -S %s -o %t.ll
; RUN: %file-check %s < %t.ll

; Test alignment precondition

target triple = "arm-unknown-linux-gnueabihf"
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

declare void @__VERIFIER_assume(i1)
declare void @__VERIFIER_assert(i1)

; CHECK-LABEL: define void @aligned_access
define void @aligned_access(i32* %ptr) {
entry:
  ; CHECK: %[[INTPTR:.*]] = ptrtoint ptr %ptr to i32
  ; CHECK-NEXT: %[[MOD:.*]] = urem i32 %[[INTPTR]], 4
  ; CHECK-NEXT: %[[ALIGNED:.*]] = icmp eq i32 %[[MOD]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED]])
  call void @dma_transfer(i32* %ptr), !static_contract !0
  ret void
}

declare void @dma_transfer(i32*)

; Contract: pointer must be 4-byte aligned
!0 = !{!"static_contract", !"preconditions=[{kind=alignment, target=Arg(0), align=4}]"}
