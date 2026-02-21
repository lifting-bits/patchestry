; RUN: %patchir-seahorn-verifier -S %s -o %t.ll
; RUN: %file-check %s < %t.ll

; Test pointer-specific edge cases including various alignments

target triple = "arm-unknown-linux-gnueabihf"
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

declare void @__VERIFIER_assume(i1)
declare void @__VERIFIER_assert(i1)

; =============================================================================
; Test 1: Alignment 1 (byte-aligned, always satisfied)
; =============================================================================

; CHECK-LABEL: define void @align_1
define void @align_1(i8* %ptr) {
entry:
  ; CHECK: %[[INTPTR:.*]] = ptrtoint ptr %ptr to i32
  ; CHECK-NEXT: %[[MOD:.*]] = urem i32 %[[INTPTR]], 1
  ; CHECK-NEXT: %[[ALIGNED:.*]] = icmp eq i32 %[[MOD]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED]])
  call void @process_byte(i8* %ptr), !static_contract !0
  ret void
}

declare void @process_byte(i8*)

!0 = !{!"static_contract", !"preconditions=[{kind=alignment, target=Arg(0), align=1}]"}

; =============================================================================
; Test 2: Alignment 2 (2-byte aligned)
; =============================================================================

; CHECK-LABEL: define void @align_2
define void @align_2(i16* %ptr) {
entry:
  ; CHECK: %[[INTPTR:.*]] = ptrtoint ptr %ptr to i32
  ; CHECK-NEXT: %[[MOD:.*]] = urem i32 %[[INTPTR]], 2
  ; CHECK-NEXT: %[[ALIGNED:.*]] = icmp eq i32 %[[MOD]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED]])
  call void @process_i16(i16* %ptr), !static_contract !1
  ret void
}

declare void @process_i16(i16*)

!1 = !{!"static_contract", !"preconditions=[{kind=alignment, target=Arg(0), align=2}]"}

; =============================================================================
; Test 3: Alignment 4 (4-byte aligned, common for i32)
; =============================================================================

; CHECK-LABEL: define void @align_4
define void @align_4(i32* %ptr) {
entry:
  ; CHECK: %[[INTPTR:.*]] = ptrtoint ptr %ptr to i32
  ; CHECK-NEXT: %[[MOD:.*]] = urem i32 %[[INTPTR]], 4
  ; CHECK-NEXT: %[[ALIGNED:.*]] = icmp eq i32 %[[MOD]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED]])
  call void @dma_transfer_i32(i32* %ptr), !static_contract !2
  ret void
}

declare void @dma_transfer_i32(i32*)

!2 = !{!"static_contract", !"preconditions=[{kind=alignment, target=Arg(0), align=4}]"}

; =============================================================================
; Test 4: Alignment 8 (8-byte aligned, common for i64/double)
; =============================================================================

; CHECK-LABEL: define void @align_8
define void @align_8(i64* %ptr) {
entry:
  ; CHECK: %[[INTPTR:.*]] = ptrtoint ptr %ptr to i32
  ; CHECK-NEXT: %[[MOD:.*]] = urem i32 %[[INTPTR]], 8
  ; CHECK-NEXT: %[[ALIGNED:.*]] = icmp eq i32 %[[MOD]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED]])
  call void @process_i64(i64* %ptr), !static_contract !3
  ret void
}

declare void @process_i64(i64*)

!3 = !{!"static_contract", !"preconditions=[{kind=alignment, target=Arg(0), align=8}]"}

; =============================================================================
; Test 5: Alignment 16 (16-byte aligned, SIMD/vector operations)
; =============================================================================

; CHECK-LABEL: define void @align_16
define void @align_16(<4 x i32>* %ptr) {
entry:
  ; CHECK: %[[INTPTR:.*]] = ptrtoint ptr %ptr to i32
  ; CHECK-NEXT: %[[MOD:.*]] = urem i32 %[[INTPTR]], 16
  ; CHECK-NEXT: %[[ALIGNED:.*]] = icmp eq i32 %[[MOD]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED]])
  call void @simd_operation(<4 x i32>* %ptr), !static_contract !4
  ret void
}

declare void @simd_operation(<4 x i32>*)

!4 = !{!"static_contract", !"preconditions=[{kind=alignment, target=Arg(0), align=16}]"}

; =============================================================================
; Test 6: Alignment 32 (32-byte aligned, AVX operations)
; =============================================================================

; CHECK-LABEL: define void @align_32
define void @align_32(<8 x i32>* %ptr) {
entry:
  ; CHECK: %[[INTPTR:.*]] = ptrtoint ptr %ptr to i32
  ; CHECK-NEXT: %[[MOD:.*]] = urem i32 %[[INTPTR]], 32
  ; CHECK-NEXT: %[[ALIGNED:.*]] = icmp eq i32 %[[MOD]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED]])
  call void @avx_operation(<8 x i32>* %ptr), !static_contract !5
  ret void
}

declare void @avx_operation(<8 x i32>*)

!5 = !{!"static_contract", !"preconditions=[{kind=alignment, target=Arg(0), align=32}]"}

; =============================================================================
; Test 7: Alignment 64 (64-byte aligned, cache line)
; =============================================================================

; CHECK-LABEL: define void @align_64
define void @align_64(i8* %ptr) {
entry:
  ; CHECK: %[[INTPTR:.*]] = ptrtoint ptr %ptr to i32
  ; CHECK-NEXT: %[[MOD:.*]] = urem i32 %[[INTPTR]], 64
  ; CHECK-NEXT: %[[ALIGNED:.*]] = icmp eq i32 %[[MOD]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED]])
  call void @cache_aligned_operation(i8* %ptr), !static_contract !6
  ret void
}

declare void @cache_aligned_operation(i8*)

!6 = !{!"static_contract", !"preconditions=[{kind=alignment, target=Arg(0), align=64}]"}

; =============================================================================
; Test 8: Alignment 128 (128-byte aligned, larger cache line)
; =============================================================================

; CHECK-LABEL: define void @align_128
define void @align_128(i8* %ptr) {
entry:
  ; CHECK: %[[INTPTR:.*]] = ptrtoint ptr %ptr to i32
  ; CHECK-NEXT: %[[MOD:.*]] = urem i32 %[[INTPTR]], 128
  ; CHECK-NEXT: %[[ALIGNED:.*]] = icmp eq i32 %[[MOD]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED]])
  call void @large_alignment_operation(i8* %ptr), !static_contract !7
  ret void
}

declare void @large_alignment_operation(i8*)

!7 = !{!"static_contract", !"preconditions=[{kind=alignment, target=Arg(0), align=128}]"}

; =============================================================================
; Test 9: Alignment 256 (256-byte aligned, maximum typical alignment)
; =============================================================================

; CHECK-LABEL: define void @align_256
define void @align_256(i8* %ptr) {
entry:
  ; CHECK: %[[INTPTR:.*]] = ptrtoint ptr %ptr to i32
  ; CHECK-NEXT: %[[MOD:.*]] = urem i32 %[[INTPTR]], 256
  ; CHECK-NEXT: %[[ALIGNED:.*]] = icmp eq i32 %[[MOD]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED]])
  call void @max_alignment_operation(i8* %ptr), !static_contract !8
  ret void
}

declare void @max_alignment_operation(i8*)

!8 = !{!"static_contract", !"preconditions=[{kind=alignment, target=Arg(0), align=256}]"}

; =============================================================================
; Test 10: Multiple pointers with different alignments
; =============================================================================

; CHECK-LABEL: define void @multiple_alignments
define void @multiple_alignments(i32* %ptr1, i64* %ptr2, <4 x i32>* %ptr3) {
entry:
  ; First pointer: 4-byte aligned
  ; CHECK: %[[INTPTR1:.*]] = ptrtoint ptr %ptr1 to i32
  ; CHECK-NEXT: %[[MOD1:.*]] = urem i32 %[[INTPTR1]], 4
  ; CHECK-NEXT: %[[ALIGNED1:.*]] = icmp eq i32 %[[MOD1]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED1]])
  call void @func_align4(i32* %ptr1), !static_contract !9

  ; Second pointer: 8-byte aligned
  ; CHECK: %[[INTPTR2:.*]] = ptrtoint ptr %ptr2 to i32
  ; CHECK-NEXT: %[[MOD2:.*]] = urem i32 %[[INTPTR2]], 8
  ; CHECK-NEXT: %[[ALIGNED2:.*]] = icmp eq i32 %[[MOD2]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED2]])
  call void @func_align8(i64* %ptr2), !static_contract !10

  ; Third pointer: 16-byte aligned
  ; CHECK: %[[INTPTR3:.*]] = ptrtoint ptr %ptr3 to i32
  ; CHECK-NEXT: %[[MOD3:.*]] = urem i32 %[[INTPTR3]], 16
  ; CHECK-NEXT: %[[ALIGNED3:.*]] = icmp eq i32 %[[MOD3]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED3]])
  call void @func_align16(<4 x i32>* %ptr3), !static_contract !11

  ret void
}

declare void @func_align4(i32*)
declare void @func_align8(i64*)
declare void @func_align16(<4 x i32>*)

!9 = !{!"static_contract", !"preconditions=[{kind=alignment, target=Arg(0), align=4}]"}
!10 = !{!"static_contract", !"preconditions=[{kind=alignment, target=Arg(0), align=8}]"}
!11 = !{!"static_contract", !"preconditions=[{kind=alignment, target=Arg(0), align=16}]"}

; =============================================================================
; Test 11: Nonnull combined with alignment
; =============================================================================

; CHECK-LABEL: define void @nonnull_and_aligned
define void @nonnull_and_aligned(i32* %ptr) {
entry:
  ; CHECK: %[[NULL:.*]] = icmp ne ptr %ptr, null
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[NULL]])
  ; CHECK: %[[INTPTR:.*]] = ptrtoint ptr %ptr to i32
  ; CHECK-NEXT: %[[MOD:.*]] = urem i32 %[[INTPTR]], 4
  ; CHECK-NEXT: %[[ALIGNED:.*]] = icmp eq i32 %[[MOD]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[ALIGNED]])
  call void @safe_and_aligned(i32* %ptr), !static_contract !12
  ret void
}

declare void @safe_and_aligned(i32*)

!12 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}, {kind=alignment, target=Arg(0), align=4}]"}
