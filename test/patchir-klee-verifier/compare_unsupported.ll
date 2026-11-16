; RUN: %patchir-klee-verifier %s --compare-original %s --target-function lookup --keep-target --harness-info %t.json -S -o %t.ll
; RUN: %file-check -check-prefix=PTR %s --input-file %t.ll
; RUN: %file-check -check-prefix=INFO %s --input-file %t.json
; RUN: not %patchir-klee-verifier %s --compare-original %s --target-function plain --reject-handler missing --keep-target -S -o %t.ll 2>&1 | %file-check -check-prefix=HANDLER %s
; RUN: not %patchir-klee-verifier %s --compare-original %s --target-function plain --arg-buffer 3:8 --keep-target -S -o %t.ll 2>&1 | %file-check -check-prefix=ARG %s
; RUN: not %patchir-klee-verifier %s --compare-original %s --target-function plain --arg-range 0:9:3 --keep-target -S -o %t.ll 2>&1 | %file-check -check-prefix=RANGE %s

; Test: what the product harness cannot model is never guessed.  A call to an
; external function that returns a pointer ends the path as inconclusive (20).

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-none-unknown-eabi"

declare ptr @find_entry(i32)

define i32 @lookup(i32 %key) {
  %p = call ptr @find_entry(i32 %key)
  %v = load i32, ptr %p
  ret i32 %v
}

define i32 @plain(i32 %x) {
  ret i32 %x
}

; PTR: define internal ptr @find_entry(i32 %0)
; PTR-NEXT: entry:
; PTR-NEXT: call void @__patchestry_path_end(i32 20)
; INFO: "kind": "unmodelled"
; HANDLER: reject handler 'missing' is not called
; ARG: --arg-buffer names argument 3
; RANGE: --arg-range names argument 0
