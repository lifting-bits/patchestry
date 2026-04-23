; RUN: %patchir-klee-verifier %s --target-function probe -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Regression test: absolute pointer constants — `inttoptr (i32 <N> to
; ptr)` with a non-null N — must be symbolized, not preserved.
;
; Decompiled firmware routinely encodes MMIO register bases,
; peripheral addresses, and vector tables as `inttoptr` constants.
; After the mandatory 32->64 retarget the integer is zero-extended
; and stored as the global's initializer (e.g. 0x0000000000001000),
; but that address no longer names a valid page in the KLEE process.
; Before this fix, emitInitWithInitializer treated every non-trivial
; pointer ConstantExpr as relocatable and emitted no store — the
; target dereferenced 0x1000 and KLEE terminated with a memory error
; on every path, defeating symbolic exploration entirely.
;
; The fix uses isRelocatablePointerConstant to distinguish
; Function/GlobalVariable/GlobalAlias (safe to preserve, loader
; rebinds them) from inttoptr/arithmetic ConstantExprs (absolute,
; must be symbolized).

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

%struct.MmioRegs = type { i32, ptr }

; Field 1 is an absolute MMIO address — not a host-valid pointer.
@regs = global %struct.MmioRegs { i32 0, ptr inttoptr (i32 4096 to ptr) }

; Top-level pointer global whose initializer is also absolute.
@top_reg = global ptr inttoptr (i32 8192 to ptr)

define i32 @probe() {
entry:
  ret i32 0
}

; --- Harness main() dispatches through __klee_init_globals ---
; CHECK:       define i32 @main()
; CHECK:       call void @__klee_init_globals()

; --- Per-global wrapper for @regs: field 0 is trivial (zero i32,
; symbolized via flat klee_make_symbolic). Field 1's initializer is
; an absolute inttoptr — not relocatable — so the wrapper must go
; through emitPointerField: malloc a fresh buffer, make it symbolic,
; and store the buffer pointer into the field. A pre-fix wrapper
; would have emitted no malloc for field 1 and left 0x1000 in place. ---
; CHECK:       define internal void @__klee_init_g_regs()
; CHECK:       call ptr @malloc(
; CHECK:       call void @klee_make_symbolic(
; CHECK:       store ptr
; CHECK:       ret void

; --- Per-global wrapper for @top_reg: top-level pointer whose
; initializer is absolute. buildTypeInitBody's pointer branch must
; run: allocate a symbolic buffer and store it into the global slot.
; A pre-fix wrapper was empty for this case. ---
; CHECK:       define internal void @__klee_init_g_top_reg()
; CHECK:       call ptr @malloc(
; CHECK:       call void @klee_make_symbolic(
; CHECK:       store ptr {{.*}}, ptr @top_reg
; CHECK:       ret void
