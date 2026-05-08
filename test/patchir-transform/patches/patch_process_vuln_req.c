// RUN: true
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;

// Per-TU loud-fail: lowers to a single trapping instruction (UDF on ARMv7-M)
// and never returns. `__builtin_trap` is a compiler intrinsic — no extern
// symbol — so the lowered patch module stays self-contained for Patcherex2's
// relocatable link, and the trap routes the CPU into HardFault instead of
// silently spinning. The patchestry intrinsic library's __patchestry_assert_fail
// pulls in fprintf/abort and is only suitable for hosted patch targets.
__attribute__((noreturn))
static void secpump_assert_fail(void) {
    __builtin_trap();
}

// SecPump seeded vuln: ProcessVulnReq accumulates 16 bytes per call into a
// static AttackBuffer[256] indexed by Attack_It; after 7 calls Attack_It
// reaches 112 and the original implementation memcpy's 112 bytes into a
// 4-byte stack buffer, overflowing the saved return address.
//
// This replacement preserves the original accumulator and reset cadence —
// same buffer geometry, same `Attack_It == OVERFLOW` reset point — and only
// drops the stack-target memcpy. A bounds guard before each write loud-fails
// if the iterator ever escapes the 256-byte buffer.
//
// The original AttackBuffer / Attack_It have file-local linkage in
// PumpService.c and are not visible here, so we keep our own statics under
// the same names.
#define OVERFLOW 112

void patch__replace__ProcessVulnReq(uint8_t *att_data) {
    static uint8_t  AttackBuffer[256];
    static uint16_t Attack_It = 0;

    for (int i = 0; i < 16; ++i, ++Attack_It) {
        if (Attack_It >= sizeof(AttackBuffer)) {
            secpump_assert_fail();
        }
        AttackBuffer[Attack_It] = att_data[i];
    }
    if (Attack_It == OVERFLOW) {
        Attack_It = 0;
    }
}
