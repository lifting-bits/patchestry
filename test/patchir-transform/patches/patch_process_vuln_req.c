// RUN: true
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;

// Per-TU assert: never returns. `static` keeps it at internal linkage so
// the lowered patch module is self-contained — Patcherex2's relocatable
// link does not need to resolve any extern when inserting the patched
// function into the firmware.
//
// The patchestry-embedded clang frontend (lib/patchestry/Passes/Compiler.cpp)
// rejects `__builtin_trap` with "use of unknown builtin", so we stick with
// the infinite-loop form that every other patch in this tree uses.
__attribute__((noreturn))
static void secpump_assert_fail(void) {
    for (;;) { }                         // halt visibly; no return path.
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
