// RUN: true
typedef unsigned char  uint8_t;
typedef unsigned int   uint32_t;

// Per-TU assert: never returns. `static` keeps it at internal linkage so
// the lowered patch module is self-contained — Patcherex2's relocatable
// link does not need to resolve any extern when inserting the patched
// function into the firmware.
__attribute__((noreturn))
static void secpump_assert_fail(void) {
    for (;;) { }
}

// SecPump seeded vuln: ProcessVulnReq accumulates 16 bytes per call into a
// static AttackBuffer[256] indexed by a static counter (Attack_It); after 7
// calls the counter reaches 112 and the original implementation memcpy's 112
// bytes into a 4-byte stack buffer, overflowing the saved return address.
//
// Both originals (AttackBuffer / Attack_It) have file-local linkage and are
// not visible to this translation unit, so the replacement keeps its own
// internal state and never invokes the 4-byte stack-target memcpy.
void patch__replace__ProcessVulnReq(uint8_t *att_data) {
    static uint8_t  safe_buf[256];
    static uint32_t safe_it = 0;

    for (int i = 0; i < 16; ++i, ++safe_it) {
        if (safe_it >= sizeof(safe_buf)) {
            secpump_assert_fail();
        }
        safe_buf[safe_it] = att_data[i];
    }
    if (safe_it >= 112) {
        safe_it = 0;
    }
}
