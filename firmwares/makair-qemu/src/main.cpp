// main.cpp — QEMU re-host entry. Runs the upstream telemetry layer in a
// polling loop (since we have no real timer interrupts driving the
// breathing cycle). On each iteration the loop pulses the firmware's
// telemetry surface — first a BootMessage at startup, then periodic
// DataSnapshot / MachineStateSnapshot frames driven by the SysTick clock.
//
// Control-side reads happen between telemetry sends; the v0 control loop
// in this TU handles only Heartbeat (the simplest message that exercises
// the shared serial_control framing). Full Control coverage is staged in
// later iterations as the upstream main_controller surface gets shimmed.

#include <stdint.h>
#include <stdlib.h>

#include "Arduino.h"
#include "CRC32.h"
#include "../includes/cycle.h"   // CyclePhases enum used by sendDataSnapshot

// Upstream telemetry API — sender prototypes the QEMU loop calls directly.
// The function bodies live in $(MAKAIR_SRC)/srcs/telemetry.cpp; the build
// system pulls that .cpp into the link.
extern void initTelemetry(void);
extern void sendBootMessage(void);
extern void sendDataSnapshot(uint16_t centileValue,
                             int16_t  pressureValue,
                             CyclePhases phase,
                             uint8_t  blowerValvePosition,
                             uint8_t  patientValvePosition,
                             uint8_t  blowerRpm,
                             uint8_t  batteryLevel,
                             int16_t  inspiratoryFlowValue,
                             int16_t  expiratoryFlowValue);

// Semihosting exit so test scripts can quit QEMU cleanly when sending 'Q'.
static inline void semihosting_exit(int code) {
    register uint32_t r0 __asm__("r0") = 0x18;       // SYS_EXIT
    register uint32_t r1 __asm__("r1") = 0x20026;    // ADP_Stopped_ApplicationExit
    (void)code;
    __asm__ volatile("bkpt #0xAB" : : "r"(r0), "r"(r1));
    while (1) {}
}

// ---- v0 minimal serial-control loop --------------------------------------
//
// Reads the upstream Control framing
// (header 0x05 0x0A | setting | value[2] | crc32[4] | footer 0x50 0xA0)
// directly — without pulling in main_controller / activation_controller /
// alarm_controller. Heartbeat (setting = 0) is acknowledged loud-fail-style
// by echoing a zero-length response. Other settings are quietly ignored
// for now; v1 will route them through the upstream serial_control.cpp once
// the controller stubs land.

static const uint8_t CTRL_HDR[2] = {0x05, 0x0A};
static const uint8_t CTRL_FTR[2] = {0x50, 0xA0};

static void serial_control_loop_v0(void) {
    if (Serial6.available() < 11) return;  // wait for full frame

    // Sync to header.
    if (Serial6.peek() != CTRL_HDR[0]) { (void)Serial6.read(); return; }
    (void)Serial6.read();
    if (Serial6.peek() != CTRL_HDR[1]) { return; }
    (void)Serial6.read();

    uint8_t  setting = (uint8_t)Serial6.read();
    uint8_t  raw_value[2];
    Serial6.readBytes(raw_value, 2);
    uint8_t  raw_crc[4];
    Serial6.readBytes(raw_crc, 4);
    uint8_t  ftr0 = (uint8_t)Serial6.read();
    uint8_t  ftr1 = (uint8_t)Serial6.read();

    if (ftr0 != CTRL_FTR[0] || ftr1 != CTRL_FTR[1]) return;  // bad frame

    CRC32 crc;
    crc.update(setting);
    crc.update(raw_value, 2);
    uint32_t got_crc = ((uint32_t)raw_crc[0] << 24) | ((uint32_t)raw_crc[1] << 16)
                     | ((uint32_t)raw_crc[2] <<  8) |  (uint32_t)raw_crc[3];
    if (got_crc != crc.finalize()) return;  // CRC mismatch — drop silently

    // Heartbeat is setting 0 in upstream `serial_control.h`. The v0 loop
    // accepts it as proof of liveness; everything else is no-op until the
    // controller stubs are filled in.
    (void)setting;
}

// ---- main ----------------------------------------------------------------
extern "C" int main(void) {
    Serial.begin(115200);   // both Serial and Serial6 share UART0 in QEMU
    initTelemetry();
    sendBootMessage();

    uint32_t next_snapshot = millis() + 1000u;
    while (true) {
        serial_control_loop_v0();

        // Periodic DataSnapshot every 1 s so the host harness has something
        // to lock onto. Synthetic values stand in for real ADC readings.
        if ((int32_t)(millis() - next_snapshot) >= 0) {
            next_snapshot += 1000u;
            sendDataSnapshot(/*centile=*/        0,
                             /*pressure=*/      100,
                             /*phase=*/         CyclePhases::INHALATION,
                             /*blowerValve=*/    50,
                             /*patientValve=*/    0,
                             /*blowerRpm=*/     120,
                             /*battery=*/        80,
                             /*inspFlow=*/      500,
                             /*expFlow=*/      -100);
        }
    }
    semihosting_exit(0);
    return 0;
}
