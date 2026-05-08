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
#include "../includes/cycle.h"             // CyclePhases, VentilationModes
#include "../includes/alarm.h"             // AlarmPriority
#include "../includes/alarm_controller.h"  // ALARMS_SIZE
#include "../includes/end_of_line_test.h"  // TestStep / TestState enums

// Upstream telemetry API — full sender list. Bodies live in
// $(MAKAIR_SRC)/srcs/telemetry.cpp (linked verbatim).
extern void initTelemetry(void);
extern void sendBootMessage(void);
extern void sendDataSnapshot(uint16_t, int16_t, CyclePhases,
                             uint8_t, uint8_t, uint8_t, uint8_t,
                             int16_t, int16_t);
extern void sendStoppedMessage(uint8_t, uint8_t, uint8_t, uint8_t, uint8_t,
                               bool, uint8_t, bool, uint8_t,
                               VentilationModes,
                               uint8_t, uint8_t, uint16_t, uint16_t,
                               uint8_t, uint8_t, uint8_t, uint8_t,
                               uint8_t, uint8_t,
                               uint16_t, uint16_t, uint16_t, uint16_t, uint16_t,
                               uint8_t, uint16_t, uint16_t,
                               uint8_t [ALARMS_SIZE], uint16_t,
                               uint8_t, uint8_t, uint16_t);
extern void sendMachineStateSnapshot(uint32_t, uint8_t, uint8_t, uint8_t,
                                     uint8_t, uint16_t, uint16_t, uint16_t,
                                     uint8_t [ALARMS_SIZE],
                                     uint16_t, uint8_t, bool, uint8_t,
                                     uint8_t, bool, uint8_t,
                                     VentilationModes,
                                     uint8_t, uint8_t, uint16_t, uint16_t,
                                     uint8_t, uint8_t, uint8_t, uint8_t,
                                     uint8_t, uint8_t,
                                     uint16_t, uint16_t, uint16_t,
                                     uint16_t, uint16_t, uint8_t,
                                     uint16_t, uint16_t, uint16_t,
                                     uint16_t, uint8_t, uint8_t, uint16_t);
extern void sendAlarmTrap(uint16_t, int16_t, CyclePhases, uint32_t,
                          uint8_t, AlarmPriority, bool,
                          uint32_t, uint32_t, uint32_t);
extern void sendControlAck(uint8_t, uint16_t);
extern void sendWatchdogRestartFatalError(void);
extern void sendCalibrationFatalError(int16_t, int16_t, int16_t,
                                      int16_t, int16_t);
extern void sendBatteryDeeplyDischargedFatalError(uint16_t);
extern void sendMassFlowMeterFatalError(void);
extern void sendInconsistentPressureFatalError(uint16_t);
extern void sendEolTestSnapshot(TestStep, TestState, char[]);

#include "../includes/rpi_watchdog.h"   // RpiWatchdog + global rpiWatchdog

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

// noinline so the function survives as a standalone symbol — the
// Vulnerability demo (see README) targets the available() call site here,
// and patchir-decomp can lift this small function but not main() in full.
static __attribute__((noinline)) void serial_control_loop_v0(void) {
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

    // Echo a ControlAck on every CRC-valid frame so the host harness can
    // verify the parser was actually reached. The synthetic per-tick
    // sendControlAck(1, 0) emitted by the polling loop is distinct (value=0),
    // so a ControlAck whose value matches the parsed raw_value uniquely
    // identifies a frame that traversed this path.
    uint16_t value = ((uint16_t)raw_value[0] << 8) | (uint16_t)raw_value[1];
    sendControlAck(setting, value);
}

// ---- main ----------------------------------------------------------------
extern "C" int main(void) {
    Serial.begin(115200);   // both Serial and Serial6 share UART0 in QEMU
    initTelemetry();
    sendBootMessage();

    uint32_t next_tick = millis() + 1000u;
    uint32_t cycle = 0;
    uint8_t  alarm_codes[ALARMS_SIZE] = {0};
    char     eol_msg[] = "OK";

    while (true) {
        serial_control_loop_v0();

        // Periodic burst once per tick: cycle through every Telemetry sender
        // so a host harness can lock onto each frame type within a few
        // seconds. Synthetic values stand in for real sensor readings.
        if ((int32_t)(millis() - next_tick) >= 0) {
            next_tick += 1000u;
            cycle++;

            // Tick the RPi heartbeat watchdog once per second (matching the
            // upstream main_state_machine.cpp:110 cadence). This drives the
            // COUNT_DOWN -> SWITCH_OFF_RASPBERRY -> SWITCH_ON_RASPBERRY ->
            // WAIT_FOR_FIRST_HEARTBEAT state machine, so a host that stops
            // sending Heartbeat Control frames will eventually trip the
            // upstream "RPi disconnected" recovery path. Each Heartbeat
            // received via serial_control.cpp resets the counter.
            rpiWatchdog.update();

            sendDataSnapshot(0, 100, CyclePhases::INHALATION,
                             50, 0, 120, 80, 500, -100);
            sendMachineStateSnapshot(cycle, 65, 30, 5, 20, 6500, 3000, 500,
                                     alarm_codes, 600, 50, true, 2, 20,
                                     false, 25, VentilationModes::PC_CMV,
                                     5, 5, 200, 3000, 30, 60, 30, 60,
                                     8, 35, 600, 200, 800, 100, 50, 60,
                                     1500, 1200, 100, 7400, 175, 0, 700);
            sendControlAck(/*setting=*/1, /*value=*/0);
            sendStoppedMessage(65, 30, 5, 20, 50, true, 2, false, 25,
                               VentilationModes::PC_CMV,
                               5, 5, 200, 3000, 30, 60, 30, 60, 8, 35,
                               600, 200, 800, 100, 50, 60, 1500, 7400,
                               alarm_codes, 175, 0, 0, 700);
            sendAlarmTrap(0, 100, CyclePhases::INHALATION, cycle,
                          /*alarmCode=*/12, AlarmPriority::ALARM_LOW,
                          /*triggered=*/true, /*expected=*/100,
                          /*measured=*/200, /*cyclesSince=*/0);
            sendEolTestSnapshot(TestStep::START, TestState::STATE_IN_PROGRESS,
                                eol_msg);
            // One fatal-error variant per tick so the host sees them all.
            switch (cycle % 4) {
            case 0: sendWatchdogRestartFatalError(); break;
            case 1: sendCalibrationFatalError(0, 0, 0, 0, 0); break;
            case 2: sendBatteryDeeplyDischargedFatalError(7000); break;
            case 3: sendMassFlowMeterFatalError(); break;
            }
        }
    }
    semihosting_exit(0);
    return 0;
}
