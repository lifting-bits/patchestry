// controller_stubs.cpp — minimal stubs for the upstream globals that
// `srcs/serial_control.cpp` calls into but which we do NOT compile from
// upstream (because they transitively pull in screen.cpp / mass_flow_meter
// / buzzer_control / pressure-sensor drivers / the full HardwareTimer-
// driven breathing cycle).
//
// We define just enough of each global's public surface to satisfy the
// linker. Each stub records the last (setting, value) it received so a
// future test can read it back via a debug telemetry path; for now the
// recordings are simply held in static storage.

#include "../includes/end_of_line_test.h"

// EolTest — only the two methods serial_control.cpp / respirator.cpp call.
// The class is defined in end_of_line_test.h; we provide just the bodies.
EolTest eolTest = EolTest();

EolTest::EolTest() : testActive(0) {}

void EolTest::activate(void)        { testActive = 1; }
bool EolTest::isRunning(void)       { return testActive == 1; }
void EolTest::onConfirm(void)       { /* user pressed confirm during EOL */ }
void EolTest::setupAndStart(void)   { /* would init pressure/buzzer/etc. */ }
