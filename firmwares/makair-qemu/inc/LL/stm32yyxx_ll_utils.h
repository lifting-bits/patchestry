// Stub for the STM32 LL utility header. The real header pulls in the
// CMSIS-Core register map; we only need LL_GetUID_Word0/1/2 declarations,
// which are provided by Arduino.h (shimmed in src/arduino_shim.cpp).
#pragma once
#include "Arduino.h"
