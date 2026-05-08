// LiquidCrystal.h — minimal stub for the Arduino LCD library that
// upstream srcs/screen.cpp pulls in. The QEMU re-host doesn't render an
// LCD; every method is a no-op.
#pragma once

#include "Arduino.h"

class LiquidCrystal {
public:
    LiquidCrystal(uint32_t /*rs*/,        uint32_t /*enable*/,
                  uint32_t /*d4*/,        uint32_t /*d5*/,
                  uint32_t /*d6*/,        uint32_t /*d7*/) {}
    void begin(uint32_t /*cols*/, uint32_t /*rows*/) {}
    void clear(void)                                  {}
    void home(void)                                   {}
    void noDisplay(void)                              {}
    void display(void)                                {}
    void setCursor(uint32_t /*col*/, uint32_t /*row*/) {}
    size_t print(const char * /*s*/)                  { return 0; }
    size_t print(int /*v*/)                           { return 0; }
    size_t print(unsigned int /*v*/)                  { return 0; }
    size_t print(long /*v*/)                          { return 0; }
    size_t print(unsigned long /*v*/)                 { return 0; }
    size_t print(uint8_t /*b*/)                       { return 0; }
    void   write(uint8_t /*b*/)                       {}
    void   createChar(uint8_t /*loc*/, uint8_t * /*p*/) {}
};
