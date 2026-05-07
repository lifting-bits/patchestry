// Copyright (C) 2019 LCIS Laboratory - Cyril Bresch
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, in version 3.
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
// This program is part of the SecPump @https://github.com/r3glisss/SecPump

#ifndef INSULIN_CONTROLLER_H
#define INSULIN_CONTROLLER_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/// Controller Function
void InsulinController(uint8_t *GlucoseBuffer);

/// Manual Controller
void InsulinManualController();

/// Reset Controller
void ResetController();

#endif
