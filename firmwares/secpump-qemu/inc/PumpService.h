// Copyright (C) 2019 LCIS Laboratory - Cyril Bresch
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Upstream Inc/PumpService.h with the BlueNRG header chain replaced by
// a single shim include. Function prototypes are unchanged.

#ifndef _PUMP_SERVICE_H_
#define _PUMP_SERVICE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "bluenrg_shim.h"

#define IDB04A1 0
#define IDB05A1 1

typedef int i32_t;

tBleStatus Add_Pump_Service(void);

/* Re-host shim addition: the upstream firmware never references these
 * handles outside PumpService.c (the GATT dispatch is in-TU). The shim's
 * main loop, which synthesizes Attribute-Modified events, needs to address
 * the value handle of each characteristic, so we expose them. */
extern uint16_t pumpServHandle, bolusCharHandle, modeCharHandle, vulnCharHandle;

void Read_Request_CB(uint16_t handle);
void Attribute_Modified_CB(uint16_t handle, uint8_t data_length, uint8_t *att_data);
void setConnectable(void);
void GAP_ConnectionComplete_CB(uint8_t addr[6], uint16_t handle);
void GAP_DisconnectionComplete_CB(void);
void user_notify(void * pData);

void ProcessModeReq(uint8_t * att_data);
void ProcessBolusReq(uint8_t * att_data);
void ProcessVulnReq(uint8_t * att_data);

void MaliciousMemCpy(void *dest, void *src, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* _PUMP_SERVICE_H_ */
