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

/**
 ******************************************************************************
 * @file    sensor_service.c
 * @version V1.0.0
 ******************************************************************************
 */

/* Upstream `bluenrg_gap_aci.h` and `bluenrg_gatt_aci.h` are folded into the
 * single `bluenrg_shim.h` (pulled in via `PumpService.h`). Everything below
 * is byte-identical to SecPump-Vuln/Src/PumpService.c. */
#include "PumpService.h"

/** @addtogroup Applications
 *  @{
 */

/** @addtogroup PumpDemo
 *  @{
 */

/** @defgroup PUMP_SERVICE
 * @{
 */

/** @defgroup PUMP_SERVICE_Private_Variables
 * @{
 */
/* Private variables ---------------------------------------------------------*/
__IO uint32_t connected = FALSE;
__IO uint8_t set_connectable = 1;
__IO uint16_t connection_handle = 0;
__IO uint8_t notification_enabled = FALSE;
uint16_t pumpServHandle, bolusCharHandle, modeCharHandle, vulnCharHandle;
/**
 * @}
 */

/** @defgroup PUMP_SERVICE_Private_Macros
 * @{
 */
/* Private macros ------------------------------------------------------------*/
#define COPY_UUID_128(uuid_struct, uuid_15, uuid_14, uuid_13, uuid_12,         \
                      uuid_11, uuid_10, uuid_9, uuid_8, uuid_7, uuid_6,        \
                      uuid_5, uuid_4, uuid_3, uuid_2, uuid_1, uuid_0)          \
  do {                                                                         \
    uuid_struct[0] = uuid_0;                                                   \
    uuid_struct[1] = uuid_1;                                                   \
    uuid_struct[2] = uuid_2;                                                   \
    uuid_struct[3] = uuid_3;                                                   \
    uuid_struct[4] = uuid_4;                                                   \
    uuid_struct[5] = uuid_5;                                                   \
    uuid_struct[6] = uuid_6;                                                   \
    uuid_struct[7] = uuid_7;                                                   \
    uuid_struct[8] = uuid_8;                                                   \
    uuid_struct[9] = uuid_9;                                                   \
    uuid_struct[10] = uuid_10;                                                 \
    uuid_struct[11] = uuid_11;                                                 \
    uuid_struct[12] = uuid_12;                                                 \
    uuid_struct[13] = uuid_13;                                                 \
    uuid_struct[14] = uuid_14;                                                 \
    uuid_struct[15] = uuid_15;                                                 \
  } while (0)

#define COPY_PUMP_SERVICE_UUID(uuid_struct)                                    \
  COPY_UUID_128(uuid_struct, 0x42, 0x82, 0x1a, 0x40, 0xe4, 0x77, 0x11, 0xe2,   \
                0x82, 0xd0, 0x00, 0x02, 0xa5, 0xd5, 0xc5, 0x1b)
#define COPY_BOLUS_CHAR_UUID(uuid_struct)                                      \
  COPY_UUID_128(uuid_struct, 0xa3, 0x2e, 0x55, 0x20, 0xe4, 0x77, 0x11, 0xe2,   \
                0xa9, 0xe3, 0x00, 0x02, 0xa5, 0xd5, 0xc5, 0x1b)
#define COPY_MODE_CHAR_UUID(uuid_struct)                                       \
  COPY_UUID_128(uuid_struct, 0xcd, 0x20, 0xc4, 0x80, 0xe4, 0x8b, 0x11, 0xe2,   \
                0x84, 0x0b, 0x00, 0x02, 0xa5, 0xd5, 0xc5, 0x1b)
#define COPY_VULNERABLE_CHAR_UUID(uuid_struct)                                 \
  COPY_UUID_128(uuid_struct, 0x01, 0xc5, 0x0b, 0x60, 0xe4, 0x8c, 0x11, 0xe2,   \
                0xa0, 0x73, 0x00, 0x02, 0xa5, 0xd5, 0xc5, 0x1b)

/* Store Value into a buffer in Little Endian Format */
#define STORE_LE_16(buf, val)                                                  \
  (((buf)[0] = (uint8_t)(val)), ((buf)[1] = (uint8_t)(val >> 8)))
/**
 * @}
 */

/** @defgroup PUMP_SERVICE_Exported_Functions
 * @{
 */

/**
 * @brief  Add the Pump service.
 *
 * @param  None
 * @retval Status
 */
tBleStatus Add_Pump_Service(void) {
  tBleStatus ret;
  uint8_t uuid[16];
  uint16_t uuid16;
  charactFormat charFormat;
  uint16_t descHandle;

  COPY_PUMP_SERVICE_UUID(uuid);
  ret = aci_gatt_add_serv(UUID_TYPE_128, uuid, PRIMARY_SERVICE, 10,
                          &pumpServHandle);
  if (ret != BLE_STATUS_SUCCESS)
    goto fail;

  /* MODE Characteristic */
  COPY_MODE_CHAR_UUID(uuid);
  ret = aci_gatt_add_char(pumpServHandle, UUID_TYPE_128, uuid, 1,
                          CHAR_PROP_WRITE, ATTR_PERMISSION_NONE,
                          GATT_NOTIFY_ATTRIBUTE_WRITE, 16, 0, &modeCharHandle);

  if (ret != BLE_STATUS_SUCCESS)
    goto fail;

  charFormat.format = FORMAT_UINT8;
  charFormat.exp = -1;
  charFormat.unit = UNIT_UNITLESS;
  charFormat.name_space = 0;
  charFormat.desc = 0;

  uuid16 = CHAR_FORMAT_DESC_UUID;

  ret = aci_gatt_add_char_desc(pumpServHandle, modeCharHandle, UUID_TYPE_16,
                               (uint8_t *)&uuid16, 7, 7, (void *)&charFormat,
                               ATTR_PERMISSION_NONE, ATTR_ACCESS_READ_WRITE, 0,
                               16, FALSE, &descHandle);

  if (ret != BLE_STATUS_SUCCESS)
    goto fail;
  printf(
      "[+] SecPump Service Created. Handle 0x%04X\r\n[+] MODE Charac handle: "
      "0x%04X\r\n",
      pumpServHandle, modeCharHandle);

  /* Bolus Characteristic */
  COPY_BOLUS_CHAR_UUID(uuid);
  ret = aci_gatt_add_char(pumpServHandle, UUID_TYPE_128, uuid, 16,
                          CHAR_PROP_WRITE, ATTR_PERMISSION_NONE,
                          GATT_NOTIFY_ATTRIBUTE_WRITE, 16, 0, &bolusCharHandle);
  if (ret != BLE_STATUS_SUCCESS)
    goto fail;

  charFormat.format = FORMAT_UINT8;
  charFormat.exp = -1;
  charFormat.unit = UNIT_UNITLESS;
  charFormat.name_space = 0;
  charFormat.desc = 0;

  uuid16 = CHAR_FORMAT_DESC_UUID;

  ret = aci_gatt_add_char_desc(pumpServHandle, bolusCharHandle, UUID_TYPE_16,
                               (uint8_t *)&uuid16, 7, 7, (void *)&charFormat,
                               ATTR_PERMISSION_NONE, ATTR_ACCESS_READ_WRITE, 0,
                               16, FALSE, &descHandle);
  if (ret != BLE_STATUS_SUCCESS)
    goto fail;
  printf("[+] BOLUS Charac handle: 0x%04X\n\r", bolusCharHandle);

  /* Vulnerable Characteristic */
  COPY_VULNERABLE_CHAR_UUID(uuid);
  ret = aci_gatt_add_char(pumpServHandle, UUID_TYPE_128, uuid, 16,
                          CHAR_PROP_WRITE, ATTR_PERMISSION_NONE,
                          GATT_NOTIFY_ATTRIBUTE_WRITE, 16, 0, &vulnCharHandle);

  if (ret != BLE_STATUS_SUCCESS)
    goto fail;

  charFormat.format = FORMAT_UINT8;
  charFormat.exp = -1;
  charFormat.unit = UNIT_UNITLESS;
  charFormat.name_space = 0;
  charFormat.desc = 0;

  uuid16 = CHAR_FORMAT_DESC_UUID;

  ret = aci_gatt_add_char_desc(pumpServHandle, vulnCharHandle, UUID_TYPE_16,
                               (uint8_t *)&uuid16, 7, 7, (void *)&charFormat,
                               ATTR_PERMISSION_NONE, ATTR_ACCESS_READ_WRITE, 0,
                               16, FALSE, &descHandle);

  if (ret != BLE_STATUS_SUCCESS)
    goto fail;
  printf("[+] VULNERABILTY Charac handle: 0x%04X\n\r", vulnCharHandle);

  return BLE_STATUS_SUCCESS;

fail:
  printf("[!] Error while adding PUMP service.\r\n");
  return BLE_STATUS_ERROR;
}

/**
 * @brief  Puts the device in connectable mode.
 *         If you want to specify a UUID list in the advertising data, those
 * data can be specified as a parameter in aci_gap_set_discoverable(). For
 * manufacture data, aci_gap_update_adv_data must be called.
 * @param  None
 * @retval None
 */
/* Ex.:
 *
 *  tBleStatus ret;
 *  const char local_name[] =
 * {AD_TYPE_COMPLETE_LOCAL_NAME,'B','l','u','e','N','R','G'}; const uint8_t
 * serviceUUIDList[] = {AD_TYPE_16_BIT_SERV_UUID,0x34,0x12}; const uint8_t
 * manuf_data[] = {4, AD_TYPE_MANUFACTURER_SPECIFIC_DATA, 0x05, 0x02, 0x01};
 *
 *  ret = aci_gap_set_discoverable(ADV_DATA_TYPE, ADV_INTERV_MIN,
 * ADV_INTERV_MAX, PUBLIC_ADDR, NO_WHITE_LIST_USE, 8, sizeof(local_name), 3,
 * serviceUUIDList, 0, 0); ret = aci_gap_update_adv_data(5, manuf_data);
 *
 */
void setConnectable(void) {
  tBleStatus ret;

  const char local_name[] = {
      AD_TYPE_COMPLETE_LOCAL_NAME, 'S', 'e', 'c', 'P', 'u', 'm', 'p'};

  /* disable scan response */
  hci_le_set_scan_resp_data(0, NULL);
  printf("[*] General Discoverable Mode.\r\n");

  ret = aci_gap_set_discoverable(ADV_DATA_TYPE, ADV_INTERV_MIN, ADV_INTERV_MAX,
                                 PUBLIC_ADDR, NO_WHITE_LIST_USE,
                                 sizeof(local_name), local_name, 0, NULL, 0, 0);
  if (ret != BLE_STATUS_SUCCESS) {
    printf("[!] Error while setting discoverable mode (%d)\r\n", ret);
  }
}

/**
 * @brief  This function is called when there is a LE Connection Complete event.
 * @param  uint8_t Address of peer device
 * @param  uint16_t Connection handle
 * @retval None
 */
void GAP_ConnectionComplete_CB(uint8_t addr[6], uint16_t handle) {
  connected = TRUE;
  connection_handle = handle;

  printf("[*] Connected to device:");
  for (uint32_t i = 5; i > 0; i--) {
    printf("%02X-", addr[i]);
  }
  printf("%02X\r\n", addr[0]);
}

/**
 * @brief  This function is called when the peer device gets disconnected.
 * @param  None
 * @retval None
 */
void GAP_DisconnectionComplete_CB(void) {
  connected = FALSE;
  printf("Disconnected\r\n");
  /* Make the device connectable again. */
  set_connectable = TRUE;
  notification_enabled = FALSE;
}

/**
 * @brief  Read request callback.
 * @param  uint16_t Handle of the attribute
 * @retval None
 */
void Read_Request_CB(uint16_t handle) {
  if (connection_handle != 0)
    aci_gatt_allow_read(connection_handle);
}

/**
 * @brief This function is called attribute value corresponding to
 * pumpCharHandle characteristic gets modified
 * @param handle : handle of the attribute
 * @param data_length : size of the modified attribute data
 * @param att_data : pointer to the modified attribute data
 * @retval None
 */
void Attribute_Modified_CB(uint16_t handle, uint8_t data_length,
                           uint8_t *att_data) {
  /* If GATT client has modified 'bolus characteristic' value, trigger the
   * bolus injection*/
  if (handle == modeCharHandle + 1) {
    /*place bolus code here*/
    printf("[*] MODE request\r\n");
    ProcessModeReq(att_data);
  }

  /* If GATT client has modified 'bolus characteristic' value, trigger the
   * bolus injection*/
  if (handle == bolusCharHandle + 1) {
    /*place bolus code here*/
    printf("[*] BOLUS request\r\n");
    ProcessBolusReq(att_data);
  }

  /* If GATT client has modified 'bolus characteristic' value, trigger the
   * bolus injection*/
  if (handle == vulnCharHandle + 1) {
    /*place bolus code here*/
    printf("[*] VULNERABILITY request\r\n");
    ProcessVulnReq(att_data);
  }
}

/**
 * @brief  Callback processing the ACI events.
 * @note   Inside this function each event must be identified and correctly
 *         parsed.
 * @param  void* Pointer to the ACI packet
 * @retval None
 */
void user_notify(void *pData) {
  hci_uart_pckt *hci_pckt = pData;
  /* obtain event packet */
  hci_event_pckt *event_pckt = (hci_event_pckt *)hci_pckt->data;

  if (hci_pckt->type != HCI_EVENT_PKT)
    return;

  switch (event_pckt->evt) {
  case EVT_DISCONN_COMPLETE: {
    GAP_DisconnectionComplete_CB();
  } break;

  case EVT_LE_META_EVENT: {
    evt_le_meta_event *evt = (void *)event_pckt->data;

    switch (evt->subevent) {
    case EVT_LE_CONN_COMPLETE: {
      evt_le_connection_complete *cc = (void *)evt->data;
      GAP_ConnectionComplete_CB(cc->peer_bdaddr, cc->handle);
    } break;
    }
  } break;

  case EVT_VENDOR: {
    evt_blue_aci *blue_evt = (void *)event_pckt->data;
    switch (blue_evt->ecode) {
    case EVT_BLUE_GATT_READ_PERMIT_REQ: {
      evt_gatt_read_permit_req *pr = (void *)blue_evt->data;
      Read_Request_CB(pr->attr_handle);
    } break;

    case EVT_BLUE_GATT_ATTRIBUTE_MODIFIED: {
      evt_gatt_attr_modified_IDB05A1 *evt =
          (evt_gatt_attr_modified_IDB05A1 *)blue_evt->data;
      Attribute_Modified_CB(evt->attr_handle, evt->data_length, evt->att_data);
    } break;
    }
  } break;
  }
}

/// Required external
extern uint8_t OPERATING_MODE;
extern uint32_t T_iterator;
extern float BolusConfig;

/// Forward decl needed because pump_cmd.c is gone; ResetController lives in
/// InsulinController.c.
extern void ResetController(void);

/// MODE Request Handler
void ProcessModeReq(uint8_t *att_data) {
  uint8_t Mode = (uint8_t)atoi((const char *)att_data);

  if (Mode == 0) {
    printf("[*] Switching to Auto MODE\r\n");
    T_iterator = 0;
    ResetController();
    OPERATING_MODE = 0;
  } else if (Mode == 1) {
    printf("[*] Switching to Manual MODE\r\n");
    OPERATING_MODE = 1;
  } else {
    printf("[!] ERROR MODE\n\r");
  }
}

/// Bolus Request Handler
void ProcessBolusReq(uint8_t *att_data) {
  BolusConfig = atof((const char *)att_data);
}

/// Vulnerable variables
static uint8_t AttackBuffer[256];
static uint16_t Attack_It = 0;

#define OVERFLOW 112

void ProcessVulnReq(uint8_t *att_data) {
  char VulnBuffer[4];
  int i;

  for (i = 0; i < 16; ++i, ++Attack_It) {
    AttackBuffer[Attack_It] = att_data[i];
  }

  printf("Iterator: %d\n\r", Attack_It);

  if (Attack_It == OVERFLOW) {
    printf("Buffer overflow:\r\n");
    MaliciousMemCpy(VulnBuffer, AttackBuffer, OVERFLOW);
    Attack_It = 0;
  }
}

/// Malicious function
void MaliciousMemCpy(void *dest, void *src, size_t n) {
  /// Typecast src and dest addresses to (char *)
  char *csrc = (char *)src;
  char *cdest = (char *)dest;

  /// Copy contents of src[] to dest[]
  for (int i = 0; i < n; i++)
    cdest[i] = csrc[i];
}
