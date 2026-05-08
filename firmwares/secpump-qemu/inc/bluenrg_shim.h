// SPDX-License-Identifier: GPL-3.0-or-later
// Minimal source-compatibility shim that lets the upstream
// SecPump-Vuln/Src/PumpService.c build and run under qemu without a
// BlueNRG-MS chip. Provides the BlueNRG/HCI/GATT type and constant surface
// referenced by PumpService.c, plus no-op implementations of the ACI and
// HCI helpers it calls. The shim never talks to a radio; instead it lets
// the host post synthesized "Attribute Modified" events into the same
// upstream dispatch path (user_notify -> Attribute_Modified_CB).

#ifndef BLUENRG_SHIM_H
#define BLUENRG_SHIM_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __IO
#define __IO volatile
#endif

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE  1
#endif

typedef uint8_t tBleStatus;

#define BLE_STATUS_SUCCESS  ((tBleStatus)0x00)
#define BLE_STATUS_ERROR    ((tBleStatus)0x01)

/* Attribute / GATT API constants used by Add_Pump_Service */
#define UUID_TYPE_16                0x01
#define UUID_TYPE_128               0x02
#define PRIMARY_SERVICE             0x01
#define CHAR_PROP_WRITE             0x08
#define ATTR_PERMISSION_NONE        0x00
#define GATT_NOTIFY_ATTRIBUTE_WRITE 0x01
#define ATTR_ACCESS_READ_WRITE      0x03
#define FORMAT_UINT8                0x04
#define UNIT_UNITLESS               0x2700
#define CHAR_FORMAT_DESC_UUID       0x2904

/* GAP advertising constants used by setConnectable */
#define AD_TYPE_COMPLETE_LOCAL_NAME 0x09
#define ADV_DATA_TYPE               0x00
#define ADV_INTERV_MIN              0x0030
#define ADV_INTERV_MAX              0x0060
#define PUBLIC_ADDR                 0x00
#define NO_WHITE_LIST_USE           0x00

/* HCI event-packet identifiers used by user_notify */
#define HCI_EVENT_PKT                0x04
#define EVT_DISCONN_COMPLETE         0x05
#define EVT_LE_META_EVENT            0x3E
#define EVT_LE_CONN_COMPLETE         0x01
#define EVT_VENDOR                   0xFF
#define EVT_BLUE_GATT_READ_PERMIT_REQ    0x0C13
#define EVT_BLUE_GATT_ATTRIBUTE_MODIFIED 0x0C01

/* Characteristic Format descriptor payload (Bluetooth Core Spec 0x2904) */
typedef struct {
    uint8_t  format;
    int8_t   exp;
    uint16_t unit;
    uint8_t  name_space;
    uint16_t desc;
} __attribute__((packed)) charactFormat;

/* HCI / event packet layouts. Packed so the byte stream the dispatcher
 * walks (cast through (void *)data) lays out the same way the BlueNRG
 * stack would deliver it. The exact field offsets only matter to
 * user_notify(), which casts these structs out of a packed buffer. */
typedef struct {
    uint8_t  type;
    uint8_t  data[1];
} __attribute__((packed)) hci_uart_pckt;

typedef struct {
    uint8_t  evt;
    uint8_t  plen;
    uint8_t  data[1];
} __attribute__((packed)) hci_event_pckt;

typedef struct {
    uint8_t  subevent;
    uint8_t  data[1];
} __attribute__((packed)) evt_le_meta_event;

typedef struct {
    uint8_t  status;
    uint16_t handle;
    uint8_t  role;
    uint8_t  peer_bdaddr_type;
    uint8_t  peer_bdaddr[6];
    uint16_t interval;
    uint16_t latency;
    uint16_t supervision_timeout;
    uint8_t  master_clock_accuracy;
} __attribute__((packed)) evt_le_connection_complete;

typedef struct {
    uint16_t ecode;
    uint8_t  data[1];
} __attribute__((packed)) evt_blue_aci;

typedef struct {
    uint16_t conn_handle;
    uint16_t attr_handle;
} __attribute__((packed)) evt_gatt_read_permit_req;

typedef struct {
    uint16_t conn_handle;
    uint16_t attr_handle;
    uint8_t  data_length;
    uint16_t offset;
    uint8_t  att_data[1];
} __attribute__((packed)) evt_gatt_attr_modified_IDB05A1;

/* ACI / HCI surface called from upstream PumpService.c. All implementations
 * are no-ops that allocate sequential handles and return success. */
tBleStatus aci_gatt_add_serv(uint8_t Service_UUID_Type,
                             const uint8_t *Service_UUID,
                             uint8_t Service_Type,
                             uint8_t Max_Attribute_Records,
                             uint16_t *Service_Handle);

tBleStatus aci_gatt_add_char(uint16_t Service_Handle,
                             uint8_t Char_UUID_Type,
                             const uint8_t *Char_UUID,
                             uint8_t Char_Value_Length,
                             uint8_t Char_Properties,
                             uint8_t Security_Permissions,
                             uint8_t GATT_Evt_Mask,
                             uint8_t Enc_Key_Size,
                             uint8_t Is_Variable,
                             uint16_t *Char_Handle);

tBleStatus aci_gatt_add_char_desc(uint16_t Service_Handle,
                                  uint16_t Char_Handle,
                                  uint8_t Char_Desc_Uuid_Type,
                                  const uint8_t *Char_Desc_Uuid,
                                  uint8_t Char_Desc_Value_Max_Len,
                                  uint8_t Char_Desc_Value_Length,
                                  const void *Char_Desc_Value,
                                  uint8_t Security_Permissions,
                                  uint8_t Access_Permissions,
                                  uint8_t GATT_Evt_Mask,
                                  uint8_t Enc_Key_Size,
                                  uint8_t Is_Variable,
                                  uint16_t *Char_Desc_Handle);

tBleStatus aci_gap_set_discoverable(uint8_t Advertising_Type,
                                    uint16_t Advertising_Interval_Min,
                                    uint16_t Advertising_Interval_Max,
                                    uint8_t Own_Address_Type,
                                    uint8_t Advertising_Filter_Policy,
                                    uint8_t Local_Name_Length,
                                    const char *Local_Name,
                                    uint8_t Service_Uuid_Length,
                                    const uint8_t *Service_Uuid_List,
                                    uint16_t Slave_Conn_Interval_Min,
                                    uint16_t Slave_Conn_Interval_Max);

tBleStatus hci_le_set_scan_resp_data(uint8_t Scan_Response_Data_Length,
                                     const uint8_t *Scan_Response_Data);

tBleStatus aci_gatt_allow_read(uint16_t Connection_Handle);

/* Driver: synthesize an EVT_BLUE_GATT_ATTRIBUTE_MODIFIED event for `handle`
 * carrying `length` bytes of `data`, and feed it through the upstream
 * user_notify() dispatcher. The call graph from here mirrors the upstream
 * radio-driven path exactly:
 *
 *     shim_post_attr_modified
 *       -> user_notify              [PumpService.c, upstream verbatim]
 *         -> Attribute_Modified_CB  [PumpService.c, upstream verbatim]
 *           -> ProcessModeReq | ProcessBolusReq | ProcessVulnReq
 */
void shim_post_attr_modified(uint16_t handle,
                             uint8_t length,
                             const uint8_t *data);

/* Drivers for the remaining upstream user_notify branches. Each builds the
 * event byte stream and hands it to user_notify(). */
void shim_post_le_connect(const uint8_t bdaddr[6], uint16_t conn_handle);
void shim_post_disconnect(void);
void shim_post_gatt_read_permit(uint16_t attr_handle);

#ifdef __cplusplus
}
#endif

#endif /* BLUENRG_SHIM_H */
