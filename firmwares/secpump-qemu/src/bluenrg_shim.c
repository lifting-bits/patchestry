// SPDX-License-Identifier: GPL-3.0-or-later
// No-op implementations of the BlueNRG-MS ACI/HCI surface called by the
// upstream PumpService.c, plus a small driver that feeds synthesized
// "Attribute Modified" events into the upstream user_notify dispatcher.

#include "bluenrg_shim.h"

/* Provided by upstream PumpService.c (compiled in verbatim). */
extern void user_notify(void *pData);
extern __IO uint16_t connection_handle;

/* Sequential GATT-handle allocator. Real BlueNRG hands out:
 *   service_handle  = N
 *   char_decl       = N+1
 *   char_value      = N+2
 *   ...
 * PumpService.c stores the *char declaration* handle and dispatches on
 * (handle == charHandle + 1) — i.e. on the value handle. We only need to
 * preserve the +1 invariant: each aci_gatt_add_char must reserve two
 * handles so the value handle is unique. The exact starting value is not
 * observable from upstream code.
 */
static uint16_t s_next_handle = 0x0001;

static uint16_t alloc_handle(uint16_t span)
{
    uint16_t h = s_next_handle;
    s_next_handle = (uint16_t)(s_next_handle + span);
    return h;
}

tBleStatus aci_gatt_add_serv(uint8_t Service_UUID_Type,
                             const uint8_t *Service_UUID,
                             uint8_t Service_Type,
                             uint8_t Max_Attribute_Records,
                             uint16_t *Service_Handle)
{
    (void)Service_UUID_Type; (void)Service_UUID;
    (void)Service_Type; (void)Max_Attribute_Records;
    *Service_Handle = alloc_handle(1);
    return BLE_STATUS_SUCCESS;
}

tBleStatus aci_gatt_add_char(uint16_t Service_Handle,
                             uint8_t Char_UUID_Type,
                             const uint8_t *Char_UUID,
                             uint8_t Char_Value_Length,
                             uint8_t Char_Properties,
                             uint8_t Security_Permissions,
                             uint8_t GATT_Evt_Mask,
                             uint8_t Enc_Key_Size,
                             uint8_t Is_Variable,
                             uint16_t *Char_Handle)
{
    (void)Service_Handle; (void)Char_UUID_Type; (void)Char_UUID;
    (void)Char_Value_Length; (void)Char_Properties; (void)Security_Permissions;
    (void)GATT_Evt_Mask; (void)Enc_Key_Size; (void)Is_Variable;
    /* Reserve two handles: declaration + value. PumpService dispatches on
     * (returned_handle + 1). */
    *Char_Handle = alloc_handle(2);
    return BLE_STATUS_SUCCESS;
}

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
                                  uint16_t *Char_Desc_Handle)
{
    (void)Service_Handle; (void)Char_Handle; (void)Char_Desc_Uuid_Type;
    (void)Char_Desc_Uuid; (void)Char_Desc_Value_Max_Len;
    (void)Char_Desc_Value_Length; (void)Char_Desc_Value;
    (void)Security_Permissions; (void)Access_Permissions;
    (void)GATT_Evt_Mask; (void)Enc_Key_Size; (void)Is_Variable;
    *Char_Desc_Handle = alloc_handle(1);
    return BLE_STATUS_SUCCESS;
}

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
                                    uint16_t Slave_Conn_Interval_Max)
{
    (void)Advertising_Type; (void)Advertising_Interval_Min;
    (void)Advertising_Interval_Max; (void)Own_Address_Type;
    (void)Advertising_Filter_Policy; (void)Local_Name_Length; (void)Local_Name;
    (void)Service_Uuid_Length; (void)Service_Uuid_List;
    (void)Slave_Conn_Interval_Min; (void)Slave_Conn_Interval_Max;
    return BLE_STATUS_SUCCESS;
}

tBleStatus hci_le_set_scan_resp_data(uint8_t Scan_Response_Data_Length,
                                     const uint8_t *Scan_Response_Data)
{
    (void)Scan_Response_Data_Length; (void)Scan_Response_Data;
    return BLE_STATUS_SUCCESS;
}

tBleStatus aci_gatt_allow_read(uint16_t Connection_Handle)
{
    (void)Connection_Handle;
    return BLE_STATUS_SUCCESS;
}

/* Synthesize an EVT_BLUE_GATT_ATTRIBUTE_MODIFIED HCI event and feed it to
 * the upstream user_notify dispatcher. The byte layout matches the field
 * accesses in PumpService.c::user_notify():
 *   hci_uart_pckt.type == HCI_EVENT_PKT
 *   hci_uart_pckt.data -> hci_event_pckt
 *   hci_event_pckt.evt == EVT_VENDOR
 *   hci_event_pckt.data -> evt_blue_aci
 *   evt_blue_aci.ecode == EVT_BLUE_GATT_ATTRIBUTE_MODIFIED
 *   evt_blue_aci.data -> evt_gatt_attr_modified_IDB05A1
 */
void shim_post_attr_modified(uint16_t handle,
                             uint8_t length,
                             const uint8_t *data)
{
    /* One static buffer is enough: dispatch is fully synchronous. */
    static uint8_t buf[256];

    hci_uart_pckt *uart = (hci_uart_pckt *)buf;
    uart->type = HCI_EVENT_PKT;

    hci_event_pckt *evt = (hci_event_pckt *)uart->data;
    evt->evt = EVT_VENDOR;

    evt_blue_aci *blue = (evt_blue_aci *)evt->data;
    blue->ecode = EVT_BLUE_GATT_ATTRIBUTE_MODIFIED;

    evt_gatt_attr_modified_IDB05A1 *m =
        (evt_gatt_attr_modified_IDB05A1 *)blue->data;
    m->conn_handle = 0x0001;
    m->attr_handle = handle;
    m->data_length = length;
    m->offset      = 0;

    /* att_data is declared as att_data[1] but the trailing bytes follow it
     * in the buffer; bound-check against the static buffer size. */
    size_t header = (size_t)((uint8_t *)m->att_data - buf);
    if ((size_t)length + header > sizeof buf) {
        length = (uint8_t)(sizeof buf - header);
        m->data_length = length;
    }
    if (length && data) {
        memcpy(m->att_data, data, length);
    }
    evt->plen = (uint8_t)((size_t)length + (header - 2 /* uart hdr */));

    user_notify(uart);
}

void shim_post_le_connect(const uint8_t bdaddr[6], uint16_t conn_handle)
{
    static uint8_t buf[64];
    hci_uart_pckt *uart = (hci_uart_pckt *)buf;
    uart->type = HCI_EVENT_PKT;

    hci_event_pckt *evt = (hci_event_pckt *)uart->data;
    evt->evt = EVT_LE_META_EVENT;
    evt->plen = (uint8_t)(1 + sizeof(evt_le_connection_complete));

    evt_le_meta_event *meta = (evt_le_meta_event *)evt->data;
    meta->subevent = EVT_LE_CONN_COMPLETE;

    evt_le_connection_complete *cc = (evt_le_connection_complete *)meta->data;
    cc->status = 0;
    cc->handle = conn_handle;
    cc->role = 0;
    cc->peer_bdaddr_type = 0;
    if (bdaddr) memcpy(cc->peer_bdaddr, bdaddr, 6);
    else        memset(cc->peer_bdaddr, 0, 6);
    cc->interval = 0;
    cc->latency = 0;
    cc->supervision_timeout = 0;
    cc->master_clock_accuracy = 0;

    user_notify(uart);
}

void shim_post_disconnect(void)
{
    static uint8_t buf[16];
    hci_uart_pckt *uart = (hci_uart_pckt *)buf;
    uart->type = HCI_EVENT_PKT;

    hci_event_pckt *evt = (hci_event_pckt *)uart->data;
    evt->evt = EVT_DISCONN_COMPLETE;
    evt->plen = 0;

    user_notify(uart);
}

void shim_post_gatt_read_permit(uint16_t attr_handle)
{
    static uint8_t buf[32];
    hci_uart_pckt *uart = (hci_uart_pckt *)buf;
    uart->type = HCI_EVENT_PKT;

    hci_event_pckt *evt = (hci_event_pckt *)uart->data;
    evt->evt = EVT_VENDOR;
    evt->plen = (uint8_t)(2 + sizeof(evt_gatt_read_permit_req));

    evt_blue_aci *blue = (evt_blue_aci *)evt->data;
    blue->ecode = EVT_BLUE_GATT_READ_PERMIT_REQ;

    evt_gatt_read_permit_req *pr = (evt_gatt_read_permit_req *)blue->data;
    pr->conn_handle = connection_handle ? connection_handle : 0x0001;
    pr->attr_handle = attr_handle;

    user_notify(uart);
}
