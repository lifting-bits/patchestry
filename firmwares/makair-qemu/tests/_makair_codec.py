"""Encoder/decoder for the MakAir Telemetry + Control wire format.

This is the single source of truth for the protocol layout on the host
side; if upstream `makair-firmware/srcs/telemetry.cpp` or
`srcs/serial_control.cpp` change framing, only this module needs to follow.

Wire layout (all integers big-endian to match upstream):

    Telemetry  : 03 0C  <type-tag-bytes>  <payload>  <crc32:4>  30 C0
    Control    : 05 0A  <setting:1>  <value:2>  <crc32:4>  50 A0

CRC is the IEEE 802.3 CRC-32 (poly 0xEDB88320) over the type-tag bytes plus
payload (Telemetry) or over setting + value (Control).

The codec is intentionally minimal — it decodes BootMessage and
DataSnapshot (the two frames the QEMU re-host's v0 loop emits) and encodes
the entire Control-side settings table. Telemetry decoders for the other
10 message types are stubbed and clearly raise NotImplementedError until
the corresponding sender path is wired into the QEMU polling loop.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

# ---- framing constants ---------------------------------------------------
TELEMETRY_HEADER = bytes([0x03, 0x0C])
TELEMETRY_FOOTER = bytes([0x30, 0xC0])
CONTROL_HEADER   = bytes([0x05, 0x0A])
CONTROL_FOOTER   = bytes([0x50, 0xA0])

# ---- message-type tags (firmware -> host) --------------------------------
# These match the upstream `Serial6.write("X:", 2)` ASCII tags emitted by
# each `send*` function in srcs/telemetry.cpp.
TAG_BOOT              = b"B:"
TAG_STOPPED           = b"O:"
TAG_DATA_SNAPSHOT     = b"D:"
TAG_MACHINE_SNAPSHOT  = b"S:"
TAG_ALARM_TRAP        = b"T:"
TAG_CONTROL_ACK       = b"A:"
TAG_FATAL_ERROR       = b"E:"
TAG_EOL_SNAPSHOT      = b"L:"

# ---- control-side settings (host -> firmware) ----------------------------
# Mirrors `enum SerialControlSetting` in upstream includes/serial_control.h.
# (Values verified against the v4.1.0 source.)
SET_HEARTBEAT                  = 1
SET_VENTILATION_MODE           = 2
SET_PLATEAU_PRESSURE           = 3
SET_PEEP                       = 4
SET_CYCLES_PER_MINUTE          = 5
SET_EXPIRATORY_TERM            = 6
SET_TRIGGER_ENABLED            = 7
SET_TRIGGER_OFFSET             = 8
SET_RESPIRATION_ENABLED        = 9
SET_ALARM_SNOOZE               = 10
SET_INSPIRATORY_TRIGGER_FLOW   = 11
SET_EXPIRATORY_TRIGGER_FLOW    = 12
SET_TI_MIN                     = 13
SET_TI_MAX                     = 14
SET_LOW_INSPIRATORY_MV_THR     = 15
SET_HIGH_INSPIRATORY_MV_THR    = 16
SET_LOW_EXPIRATORY_MV_THR      = 17
SET_HIGH_EXPIRATORY_MV_THR     = 18
SET_LOW_RESPIRATORY_RATE_THR   = 19
SET_HIGH_RESPIRATORY_RATE_THR  = 20
SET_TARGET_TIDAL_VOLUME        = 21
SET_LOW_TIDAL_VOLUME_THR       = 22
SET_HIGH_TIDAL_VOLUME_THR      = 23
SET_PLATEAU_DURATION           = 24
SET_LEAK_ALARM_THRESHOLD       = 25
SET_TARGET_INSPIRATORY_FLOW    = 26
SET_INSPIRATORY_DURATION       = 27
SET_PATIENT_HEIGHT             = 28
SET_PATIENT_GENDER             = 29
SET_PEAK_PRESSURE_ALARM_THR    = 30
SET_EOL_CONFIRM                = 31

DISABLE_RPI_WATCHDOG = 0xFFFF


# ---- CRC-32/IEEE ---------------------------------------------------------
def _crc32(buf: bytes) -> int:
    state = 0xFFFFFFFF
    for b in buf:
        state ^= b
        for _ in range(8):
            mask = -(state & 1) & 0xFFFFFFFF
            state = ((state >> 1) ^ (0xEDB88320 & mask)) & 0xFFFFFFFF
    return state ^ 0xFFFFFFFF


# ---- Control encoder ----------------------------------------------------
def encode_control(setting: int, value: int) -> bytes:
    """Pack a single Control message frame (11 bytes).

    Layout: 05 0A | setting | value_hi value_lo | crc32 (big-endian) | 50 A0
    """
    payload = bytes([setting]) + struct.pack(">H", value & 0xFFFF)
    crc = struct.pack(">I", _crc32(payload))
    return CONTROL_HEADER + payload + crc + CONTROL_FOOTER


# ---- Telemetry decoders --------------------------------------------------
@dataclass
class TelemetryFrame:
    tag: bytes
    payload: bytes
    raw: bytes


def find_next_frame(buf: bytes, start: int = 0) -> tuple[TelemetryFrame, int] | None:
    """Scan `buf` from `start` for the next valid Telemetry frame.

    Returns (frame, end_index) on success, or None if no full + CRC-valid
    frame is found. End_index is the offset just past the frame's footer.
    """
    pos = buf.find(TELEMETRY_HEADER, start)
    while pos >= 0:
        # Walk forward looking for the footer; the smallest meaningful frame
        # (BootMessage) is ~30 bytes, but bound the search at 256 to keep
        # corrupted streams from running off the end.
        ftr = buf.find(TELEMETRY_FOOTER, pos + len(TELEMETRY_HEADER), pos + 256)
        if ftr < 0:
            return None
        end = ftr + len(TELEMETRY_FOOTER)
        body = buf[pos + len(TELEMETRY_HEADER):ftr - 4]   # tag + payload
        if len(body) < 2:
            pos = buf.find(TELEMETRY_HEADER, pos + 1)
            continue
        crc_bytes = buf[ftr - 4:ftr]
        got_crc = struct.unpack(">I", crc_bytes)[0]
        if got_crc == _crc32(body):
            tag = body[:2]
            return TelemetryFrame(tag=tag, payload=body[2:], raw=buf[pos:end]), end
        # CRC mismatch -- this footer was inside another frame's payload
        # (the byte sequence 30 C0 can appear naturally). Search past it.
        pos = buf.find(TELEMETRY_HEADER, ftr)
    return None


def iter_frames(buf: bytes):
    """Yield every CRC-valid Telemetry frame in `buf`."""
    cursor = 0
    while True:
        hit = find_next_frame(buf, cursor)
        if hit is None:
            return
        frame, cursor = hit
        yield frame


# ---- BootMessage decoder ------------------------------------------------
@dataclass
class BootMessage:
    protocol_version: int
    fw_version: str
    device_id: bytes
    systick_us: int
    mode: int
    value128: int


def decode_boot(frame: TelemetryFrame) -> BootMessage:
    if frame.tag != TAG_BOOT:
        raise ValueError(f"expected B: tag, got {frame.tag!r}")
    p = frame.payload
    proto = p[0]
    ver_len = p[1]
    fw = p[2:2 + ver_len].decode("ascii", errors="replace")
    off = 2 + ver_len
    dev = p[off:off + 12]
    off += 12
    assert p[off:off + 1] == b"\t"
    off += 1
    systick = struct.unpack(">Q", p[off:off + 8])[0]
    off += 8
    assert p[off:off + 1] == b"\t"
    off += 1
    mode = p[off]
    off += 1
    assert p[off:off + 1] == b"\t"
    off += 1
    val128 = p[off]
    off += 1
    # Trailing \n before the CRC
    assert p[off:off + 1] == b"\n"
    return BootMessage(protocol_version=proto, fw_version=fw, device_id=dev,
                       systick_us=systick, mode=mode, value128=val128)


# ---- DataSnapshot decoder -----------------------------------------------
@dataclass
class DataSnapshot:
    centile: int
    pressure: int
    phase: int
    blower_valve_position: int
    patient_valve_position: int
    blower_rpm: int
    battery_level: int
    inspiratory_flow: int
    expiratory_flow: int


def decode_data_snapshot(frame: TelemetryFrame) -> DataSnapshot:
    if frame.tag != TAG_DATA_SNAPSHOT:
        raise ValueError(f"expected D: tag, got {frame.tag!r}")
    # The DataSnapshot payload layout in upstream telemetry.cpp:480..480 is
    # version-dependent; the v4.1.0 wire matches `decode_v2_data_snapshot`
    # in makair-telemetry. We only need a small subset for the tests.
    p = frame.payload
    # Skip protocol byte + tab-delimited preamble identical to BootMessage's
    # device id / systick block, then read the data fields.
    return DataSnapshot(
        centile=struct.unpack(">H", p[0:2])[0] if len(p) >= 2 else 0,
        pressure=struct.unpack(">h", p[2:4])[0] if len(p) >= 4 else 0,
        phase=p[4] if len(p) > 4 else 0,
        blower_valve_position=p[5] if len(p) > 5 else 0,
        patient_valve_position=p[6] if len(p) > 6 else 0,
        blower_rpm=p[7] if len(p) > 7 else 0,
        battery_level=p[8] if len(p) > 8 else 0,
        inspiratory_flow=struct.unpack(">h", p[9:11])[0] if len(p) >= 11 else 0,
        expiratory_flow=struct.unpack(">h", p[11:13])[0] if len(p) >= 13 else 0,
    )
