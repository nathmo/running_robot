"""CubeMars force-control ("MIT mode") frames for the AK drives on DASH-01.

READ THIS BEFORE CHANGING ANYTHING HERE
---------------------------------------
CubeMars does NOT implement the classic MIT protocol. There are no ENTER/EXIT/ZERO magic frames
(FF..FC/FD/FE) and no standard-id addressing. Force control is tunnelled as one more VESC-style
EXTENDED-id command, documented in the AK Series Module Product Manual v3.0.0 section 4.2, and it
is live alongside servo mode with no reflash and no mode switch (proven on this robot 2026-08-26:
0.30 N*m commanded, 0.72 A drawn, on the free left thigh).

    arbitration id (EXTENDED, DLC 8):  bits[28:8] = 8 (force control),  bits[7:0] = drive id
    byte order:  KP, KD, POSITION, VELOCITY, TORQUE      <- kp FIRST, not position-first
    widths:      P 16 bit, V/KP/KD/T 12 bit
    scaling:     float_to_uint uses (1 << bits) / span, NOT ((1 << bits) - 1) / span

Every mit_*.py, ak_position_sweep.py and can_scan_motors.py in this repo implements the CLASSIC
protocol instead (standard id, ENTER/EXIT, position-first). This firmware ignores those frames
entirely, which is why every earlier MIT attempt "failed". Do not copy from them.

THE THREE RULES THIS MODULE ENFORCES IN CODE
--------------------------------------------
1. DLC IS ALWAYS 8. On 2026-08-26 a three-byte frame to id 0x86A drew 62.5 A into a stalled rotor
   and was stopped only by a speed watchdog. The missing bytes are not "unset" -- they read as
   ZERO, and zero in the position field is the range MINIMUM (-12.56 rad), which at the kp the
   first byte happened to encode was a -719 degree target at 195 N*m/rad. `pack()` cannot produce
   a short frame; there is no code path to one.

2. VELOCITY AND TORQUE ARE ZERO unless their ranges have been identified. The V and T spans are
   per-model firmware constants and the AKE90-8 and AK60-39 are NOT in the manual's table -- so
   ours are unknown, and a wrong span silently rescales the value with no error anywhere.
   Zero is the one value that is immune: with the (1<<bits)/span convention, 0 encodes to exactly
   the mid-code and decodes to exactly 0.0 for ANY span. So commanding v_des = 0 and tau_ff = 0
   makes the unidentified constants irrelevant, and what the drive runs is
       tau = kp * (p_des - p) - kd * v
   which is precisely the MuJoCo position actuator the policy was trained against
   (gainprm[0] = kp, biasprm[2] = -kv). Passing a nonzero velocity or torque requires passing the
   identified range explicitly, so it cannot happen by default or by accident.

3. POSITION, KP AND KD ARE CLAMPED to the documented spans before encoding, and the clamp is
   REPORTED. A silently saturated command is a control law that is not the one that was verified.
   (For the deployed bundle the impedance channel spans kp 40-500 and kd 1.0-5.0, which fits the
   0-500 / 0-5 wire ranges exactly -- so a clamp here means something upstream changed.)

The FEEDBACK path is unchanged by force control: the drives keep sending the same 0x29 servo
status frame, so `canio.parse_status` is still the decoder. Only the command changes.
"""
import struct

# --- documented spans (manual v3.0.0 p39) -------------------------------------------------------
# Position is +-12.56 rad for EVERY model; Kp 0-500 N*m/rad and Kd 0-5 N*m*s/rad likewise. These
# three are safe to rely on. V and T are per-model and ours are not in the table.
P_MIN, P_MAX = -12.56, 12.56
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0

CMD_FORCE_CONTROL = 8            # the control-mode id in bits[28:8] of the extended arbitration id
DLC = 8


class RangeNotIdentified(Exception):
    """Raised on any attempt to send a nonzero velocity or torque without its measured span."""


def arbitration_id(node_id):
    """Extended CAN id for a force-control frame to `node_id`. Node 106 -> 0x86A."""
    if not 0 <= int(node_id) <= 0xFF:
        raise ValueError("node id {} is not a byte".format(node_id))
    return (CMD_FORCE_CONTROL << 8) | int(node_id)


def float_to_uint(x, lo, hi, bits):
    """CubeMars scaling. Note (1 << bits), not ((1 << bits) - 1) -- the classic MIT code in this
    repo uses the other one, and mixing them puts a half-LSB bias on every field."""
    span = hi - lo
    x = lo if x < lo else (hi if x > hi else x)
    v = int((x - lo) * ((1 << bits) / span))
    return min(v, (1 << bits) - 1)


def uint_to_float(v, lo, hi, bits):
    return v * ((hi - lo) / (1 << bits)) + lo


def _clamp(v, lo, hi):
    return (lo if v < lo else (hi if v > hi else v)), (v < lo or v > hi)


def pack(p_des, kp, kd, v_des=0.0, tau_ff=0.0, v_range=None, t_range=None):
    """Build the 8 payload bytes. Returns (bytes, clamped_fields).

    p_des  rad, drive origin (the same origin the 0x29 status frame reports)
    kp     N*m/rad at the OUTPUT shaft
    kd     N*m*s/rad at the output shaft
    v_des  rad/s -- must be 0.0 unless v_range=(lo, hi) has been MEASURED for this motor
    tau_ff N*m   -- must be 0.0 unless t_range=(lo, hi) has been MEASURED for this motor
    """
    if v_des != 0.0 and v_range is None:
        raise RangeNotIdentified(
            "v_des={} but the velocity span of this drive has not been identified. Zero is the "
            "only velocity that encodes correctly under an unknown span; pass the measured "
            "v_range=(lo, hi) if you have one.".format(v_des))
    if tau_ff != 0.0 and t_range is None:
        raise RangeNotIdentified(
            "tau_ff={} but the torque span of this drive has not been identified. See the module "
            "docstring -- a wrong span rescales the torque with no error reported anywhere."
            .format(tau_ff))
    clamped = []
    p, c = _clamp(float(p_des), P_MIN, P_MAX)
    if c:
        clamped.append("position")
    k, c = _clamp(float(kp), KP_MIN, KP_MAX)
    if c:
        clamped.append("kp")
    d, c = _clamp(float(kd), KD_MIN, KD_MAX)
    if c:
        clamped.append("kd")
    vlo, vhi = v_range if v_range else (-1.0, 1.0)
    tlo, thi = t_range if t_range else (-1.0, 1.0)

    kpi = float_to_uint(k, KP_MIN, KP_MAX, 12)
    kdi = float_to_uint(d, KD_MIN, KD_MAX, 12)
    pi = float_to_uint(p, P_MIN, P_MAX, 16)
    vi = float_to_uint(float(v_des), vlo, vhi, 12)
    ti = float_to_uint(float(tau_ff), tlo, thi, 12)

    buf = bytes([
        (kpi >> 4) & 0xFF,
        ((kpi & 0xF) << 4) | ((kdi >> 8) & 0xF),
        kdi & 0xFF,
        (pi >> 8) & 0xFF,
        pi & 0xFF,
        (vi >> 4) & 0xFF,
        ((vi & 0xF) << 4) | ((ti >> 8) & 0xF),
        ti & 0xFF,
    ])
    assert len(buf) == DLC, "force-control frames are always 8 bytes"
    return buf, clamped


def unpack(buf, v_range=None, t_range=None):
    """Decode a command payload back to engineering units. For tests and for the black box --
    a recorded frame must be readable without re-deriving the bit layout by hand."""
    if len(buf) != DLC:
        raise ValueError("force-control payload must be exactly {} bytes, got {}".format(
            DLC, len(buf)))
    b = bytes(buf)
    kpi = (b[0] << 4) | (b[1] >> 4)
    kdi = ((b[1] & 0xF) << 8) | b[2]
    pi = (b[3] << 8) | b[4]
    vi = (b[5] << 4) | (b[6] >> 4)
    ti = ((b[6] & 0xF) << 8) | b[7]
    vlo, vhi = v_range if v_range else (-1.0, 1.0)
    tlo, thi = t_range if t_range else (-1.0, 1.0)
    return {"kp": uint_to_float(kpi, KP_MIN, KP_MAX, 12),
            "kd": uint_to_float(kdi, KD_MIN, KD_MAX, 12),
            "p_des": uint_to_float(pi, P_MIN, P_MAX, 16),
            "v_des": uint_to_float(vi, vlo, vhi, 12),
            "tau_ff": uint_to_float(ti, tlo, thi, 12)}


def limp_payload():
    """kp = kd = 0, everything else zero: a VALID, fully-formed frame that commands nothing.

    Measured on the robot: 0.07 A, no motion. This is the force-control equivalent of streaming
    SET_CURRENT 0, and like it, it must be STREAMED -- the drive holds its last command, so
    "stop sending" is not "stop commanding"."""
    return pack(0.0, 0.0, 0.0)[0]


def resolution():
    """LSB size of each field, for the record. Position quantisation is 0.022 deg and kp is
    0.12 N*m/rad -- both far below anything the mechanism can resolve, which is worth knowing
    before someone blames a wobble on the wire format."""
    return {"position_rad": (P_MAX - P_MIN) / (1 << 16),
            "kp_nm_rad": (KP_MAX - KP_MIN) / (1 << 12),
            "kd_nm_s_rad": (KD_MAX - KD_MIN) / (1 << 12)}
