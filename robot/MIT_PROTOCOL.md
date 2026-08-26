# CubeMars AK Series MIT Mode (Force Control) Protocol

**Source:** CubeMars AK Series Module Product Manual v3.0.0, §4.2 (tested and verified 2026-08-26 on AKE90-8 and AK60-39)

**This firmware revision is NOT the classic MIT protocol.** There are no ENTER/EXIT/ZERO magic frames or standard-ID addressing. Force control is tunnelled as one more VESC-style extended-ID servo command, **always live alongside servo mode** — no mode switch required.

## Quick Start

```python
import struct, can

bus = can.interface.Bus(channel="can1", interface="socketcan", bitrate=1_000_000)

# Ranges for AKE80-8 (closest published sibling to your AKE90-8)
P_MIN, P_MAX = -12.56, 12.56      # rad, +/- 2 pi
V_MIN, V_MAX = -20.0, 20.0        # rad/s
KP_MIN, KP_MAX = 0.0, 500.0       # Nm/rad
KD_MIN, KD_MAX = 0.0, 5.0         # Nm*s/rad
T_MIN, T_MAX = -35.0, 35.0        # Nm (AKE80-8; yours unidentified)

def float_to_uint(x, x_min, x_max, bits):
    x = max(x_min, min(x_max, x))
    return int((x - x_min) * ((1 << bits) / (x_max - x_min)))

def pack_mit(pos, vel, kp, kd, torque):
    """Pack a MIT command into 8 bytes (extended-ID frame format)."""
    P = float_to_uint(pos, P_MIN, P_MAX, 16)
    V = float_to_uint(vel, V_MIN, V_MAX, 12)
    KP = float_to_uint(kp, KP_MIN, KP_MAX, 12)
    KD = float_to_uint(kd, KD_MIN, KD_MAX, 12)
    T = float_to_uint(torque, T_MIN, T_MAX, 12)
    
    return bytes([
        KP >> 4,                          # byte 0: KP high 8 bits
        ((KP & 0xF) << 4) | (KD >> 8),   # byte 1: KP low 4 + KD high 4
        KD & 0xFF,                        # byte 2: KD low 8
        P >> 8,                           # byte 3: position high 8
        P & 0xFF,                         # byte 4: position low 8
        V >> 4,                           # byte 5: velocity high 8
        ((V & 0xF) << 4) | (T >> 8),     # byte 6: velocity low 4 + torque high 4
        T & 0xFF                          # byte 7: torque low 8
    ])

# Command the motor: hold at current position with 1.0 Nm*s/rad damping, no position target
node_id = 106  # CAN node
data = pack_mit(pos=0, vel=0, kp=0, kd=1.0, torque=0)
msg = can.Message(arbitration_id=node_id | (8 << 8), data=data, is_extended_id=True)
bus.send(msg)

# Read feedback (same 0x29 status frame as servo mode, timed upload configurable 1-500 Hz)
m = bus.recv(timeout=0.01)
if m is not None and m.arbitration_id == (0x29 << 8) | node_id:
    pos = struct.unpack(">h", m.data[0:2])[0] / 10.0        # degrees
    spd = struct.unpack(">h", m.data[2:4])[0] * 10.0        # ERPM
    cur = struct.unpack(">h", m.data[4:6])[0] / 100.0       # Amps
    temp = m.data[6]                                         # Celsius
    err = m.data[7]                                          # error code
```

---

## CAN Frame Format

### Command (Master -> Drive)

| Field | Value |
|---|---|
| **CAN ID** | Extended (29-bit): bits[28:8] = 8 (control mode ID), bits[7:0] = node_id |
| **Example** | Node 106 -> arbitration_id = (8 << 8) \| 106 = 0x86A |
| **Frame Type** | Extended frame |
| **DLC** | 8 bytes (must be full, no padding) |

### Byte Packing (8 bytes)

| Byte | Bits | Field | Meaning |
|---|---|---|---|
| 0 | 7-0 | KP[15:8] | Kp high 8 bits |
| 1 | 7-4 | KP[3:0] | Kp low 4 bits |
| 1 | 3-0 | KD[11:8] | Kd high 4 bits |
| 2 | 7-0 | KD[7:0] | Kd low 8 bits |
| 3 | 7-0 | P[15:8] | Position high 8 bits |
| 4 | 7-0 | P[7:0] | Position low 8 bits |
| 5 | 7-0 | V[11:4] | Velocity high 8 bits |
| 6 | 7-4 | V[3:0] | Velocity low 4 bits |
| 6 | 3-0 | T[11:8] | Torque high 4 bits |
| 7 | 7-0 | T[7:0] | Torque low 8 bits |

**Total:** KP 12-bit, KD 12-bit, position 16-bit, velocity 12-bit, torque 12-bit.

### Feedback (Drive -> Master)

Same as servo mode: extended-ID cmd 41 (0x29), timed upload 1-500 Hz.

| Byte | Field | Conversion |
|---|---|---|
| 0-1 | Position (int16 BE) | pos_deg = value / 10.0; to rad: / 57.3 |
| 2-3 | Speed (int16 BE) | speed_rpm = value * 10.0; to rad/s: * 2*pi/60 / 9.55 |
| 4-5 | Current (int16 BE) | current_A = value / 100.0 |
| 6 | Temperature | temp_C = value |
| 7 | Error code | 0 = OK |

---

## Parameter Ranges

**Position:** +/- 12.56 rad for ALL models (fixed).

**Per-model table** (CubeMars AK Series Manual v3.0.0 p39):

| Model | V_MAX (rad/s) | T_MAX (Nm) |
|---|---|---|
| AK10-9 | 28 | 54 |
| AK60-6 | 60 | 12 |
| AK70-9 | 30 | 32 |
| AK80-9 | 65 | 18 |
| AKE60-8 | 40 | 15 |
| AKE80-8 | 20 | 35 |
| **AKE90-8** | **UNIDENTIFIED** | **UNIDENTIFIED** |
| **AK60-39** | **UNIDENTIFIED** | **UNIDENTIFIED** |

WARNING: DASH-01's motors are not in the table. Use `robot/tools/mit_identify.py --mode osc` to measure ranges before trusting absolute velocity/torque commands.

**Gains (all models):**
- Kp: 0-500 Nm/rad
- Kd: 0-5 Nm*s/rad

---

## Control Modes

The MIT frame carries five parameters. Behaviour is determined by which are nonzero:

### Position Loop (Kp > 0)

Closed-loop: tau = Kp*(pos_des - pos) + Kd*(vel_des - vel) + tau_ff

**WARNING:** If P_range is wrong, the motor will seek a far-away position. With high Kp, this draws instant high current into a stalled rotor. **Always send kp=0 until ranges are measured** — zero stiffness makes wrong P mapping inert.

### Velocity Loop (Kd > 0, Kp = 0)

tau = Kd*(vel_des - vel) + tau_ff. Velocity reference without position target.

### Torque (Tau_ff > 0, Kp = Kd = 0)

Direct torque: tau = tau_ff. On a free rotor, accelerates without bound.

### Impedance (Kp > 0, Kd > 0)

Full impedance: tau = Kp*(pos_des - pos) + Kd*(vel_des - vel) + tau_ff

---

## Key Properties

### Transport Delay

**~6-7 ms at 200 Hz command rate** (1.3 command periods). Inertia-independent, survives leg attachment. Servo mode was ~200 ms; MIT is **28x faster**.

### Feedback Rate

Poll-only drives do NOT free-run. Commanding a MIT frame at 200 Hz elicits feedback reply 1:1, giving ~202 Hz feedback. No separate config needed.

In MIT mode, **always poll for feedback** (it's free, you're sending frames anyway) rather than relying on broadcast.

### Safety

- Hardware phase/bus current limits are hard stops. Use Kd as your safety valve, never the drive limit.
- The **62 A accident (2026-08-26):** one frame with kp=195 and pos=-12.56 (wrong range) commanded full stiffness against max extension. Always send kp=0 until ranges measured.
- Ramp torque gradually to avoid shocks.

---

## Debugging

### Motor Does Not Respond

1. Check node_id matches (default 1).
2. Verify frame is extended-ID, not standard.
3. Command byte is 8 (bits[28:8] of arbitration_id).
4. Motor is powered, error state is 0.
5. Try a safe kp=0 frame: `data = bytes([0]*8)`
6. Periodic feedback enabled on drive (via CubeMarsTool).

### Movement Is Wrong

- Position is 2x larger: P_range is halved. Identify ranges.
- Motor stalls: Kp too high against wrong P_range. Send kp=0.
- Motor overshoots: Kd too low. Increase kd (max 5.0).
- No motion: pos is clamped to +/-12.56 rad.

### Bus Saturated

- 200 Hz, 6 motors = 1200 frames/s = 15% load at 1 Mbit.
- Adding broadcast = 45% total. Solution: poll instead.
- 500 Hz on 6 motors = 40% load (acceptable).

---

## References

- CubeMars AK Series Module Product Manual v3.0.0, Section 4.2, pages 38-42
- MIT_BANDWIDTH_MEASURED memory: 28x lag reduction, 6-7 ms transport delay
- CAN_CMD8_DANGER memory: safety lessons from the 62 A incident
