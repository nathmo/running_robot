#!/usr/bin/env python3
"""Waveshare **Sense HAT (B)** driver for the DASH-01 web UI (I2C bus 1 on the Pi).

Five chips, all confirmed present on the robot (`i2cdetect` + WHO_AM_I):

    0x68  ICM-20948   9-DOF IMU: accel + gyro (+ AK09916 magnetometer at 0x0C via bypass)
    0x70  SHTC3       air temperature + relative humidity
    0x5C  LPS22HB     barometric pressure + its own die temperature
    0x29  TCS34725    RGB colour / ambient light  -> lux + correlated colour temperature
    0x48  ADS1015     4-channel 12-bit ADC on the HAT's external analog header

Everything here is register-level over `smbus2`; no Waveshare vendor code and no GPIO. The chips
are read by ONE background thread (`SenseHat`), never by the Flask handlers and never by the
200 Hz CAN daemon — an I2C stall must not be able to delay a motor command. Handlers only read the
published snapshot / ring buffer, exactly like motor telemetry.

Degrades instead of failing: a missing `smbus2`, a missing /dev/i2c-1 or an unplugged HAT leaves
`available = False` with the reason in `error`, and the rest of the web UI is unaffected.

Frames. The AK09916 sits in a rotated frame inside the ICM-20948 package and is remapped here so
all nine axes agree in the chip frame. Getting from there to the ROBOT's axes is a mounting
question, and on this robot the HAT is bolted underneath (it reads gravity on chip -Z), so the two
frames are nowhere near each other. `mountcal.py` measures that rotation from two captures instead
of asserting it; until it has been run, values are published in the raw chip frame and the panel
says so.

Attitude comes from a 6-axis Madgwick filter (accel + gyro) fed the BODY-frame signals. Roll/pitch
are gravity-referenced and absolute; YAW IS GYRO-INTEGRATED AND DRIFTS — the magnetometer
deliberately does not correct it, because the mag sits centimetres from two brushless motors and a
battery, so its heading is only published as an advisory number, never fused into the attitude.

The IMU is also offset from the base centre, so while the robot rotates it measures the lever-arm
terms on top of gravity; when a lever arm is known (CAD or fitted, see mountcal) those are
subtracted before the filter sees the accelerometer.
"""
import math
import threading
import time

import numpy as np

import mountcal
import paths  # noqa: F401  — path bootstrap; every webui module imports it first
from ringbuffer import ScalarRing

try:
    import smbus2
except ImportError:                                     # dev machines / the Pi before pip install
    smbus2 = None

I2C_BUS = 1
IMU_HZ = 100.0          # AHRS update rate (also the I2C read rate for accel/gyro)
LOG_HZ = 20.0           # ring-buffer/chart rate, matched to the motor telemetry stream
ENV_PERIOD_S = 1.0      # SHTC3 + LPS22HB (slow, and self-heating if hammered)
COLOR_PERIOD_S = 0.5    # TCS34725 (its integration time is 154 ms)
ADC_PERIOD_S = 0.2      # ADS1015, one channel per tick -> all four at 1.25 Hz

OMEGA_TAU_S = 0.02      # low-pass on the rate gyro before differentiating it for angular accel

RING_FIELDS = ("ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz",
               "roll", "pitch", "yaw", "heading", "acc_mag", "gyro_mag",
               "temp", "humidity", "pressure", "temp_baro", "temp_imu",
               "lux", "cct", "adc0", "adc1", "adc2", "adc3")


def _clean(v):
    """JSON-safe scalar: NaN/inf become None (the UI draws a gap, not a bogus number). Lists (the
    raw RGB triple) pass through element-wise."""
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if v is None:
        return None
    v = float(v)
    return None if not math.isfinite(v) else round(v, 4)


# ============================================================================ ICM-20948 (IMU)
class ICM20948:
    """9-DOF IMU. Bank-switched register file: BANK_SEL (0x7F) picks which of the four 128-byte
    banks the 0x00-0x7E window maps to, so every access states its bank."""
    ADDR = 0x68
    WHO_AM_I = 0xEA
    MAG_ADDR = 0x0C                 # AK09916, reachable once I2C bypass is enabled
    MAG_WHO_AM_I = 0x09
    MAG_UT_PER_LSB = 0.15           # datasheet: 0.15 uT/LSB, fixed (no range setting)

    ACC_LSB_PER_G = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
    GYR_LSB_PER_DPS = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}

    # A legged robot slams the foot down (accel spikes well past 2 g) and swings the shank fast
    # (hundreds of deg/s), so the default ranges are wide enough not to clip during a gait.
    ACC_RANGE_G = 4
    GYR_RANGE_DPS = 1000

    def __init__(self, bus):
        self.bus = bus
        self._bank = None
        self.mag_ok = False
        self.acc_scale = 1.0 / self.ACC_LSB_PER_G[self.ACC_RANGE_G]
        self.gyr_scale = 1.0 / self.GYR_LSB_PER_DPS[self.GYR_RANGE_DPS]

    # ---- register plumbing
    def _bank_sel(self, bank):
        if self._bank != bank:
            self.bus.write_byte_data(self.ADDR, 0x7F, bank << 4)
            self._bank = bank

    def _w(self, bank, reg, val):
        self._bank_sel(bank)
        self.bus.write_byte_data(self.ADDR, reg, val)

    def _r(self, bank, reg, n=1):
        self._bank_sel(bank)
        return self.bus.read_i2c_block_data(self.ADDR, reg, n)

    def init(self):
        who = self._r(0, 0x00)[0]
        if who != self.WHO_AM_I:
            raise RuntimeError(f"ICM-20948 WHO_AM_I 0x{who:02X}, expected 0x{self.WHO_AM_I:02X}")
        self._w(0, 0x06, 0x80)                  # PWR_MGMT_1: device reset
        time.sleep(0.05)
        self._bank = None                       # reset also resets the bank pointer
        self._w(0, 0x06, 0x01)                  # wake, auto clock source
        time.sleep(0.02)
        self._w(0, 0x07, 0x00)                  # PWR_MGMT_2: accel + gyro all axes on

        # Sample rate 1.125 kHz / (1 + div) with the low-pass filters engaged (FCHOICE = 1). div=8
        # -> 125 Hz, comfortably above our 100 Hz read rate so we never read the same sample twice.
        fs_g = {250: 0, 500: 1, 1000: 2, 2000: 3}[self.GYR_RANGE_DPS]
        fs_a = {2: 0, 4: 1, 8: 2, 16: 3}[self.ACC_RANGE_G]
        self._w(2, 0x00, 8)                     # GYRO_SMPLRT_DIV
        self._w(2, 0x01, (3 << 3) | (fs_g << 1) | 1)   # GYRO_CONFIG_1: DLPF 3 (~51 Hz)
        self._w(2, 0x10, 0)                     # ACCEL_SMPLRT_DIV_1 (high byte)
        self._w(2, 0x11, 8)                     # ACCEL_SMPLRT_DIV_2 (low byte)
        self._w(2, 0x14, (3 << 3) | (fs_a << 1) | 1)   # ACCEL_CONFIG: DLPF 3 (~50 Hz)

        # Expose the AK09916 directly on the Pi's bus (bypass) instead of running the ICM's own I2C
        # master: one fewer moving part, and the mag is a low-rate advisory signal here anyway.
        self._w(0, 0x03, 0x00)                  # USER_CTRL: I2C master off
        self._w(0, 0x0F, 0x02)                  # INT_PIN_CFG: BYPASS_EN
        time.sleep(0.01)
        self._init_mag()

    def _init_mag(self):
        try:
            if self.bus.read_i2c_block_data(self.MAG_ADDR, 0x01, 1)[0] != self.MAG_WHO_AM_I:
                return
            self.bus.write_byte_data(self.MAG_ADDR, 0x32, 0x01)      # CNTL3: soft reset
            time.sleep(0.02)
            self.bus.write_byte_data(self.MAG_ADDR, 0x31, 0x08)      # CNTL2: continuous mode 4 (100 Hz)
            time.sleep(0.01)
            self.mag_ok = True
        except OSError:
            self.mag_ok = False

    # ---- reads
    @staticmethod
    def _s16(hi, lo):
        v = (hi << 8) | lo
        return v - 65536 if v & 0x8000 else v

    def read_motion(self):
        """One 14-byte burst: accel (g), gyro (deg/s), die temperature (degC)."""
        d = self._r(0, 0x2D, 14)
        acc = [self._s16(d[i], d[i + 1]) * self.acc_scale for i in (0, 2, 4)]
        gyr = [self._s16(d[i], d[i + 1]) * self.gyr_scale for i in (6, 8, 10)]
        temp = self._s16(d[12], d[13]) / 333.87 + 21.0
        return acc, gyr, temp

    def read_mag(self):
        """Magnetometer in uT, rotated into the accel/gyro frame, or None if not ready.

        The AK09916 die is mounted rotated inside the ICM-20948 package: its (X, Y, Z) correspond
        to the accel/gyro (Y, X, -Z). ST2 must be read to release the data registers, and its HOFL
        bit marks a sample that saturated the sensor (near a motor, that happens)."""
        if not self.mag_ok:
            return None
        try:
            if not (self.bus.read_i2c_block_data(self.MAG_ADDR, 0x10, 1)[0] & 0x01):   # ST1.DRDY
                return None
            d = self.bus.read_i2c_block_data(self.MAG_ADDR, 0x11, 8)                   # HXL..ST2
            if d[7] & 0x08:                                                            # ST2.HOFL
                return None
            mx = self._s16(d[1], d[0]) * self.MAG_UT_PER_LSB      # little-endian, unlike the ICM
            my = self._s16(d[3], d[2]) * self.MAG_UT_PER_LSB
            mz = self._s16(d[5], d[4]) * self.MAG_UT_PER_LSB
            return [my, mx, -mz]
        except OSError:
            return None


# ============================================================================ SHTC3 (T + RH)
class SHTC3:
    """Temperature + humidity. Command-word device (no register file); sleeps between reads to keep
    self-heating out of the temperature it is supposed to be measuring."""
    ADDR = 0x70

    def __init__(self, bus):
        self.bus = bus

    def _cmd(self, word):
        self.bus.i2c_rdwr(smbus2.i2c_msg.write(self.ADDR, [word >> 8, word & 0xFF]))

    def _read(self, n):
        msg = smbus2.i2c_msg.read(self.ADDR, n)
        self.bus.i2c_rdwr(msg)
        return list(msg)

    @staticmethod
    def _crc_ok(msb, lsb, crc):
        c = 0xFF
        for b in (msb, lsb):
            c ^= b
            for _ in range(8):
                c = ((c << 1) ^ 0x31) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        return c == crc

    def init(self):
        self._cmd(0x3517)                       # wake
        time.sleep(0.001)
        self._cmd(0xEFC8)                       # read ID
        time.sleep(0.005)
        d = self._read(3)
        self._cmd(0xB098)                       # sleep
        if ((d[0] << 8) | d[1]) & 0x083F != 0x0807:
            raise RuntimeError(f"SHTC3 ID 0x{(d[0] << 8) | d[1]:04X} unexpected")

    def read(self):
        """(temperature degC, relative humidity %) — normal-mode, clock stretching disabled."""
        self._cmd(0x3517)                       # wake
        time.sleep(0.001)
        self._cmd(0x7866)                       # measure T first, then RH
        time.sleep(0.015)                       # max conversion time is 12.1 ms
        d = self._read(6)
        self._cmd(0xB098)                       # sleep
        if not (self._crc_ok(d[0], d[1], d[2]) and self._crc_ok(d[3], d[4], d[5])):
            return None
        traw = (d[0] << 8) | d[1]
        hraw = (d[3] << 8) | d[4]
        return -45.0 + 175.0 * traw / 65536.0, 100.0 * hraw / 65536.0


# ============================================================================ LPS22HB (pressure)
class LPS22HB:
    """Barometer, read one-shot.

    IF_ADD_INC (CTRL_REG2 bit 4) must stay set in every write to that register: it is what makes
    the address pointer walk 0x28..0x2C during the block read. Clearing it while triggering
    ONE_SHOT re-reads PRESS_OUT_XL five times instead, which silently yields a *constant* pressure
    (0x333333 / 4096 = 819.2 hPa) that looks like a plausible reading."""
    ADDR = 0x5C
    WHO_AM_I = 0xB1
    ADDR_INC = 0x10

    def __init__(self, bus):
        self.bus = bus

    def init(self):
        who = self.bus.read_byte_data(self.ADDR, 0x0F)
        if who != self.WHO_AM_I:
            raise RuntimeError(f"LPS22HB WHO_AM_I 0x{who:02X}, expected 0x{self.WHO_AM_I:02X}")
        self.bus.write_byte_data(self.ADDR, 0x11, self.ADDR_INC)   # CTRL_REG2: auto-increment on
        time.sleep(0.01)
        self.bus.write_byte_data(self.ADDR, 0x10, 0x02)      # CTRL_REG1: power-down/one-shot, BDU on

    def read(self):
        """(pressure hPa, die temperature degC) via a one-shot conversion."""
        self.bus.write_byte_data(self.ADDR, 0x11, self.ADDR_INC | 0x01)     # ONE_SHOT
        # BOTH data-available bits, not either: pressure completes first, and settling for P_DA
        # alone reads the temperature registers before the conversion has written them (the first
        # one-shot after power-up then reports 0.0 degC and a nonsense pressure).
        for _ in range(30):                                  # ~2 ms each; the conversion is faster
            if self.bus.read_byte_data(self.ADDR, 0x27) & 0x03 == 0x03:   # STATUS: P_DA and T_DA
                break
            time.sleep(0.002)
        else:
            return None
        d = self.bus.read_i2c_block_data(self.ADDR, 0x28, 5)      # PRESS_XL..TEMP_H
        praw = (d[2] << 16) | (d[1] << 8) | d[0]
        traw = (d[4] << 8) | d[3]
        if traw & 0x8000:
            traw -= 65536
        return praw / 4096.0, traw / 100.0


# ============================================================================ TCS34725 (light)
class TCS34725:
    """RGBC light sensor. Every register access sets the command bit (0x80)."""
    ADDR = 0x29
    ATIME = 0xC0            # 154 ms integration -> 64 cycles, full-scale count 65535
    GAIN_CODE = 1           # indoor light levels, not sunlight
    _GAIN_X = {0: 1, 1: 4, 2: 16, 3: 60}

    def __init__(self, bus):
        self.bus = bus
        self.integ_ms = 2.4 * (256 - self.ATIME)

    def _w(self, reg, val):
        self.bus.write_byte_data(self.ADDR, 0x80 | reg, val)

    def _r16(self, reg):
        d = self.bus.read_i2c_block_data(self.ADDR, 0x80 | reg, 2)
        return (d[1] << 8) | d[0]

    def init(self):
        who = self.bus.read_byte_data(self.ADDR, 0x80 | 0x12)
        if who not in (0x44, 0x4D):
            raise RuntimeError(f"TCS34725 ID 0x{who:02X}, expected 0x44/0x4D")
        self._w(0x01, self.ATIME)
        self._w(0x0F, self.GAIN_CODE)           # CONTROL: gain
        self._w(0x00, 0x01)                     # ENABLE: power on
        time.sleep(0.003)
        self._w(0x00, 0x03)                     # ENABLE: power on + RGBC enable

    def read(self):
        """(clear, r, g, b, lux, colour temperature K) — lux/CCT per the AMS DN40 app note."""
        if not (self.bus.read_byte_data(self.ADDR, 0x80 | 0x13) & 0x01):    # STATUS.AVALID
            return None
        c = self._r16(0x14)
        r = self._r16(0x16)
        g = self._r16(0x18)
        b = self._r16(0x1A)
        # DN40: strip the infrared component the silicon sees but the eye does not, then weight the
        # corrected channels by the eye's response and normalise by integration time and gain.
        ir = max(0.0, (r + g + b - c) / 2.0)
        rc, gc, bc = r - ir, g - ir, b - ir
        cpl = (self.integ_ms * self._GAIN_X[self.GAIN_CODE]) / 310.0    # counts per lux
        lux = max(0.0, (0.136 * rc + 1.0 * gc - 0.444 * bc) / cpl) if cpl else 0.0
        cct = 3810.0 * (bc / rc) + 1391.0 if rc > 0 else None           # DN40 approximation
        return c, r, g, b, lux, cct


# ============================================================================ ADS1015 (ADC)
class ADS1015:
    """4-channel 12-bit ADC on the HAT's analog header, read single-shot one channel at a time."""
    ADDR = 0x48
    FS_V = 4.096            # PGA = +/-4.096 V, i.e. 2 mV per LSB over 12 bits

    def __init__(self, bus):
        self.bus = bus

    def init(self):
        self.bus.read_i2c_block_data(self.ADDR, 0x01, 2)    # config register must be readable

    def read_channel(self, ch):
        """Volts on AIN<ch>, single-ended against GND."""
        cfg = (0x8000                       # OS: start a single conversion
               | ((4 + ch) << 12)           # MUX: AIN<ch> single-ended
               | (0x1 << 9)                 # PGA: +/-4.096 V
               | 0x0100                     # MODE: single-shot
               | (0x4 << 5)                 # DR: 1600 SPS
               | 0x0003)                    # COMP_QUE: comparator disabled
        self.bus.write_i2c_block_data(self.ADDR, 0x01, [cfg >> 8, cfg & 0xFF])
        time.sleep(0.002)                   # 1600 SPS -> 0.63 ms; round up for the bus
        d = self.bus.read_i2c_block_data(self.ADDR, 0x00, 2)
        raw = ((d[0] << 8) | d[1]) >> 4     # 12 bits, left-aligned in the 16-bit register
        if raw & 0x800:
            raw -= 4096
        return raw * self.FS_V / 2048.0


# ============================================================================ attitude
class Madgwick:
    """6-axis (accel + gyro) Madgwick gradient-descent filter.

    Accel-only correction, so roll and pitch are absolute (gravity-referenced) while yaw is pure
    gyro integration and drifts — see the module docstring for why the magnetometer is kept out."""

    def __init__(self, beta=0.06):
        self.beta = beta                    # correction gain: bigger = trusts accel over gyro more
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def reset(self, acc=None):
        """Restart from the attitude the accelerometer implies (yaw arbitrarily 0), so the filter
        does not have to converge from level after a bias calibration."""
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        if acc is None:
            return
        ax, ay, az = acc
        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n < 1e-6:
            return
        ax, ay, az = ax / n, ay / n, az / n
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        self.q = np.array([cr * cp, sr * cp, cr * sp, -sr * sp])

    def update(self, gyro_dps, acc_g, dt):
        q0, q1, q2, q3 = self.q
        gx, gy, gz = (math.radians(v) for v in gyro_dps)

        # quaternion rate from the rate gyro
        qd = 0.5 * np.array([-q1 * gx - q2 * gy - q3 * gz,
                             q0 * gx + q2 * gz - q3 * gy,
                             q0 * gy - q1 * gz + q3 * gx,
                             q0 * gz + q1 * gy - q2 * gx])

        n = math.sqrt(sum(v * v for v in acc_g))
        if n > 1e-6:
            ax, ay, az = (v / n for v in acc_g)
            # gradient of |predicted gravity - measured gravity|^2 w.r.t. the quaternion
            f = np.array([2 * (q1 * q3 - q0 * q2) - ax,
                          2 * (q0 * q1 + q2 * q3) - ay,
                          2 * (0.5 - q1 * q1 - q2 * q2) - az])
            j = np.array([[-2 * q2, 2 * q3, -2 * q0, 2 * q1],
                          [2 * q1, 2 * q0, 2 * q3, 2 * q2],
                          [0.0, -4 * q1, -4 * q2, 0.0]])
            step = j.T @ f
            sn = np.linalg.norm(step)
            if sn > 1e-9:
                qd -= self.beta * step / sn

        q = self.q + qd * dt
        self.q = q / np.linalg.norm(q)
        return self.rpy()

    def up_body(self):
        """The world's UP direction expressed in body axes — what a still accelerometer reads.
        Gravity is the negative of this, scaled by g."""
        q0, q1, q2, q3 = self.q
        return np.array([2 * (q1 * q3 - q0 * q2),
                         2 * (q0 * q1 + q2 * q3),
                         1 - 2 * (q1 * q1 + q2 * q2)])

    def rpy(self):
        """(roll, pitch, yaw) in degrees, aerospace Z-Y-X convention."""
        q0, q1, q2, q3 = self.q
        roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        s = max(-1.0, min(1.0, 2 * (q0 * q2 - q3 * q1)))
        pitch = math.asin(s)
        yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def tilt_compensated_heading(mag, roll_deg, pitch_deg):
    """Magnetic heading in degrees (0 = +X axis pointing at magnetic north), de-rotated by the
    measured roll/pitch. Advisory only: uncalibrated for hard/soft iron, and the motors move it."""
    if mag is None:
        return None
    mx, my, mz = mag
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    xh = mx * math.cos(p) + mz * math.sin(p)
    yh = (mx * math.sin(r) * math.sin(p) + my * math.cos(r) - mz * math.sin(r) * math.cos(p))
    return math.degrees(math.atan2(-yh, xh)) % 360.0


# ============================================================================ simulated IMU
# A known mount and a known lever arm, so the calibration procedure can be walked end to end off
# the robot and checked against the answer it is supposed to find. The rotation below is "HAT
# bolted underneath, turned 90 deg" — chip axes nowhere near body axes, like the real one.
MOCK_R = np.array([[0.0, 1.0, 0.0],          # body X (fwd)  = chip +Y
                   [1.0, 0.0, 0.0],          # body Y (left) = chip +X
                   [0.0, 0.0, -1.0]])        # body Z (up)   = chip -Z
MOCK_LEVER = np.array([0.021, -0.014, -0.115])       # base centre -> IMU, metres
MOCK_TILT_DEG = 15.0
MOCK_MAG_BODY = np.array([21.0, -4.0, -42.0])        # uT, fixed field in body axes


class MockImu:
    """Rigid-body IMU simulator: poses/rocks a body, mounts an IMU on it at MOCK_LEVER through
    MOCK_R, and reports what that chip would read — including the lever-arm terms."""

    def __init__(self):
        self.mode = "still"
        self.q = np.array([1.0, 0.0, 0.0, 0.0])      # body -> world
        self.t = 0.0
        self.noise = 0.004

    def set_mode(self, mode):
        if mode not in ("still", "tilt", "rock"):
            return False, f"unknown mock pose '{mode}' (still, tilt, rock)"
        self.mode = mode
        self.t = 0.0
        # +theta about the body's left axis swings the nose DOWN, which is the pose the fore-aft
        # capture asks for.
        self.q = _quat_from_pitch(math.radians(MOCK_TILT_DEG) if mode == "tilt" else 0.0)
        return True, None

    def step(self, dt, rng):
        self.t += dt
        if self.mode == "rock":
            w = (2.0 * np.array([1.0, 0.0, 0.0]) * math.sin(2 * math.pi * 0.7 * self.t)
                 + 1.6 * np.array([0.0, 1.0, 0.0]) * math.sin(2 * math.pi * 1.1 * self.t))
            al = (2.0 * 2 * math.pi * 0.7 * np.array([1.0, 0.0, 0.0]) * math.cos(2 * math.pi * 0.7 * self.t)
                  + 1.6 * 2 * math.pi * 1.1 * np.array([0.0, 1.0, 0.0]) * math.cos(2 * math.pi * 1.1 * self.t))
            self.q = _quat_integrate(self.q, w, dt)
        else:
            w = np.zeros(3)
            al = np.zeros(3)

        up_body = _quat_up_body(self.q)
        a_imu = np.cross(al, MOCK_LEVER) + np.cross(w, np.cross(w, MOCK_LEVER))
        f_body = a_imu / mountcal.G0 + up_body                    # specific force, g
        n = rng.normal(0.0, self.noise, 3)
        acc_chip = MOCK_R.T @ f_body + n
        gyr_chip = MOCK_R.T @ np.degrees(w) + n * 20.0
        mag_chip = MOCK_R.T @ MOCK_MAG_BODY + n * 5.0
        return list(acc_chip), list(gyr_chip), list(mag_chip)


def _quat_from_pitch(theta):
    return np.array([math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0])


def _quat_integrate(q, w, dt):
    q0, q1, q2, q3 = q
    gx, gy, gz = w
    qd = 0.5 * np.array([-q1 * gx - q2 * gy - q3 * gz,
                         q0 * gx + q2 * gz - q3 * gy,
                         q0 * gy - q1 * gz + q3 * gx,
                         q0 * gz + q1 * gy - q2 * gx])
    q = q + qd * dt
    return q / np.linalg.norm(q)


def _quat_up_body(q):
    q0, q1, q2, q3 = q
    return np.array([2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2)])


# ============================================================================ the poll thread
class SenseHat(threading.Thread):
    """Owns the I2C bus and publishes a snapshot + a ScalarRing of history.

    `mock=True` synthesises plausible values so the panel can be developed off the robot (matching
    the web UI's existing `--mock` motor path)."""

    CAPTURE_S = 1.5             # how long a still-robot average (gyro bias, level, tilt) runs
    GYRO_STILL_DPS = 3.0        # per-axis gyro spread allowed during any capture
    ACC_STILL_G = 0.04          # how far |a| may sit from 1 g during a pose capture
    LEVER_MAX_S = 40.0          # cap on a lever-arm excitation recording

    def __init__(self, bus_num=I2C_BUS, mock=False, mount=None):
        super().__init__(daemon=True, name="sensehat")
        self.bus_num = bus_num
        self.mock = mock
        self.available = False
        self.error = None if (smbus2 or mock) else "smbus2 not installed (pip install smbus2)"
        self.stop_event = threading.Event()
        self.ring = ScalarRing(RING_FIELDS, capacity=2048)
        self.filt = Madgwick()
        self.mount = mount if mount is not None else mountcal.MountCal.load_or_new()
        self.gyro_bias = np.zeros(3)
        self.bias_status = {"state": "pending", "msg": "not calibrated yet"}
        # One capture machine for all three still-robot averages; `kind` says which is running.
        self._cap = None
        self.capture_status = {"kind": None, "state": "idle", "msg": ""}
        self._lever_buf = None       # list of (f_body, omega, alpha, up_body) while recording
        self.lever_status = {"state": "idle", "msg": "", "n": 0}
        self._lock = threading.Lock()
        self._snap = {}
        self._chips = {}
        self._t0 = time.time()
        self._err_count = 0
        self._last_err = None
        self._mock = MockImu() if mock else None
        self._rng = np.random.default_rng(0)

    # ---- public API (Flask side) -----------------------------------------------------------
    def snapshot(self):
        with self._lock:
            return dict(self._snap)

    def start_capture(self, kind):
        """Ask the poll thread for a still-robot average: `gyro` (zero-rate bias), `level` (the
        upright reference pose) or `forward` (the nose-down tilt that pins the fore-aft axis).
        Returns immediately; watch `capture_status` / `bias_status` for the verdict."""
        if kind not in ("gyro", "level", "forward"):
            return {"ok": False, "error": f"unknown capture '{kind}'"}
        if not self.available:
            return {"ok": False, "error": self.error or "sensors unavailable"}
        if kind == "forward" and self.mount.up_chip is None:
            return {"ok": False, "error": "capture the level reference first — the tilt is measured "
                                          "as a change from it"}
        with self._lock:
            if self._cap is not None:
                return {"ok": False, "error": f"a {self._cap['kind']} capture is already running"}
            self._cap = {"kind": kind, "acc": [], "gyr": []}
            self.capture_status = {"kind": kind, "state": "running",
                                   "msg": f"averaging {self.CAPTURE_S:.1f} s — hold the robot still"}
        return {"ok": True}

    def lever_start(self):
        """Begin recording a rocking excitation for the lever-arm fit."""
        if not self.available:
            return {"ok": False, "error": self.error or "sensors unavailable"}
        if not self.mount.calibrated:
            return {"ok": False, "error": "calibrate the mount rotation first — the fit is done in "
                                          "body axes"}
        with self._lock:
            self._lever_buf = []
            self.lever_status = {"state": "recording", "n": 0,
                                 "msg": "rock the robot by hand about TWO different axes"}
        return {"ok": True}

    def lever_stop(self):
        """Stop recording and fit. Returns the estimate (or why it could not be made)."""
        with self._lock:
            buf, self._lever_buf = self._lever_buf, None
        if not buf:
            self.lever_status = {"state": "error", "msg": "nothing was recorded", "n": 0}
            return {"ok": False, "error": "nothing was recorded"}
        fit = mountcal.estimate_lever(buf)
        if not fit.get("ok"):
            self.lever_status = {"state": "error", "msg": fit["error"], "n": len(buf)}
            return {"ok": False, "error": fit["error"]}
        self.mount.set_lever_fit(fit, about=self.mount.reference)
        self.lever_status = {"state": "ok", "n": len(buf),
                             "msg": f"fit from {fit['samples']} rotating samples, residual "
                                    f"{fit['residual_ms2']:.2f} m/s²"}
        return {"ok": True, "fit": fit}

    def mock_pose(self, mode):
        """Pose the simulated robot (mock only): `still` upright, `tilt` nose-down, `rock` a
        two-axis hand-rock — the three motions the calibration procedure asks for."""
        if not self.mock:
            return {"ok": False, "error": "mock mode only"}
        ok, why = self._mock.set_mode(mode)
        return {"ok": ok} if ok else {"ok": False, "error": why}

    def stop(self):
        self.stop_event.set()

    # ---- setup -----------------------------------------------------------------------------
    def _setup(self):
        if self.mock:
            self.available = True
            return
        if smbus2 is None:
            return
        try:
            self.bus = smbus2.SMBus(self.bus_num)
        except Exception as e:
            self.error = f"cannot open /dev/i2c-{self.bus_num}: {e} (enable I2C with raspi-config)"
            return
        # Each chip is optional: a HAT with, say, no colour sensor populated must still give an IMU.
        for name, cls in (("imu", ICM20948), ("shtc3", SHTC3), ("lps22hb", LPS22HB),
                          ("tcs34725", TCS34725), ("ads1015", ADS1015)):
            try:
                chip = cls(self.bus)
                chip.init()
                self._chips[name] = chip
            except Exception as e:
                print(f"!! Sense HAT: {name} unavailable — {e}")
        if "imu" not in self._chips:
            self.error = "no chips answered on I2C — is the Sense HAT (B) seated on the header?"
            self.available = bool(self._chips)
            return
        self.available = True
        self.error = None
        self.start_capture("gyro")          # first bias average as soon as we start reading

    # ---- the loop --------------------------------------------------------------------------
    def run(self):
        self._setup()
        if not self.available:
            with self._lock:
                self._snap = {"available": False, "error": self.error}
            print(f"!! Sense HAT (B) unavailable: {self.error}")
            return
        print(f"Sense HAT (B): {'MOCK' if self.mock else ', '.join(sorted(self._chips))}")

        dt_nom = 1.0 / IMU_HZ
        next_imu = time.time()
        next_env = next_color = next_adc = 0.0
        next_log = 0.0
        last_t = time.time()
        env = {}
        adc_ch = 0
        w_lp = np.zeros(3)          # low-passed body rate, differentiated for angular acceleration
        mount_version = self.mount.version

        while not self.stop_event.is_set():
            now = time.time()
            if now < next_imu:
                time.sleep(min(next_imu - now, 0.005))
                continue
            next_imu = max(now, next_imu + dt_nom)
            dt = min(max(now - last_t, 1e-4), 0.1)
            last_t = now

            acc, gyr, mag, temp_imu = self._read_motion(dt)
            if acc is None:
                continue

            # still-robot averages (gyro bias / level / tilt) all run on the RAW chip-frame vectors:
            # the mount rotation is what they are being measured to produce.
            self._tick_capture(acc, gyr)

            # chip -> body. The gyro bias is measured in chip axes, so subtract before rotating.
            R = self.mount.R
            gyr_c = np.asarray(gyr, float) - self.gyro_bias
            acc_b = R @ np.asarray(acc, float)
            gyr_b = R @ gyr_c
            mag_b = R @ np.asarray(mag, float) if mag else None

            # The frame just moved under the filter: its quaternion describes an attitude in the
            # OLD body axes, which can be arbitrarily far from the truth in the new ones — and a
            # near-antipodal Madgwick error sits on a saddle where the correction vanishes, so it
            # would never converge out of it. Restart from what the accelerometer says instead.
            if self.mount.version != mount_version:
                mount_version = self.mount.version
                self.filt.reset(acc_b)
                w_lp = np.zeros(3)

            # Angular acceleration for the lever-arm terms: differentiating a rate gyro amplifies
            # its noise, so the rate is low-passed first and the difference taken on that. The SAME
            # low-passed rate then feeds the omega x (omega x r) term — pairing a raw omega with an
            # alpha derived from a filtered one mismatches their phase and biases the fit.
            w_rad = np.radians(gyr_b)
            a_lp = dt / (OMEGA_TAU_S + dt)
            w_prev = w_lp
            w_lp = (1 - a_lp) * w_lp + a_lp * w_rad
            alpha = (w_lp - w_prev) / dt

            # Subtract what the offset alone accounts for, so the filter sees the specific force at
            # the base centre rather than at the IMU. Zero at rest; only matters while rotating.
            acc_filt = acc_b
            lever = self.mount.lever()
            if lever is not None:
                rot_ms2 = np.cross(alpha, lever) + np.cross(w_lp, np.cross(w_lp, lever))
                acc_filt = acc_b - rot_ms2 / mountcal.G0

            roll, pitch, yaw = self.filt.update(gyr_b, acc_filt, dt)
            heading = tilt_compensated_heading(mag_b, roll, pitch)

            if self._lever_buf is not None:
                self._tick_lever(acc_b, w_lp, alpha)

            if now >= next_env:
                next_env = now + ENV_PERIOD_S
                env.update(self._read_env())
            if now >= next_color:
                next_color = now + COLOR_PERIOD_S
                env.update(self._read_color())
            if now >= next_adc:
                next_adc = now + ADC_PERIOD_S
                env.update(self._read_adc(adc_ch))
                adc_ch = (adc_ch + 1) % 4

            sample = {
                "ax": acc_b[0], "ay": acc_b[1], "az": acc_b[2],
                "gx": gyr_b[0], "gy": gyr_b[1], "gz": gyr_b[2],
                "mx": mag_b[0] if mag_b is not None else None,
                "my": mag_b[1] if mag_b is not None else None,
                "mz": mag_b[2] if mag_b is not None else None,
                "roll": roll, "pitch": pitch, "yaw": yaw, "heading": heading,
                "acc_mag": float(np.linalg.norm(acc_b)), "gyro_mag": float(np.linalg.norm(gyr_b)),
                "temp_imu": temp_imu, **env,
            }
            # The AHRS runs at IMU_HZ, but nothing downstream reads faster than the UI polls, so
            # the ring push and the snapshot rebuild are both decimated to LOG_HZ.
            if now >= next_log:
                next_log = max(now, next_log + 1.0 / LOG_HZ)
                self.ring.push(now - self._t0, sample)
                self._publish(sample, acc, mag is not None)

    # ---- calibration captures --------------------------------------------------------------
    def _tick_capture(self, acc, gyr):
        """Collect one sample into a running still-robot average and finish it when full."""
        with self._lock:
            cap = self._cap
        if cap is None:
            return
        cap["acc"].append(acc)
        cap["gyr"].append(gyr)
        if len(cap["gyr"]) < int(self.CAPTURE_S * IMU_HZ):
            return

        g = np.array(cap["gyr"])
        a = np.array(cap["acc"])
        spread = float(np.max(g.max(0) - g.min(0)))
        acc_mean = a.mean(0)
        mag = float(np.linalg.norm(acc_mean))
        kind = cap["kind"]
        with self._lock:
            self._cap = None

        # One stillness test for all three: a moving robot invalidates a gyro bias and a pose
        # capture alike, and saying so beats storing a quietly wrong calibration.
        if spread > self.GYRO_STILL_DPS:
            msg = (f"robot moved during the capture ({spread:.1f} deg/s spread > "
                   f"{self.GYRO_STILL_DPS}) — hold it still and retry")
            self._capture_failed(kind, "moving", msg)
            return
        if kind in ("level", "forward") and abs(mag - 1.0) > self.ACC_STILL_G:
            msg = (f"|a| was {mag:.3f} g, not 1 g — the robot was accelerating (or the sensor is "
                   f"badly scaled); retry with it hanging still")
            self._capture_failed(kind, "moving", msg)
            return

        meta = {"spread_dps": round(spread, 2), "acc_mag_g": round(mag, 4), "n": len(g)}
        if kind == "gyro":
            self.gyro_bias = g.mean(0)
            self.bias_status = {"state": "ok",
                                "msg": f"bias {np.array2string(self.gyro_bias, precision=2)} deg/s"}
            self.filt.reset(self.mount.R @ acc_mean)
            with self._lock:
                self.capture_status = {"kind": "gyro", "state": "ok", "msg": self.bias_status["msg"]}
        elif kind == "level":
            self.mount.set_level(acc_mean, meta)
            self.filt.reset(self.mount.R @ acc_mean)
            with self._lock:
                self.capture_status = {"kind": "level", "state": "ok",
                                       "msg": f"upright reference captured (|a| {mag:.3f} g)"}
        else:
            ok, why = self.mount.set_forward_from_tilt(acc_mean, meta)
            if not ok:
                self._capture_failed("forward", "tilt", why)
                return
            tilt = self.mount.captures["forward"]["tilt_deg"]
            with self._lock:
                self.capture_status = {"kind": "forward", "state": "ok",
                                       "msg": f"fore-aft axis fixed from a {tilt:.1f}° nose-down tilt"}

    def _capture_failed(self, kind, state, msg):
        with self._lock:
            self.capture_status = {"kind": kind, "state": state, "msg": msg}
        if kind == "gyro":
            self.bias_status = {"state": state, "msg": msg}

    def _tick_lever(self, acc_b, w_rad, alpha):
        """Append one excitation sample (body axes, SI units) for the lever-arm fit."""
        buf = self._lever_buf
        if buf is None:
            return
        buf.append((acc_b * mountcal.G0, w_rad.copy(), alpha.copy(), self.filt.up_body()))
        if len(buf) % 20 == 0:
            with self._lock:
                self.lever_status = dict(self.lever_status, n=len(buf))
        if len(buf) >= int(self.LEVER_MAX_S * IMU_HZ):
            self.lever_stop()

    def _publish(self, sample, acc_chip, mag_live):
        with self._lock:
            self._snap = {
                "available": True, "error": None, "mock": self.mock,
                "chips": sorted(self._chips) if not self.mock else ["mock"],
                "mag_live": mag_live,
                "gyro_bias": [round(float(v), 3) for v in self.gyro_bias],
                "bias_status": dict(self.bias_status),
                "capture_status": dict(self.capture_status),
                "lever_status": dict(self.lever_status),
                "i2c_errors": self._err_count, "last_error": self._last_err,
                # the raw chip-frame accelerometer, kept alongside the body-frame values: it is what
                # the mount panel checks against, and it is the only thing shown before calibration
                "acc_chip": [_clean(v) for v in acc_chip],
                "mount": self.mount.snapshot(),
                "values": {k: _clean(v) for k, v in sample.items()},
            }

    # ---- per-chip reads, each isolated so one sulking chip cannot stop the others -----------
    def _fail(self, chip, e):
        """Record a chip read failure. Broad on purpose: the robot is in the field and one flaky
        sensor must never take the IMU stream (or the poll thread) down with it — but the reason is
        counted and published, so a failure is visible in the panel rather than silent."""
        self._err_count += 1
        self._last_err = f"{chip}: {type(e).__name__}: {e}"
        if self._err_count % 200 == 1:
            print(f"!! Sense HAT {self._last_err} ({self._err_count} read errors so far)")

    def _read_motion(self, dt=1.0 / IMU_HZ):
        if self.mock:
            t = time.time() - self._t0
            # the simulator must advance on the SAME clock the loop differentiates with, or the
            # angular acceleration it implies and the one the driver computes disagree
            acc, gyr, mag = self._mock.step(dt, self._rng)
            return acc, gyr, mag, 34.0 + 0.5 * math.sin(t * 0.05)
        try:
            acc, gyr, temp = self._chips["imu"].read_motion()
            return acc, gyr, self._chips["imu"].read_mag(), temp
        except Exception as e:
            self._fail("imu", e)
            return None, None, None, None

    def _read_env(self):
        if self.mock:
            t = time.time() - self._t0
            return {"temp": 24.0 + 0.4 * math.sin(t * 0.03), "humidity": 45.0 + 2 * math.sin(t * 0.02),
                    "pressure": 1013.2 + 0.3 * math.sin(t * 0.01), "temp_baro": 25.5}
        out = {}
        try:
            r = self._chips["shtc3"].read() if "shtc3" in self._chips else None
            if r:
                out["temp"], out["humidity"] = r
        except Exception as e:
            self._fail("shtc3", e)
        try:
            r = self._chips["lps22hb"].read() if "lps22hb" in self._chips else None
            if r:
                out["pressure"], out["temp_baro"] = r
        except Exception as e:
            self._fail("lps22hb", e)
        return out

    def _read_color(self):
        if self.mock:
            t = time.time() - self._t0
            return {"lux": 180 + 40 * math.sin(t * 0.11), "cct": 4300 + 200 * math.cos(t * 0.07),
                    "rgb": [120, 140, 160], "clear": 420}
        if "tcs34725" not in self._chips:
            return {}
        try:
            r = self._chips["tcs34725"].read()
            if not r:
                return {}
            c, red, green, blue, lux, cct = r
            return {"lux": lux, "cct": cct, "rgb": [red, green, blue], "clear": c}
        except Exception as e:
            self._fail("tcs34725", e)
            return {}

    def _read_adc(self, ch):
        if self.mock:
            t = time.time() - self._t0
            return {f"adc{ch}": 1.5 + 0.4 * math.sin(t * 0.4 + ch)}
        if "ads1015" not in self._chips:
            return {}
        try:
            return {f"adc{ch}": self._chips["ads1015"].read_channel(ch)}
        except Exception as e:
            self._fail("ads1015", e)
            return {}


# ============================================================================ CLI self-test
def main():
    """`python fixed_gait/webui/sensehat.py` — print a live line per second, no web UI needed."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--bus", type=int, default=I2C_BUS)
    args = ap.parse_args()

    sh = SenseHat(bus_num=args.bus, mock=args.mock)
    sh.start()
    time.sleep(1.0)
    if not sh.available:
        print(f"unavailable: {sh.error}")
        return
    try:
        while True:
            v = sh.snapshot().get("values", {})
            g = lambda k, d=1: "—" if v.get(k) is None else f"{v[k]:.{d}f}"     # noqa: E731
            print(f"rpy {g('roll')},{g('pitch')},{g('yaw')}°  "
                  f"acc {g('ax',2)},{g('ay',2)},{g('az',2)} g  "
                  f"gyr {g('gx')},{g('gy')},{g('gz')} °/s  "
                  f"mag {g('mx')},{g('my')},{g('mz')} µT  "
                  f"{g('temp')}°C {g('humidity')}% {g('pressure',2)} hPa ({g('temp_baro')}°C)  "
                  f"{g('lux')} lx {g('cct',0)} K  adc {g('adc0',3)}V")
            time.sleep(1.0)
    except KeyboardInterrupt:
        sh.stop()


if __name__ == "__main__":
    main()
