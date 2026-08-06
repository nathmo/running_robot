#!/usr/bin/env python3
"""Measure the Sense HAT (B) IMU's real properties on this Pi — the numbers behind robot/IMU.md.

    sudo systemctl stop runningrobot-webui        # it owns the I2C bus; two owners corrupt
    python robot/fixed_gait/webui/tools/imu_bench.py      # the ICM's bank-select state
    sudo systemctl start runningrobot-webui

    --dlpf     also sweep the chip's internal low-pass and check noise scales as sqrt(bandwidth)
    --seconds  length of the noise record (default 20)

**The robot must be genuinely at rest, and the tool checks.** A sensor-noise measurement taken
while the robot sways on its rig reports the sway, and there is nothing in the resulting number to
say so — a swinging robot once produced a "noise" figure 8x too high on the yaw axis alone, and a
low-pass sweep in which a *narrower* filter measured *more* noise. Every noise figure below is
therefore gated on a motion check and labelled CONTAMINATED rather than quietly printed.
"""
import argparse
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import smbus2                       # noqa: E402
import sensehat                     # noqa: E402

# Datasheet, FCHOICE=1: DLPF config -> (3 dB bandwidth, noise bandwidth) in Hz. NOT measured here;
# --dlpf checks whether the noise scales the way this table implies.
GYR_BW = {0: (196.6, 229.8), 1: (151.8, 187.6), 2: (119.5, 154.3), 3: (51.2, 73.3),
          4: (23.9, 35.9), 5: (11.6, 17.8), 6: (5.7, 8.9)}
ACC_BW = {0: (246.0, 265.0), 1: (246.0, 265.0), 2: (111.4, 136.0), 3: (50.4, 68.8),
          4: (23.9, 34.4), 5: (11.5, 16.8), 6: (5.7, 8.3)}

# A still robot on the ground. Above these, whatever is measured is the robot, not the sensor.
STILL_GYRO_RANGE_DPS = 0.6
STILL_ACC_RANGE_G = 0.02


def collect(imu, seconds, hz=200.0):
    """Sample at a steady rate; returns (accel g, gyro dps, timestamps)."""
    acc, gyr, ts = [], [], []
    nxt = time.perf_counter()
    for _ in range(int(seconds * hz)):
        nxt += 1.0 / hz
        while time.perf_counter() < nxt:
            time.sleep(0)
        a, g, _ = imu.read_motion()
        ts.append(time.perf_counter())
        acc.append(a)
        gyr.append(g)
    return np.array(acc), np.array(gyr), np.array(ts)


def stillness(acc, gyr):
    """(is_still, one-line reason). Peak-to-peak, not sd: a slow drift is exactly what a swaying
    robot looks like and it barely moves the standard deviation."""
    gr = float(np.max(gyr.max(0) - gyr.min(0)))
    ar = float(np.max(acc.max(0) - acc.min(0)))
    if gr > STILL_GYRO_RANGE_DPS or ar > STILL_ACC_RANGE_G:
        return False, (f"MOVING: gyro swing {gr:.2f} dps (limit {STILL_GYRO_RANGE_DPS}), "
                       f"accel swing {ar * 1000:.1f} mg (limit {STILL_ACC_RANGE_G * 1000:.0f}) "
                       f"— set the robot down on the floor, off any rig that lets it sway")
    return True, f"still (gyro swing {gr:.2f} dps, accel swing {ar * 1000:.1f} mg)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", type=int, default=sensehat.I2C_BUS)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--dlpf", action="store_true", help="sweep the internal low-pass too")
    args = ap.parse_args()

    bus = smbus2.SMBus(args.bus)
    imu = sensehat.ICM20948(bus)
    imu.init()
    time.sleep(0.3)
    print(f"ICM-20948: +/-{imu.ACC_RANGE_G} g, +/-{imu.GYR_RANGE_DPS} dps, "
          f"DLPF cfg {imu.DLPF_CFG}, ODR {imu.odr_hz:.1f} Hz\n")

    # ---- I2C throughput
    def rate(fn, secs=3.0):
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < secs:
            fn()
            n += 1
        return n / (time.perf_counter() - t0)

    r_ag = rate(lambda: imu.read_motion())
    r_agm = rate(lambda: (imu.read_motion(), imu.read_mag()))
    print("I2C throughput")
    print(f"  accel+gyro+temp burst : {r_ag:6.0f} reads/s ({1000 / r_ag:.2f} ms each)")
    print(f"  + magnetometer        : {r_agm:6.0f} reads/s ({1000 / r_agm:.2f} ms each)")

    # ---- true output data rate, by counting reads that returned an unchanged sample
    prev, dup, tot = None, 0, 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3.0:
        a, _, _ = imu.read_motion()
        tot += 1
        if prev is not None and a == prev:
            dup += 1
        prev = a
    polled = tot / 3.0
    print(f"\nOutput data rate: polled {polled:.0f}/s, {100 * dup / tot:.0f}% unchanged "
          f"-> ~{polled * (1 - dup / tot):.0f} Hz fresh (configured {imu.odr_hz:.0f} Hz)")

    # ---- noise / bias / resolution, gated on the robot actually being still
    acc, gyr, ts = collect(imu, args.seconds)
    ok, why = stillness(acc, gyr)
    dt = np.diff(ts)
    print(f"\nAt rest, {len(acc)} samples: {why}")
    print(f"  achieved rate  : {1 / dt.mean():.1f} Hz, jitter sd {dt.std() * 1000:.2f} ms, "
          f"max gap {dt.max() * 1000:.2f} ms")
    label = "" if ok else "   <-- CONTAMINATED BY MOTION, not a sensor property"
    print(f"  accel mean [g] : {np.array2string(acc.mean(0), precision=4)}  "
          f"|a| {np.linalg.norm(acc.mean(0)):.4f} "
          f"(scale error {100 * (np.linalg.norm(acc.mean(0)) - 1):+.2f}%)")
    print(f"  accel noise    : {np.array2string(acc.std(0) * 1000, precision=2)} mg RMS{label}")
    print(f"  gyro bias      : {np.array2string(gyr.mean(0), precision=3)} dps")
    print(f"  gyro noise     : {np.array2string(gyr.std(0), precision=4)} dps RMS{label}")
    if ok:
        nbw_a, nbw_g = ACC_BW[imu.DLPF_CFG][1], GYR_BW[imu.DLPF_CFG][1]
        print(f"  -> density     : {acc.std(0).mean() * 1e6 / np.sqrt(nbw_a):.0f} ug/sqrt(Hz), "
              f"{gyr.std(0).mean() / np.sqrt(nbw_g):.4f} dps/sqrt(Hz)")
    # in-run bias stability: how far a 1 s average wanders over the record
    w = int(round(1 / dt.mean()))
    blocks = gyr[:len(gyr) // w * w].reshape(-1, w, 3).mean(1)
    print(f"  gyro 1 s-average wander: {np.array2string(blocks.std(0), precision=4)} dps sd{label}")
    for name, arr, scale, unit, lsb in (("accel", acc, 1000, "mg", 1000 / 8192.),
                                        ("gyro", gyr, 1, "dps", 1 / 32.8)):
        d = np.abs(np.diff(arr, axis=0))
        d = d[d > 0]
        if len(d):
            print(f"  {name} smallest step: {d.min() * scale:.4f} {unit} (1 LSB = {lsb:.4f})")

    # ---- optional: does the internal low-pass behave as the datasheet says?
    if args.dlpf:
        print("\nDLPF sweep (noise should fall as sqrt(noise bandwidth) from cfg 0)")
        print(" cfg | accel 3dB  | accel RMS  predicted | gyro 3dB   | gyro RMS   predicted | still?")
        base_a = base_g = None
        original = sensehat.ICM20948.DLPF_CFG
        for cfg in range(7):
            sensehat.ICM20948.DLPF_CFG = cfg
            probe = sensehat.ICM20948(bus)
            probe.init()
            time.sleep(0.35)
            a, g, _ = collect(probe, 6.0)
            still, _ = stillness(a, g)
            am, gm = a.std(0).mean() * 1000, g.std(0).mean()
            if base_a is None:
                base_a, base_g = am, gm
            pa = base_a * np.sqrt(ACC_BW[cfg][1] / ACC_BW[0][1])
            pg = base_g * np.sqrt(GYR_BW[cfg][1] / GYR_BW[0][1])
            print(f"  {cfg}  | {ACC_BW[cfg][0]:6.1f} Hz | {am:7.3f} mg {pa:8.3f} | "
                  f"{GYR_BW[cfg][0]:6.1f} Hz | {gm:7.4f} dps {pg:7.4f} | "
                  f"{'yes' if still else 'MOVING'}")
        sensehat.ICM20948.DLPF_CFG = original
        print("\nRows marked MOVING measured the robot, not the sensor — rerun them at rest.")


if __name__ == "__main__":
    main()
