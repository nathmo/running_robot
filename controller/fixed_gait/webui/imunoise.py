#!/usr/bin/env python3
"""IMU noise from a still record: the web UI's "record while still" button and tools/imu_bench.py
share this, so a record is judged by the same stillness gate whichever way it was taken.

**A noise figure from a robot that was not still is the robot, not the sensor** — a swinging robot
once produced a "noise" figure 8x too high on one axis. Every figure is therefore returned next to
a stillness verdict, and the verdict travels with the saved record.
"""
import math
import os
import time

import numpy as np

# Datasheet, FCHOICE=1: DLPF config -> (3 dB bandwidth, noise bandwidth) in Hz. NOT measured here;
# --dlpf checks whether the noise scales the way this table implies.
GYR_BW = {0: (196.6, 229.8), 1: (151.8, 187.6), 2: (119.5, 154.3), 3: (51.2, 73.3),
          4: (23.9, 35.9), 5: (11.6, 17.8), 6: (5.7, 8.9)}
ACC_BW = {0: (246.0, 265.0), 1: (246.0, 265.0), 2: (111.4, 136.0), 3: (50.4, 68.8),
          4: (23.9, 34.4), 5: (11.5, 16.8), 6: (5.7, 8.3)}

# A still robot on the ground. Above these, whatever is measured is the robot, not the sensor.
# Vibration bounds: ~4x the datasheet noise at DLPF cfg 3 (and still >2x margin at cfg 0, whose
# noise bandwidth is ~4x wider). Sway bounds are on 1 s block means, where rocking lives: 6 mg
# is a steady 0.4 deg of tilt, 0.2 dps is far above the measured 0.012 dps bias instability.
STILL_GYRO_RMS_DPS = 0.35
STILL_ACC_RMS_G = 0.007
STILL_GYRO_DRIFT_DPS = 0.2
STILL_ACC_DRIFT_G = 0.006
STILL_PP_MARGIN = 1.6


def stillness(acc, gyr, fs):
    """(is_still, one-line reason). Three failure modes, three checks. Vibration inflates the
    RMS itself. Sway is low-frequency and lives in the 1 s block means (a swaying robot
    ROTATES — the gyro block means are the sharp detector; sd barely moves). Bumps are
    transients, caught by the raw peak-to-peak against what this record's own RMS predicts
    for gaussian noise (2*sqrt(2 ln n)*sigma). A FIXED raw peak-to-peak bound is wrong: the
    expected extremes grow with sample count, so a perfectly still sensor trips it once the
    record is long enough — the old 0.6 dps / 20 mg gate did exactly that on the first
    300 s record (gyro swing 0.73 dps, 0.9x the white-noise expectation, block means clean)."""
    n = len(gyr)
    w = max(1, int(round(fs)))
    checks = []
    for name, x, rms_lim, drift_lim, scale, unit in (
            ("gyro", gyr, STILL_GYRO_RMS_DPS, STILL_GYRO_DRIFT_DPS, 1.0, "dps"),
            ("accel", acc, STILL_ACC_RMS_G, STILL_ACC_DRIFT_G, 1000.0, "mg")):
        sd = x.std(0)
        bm = x[:n // w * w].reshape(-1, w, x.shape[1]).mean(1)
        drift = float(np.max(bm.max(0) - bm.min(0)))
        pp = float(np.max((x.max(0) - x.min(0)) / (2 * np.sqrt(2 * np.log(n)) * sd)))
        if float(sd.max()) > rms_lim:
            return False, (f"MOVING (vibration): {name} RMS {sd.max() * scale:.2f} {unit} "
                           f"(limit {rms_lim * scale:.2f}) — something is buzzing the robot")
        if drift > drift_lim:
            return False, (f"MOVING (sway): {name} 1 s-average swing {drift * scale:.2f} {unit} "
                           f"(limit {drift_lim * scale:.2f}) — set the robot down on the "
                           f"floor, off any rig that lets it rock")
        if pp > STILL_PP_MARGIN:
            return False, (f"MOVING (bumps): {name} peak-to-peak {pp:.2f}x the white-noise "
                           f"expectation (limit {STILL_PP_MARGIN}) — something knocked the "
                           f"robot mid-record")
        checks.append(f"{name} drift {drift * scale:.2f} {unit}, p-p {pp:.2f}x white")
    return True, "still (" + "; ".join(checks) + ")"


def _block_means(x, w):
    n = len(x) // w * w
    return x[:n].reshape(-1, w, x.shape[1]).mean(1) if n else x[:0]


def analyze(acc_g, gyr_dps, t_s, pitch_deg=None, roll_deg=None, dlpf_cfg=3):
    """Summary of a still record. `acc_g` / `gyr_dps`: (n, 3), in whatever frame they were
    recorded in (the web UI records BODY axes, gyro with the current zero subtracted, so the gyro
    mean is the zero's residual). `pitch_deg` / `roll_deg`: the attitude filter's output over the
    same samples — the noise the balance loop actually steers on."""
    acc = np.asarray(acc_g, float)
    gyr = np.asarray(gyr_dps, float)
    t = np.asarray(t_s, float)
    n = len(acc)
    if n < 50:
        return {"ok": False, "error": f"only {n} samples — record for at least a few seconds"}
    dt = np.diff(t)
    fs = 1.0 / float(dt.mean()) if len(dt) else 0.0
    still, why = stillness(acc, gyr, fs)
    nbw_a, nbw_g = ACC_BW[dlpf_cfg][1], GYR_BW[dlpf_cfg][1]
    w = max(1, int(round(fs)))
    out = {
        "ok": True, "n": n, "seconds": float(t[-1] - t[0]), "rate_hz": fs,
        "jitter_ms": float(dt.std() * 1000), "max_gap_ms": float(dt.max() * 1000),
        "still": bool(still), "still_why": why, "dlpf_cfg": dlpf_cfg,
        "acc_mean_g": acc.mean(0).tolist(),
        "acc_mag_g": float(np.linalg.norm(acc.mean(0))),
        "acc_rms_mg": (acc.std(0) * 1000).tolist(),
        "acc_density_ug": (acc.std(0) * 1e6 / math.sqrt(nbw_a)).tolist(),
        "gyr_mean_dps": gyr.mean(0).tolist(),
        "gyr_rms_dps": gyr.std(0).tolist(),
        "gyr_density_dps": (gyr.std(0) / math.sqrt(nbw_g)).tolist(),
        "gyr_wander_dps": _block_means(gyr, w).std(0).tolist() if n >= 2 * w else None,
    }
    for name, a in (("pitch", pitch_deg), ("roll", roll_deg)):
        if a is not None and len(a) == n:
            a = np.asarray(a, float)
            out[name] = {"mean": float(a.mean()), "rms": float(a.std()),
                         "pp": float(a.max() - a.min())}
    return out


def save_record(directory, acc_g, gyr_dps, t_s, summary, pitch_deg=None, roll_deg=None,
                frame="body", acc_range_g=None, gyr_range_dps=None):
    """Dump the raw record next to its verdict (same keys as imu_bench --save). Returns the path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, time.strftime("imu_noise_%Y%m%d_%H%M%S.npz"))
    cfg = summary.get("dlpf_cfg", 3)
    np.savez(path, acc_g=np.asarray(acc_g), gyr_dps=np.asarray(gyr_dps), t_s=np.asarray(t_s),
             still=bool(summary.get("still")), frame=frame, dlpf_cfg=cfg,
             pitch_deg=np.asarray(pitch_deg if pitch_deg is not None else []),
             roll_deg=np.asarray(roll_deg if roll_deg is not None else []),
             acc_range_g=acc_range_g if acc_range_g is not None else np.nan,
             gyr_range_dps=gyr_range_dps if gyr_range_dps is not None else np.nan,
             acc_nbw_hz=ACC_BW[cfg][1], gyr_nbw_hz=GYR_BW[cfg][1])
    return path
