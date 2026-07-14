"""Fourier cyclic-gait reconstruction for DASH-01 — PURE NUMPY (Pi-shareable, like fixed_gait/gait.py).

The RL policy emits, ONCE PER GAIT CYCLE, a compact action that this module turns into the 6 PD motor
targets at any point in the cycle:
  - cam + thigh (the sagittal 4-bar, propulsion): a Fourier series over the gait phase phi, SYMMETRIC
    across legs (right = mirror of left, half a cycle out of phase — the fixed_gait/gait.py convention),
    centered on the per-episode stance pose. Harmonics are weighted DOWN by 1/k so the offset + low
    harmonics dominate (smooth, hardware-friendly) and the total excursion stays ~bounded.
  - hip_roll (abduction, lateral balance): a learned LINEAR REFLEX on roll / roll-rate evaluated at
    50 Hz INSIDE the cycle (feedback, not a function of phi). Inert while roll is locked (M1-M3).

Action layout (all entries in [-1, 1]), for N harmonics:
    [ cam:   a0,a1,b1,...,aN,bN   (1+2N) ]
    [ thigh: a0,a1,b1,...,aN,bN   (1+2N) ]
    [ freq_raw                    (1)    ]
    [ reflex: kp, kd, bias        (3)    ]
Total = 2*(1+2N) + 4.

Actuator/ctrl order matches mujoco/dash01/dash01.xml <actuator> and fixed_gait/gait.py:
    0 hip_roll_L   1 cam_L   2 thigh_L   3 hip_roll_R   4 cam_R   5 thigh_R
"""
import numpy as np

HIP_ROLL_L, CAM_L, THIGH_L, HIP_ROLL_R, CAM_R, THIGH_R = range(6)
N_REFLEX = 3   # kp, kd, bias


def per_joint(n_harmonics):
    return 1 + 2 * n_harmonics


def action_dim(n_harmonics):
    """Size of the per-cycle policy action: cam + thigh Fourier coeffs + frequency + reflex gains."""
    return 2 * per_joint(n_harmonics) + 1 + N_REFLEX


def _weights(n_harmonics):
    """Amplitude budget: offset + harmonic k weighted by 1/k, normalized to sum 1. Keeps the
    reconstruction dominated by the low harmonics (smooth) and |series| ~<= 1 before the amp scale."""
    w = np.array([1.0] + [1.0 / k for k in range(1, n_harmonics + 1)])
    return w / w.sum()


def decode(action, n_harmonics):
    """Split the raw [-1,1] action into (cam_coeffs, thigh_coeffs, freq_raw, reflex_raw)."""
    pj = per_joint(n_harmonics)
    cam = np.asarray(action[0:pj], dtype=float)
    thigh = np.asarray(action[pj:2 * pj], dtype=float)
    freq_raw = float(action[2 * pj])
    reflex = np.asarray(action[2 * pj + 1: 2 * pj + 1 + N_REFLEX], dtype=float)
    return cam, thigh, freq_raw, reflex


def frequency(freq_raw, freq_range):
    """Map freq_raw in [-1,1] to a gait frequency (Hz) in freq_range."""
    lo, hi = freq_range
    return lo + (hi - lo) * 0.5 * (float(freq_raw) + 1.0)


def _series(coeffs, phi, weights, n_harmonics):
    """weighted a0 + sum_k wk*(ak cos k phi + bk sin k phi). coeffs = [a0, a1,b1, a2,b2, ...]."""
    val = weights[0] * coeffs[0]
    for k in range(1, n_harmonics + 1):
        ak, bk = coeffs[2 * k - 1], coeffs[2 * k]
        val += weights[k] * (ak * np.cos(k * phi) + bk * np.sin(k * phi))
    return val


def assemble(cam_coeffs, thigh_coeffs, reflex, phi, roll, roll_rate, nominal, cfg):
    """Return the 6 PD motor targets (rad) at gait phase phi.

    nominal   : 6-vector stance ctrl = the per-episode centers (mirrored L/R already).
    roll,roll_rate : current base roll and roll-rate (for the abduction reflex).
    cfg       : Config (reads n_harmonics, cam_amp, thigh_amp, reflex_*_scale).
    """
    N = cfg.n_harmonics
    w = _weights(N)
    # --- cam + thigh: symmetric Fourier, right = mirror (negate) + antiphase (phi + pi) ---
    # The oscillation DELTA is clipped to +-amp of the nominal. `nominal` is always a validated
    # in-band posture (the stand keyframe / M1 ride-height LUT), so bounding the deviation keeps the
    # coupled cam-thigh 4-bar inside its assemblable band without assuming an absolute cam sign/center
    # (stand cam ctrl is ~0, valid cam ranges ~[-0.6,0.6] here — NOT gait.py's in-air +0.6 center).
    ca, ta = cfg.cam_amp, cfg.thigh_amp
    d_cam = np.clip(ca * _series(cam_coeffs, phi, w, N), -ca, ca)
    d_thigh = np.clip(ta * _series(thigh_coeffs, phi, w, N), -ta, ta)
    d_cam_a = np.clip(ca * _series(cam_coeffs, phi + np.pi, w, N), -ca, ca)
    d_thigh_a = np.clip(ta * _series(thigh_coeffs, phi + np.pi, w, N), -ta, ta)
    cam_L = nominal[CAM_L] + d_cam
    thigh_L = nominal[THIGH_L] + d_thigh
    cam_R = nominal[CAM_R] - d_cam_a
    thigh_R = nominal[THIGH_R] - d_thigh_a
    # --- hip_roll: learned linear balance reflex (feedback on roll / roll-rate); sign is learned ---
    u = (cfg.reflex_kp_scale * reflex[0] * roll
         + cfg.reflex_kd_scale * reflex[1] * roll_rate
         + cfg.reflex_bias_scale * reflex[2])
    hr_L = nominal[HIP_ROLL_L] + u
    hr_R = nominal[HIP_ROLL_R] - u

    out = np.empty(6)
    out[HIP_ROLL_L], out[CAM_L], out[THIGH_L] = hr_L, cam_L, thigh_L
    out[HIP_ROLL_R], out[CAM_R], out[THIGH_R] = hr_R, cam_R, thigh_R
    return out
