"""The v2 gait generator: one Fourier series per joint family, five relationship knobs, phase-
scheduled impedance, two reflexes, a per-tick residual (artifact §02, §06).

Backend-agnostic on purpose: every function takes `xp` (jax.numpy by default, numpy for the
deploy reference and for cross-checking the CPU implementation the walk_mit/ arm is building).
Only ops both backends share are used, so the same source is the training law AND the
reference. No dynamic shapes, no Python control flow on array values.

ACTION LAYOUT (50; all entries in [-1, 1])
    [ 0: 7]  S_cam    a0, a1,b1, a2,b2, a3,b3     N=3 series, x cam_amp           latched
    [ 7:14]  S_thigh                               x thigh_amp                     latched
    [14:21]  S_hip                                 x roll_amp                      latched
    [21:28]  kp(phi) series -> exp map            x2.5 / /3 (plant kp per family) latched
    [28:35]  kd(phi) series -> exp map            soften-only /4                  latched
    [35]     freq_raw  linear -> [0.5, 5.0] Hz                                     latched
    [36:39]  roll reflex kp, kd, bias   u = kp*roll + kd*roll_rate + bias -> hip (+,+)   latched
    [39]     Delta  x delta_max rad   right footfall pi+Delta after the left        latched
    [40]     s      g_L = 1+s, g_R = 1-s                                             latched
    [41:44]  o_cam, o_thigh, o_hip  x o_max, joint bias (+,+)                       latched
    [44:50]  residual r0..r5, x residual_scale rad, per tick

    u_L(phi) = n_L + (1+s) A S(phi)             + o
    u_R(phi) = n_R - (1-s) A S(phi - pi - Delta) + o
The structural minus + half-cycle shift IS the mirror; knobs at zero is exactly the mirror-
symmetric gait (§06). Actuator order: hip_roll_L, cam_L, thigh_L, hip_roll_R, cam_R, thigh_R.

SIGN RULE (verified pure-FK, §06): joint pattern (+,-) is mirror-SYMMETRIC and (+,+) is
ANTISYMMETRIC for every family (cam/thigh axes mirrored L/R, hip_roll both +X). Hence the
fixed pitch reflex enters the thighs (+,-) (both feet the same fore-aft way), and the offsets o
and the roll reflex enter (+,+): o_cam = fore-aft split, o_hip and the reflex = the lean.
"""
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

N_HARMONICS = 3
PER_JOINT = 1 + 2 * N_HARMONICS          # 7
I_S_CAM = slice(0, 7)
I_S_THIGH = slice(7, 14)
I_S_HIP = slice(14, 21)
I_KP = slice(21, 28)
I_KD = slice(28, 35)
I_FREQ = 35
I_REFLEX = slice(36, 39)
I_DELTA = 39
I_S = 40
I_O = slice(41, 44)
SPEC_DIM = 44
N_RESIDUAL = 6
ACTION_DIM = SPEC_DIM + N_RESIDUAL       # 50
KNOB_IDX = (39, 40, 41, 42, 43)
HIP_L, CAM_L, THIGH_L, HIP_R, CAM_R, THIGH_R = range(6)
# mirror permutation of the 6 actuators (L <-> R)
MIRROR_PERM = np.array([HIP_R, CAM_R, THIGH_R, HIP_L, CAM_L, THIGH_L])
# library variant: residual first, then the 3 catch-event latched dims
LIB_N_RESIDUAL = 6
LIB_CATCH = 3
LIB_ACTION_DIM = LIB_N_RESIDUAL + LIB_CATCH


def _weights():
    w = np.array([1.0] + [1.0 / np.sqrt(k) for k in range(1, N_HARMONICS + 1)])
    return w / w.sum()


WEIGHTS = _weights()                       # (4,) offset + 3 harmonics
HARMONIC_K = np.arange(1.0, N_HARMONICS + 1.0)


class GaitParams(NamedTuple):
    """Every number that turns an action into targets + gains. Plain floats/tuples: identical
    under jit (baked constants) and in numpy."""
    cam_amp: float
    thigh_amp: float
    roll_amp: float
    delta_max: float
    o_max: tuple                 # (o_cam_max, o_thigh_max, o_hip_max)
    imp_kp_up: float
    imp_kp_dn: float
    imp_kd_up: float
    imp_kd_dn: float
    reflex_kp_scale: float
    reflex_kd_scale: float
    reflex_bias_scale: float
    pitch_kp: float
    pitch_kd: float
    pitch_bias: float
    pitch_clip: float
    residual_scale: float
    freq_lo: float
    freq_hi: float
    drive_kp: tuple              # 6, actuator order (base gains the profile scales)
    drive_kd: tuple

    @classmethod
    def from_cfg(cls, cfg):
        return cls(cam_amp=float(cfg.cam_amp), thigh_amp=float(cfg.thigh_amp),
                   roll_amp=float(cfg.roll_amp), delta_max=float(cfg.delta_max),
                   o_max=(float(cfg.o_cam_max), float(cfg.o_thigh_max), float(cfg.o_hip_max)),
                   imp_kp_up=float(cfg.imp_kp_up), imp_kp_dn=float(cfg.imp_kp_dn),
                   imp_kd_up=float(cfg.imp_kd_up), imp_kd_dn=float(cfg.imp_kd_dn),
                   reflex_kp_scale=float(cfg.reflex_kp_scale),
                   reflex_kd_scale=float(cfg.reflex_kd_scale),
                   reflex_bias_scale=float(cfg.reflex_bias_scale),
                   pitch_kp=float(cfg.pitch_kp), pitch_kd=float(cfg.pitch_kd),
                   pitch_bias=float(cfg.pitch_bias), pitch_clip=float(cfg.pitch_clip),
                   residual_scale=float(cfg.residual_scale),
                   freq_lo=float(cfg.gait_freq_hz[0]), freq_hi=float(cfg.gait_freq_hz[1]),
                   drive_kp=tuple(float(x) for x in cfg.drive_kp),
                   drive_kd=tuple(float(x) for x in cfg.drive_kd))


# ------------------------------------------------------------------ primitives
def series(coeffs, phi, xp=jnp):
    """weighted a0 + sum_k w_k (a_k cos k phi + b_k sin k phi). coeffs [..., 7], phi [...]."""
    phi = xp.asarray(phi)[..., None]
    k = xp.asarray(HARMONIC_K)
    c, s = xp.cos(k * phi), xp.sin(k * phi)
    w = xp.asarray(WEIGHTS)
    a = coeffs[..., 1::2]
    b = coeffs[..., 2::2]
    return w[0] * coeffs[..., 0] + xp.sum(w[1:] * (a * c + b * s), axis=-1)


def series_dphi(coeffs, phi, xp=jnp):
    """d/dphi of `series`."""
    phi = xp.asarray(phi)[..., None]
    k = xp.asarray(HARMONIC_K)
    c, s = xp.cos(k * phi), xp.sin(k * phi)
    w = xp.asarray(WEIGHTS)
    a = coeffs[..., 1::2]
    b = coeffs[..., 2::2]
    return xp.sum(w[1:] * k * (-a * s + b * c), axis=-1)


def frequency(freq_raw, p: GaitParams, xp=jnp):
    return p.freq_lo + (p.freq_hi - p.freq_lo) * 0.5 * (xp.clip(freq_raw, -1.0, 1.0) + 1.0)


def exp_map(x, up, dn, xp=jnp):
    """[-1, 1] -> multiplier: +1 -> up, -1 -> 1/dn, 0 -> 1. up=1 makes the +side inert."""
    x = xp.clip(x, -1.0, 1.0)
    return xp.exp(xp.where(x >= 0.0, x * np.log(up), x * np.log(dn)))


def wrap_pi(x, xp=jnp):
    """Wrap to (-pi, pi]."""
    return x - 2.0 * np.pi * xp.floor((x + np.pi) / (2.0 * np.pi))


# ------------------------------------------------------------------ the generator
def phases(spec, phi, p: GaitParams, xp=jnp):
    """(phi_L, phi_R): the right leg reads everything pi + Delta later."""
    delta = p.delta_max * xp.clip(spec[..., I_DELTA], -1.0, 1.0)
    return phi, phi - np.pi - delta


def feedforward(spec, phi, nominal, p: GaitParams, xp=jnp):
    """Series + knobs only: the reference joint targets q_ref(phi) [..., 6] with no reflex and
    no residual (what the library once-block reports and what the fixed-point solver plays)."""
    phi_l, phi_r = phases(spec, phi, p, xp)
    s = xp.clip(spec[..., I_S], -1.0, 1.0)
    g_l, g_r = 1.0 + s, 1.0 - s
    o = xp.clip(spec[..., I_O], -1.0, 1.0) * xp.asarray(p.o_max)
    ca, ta, ra = p.cam_amp, p.thigh_amp, p.roll_amp
    d_cam_l = xp.clip(ca * series(spec[..., I_S_CAM], phi_l, xp), -ca, ca)
    d_cam_r = xp.clip(ca * series(spec[..., I_S_CAM], phi_r, xp), -ca, ca)
    d_th_l = xp.clip(ta * series(spec[..., I_S_THIGH], phi_l, xp), -ta, ta)
    d_th_r = xp.clip(ta * series(spec[..., I_S_THIGH], phi_r, xp), -ta, ta)
    d_hip_l = xp.clip(ra * series(spec[..., I_S_HIP], phi_l, xp), -ra, ra)
    d_hip_r = xp.clip(ra * series(spec[..., I_S_HIP], phi_r, xp), -ra, ra)
    n = xp.asarray(nominal)
    q = xp.stack([
        n[..., HIP_L] + g_l * d_hip_l + o[..., 2],
        n[..., CAM_L] + g_l * d_cam_l + o[..., 0],
        n[..., THIGH_L] + g_l * d_th_l + o[..., 1],
        n[..., HIP_R] - g_r * d_hip_r + o[..., 2],
        n[..., CAM_R] - g_r * d_cam_r + o[..., 0],
        n[..., THIGH_R] - g_r * d_th_r + o[..., 1],
    ], axis=-1)
    return q


def feedforward_dphi(spec, phi, p: GaitParams, xp=jnp):
    """d q_ref / d phi [..., 6] (clip ignored: the derivative of the unclipped series)."""
    phi_l, phi_r = phases(spec, phi, p, xp)
    s = xp.clip(spec[..., I_S], -1.0, 1.0)
    g_l, g_r = 1.0 + s, 1.0 - s
    ca, ta, ra = p.cam_amp, p.thigh_amp, p.roll_amp
    return xp.stack([
        g_l * ra * series_dphi(spec[..., I_S_HIP], phi_l, xp),
        g_l * ca * series_dphi(spec[..., I_S_CAM], phi_l, xp),
        g_l * ta * series_dphi(spec[..., I_S_THIGH], phi_l, xp),
        -g_r * ra * series_dphi(spec[..., I_S_HIP], phi_r, xp),
        -g_r * ca * series_dphi(spec[..., I_S_CAM], phi_r, xp),
        -g_r * ta * series_dphi(spec[..., I_S_THIGH], phi_r, xp),
    ], axis=-1)


def impedance(spec, phi, p: GaitParams, xp=jnp):
    """(kp[...,6], kd[...,6]) from the phase-scheduled profiles, one profile read pi+Delta
    later by the right leg (a gain has no handedness)."""
    phi_l, phi_r = phases(spec, phi, p, xp)
    # the profile's LEVEL is a0: divide the normalized series by w0 so a0 = +-1 alone spans the
    # whole exp-map range and the harmonics modulate around it (clipped to +-1 inside exp_map)
    w0 = float(WEIGHTS[0])
    kp_l = exp_map(series(spec[..., I_KP], phi_l, xp) / w0, p.imp_kp_up, p.imp_kp_dn, xp)
    kp_r = exp_map(series(spec[..., I_KP], phi_r, xp) / w0, p.imp_kp_up, p.imp_kp_dn, xp)
    kd_l = exp_map(series(spec[..., I_KD], phi_l, xp) / w0, p.imp_kd_up, p.imp_kd_dn, xp)
    kd_r = exp_map(series(spec[..., I_KD], phi_r, xp) / w0, p.imp_kd_up, p.imp_kd_dn, xp)
    kp0 = xp.asarray(p.drive_kp)
    kd0 = xp.asarray(p.drive_kd)
    kp = xp.stack([kp_l, kp_l, kp_l, kp_r, kp_r, kp_r], axis=-1) * kp0
    kd = xp.stack([kd_l, kd_l, kd_l, kd_r, kd_r, kd_r], axis=-1) * kd0
    return kp, kd


def reflexes(spec, roll, roll_rate, pitch, pitch_rate, p: GaitParams, xp=jnp):
    """(u_roll, u_pitch): the learned roll reflex (gains latched, feedback every tick) and the
    fixed pitch reflex. Both scalars per env."""
    r = xp.clip(spec[..., I_REFLEX], -1.0, 1.0)
    u_roll = (p.reflex_kp_scale * r[..., 0] * roll + p.reflex_kd_scale * r[..., 1] * roll_rate
              + p.reflex_bias_scale * r[..., 2])
    u_pitch = -xp.clip(p.pitch_kp * pitch + p.pitch_kd * pitch_rate + p.pitch_bias,
                       -p.pitch_clip, p.pitch_clip)
    return u_roll, u_pitch


def assemble(spec, residual, phi, roll, roll_rate, pitch, pitch_rate, nominal, p: GaitParams,
             xp=jnp):
    """Full control law for one tick: (target[...,6], kp[...,6], kd[...,6], q_ref[...,6]).

    target = feedforward(spec, phi) + reflexes + residual_scale * residual
      pitch reflex on the thighs (+,-); roll reflex on the hips (+,+)."""
    q_ref = feedforward(spec, phi, nominal, p, xp)
    u_roll, u_pitch = reflexes(spec, roll, roll_rate, pitch, pitch_rate, p, xp)
    add = xp.stack([u_roll, xp.zeros_like(u_roll), u_pitch,
                    u_roll, xp.zeros_like(u_roll), -u_pitch], axis=-1)
    target = q_ref + add + p.residual_scale * xp.clip(residual, -1.0, 1.0)
    kp, kd = impedance(spec, phi, p, xp)
    return target, kp, kd, q_ref


def knobs(spec, p: GaitParams, xp=jnp):
    """k = (Delta/Delta_max, s, o/o_max) in [-1,1]^5 — what the standing price and the symmetry
    loss read. In action units the knobs are already normalized, so this is a slice."""
    return xp.clip(spec[..., 39:44], -1.0, 1.0)


def stance_indicator(phi, stance_ratio, xp=jnp, edge=0.1 * 2.0 * np.pi):
    """Smooth expected-stance indicator in [0,1] for the LEFT foot on the latched clock: ~1 in
    [0, 2 pi sr), raised-cosine edges. The right foot reads it at phi_R. Vectorized copy of
    walk_mit's."""
    center = np.pi * stance_ratio
    half = np.pi * stance_ratio
    dc = xp.mod(phi - center + np.pi, 2.0 * np.pi) - np.pi
    x = xp.clip((half - xp.abs(dc)) / edge + 0.5, 0.0, 1.0)
    return 0.5 - 0.5 * xp.cos(np.pi * x)


# ------------------------------------------------------------------ the mirror (§06)
def mirror_action(a, xp=jnp):
    """M on the action: series and impedance profiles unchanged, frequency unchanged, reflex
    kp/kd unchanged and bias negated, the five knobs negated, the residual L<->R and negated."""
    out = a
    out = out.at[..., 38].set(-a[..., 38]) if hasattr(out, "at") else _np_set(out, 38, -a[..., 38])
    for i in KNOB_IDX:
        out = out.at[..., i].set(-a[..., i]) if hasattr(out, "at") else _np_set(out, i, -a[..., i])
    res = -a[..., SPEC_DIM:][..., MIRROR_PERM]
    return xp.concatenate([out[..., :SPEC_DIM], res], axis=-1)


def _np_set(x, i, v):
    x = np.array(x, copy=True)
    x[..., i] = v
    return x


def mirror_frame(f, xp=jnp):
    """M on one 34-dim per-tick frame [q6 v6 tau6 grav3 gyro3 lpyaw1 phase2 prev_res6 heading1].
    Joints: L<->R and negate. Gravity (gx,gy,gz)->(gx,-gy,gz). Gyro (wx,wy,wz)->(-wx,wy,-wz).
    LP yaw negates. Phase: handled by the caller (it needs Delta). prev residual as joints.
    Heading negates for the same reason the yaw rate does: the mirrored robot that was drifting
    left is drifting right, so a policy symmetric under M corrects both the same way."""
    q, v, t = f[..., 0:6], f[..., 6:12], f[..., 12:18]
    g, w, y = f[..., 18:21], f[..., 21:24], f[..., 24:25]
    ph, r, h = f[..., 25:27], f[..., 27:33], f[..., 33:34]
    sw = lambda x: -x[..., MIRROR_PERM]
    g2 = xp.stack([g[..., 0], -g[..., 1], g[..., 2]], axis=-1)
    w2 = xp.stack([-w[..., 0], w[..., 1], -w[..., 2]], axis=-1)
    return xp.concatenate([sw(q), sw(v), sw(t), g2, w2, -y, ph, sw(r), -h], axis=-1)


def mirror_phase(cos_sin, delta_rad, xp=jnp):
    """Mirrored clock: phi' = phi - pi - Delta (the mirrored left leg is where the right was)."""
    c, s = cos_sin[..., 0], cos_sin[..., 1]
    cd, sd = xp.cos(np.pi + delta_rad), xp.sin(np.pi + delta_rad)
    # cos(phi - a) = c cd + s sd ; sin(phi - a) = s cd - c sd
    return xp.stack([c * cd + s * sd, s * cd - c * sd], axis=-1)
