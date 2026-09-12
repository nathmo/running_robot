"""The v2 gait generator, in numpy. Vendored from `walk_v2/gait.py` (artifact §02, §06).

WHY A COPY AND NOT AN IMPORT
----------------------------
`walk_v2/gait.py` is backend-agnostic by design -- every function takes `xp`, and `xp=numpy` is
the deploy reference. But the module imports `jax.numpy` at the top to supply its default, and jax
is not on the robot and never will be. So the functions are copied here with `xp` removed and
numpy inlined, in the SAME arithmetic order, and `tests/test_v2_deploy.py` diffs this against the
trained law on the recorded MJX trace (`walk_v2/results/trace_mjx.json`) -- the same fixture the
two training arms cross-check each other on.

Read this next to `controller_v2.py`: this module is the pure algebra (a spec and a phase in,
targets and gains out) and knows nothing about the latch, the clock or the observation.

TWO IMPLEMENTATIONS, ON PURPOSE. The free functions below are the reference: one series call per
scheduled quantity per leg, in the trainer's order, easy to read against `walk_v2/gait.py` and
checked against the recorded MJX trace. `GaitEval` at the bottom is the same algebra batched into
a fifth of the numpy calls, and it is what the robot actually runs -- on a Pi 3B the reference is
5.2 ms of an 8.2 ms control tick, and a 100 Hz bundle has 10 ms. `tests/test_v2_deploy.py` pins
the two together EXACTLY, over specs pushed past every clip, in float32 and float64.

ACTION LAYOUT (50; all entries clipped to [-1, 1])
    [ 0: 7]  S_cam    a0, a1,b1, a2,b2, a3,b3     N=3 series, x cam_amp            latched
    [ 7:14]  S_thigh                               x thigh_amp                      latched
    [14:21]  S_hip                                 x roll_amp                       latched
    [21:28]  kp(phi) series -> exp map             x2.5 / /3 on the plant's base kp latched
    [28:35]  kd(phi) series -> exp map             soften-only /4                   latched
    [35]     freq_raw  linear -> [freq_lo, freq_hi] Hz                              latched
    [36:39]  roll reflex kp, kd, bias                                               latched
    [39]     Delta  x delta_max rad   right footfall pi+Delta after the left        latched
    [40]     s      g_L = 1+s, g_R = 1-s                                            latched
    [41:44]  o_cam, o_thigh, o_hip  x o_max, joint bias (+,+)                       latched
    [44:50]  residual r0..r5, x residual_scale rad                                  EVERY tick

    u_L(phi) = n_L + (1+s) A S(phi)             + o
    u_R(phi) = n_R - (1-s) A S(phi - pi - Delta) + o

Actuator order throughout: hip_roll_L, cam_L, thigh_L, hip_roll_R, cam_R, thigh_R.
"""
from typing import NamedTuple

import numpy as np

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
HIP_L, CAM_L, THIGH_L, HIP_R, CAM_R, THIGH_R = range(6)


def _weights():
    w = np.array([1.0] + [1.0 / np.sqrt(k) for k in range(1, N_HARMONICS + 1)])
    return w / w.sum()


WEIGHTS = _weights()                       # (4,) offset + 3 harmonics
HARMONIC_K = np.arange(1.0, N_HARMONICS + 1.0)


class GaitParams(NamedTuple):
    """Every number that turns a spec into targets + gains. Field-for-field the training
    `gait.GaitParams`, so `GaitParams.from_meta(bundle.meta["gait"])` is the whole conversion."""
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
    drive_kp: tuple              # 6, actuator order (base gains the phase profile scales)
    drive_kd: tuple

    @classmethod
    def from_meta(cls, d):
        """From the bundle's meta["gait"] blob (JSON, so every tuple arrived as a list)."""
        f = {k: (tuple(v) if isinstance(v, (list, tuple)) else float(v))
             for k, v in d.items() if k in cls._fields}
        missing = [k for k in cls._fields if k not in f]
        if missing:
            raise ValueError("the bundle's gait block is missing {} -- it was written by an "
                             "exporter this runtime does not know".format(missing))
        return cls(**f)


# ------------------------------------------------------------------ primitives
def series(coeffs, phi):
    """weighted a0 + sum_k w_k (a_k cos k phi + b_k sin k phi). coeffs [..., 7], phi scalar."""
    phi = np.asarray(phi)[..., None]
    c, s = np.cos(HARMONIC_K * phi), np.sin(HARMONIC_K * phi)
    a = coeffs[..., 1::2]
    b = coeffs[..., 2::2]
    return WEIGHTS[0] * coeffs[..., 0] + np.sum(WEIGHTS[1:] * (a * c + b * s), axis=-1)


def frequency(freq_raw, p):
    """Latched gait frequency, Hz."""
    return p.freq_lo + (p.freq_hi - p.freq_lo) * 0.5 * (np.clip(freq_raw, -1.0, 1.0) + 1.0)


def exp_map(x, up, dn):
    """[-1, 1] -> multiplier: +1 -> up, -1 -> 1/dn, 0 -> 1. up=1 makes the +side inert."""
    x = np.clip(x, -1.0, 1.0)
    return np.exp(np.where(x >= 0.0, x * np.log(up), x * np.log(dn)))


def phases(spec, phi, p):
    """(phi_L, phi_R): the right leg reads everything pi + Delta later."""
    delta = p.delta_max * np.clip(spec[..., I_DELTA], -1.0, 1.0)
    return phi, phi - np.pi - delta


# ------------------------------------------------------------------ the generator
def feedforward(spec, phi, nominal, p):
    """Series + knobs only: the reference joint targets q_ref(phi) [..., 6], no reflex and no
    residual."""
    phi_l, phi_r = phases(spec, phi, p)
    s = np.clip(spec[..., I_S], -1.0, 1.0)
    g_l, g_r = 1.0 + s, 1.0 - s
    o = np.clip(spec[..., I_O], -1.0, 1.0) * np.asarray(p.o_max)
    ca, ta, ra = p.cam_amp, p.thigh_amp, p.roll_amp
    d_cam_l = np.clip(ca * series(spec[..., I_S_CAM], phi_l), -ca, ca)
    d_cam_r = np.clip(ca * series(spec[..., I_S_CAM], phi_r), -ca, ca)
    d_th_l = np.clip(ta * series(spec[..., I_S_THIGH], phi_l), -ta, ta)
    d_th_r = np.clip(ta * series(spec[..., I_S_THIGH], phi_r), -ta, ta)
    d_hip_l = np.clip(ra * series(spec[..., I_S_HIP], phi_l), -ra, ra)
    d_hip_r = np.clip(ra * series(spec[..., I_S_HIP], phi_r), -ra, ra)
    n = np.asarray(nominal)
    return np.stack([
        n[..., HIP_L] + g_l * d_hip_l + o[..., 2],
        n[..., CAM_L] + g_l * d_cam_l + o[..., 0],
        n[..., THIGH_L] + g_l * d_th_l + o[..., 1],
        n[..., HIP_R] - g_r * d_hip_r + o[..., 2],
        n[..., CAM_R] - g_r * d_cam_r + o[..., 0],
        n[..., THIGH_R] - g_r * d_th_r + o[..., 1],
    ], axis=-1)


def impedance(spec, phi, p):
    """(kp[6], kd[6]) from the phase-scheduled profiles, the right leg reading the same profile
    pi + Delta later (a gain has no handedness). Already multiplied by the plant's base gains, so
    what comes out is what goes into the force-control frame."""
    phi_l, phi_r = phases(spec, phi, p)
    # the profile's LEVEL is a0: divide the normalized series by w0 so a0 = +-1 alone spans the
    # whole exp-map range and the harmonics modulate around it (clipped to +-1 inside exp_map)
    w0 = float(WEIGHTS[0])
    kp_l = exp_map(series(spec[..., I_KP], phi_l) / w0, p.imp_kp_up, p.imp_kp_dn)
    kp_r = exp_map(series(spec[..., I_KP], phi_r) / w0, p.imp_kp_up, p.imp_kp_dn)
    kd_l = exp_map(series(spec[..., I_KD], phi_l) / w0, p.imp_kd_up, p.imp_kd_dn)
    kd_r = exp_map(series(spec[..., I_KD], phi_r) / w0, p.imp_kd_up, p.imp_kd_dn)
    kp0 = np.asarray(p.drive_kp)
    kd0 = np.asarray(p.drive_kd)
    kp = np.stack([kp_l, kp_l, kp_l, kp_r, kp_r, kp_r], axis=-1) * kp0
    kd = np.stack([kd_l, kd_l, kd_l, kd_r, kd_r, kd_r], axis=-1) * kd0
    return kp, kd


def reflexes(spec, roll, roll_rate, pitch, pitch_rate, p):
    """(u_roll, u_pitch): the learned roll reflex (gains latched, feedback every tick) and the
    fixed pitch reflex. Both scalars.

    `roll` and `pitch` are GRAVITY COMPONENTS, not angles: grav[1] and grav[0] of world-DOWN
    expressed in body axes, exactly as the sim reads them."""
    r = np.clip(spec[..., I_REFLEX], -1.0, 1.0)
    u_roll = (p.reflex_kp_scale * r[..., 0] * roll + p.reflex_kd_scale * r[..., 1] * roll_rate
              + p.reflex_bias_scale * r[..., 2])
    u_pitch = -np.clip(p.pitch_kp * pitch + p.pitch_kd * pitch_rate + p.pitch_bias,
                       -p.pitch_clip, p.pitch_clip)
    return u_roll, u_pitch


def assemble(spec, residual, phi, roll, roll_rate, pitch, pitch_rate, nominal, p):
    """Full control law for one tick: (target[6], kp[6], kd[6], q_ref[6]).

    target = feedforward(spec, phi) + reflexes + residual_scale * residual
      pitch reflex on the thighs (+,-); roll reflex on the hips (+,+)."""
    q_ref = feedforward(spec, phi, nominal, p)
    u_roll, u_pitch = reflexes(spec, roll, roll_rate, pitch, pitch_rate, p)
    add = np.stack([u_roll, np.zeros_like(u_roll), u_pitch,
                    u_roll, np.zeros_like(u_roll), -u_pitch], axis=-1)
    target = q_ref + add + p.residual_scale * np.clip(residual, -1.0, 1.0)
    kp, kd = impedance(spec, phi, p)
    return target, kp, kd, q_ref


def slew_limit(target, prev_target, prev_vel, vel_limit, accel_limit, dt):
    """The no-load command cap, vendored from `walk_v2/drive.py`: the commanded target may not
    move faster than the motor can. Returns (target, commanded joint velocity)."""
    v_des = (target - prev_target) / dt
    if accel_limit > 0.0:
        dv = accel_limit * dt
        v_des = np.clip(v_des, prev_vel - dv, prev_vel + dv)
    v_des = np.clip(v_des, -vel_limit, vel_limit)
    return prev_target + v_des * dt, v_des


def wrap_pi(x):
    """Wrap to (-pi, pi]."""
    return x - 2.0 * np.pi * np.floor((x + np.pi) / (2.0 * np.pi))


# ------------------------------------------------------------------ the deployed hot path
# Rows of the batched coefficient block, in the order the series are evaluated: each of the five
# scheduled quantities at the LEFT phase and then at the RIGHT one.
_SERIES_ROWS = np.array([list(range(s.start, s.stop))
                         for s in (I_S_CAM, I_S_THIGH, I_S_HIP, I_KP, I_KD)])
_SERIES_IDX = np.repeat(_SERIES_ROWS, 2, axis=0)          # (10, 7)
I_CAM_L, I_CAM_R, I_TH_L, I_TH_R, I_HIP_L, I_HIP_R = range(6)


class GaitEval:
    """One tick of the generator -- (target, kp, kd, q_ref) -- as `assemble` computes it, in about
    a fifth of the time. This is what the robot runs; `assemble` above is the reference it is
    pinned to, exactly, by `tests/test_v2_deploy.py`.

    WHY. Measured on the Pi 3B with the real bundle, a control tick costs 8.2 ms against a 10 ms
    budget at 100 Hz, and 5.2 ms of it is this generator -- against 1.8 ms for both neural nets.
    Almost none of that is arithmetic. `assemble` evaluates the Fourier series TEN times (cam,
    thigh, hip, kp, kd, each at the left and the right phase), clips twenty-one scalars and stacks
    four six-vectors, and every one of those is a separate numpy dispatch over three to seven
    elements: a few hundred flops behind ~50 interpreter round trips at 10-40 us each on a 1.2 GHz
    A53. So the fix is not faster maths, it is fewer calls. The ten series share one (10, 7)
    coefficient block and one weighted sum; the scalar clips are Python min/max; the six-vectors
    are written into preallocated buffers. Measured 3.62 ms -> 0.76 ms on the robot, bit-identical.

    THE COSINES ARE STILL FOUR (3,) CALLS at the two phases, not one call over (10, 3). numpy can
    take a different SIMD path for a longer array and land a different last ULP, and bit-equality
    with the reference is the whole reason this substitution is safe to make. It costs ~30 us.

    The plant constants (`p`) and the stance (`nominal`) are bound once at construction because
    they are constants of the robot, not of the tick. A spec with leading batch dimensions is not
    handled here at all -- call `assemble` for that; this is the deployment path and the robot has
    one body. For the same reason an instance is NOT reentrant: it owns scratch buffers, so one
    per control loop, called from the thread that owns the loop. (Nothing it returns aliases them.)"""

    __slots__ = ("p", "nominal", "_idx", "_amp", "_o_max", "_kp0", "_kd0", "_log_up", "_log_dn",
                 "_w0", "_wk", "_cos", "_sin", "_add", "_kp", "_kd", "_qref")

    def __init__(self, p, nominal):
        self.p = p
        self.nominal = np.asarray(nominal, float)
        if self.nominal.shape != (6,):
            raise ValueError("the stance must be one 6-vector in actuator order, got {}"
                             .format(self.nominal.shape))
        self._idx = _SERIES_IDX
        # cam, cam, thigh, thigh, hip, hip -- the amplitude each of the first six series is scaled
        # by, and clipped to, exactly as feedforward() does one at a time
        self._amp = np.array([p.cam_amp, p.cam_amp, p.thigh_amp, p.thigh_amp,
                              p.roll_amp, p.roll_amp], float)
        self._o_max = np.asarray(p.o_max, float)
        self._kp0 = np.asarray(p.drive_kp, float)
        self._kd0 = np.asarray(p.drive_kd, float)
        # exp_map's two branches, precomputed for the four scheduled gains (kp_L, kp_R, kd_L, kd_R)
        self._log_up = np.array([np.log(p.imp_kp_up), np.log(p.imp_kp_up),
                                 np.log(p.imp_kd_up), np.log(p.imp_kd_up)])
        self._log_dn = np.array([np.log(p.imp_kp_dn), np.log(p.imp_kp_dn),
                                 np.log(p.imp_kd_dn), np.log(p.imp_kd_dn)])
        # WEIGHTS[0] stays a numpy float64 SCALAR and is not unwrapped to a Python float: under
        # NEP 50 a Python float is weak and would let the float32 spec decide the dtype of the
        # constant term, computing a0 in float32. That is a 5e-5 drift in the gains, and invisible.
        self._w0 = WEIGHTS[0]
        self._wk = WEIGHTS[1:]
        # scratch. The four RETURNED arrays are fresh every tick -- the caller keeps them (the
        # controller's previous target, the governor's command) and a reused buffer would alias.
        self._cos = np.empty((10, N_HARMONICS))
        self._sin = np.empty((10, N_HARMONICS))
        self._add = np.zeros(6)

    def __call__(self, spec, residual, phi, roll, roll_rate, pitch, pitch_rate):
        p, n = self.p, self.nominal
        # ---- the two phases. phases() itself, NOT a Python-float rewrite of it: the spec is
        #      float32, so numpy's promotion rules round `delta` -- and with it the right leg's
        #      whole phase -- to float32, and doing that arithmetic in float64 instead moves the
        #      right-leg gains by 5e-6 relative. Faithful beats tidy: the trainer is float32.
        phi_l, phi_r = phases(spec, phi, p)
        # ---- the ten series, one block. Four (3,) trig calls: see the class docstring.
        c, s = self._cos, self._sin
        c[0::2] = np.cos(HARMONIC_K * phi_l)
        s[0::2] = np.sin(HARMONIC_K * phi_l)
        c[1::2] = np.cos(HARMONIC_K * phi_r)
        s[1::2] = np.sin(HARMONIC_K * phi_r)
        coef = spec[self._idx]                                          # (10, 7)
        v = self._w0 * coef[:, 0] + np.sum(self._wk * (coef[:, 1::2] * c + coef[:, 2::2] * s),
                                           axis=-1)
        # ---- feedforward
        sk = np.clip(spec[I_S], -1.0, 1.0)      # float32 in, float32 out -- see the phases note
        g_l, g_r = 1.0 + sk, 1.0 - sk
        o = np.clip(spec[I_O], -1.0, 1.0) * self._o_max
        amp = self._amp
        d = np.clip(amp * v[:6], -amp, amp)
        q_ref = np.empty(6)
        q_ref[HIP_L] = n[HIP_L] + g_l * d[I_HIP_L] + o[2]
        q_ref[CAM_L] = n[CAM_L] + g_l * d[I_CAM_L] + o[0]
        q_ref[THIGH_L] = n[THIGH_L] + g_l * d[I_TH_L] + o[1]
        q_ref[HIP_R] = n[HIP_R] - g_r * d[I_HIP_R] + o[2]
        q_ref[CAM_R] = n[CAM_R] - g_r * d[I_CAM_R] + o[0]
        q_ref[THIGH_R] = n[THIGH_R] - g_r * d[I_TH_R] + o[1]
        # ---- the reflexes, and the target
        r = np.clip(spec[I_REFLEX], -1.0, 1.0)
        u_roll = (p.reflex_kp_scale * r[0] * roll + p.reflex_kd_scale * r[1] * roll_rate
                  + p.reflex_bias_scale * r[2])
        u_pitch = -_clip(p.pitch_kp * pitch + p.pitch_kd * pitch_rate + p.pitch_bias, p.pitch_clip)
        add = self._add
        add[HIP_L] = add[HIP_R] = u_roll
        add[THIGH_L] = u_pitch
        add[THIGH_R] = -u_pitch
        target = q_ref + add + p.residual_scale * np.clip(residual, -1.0, 1.0)
        # ---- the gains: one exp_map over the four scheduled values
        x = np.clip(v[6:] / self._w0, -1.0, 1.0)
        e = np.exp(x * np.where(x >= 0.0, self._log_up, self._log_dn))
        kp = np.empty(6)
        kp[0:3] = e[0]
        kp[3:6] = e[1]
        kd = np.empty(6)
        kd[0:3] = e[2]
        kd[3:6] = e[3]
        return target, kp * self._kp0, kd * self._kd0, q_ref


def _clip(x, c):
    """np.clip(x, -c, c) for one PYTHON float, without the dispatch. Only safe where the reference
    also works in float64 -- anything carrying the spec's float32 keeps going through np.clip."""
    return -c if x < -c else (c if x > c else x)
