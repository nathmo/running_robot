"""DASH-01 Walker v2 gait spec — LATCHED, one series + five relationship knobs (pure numpy, Pi-shareable).

Design ground truth: the "DASH-01 Walker v2" artifact (rev 2026-09-09), §02 / §06. This module is
the generator the artifact draws between the LATCH REGISTER and the PLANT: it turns the live
44-dim gait spec + the current phase into 6 PD targets and 6 (kp, kd) gain multipliers.

Action vector — 50 dims, every entry in [-1, 1]:

    [ S cam    a0,a1,b1,a2,b2,a3,b3   (7) ]   one Fourier series per joint family, N=3,
    [ S thigh                        (7) ]   shared by BOTH legs (the right leg reads the same
    [ S hip_roll                     (7) ]   series mirrored + shifted, see below)
    [ kp(phi)                        (7) ]   stiffness profile on phase, series -> exp map
    [ kd(phi)                        (7) ]   damping profile on phase, soften-only
    [ freq_raw                       (1) ]   linear -> cfg.gait_freq_hz (0.5 .. 5 Hz)
    [ reflex kp, kd, bias            (3) ]   learned roll reflex GAINS (the multiplication runs
                                             every tick; only the gains are latched)
    [ Delta                          (1) ]   lag: right footfall lands pi + Delta after the left
    [ s                              (1) ]   stride asymmetry: g_L = 1 + s, g_R = 1 - s
    [ o cam, o thigh, o hip          (3) ]   joint bias, enters BOTH legs with the same sign
    ---------------------------------------- 44 = the SPEC, latched once per cycle at phi-wrap
    [ residual r0..r5                (6) ]   per-tick position correction, +-residual_scale rad
    ---------------------------------------- 50

The generator (joint space, per family, A = family amplitude, n = per-episode stance nominal):

    u_L(phi) = n_L + (1 + s) * A * S(phi)               + o
    u_R(phi) = n_R - (1 - s) * A * S(phi - pi - Delta)  + o

The structural minus + half-cycle shift IS the mirror; knobs at zero is exactly the mirror-symmetric
gait (u_L(phi) + u_R(phi + pi) = 2n <=> Delta = s = o = 0). Asymmetry is therefore five priced
numbers, and shape asymmetry (a different waveform per leg, e.g. a duty-cycle limp) is
unrepresentable. The offset o enters (+,+) because the FK test on the env's own in-air rig showed
that joint pattern is the ANTISYMMETRIC one for every joint family (thigh (+,-) moves both feet the
same way in x; hip_roll (+,-) moves them oppositely in y). That same test is why the roll reflex
enters (+,+) here — fourier_gait.py wires it (+,-), which is a symmetric widen/narrow and cannot
make a directional lateral capture step. The fixed pitch reflex (thigh (+,-) = both feet the same
way in x) checks out and is kept as is.

Impedance: kp(phi) and kd(phi) are ONE profile with no handedness, each leg reading it at its own
phase (left at phi, right at phi - pi - Delta) so stance stiffness lands on whichever leg is in
stance. Exp map with the measured MIT-frame headroom: kp x2.5 / /3, kd soften-only /4.

Actuator/ctrl order matches model/dash01.xml <actuator>:
    0 hip_roll_L   1 cam_L   2 thigh_L   3 hip_roll_R   4 cam_R   5 thigh_R
"""
import numpy as np

from fourier_gait import _weights, _harmonic_index, _series, stance_indicator  # noqa: F401

HIP_ROLL_L, CAM_L, THIGH_L, HIP_ROLL_R, CAM_R, THIGH_R = range(6)
N_RESIDUAL = 6
N_REFLEX = 3
N_KNOBS = 5          # Delta, s, o_cam, o_thigh, o_hip
# per-episode L/R mirror of the actuator vector: swap legs, negate (every joint family is (+,-)
# symmetric under the mirror -- the FK sign rule)
MIRROR_PERM6 = np.array([3, 4, 5, 0, 1, 2])
MIRROR_SIGN6 = -np.ones(6)


def per_joint(n):
    return 1 + 2 * n


class Layout:
    """Index map of the 50-dim action for N harmonics (N=3 -> 44 + 6). Everything downstream
    (env, masked policy, diagnostics, deploy) reads slices from here, never hard-coded ints."""

    def __init__(self, n_harmonics=3):
        self.N = int(n_harmonics)
        pj = per_joint(self.N)
        k = 0
        self.s_cam = slice(k, k + pj); k += pj
        self.s_thigh = slice(k, k + pj); k += pj
        self.s_hip = slice(k, k + pj); k += pj
        self.kp_prof = slice(k, k + pj); k += pj
        self.kd_prof = slice(k, k + pj); k += pj
        self.freq = k; k += 1
        self.reflex = slice(k, k + N_REFLEX); k += N_REFLEX
        self.delta = k; k += 1
        self.stride = k; k += 1
        self.offset = slice(k, k + 3); k += 3
        self.spec_dim = k
        self.residual = slice(k, k + N_RESIDUAL); k += N_RESIDUAL
        self.action_dim = k
        self.knobs = [self.delta, self.stride, self.offset.start, self.offset.start + 1,
                      self.offset.start + 2]

    # ---- the action-side mirror (for the symmetry/equivariance loss) ----
    def mirror_perm_sign(self):
        """(perm, sign) such that a_mirrored = sign * a[perm]: negate the five knobs and the
        reflex bias, swap+negate the residual; series / profiles / frequency / reflex gains are
        invariant (S is shared by both legs, a gain has no handedness)."""
        perm = np.arange(self.action_dim)
        sign = np.ones(self.action_dim)
        sign[self.knobs] = -1.0
        sign[self.reflex.start + 2] = -1.0          # bias
        r0 = self.residual.start
        perm[r0:r0 + 6] = r0 + MIRROR_PERM6
        sign[r0:r0 + 6] = MIRROR_SIGN6
        return perm, sign


LAYOUT = Layout(3)


def spec_dim(n_harmonics=3):
    return Layout(n_harmonics).spec_dim


def action_dim(n_harmonics=3):
    return Layout(n_harmonics).action_dim


def decode(action, layout=LAYOUT):
    """(spec[44], residual[6]) from the raw action."""
    a = np.asarray(action, dtype=float)
    return a[:layout.spec_dim], a[layout.residual]


def frequency(freq_raw, freq_range):
    """freq_raw in [-1, 1] -> Hz, linear across cfg.gait_freq_hz (latched, so it can neither pause
    nor warp mid-cycle)."""
    lo, hi = freq_range
    return lo + (hi - lo) * 0.5 * (float(np.clip(freq_raw, -1.0, 1.0)) + 1.0)


def freq_raw_of(hz, freq_range):
    """Inverse of `frequency` (library entries store a physical frequency)."""
    lo, hi = freq_range
    return float(np.clip(2.0 * (hz - lo) / (hi - lo) - 1.0, -1.0, 1.0))


def _exp_map(x, up, dn):
    """Asymmetric exp map: x in [-1,1] -> multiplier in [1/dn, up], neutral 0 -> 1.0."""
    x = np.clip(x, -1.0, 1.0)
    return np.exp(np.where(x >= 0.0, x * np.log(up), x * np.log(dn)))


def leg_phases(phi, spec, cfg, layout=LAYOUT):
    """(phi_L, phi_R): the left leg reads the series at phi, the right at phi - pi - Delta."""
    delta = float(cfg.delta_max_rad) * float(np.clip(spec[layout.delta], -1.0, 1.0))
    return float(phi), float(phi) - np.pi - delta, delta


def assemble(spec, phi, roll, roll_rate, nominal, cfg, pitch=0.0, pitch_rate=0.0,
             layout=LAYOUT, reflexes=True):
    """6 PD targets (rad) at phase phi from the LIVE spec. Residuals are added by the env.

    nominal          : 6-vector per-episode stance ctrl (already L/R mirrored).
    roll, roll_rate  : base roll (grav_y) and roll rate (gyro x) for the learned roll reflex.
    pitch,pitch_rate : base pitch (grav_x, + = nose-down) and rate for the FIXED pitch reflex.
    reflexes=False   : pure function of phi (the library solver's open-loop return map).
    """
    N = layout.N
    w = _weights(N)
    kk = _harmonic_index(N)
    phi_L, phi_R, _ = leg_phases(phi, spec, cfg, layout)
    s = float(np.clip(spec[layout.stride], -1.0, 1.0))
    gL, gR = 1.0 + s, 1.0 - s
    o = np.asarray(cfg.offset_max_rad, dtype=float) * np.clip(spec[layout.offset], -1.0, 1.0)
    cL, sL = np.cos(kk * phi_L), np.sin(kk * phi_L)
    cR, sR = np.cos(kk * phi_R), np.sin(kk * phi_R)
    out = np.empty(6)
    fams = ((layout.s_cam, float(cfg.cam_amp), CAM_L, CAM_R, 0),
            (layout.s_thigh, float(cfg.thigh_amp), THIGH_L, THIGH_R, 1),
            (layout.s_hip, float(cfg.roll_amp), HIP_ROLL_L, HIP_ROLL_R, 2))
    for sl, amp, iL, iR, oi in fams:
        coef = spec[sl]
        # +-amp clip flat-tops the (approximately) unit-bounded series; the steer gain is applied
        # AFTER the clip (bounded by amp*(1+|s|)); the env's ctrlrange clip is the last word
        dL = np.clip(amp * _series(coef, phi_L, w, N, cL, sL), -amp, amp)
        dR = np.clip(amp * _series(coef, phi_R, w, N, cR, sR), -amp, amp)
        out[iL] = nominal[iL] + gL * dL + o[oi]
        out[iR] = nominal[iR] - gR * dR + o[oi]
    if reflexes:
        # FIXED pitch reflex: symmetric fore-aft foot shift, thigh (+,-) = both feet the same way
        u_p = -np.clip(cfg.pitch_kp * pitch + cfg.pitch_kd * pitch_rate + cfg.pitch_bias,
                       -cfg.pitch_clip, cfg.pitch_clip)
        out[THIGH_L] += u_p
        out[THIGH_R] -= u_p
        # LEARNED roll reflex, gains latched, evaluated every tick, enters (+,+) = the lean: both
        # feet shift the same physical way -> a directional lateral capture step exists
        r = spec[layout.reflex]
        u = (cfg.reflex_kp_scale * float(r[0]) * roll
             + cfg.reflex_kd_scale * float(r[1]) * roll_rate
             + cfg.reflex_bias_scale * float(r[2]))
        out[HIP_ROLL_L] += u
        out[HIP_ROLL_R] += u
    return out


def gains(spec, phi, cfg, layout=LAYOUT):
    """Per-actuator (kp_scale[6], kd_scale[6]) from the latched impedance profiles, each leg at
    its own phase. Multipliers on the plant's nominal gains; neutral profile -> 1.0 everywhere."""
    N = layout.N
    w = _weights(N)
    kk = _harmonic_index(N)
    phi_L, phi_R, _ = leg_phases(phi, spec, cfg, layout)
    kp_s = np.empty(6)
    kd_s = np.empty(6)
    for ph, idx in ((phi_L, (HIP_ROLL_L, CAM_L, THIGH_L)), (phi_R, (HIP_ROLL_R, CAM_R, THIGH_R))):
        ck, sk = np.cos(kk * ph), np.sin(kk * ph)
        xp = _series(spec[layout.kp_prof], ph, w, N, ck, sk)
        xd = _series(spec[layout.kd_prof], ph, w, N, ck, sk)
        kp = float(_exp_map(xp, cfg.imp_kp_up, cfg.imp_kp_dn))
        kd = float(_exp_map(xd, cfg.imp_kd_up, cfg.imp_kd_dn))
        for i in idx:
            kp_s[i] = kp
            kd_s[i] = kd
    return kp_s, kd_s


def knob_vector(spec, layout=LAYOUT):
    """k = (Delta/Delta_max, s, o/o_max) in raw units -- the five priced numbers."""
    return np.asarray(spec, dtype=float)[layout.knobs]


def neutral_spec(layout=LAYOUT, freq_raw=0.0):
    """The standing spec: zero amplitudes, neutral gains, knobs zero, mid-range clock."""
    s = np.zeros(layout.spec_dim)
    s[layout.freq] = float(freq_raw)
    return s


def embed_reduced(theta_r, cfg, layout=LAYOUT):
    """Stage-1 search variables theta_r (17) -> full raw spec (44), the artifact's embedding:
    f, cam a0 a1 b1 a2 b2, thigh a0 a1 b1 a2 b2, Delta, s, o_cam, kp_lvl, kd_lvl, hip a1.
    Third harmonics and the hip profile beyond a1 are 0; reflex gains are 0 (open-loop solve)."""
    t = np.asarray(theta_r, dtype=float)
    assert t.size == 17, t.size
    s = neutral_spec(layout)
    s[layout.freq] = freq_raw_of(t[0], cfg.gait_freq_hz)
    s[layout.s_cam.start:layout.s_cam.start + 5] = t[1:6]
    s[layout.s_thigh.start:layout.s_thigh.start + 5] = t[6:11]
    s[layout.delta] = t[11]
    s[layout.stride] = t[12]
    s[layout.offset.start] = t[13]
    s[layout.kp_prof.start] = t[14]
    s[layout.kd_prof.start] = t[15]
    s[layout.s_hip.start + 1] = t[16]
    return np.clip(s, -1.0, 1.0)


def reduce_full(spec, cfg, layout=LAYOUT):
    """Inverse of embed_reduced on the entries it covers (other entries are dropped)."""
    s = np.asarray(spec, dtype=float)
    return np.array([frequency(s[layout.freq], cfg.gait_freq_hz),
                     *s[layout.s_cam.start:layout.s_cam.start + 5],
                     *s[layout.s_thigh.start:layout.s_thigh.start + 5],
                     s[layout.delta], s[layout.stride], s[layout.offset.start],
                     s[layout.kp_prof.start], s[layout.kd_prof.start], s[layout.s_hip.start + 1]])


def mirror_defect(spec, nominal, cfg, layout=LAYOUT, n=64):
    """max_phi |u_L(phi) + u_R(phi + pi) - (n_L + n_R)| over the three families, reflexes off:
    0 exactly when Delta = s = o = 0 (the §06 identity). A diagnostic, not a reward term."""
    worst = 0.0
    for phi in np.linspace(0.0, 2 * np.pi, n, endpoint=False):
        a = assemble(spec, phi, 0.0, 0.0, nominal, cfg, layout=layout, reflexes=False)
        b = assemble(spec, phi + np.pi, 0.0, 0.0, nominal, cfg, layout=layout, reflexes=False)
        for iL, iR in ((CAM_L, CAM_R), (THIGH_L, THIGH_R), (HIP_ROLL_L, HIP_ROLL_R)):
            worst = max(worst, abs(a[iL] + b[iR] - (nominal[iL] + nominal[iR])))
    return worst
