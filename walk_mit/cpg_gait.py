"""Central Pattern Generator gait for DASH-01 — PURE NUMPY (Pi-shareable).

The Ijspeert-school alternative to `fourier_gait.py`. Where the Fourier module lets the policy
re-draw the whole joint waveform every control step, here the gait is produced by a small
DYNAMICAL SYSTEM — a pair of coupled amplitude-controlled phase oscillators, one per leg — and the
policy only TUNES that oscillator. This is the CPG-RL formulation of Bellegarda & Ijspeert (2022),
"CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion", itself built on the Hopf /
amplitude-phase oscillator networks of Ijspeert (2008) and the salamander CPG line at BioRob.

Oscillator (per leg i, integrated at the control rate):

    r_i''  = a * ( a/4 * (mu_i - r_i) - r_i' )                       amplitude, critically damped
    th_i'  = 2*pi*f_i + sum_j  r_j * w_ij * sin(th_j - th_i - phi_ij)  phase, with inter-leg coupling

Three properties are what we are actually testing against the Fourier stack:
  1. STRUCTURAL SMOOTHNESS. r converges to mu through a second-order critically damped filter, so a
     jittery policy output cannot become a jittery joint target. The Fourier action has no such
     filter — it is read straight into the waveform. (This is the direct probe of the m7 ~6 Hz
     footfall limit cycle: if the chatter needs a fast open path from action to target, the CPG
     cannot express it.)
  2. PHASE IS STATE, NOT A CLOCK. Each leg owns its phase and the legs are held in antiphase by a
     coupling term, so the gait can be perturbed, drift, and RE-SYNCHRONISE. The Fourier stack has
     one global clock and hard-codes the right leg at phi + pi, so it can never phase-reset.
  3. FEW PARAMETERS. 9 gait dims here vs 21 there (both plus the 6 residuals) — the CPG carries the
     locomotion prior in its structure instead of asking the policy to relearn it every step.

Oscillator -> joint mapping. CPG-RL writes the mapping in FOOT space (phase drives fore-aft travel
plus a swing-phase lift, then IK). We keep that, because on this robot the naive joint-space reading
is measurably wrong: from the stand pose both cam and thigh move the toe mostly FORE-AFT and either
sign of cam shortens the near-straight leg, so "thigh = swing, cam = lift" would not produce a step.
The IK is a measured lookup table (see build_cpg_lut.py) rather than a formula, since the sagittal
leg is a cam crank driving a parallel 4-bar. The mapping is FIXED — the policy tunes the oscillator,
never the mapping. That is the whole point of the comparison.

    x_i(th) = stride * r_i * cos(th_i)                  fore-aft foot travel
    z_i(th) = clearance * r_i * bump(th_i, stance_ratio) foot lift, nonzero only in the swing window

`bump` is a raised cosine over the SWING window of the phase-gated contact schedule, so the CPG's
own idea of which phase is stance agrees, by construction, with the window the reward pays for
(fourier_gait.stance_indicator). At stance_ratio 0.5 this is exactly the paper's `sin(th) > 0` rule;
below 0.5 the lift window widens with the flight phase the curriculum is demanding.

Action layout (all entries in [-1, 1]):
    [ mu:     mu_L, mu_R      (2) ]   amplitude setpoint -> stride/clearance scale
    [ freq:   f_L, f_R        (2) ]   intrinsic frequency, mapped into cfg.gait_freq_hz
    [ psi                     (1) ]   inter-leg phase bias around antiphase, +- cfg.cpg_psi_range
    [ reflex: kp, kd, bias    (3) ]   the SAME learned abduction reflex as fourier_gait
    [ steer:  stride, width   (2) ]   optional, same two mechanisms as fourier_gait
    [ residual: r0..r5        (6) ]   optional PMTG channel (cfg.cpg_residual off = ablation arm)
Total = 8 (+2 steer) + 6 = 14 / 16 with steering, or 8 / 10 with residuals disabled.

Actuator/ctrl order matches model/dash01.xml <actuator>, as in fourier_gait:
    0 hip_roll_L   1 cam_L   2 thigh_L   3 hip_roll_R   4 cam_R   5 thigh_R
"""
import os
import numpy as np

HIP_ROLL_L, CAM_L, THIGH_L, HIP_ROLL_R, CAM_R, THIGH_R = range(6)
N_MU = 2
N_FREQ = 2
N_PSI = 1
N_REFLEX = 3      # kp, kd, bias — identical to fourier_gait
N_STEER = 2
N_RESIDUAL = 6

_LUT = {}


def _lut_path(name="cpg_foot_lut.npz"):
    if os.path.isabs(name):
        return name
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", name)


def load_lut(name="cpg_foot_lut.npz"):
    """Load (and cache) a measured foot-IK table built by build_cpg_lut.py.

    Cached PER PATH, so several tables can coexist in one process and, more importantly, so a run
    is pinned to the table its config names — adding a new table for a new arm can never silently
    change the mapping under an arm that is already training."""
    key = _lut_path(name)
    if key not in _LUT:
        z = np.load(key, allow_pickle=True)
        _LUT[key] = dict(dx_grid=z["dx_grid"], dz_grid=z["dz_grid"], inv=z["inv"],
                         nominal_toe=z["nominal_toe"])
    return _LUT[key]


def spec_dim(n_steer=0):
    """Size of the gait SPEC — the slice the phase-gated coef_rate penalty bills. As in
    fourier_gait, the per-step residuals are excluded by design."""
    return N_MU + N_FREQ + N_PSI + N_REFLEX + n_steer


def action_dim(n_steer=0, residual=True):
    return spec_dim(n_steer) + (N_RESIDUAL if residual else 0)


def decode(action, n_steer=0, residual=True):
    """Split the raw [-1,1] action into (mu, freq_raw, psi_raw, reflex, steer, residual).
    `mu` and `freq_raw` are 2-vectors ordered [left, right]."""
    a = np.asarray(action, dtype=float)
    mu = a[0:2]
    freq_raw = a[2:4]
    psi_raw = float(a[4])
    k = 5
    reflex = a[k:k + N_REFLEX]
    k += N_REFLEX
    steer = a[k:k + n_steer] if n_steer else np.zeros(N_STEER)
    k += n_steer
    res = a[k:k + N_RESIDUAL] if residual else np.zeros(N_RESIDUAL)
    return mu, freq_raw, psi_raw, reflex, steer, res


def frequency(freq_raw, freq_range):
    """Map freq_raw in [-1,1] to an intrinsic frequency (Hz) in freq_range — same convention as
    fourier_gait.frequency, so the two arms get an identical cadence envelope."""
    lo, hi = freq_range
    return lo + (hi - lo) * 0.5 * (np.asarray(freq_raw, dtype=float) + 1.0)


def amplitude_setpoint(mu_raw, cfg):
    """Map mu_raw in [-1,1] to the amplitude setpoint in [cpg_mu_min, cpg_mu_max]. r converges to
    mu, and the foot excursion is proportional to r, so mu IS the stride-length knob."""
    lo, hi = cfg.cpg_mu_min, cfg.cpg_mu_max
    return lo + (hi - lo) * 0.5 * (np.clip(np.asarray(mu_raw, dtype=float), -1.0, 1.0) + 1.0)


def integrate(state, mu, f_hz, psi, dt, cfg):
    """Advance the two-oscillator network by dt. `state` = (r, rdot, th), each a 2-vector
    [left, right]; returns the new tuple. Sub-stepped so a stiff `a` stays stable at any control
    rate (the 200 Hz stack runs dt = 5 ms, but the same code must hold at 50 Hz).

        r'' = a (a/4 (mu - r) - r')                       critically damped -> no overshoot
        th' = 2 pi f + sum_j r_j w sin(th_j - th_i - phi_ij)

    The coupling target phi_ij for a biped is antiphase (pi), shifted by the policy's psi. Coupling
    is scaled by the NEIGHBOUR's amplitude r_j (Ijspeert's convention): a leg that is barely
    oscillating does not drag its partner around."""
    r, rdot, th = (np.array(x, dtype=float) for x in state)
    a = float(cfg.cpg_a)
    w = float(cfg.cpg_coupling)
    n = max(1, int(cfg.cpg_substeps))
    h = dt / n
    target = np.pi + psi                       # desired (th_R - th_L)
    for _ in range(n):
        rddot = a * (0.25 * a * (mu - r) - rdot)
        rdot = rdot + h * rddot
        r = np.maximum(0.0, r + h * rdot)      # amplitude is a radius: never negative
        d = th[1] - th[0]                      # phase of R relative to L
        # symmetric pull toward `target`: L is pushed one way, R the other
        coupL = r[1] * w * np.sin(d - target)
        coupR = r[0] * w * np.sin(-d + target)
        th = th + h * (2.0 * np.pi * f_hz + np.array([coupL, coupR]))
    th = np.mod(th, 2.0 * np.pi)
    return r, rdot, th


def swing_bump(th, stance_ratio, ):
    """Foot-lift profile: a raised cosine that is 0 through the whole STANCE window
    [0, 2*pi*sr) and rises to 1 at mid-swing. Shares the phase convention of
    fourier_gait.stance_indicator, so the leg is off the ground exactly when the reward's
    phase-gated contact term expects it to be. Vectorised over th."""
    two_pi = 2.0 * np.pi
    sr = float(np.clip(stance_ratio, 0.05, 0.95))
    start = two_pi * sr                       # swing window is [2*pi*sr, 2*pi)
    width = two_pi - start
    u = (np.mod(np.asarray(th, dtype=float) - start, two_pi)) / width
    return np.where(u <= 1.0, 0.5 - 0.5 * np.cos(two_pi * np.clip(u, 0.0, 1.0)), 0.0)


def foot_ik(dx, dz, lut):
    """Bilinear lookup of the measured inverse map: desired toe delta (dx fore-aft, dz up, metres,
    body frame) -> (dcam, dthigh) ctrl delta for the LEFT leg. Inputs are clipped to the table's
    box, so the mapping saturates instead of extrapolating into the 4-bar's folded branch."""
    xg, zg, inv = lut["dx_grid"], lut["dz_grid"], lut["inv"]
    x = np.clip(dx, xg[0], xg[-1])
    z = np.clip(dz, zg[0], zg[-1])
    fx = (x - xg[0]) / (xg[-1] - xg[0]) * (len(xg) - 1)
    fz = (z - zg[0]) / (zg[-1] - zg[0]) * (len(zg) - 1)
    i0 = int(np.clip(np.floor(fx), 0, len(xg) - 2))
    j0 = int(np.clip(np.floor(fz), 0, len(zg) - 2))
    tx, tz = fx - i0, fz - j0
    return ((1 - tx) * (1 - tz) * inv[i0, j0] + tx * (1 - tz) * inv[i0 + 1, j0]
            + (1 - tx) * tz * inv[i0, j0 + 1] + tx * tz * inv[i0 + 1, j0 + 1])


def assemble(state, reflex, roll, roll_rate, nominal, cfg, stance_ratio=0.5,
             pitch=0.0, pitch_rate=0.0, steer=None, lut=None):
    """Return the 6 PD motor targets (rad) for the current oscillator state.

    Deliberately mirrors fourier_gait.assemble: same nominal-relative construction, the same
    mirrored L/R sign convention, the SAME learned abduction reflex and the SAME fixed pitch
    reflex, and the same two steering mechanisms. Only the waveform source differs — which is what
    makes the CPG-vs-Fourier comparison an experiment about the generator rather than about the
    balance controller.

    state : (r, rdot, th) from integrate(); only r and th are used here.
    """
    lut = lut if lut is not None else load_lut(getattr(cfg, 'cpg_lut', 'cpg_foot_lut.npz'))
    r, _, th = state
    if steer is None:
        s_stride = s_width = 0.0
    else:
        s_stride = cfg.steer_stride_scale * float(steer[0])
        s_width = cfg.steer_width_scale * float(steer[1])
    gain = np.array([1.0 + s_stride, 1.0 - s_stride])          # [L, R] differential stride

    # --- foot trajectory from the oscillator, then measured IK -> cam/thigh -------------------
    dx = cfg.cpg_stride * gain * r * np.cos(th)
    dz = cfg.cpg_clearance * r * swing_bump(th, stance_ratio)
    jL = foot_ik(dx[0], dz[0], lut)
    jR = foot_ik(dx[1], dz[1], lut)
    # the +-amp clip mirrors fourier_gait: keep the coupled cam-thigh 4-bar inside its assemblable
    # band without assuming an absolute cam sign or centre (the env's ctrlrange clip is the last word)
    ca, ta = cfg.cam_amp, cfg.thigh_amp
    cam_L, thigh_L = np.clip(jL[0], -ca, ca), np.clip(jL[1], -ta, ta)
    cam_R, thigh_R = np.clip(jR[0], -ca, ca), np.clip(jR[1], -ta, ta)
    cam_L = nominal[CAM_L] + cam_L
    thigh_L = nominal[THIGH_L] + thigh_L
    cam_R = nominal[CAM_R] - cam_R       # right leg: mirrored axes (same convention as fourier_gait)
    thigh_R = nominal[THIGH_R] - thigh_R

    # --- FIXED pitch reflex: symmetric fore-aft thigh shift (capture point). Identical to
    # fourier_gait, including being applied AFTER the amp clip so a big stride cannot eat it.
    u_p = -np.clip(cfg.pitch_kp * pitch + cfg.pitch_kd * pitch_rate + cfg.pitch_bias,
                   -cfg.pitch_clip, cfg.pitch_clip)
    thigh_L += u_p
    thigh_R -= u_p
    # --- hip_roll: the learned linear abduction reflex, verbatim from fourier_gait --------------
    u = (cfg.reflex_kp_scale * reflex[0] * roll
         + cfg.reflex_kd_scale * reflex[1] * roll_rate
         + cfg.reflex_bias_scale * reflex[2])
    hr_L = nominal[HIP_ROLL_L] + u + s_width
    hr_R = nominal[HIP_ROLL_R] - u + s_width

    out = np.empty(6)
    out[HIP_ROLL_L], out[CAM_L], out[THIGH_L] = hr_L, cam_L, thigh_L
    out[HIP_ROLL_R], out[CAM_R], out[THIGH_R] = hr_R, cam_R, thigh_R
    return out
