"""Fourier gait reconstruction for DASH-01 — PURE NUMPY (Pi-shareable).

The policy emits, at EVERY 50 Hz control step, a gait-spec action that this module turns into the
6 PD motor targets at the current gait phase phi (CPG-RL-style per-step re-parameterization,
Bellegarda & Ijspeert 2022), plus a small per-step residual channel (PMTG-style, Iscen et al. 2018)
that the env adds on top of the reconstruction:

  - cam + thigh (the sagittal 4-bar, propulsion): a Fourier series over phi, SYMMETRIC across legs
    (right = mirror of left, half a cycle out of phase), centered on the per-episode stance pose.
    Harmonics are weighted DOWN by 1/sqrt(k) so the offset + low harmonics dominate while leaving
    enough high-harmonic authority for the impulsive push-off a running stride needs (the old 1/k
    weighting biased everything toward near-sinusoidal walking).
  - hip_roll (abduction, lateral balance): a learned LINEAR REFLEX on roll / roll-rate evaluated at
    50 Hz (feedback, not a function of phi). Inert while roll is locked (M1-M3 rails).
  - pitch reflex (sagittal balance): a FIXED (hand-tuned, not learned) PD from base pitch /
    pitch-rate onto a SYMMETRIC fore-aft thigh offset (both feet shift together toward a fall to
    catch the CoM — capture-point control). Always active, evaluated at 50 Hz; inert while pitch is
    locked (M1-M2). Config gains cfg.pitch_k*; assemble() reads pitch/pitch_rate kwargs.
  - steer (2): the ASYMMETRY channel — the only way this action space can turn. Everything else
    here is mirror-symmetric by construction (right = negated left, half a cycle out of phase), and
    a symmetric gait can only go straight. See `assemble` for the two mechanisms. Part of the gait
    SPEC (coef_rate-billed), because a turn is a gait change, not a one-off correction.
  - residual (6): per-step joint-target corrections added AFTER reconstruction — the fast feedback
    channel the literature says running requires (one-off stumble corrections don't have to rewrite
    the whole gait spec through the coef_rate penalty). Scaled by cfg.residual_scale in the env.

Action layout (all entries in [-1, 1]), for N harmonics:
    [ cam:   a0,a1,b1,...,aN,bN   (1+2N) ]
    [ thigh: a0,a1,b1,...,aN,bN   (1+2N) ]
    [ freq_raw                    (1)    ]
    [ reflex: kp, kd, bias        (3)    ]
    [ steer:  stride, width       (2)    ]   <- end of the GAIT SPEC (spec_dim, coef_rate-gated)
    [ residual: r0..r5            (6)    ]   <- per-step, NOT part of the spec / coef_rate
Total = 2*(1+2N) + 6 + 6   (N=3 -> 26).

Actuator/ctrl order matches model/dash01.xml <actuator>:
    0 hip_roll_L   1 cam_L   2 thigh_L   3 hip_roll_R   4 cam_R   5 thigh_R
"""
import numpy as np

HIP_ROLL_L, CAM_L, THIGH_L, HIP_ROLL_R, CAM_R, THIGH_R = range(6)
N_REFLEX = 3    # kp, kd, bias
N_STEER = 2     # stride asymmetry, stance-width asymmetry
N_RESIDUAL = 6  # one per actuator


def per_joint(n_harmonics):
    return 1 + 2 * n_harmonics


def spec_dim(n_harmonics, n_steer=0):
    """Size of the gait SPEC (Fourier coeffs + frequency + reflex + steer) — the slice the
    phase-gated coef_rate penalty bills. Residuals are per-step by design and excluded.

    n_steer defaults to 0 — no steering dims at all, so every pre-steering preset keeps its
    original 24-dim action and its checkpoints keep loading. Steering is opt-in end to end: a bare
    call to anything in this module gives the legacy mirror-symmetric behaviour, matching
    assemble(steer=None). Only env.py, driven by cfg.steer_enable, passes N_STEER."""
    return 2 * per_joint(n_harmonics) + 1 + N_REFLEX + n_steer


def action_dim(n_harmonics, n_steer=0):
    """Full policy action: gait spec + per-step residuals."""
    return spec_dim(n_harmonics, n_steer) + N_RESIDUAL


def _weights(n_harmonics):
    """Amplitude budget: offset + harmonic k weighted by 1/sqrt(k), normalized to sum 1.
    Low-harmonic-dominated but with more high-harmonic authority than a 1/k schedule (running
    push-off is impulsive). NOTE the bound is approximate: each harmonic pair reaches sqrt(2) at
    coeffs +-1, so |series| can reach ~1.24 — the +-amp clip in assemble() flat-tops those
    extremes (intentional: a bounded, still-smooth waveform)."""
    w = np.array([1.0] + [1.0 / np.sqrt(k) for k in range(1, n_harmonics + 1)])
    return w / w.sum()


def decode(action, n_harmonics, n_steer=0):
    """Split the raw [-1,1] action into
    (cam_coeffs, thigh_coeffs, freq_raw, reflex, steer, residual).
    With n_steer=0 the action carries no steering dims and `steer` comes back as zeros — the gait
    is then mirror-symmetric, i.e. straight-line only, exactly as before steering existed."""
    pj = per_joint(n_harmonics)
    cam = np.asarray(action[0:pj], dtype=float)
    thigh = np.asarray(action[pj:2 * pj], dtype=float)
    freq_raw = float(action[2 * pj])
    k = 2 * pj + 1
    reflex = np.asarray(action[k:k + N_REFLEX], dtype=float)
    k += N_REFLEX
    steer = np.asarray(action[k:k + n_steer], dtype=float) if n_steer else np.zeros(N_STEER)
    k += n_steer
    residual = np.asarray(action[k:k + N_RESIDUAL], dtype=float)
    return cam, thigh, freq_raw, reflex, steer, residual


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


def assemble(cam_coeffs, thigh_coeffs, reflex, phi, roll, roll_rate, nominal, cfg,
             pitch=0.0, pitch_rate=0.0, steer=None):
    """Return the 6 PD motor targets (rad) at gait phase phi (residuals are added by the env).

    nominal          : 6-vector stance ctrl = the per-episode centers (mirrored L/R already).
    roll,roll_rate   : current base roll and roll-rate (for the learned abduction reflex).
    pitch,pitch_rate : current base pitch (grav_x ~ sin(pitch), + = nose-down) and pitch rate,
                       for the FIXED pitch-stabilizing reflex. Default 0 keeps every legacy caller
                       (and the m1/m2 rail, where pitch is locked) byte-identical.
    steer            : 2-vector [stride_asym, width_asym] in [-1,1], or None for a symmetric
                       (straight-only) gait — keeps pre-steering callers byte-identical.
    cfg              : Config (n_harmonics, cam_amp, thigh_amp, reflex_*_scale, pitch_k*, steer_*).
    """
    N = cfg.n_harmonics
    w = _weights(N)
    # --- steering: the ONLY asymmetry in this action space -------------------------------------
    # Everything else below is mirror-symmetric by construction, and a mirror-symmetric gait can
    # only run straight. Two mechanisms, both learned in sign (nothing here assumes which way a
    # positive value turns — the policy discovers that from the yaw-rate tracking reward):
    #   s_stride : differential stride AMPLITUDE. One leg reaches further per cycle than the other,
    #              so the feet travel unequal arcs -> the body yaws toward the short-stride side.
    #              This is how a legged robot with no yaw actuator actually turns.
    #   s_width  : differential stance WIDTH. Same-signed hip_roll offset on both legs; because the
    #              L/R roll axes are MIRRORED, same sign in joint space = one leg abducts out while
    #              the other adducts in -> an asymmetric support polygon that biases the turn.
    if steer is None:
        s_stride = s_width = 0.0
    else:
        s_stride = cfg.steer_stride_scale * float(steer[0])
        s_width = cfg.steer_width_scale * float(steer[1])
    gL, gR = 1.0 + s_stride, 1.0 - s_stride
    # --- cam + thigh: Fourier, right = mirror (negate) + antiphase (phi + pi), then steer-scaled ---
    # The oscillation DELTA is clipped to +-amp of the nominal. `nominal` is always a validated
    # in-band posture (the stand keyframe / M1 ride-height LUT), so bounding the deviation keeps the
    # coupled cam-thigh 4-bar inside its assemblable band without assuming an absolute cam sign or
    # center (the physical ctrl-range clip in the env is the final guard). The steer gain is applied
    # AFTER the clip, so it can reach amp*(1+steer_stride_scale) — bounded, and the env's ctrlrange
    # clip is still the last word.
    ca, ta = cfg.cam_amp, cfg.thigh_amp
    d_cam = np.clip(ca * _series(cam_coeffs, phi, w, N), -ca, ca)
    d_thigh = np.clip(ta * _series(thigh_coeffs, phi, w, N), -ta, ta)
    d_cam_a = np.clip(ca * _series(cam_coeffs, phi + np.pi, w, N), -ca, ca)
    d_thigh_a = np.clip(ta * _series(thigh_coeffs, phi + np.pi, w, N), -ta, ta)
    cam_L = nominal[CAM_L] + gL * d_cam
    thigh_L = nominal[THIGH_L] + gL * d_thigh
    cam_R = nominal[CAM_R] - gR * d_cam_a
    thigh_R = nominal[THIGH_R] - gR * d_thigh_a
    # --- FIXED pitch reflex: symmetric fore-aft foot shift on both thighs (added AFTER the +-amp
    # clip so a large gait swing can't eat the reflex authority right when push-off needs it). The
    # thigh axes are MIRRORED (L: +Y, R: -Y), so thigh_L += u_p, thigh_R -= u_p moves BOTH feet the
    # same physical direction. u_p = -clip(kp*pitch + kd*pitch_rate + bias): when nose-down
    # (pitch > 0) u_p < 0 -> feet move forward under the falling CoM (capture point). The env's
    # ctrl-range clip is the final guard. clip=0 disables the reflex.
    u_p = -np.clip(cfg.pitch_kp * pitch + cfg.pitch_kd * pitch_rate + cfg.pitch_bias,
                   -cfg.pitch_clip, cfg.pitch_clip)
    thigh_L += u_p
    thigh_R -= u_p
    # --- hip_roll: learned linear balance reflex (feedback on roll / roll-rate); sign is learned ---
    # +u/-u in joint space = both feet shift the SAME physical way (lateral CoM shift, the balance
    # channel); +w/+w = opposite physical ways (stance-width asymmetry, the steering channel).
    u = (cfg.reflex_kp_scale * reflex[0] * roll
         + cfg.reflex_kd_scale * reflex[1] * roll_rate
         + cfg.reflex_bias_scale * reflex[2])
    hr_L = nominal[HIP_ROLL_L] + u + s_width
    hr_R = nominal[HIP_ROLL_R] - u + s_width

    out = np.empty(6)
    out[HIP_ROLL_L], out[CAM_L], out[THIGH_L] = hr_L, cam_L, thigh_L
    out[HIP_ROLL_R], out[CAM_R], out[THIGH_R] = hr_R, cam_R, thigh_R
    return out


def stance_indicator(phi, stance_ratio, edge=0.1 * 2.0 * np.pi):
    """Smooth expected-stance indicator I(phi) in [0,1] for the LEFT foot: ~1 inside the stance
    window [0, 2*pi*stance_ratio), ~0 in the swing window, raised-cosine transitions of width
    `edge` centered on the two boundaries (Siekmann-style phase gating; cosine bumps instead of
    von Mises — same shape, no scipy). The RIGHT foot uses stance_indicator(phi + pi, ...): the
    reward windows share the exact antiphase convention the action mirroring uses.
    stance_ratio < 0.5 leaves a window where BOTH feet are expected in swing -> demanding it
    forces a flight phase (running). Computed in the WINDOW-CENTERED frame — circular distance
    from the window center vs. its half-width — which stays unambiguous for any window width
    (a naive product of two circular edges breaks for windows wider than half a cycle) and is
    continuous across the phase wrap by construction. I = 0.5 exactly on each boundary."""
    two_pi = 2.0 * np.pi
    center = np.pi * float(stance_ratio)              # stance window is [0, 2*pi*sr)
    half = np.pi * float(stance_ratio)

    def smooth(x):                       # 0 -> 1 raised-cosine step on [0, 1]
        x = np.clip(x, 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * x)

    dc = np.mod(phi - center + np.pi, two_pi) - np.pi   # signed circular distance from center
    return float(smooth((half - abs(dc)) / edge + 0.5))
